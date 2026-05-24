"""
Glucose Prediction Models using PyTorch.

Implements LSTM and Transformer architectures with Physics-Informed Neural Network (PINN)
components that incorporate physiological constraints based on the Bergman Minimal Model.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class GlucoseLSTM(nn.Module):
    """
    LSTM-based glucose prediction model.

    Architecture:
    - Input embedding layer
    - Bidirectional LSTM layers
    - Attention mechanism
    - Multi-horizon prediction heads
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_horizons: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_horizons = num_horizons

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_horizons)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Predictions of shape (batch, num_horizons)
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)

        # Take the last timestep
        last_hidden = attn_out[:, -1, :]

        # Output projection
        out = self.dropout(F.relu(self.fc1(last_hidden)))
        out = self.fc2(out)

        return out


class GlucoseTransformer(nn.Module):
    """
    Transformer-based glucose prediction model.

    Architecture:
    - Input embedding with positional encoding
    - Transformer encoder layers
    - Multi-horizon prediction heads
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_horizons: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_horizons = num_horizons

        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_horizons)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            mask: Optional attention mask

        Returns:
            Predictions of shape (batch, num_horizons)
        """
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq, d_model)

        # Positional encoding expects (seq, batch, d_model)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # Back to (batch, seq, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x, mask=mask)

        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)

        # Output projection
        x = self.output_norm(x)
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.fc2(x)

        return x


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Neural Network (PINN) loss component.

    Incorporates physiological constraints from the Bergman Minimal Model
    to ensure predictions are biologically plausible.

    The Bergman Minimal Model describes glucose-insulin dynamics:
    dG/dt = -p1*(G - Gb) - X*G + Ra(t)
    dX/dt = -p2*X + p3*(I - Ib)

    Where:
    - G: glucose concentration
    - X: remote insulin action
    - I: plasma insulin concentration (approximated via rapid IOB)
    - Gb: basal glucose concentration (target equilibrium)
    - p1: glucose effectiveness (min^-1)
    - p2: remote insulin action decay rate (min^-1)
    - p3: insulin sensitivity (min^-1 / (uU/mL))
    - Ra(t): glucose appearance rate from meal carbohydrates
    """

    def __init__(
        self,
        p1: float = 0.028,      # Glucose effectiveness (min^-1)
        p2: float = 0.025,      # Rate of insulin action decay (min^-1)
        p3: float = 5.0e-5,     # Insulin sensitivity
        gb: float = 110.0,      # Basal glucose (mg/dL)
        dt: float = 30.0,       # Step size between prediction horizons (minutes)
        lambda_physics: float = 0.1,  # Weight of physics loss
    ):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.gb = gb
        self.dt = dt
        self.lambda_physics = lambda_physics

    def forward(
        self,
        pred: torch.Tensor,                  # Predicted glucose (batch, 4) at 30, 60, 90, 120 mins
        target: torch.Tensor,                # Target glucose (batch, 4)
        glucose_history: torch.Tensor,       # Historical glucose (batch, seq_len)
        iob: torch.Tensor,                   # Rapid insulin on board at t=0 (batch,)
        cob: torch.Tensor,                   # Carbs on board at t=0 (batch,)
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined MSE + Physics loss using Bergman Minimal Model ODE residuals.
        """
        # Standard prediction loss (MSE)
        mse_loss = F.mse_loss(pred, target)

        # Concatenate current glucose G(t=0) with predictions to get full trajectory G(t) at [0, 30, 60, 90, 120] min
        g0 = glucose_history[:, -1].unsqueeze(1)  # Shape (batch, 1)
        g_traj = torch.cat([g0, pred], dim=1)     # Shape (batch, 5)

        physics_loss = torch.tensor(0.0, device=pred.device)
        dt = self.dt

        # Check that we have valid arrays
        batch_size = pred.size(0)
        iob = iob.view(batch_size)
        cob = cob.view(batch_size)

        # Compute ODE residuals for each of the 4 prediction horizons: 30, 60, 90, 120 min
        for k in range(1, 5):
            tk = k * dt

            # Extract glucose at current step and previous step
            g_k = g_traj[:, k]
            g_prev = g_traj[:, k-1]

            # 1. Estimate dG/dt using finite differences (mg/dL per minute)
            dG_dt = (g_k - g_prev) / dt

            # 2. Model rapid-acting IOB decay curve to estimate active insulin action X(t_k)
            # Rapid insulin durations typically ~4 hours (240 mins)
            # Fraction remaining modeled exponentially: (1 - t/240)^1.5
            iob_fraction = torch.clamp(1.0 - torch.tensor(tk / 240.0, device=pred.device), min=0.0) ** 1.5
            iob_k = iob * iob_fraction

            # Bergman: X(t) represents remote compartment insulin action.
            # Active insulin action is proportional to the active insulin on board.
            x_k = self.p3 * iob_k

            # 3. Model meal carbohydrate absorption to estimate rate of glucose appearance Ra(t_k)
            # Meal absorption duration typically ~3 hours (180 mins)
            # Fraction remaining modeled exponentially: (1 - t/180)^1.2
            cob_fraction_prev = torch.clamp(1.0 - torch.tensor((tk - dt) / 180.0, device=pred.device), min=0.0) ** 1.2
            cob_fraction_k = torch.clamp(1.0 - torch.tensor(tk / 180.0, device=pred.device), min=0.0) ** 1.2

            cob_prev = cob * cob_fraction_prev
            cob_k = cob * cob_fraction_k

            # Ra(t) is the rate of appearance from meal absorption (mg/dL per minute)
            # modeled as the rate of carb decay multiplied by systemic absorption scaling (~0.15 mg/dL/g/min)
            ra_k = ((cob_prev - cob_k) / dt) * 0.15

            # 4. Compute Bergman minimal model ODE residual:
            # dG/dt - [ -p1*(G_k - Gb) - X_k*G_k + Ra_k ] = 0
            pinn_residual = dG_dt - (-self.p1 * (g_k - self.gb) - x_k * g_k + ra_k)

            # Accumulate mean square residuals
            physics_loss += (pinn_residual ** 2).mean()

        # Bounds constraint: glucose should stay within physiological range [40, 400] mg/dL
        lower_violation = F.relu(40.0 - pred)
        upper_violation = F.relu(pred - 400.0)
        bounds_loss = (lower_violation.mean() + upper_violation.mean()) * 0.01

        physics_loss += bounds_loss

        # Combined loss
        total_loss = mse_loss + self.lambda_physics * physics_loss

        return total_loss, {
            "mse_loss": mse_loss.item(),
            "physics_loss": physics_loss.item(),
            "total_loss": total_loss.item(),
        }


class GlucosePredictor(nn.Module):
    """
    Main glucose prediction model wrapper.

    Supports both LSTM and Transformer architectures with Physics-Informed loss.
    """

    def __init__(
        self,
        input_size: int,
        model_type: str = "transformer",
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        num_horizons: int = 4,
        dropout: float = 0.1,
        use_pinn: bool = True,
        pinn_lambda: float = 0.1,
    ):
        super().__init__()

        self.model_type = model_type
        self.num_horizons = num_horizons
        self.use_pinn = use_pinn

        if model_type == "lstm":
            self.model = GlucoseLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_horizons=num_horizons,
                dropout=dropout,
            )
        elif model_type == "transformer":
            self.model = GlucoseTransformer(
                input_size=input_size,
                d_model=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                num_horizons=num_horizons,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.pinn_loss = PhysicsInformedLoss(lambda_physics=pinn_lambda) if use_pinn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        glucose_history: Optional[torch.Tensor] = None,
        iob: Optional[torch.Tensor] = None,
        cob: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss with optional physics constraints."""
        if self.use_pinn and self.pinn_loss is not None and glucose_history is not None and iob is not None and cob is not None:
            return self.pinn_loss(pred, target, glucose_history, iob, cob)
        else:
            loss = F.mse_loss(pred, target)
            return loss, {"mse_loss": loss.item(), "total_loss": loss.item()}

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence intervals.

        Uses Monte Carlo dropout for uncertainty estimation.

        Returns:
            mean_pred: Mean prediction
            std_pred: Standard deviation (uncertainty)
        """
        self.train()  # Enable dropout
        n_samples = 20

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        self.eval()
        return mean_pred, std_pred


class EnsemblePredictor(nn.Module):
    """
    Ensemble of glucose predictors for improved accuracy and uncertainty estimation.
    """

    def __init__(
        self,
        input_size: int,
        n_models: int = 3,
        model_configs: Optional[list[dict]] = None,
    ):
        super().__init__()

        self.n_models = n_models

        if model_configs is None:
            # Default: mix of LSTM and Transformer
            model_configs = [
                {"model_type": "transformer", "hidden_size": 128, "num_layers": 4},
                {"model_type": "transformer", "hidden_size": 64, "num_layers": 3},
                {"model_type": "lstm", "hidden_size": 128, "num_layers": 2},
            ]

        self.models = nn.ModuleList([
            GlucosePredictor(input_size=input_size, **config)
            for config in model_configs[:n_models]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean prediction from ensemble."""
        predictions = torch.stack([model(x) for model in self.models])
        return predictions.mean(dim=0)

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and std from ensemble predictions."""
        predictions = torch.stack([model(x) for model in self.models])
        return predictions.mean(dim=0), predictions.std(dim=0)
