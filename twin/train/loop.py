"""Training loop.

Model selection is on **validation MAE at the shortest horizon**, computed on a
purged, time-ordered validation tail. The legacy pipeline selected on a validation
set drawn by ``randperm`` from overlapping windows, so its selection signal was a
near-copy of the training data.

The checkpoint is self-describing: it carries the resolved config, the fitted
scaler, the feature-name contract, and the fold identity. A checkpoint that cannot
state which scaler and which fold produced it cannot be evaluated reproducibly, and
the legacy artefacts could not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from twin.config import Config
from twin.data.dataset import FittedScaler
from twin.data.features import FEATURE_NAMES
from twin.models.forecaster import ForecastOutput, PhysicsGuidedForecaster
from twin.train.loss import AdaptiveWeights, compute_loss, physics_ramp


@dataclass
class EpochRecord:
    """One epoch's metrics, for the learning-curve figure."""

    epoch: int
    train_loss: float
    val_mae_30: float
    val_mae_mean: float
    seconds: float
    prior_gate: float | None = None
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_mae_30": self.val_mae_30,
            "val_mae_mean": self.val_mae_mean,
            "seconds": self.seconds,
        }
        if self.prior_gate is not None:
            out["prior_gate"] = self.prior_gate
        out.update(self.extras)
        return out


@dataclass
class TrainingResult:
    """Outcome of one training run."""

    history: list[EpochRecord]
    best_epoch: int
    best_val_mae_30: float
    checkpoint_path: Path | None = None

    def history_frame(self):
        import pandas as pd

        return pd.DataFrame([record.as_dict() for record in self.history])


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def predict_loader(
    model: PhysicsGuidedForecaster,
    loader: DataLoader,
    device: str,
) -> dict[str, np.ndarray]:
    """Run the model over a loader and return predictions, targets, and diagnostics.

    Also returns per-window insulin sensitivity and the subject index, which the
    ``S_I`` validation needs -- it is the same forward pass, so recomputing it later
    would risk a different result.
    """
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sensitivity: list[np.ndarray] = []
    subjects: list[np.ndarray] = []

    for batch in loader:
        batch = _to_device(batch, device)
        output: ForecastOutput = model(batch)
        predictions.append(output.horizons.float().cpu().numpy())
        targets.append(batch["targets"].float().cpu().numpy())
        sensitivity.append(output.insulin_sensitivity.float().cpu().numpy())
        subjects.append(batch["subject_index"].cpu().numpy())

    return {
        "predictions": np.concatenate(predictions, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "insulin_sensitivity": np.concatenate(sensitivity, axis=0),
        "subject_index": np.concatenate(subjects, axis=0),
    }


def train_model(
    model: PhysicsGuidedForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    *,
    scaler: FittedScaler,
    fold_name: str,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train one model, selecting on purged validation MAE at 30 minutes."""
    device = config.resolve_device()
    model = model.to(device)
    weights = AdaptiveWeights(config).to(device)

    parameters = list(model.parameters()) + list(weights.parameters())
    optimiser = torch.optim.AdamW(
        parameters, lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(config.train.epochs, 1)
    )

    history: list[EpochRecord] = []
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0

    for epoch in range(1, config.train.epochs + 1):
        started = time.time()
        # Parameters are frozen at population values for the warmup epochs: the
        # encoder has no useful signal yet, and an unconstrained parameter estimate
        # at that point collapses to whatever fits the initial noise.
        use_population = epoch <= config.physics.param_warmup_epochs

        model.train()
        running: list[float] = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimiser.zero_grad(set_to_none=True)
            output = model(batch, use_population_params=use_population)
            breakdown = compute_loss(output, batch, weights, config, epoch=epoch)
            breakdown.total.backward()
            nn.utils.clip_grad_norm_(parameters, config.train.grad_clip)
            optimiser.step()
            running.append(float(breakdown.total.detach()))

        evaluation = predict_loader(model, val_loader, device)
        errors = np.abs(evaluation["predictions"] - evaluation["targets"])
        val_mae_30 = float(errors[:, 0].mean())
        val_mae_mean = float(errors.mean())

        record = EpochRecord(
            epoch=epoch,
            train_loss=float(np.mean(running)) if running else float("nan"),
            val_mae_30=val_mae_30,
            val_mae_mean=val_mae_mean,
            seconds=time.time() - started,
            prior_gate=float(torch.sigmoid(model.prior_logit).detach()) if model.hybrid else None,
            extras={"physics_ramp": physics_ramp(epoch, config)},
        )
        history.append(record)
        schedule.step()

        if verbose:
            gate = f" gate {record.prior_gate:.3f}" if record.prior_gate is not None else ""
            print(
                f"epoch {epoch:3d}  train {record.train_loss:8.4f}  "
                f"val MAE@30 {val_mae_30:6.2f}  mean {val_mae_mean:6.2f}"
                f"{gate}  ramp {record.extras['physics_ramp']:.2f}  "
                f"{record.seconds:.1f}s",
                flush=True,
            )

        if val_mae_30 < best_val - 1e-4:
            best_val = val_mae_30
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.train.early_stopping_patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best {best_epoch})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path: Path | None = None
    if out_dir is not None:
        checkpoint_path = Path(out_dir) / "best_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "adaptive_weights_state_dict": weights.state_dict(),
                "config": config.to_dict(),
                "scaler": scaler.state_dict(),
                "feature_names": list(FEATURE_NAMES),
                "fold_name": fold_name,
                "best_epoch": best_epoch,
                "best_val_mae_30": best_val,
                "history": [record.as_dict() for record in history],
            },
            checkpoint_path,
        )

    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_val_mae_30=best_val,
        checkpoint_path=checkpoint_path,
    )


def load_checkpoint(path: str | Path) -> dict[str, object]:
    """Load a checkpoint and verify it matches the current feature contract.

    A checkpoint trained under a different feature list cannot be evaluated with the
    current one: the legacy deployment silently zero-filled six missing features and
    fed them through a scaler that mapped the zeros to large negative z-scores.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    stored = tuple(state.get("feature_names", ()))
    if stored != FEATURE_NAMES:
        missing = set(FEATURE_NAMES) - set(stored)
        extra = set(stored) - set(FEATURE_NAMES)
        raise ValueError(
            f"{path}: checkpoint feature contract does not match the current one. "
            f"missing={sorted(missing)} unexpected={sorted(extra)}"
        )
    return state


__all__ = [
    "EpochRecord",
    "TrainingResult",
    "load_checkpoint",
    "predict_loader",
    "train_model",
]
