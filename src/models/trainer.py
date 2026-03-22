"""
Training pipeline for glucose prediction models.

Includes:
- Data loading and batching
- Training loop with early stopping
- MLflow experiment tracking
- Model checkpointing
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.models.glucose_predictor import GlucosePredictor

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 100
    early_stopping_patience: int = 10
    val_split: float = 0.2
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    model_type: str = "transformer"
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    use_pinn: bool = True
    pinn_lambda: float = 0.1
    checkpoint_dir: str = "./checkpoints"
    experiment_name: str = "glucose_prediction"


class GlucoseDataset(Dataset):
    """PyTorch Dataset for glucose prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray, glucose_history: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.glucose_history = torch.FloatTensor(glucose_history) if glucose_history is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.glucose_history is not None:
            return self.X[idx], self.y[idx], self.glucose_history[idx]
        return self.X[idx], self.y[idx]


class Trainer:
    """Training pipeline for glucose prediction models."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Training on device: {self.device}")

    def create_model(self, input_size: int, num_horizons: int = 4) -> GlucosePredictor:
        """Create and initialize the model."""
        self.model = GlucosePredictor(
            input_size=input_size,
            model_type=self.config.model_type,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            num_horizons=num_horizons,
            dropout=self.config.dropout,
            use_pinn=self.config.use_pinn,
            pinn_lambda=self.config.pinn_lambda,
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        return self.model

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        glucose_history: Optional[np.ndarray] = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        dataset = GlucoseDataset(X, y, glucose_history)

        val_size = int(len(dataset) * self.config.val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_physics = 0
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 3:
                X, y, glucose_hist = batch
                glucose_hist = glucose_hist.to(self.device)
            else:
                X, y = batch
                glucose_hist = None

            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(X)

            # Compute loss
            loss, loss_dict = self.model.compute_loss(pred, y, glucose_hist)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            self.optimizer.step()

            total_loss += loss_dict["total_loss"]
            total_mse += loss_dict["mse_loss"]
            total_physics += loss_dict.get("physics_loss", 0)
            n_batches += 1

        return {
            "train_loss": total_loss / n_batches,
            "train_mse": total_mse / n_batches,
            "train_physics": total_physics / n_batches,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        all_preds = []
        all_targets = []
        n_batches = 0

        for batch in val_loader:
            if len(batch) == 3:
                X, y, glucose_hist = batch
                glucose_hist = glucose_hist.to(self.device)
            else:
                X, y = batch
                glucose_hist = None

            X = X.to(self.device)
            y = y.to(self.device)

            pred = self.model(X)
            loss, loss_dict = self.model.compute_loss(pred, y, glucose_hist)

            total_loss += loss_dict["total_loss"]
            total_mse += loss_dict["mse_loss"]
            n_batches += 1

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Compute metrics
        mae = torch.abs(all_preds - all_targets).mean().item()
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()

        # Per-horizon metrics
        horizon_mae = torch.abs(all_preds - all_targets).mean(dim=0).tolist()

        return {
            "val_loss": total_loss / n_batches,
            "val_mse": total_mse / n_batches,
            "val_mae": mae,
            "val_rmse": rmse,
            "horizon_mae": horizon_mae,
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
        }

        # Save latest checkpoint
        latest_path = Path(self.config.checkpoint_dir) / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {metrics['val_loss']:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"], checkpoint["metrics"]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        glucose_history: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Full training loop.

        Args:
            X: Feature sequences (n_samples, seq_len, n_features)
            y: Target values (n_samples, n_horizons)
            glucose_history: Optional glucose history for PINN loss

        Returns:
            Dictionary with training history and final metrics
        """
        input_size = X.shape[2]
        num_horizons = y.shape[1]

        # Create model
        self.create_model(input_size, num_horizons)

        # Prepare data
        train_loader, val_loader = self.prepare_data(X, y, glucose_history)

        # Initialize MLflow tracking
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run()
            mlflow.log_params({
                "model_type": self.config.model_type,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "use_pinn": self.config.use_pinn,
            })

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
        }

        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

        for epoch in range(self.config.epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Record history
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_mae"].append(val_metrics["val_mae"])
            history["val_rmse"].append(val_metrics["val_rmse"])

            # Log metrics
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "train_loss": train_metrics["train_loss"],
                    "val_loss": val_metrics["val_loss"],
                    "val_mae": val_metrics["val_mae"],
                    "val_rmse": val_metrics["val_rmse"],
                }, step=epoch)

            # Check for improvement
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Logging
            if epoch % 5 == 0 or is_best:
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val MAE: {val_metrics['val_mae']:.2f} | "
                    f"Val RMSE: {val_metrics['val_rmse']:.2f}"
                )

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
        if best_path.exists():
            self.load_checkpoint(str(best_path))

        # Final validation
        final_metrics = self.validate(val_loader)

        # Log final model to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.log_metrics({
                "final_val_loss": final_metrics["val_loss"],
                "final_val_mae": final_metrics["val_mae"],
                "final_val_rmse": final_metrics["val_rmse"],
            })
            mlflow.end_run()

        logger.info(f"Training complete. Best Val Loss: {self.best_val_loss:.4f}")

        return {
            "history": history,
            "final_metrics": final_metrics,
            "best_val_loss": self.best_val_loss,
        }


def train_glucose_model(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[TrainingConfig] = None,
) -> tuple[GlucosePredictor, dict]:
    """
    Convenience function to train a glucose prediction model.

    Args:
        X: Feature sequences
        y: Target values
        config: Optional training configuration

    Returns:
        Trained model and training results
    """
    if config is None:
        config = TrainingConfig()

    trainer = Trainer(config)
    results = trainer.train(X, y)

    return trainer.model, results
