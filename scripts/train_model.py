#!/usr/bin/env python3
"""
Training script for glucose prediction models.

Usage:
    python scripts/train_model.py                    # Train with defaults
    python scripts/train_model.py --model transformer --epochs 50
    python scripts/train_model.py --model lstm --batch-size 64
    python scripts/train_model.py --shap             # Run SHAP analysis after training
"""

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import GlucoseFeatureEngine
from src.models.glucose_predictor import GlucosePredictor

# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logging(verbose: bool = True):
    """Setup logging with colors and appropriate verbosity."""
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        "%(asctime)s │ %(levelname)s │ %(message)s",
        datefmt="%H:%M:%S"
    ))

    logging.basicConfig(level=level, handlers=[handler])

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Real-time progress tracking with ETA estimation."""

    def __init__(self, total_epochs: int, total_batches: int):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.epoch_start_time = None
        self.training_start_time = None
        self.batch_times = []
        self.epoch_times = []

    def start_training(self):
        """Mark training start."""
        self.training_start_time = time.time()
        print("\n" + "=" * 80)
        print("  TRAINING STARTED")
        print("=" * 80 + "\n")

    def start_epoch(self, epoch: int):
        """Mark epoch start."""
        self.epoch_start_time = time.time()
        self.batch_times = []
        print(f"\n{'─' * 80}")
        print(f"  EPOCH {epoch + 1}/{self.total_epochs}")
        print(f"{'─' * 80}")

    def update_batch(self, batch_idx: int, batch_loss: float, batch_size: int):
        """Update progress after each batch."""
        self.batch_times.append(time.time())

        # Calculate progress
        progress = (batch_idx + 1) / self.total_batches
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Calculate speed
        if len(self.batch_times) > 1:
            recent_times = self.batch_times[-10:]
            samples_per_sec = batch_size * len(recent_times) / (recent_times[-1] - recent_times[0])
        else:
            samples_per_sec = 0

        # ETA for this epoch
        elapsed = time.time() - self.epoch_start_time
        if progress > 0:
            eta_epoch = elapsed / progress - elapsed
        else:
            eta_epoch = 0

        # Print progress bar (overwrite line)
        sys.stdout.write(f"\r  [{bar}] {progress*100:5.1f}% │ "
                        f"Batch {batch_idx+1:4d}/{self.total_batches} │ "
                        f"Loss: {batch_loss:.4f} │ "
                        f"{samples_per_sec:6.1f} samples/s │ "
                        f"ETA: {self._format_time(eta_epoch)}")
        sys.stdout.flush()

    def end_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float, is_best: bool):
        """Print epoch summary."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        # Clear progress bar line
        sys.stdout.write("\r" + " " * 100 + "\r")

        # Calculate total ETA
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - epoch - 1
        total_eta = avg_epoch_time * remaining_epochs

        # Best marker
        best_marker = " ★ NEW BEST" if is_best else ""

        # Print summary
        print(f"\n  ┌{'─' * 76}┐")
        print(f"  │ {'EPOCH SUMMARY':^74} │")
        print(f"  ├{'─' * 76}┤")
        print(f"  │ {'Training Loss:':<20} {train_metrics['train_loss']:>10.4f}    │ {'Physics Loss:':<18} {train_metrics.get('train_physics', 0):>10.4f} │")
        print(f"  │ {'Validation Loss:':<20} {val_metrics['val_loss']:>10.4f}    │ {'Learning Rate:':<18} {lr:>10.2e} │")
        print(f"  │ {'Val MAE:':<20} {val_metrics['val_mae']:>10.2f} mg/dL │ {'Val RMSE:':<18} {val_metrics['val_rmse']:>10.2f} mg/dL │")
        print(f"  ├{'─' * 76}┤")
        print(f"  │ {'Time:':<12} {self._format_time(epoch_time):>10} │ {'ETA:':<12} {self._format_time(total_eta):>10} │ {best_marker:^26} │")
        print(f"  └{'─' * 76}┘")

        # Per-horizon breakdown
        if 'horizon_mae' in val_metrics:
            horizons = [30, 60, 90, 120]
            print(f"\n  Per-Horizon MAE (mg/dL):")
            for i, (h, mae) in enumerate(zip(horizons, val_metrics['horizon_mae'])):
                bar_len = int(mae / 2)  # Scale for display
                bar = "▓" * min(bar_len, 30)
                print(f"    {h:3d} min: {mae:6.2f} │{bar}")

    def end_training(self, final_metrics: dict, best_val_loss: float):
        """Print training complete summary."""
        total_time = time.time() - self.training_start_time

        print("\n" + "=" * 80)
        print("  TRAINING COMPLETE")
        print("=" * 80)
        print(f"\n  Total Training Time: {self._format_time(total_time)}")
        print(f"  Best Validation Loss: {best_val_loss:.4f}")
        print(f"  Final MAE: {final_metrics['val_mae']:.2f} mg/dL")
        print(f"  Final RMSE: {final_metrics['val_rmse']:.2f} mg/dL")
        print()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


# ============================================================================
# DATASET
# ============================================================================

class GlucoseDataset(Dataset):
    """PyTorch Dataset for glucose prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray, glucose_history: np.ndarray, iob: np.ndarray, cob: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.glucose_history = torch.from_numpy(glucose_history).float()
        self.iob = torch.from_numpy(iob).float()
        self.cob = torch.from_numpy(cob).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.glucose_history[idx], self.iob[idx], self.cob[idx]


# ============================================================================
# TRAINING CONFIG
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 100
    early_stopping_patience: int = 15
    val_split: float = 0.2
    gradient_clip: float = 1.0
    model_type: str = "transformer"
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    use_pinn: bool = True
    pinn_lambda: float = 0.1
    checkpoint_dir: str = "./checkpoints"


# ============================================================================
# VERBOSE TRAINER
# ============================================================================

class VerboseTrainer:
    """Training pipeline with detailed progress logging."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.progress = None

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

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

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Model architecture: {self.config.model_type.upper()}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6,
        )

        return self.model

    def prepare_data(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        train_gh: np.ndarray,
        train_iob: np.ndarray,
        train_cob: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        val_gh: np.ndarray,
        val_iob: np.ndarray,
        val_cob: np.ndarray,
    ) -> tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders with clean patient-level splitting."""
        train_dataset = GlucoseDataset(train_X, train_y, train_gh, train_iob, train_cob)
        val_dataset = GlucoseDataset(val_X, val_y, val_gh, val_iob, val_cob)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        logger.info(f"Training samples: {len(train_dataset):,}")
        logger.info(f"Validation samples: {len(val_dataset):,}")
        logger.info(f"Batches per epoch: {len(train_loader)}")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """Train for one epoch with progress updates."""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_physics = 0
        n_batches = 0

        self.progress.start_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            X, y, gh, iob, cob = batch
            X = X.to(self.device)
            y = y.to(self.device)
            gh = gh.to(self.device)
            iob = iob.to(self.device)
            cob = cob.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(X)
            loss, loss_dict = self.model.compute_loss(pred, y, gh, iob, cob)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            total_loss += loss_dict["total_loss"]
            total_mse += loss_dict["mse_loss"]
            total_physics += loss_dict.get("physics_loss", 0)
            n_batches += 1

            # Update progress bar
            self.progress.update_batch(batch_idx, loss_dict["total_loss"], len(X))

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
            X, y, gh, iob, cob = batch
            X = X.to(self.device)
            y = y.to(self.device)
            gh = gh.to(self.device)
            iob = iob.to(self.device)
            cob = cob.to(self.device)

            pred = self.model(X)
            loss, loss_dict = self.model.compute_loss(pred, y, gh, iob, cob)

            total_loss += loss_dict["total_loss"]
            total_mse += loss_dict["mse_loss"]
            n_batches += 1

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        mae = torch.abs(all_preds - all_targets).mean().item()
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
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

        latest_path = Path(self.config.checkpoint_dir) / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)

    def train(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        train_glucose_history: np.ndarray,
        train_iob: np.ndarray,
        train_cob: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        val_glucose_history: np.ndarray,
        val_iob: np.ndarray,
        val_cob: np.ndarray,
    ) -> dict:
        """Full training loop with verbose output."""
        input_size = train_X.shape[2]
        num_horizons = train_y.shape[1]

        # Create model
        self.create_model(input_size, num_horizons)

        # Prepare data
        train_loader, val_loader = self.prepare_data(
            train_X, train_y, train_glucose_history, train_iob, train_cob,
            val_X, val_y, val_glucose_history, val_iob, val_cob
        )

        # Initialize progress tracker
        self.progress = ProgressTracker(self.config.epochs, len(train_loader))
        self.progress.start_training()

        history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

        for epoch in range(self.config.epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation
            val_metrics = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_mae"].append(val_metrics["val_mae"])
            history["val_rmse"].append(val_metrics["val_rmse"])

            # Check for improvement
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Print epoch summary
            self.progress.end_epoch(epoch, train_metrics, val_metrics, current_lr, is_best)

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.warning(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model
        best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Final validation
        final_metrics = self.validate(val_loader)
        self.progress.end_training(final_metrics, self.best_val_loss)

        return {
            "history": history,
            "final_metrics": final_metrics,
            "best_val_loss": self.best_val_loss,
            "model": self.model,
            "val_loader": val_loader,
        }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed CSV files."""
    logger.info(f"Loading data from {data_dir}")

    glucose_df = pd.read_csv(data_dir / "glucose_real.csv")
    insulin_df = pd.read_csv(data_dir / "insulin_real.csv")
    meals_df = pd.read_csv(data_dir / "meals_real.csv")

    glucose_df = glucose_df.rename(columns={"timestamp": "time"})
    insulin_df = insulin_df.rename(columns={"timestamp": "time"})
    meals_df = meals_df.rename(columns={"timestamp": "time"})

    glucose_df["time"] = pd.to_datetime(glucose_df["time"])
    insulin_df["time"] = pd.to_datetime(insulin_df["time"])
    meals_df["time"] = pd.to_datetime(meals_df["time"])

    if "trend" not in glucose_df.columns:
        glucose_df["trend"] = "STABLE"

    logger.info(f"Loaded {len(glucose_df):,} glucose readings")
    logger.info(f"Loaded {len(insulin_df):,} insulin records")
    logger.info(f"Loaded {len(meals_df):,} meal records")

    return glucose_df, insulin_df, meals_df


def interpolate_glucose(patient_glucose: pd.DataFrame, interval_minutes: int = 5) -> pd.DataFrame:
    """Interpolate glucose readings to regular intervals."""
    if len(patient_glucose) < 2:
        return patient_glucose

    patient_glucose = patient_glucose.sort_values("time")
    start_time = patient_glucose["time"].min()
    end_time = patient_glucose["time"].max()
    regular_times = pd.date_range(start=start_time, end=end_time, freq=f"{interval_minutes}min")

    patient_glucose = patient_glucose.set_index("time")
    interpolated = patient_glucose.reindex(regular_times)
    interpolated["glucose_mg_dl"] = interpolated["glucose_mg_dl"].interpolate(method="linear")
    interpolated["patient_id"] = interpolated["patient_id"].ffill().bfill()
    interpolated["trend"] = interpolated["trend"].ffill().bfill()
    interpolated = interpolated.reset_index().rename(columns={"index": "time"})

    return interpolated.dropna(subset=["glucose_mg_dl"])


def load_static_clinical_profiles(pima_path: Path, hosp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PIMA and 130-Hospitals datasets and extract clinical profiles."""
    logger.info("Loading clinical profiles for multi-dataset fusion...")
    
    # 1. Load PIMA profiles
    if pima_path.exists():
        pima_df = pd.read_csv(pima_path, header=None)
        pima_profiles = pd.DataFrame()
        pima_profiles["pregnancies"] = pima_df[0]
        pima_profiles["glucose"] = pima_df[1].replace(0, 120.0)
        pima_profiles["blood_pressure"] = pima_df[2].replace(0, 70.0)
        pima_profiles["skin_thickness"] = pima_df[3].replace(0, 20.0)
        pima_profiles["insulin"] = pima_df[4].replace(0, 80.0)
        pima_profiles["bmi"] = pima_df[5].replace(0.0, 28.0)
        pima_profiles["diabetes_pedigree"] = pima_df[6]
        pima_profiles["age"] = pima_df[7]
        pima_profiles["outcome"] = pima_df[8]
        pima_profiles["estimated_hba1c"] = (pima_profiles["glucose"] + 46.7) / 28.7
    else:
        logger.warning(f"PIMA dataset not found at {pima_path}, creating fallback profiles")
        pima_profiles = pd.DataFrame({
            "pregnancies": [2] * 100,
            "glucose": [120.0] * 100,
            "blood_pressure": [72.0] * 100,
            "skin_thickness": [23.0] * 100,
            "insulin": [79.0] * 100,
            "bmi": [29.0] * 100,
            "diabetes_pedigree": [0.47] * 100,
            "age": [33.0] * 100,
            "outcome": [0] * 100,
            "estimated_hba1c": [5.8] * 100
        })

    # 2. Load 130-Hospitals profiles
    csv_path = hosp_path / "diabetic_data.csv"
    if csv_path.exists():
        hosp_df = pd.read_csv(csv_path, low_memory=False)
        hosp_df = hosp_df.replace("?", np.nan)
        hosp_profiles = pd.DataFrame()
        hosp_profiles["time_in_hospital"] = pd.to_numeric(hosp_df["time_in_hospital"], errors="coerce").fillna(4.0)
        hosp_profiles["num_lab_procedures"] = pd.to_numeric(hosp_df["num_lab_procedures"], errors="coerce").fillna(43.0)
        hosp_profiles["num_procedures"] = pd.to_numeric(hosp_df["num_procedures"], errors="coerce").fillna(1.0)
        hosp_profiles["num_medications"] = pd.to_numeric(hosp_df["num_medications"], errors="coerce").fillna(16.0)
        hosp_profiles["number_diagnoses"] = pd.to_numeric(hosp_df["number_diagnoses"], errors="coerce").fillna(7.0)
    else:
        logger.warning(f"130-Hospitals dataset not found at {csv_path}, creating fallback profiles")
        hosp_profiles = pd.DataFrame({
            "time_in_hospital": [4.0] * 100,
            "num_lab_procedures": [43.0] * 100,
            "num_procedures": [1.0] * 100,
            "num_medications": [16.0] * 100,
            "number_diagnoses": [7.0] * 100
        })

    return pima_profiles, hosp_profiles


def prepare_patient_data(
    patient_id: int,
    glucose_df: pd.DataFrame,
    insulin_df: pd.DataFrame,
    meals_df: pd.DataFrame,
    feature_engine: GlucoseFeatureEngine,
    pima_profile: pd.Series,
    hosp_profile: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray] | None:
    """Prepare training data for a single patient."""
    patient_glucose = glucose_df[glucose_df["patient_id"] == patient_id].copy()
    patient_insulin = insulin_df[insulin_df["patient_id"] == patient_id].copy()
    patient_meals = meals_df[meals_df["patient_id"] == patient_id].copy()

    min_required = feature_engine.sequence_length + max(feature_engine.prediction_horizons) + 10
    if len(patient_glucose) < min_required:
        return None

    patient_glucose = interpolate_glucose(patient_glucose, feature_engine.cgm_interval_minutes)

    if len(patient_glucose) < min_required:
        return None

    cgm_features = feature_engine.create_cgm_features(patient_glucose)
    temporal_features = feature_engine.create_temporal_features(patient_glucose["time"])
    insulin_features = feature_engine.create_insulin_features(patient_glucose["time"], patient_insulin)
    meal_features = feature_engine.create_meal_features(patient_glucose["time"], patient_meals)
    activity_features = feature_engine.create_activity_features(patient_glucose["time"], pd.DataFrame())

    all_features = pd.concat([
        cgm_features.reset_index(drop=True),
        temporal_features.reset_index(drop=True),
        insulin_features.reset_index(drop=True),
        meal_features.reset_index(drop=True),
        activity_features.reset_index(drop=True),
    ], axis=1)

    all_features = all_features.loc[:, ~all_features.columns.duplicated()]

    # Extract raw unscaled inputs before dropping columns or scaling!
    raw_glucose = all_features['glucose_mg_dl'].values
    raw_iob = all_features['iob_rapid'].values
    raw_cob = all_features['cob'].values

    # Ingress static clinical covariates (Multi-Dataset Fusion)
    all_features["static_age"] = pima_profile["age"]
    all_features["static_bmi"] = pima_profile["bmi"]
    all_features["static_hba1c"] = pima_profile["estimated_hba1c"]
    all_features["static_num_meds"] = hosp_profile["num_medications"]

    # Drop non-numeric columns that can't be used as features
    cols_to_drop = ['time', 'trend', 'patient_id']
    all_features = all_features.drop(columns=[c for c in cols_to_drop if c in all_features.columns], errors='ignore')

    # Ensure all columns are numeric
    for col in all_features.columns:
        all_features[col] = pd.to_numeric(all_features[col], errors='coerce')

    all_features = all_features.ffill().fillna(0)

    # Get feature names
    feature_names = [c for c in all_features.columns if c != 'glucose_mg_dl']

    X, y = feature_engine.create_sequences(all_features)

    # Reconstruct matching raw unscaled sequences for PINN ODE calculations
    glucose_history = []
    iob_history = []
    cob_history = []
    max_horizon = max(feature_engine.prediction_horizons)

    for i in range(feature_engine.sequence_length, len(all_features) - max_horizon):
        glucose_history.append(raw_glucose[i - feature_engine.sequence_length : i])
        iob_history.append(raw_iob[i - 1])
        cob_history.append(raw_cob[i - 1])

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    glucose_history = np.array(glucose_history, dtype=np.float32)
    iob_history = np.array(iob_history, dtype=np.float32)
    cob_history = np.array(cob_history, dtype=np.float32)

    return X, y, feature_names, glucose_history, iob_history, cob_history


def prepare_all_data(
    glucose_df: pd.DataFrame,
    insulin_df: pd.DataFrame,
    meals_df: pd.DataFrame,
    pima_path: Path,
    hosp_path: Path,
    sequence_length: int = 24,
    prediction_horizons: list[int] = None,
    val_split: float = 0.2,
) -> dict:
    """Prepare training data from all patients using a rigorous data leakage-free patient split."""
    feature_engine = GlucoseFeatureEngine(
        sequence_length=sequence_length,
        prediction_horizons=prediction_horizons or [6, 12, 18, 24],
        cgm_interval_minutes=5,
    )

    # Load static clinical datasets
    pima_profiles, hosp_profiles = load_static_clinical_profiles(pima_path, hosp_path)

    patient_ids = sorted(glucose_df["patient_id"].unique())
    logger.info(f"Splitting {len(patient_ids)} patients to prevent data leakage...")
    
    # Enforce patient-level separation
    val_count = int(len(patient_ids) * val_split)
    np.random.seed(42)
    shuffled_ids = np.random.permutation(patient_ids).tolist()
    val_patient_ids = shuffled_ids[:val_count]
    train_patient_ids = shuffled_ids[val_count:]
    
    logger.info(f"Train patients ({len(train_patient_ids)}): {train_patient_ids}")
    logger.info(f"Validation patients ({len(val_patient_ids)}): {val_patient_ids}")

    train_X_list, train_y_list, train_gh_list, train_iob_list, train_cob_list = [], [], [], [], []
    val_X_list, val_y_list, val_gh_list, val_iob_list, val_cob_list = [], [], [], [], []
    feature_names = None

    # Helper function to process patient list
    def process_patients(patient_list, X_list, y_list, gh_list, iob_list, cob_list):
        nonlocal feature_names
        successful = 0
        for pid in patient_list:
            # Deterministic mapping to clinical profiles
            pima_idx = (pid * 13) % len(pima_profiles)
            hosp_idx = (pid * 101) % len(hosp_profiles)
            pima_p = pima_profiles.iloc[pima_idx]
            hosp_p = hosp_profiles.iloc[hosp_idx]

            result = prepare_patient_data(
                pid, glucose_df, insulin_df, meals_df, feature_engine, pima_p, hosp_p
            )
            if result is not None:
                p_X, p_y, names, p_gh, p_iob, p_cob = result
                X_list.append(p_X)
                y_list.append(p_y)
                gh_list.append(p_gh)
                iob_list.append(p_iob)
                cob_list.append(p_cob)
                if feature_names is None:
                    feature_names = names
                successful += 1
        return successful

    train_count = process_patients(train_patient_ids, train_X_list, train_y_list, train_gh_list, train_iob_list, train_cob_list)
    val_count = process_patients(val_patient_ids, val_X_list, val_y_list, val_gh_list, val_iob_list, val_cob_list)

    logger.info(f"Processed {train_count} train patients and {val_count} validation patients")

    if not train_X_list or not val_X_list:
        raise ValueError("No valid training or validation data found!")

    # Concatenate per split
    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    train_gh = np.concatenate(train_gh_list, axis=0)
    train_iob = np.concatenate(train_iob_list, axis=0)
    train_cob = np.concatenate(train_cob_list, axis=0)

    val_X = np.concatenate(val_X_list, axis=0)
    val_y = np.concatenate(val_y_list, axis=0)
    val_gh = np.concatenate(val_gh_list, axis=0)
    val_iob = np.concatenate(val_iob_list, axis=0)
    val_cob = np.concatenate(val_cob_list, axis=0)

    # Fit scaling ONLY on training features to prevent validation leakage!
    train_flat = train_X.reshape(-1, train_X.shape[2])
    feature_engine.scaler.fit(train_flat)
    feature_engine._is_fitted = True

    # Scale both train and validation
    train_X_scaled = feature_engine.scaler.transform(train_flat).reshape(train_X.shape).astype(np.float32)
    val_flat = val_X.reshape(-1, val_X.shape[2])
    val_X_scaled = feature_engine.scaler.transform(val_flat).reshape(val_X.shape).astype(np.float32)

    return {
        "train_X": train_X_scaled,
        "train_y": train_y,
        "train_gh": train_gh,
        "train_iob": train_iob,
        "train_cob": train_cob,
        "val_X": val_X_scaled,
        "val_y": val_y,
        "val_gh": val_gh,
        "val_iob": val_iob,
        "val_cob": val_cob,
        "feature_engine": feature_engine,
        "feature_names": feature_names,
    }


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def run_shap_analysis(model, val_loader, feature_names: list[str], output_dir: Path, device: torch.device):
    """Run SHAP analysis on the trained model."""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap matplotlib")
        return

    logger.info("Running SHAP analysis...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Get a batch of data for SHAP
    X_sample = []
    for batch in val_loader:
        X_sample.append(batch[0])
        if len(X_sample) * val_loader.batch_size >= 200:
            break

    X_sample = torch.cat(X_sample)[:200].to(device)

    # Use last timestep features for interpretation
    X_last = X_sample[:, -1, :].cpu().numpy()

    # Create a wrapper for SHAP
    def model_predict(X_np):
        with torch.no_grad():
            # Repeat last timestep to match sequence length
            X_seq = np.tile(X_np[:, np.newaxis, :], (1, X_sample.shape[1], 1))
            X_tensor = torch.FloatTensor(X_seq).to(device)
            return model(X_tensor).cpu().numpy()

    # Use KernelExplainer for model-agnostic SHAP values
    logger.info("Computing SHAP values (this may take a few minutes)...")
    background = X_last[:50]
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_last[:100], nsamples=100)

    # Handle multi-output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # First prediction horizon

    # Handle multi-output case - flatten to 2D if needed
    if len(shap_values.shape) > 2:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)

    # Truncate feature names if needed
    if len(feature_names) != X_last.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_last.shape[1])]

    # Ensure shap_values matches feature count
    if shap_values.shape[1] != len(feature_names):
        # Average across extra dimensions if model has multiple outputs
        if shap_values.shape[1] > len(feature_names):
            n_features = len(feature_names)
            shap_values = shap_values[:, :n_features]
        else:
            feature_names = feature_names[:shap_values.shape[1]]

    # Save summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_last[:100], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot to {output_dir / 'shap_summary.png'}")

    # Save bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_last[:100], feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP importance plot to {output_dir / 'shap_importance.png'}")

    # Compute mean absolute SHAP values for ranking
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Ensure mean_shap is 1D array of scalars
    if len(mean_shap.shape) > 1:
        mean_shap = mean_shap.flatten()[:len(feature_names)]

    # Convert to list of floats for sorting
    mean_shap_list = [float(x) for x in mean_shap]
    feature_importance = sorted(zip(feature_names, mean_shap_list), key=lambda x: x[1], reverse=True)

    max_importance = max(mean_shap_list) if mean_shap_list else 1.0

    logger.info("\nTop 15 Most Important Features (SHAP):")
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │            FEATURE IMPORTANCE (SHAP)            │")
    print("  ├─────────────────────────────────────────────────┤")
    for i, (name, importance) in enumerate(feature_importance[:15]):
        bar_len = int(importance / max_importance * 25)
        bar = "█" * bar_len
        print(f"  │ {i+1:2d}. {name:<25} {importance:8.4f} {bar:<25} │")
    print("  └─────────────────────────────────────────────────┘\n")

    # Save feature importance to CSV
    importance_df = pd.DataFrame(feature_importance, columns=["feature", "importance"])
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info(f"Saved feature importance to {output_dir / 'feature_importance.csv'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train glucose prediction model")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "lstm"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--no-pinn", action="store_true", help="Disable physics-informed loss")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--data-dir", type=str, default="./data/processed")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis after training")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Print header
    print("\n" + "═" * 80)
    print("  DIABETES DIGITAL TWIN - GLUCOSE PREDICTION MODEL TRAINING")
    print("═" * 80 + "\n")

    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    glucose_df, insulin_df, meals_df = load_data(data_dir)

    # Prepare training data
    logger.info("Preparing training data...")
    pima_path = PROJECT_ROOT / "data" / "raw" / "pima" / "pima_diabetes.csv"
    hosp_path = PROJECT_ROOT / "data" / "raw" / "diabetes_130_hospitals"
    
    data_dict = prepare_all_data(
        glucose_df, insulin_df, meals_df,
        pima_path=pima_path, hosp_path=hosp_path,
        sequence_length=args.seq_length,
    )
    
    feature_names = data_dict["feature_names"]
    feature_engine = data_dict["feature_engine"]

    # Free memory from dataframes
    del glucose_df, insulin_df, meals_df
    gc.collect()

    # Log feature info
    logger.info(f"Features ({len(feature_names)}): {', '.join(feature_names[:10])}...")

    # Create training config
    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_pinn=not args.no_pinn,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Print config
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │              TRAINING CONFIGURATION             │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │ Model:              {config.model_type:>27} │")
    print(f"  │ Hidden Size:        {config.hidden_size:>27} │")
    print(f"  │ Num Layers:         {config.num_layers:>27} │")
    print(f"  │ Learning Rate:      {config.learning_rate:>27} │")
    print(f"  │ Batch Size:         {config.batch_size:>27} │")
    print(f"  │ Epochs:             {config.epochs:>27} │")
    print(f"  │ Physics Loss:       {str(config.use_pinn):>27} │")
    print(f"  │ Sequence Length:    {args.seq_length:>27} │")
    print("  └─────────────────────────────────────────────────┘\n")

    # Train model
    trainer = VerboseTrainer(config)
    results = trainer.train(
        train_X=data_dict["train_X"],
        train_y=data_dict["train_y"],
        train_glucose_history=data_dict["train_gh"],
        train_iob=data_dict["train_iob"],
        train_cob=data_dict["train_cob"],
        val_X=data_dict["val_X"],
        val_y=data_dict["val_y"],
        val_glucose_history=data_dict["val_gh"],
        val_iob=data_dict["val_iob"],
        val_cob=data_dict["val_cob"],
    )

    # Save final model path
    print(f"\n  Model saved to: {args.checkpoint_dir}/best_model.pt")

    # Run SHAP analysis if requested
    if args.shap:
        shap_output_dir = Path(args.checkpoint_dir) / "shap"
        run_shap_analysis(
            results["model"],
            results["val_loader"],
            feature_names,
            shap_output_dir,
            device,
        )

    return results


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def run_shap_analysis(model, val_loader, feature_names: list[str], output_dir: Path, device: torch.device):
    """Run SHAP analysis on the trained model."""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap matplotlib")
        return

    logger.info("Running SHAP analysis...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Get a batch of data for SHAP
    X_sample = []
    for batch in val_loader:
        X_sample.append(batch[0])
        if len(X_sample) * val_loader.batch_size >= 200:
            break

    X_sample = torch.cat(X_sample)[:200].to(device)

    # Use last timestep features for interpretation
    X_last = X_sample[:, -1, :].cpu().numpy()

    # Create a wrapper for SHAP
    def model_predict(X_np):
        with torch.no_grad():
            # Repeat last timestep to match sequence length
            X_seq = np.tile(X_np[:, np.newaxis, :], (1, X_sample.shape[1], 1))
            X_tensor = torch.FloatTensor(X_seq).to(device)
            return model(X_tensor).cpu().numpy()

    # Use KernelExplainer for model-agnostic SHAP values
    logger.info("Computing SHAP values (this may take a few minutes)...")
    background = X_last[:50]
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_last[:100], nsamples=100)

    # Handle multi-output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # First prediction horizon

    # Handle multi-output case - flatten to 2D if needed
    if len(shap_values.shape) > 2:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)

    # Truncate feature names if needed
    if len(feature_names) != X_last.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_last.shape[1])]

    # Ensure shap_values matches feature count
    if shap_values.shape[1] != len(feature_names):
        # Average across extra dimensions if model has multiple outputs
        if shap_values.shape[1] > len(feature_names):
            n_features = len(feature_names)
            shap_values = shap_values[:, :n_features]
        else:
            feature_names = feature_names[:shap_values.shape[1]]

    # Save summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_last[:100], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot to {output_dir / 'shap_summary.png'}")

    # Save bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_last[:100], feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP importance plot to {output_dir / 'shap_importance.png'}")

    # Compute mean absolute SHAP values for ranking
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Ensure mean_shap is 1D array of scalars
    if len(mean_shap.shape) > 1:
        mean_shap = mean_shap.flatten()[:len(feature_names)]

    # Convert to list of floats for sorting
    mean_shap_list = [float(x) for x in mean_shap]
    feature_importance = sorted(zip(feature_names, mean_shap_list), key=lambda x: x[1], reverse=True)

    max_importance = max(mean_shap_list) if mean_shap_list else 1.0

    logger.info("\nTop 15 Most Important Features (SHAP):")
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │            FEATURE IMPORTANCE (SHAP)            │")
    print("  ├─────────────────────────────────────────────────┤")
    for i, (name, importance) in enumerate(feature_importance[:15]):
        bar_len = int(importance / max_importance * 25)
        bar = "█" * bar_len
        print(f"  │ {i+1:2d}. {name:<25} {importance:8.4f} {bar:<25} │")
    print("  └─────────────────────────────────────────────────┘\n")

    # Save feature importance to CSV
    importance_df = pd.DataFrame(feature_importance, columns=["feature", "importance"])
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info(f"Saved feature importance to {output_dir / 'feature_importance.csv'}")



if __name__ == "__main__":
    main()

