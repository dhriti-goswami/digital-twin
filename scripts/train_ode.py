#!/usr/bin/env python3
"""
Train the GlucoseTransformer on UVA/Padova ODE-simulated data.

Uses vectorized feature engineering (ode_features.py) which is 20× faster
than the loop-based GlucoseFeatureEngine.

Run generate_training_data.py first, then:
    python scripts/train_ode.py
    python scripts/train_ode.py --epochs 150 --shap
"""

import argparse
import gc
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ode_features import load_all_patients, FEATURE_NAMES
from src.models.glucose_predictor import GlucosePredictor
from scripts.train_model import (
    TrainingConfig,
    ProgressTracker,
    setup_logging,
    run_shap_analysis,
)

logger = logging.getLogger(__name__)

SIM_DIR = PROJECT_ROOT / "data" / "raw" / "simulated"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
HORIZONS = [30, 60, 90, 120]


# ──────────────────────────────────────────────────────────────────────────────
# Clinical penalty loss (from paper eq 5-9)
# ──────────────────────────────────────────────────────────────────────────────

def clinical_penalty_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Weighted MSE with clinical penalty:
      P = 2 if actual < 70 and predicted ≥ 70  (missed hypoglycemia)
      P = 6 if actual > 180 and predicted ≤ 180 (missed hyperglycemia)
      P = 1 otherwise
    """
    P = torch.ones_like(target)
    hypo_miss = (target < 70) & (pred >= 70)
    hyper_miss = (target > 180) & (pred <= 180)
    P = torch.where(hypo_miss, torch.full_like(P, 2.0), P)
    P = torch.where(hyper_miss, torch.full_like(P, 6.0), P)
    return (P * (pred - target) ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class ODETrainer:
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _make_loader(self, X, y, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

    def train(
        self,
        data: dict,
        model: GlucosePredictor,
    ) -> dict:
        train_loader = self._make_loader(data["train_X"], data["train_y"], shuffle=True)
        val_loader = self._make_loader(data["val_X"], data["val_y"], shuffle=False)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        progress = ProgressTracker(self.config.epochs, len(train_loader))
        progress.start_training()

        history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
        scaler = data["scaler"]
        feature_names = data["feature_names"]

        for epoch in range(self.config.epochs):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            total_loss = 0
            progress.start_epoch(epoch)

            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = clinical_penalty_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()
                total_loss += loss.item()
                progress.update_batch(i, loss.item(), len(xb))

            # ── Validate ──────────────────────────────────────────────────────
            model.eval()
            val_preds, val_targets = [], []
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    val_loss += clinical_penalty_loss(pred, yb).item()
                    val_preds.append(pred.cpu())
                    val_targets.append(yb.cpu())

            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)
            mae = float(torch.abs(val_preds - val_targets).mean())
            rmse = float(torch.sqrt(torch.mean((val_preds - val_targets) ** 2)))
            horizon_mae = torch.abs(val_preds - val_targets).mean(dim=0).tolist()
            avg_val_loss = val_loss / len(val_loader)
            avg_train_loss = total_loss / len(train_loader)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_mae"].append(mae)
            history["val_rmse"].append(rmse)

            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self._save(epoch, {"val_loss": avg_val_loss, "val_mae": mae, "val_rmse": rmse,
                                "horizon_mae": horizon_mae},
                       model, optimizer, scheduler, scaler, feature_names, is_best)

            progress.end_epoch(
                epoch,
                {"train_loss": avg_train_loss, "train_mse": avg_train_loss, "train_physics": 0},
                {"val_loss": avg_val_loss, "val_mae": mae, "val_rmse": rmse, "horizon_mae": horizon_mae},
                optimizer.param_groups[0]["lr"],
                is_best,
            )

            scheduler.step()

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.warning("Early stopping at epoch %d", epoch + 1)
                break

        # Load best
        best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        progress.end_training({"val_mae": history["val_mae"][-1], "val_rmse": history["val_rmse"][-1]},
                               self.best_val_loss)

        # Save learning curves for evaluation script
        hist_path = Path(self.config.checkpoint_dir) / "training_history.npz"
        np.savez(str(hist_path), **{k: np.array(v) for k, v in history.items()})
        logger.info("History saved → %s", hist_path)

        return {"history": history, "model": model, "val_loader": val_loader}

    def _save(self, epoch, metrics, model, optimizer, scheduler, scaler, feature_names, is_best):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "scaler": pickle.dumps(scaler),
            "feature_names": feature_names,
        }
        p = Path(self.config.checkpoint_dir)
        torch.save(ckpt, p / "latest_checkpoint.pt")
        if is_best:
            torch.save(ckpt, p / "best_model.pt")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train glucose transformer on ODE simulated data")
    p.add_argument("--sim-dir", default=str(SIM_DIR))
    p.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    p.add_argument("--model", default="transformer", choices=["transformer", "lstm"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--seq-length", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--shap", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(verbose=True)

    print("\n" + "═" * 80)
    print("  DIABETES DIGITAL TWIN — ODE-BASED TRAINING  (Clinical Penalty Loss)")
    print("═" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s  (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # 1. Load + build features
    data = load_all_patients(
        sim_dir=Path(args.sim_dir),
        seq_len=args.seq_length,
        horizons=[6, 12, 18, 24],
        val_split=args.val_split,
    )
    logger.info("Features (%d): %s …", len(FEATURE_NAMES), ", ".join(FEATURE_NAMES[:6]))
    gc.collect()

    # 2. Build model
    model = GlucosePredictor(
        input_size=data["n_features"],
        model_type=args.model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_horizons=4,
        dropout=args.dropout,
        use_pinn=False,   # clinical penalty replaces PINN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s  params: %s", args.model.upper(), f"{total_params:,}")

    # Print config box
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │              TRAINING CONFIGURATION             │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │ Model:        {args.model:>33} │")
    print(f"  │ Hidden size:  {args.hidden_size:>33} │")
    print(f"  │ Layers:       {args.num_layers:>33} │")
    print(f"  │ LR:           {args.lr:>33} │")
    print(f"  │ Batch:        {args.batch_size:>33} │")
    print(f"  │ Epochs:       {args.epochs:>33} │")
    print(f"  │ Train seq:    {len(data['train_X']):>33,} │")
    print(f"  │ Val seq:      {len(data['val_X']):>33,} │")
    print(f"  │ Features:     {data['n_features']:>33} │")
    print("  └─────────────────────────────────────────────────┘\n")

    # 3. Train
    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        use_pinn=False,
        early_stopping_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer = ODETrainer(config, device)
    results = trainer.train(data, model)

    print(f"\n  Checkpoint → {args.checkpoint_dir}/best_model.pt")

    # 4. SHAP
    if args.shap:
        shap_dir = Path(args.checkpoint_dir) / "shap"
        run_shap_analysis(results["model"], results["val_loader"], FEATURE_NAMES, shap_dir, device)

    return results


if __name__ == "__main__":
    main()
