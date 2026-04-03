#!/usr/bin/env python3
"""
Model Validation Script.

Validates the trained glucose prediction model with comprehensive testing:
- Model loading and architecture verification
- Performance metrics on validation data
- Inference speed benchmarking
- Clinical accuracy assessment
- Export validation report

Usage:
    python scripts/validate_model.py
    python scripts/validate_model.py --model checkpoints/best_model.pt
    python scripts/validate_model.py --export-report
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_model import TrainingConfig
from src.models.glucose_predictor import GlucosePredictor
from src.data.preprocessing import GlucoseFeatureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Model validation report."""
    timestamp: str
    model_path: str
    model_type: str
    total_parameters: int
    checkpoint_size_mb: float

    # Performance metrics
    val_loss: float
    val_mae: float
    val_rmse: float
    horizon_mae: dict

    # Clinical assessment
    clinical_rating: str
    clinical_suitable: bool

    # Inference performance
    inference_time_ms: float
    throughput_samples_per_sec: float

    # Configuration
    config: dict


def load_checkpoint(model_path: str, device: torch.device) -> dict:
    """Load model checkpoint."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def get_model_info(checkpoint: dict) -> dict:
    """Extract model information from checkpoint."""
    config = checkpoint.get("config")
    state_dict = checkpoint.get("model_state_dict", {})
    metrics = checkpoint.get("metrics", {})

    total_params = sum(t.numel() for t in state_dict.values())

    return {
        "config": config,
        "total_params": total_params,
        "metrics": metrics,
        "epoch": checkpoint.get("epoch", -1),
    }


def benchmark_inference(model: GlucosePredictor, device: torch.device,
                       batch_size: int = 32, seq_len: int = 24,
                       n_features: int = 42, n_runs: int = 100) -> dict:
    """Benchmark model inference speed."""
    model.eval()

    # Create dummy input
    X = torch.randn(batch_size, seq_len, n_features).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(X)

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(X)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    total_samples = batch_size * n_runs
    avg_time_per_batch = (elapsed / n_runs) * 1000  # ms
    throughput = total_samples / elapsed

    return {
        "inference_time_ms": avg_time_per_batch,
        "throughput_samples_per_sec": throughput,
        "batch_size": batch_size,
        "n_runs": n_runs,
    }


def assess_clinical_suitability(mae: float, horizon_mae: list) -> tuple[str, bool]:
    """Assess clinical suitability of model predictions."""
    # Clinical standards for glucose prediction
    # FDA guidance suggests MARD < 15% for CGM accuracy
    # For absolute error, < 15 mg/dL is considered good

    if mae < 10:
        rating = "EXCELLENT"
        suitable = True
        note = "Exceeds clinical standards for glucose prediction"
    elif mae < 15:
        rating = "GOOD"
        suitable = True
        note = "Meets clinical standards for glucose prediction"
    elif mae < 20:
        rating = "ACCEPTABLE"
        suitable = True
        note = "Acceptable for patient guidance with monitoring"
    elif mae < 30:
        rating = "MARGINAL"
        suitable = False
        note = "May provide guidance but requires careful interpretation"
    else:
        rating = "INSUFFICIENT"
        suitable = False
        note = "Not suitable for clinical use - requires retraining"

    # Check longer horizons (90-120min typically less accurate)
    if len(horizon_mae) >= 4:
        if horizon_mae[3] > 30:  # 120-min horizon
            note += " (Note: Long-horizon predictions less reliable)"

    return rating, suitable


def validate_model(model_path: str, export_report: bool = False) -> ValidationReport:
    """Run full model validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info(f"Loading model from: {model_path}")
    checkpoint = load_checkpoint(model_path, device)
    model_info = get_model_info(checkpoint)

    config = model_info["config"]
    metrics = model_info["metrics"]

    # Print validation header
    print("\n" + "=" * 70)
    print("  GLUCOSE PREDICTION MODEL - VALIDATION REPORT")
    print("=" * 70)

    # Model Configuration
    print("\n[1. MODEL CONFIGURATION]")
    print(f"    Model Type:       {config.model_type}")
    print(f"    Hidden Size:      {config.hidden_size}")
    print(f"    Num Layers:       {config.num_layers}")
    print(f"    Num Heads:        {config.num_heads}")
    print(f"    Dropout:          {config.dropout}")
    print(f"    Physics Loss:     {config.use_pinn}")
    print(f"    Total Parameters: {model_info['total_params']:,}")

    # Performance Metrics
    print("\n[2. PERFORMANCE METRICS]")
    val_mae = metrics.get("val_mae", 0)
    val_rmse = metrics.get("val_rmse", 0)
    val_loss = metrics.get("val_loss", 0)

    print(f"    Validation Loss:  {val_loss:.4f}")
    print(f"    Mean Abs Error:   {val_mae:.2f} mg/dL")
    print(f"    Root MSE:         {val_rmse:.2f} mg/dL")

    horizon_mae = metrics.get("horizon_mae", [])
    horizons = [30, 60, 90, 120]
    horizon_dict = {}

    if horizon_mae:
        print("\n    Per-Horizon MAE:")
        for h, mae in zip(horizons, horizon_mae):
            horizon_dict[f"{h}min"] = mae
            quality = "Excellent" if mae < 10 else "Good" if mae < 15 else "Acceptable" if mae < 20 else "Marginal"
            bar = "█" * int(mae / 2) if mae < 50 else "█" * 25
            print(f"      {h:3d} min: {mae:6.2f} mg/dL │{bar:25s}│ {quality}")

    # Clinical Assessment
    print("\n[3. CLINICAL ASSESSMENT]")
    rating, suitable = assess_clinical_suitability(val_mae, horizon_mae)
    print(f"    Clinical Rating:  {rating}")
    print(f"    Suitable for Use: {'Yes' if suitable else 'No'}")

    # Load model for inference benchmark
    print("\n[4. INFERENCE PERFORMANCE]")

    # Infer input size
    state_dict = checkpoint["model_state_dict"]
    input_size = 42  # default
    for key, tensor in state_dict.items():
        if "input" in key and "weight" in key and len(tensor.shape) == 2:
            input_size = tensor.shape[1]
            break

    model = GlucosePredictor(
        input_size=input_size,
        model_type=config.model_type,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_horizons=4,
        use_pinn=config.use_pinn,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    bench = benchmark_inference(model, device, n_features=input_size)
    print(f"    Avg Inference Time: {bench['inference_time_ms']:.2f} ms/batch")
    print(f"    Throughput:         {bench['throughput_samples_per_sec']:.0f} samples/sec")
    print(f"    Device:             {device}")

    # Model Size
    model_size = Path(model_path).stat().st_size / 1024 / 1024
    print("\n[5. MODEL SIZE]")
    print(f"    Checkpoint Size:  {model_size:.2f} MB")
    print(f"    Parameters:       {model_info['total_params']:,}")

    # Summary
    print("\n" + "=" * 70)
    if suitable:
        print("  VALIDATION: PASSED - Model is ready for production use")
    else:
        print("  VALIDATION: NEEDS IMPROVEMENT - Consider retraining")
    print("=" * 70 + "\n")

    # Create report
    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        model_path=str(model_path),
        model_type=config.model_type,
        total_parameters=model_info["total_params"],
        checkpoint_size_mb=model_size,
        val_loss=val_loss,
        val_mae=val_mae,
        val_rmse=val_rmse,
        horizon_mae=horizon_dict,
        clinical_rating=rating,
        clinical_suitable=suitable,
        inference_time_ms=bench["inference_time_ms"],
        throughput_samples_per_sec=bench["throughput_samples_per_sec"],
        config={
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "use_pinn": config.use_pinn,
        },
    )

    if export_report:
        report_path = Path(model_path).parent / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Report exported to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate trained glucose prediction model")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--export-report", action="store_true",
                       help="Export validation report to JSON")
    args = parser.parse_args()

    try:
        report = validate_model(args.model, args.export_report)
        sys.exit(0 if report.clinical_suitable else 1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
