#!/usr/bin/env python3
"""
Comprehensive evaluation of the trained GlucoseTransformer.

Computes all clinical metrics from the T1DSim_AI paper (Table 3):
  - RMSE per prediction horizon (30/60/90/120 min)
  - TIR (70–180 mg/dL), TAR (>180), TBR (<70)
  - LBGI, HBGI, Mean Glucose
  - Clarke Error Grid Analysis
  - SHAP feature importance

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --shap
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model import setup_logging, run_shap_analysis
from src.models.inference import GlucoseInferenceService

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
HORIZONS = [30, 60, 90, 120]   # minutes


# ──────────────────────────────────────────────────────────────────────────────
# Clinical metrics
# ──────────────────────────────────────────────────────────────────────────────

def tir_metrics(glucose: np.ndarray) -> dict:
    """Time in range / above / below."""
    tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
    tar1 = np.mean((glucose > 180) & (glucose <= 250)) * 100
    tar2 = np.mean(glucose > 250) * 100
    tbr1 = np.mean((glucose >= 54) & (glucose < 70)) * 100
    tbr2 = np.mean(glucose < 54) * 100
    return {"TIR": tir, "TAR1": tar1, "TAR2": tar2, "TBR1": tbr1, "TBR2": tbr2}


def lbgi_hbgi(glucose: np.ndarray) -> tuple[float, float]:
    """Low/High Blood Glucose Index (Kovatchev 2000)."""
    f = 1.509 * (np.log(np.maximum(glucose, 1)) ** 1.084 - 5.381)
    rl = 10 * f ** 2 * (f < 0).astype(float)
    rh = 10 * f ** 2 * (f > 0).astype(float)
    return float(rl.mean()), float(rh.mean())


def clarke_ega(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Clarke Error Grid Analysis.

    Returns zone counts (A–E) and percentages.
    """
    n = len(actual)
    zones = np.full(n, "E")

    for i in range(n):
        ref = actual[i]
        pred = predicted[i]

        # Zone A
        if abs(ref - pred) / max(ref, 1e-9) <= 0.2 or (ref < 70 and pred < 70):
            zones[i] = "A"
            continue

        # Zone D
        if (ref < 70 and pred > 180) or (ref > 240 and pred < 70):
            zones[i] = "D"
            continue

        # Zone E
        if (ref > 180 and pred < 70) or (ref < 70 and pred > 180):
            zones[i] = "E"
            continue

        # Zone B
        above = pred > ref
        if ref >= 70 and ref <= 290:
            upper_b = ref + 0.25 * ref
            lower_b = ref - 0.25 * ref
            if lower_b <= pred <= upper_b:
                zones[i] = "B"
                continue

        # Zone C
        if (ref >= 130 and ref <= 180 and pred > ref + 0.3 * ref) or \
           (ref >= 70 and ref <= 130 and pred < ref - 0.3 * ref):
            zones[i] = "C"
            continue

        zones[i] = "B"  # default to B if no other condition met

    counts = {z: int(np.sum(zones == z)) for z in "ABCDE"}
    pcts = {z: counts[z] / n * 100 for z in "ABCDE"}
    return {"counts": counts, "pcts": pcts}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation on val set
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, val_X: np.ndarray, val_y: np.ndarray, device: torch.device) -> dict:
    """Run model on all validation data and collect predictions."""
    model.eval()
    all_preds, all_targets = [], []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(val_X), batch_size):
            xb = torch.from_numpy(val_X[i: i + batch_size]).to(device)
            yb = val_y[i: i + batch_size]
            pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(yb)

    preds = np.concatenate(all_preds)     # (N, 4)
    targets = np.concatenate(all_targets) # (N, 4)
    return {"preds": preds, "targets": targets}


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(history: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    ax.plot(history["val_loss"], label="Val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["val_mae"], label="Val MAE", linewidth=1.5, color="green")
    ax.plot(history["val_rmse"], label="Val RMSE", linewidth=1.5, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mg/dL")
    ax.set_title("Validation MAE / RMSE")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_prediction_traces(preds: np.ndarray, targets: np.ndarray, out_dir: Path, n: int = 3):
    """Plot a few example glucose traces: actual vs. predicted at all horizons."""
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    horizon_labels = [f"{h}min" for h in HORIZONS]

    for i, ax in enumerate(axes):
        idx = i * (len(targets) // n)
        actual = targets[max(0, idx - 6): idx + 1, 0]

        ax.plot(range(len(actual)), actual, "k-", linewidth=1.5, label="Actual (30-min)")
        for j, (h, c, lbl) in enumerate(zip(HORIZONS, colors, horizon_labels)):
            pred_val = preds[idx, j]
            ax.plot(len(actual) - 1 + j + 1, pred_val, "o", color=c, markersize=8, label=f"Pred {lbl}")
            ax.axvline(len(actual) - 1, color="gray", linestyle="--", alpha=0.5)

        ax.axhspan(70, 180, alpha=0.08, color="green")
        ax.axhline(70, color="orange", linestyle=":", alpha=0.7, linewidth=0.8)
        ax.axhline(180, color="orange", linestyle=":", alpha=0.7, linewidth=0.8)
        ax.set_ylim(40, 350)
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(f"Sample {i + 1}: Actual={targets[idx, 0]:.0f} mg/dL")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Timestep (5 min)")
    fig.tight_layout()
    path = out_dir / "prediction_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_scatter(preds: np.ndarray, targets: np.ndarray, out_dir: Path):
    """Actual vs. predicted scatter for each horizon."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for j, (h, ax) in enumerate(zip(HORIZONS, axes)):
        p = preds[:, j]
        t = targets[:, j]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))

        lim_lo = min(p.min(), t.min()) - 10
        lim_hi = max(p.max(), t.max()) + 10
        ax.scatter(t, p, alpha=0.3, s=3, color="#2196F3")
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", linewidth=1.5)
        ax.set_xlabel("Actual (mg/dL)")
        ax.set_ylabel("Predicted (mg/dL)")
        ax.set_title(f"{h}-min  RMSE={rmse:.1f}  MAE={mae:.1f}")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "scatter_actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_clarke_ega(actual: np.ndarray, predicted: np.ndarray, out_dir: Path, horizon_label: str = "30min"):
    """Clarke Error Grid plot."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ega = clarke_ega(actual, predicted)
    zone_colors = {"A": "#4CAF50", "B": "#8BC34A", "C": "#FF9800", "D": "#FF5722", "E": "#F44336"}

    # Background zones (simplified)
    ax.fill_between([0, 70], [0, 70], [0, 0], color=zone_colors["A"], alpha=0.1)
    ax.fill_between([70, 400], [0, 400], [70, 70], color=zone_colors["B"], alpha=0.1)

    ax.scatter(actual, predicted, alpha=0.4, s=4, c="#2196F3")
    ax.plot([0, 400], [0, 400], "k--", linewidth=1)
    ax.plot([0, 400], [0, 480], "g:", linewidth=0.8)  # +20% upper
    ax.plot([0, 400], [0, 320], "g:", linewidth=0.8)  # -20% lower

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Actual Glucose (mg/dL)")
    ax.set_ylabel("Predicted Glucose (mg/dL)")
    ax.set_title(f"Clarke Error Grid — {horizon_label}")

    # Add zone annotations
    zone_text = "  ".join([f"{z}: {ega['pcts'][z]:.1f}%" for z in "ABCDE"])
    ax.text(0.02, 0.98, zone_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"clarke_ega_{horizon_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return ega


def plot_horizon_rmse(preds: np.ndarray, targets: np.ndarray, out_dir: Path):
    """Bar chart of RMSE per prediction horizon — mirrors paper Table 3."""
    rmse = [float(np.sqrt(np.mean((preds[:, j] - targets[:, j]) ** 2))) for j in range(4)]
    mae = [float(np.mean(np.abs(preds[:, j] - targets[:, j]))) for j in range(4)]

    x = np.arange(4)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, rmse, w, label="RMSE", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + w / 2, mae, w, label="MAE", color="#FF9800", alpha=0.85)

    ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in HORIZONS])
    ax.set_ylabel("mg/dL")
    ax.set_title("Prediction Error by Horizon")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "horizon_rmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained GlucoseTransformer")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "best_model.pt"))
    p.add_argument("--sim-dir", default=str(PROJECT_ROOT / "data" / "raw" / "simulated"))
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--shap", action="store_true", help="Run SHAP analysis")
    p.add_argument("--seq-length", type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(verbose=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 80)
    print("  DIABETES DIGITAL TWIN — MODEL EVALUATION")
    print("═" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        logger.error("Run: python scripts/train_ode.py first")
        sys.exit(1)

    svc = GlucoseInferenceService(model_path=str(checkpoint_path), device=str(device))
    if not svc.model_loaded:
        logger.error("Failed to load model")
        sys.exit(1)

    model = svc.model
    logger.info("Model loaded — input_size=%d", svc.input_size)

    # 2. Rebuild validation data (same split as training)
    from src.data.ode_features import load_all_patients, FEATURE_NAMES
    data = load_all_patients(Path(args.sim_dir), seq_len=args.seq_length)
    val_X = data["val_X"]
    val_y = data["val_y"]
    feature_names = FEATURE_NAMES
    logger.info("Validation set: %d samples", len(val_X))

    # 3. Run model predictions
    logger.info("Running inference …")
    res = evaluate_model(model, val_X, val_y, device)
    preds = res["preds"]
    targets = res["targets"]

    # 4. Compute all metrics
    print("\n" + "─" * 80)
    print("  METRICS (matches paper Table 3 format)")
    print("─" * 80)

    metrics_rows = []
    for j, h in enumerate(HORIZONS):
        p = preds[:, j]
        t = targets[:, j]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        mard = float(np.mean(np.abs(p - t) / np.maximum(t, 1e-9)) * 100)
        tir = tir_metrics(p)
        lbgi, hbgi = lbgi_hbgi(p)
        ega = clarke_ega(t, p)

        metrics_rows.append({
            "horizon_min": h,
            "RMSE_mg_dL": round(rmse, 2),
            "MAE_mg_dL": round(mae, 2),
            "MARD_%": round(mard, 2),
            "TIR_%": round(tir["TIR"], 1),
            "TAR1_%": round(tir["TAR1"], 1),
            "TAR2_%": round(tir["TAR2"], 1),
            "TBR1_%": round(tir["TBR1"], 1),
            "TBR2_%": round(tir["TBR2"], 1),
            "LBGI": round(lbgi, 2),
            "HBGI": round(hbgi, 2),
            "EGA_A_%": round(ega["pcts"]["A"], 1),
            "EGA_B_%": round(ega["pcts"]["B"], 1),
        })

        print(f"\n  {h}-min horizon:")
        print(f"    RMSE = {rmse:.2f} mg/dL   MAE = {mae:.2f} mg/dL   MARD = {mard:.2f}%")
        print(f"    TIR  = {tir['TIR']:.1f}%   TAR = {tir['TAR1']+tir['TAR2']:.1f}%   TBR = {tir['TBR1']+tir['TBR2']:.1f}%")
        print(f"    LBGI = {lbgi:.2f}   HBGI = {hbgi:.2f}")
        print(f"    Clarke EGA: A={ega['pcts']['A']:.1f}%  B={ega['pcts']['B']:.1f}%  C={ega['pcts']['C']:.1f}%  D={ega['pcts']['D']:.1f}%  E={ega['pcts']['E']:.1f}%")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics saved → %s", metrics_path)

    # 5. Generate plots
    print("\n  Generating plots …")
    plot_scatter(preds, targets, out_dir)
    plot_horizon_rmse(preds, targets, out_dir)
    plot_prediction_traces(preds, targets, out_dir)
    plot_clarke_ega(targets[:, 0], preds[:, 0], out_dir, horizon_label="30min")
    plot_clarke_ega(targets[:, 2], preds[:, 2], out_dir, horizon_label="90min")

    # Learning curves (if history file exists from training)
    history_path = Path(args.checkpoint).parent / "training_history.npz"
    if history_path.exists():
        hist = dict(np.load(history_path))
        plot_learning_curves(hist, out_dir)

    # 6. SHAP
    if args.shap:
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        ds = TensorDataset(
            torch.from_numpy(val_X[:1000]),
            torch.from_numpy(val_y[:1000]),
            torch.zeros(min(1000, len(val_X)), args.seq_length),
            torch.zeros(min(1000, len(val_X))),
            torch.zeros(min(1000, len(val_X))),
        )
        val_loader = DataLoader(ds, batch_size=256, shuffle=False)
        shap_dir = out_dir / "shap"
        shap_dir.mkdir(exist_ok=True)
        run_shap_analysis(model, val_loader, feature_names, shap_dir, device)

    # 7. Summary table
    print("\n" + "─" * 80)
    print("  SUMMARY TABLE (30/60/90/120-min horizons)")
    print("─" * 80)
    cols = ["horizon_min", "RMSE_mg_dL", "MAE_mg_dL", "TIR_%", "EGA_A_%"]
    print(metrics_df[cols].to_string(index=False))
    print("\n  All results saved to:", out_dir)


if __name__ == "__main__":
    main()
