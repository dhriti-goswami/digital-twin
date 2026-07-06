#!/usr/bin/env python3
"""
Comprehensive evaluation of the trained GlucoseTransformer.

Produces:
  - Scatter plots (actual vs predicted, all 4 horizons)
  - Horizon RMSE/MAE bar chart
  - Prediction traces
  - Clarke Error Grid Analysis (30-min and 90-min)
  - Learning curves
  - TIR/TAR/TBR stacked bar chart (actual vs predicted)
  - Error distribution histograms
  - Per-cohort (adolescent/adult/child) performance table
  - Persistence baseline comparison
  - Meal scenario simulation (3 meal sizes × 3 ICR ratios)
  - Clinical summary table (mean±SD)
  - SHAP feature importance (optional)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --shap
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ode_features import load_all_patients, build_features, make_sequences, FEATURE_NAMES
from src.models.glucose_predictor import GlucosePredictor


def setup_logging(verbose: bool = True):
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
HORIZONS = [30, 60, 90, 120]
COHORT_PREFIXES = {"adolescent": "Adolescent", "adult": "Adult", "child": "Child"}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device):
    """Load model directly from checkpoint (bypasses GlucoseInferenceService)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    feature_names = ckpt.get("feature_names", FEATURE_NAMES)
    n_features = len(feature_names)

    model = GlucosePredictor(
        input_size=n_features,
        model_type=getattr(config, "model_type", "transformer"),
        hidden_size=getattr(config, "hidden_size", 128),
        num_layers=getattr(config, "num_layers", 4),
        num_horizons=4,
        dropout=getattr(config, "dropout", 0.1),
        use_pinn=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = pickle.loads(ckpt["scaler"]) if ckpt.get("scaler") else None
    metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", "?")
    logger.info(
        "Model loaded — epoch=%s  val_mae=%.3f  input_size=%d",
        epoch, metrics.get("val_mae", float("nan")), n_features,
    )
    return model, scaler, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Data loading with per-patient tracking
# ──────────────────────────────────────────────────────────────────────────────

def load_val_per_patient(sim_dir: Path, scaler, seq_len: int = 24, horizons=None, val_split=0.2, seed=42):
    """
    Load validation patients individually, returning sequences grouped by patient.

    Returns list of dicts: [{name, cohort, X, y}, ...]
    """
    if horizons is None:
        horizons = [6, 12, 18, 24]

    files = sorted(sim_dir.glob("*.csv"))
    rng = np.random.RandomState(seed)
    idxs = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_split))
    val_files = [files[i] for i in idxs[:n_val]]

    patients = []
    for f in sorted(val_files):
        df = pd.read_csv(f)
        feats = build_features(df)
        glucose = df["cgm_mg_dl"].values.astype(np.float32)
        X, y = make_sequences(feats, glucose, seq_len, horizons)
        if len(X) == 0:
            continue
        # Scale features using the training scaler
        flat = X.reshape(-1, X.shape[2])
        X_scaled = scaler.transform(flat).reshape(X.shape).astype(np.float32)

        name = f.stem
        cohort = next((k for k in COHORT_PREFIXES if name.startswith(k)), "unknown")
        patients.append({"name": name, "cohort": cohort, "X": X_scaled, "y": y})

    return patients


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i: i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def persistence_baseline(scaler, val_patients: list) -> np.ndarray:
    """
    Persistence baseline: predict the CURRENT glucose for all horizons.

    Current glucose = feature index 0 (glucose_mg_dl), last timestep, unscaled.
    """
    all_current = []
    for p in val_patients:
        X_scaled = p["X"]          # (N, seq_len, n_features)
        last_step_scaled = X_scaled[:, -1, :]  # (N, n_features)
        # inverse-transform the full feature vector, then take feature 0
        last_step_orig = scaler.inverse_transform(last_step_scaled)
        current_g = last_step_orig[:, 0]   # glucose_mg_dl is feature 0
        all_current.append(current_g)

    current_g = np.concatenate(all_current)
    return np.stack([current_g] * 4, axis=1)   # (N, 4) — same value for all horizons


# ──────────────────────────────────────────────────────────────────────────────
# Clinical metrics
# ──────────────────────────────────────────────────────────────────────────────

def tir_metrics(glucose: np.ndarray) -> dict:
    tir  = np.mean((glucose >= 70) & (glucose <= 180)) * 100
    tar1 = np.mean((glucose > 180) & (glucose <= 250)) * 100
    tar2 = np.mean(glucose > 250) * 100
    tbr1 = np.mean((glucose >= 54) & (glucose < 70)) * 100
    tbr2 = np.mean(glucose < 54) * 100
    return {"TIR": tir, "TAR1": tar1, "TAR2": tar2, "TBR1": tbr1, "TBR2": tbr2}


def lbgi_hbgi(glucose: np.ndarray) -> tuple:
    f = 1.509 * (np.log(np.maximum(glucose, 1)) ** 1.084 - 5.381)
    rl = 10 * f ** 2 * (f < 0).astype(float)
    rh = 10 * f ** 2 * (f > 0).astype(float)
    return float(rl.mean()), float(rh.mean())


def clarke_ega(actual: np.ndarray, predicted: np.ndarray) -> dict:
    n = len(actual)
    zones = np.full(n, "B", dtype="<U1")
    for i in range(n):
        r, p = float(actual[i]), float(predicted[i])
        if (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
            zones[i] = "E"
        elif (r <= 70 and 70 <= p <= 180) or (r >= 240 and 70 <= p <= 180):
            zones[i] = "D"
        elif r <= 70 and p <= 70:
            zones[i] = "A"
        elif r >= 290:
            zones[i] = "A" if p >= 290 * 0.7 else "B"
        elif abs(p - r) / r <= 0.20:
            zones[i] = "A"
        elif 130 <= r <= 180 and p > 1.30 * r:
            zones[i] = "C"
        elif 70 <= r <= 130 and p < 0.70 * r:
            zones[i] = "C"
    counts = {z: int(np.sum(zones == z)) for z in "ABCDE"}
    pcts = {z: counts[z] / n * 100 for z in "ABCDE"}
    return {"counts": counts, "pcts": pcts, "zones": zones}


def compute_per_patient_metrics(patients_data: list, model, device: torch.device) -> pd.DataFrame:
    """Compute RMSE/MAE per patient, with cohort labels."""
    rows = []
    for p in patients_data:
        preds = run_inference(model, p["X"], device)
        targets = p["y"]
        for j, h in enumerate(HORIZONS):
            err = preds[:, j] - targets[:, j]
            rows.append({
                "patient": p["name"],
                "cohort": COHORT_PREFIXES.get(p["cohort"], "Unknown"),
                "horizon_min": h,
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "MAE": float(np.mean(np.abs(err))),
                "n": len(preds),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(history: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    ax.plot(history["val_loss"], label="Val", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["val_mae"], label="Val MAE", linewidth=1.5, color="green")
    ax.plot(history["val_rmse"], label="Val RMSE", linewidth=1.5, color="orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mg/dL")
    ax.set_title("Validation MAE / RMSE"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)


def plot_scatter(preds: np.ndarray, targets: np.ndarray, out_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for j, (h, ax) in enumerate(zip(HORIZONS, axes)):
        p, t = preds[:, j], targets[:, j]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae  = float(np.mean(np.abs(p - t)))
        r2   = float(1 - np.sum((p - t) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-9))
        lim_lo = max(30,  min(p.min(), t.min()) - 10)
        lim_hi = min(420, max(p.max(), t.max()) + 10)
        ax.scatter(t, p, alpha=0.20, s=2, color="#2196F3", rasterized=True)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", linewidth=1.5)
        ax.set_xlabel("Reference (mg/dL)", fontsize=9)
        ax.set_ylabel("Predicted (mg/dL)", fontsize=9)
        ax.set_title(f"{h}-min", fontsize=10)
        ax.text(0.05, 0.95, f"RMSE={rmse:.1f}\nMAE={mae:.1f}\nR²={r2:.3f}",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    fig.suptitle("Actual vs. Predicted Glucose — All Horizons", fontsize=11, y=1.01)
    fig.tight_layout()
    path = out_dir / "scatter_actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)


def plot_horizon_rmse(preds: np.ndarray, targets: np.ndarray, pers_preds: np.ndarray, out_dir: Path):
    rmse_model = [float(np.sqrt(np.mean((preds[:, j] - targets[:, j]) ** 2))) for j in range(4)]
    mae_model  = [float(np.mean(np.abs(preds[:, j] - targets[:, j])))         for j in range(4)]
    rmse_pers  = [float(np.sqrt(np.mean((pers_preds[:, j] - targets[:, j]) ** 2))) for j in range(4)]

    x = np.arange(4); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w, rmse_model, w, label="Transformer RMSE", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x,     mae_model,  w, label="Transformer MAE",  color="#FF9800", alpha=0.85)
    b3 = ax.bar(x + w, rmse_pers,  w, label="Persistence RMSE", color="#9E9E9E", alpha=0.85, hatch="//")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b3, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{h} min" for h in HORIZONS])
    ax.set_ylabel("mg/dL"); ax.set_title("Prediction Error by Horizon vs Persistence Baseline")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "horizon_rmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)


def plot_prediction_traces(preds: np.ndarray, targets: np.ndarray, out_dir: Path, n: int = 3):
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1: axes = [axes]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for i, ax in enumerate(axes):
        idx = i * (len(targets) // n)
        actual = targets[max(0, idx - 6): idx + 1, 0]
        ax.plot(range(len(actual)), actual, "k-", linewidth=1.5, label="Actual (30-min)")
        for j, (h, c) in enumerate(zip(HORIZONS, colors)):
            ax.plot(len(actual) - 1 + j + 1, preds[idx, j], "o", color=c,
                    markersize=8, label=f"Pred {h}min")
        ax.axvline(len(actual) - 1, color="gray", linestyle="--", alpha=0.5)
        ax.axhspan(70, 180, alpha=0.08, color="green")
        ax.axhline(70,  color="orange", linestyle=":", alpha=0.7, linewidth=0.8)
        ax.axhline(180, color="orange", linestyle=":", alpha=0.7, linewidth=0.8)
        ax.set_ylim(40, 350); ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(f"Sample {i+1}"); ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Timestep (5 min)")
    fig.tight_layout()
    path = out_dir / "prediction_traces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)


def plot_clarke_ega(actual: np.ndarray, predicted: np.ndarray, out_dir: Path, horizon_label: str = "30min"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ega = clarke_ega(actual, predicted)
    zone_colours = {"A": "#4CAF50", "B": "#8BC34A", "C": "#FF9800", "D": "#FF5722", "E": "#F44336"}
    for z in "ABCDE":
        mask = ega["zones"] == z
        if mask.sum() > 0:
            ax.scatter(actual[mask], predicted[mask],
                       alpha=0.5, s=4, c=zone_colours[z],
                       label=f"Zone {z} ({ega['pcts'][z]:.1f}%)", rasterized=True)
    xs = np.linspace(0, 400, 200)
    ax.plot(xs, xs, "k--", linewidth=1.2)
    ax.plot(xs, xs * 1.20, "--", color="gray", linewidth=0.8, alpha=0.7)
    ax.plot(xs, xs * 0.80, "--", color="gray", linewidth=0.8, alpha=0.7)
    for v in [70, 180]:
        ax.axvline(v, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.axhline(v, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=10)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=10)
    ax.set_title(f"Clarke Error Grid — {horizon_label}", fontsize=11)
    for z, txt_xy in [("A",(15,340)),("B",(200,350)),("C",(150,320)),("D",(340,80)),("E",(80,360))]:
        ax.text(*txt_xy, z, fontsize=14, fontweight="bold", color=zone_colours[z], alpha=0.5)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9, markerscale=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = out_dir / f"clarke_ega_{horizon_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)
    return ega


def plot_tir_bars(preds: np.ndarray, targets: np.ndarray, out_dir: Path):
    """Stacked bar chart comparing actual vs predicted TIR/TAR/TBR at each horizon."""
    categories = ["TBR2\n(<54)", "TBR1\n(54–70)", "TIR\n(70–180)", "TAR1\n(180–250)", "TAR2\n(>250)"]
    colors     = ["#D32F2F", "#FF7043", "#66BB6A", "#FFA726", "#EF5350"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    x = np.arange(2)
    width = 0.5

    for j, (h, ax) in enumerate(zip(HORIZONS, axes)):
        t_vals = targets[:, j]
        p_vals = preds[:, j]

        def get_fracs(g):
            return [
                np.mean(g < 54) * 100,
                np.mean((g >= 54) & (g < 70)) * 100,
                np.mean((g >= 70) & (g <= 180)) * 100,
                np.mean((g > 180) & (g <= 250)) * 100,
                np.mean(g > 250) * 100,
            ]

        actual_fracs = get_fracs(t_vals)
        pred_fracs   = get_fracs(p_vals)

        bottoms = [0.0, 0.0]
        for ci, (cat, col) in enumerate(zip(categories, colors)):
            vals = [actual_fracs[ci], pred_fracs[ci]]
            bars = ax.bar(x, vals, width, bottom=bottoms, color=col, label=cat if j == 0 else "")
            for bar, val in zip(bars, vals):
                if val > 4:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + val/2,
                            f"{val:.0f}%", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax.set_xticks(x); ax.set_xticklabels(["Actual", "Predicted"], fontsize=9)
        ax.set_title(f"{h}-min", fontsize=10); ax.set_ylim(0, 105)
        if j == 0: ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.2)

    handles = [mpatches.Patch(color=c, label=cat) for cat, c in zip(categories, colors)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.suptitle("Glycaemic Distribution — Actual vs. Predicted", fontsize=12, y=1.01)
    fig.tight_layout()
    path = out_dir / "tir_bars_actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)


def plot_error_histograms(preds: np.ndarray, targets: np.ndarray, out_dir: Path):
    """Per-horizon prediction error distributions."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for j, (h, ax, col) in enumerate(zip(HORIZONS, axes, colors)):
        err = preds[:, j] - targets[:, j]
        ax.hist(err, bins=60, color=col, alpha=0.75, edgecolor="none", density=True)
        mu, sigma = err.mean(), err.std()
        x = np.linspace(err.min(), err.max(), 200)
        ax.plot(x, np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi)),
                "k-", linewidth=1.5, label=f"N({mu:.1f},{sigma:.1f})")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Error (mg/dL)", fontsize=9)
        ax.set_ylabel("Density" if j == 0 else "", fontsize=9)
        ax.set_title(f"{h}-min: μ={mu:.1f} σ={sigma:.1f}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle("Prediction Error Distributions", fontsize=12, y=1.01)
    fig.tight_layout()
    path = out_dir / "error_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)


def plot_cohort_metrics(per_patient_df: pd.DataFrame, out_dir: Path):
    """Per-cohort RMSE bar chart across prediction horizons."""
    cohorts = ["Adolescent", "Adult", "Child"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax in zip(["RMSE", "MAE"], axes):
        df_pivot = (per_patient_df.groupby(["cohort", "horizon_min"])[metric]
                                  .mean().reset_index()
                                  .pivot(index="horizon_min", columns="cohort", values=metric))
        x = np.arange(4)
        w = 0.25
        pal = {"Adolescent": "#5C6BC0", "Adult": "#26A69A", "Child": "#EF5350"}
        for ci, cohort in enumerate([c for c in cohorts if c in df_pivot.columns]):
            offset = (ci - 1) * w
            bars = ax.bar(x + offset, df_pivot[cohort], w,
                          label=cohort, color=pal.get(cohort, "#999"), alpha=0.85)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{h}min" for h in HORIZONS])
        ax.set_ylabel("mg/dL"); ax.set_title(f"Per-Cohort {metric}")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Performance by Patient Cohort", fontsize=12)
    fig.tight_layout()
    path = out_dir / "cohort_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)

    # Also save cohort CSV
    tbl = per_patient_df.groupby(["cohort", "horizon_min"])[["RMSE", "MAE"]].mean().round(2)
    tbl.to_csv(out_dir / "cohort_metrics.csv")
    logger.info("Saved: %s", out_dir / "cohort_metrics.csv")


def plot_meal_scenario(out_dir: Path, n_patients: int = 10):
    """
    Meal scenario simulation figure — mirrors paper Figure 4.

    3 panels (30g / 60g / 90g carbs) × 3 lines (ICR 8/15/25 g/U).
    Runs the UVA/Padova ODE for n_patients adult patients.
    Shows median ± IQR bands.
    """
    try:
        from src.models.t1d_ode import T1DPatient, list_patient_names
    except ImportError as e:
        logger.warning("Cannot generate meal scenario: %s", e)
        return

    # Use adult patients only for stable, realistic responses
    all_names = list_patient_names()
    patient_names = [n for n in all_names if "adult" in n][:n_patients]

    meal_sizes = [30, 60, 90]
    icr_values = [8, 15, 25]
    icr_colors = ["#1565C0", "#2E7D32", "#E65100"]

    STEPS = 60      # 5h window at 5-min intervals
    PRE   = 12      # 1h pre-meal baseline steps
    TOTAL = PRE + STEPS

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, carbs in zip(axes, meal_sizes):
        for icr, col in zip(icr_values, icr_colors):
            traces = []
            for pname in patient_names:
                try:
                    patient = T1DPatient.from_name(pname)
                    patient.reset()
                    basal = patient.basal_rate

                    # 1 h pre-meal warm-up at basal only
                    for _ in range(PRE):
                        patient.step(0.0, basal)

                    traj = [patient.cgm]
                    # Give meal + bolus at t=0
                    bolus_u = carbs / icr
                    bolus_u_min = bolus_u / 5.0

                    for step in range(1, STEPS + 1):
                        cho_s = float(carbs) if step == 1 else 0.0
                        ins_s = basal + (bolus_u_min if step == 1 else 0.0)
                        patient.step(cho_s, ins_s)
                        traj.append(max(40.0, min(400.0, patient.cgm)))

                    traces.append(traj)
                except Exception:
                    continue

            if not traces:
                continue

            traces = np.array(traces)   # (n_patients, STEPS+1)
            t_hours = np.arange(len(traces[0])) * 5 / 60 - (PRE * 5 / 60)
            med = np.median(traces, axis=0)
            q25 = np.percentile(traces, 25, axis=0)
            q75 = np.percentile(traces, 75, axis=0)

            ax.plot(t_hours, med, color=col, linewidth=2,
                    label=f"ICR {icr} g/U")
            ax.fill_between(t_hours, q25, q75, color=col, alpha=0.15)

        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.axhspan(70, 180, alpha=0.07, color="green")
        ax.axhline(70,  color="orange", linestyle=":", linewidth=0.8)
        ax.axhline(180, color="orange", linestyle=":", linewidth=0.8)
        ax.set_title(f"Meal: {carbs} g Carbohydrates", fontsize=10)
        ax.set_xlabel("Time from Meal (h)", fontsize=9)
        ax.set_xlim(t_hours[0], t_hours[-1])
        ax.set_ylim(40, 350)
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("Glucose (mg/dL)", fontsize=9)
            ax.legend(title="Insulin-to-Carb\nRatio", fontsize=8, title_fontsize=8)

    fig.suptitle("Simulated Glucose Response to Meal Scenarios\n(UVA/Padova ODE, median ± IQR)",
                 fontsize=11)
    fig.tight_layout()
    path = out_dir / "meal_scenario_simulation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved: %s", path)


def run_shap_analysis(model, val_X: np.ndarray, feature_names: list, output_dir: Path, device: torch.device):
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap")
        return

    logger.info("Running SHAP analysis …")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    X_sample = torch.from_numpy(val_X[:200]).to(device)
    X_last = X_sample[:, -1, :].cpu().numpy()

    def model_predict(X_np):
        with torch.no_grad():
            X_seq = np.tile(X_np[:, np.newaxis, :], (1, X_sample.shape[1], 1))
            return model(torch.FloatTensor(X_seq).to(device)).cpu().numpy()

    background = X_last[:50]
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_last[:100], nsamples=100)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) > 2:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)
    shap_values = shap_values[:, :len(feature_names)]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_last[:100], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    logger.info("SHAP summary → %s", output_dir / "shap_summary.png")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_last[:100], feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()
    logger.info("SHAP importance → %s", output_dir / "shap_importance.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained GlucoseTransformer")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "best_model.pt"))
    p.add_argument("--sim-dir",  default=str(PROJECT_ROOT / "data" / "raw" / "simulated"))
    p.add_argument("--out-dir",  default=str(RESULTS_DIR))
    p.add_argument("--shap",     action="store_true", help="Run SHAP analysis")
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

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    # 1. Load model + scaler from checkpoint
    model, scaler, feature_names = load_model(checkpoint_path, device)

    # 2. Load validation data
    data = load_all_patients(Path(args.sim_dir), seq_len=args.seq_length)
    val_X, val_y = data["val_X"], data["val_y"]
    logger.info("Validation set: %d sequences", len(val_X))

    # Per-patient data for cohort analysis
    val_patients = load_val_per_patient(
        Path(args.sim_dir), scaler, seq_len=args.seq_length
    )
    logger.info("Val patients loaded: %d", len(val_patients))

    # 3. Inference
    logger.info("Running inference …")
    preds   = run_inference(model, val_X, device)
    targets = val_y
    pers    = persistence_baseline(scaler, val_patients)

    # 4. Compute metrics
    print("\n" + "─" * 80)
    print("  METRICS PER HORIZON")
    print("─" * 80)
    metrics_rows = []
    for j, h in enumerate(HORIZONS):
        p, t = preds[:, j], targets[:, j]
        pp   = pers[:, j]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae  = float(np.mean(np.abs(p - t)))
        mard = float(np.mean(np.abs(p - t) / np.maximum(t, 1e-9)) * 100)
        r2   = float(1 - np.sum((p - t) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-9))
        rmse_pers = float(np.sqrt(np.mean((pp - t) ** 2)))

        tir_act  = tir_metrics(t)
        tir_pred = tir_metrics(p)
        lbgi_a, hbgi_a = lbgi_hbgi(t)
        lbgi_p, hbgi_p = lbgi_hbgi(p)
        ega = clarke_ega(t, p)

        metrics_rows.append({
            "horizon_min": h,
            "RMSE_mg_dL": round(rmse, 2),
            "MAE_mg_dL": round(mae, 2),
            "R2": round(r2, 4),
            "MARD_%": round(mard, 2),
            "RMSE_persistence": round(rmse_pers, 2),
            "TIR_actual_%": round(tir_act["TIR"], 1),
            "TIR_pred_%": round(tir_pred["TIR"], 1),
            "TAR_actual_%": round(tir_act["TAR1"] + tir_act["TAR2"], 1),
            "TAR_pred_%": round(tir_pred["TAR1"] + tir_pred["TAR2"], 1),
            "TBR_actual_%": round(tir_act["TBR1"] + tir_act["TBR2"], 1),
            "TBR_pred_%": round(tir_pred["TBR1"] + tir_pred["TBR2"], 1),
            "LBGI_actual": round(lbgi_a, 2),
            "LBGI_pred": round(lbgi_p, 2),
            "HBGI_actual": round(hbgi_a, 2),
            "HBGI_pred": round(hbgi_p, 2),
            "MeanGluc_actual": round(float(t.mean()), 1),
            "MeanGluc_pred": round(float(p.mean()), 1),
            "EGA_A_%": round(ega["pcts"]["A"], 1),
            "EGA_B_%": round(ega["pcts"]["B"], 1),
        })

        print(f"\n  {h}-min:  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  (Persistence RMSE={rmse_pers:.2f})")
        print(f"          TIR: actual={tir_act['TIR']:.1f}%  predicted={tir_pred['TIR']:.1f}%")
        print(f"          TBR: actual={tir_act['TBR1']+tir_act['TBR2']:.1f}%  predicted={tir_pred['TBR1']+tir_pred['TBR2']:.1f}%")
        print(f"          Clarke EGA: A={ega['pcts']['A']:.1f}%  B={ega['pcts']['B']:.1f}%  C={ega['pcts']['C']:.1f}%  D={ega['pcts']['D']:.1f}%  E={ega['pcts']['E']:.1f}%")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics saved → %s", metrics_path)

    # 5. Per-patient / cohort metrics
    logger.info("Computing per-patient metrics …")
    per_patient_df = compute_per_patient_metrics(val_patients, model, device)
    per_patient_df.to_csv(out_dir / "per_patient_metrics.csv", index=False)
    logger.info("Per-patient metrics → %s", out_dir / "per_patient_metrics.csv")

    # 6. Generate all plots
    print("\n  Generating plots …")
    plot_scatter(preds, targets, out_dir)
    plot_horizon_rmse(preds, targets, pers, out_dir)
    plot_prediction_traces(preds, targets, out_dir)
    plot_clarke_ega(targets[:, 0], preds[:, 0], out_dir, horizon_label="30min")
    plot_clarke_ega(targets[:, 2], preds[:, 2], out_dir, horizon_label="90min")
    plot_tir_bars(preds, targets, out_dir)
    plot_error_histograms(preds, targets, out_dir)
    plot_cohort_metrics(per_patient_df, out_dir)
    plot_meal_scenario(out_dir)

    history_path = checkpoint_path.parent / "training_history.npz"
    if history_path.exists():
        hist = dict(np.load(history_path))
        plot_learning_curves(hist, out_dir)

    # 7. SHAP
    if args.shap:
        shap_dir = out_dir / "shap"
        run_shap_analysis(model, val_X, feature_names, shap_dir, device)

    # 8. Clinical summary table
    print("\n" + "─" * 80)
    print("  CLINICAL SUMMARY TABLE (actual vs predicted)")
    print("─" * 80)
    fmt = "  {:<10} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8}"
    print(fmt.format("Horizon", "RMSE", "MAE", "R²", "TIR-act", "TIR-pred", "EGA-A%", "Pers-RMSE"))
    print("  " + "-" * 78)
    for _, row in metrics_df.iterrows():
        print(fmt.format(
            f"{int(row['horizon_min'])}min",
            f"{row['RMSE_mg_dL']:.2f}",
            f"{row['MAE_mg_dL']:.2f}",
            f"{row['R2']:.4f}",
            f"{row['TIR_actual_%']:.1f}%",
            f"{row['TIR_pred_%']:.1f}%",
            f"{row['EGA_A_%']:.1f}%",
            f"{row['RMSE_persistence']:.2f}",
        ))
    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
