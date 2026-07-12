"""
Fine-tune the pretrained GlucoseTransformer on the OhioT1DM training split,
then evaluate on the test split and save results to results/ohio_finetuned/.

Strategy:
  - Keep the simulation-trained weights as starting point
  - Keep the simulation scaler (do NOT refit — model was trained on those statistics)
  - Fine-tune with small LR to correct the distribution shift
  - Save fine-tuned checkpoint to checkpoints/best_model_ohio_ft.pt
"""

import sys, pickle, logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ode_features import build_features, make_sequences, FEATURE_NAMES, N_FEATURES
from src.models.glucose_predictor import GlucosePredictor
from scripts.evaluate_ohio import (
    parse_ohio_xml, run_inference, rmse, mae, mard,
    tir_metrics, ega_zones, clarke_zone,
    plot_clarke, plot_scatter_grid, plot_error_hist, plot_tir_bars, plot_traces,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_IN  = ROOT / "checkpoints/best_model.pt"
CHECKPOINT_OUT = ROOT / "checkpoints/best_model_ohio_ft.pt"
OHIO_DIR       = ROOT / "OhioT1DM"
OUT_DIR        = ROOT / "results/ohio_finetuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS  = [6, 12, 18, 24]
SEQ_LEN   = 24

LR          = 5e-5   # small to avoid catastrophic forgetting
WEIGHT_DECAY = 0.01
BATCH_SIZE  = 64
MAX_EPOCHS  = 30
PATIENCE    = 6


def load_base_model(device):
    @dataclass
    class TrainingConfig:
        batch_size: int = 32; learning_rate: float = 1e-3; weight_decay: float = 0.01
        epochs: int = 100; early_stopping_patience: int = 15; val_split: float = 0.2
        gradient_clip: float = 1.0; model_type: str = "transformer"
        hidden_size: int = 128; num_layers: int = 4; num_heads: int = 8
        dropout: float = 0.1; use_pinn: bool = True; pinn_lambda: float = 0.1
        checkpoint_dir: str = "./checkpoints"

    import __main__
    __main__.TrainingConfig = TrainingConfig

    ckpt = torch.load(CHECKPOINT_IN, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = GlucosePredictor(
        input_size=N_FEATURES,
        model_type=getattr(config, "model_type", "transformer"),
        hidden_size=getattr(config, "hidden_size", 128),
        num_layers=getattr(config, "num_layers", 4),
        num_horizons=4,
        dropout=getattr(config, "dropout", 0.1),
        use_pinn=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    scaler_bytes = ckpt.get("scaler")
    if scaler_bytes is None:
        raise RuntimeError("No scaler in checkpoint.")
    scaler = pickle.loads(scaler_bytes)

    logger.info("Loaded pretrained model — val MAE: %.2f mg/dL",
                ckpt.get("metrics", {}).get("val_mae", float("nan")))
    return model, scaler, ckpt


def build_ohio_dataset(xml_files, scaler):
    """Parse Ohio XMLs → scaled (X, y) tensors."""
    all_X, all_y = [], []
    for xml_path in sorted(xml_files):
        pid = xml_path.stem.split("-")[0]
        df = parse_ohio_xml(xml_path)
        if df.empty or len(df) < SEQ_LEN + max(HORIZONS) + 10:
            logger.warning("  Skipping %s — insufficient data", pid)
            continue

        feat_df = df[["t_min", "cgm_mg_dl", "insulin_u_h", "cho_g", "basal_u_h"]].copy()
        feat_df = feat_df[~df["cgm_mg_dl"].isna()].reset_index(drop=True)

        if len(feat_df) < SEQ_LEN + max(HORIZONS) + 10:
            continue

        feat_matrix = build_features(feat_df)
        # Inject real exercise features
        df_clean = df[~df["cgm_mg_dl"].isna()].reset_index(drop=True)
        ex_cols = ["is_exercising", "exercise_intensity", "time_since_exercise", "exercise_minutes_2h"]
        for i, col in enumerate(ex_cols):
            if col in df_clean.columns:
                feat_matrix[:, 31 + i] = df_clean[col].values.astype(np.float32)

        glucose = feat_df["cgm_mg_dl"].values.astype(np.float32)
        X, y = make_sequences(feat_matrix, glucose, SEQ_LEN, HORIZONS)
        if len(X) == 0:
            continue

        # Apply the simulation scaler
        flat = X.reshape(-1, N_FEATURES)
        X_scaled = scaler.transform(flat).reshape(X.shape).astype(np.float32)

        all_X.append(X_scaled)
        all_y.append(y)
        logger.info("  Patient %s: %d sequences", pid, len(X))

    if not all_X:
        return None, None

    X_cat = np.concatenate(all_X, axis=0)
    y_cat = np.concatenate(all_y, axis=0)
    return torch.from_numpy(X_cat), torch.from_numpy(y_cat)


def fine_tune(model, train_loader, val_X, val_y, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR * 0.1)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    patience_count = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = []
            for i in range(0, len(val_X), 512):
                out = model(val_X[i:i+512].to(device)).cpu()
                val_pred.append(out)
            val_pred = torch.cat(val_pred, dim=0).numpy()

        val_mae_30 = float(np.mean(np.abs(val_pred[:, 0] - val_y[:, 0].numpy())))
        val_loss = float(np.mean((val_pred - val_y.numpy()) ** 2))
        train_loss = float(np.mean(train_losses))
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_mae_30": val_mae_30})
        logger.info("Epoch %2d | train_loss=%.2f | val_loss=%.2f | val_MAE_30=%.2f",
                    epoch, train_loss, val_loss, val_mae_30)

        if val_mae_30 < best_val_mae - 0.1:
            best_val_mae = val_mae_30
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info("Early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    logger.info("Best val MAE (30 min): %.2f mg/dL", best_val_mae)
    return model, pd.DataFrame(history)


def plot_training_curve(history, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["val_mae_30"], "o-", color="#1f6f9a",
            lw=2, ms=5, label="Val MAE (30 min)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("Fine-tuning convergence — OhioT1DM", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, scaler, orig_ckpt = load_base_model(device)

    # ── Build fine-tuning dataset from train XMLs ─────────────────────────────
    train_files_2018 = sorted((OHIO_DIR / "2018" / "train").glob("*.xml"))
    train_files_2020 = sorted((OHIO_DIR / "2020" / "train").glob("*.xml"))
    all_train_files  = train_files_2018 + train_files_2020

    logger.info("Building dataset from %d training patients…", len(all_train_files))
    X_all, y_all = build_ohio_dataset(all_train_files, scaler)

    if X_all is None:
        logger.error("No training data built — check Ohio paths.")
        return

    logger.info("Total training sequences: %d", len(X_all))

    # Patient-level val split: hold out last 20% of samples as val
    # (time-ordered, so this is the most recent data within the training set)
    n_val = max(1, int(len(X_all) * 0.15))
    # Shuffle first to mix patients
    perm = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(42))
    X_all, y_all = X_all[perm], y_all[perm]
    X_train, y_train = X_all[n_val:], y_all[n_val:]
    X_val, y_val = X_all[:n_val], y_all[:n_val]

    logger.info("Train sequences: %d | Val sequences: %d", len(X_train), len(X_val))

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    model, history = fine_tune(model, train_loader, X_val, y_val, device)

    # ── Save fine-tuned checkpoint ────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": orig_ckpt["config"],
        "scaler": orig_ckpt["scaler"],
        "feature_names": orig_ckpt.get("feature_names"),
        "metrics": {
            "val_mae": float(history["val_mae_30"].min()),
            "finetune_epochs": len(history),
        },
    }, CHECKPOINT_OUT)
    logger.info("Saved fine-tuned checkpoint to %s", CHECKPOINT_OUT)

    plot_training_curve(history, OUT_DIR / "finetune_curve.png")

    # ── Evaluate on test split ────────────────────────────────────────────────
    test_files = (
        sorted((OHIO_DIR / "2018" / "test").glob("*.xml")) +
        sorted((OHIO_DIR / "2020" / "test").glob("*.xml"))
    )
    logger.info("Evaluating on %d test patients…", len(test_files))

    model.eval()
    all_preds, all_targets = [], []
    patient_results, patient_data = [], {}

    for xml_path in sorted(test_files):
        pid = xml_path.stem.split("-")[0]
        year = xml_path.parts[-3]

        df = parse_ohio_xml(xml_path)
        if df.empty or len(df) < SEQ_LEN + max(HORIZONS) + 10:
            logger.warning("  Skipped %s — insufficient data", pid)
            continue

        preds, targets = run_inference(df, model, scaler, device)
        if len(preds) == 0:
            logger.warning("  No valid sequences for patient %s", pid)
            continue

        logger.info("  %s | seqs=%d | RMSE_30=%.1f | MAE_30=%.1f",
                    pid, len(preds), rmse(targets[:,0], preds[:,0]), mae(targets[:,0], preds[:,0]))

        all_preds.append(preds)
        all_targets.append(targets)
        patient_data[f"{pid}({year})"] = (preds, targets)

        row = {"patient_id": pid, "year": year, "n_sequences": len(preds)}
        for j, h in enumerate([30, 60, 90, 120]):
            p, t = preds[:, j], targets[:, j]
            zones = ega_zones(t, p)
            row[f"RMSE_{h}min"] = round(rmse(t, p), 2)
            row[f"MAE_{h}min"]  = round(mae(t, p), 2)
            row[f"EGA_A_{h}min"] = round(zones["A"], 1)
        patient_results.append(row)

    all_preds   = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metric_rows = []
    for j, h in enumerate([30, 60, 90, 120]):
        p, t = all_preds[:, j], all_targets[:, j]
        zones = ega_zones(t, p)
        ta = tir_metrics(t)
        tp = tir_metrics(p)
        row = {
            "horizon_min": h,
            "RMSE_mg_dL": round(rmse(t, p), 2),
            "MAE_mg_dL":  round(mae(t, p), 2),
            "R2":         round(r2_score(t, p), 4),
            "MARD_%":     round(mard(t, p), 2),
            "TIR_actual_%": round(ta["tir"], 1),
            "TIR_pred_%":   round(tp["tir"], 1),
            "EGA_A_%": round(zones["A"], 1),
            "EGA_B_%": round(zones["B"], 1),
            "EGA_C_%": round(zones["C"], 1),
            "EGA_D_%": round(zones["D"], 1),
        }
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    per_patient_df = pd.DataFrame(patient_results)
    metrics_df.to_csv(OUT_DIR / "metrics_ohio_ft.csv", index=False)
    per_patient_df.to_csv(OUT_DIR / "per_patient_ohio_ft.csv", index=False)

    print("\n=== Fine-tuned results ===")
    print(metrics_df.to_string(index=False))

    # ── Zero-shot vs fine-tuned comparison ────────────────────────────────────
    zs = pd.read_csv(ROOT / "results/ohio/metrics_ohio.csv")
    print("\n=== Zero-shot vs Fine-tuned (30 min) ===")
    for col in ["RMSE_mg_dL", "MAE_mg_dL", "EGA_A_%"]:
        zv = float(zs.loc[zs["horizon_min"]==30, col].values[0])
        fv = float(metrics_df.loc[metrics_df["horizon_min"]==30, col].values[0])
        print(f"  {col}: {zv:.1f} → {fv:.1f}  (Δ={fv-zv:+.1f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_clarke(all_targets[:, 0], all_preds[:, 0], 30,  OUT_DIR / "clarke_ft_30min.png")
    plot_clarke(all_targets[:, 1], all_preds[:, 1], 60,  OUT_DIR / "clarke_ft_60min.png")
    plot_scatter_grid(all_preds, all_targets,             OUT_DIR / "scatter_ft.png")
    plot_error_hist(all_preds, all_targets,               OUT_DIR / "error_hist_ft.png")
    plot_tir_bars(all_preds, all_targets,                 OUT_DIR / "tir_bars_ft.png")
    plot_traces(patient_data,                             OUT_DIR / "traces_ft.png")
    plot_three_way(zs, metrics_df,                        OUT_DIR / "three_way_comparison.png")

    logger.info("Done — results in %s", OUT_DIR)


def plot_three_way(zs_df, ft_df, out_path):
    """Bar chart: in-silico vs zero-shot vs fine-tuned across horizons."""
    sim = pd.read_csv(ROOT / "results/metrics.csv")
    horizons = [30, 60, 90, 120]
    x = np.arange(4)
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pairs = [
        ("RMSE_mg_dL", "RMSE (mg/dL)"),
        ("MAE_mg_dL",  "MAE (mg/dL)"),
        ("EGA_A_%",    "Clarke A zone (%)"),
    ]
    colors = ["#1f6f9a", "#e74c3c", "#2ecc71"]

    for ax, (col, title) in zip(axes, pairs):
        sim_vals = [float(sim.loc[sim["horizon_min"]==h, col].values[0]) for h in horizons]
        zs_vals  = [float(zs_df.loc[zs_df["horizon_min"]==h, col].values[0]) for h in horizons]
        ft_vals  = [float(ft_df.loc[ft_df["horizon_min"]==h, col].values[0]) for h in horizons]

        ax.bar(x - width, sim_vals, width, label="In-silico", color=colors[0], alpha=0.85)
        ax.bar(x,          zs_vals, width, label="Zero-shot",  color=colors[1], alpha=0.85)
        ax.bar(x + width,  ft_vals, width, label="Fine-tuned", color=colors[2], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}min" for h in horizons])
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("In-silico → Zero-shot → Fine-tuned on OhioT1DM", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
