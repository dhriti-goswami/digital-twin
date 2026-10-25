"""Split-conformal recalibration of the final quantile-head model (S-10, round 4).

Problem (reviewer): below 80 mg/dL the model's q10/q90 quantiles are badly
miscalibrated -- the q0.10..q0.90 interval covers only 76.4% of truth at 30 min
(down to 53.3% at 120 min), and q0.10 sits ABOVE the true value 22.4% of the
time in that low-glucose subset (should be ~10% if q10 is well calibrated).

Method: per-horizon, per-tail ADDITIVE conformal shift, calibrated on the
VALIDATION split only (never test), applied to the test-set quantile
predictions. This is the single-quantile analogue of split conformal
prediction (Romano et al. 2019 do the symmetric CQR interval version; here we
calibrate q10 and q90 independently since the paper's existing coverage
analysis is per-quantile, not just per-interval).

For quantile level tau and horizon h:
    residual_i = y_i - q_tau_pred_i                  (validation set)
    correction = empirical (1 - tau)-quantile of residual_i   [for tau=0.10]
               = empirical tau-quantile of residual_i          [for tau=0.90]
    q_tau_test_recalibrated = q_tau_test_pred + correction

The correction for q10 is chosen so that, on the validation set itself,
P(y <= q10 + correction) = 0.10 exactly (by construction, since `correction`
is defined as the empirical quantile of the residual needed to hit that
nominal rate). Same idea for q90 at the 0.90 level.

No retraining; this re-runs the already-trained, already-selected checkpoint
(`artifacts/official/quantile/official/best_model.pt`) on the validation
loader to get validation-set quantile predictions (which were never saved to
disk), then applies the fitted correction to the existing test-set
predictions in `test_diagnostics.npz`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, load_corpus, FittedScaler
from twin.data.splits import official_split, verify_no_leakage
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.train.loop import load_checkpoint, predict_loader

RESOLVED_CONFIG = "artifacts/official/quantile/resolved_config.yaml"
CHECKPOINT = "artifacts/official/quantile/official/best_model.pt"
TEST_DIAGNOSTICS = "artifacts/official/quantile/official/test_diagnostics.npz"
HORIZONS_MIN = [30, 60, 90, 120]
LOW_GLUCOSE_THRESHOLD = 80.0
HYPO_ALARM_DAYS_OBSERVED_FALSE_ALARMS_PER_DAY = 2.2  # reported elsewhere, for reference only


def rebuild_val_test_loaders(config: Config):
    corpus = load_corpus(config)
    train_sets = {k: v.windows for k, v in corpus["train"].items()}
    test_sets = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=config.split.val_fraction,
        purge_steps=config.split.purge_steps,
    )
    verify_no_leakage(fold, train_sets, test_sets)
    return corpus, fold


def main() -> None:
    config = Config.from_yaml(RESOLVED_CONFIG)
    state = load_checkpoint(CHECKPOINT)
    scaler = FittedScaler.from_state_dict(state["scaler"])

    corpus, fold = rebuild_val_test_loaders(config)
    val_dataset = build_dataset(fold, "val", corpus, scaler, config)
    val_loader = build_loader(val_dataset, config, shuffle=False)

    model = PhysicsGuidedForecaster(len(scaler.feature_names), config)
    model.load_state_dict(state["model_state_dict"])
    device = config.resolve_device()
    model.to(device)

    val_eval = predict_loader(model, val_loader, device)
    val_targets = val_eval["targets"]  # (N_val, 4)
    val_quantiles = val_eval["quantile_predictions"]  # (N_val, 3, 4)
    quantile_levels = val_eval["quantile_levels"]  # [0.1, 0.5, 0.9]
    assert np.allclose(quantile_levels, [0.1, 0.5, 0.9]), quantile_levels

    test = np.load(TEST_DIAGNOSTICS)
    test_targets = test["targets"]  # (N_test, 4)
    test_quantiles = test["quantile_predictions"].copy()  # (N_test, 3, 4)

    corrections = {}  # (tail, horizon_idx) -> additive shift
    for h_idx, horizon in enumerate(HORIZONS_MIN):
        # q10: want P(y <= q10 + c) = 0.10 on validation.
        # residual = y - q10_pred; c = 0.10-quantile of residual (so that
        # y - q10_pred <= c happens 10% of the time, i.e. y <= q10_pred + c 10% of the time).
        resid_q10 = val_targets[:, h_idx] - val_quantiles[:, 0, h_idx]
        c_q10 = float(np.quantile(resid_q10, 0.10))
        # q90: want P(y <= q90 + c) = 0.90 on validation -> c = 0.90-quantile of residual.
        resid_q90 = val_targets[:, h_idx] - val_quantiles[:, 2, h_idx]
        c_q90 = float(np.quantile(resid_q90, 0.90))
        corrections[("q10", horizon)] = c_q10
        corrections[("q90", horizon)] = c_q90
        test_quantiles[:, 0, h_idx] = test["quantile_predictions"][:, 0, h_idx] + c_q10
        test_quantiles[:, 2, h_idx] = test["quantile_predictions"][:, 2, h_idx] + c_q90

    def coverage_stats(targets, quantiles, mask=None):
        rows = []
        for h_idx, horizon in enumerate(HORIZONS_MIN):
            y = targets[:, h_idx]
            q10 = quantiles[:, 0, h_idx]
            q90 = quantiles[:, 2, h_idx]
            sel = np.ones_like(y, dtype=bool) if mask is None else mask[:, h_idx]
            n = int(sel.sum())
            if n == 0:
                continue
            interval_cov = float(np.mean((y[sel] >= q10[sel]) & (y[sel] <= q90[sel])))
            q10_above_truth = float(np.mean(q10[sel] > y[sel]))  # should be ~10% in low-glucose subset miscalibration check
            q10_miss_rate = float(np.mean(y[sel] < q10[sel]))  # nominal 10%
            q90_coverage = float(np.mean(y[sel] <= q90[sel]))  # nominal 90%
            rows.append({
                "horizon_min": horizon, "n": n,
                "interval_80pct_coverage": interval_cov,
                "q10_above_truth_rate": q10_above_truth,
                "q10_miss_rate": q10_miss_rate,
                "q90_coverage": q90_coverage,
            })
        return pd.DataFrame(rows)

    overall_before = coverage_stats(test_targets, test["quantile_predictions"])
    overall_after = coverage_stats(test_targets, test_quantiles)

    low_mask = test_targets < LOW_GLUCOSE_THRESHOLD
    low_before = coverage_stats(test_targets, test["quantile_predictions"], mask=low_mask)
    low_after = coverage_stats(test_targets, test_quantiles, mask=low_mask)

    def alarm_metrics(quantiles, threshold=70.0):
        # Alarm fires if predicted q10 < threshold at the 30-min horizon.
        h_idx = HORIZONS_MIN.index(30)
        pred_alarm = quantiles[:, 0, h_idx] < threshold
        true_hypo = test_targets[:, h_idx] < threshold
        tp = int(np.sum(pred_alarm & true_hypo))
        fp = int(np.sum(pred_alarm & ~true_hypo))
        fn = int(np.sum(~pred_alarm & true_hypo))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        return sensitivity, precision, tp, fp, fn

    sens_before, prec_before, tp_b, fp_b, fn_b = alarm_metrics(test["quantile_predictions"])
    sens_after, prec_after, tp_a, fp_a, fn_a = alarm_metrics(test_quantiles)

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = overall_before.merge(overall_after, on=["horizon_min", "n"], suffixes=("_before", "_after"))
    overall.to_csv(out_dir / "conformal_recalibration.csv", index=False)

    low = low_before.merge(low_after, on="horizon_min", suffixes=("_before", "_after"), how="outer")
    low.to_csv(out_dir / "conformal_low_glucose_coverage.csv", index=False)

    corrections_df = pd.DataFrame(
        [{"tail": tail, "horizon_min": h, "additive_shift_mgdl": c} for (tail, h), c in corrections.items()]
    )
    corrections_df.to_csv(out_dir / "conformal_corrections.csv", index=False)

    alarm_df = pd.DataFrame([
        {"stage": "before", "threshold_mgdl": 70.0, "sensitivity": sens_before, "precision": prec_before, "tp": tp_b, "fp": fp_b, "fn": fn_b},
        {"stage": "after", "threshold_mgdl": 70.0, "sensitivity": sens_after, "precision": prec_after, "tp": tp_a, "fp": fp_a, "fn": fn_a},
    ])
    alarm_df.to_csv(out_dir / "conformal_alarm_metrics.csv", index=False)

    print("=== Conformal corrections (fit on validation, mg/dL) ===")
    print(corrections_df.to_string(index=False))
    print("\n=== Overall coverage, before vs after ===")
    print(overall.to_string(index=False))
    print("\n=== Low-glucose (<80 mg/dL) conditional coverage, before vs after ===")
    print(low.to_string(index=False))
    print("\n=== Hypoglycemia alarm (q10 < 70 mg/dL @ 30min), before vs after ===")
    print(alarm_df.to_string(index=False))


if __name__ == "__main__":
    main()
