"""Round-5 fix for TODO_ROUND5.md S-11 (Must).

Round 4's `scripts/run_extra_baselines_round4.py` tuned the persistence and
linear-extrapolation alarm decision thresholds on the TEST set to match the
model's false-alarm rate there (2.21 FA/day) -- a test-set leak the model's
own alarm does not have (its 70 mg/dL threshold was fixed in advance, before
ever touching the test set).

Fix: tune each baseline's threshold on the VALIDATION split only, to match
the MODEL'S OWN alarm's false-alarm rate on the validation split (not the
test-set rate), then evaluate sensitivity/specificity/false-alarms-per-day/
lead-time for the baselines on the held-out TEST split, exactly as the model
itself is evaluated.

The model's validation-split false-alarm rate is computed by re-running the
already-trained, already-selected checkpoint
(artifacts/official/quantile/official/best_model.pt) on the validation
loader (predictions were never cached to disk) -- same technique as
scripts/conformal_recalibration.py (S-10, round 4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, load_corpus, FittedScaler
from twin.data.splits import official_split, verify_no_leakage
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.train.loop import load_checkpoint, predict_loader
from twin.metrics.clinical import LOW_MAX, event_based_hypoglycemia_detection

RESOLVED_CONFIG = "artifacts/official/quantile/resolved_config.yaml"
CHECKPOINT = "artifacts/official/quantile/official/best_model.pt"
HORIZON_IDX_30MIN = 0
N_LAGS = 6
GRID_MINUTES = 5


def load_fold_and_model():
    config = Config.from_yaml(RESOLVED_CONFIG)
    state = load_checkpoint(CHECKPOINT)
    scaler = FittedScaler.from_state_dict(state["scaler"])

    corpus = load_corpus(config)
    train_sets = {k: v.windows for k, v in corpus["train"].items()}
    test_sets = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(
        list(train_sets.values()), list(test_sets.values()),
        val_fraction=config.split.val_fraction, purge_steps=config.split.purge_steps,
    )
    verify_no_leakage(fold, train_sets, test_sets)

    model = PhysicsGuidedForecaster(len(scaler.feature_names), config)
    model.load_state_dict(state["model_state_dict"])
    device = config.resolve_device()
    model.to(device)
    return config, corpus, fold, model, scaler, device


def model_val_fa_rate(corpus, fold, model, scaler, config, device):
    """Model's own q10<70 alarm at 30 min, evaluated on the VALIDATION split."""
    val_dataset = build_dataset(fold, "val", corpus, scaler, config)
    val_loader = build_loader(val_dataset, config, shuffle=False)
    val_eval = predict_loader(model, val_loader, device)

    val_targets = val_eval["targets"][:, HORIZON_IDX_30MIN]
    val_q10 = val_eval["quantile_predictions"][:, 0, HORIZON_IDX_30MIN]
    subj_idx = val_eval["subject_index"]
    subject_ids = np.asarray(val_dataset.subject_ids)[subj_idx]

    report = event_based_hypoglycemia_detection(val_targets, val_q10, subject_ids, threshold=LOW_MAX)
    print(f"Model's own q10<70 alarm on VALIDATION: {report.false_alarms_per_day:.4f} FA/day "
          f"(sensitivity {report.event_sensitivity:.4f}, specificity {report.specificity:.4f}, "
          f"{report.detected_episodes}/{report.total_episodes} events, "
          f"{report.monitored_days:.1f} monitored patient-days)")
    return report.false_alarms_per_day


def build_alarm_inputs(corpus, fold, subjects, part: str):
    """Per-subject, time-ordered (true, persistence, linear-extrap, subject_id) for `part`."""
    all_true, all_persist, all_extrap, all_subj = [], [], [], []
    selections = fold.val if part == "val" else fold.test
    source = corpus["train"] if part == "val" else corpus["test"]
    for s in subjects:
        data = source[s]
        sel = next((x for x in selections if x.subject_id == s), None)
        if sel is None or len(sel) == 0:
            continue
        g = data.frame["glucose_filled"].to_numpy()
        anchors = data.windows.anchors[sel.indices]
        order = np.argsort(anchors)
        anchors = anchors[order]
        targets = data.windows.targets[sel.indices][order]

        y_true_30 = targets[:, HORIZON_IDX_30MIN]
        persist_pred = g[anchors]

        lag_minutes = np.arange(N_LAGS) * GRID_MINUTES
        x_fit = -lag_minutes
        lag_vals = np.stack([g[anchors - lag] for lag in range(N_LAGS)], axis=1)
        x_mean = x_fit.mean()
        y_mean = lag_vals.mean(axis=1)
        x_centered = x_fit - x_mean
        denom = np.sum(x_centered**2)
        slope = (lag_vals * x_centered).sum(axis=1) / denom
        intercept = y_mean - slope * x_mean
        extrap_pred = intercept + slope * 30.0

        all_true.append(y_true_30)
        all_persist.append(persist_pred)
        all_extrap.append(extrap_pred)
        all_subj.extend([s] * len(y_true_30))

    return (
        np.concatenate(all_true), np.concatenate(all_persist),
        np.concatenate(all_extrap), np.array(all_subj),
    )


def tune_on_val_eval_on_test(y_val, pred_val, subj_val, y_test, pred_test, subj_test,
                              target_fa_per_day, label):
    candidates = np.arange(40.0, 110.01, 0.5)
    best = None
    for thr in candidates:
        shifted_val = pred_val - thr + LOW_MAX
        report_val = event_based_hypoglycemia_detection(y_val, shifted_val, subj_val, threshold=LOW_MAX)
        diff = abs(report_val.false_alarms_per_day - target_fa_per_day)
        if best is None or diff < best[0]:
            best = (diff, thr, report_val)
    diff, thr, report_val = best

    shifted_test = pred_test - thr + LOW_MAX
    report_test = event_based_hypoglycemia_detection(y_test, shifted_test, subj_test, threshold=LOW_MAX)

    print(f"\n[{label}] threshold {thr:.1f} mg/dL chosen on VALIDATION "
          f"(target {target_fa_per_day:.3f} FA/day, achieved on val {report_val.false_alarms_per_day:.3f})")
    print(f"  TEST: sensitivity {report_test.detected_episodes}/{report_test.total_episodes} "
          f"= {report_test.event_sensitivity:.4f}, specificity {report_test.specificity:.4f}, "
          f"{report_test.false_alarms_per_day:.3f} FA/day, "
          f"median lead time {report_test.median_lead_time_min:.1f} min")
    return {
        "baseline": label,
        "decision_threshold_mgdl": thr,
        "target_fa_per_day_val": target_fa_per_day,
        "achieved_fa_per_day_val": report_val.false_alarms_per_day,
        "test_total_episodes": report_test.total_episodes,
        "test_detected_episodes": report_test.detected_episodes,
        "test_event_sensitivity": report_test.event_sensitivity,
        "test_specificity": report_test.specificity,
        "test_fa_per_day": report_test.false_alarms_per_day,
        "test_median_lead_time_min": report_test.median_lead_time_min,
        "test_mean_lead_time_min": report_test.mean_lead_time_min,
        "test_monitored_patient_days": report_test.monitored_days,
    }


def main():
    config, corpus, fold, model, scaler, device = load_fold_and_model()
    subjects = sorted(corpus["test"].keys())

    target_fa = model_val_fa_rate(corpus, fold, model, scaler, config, device)

    y_val, persist_val, extrap_val, subj_val = build_alarm_inputs(corpus, fold, subjects, "val")
    y_test, persist_test, extrap_test, subj_test = build_alarm_inputs(corpus, fold, subjects, "test")
    n_test = len(y_test)
    print(f"\nval windows: {len(y_val)}, test windows: {n_test} (expect 26,498)")
    assert n_test == 26498, f"test window count mismatch: {n_test} != 26498"

    rows = [
        tune_on_val_eval_on_test(y_val, persist_val, subj_val, y_test, persist_test, subj_test,
                                  target_fa, "Persistence alarm"),
        tune_on_val_eval_on_test(y_val, extrap_val, subj_val, y_test, extrap_test, subj_test,
                                  target_fa, "Linear-extrapolation alarm"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/hypo_alarm_baseline_round5.csv", index=False)
    print("\nWrote results/tables/hypo_alarm_baseline_round5.csv")
    print(df.to_string(index=False))
    print("\nModel's own q=0.10 alarm at 30 min (fixed threshold 70 mg/dL, test set, main.tex): "
          "sensitivity 67.9% (38/56), specificity 95.4%, 2.21 FA/day, median lead time 15.0 min.")


if __name__ == "__main__":
    main()
