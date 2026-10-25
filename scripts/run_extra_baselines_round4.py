"""Round-4 baselines (TODO_ROUND4.md S-11, Must): a non-ML hypoglycemia alarm
baseline and a linear quantile-regression probabilistic baseline, both on the
identical 26,498-window official test split used throughout the paper.

1. Hypoglycemia alarm baseline: trigger on EITHER (a) the current glucose
   already being near/below threshold (persistence) or (b) a linear
   extrapolation of the last 25 minutes of readings to the 30-min horizon.
   The alarm's decision threshold is tuned on the test set so its false-alarm
   rate matches the paper's own A7 q=0.10 alarm (2.21 false alarms/day,
   main.tex line ~1006/1031), then event sensitivity and lead time are read
   off at that matched operating point for a fair comparison.

   Reuses twin.metrics.clinical.event_based_hypoglycemia_detection exactly as
   the paper's own alarm evaluation does (same episode definition: >=15 min
   continuously <70 mg/dL, detection window 6 steps = 30 min lookback). That
   function always compares (y_pred < threshold) with threshold=LOW_MAX=70
   used for both the true-episode definition and the alarm; to sweep a
   *decision* threshold distinct from the clinical truth threshold, each
   candidate glucose/extrapolation value is shifted by
   (decision_threshold - LOW_MAX) before being passed in, so
   `pred_shifted < LOW_MAX  <=>  pred_raw < decision_threshold`.

2. Probabilistic baseline: per-subject linear quantile regression (pinball
   loss, q=0.1/0.5/0.9) on the same 35-feature vector as the Ridge-35feat
   baseline (scripts/run_extra_baselines.py), one regression per subject per
   horizon per quantile (matching that script's per-subject-personalized
   methodology). Reports per-horizon q10/q90 coverage, mean interval width,
   and CRPS via the same trapezoidal formula in scripts/compute_crps.py, so a
   row can be added to Table X in the same format.

Reuses the exact corpus/split/windows pipeline from scripts/run_ar_arx.py
(same Config(), load_corpus, official_split) -- confirmed 26,498 total test
windows across the 12 official test subjects, matching every other row in
Table V/VII/X.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from twin.config import Config
from twin.data.dataset import load_corpus
from twin.data.features import FEATURE_NAMES
from twin.data.splits import official_split
from twin.metrics.clinical import LOW_MAX, event_based_hypoglycemia_detection

HORIZONS_MIN = [30, 60, 90, 120]
HORIZON_IDX_30MIN = 0
N_LAGS = 6  # 0,5,...,25 min back, matching AR(6)'s lookback
GRID_MINUTES = 5


def crps_from_three_quantiles(pb10: float, pb50: float, pb90: float) -> float:
    """Same trapezoidal-rule CRPS approximation as scripts/compute_crps.py."""
    return 0.5 * pb10 + 0.8 * pb50 + 0.5 * pb90


def pinball_loss(quantile: float, pred: np.ndarray, true: np.ndarray) -> float:
    error = true - pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def load_fold():
    config = Config()
    corpus = load_corpus(config)
    train_sets = {k: v.windows for k, v in corpus["train"].items()}
    test_sets = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=config.split.val_fraction,
        purge_steps=config.split.purge_steps,
    )
    return config, corpus, fold


def build_hypo_alarm_inputs(corpus, fold, subjects):
    """Per-subject, time-ordered arrays: true 30-min target, persistence pred,
    linear-extrapolation pred, subject id."""
    all_true, all_persist, all_extrap, all_subj = [], [], [], []
    for s in subjects:
        test_data = corpus["test"][s]
        test_sel = next(sel for sel in fold.test if sel.subject_id == s)
        g = test_data.frame["glucose_filled"].to_numpy()
        anchors = test_data.windows.anchors[test_sel.indices]
        order = np.argsort(anchors)
        anchors = anchors[order]
        targets = test_data.windows.targets[test_sel.indices][order]

        y_true_30 = targets[:, HORIZON_IDX_30MIN]
        persist_pred = g[anchors]  # "current glucose" persistence forecast

        # Linear extrapolation over last N_LAGS points (0..25 min back) to +30 min.
        lag_minutes = np.arange(N_LAGS) * GRID_MINUTES  # 0,5,...,25 (back in time)
        x_fit = -lag_minutes  # time axis, most recent = 0, older = negative
        lag_vals = np.stack([g[anchors - lag] for lag in range(N_LAGS)], axis=1)  # (n, N_LAGS)
        # vectorized per-row OLS slope/intercept on x_fit (same x for every row)
        x_mean = x_fit.mean()
        y_mean = lag_vals.mean(axis=1)
        x_centered = x_fit - x_mean
        denom = np.sum(x_centered**2)
        slope = (lag_vals * x_centered).sum(axis=1) / denom
        intercept = y_mean - slope * x_mean
        extrap_pred = intercept + slope * 30.0  # extrapolate to +30 min

        all_true.append(y_true_30)
        all_persist.append(persist_pred)
        all_extrap.append(extrap_pred)
        all_subj.extend([s] * len(y_true_30))

    return (
        np.concatenate(all_true),
        np.concatenate(all_persist),
        np.concatenate(all_extrap),
        np.array(all_subj),
    )


def sweep_to_match_fa_rate(y_true, raw_pred, subj_ids, target_fa_per_day, label):
    """Sweep a decision threshold so false_alarms_per_day matches target_fa_per_day
    as closely as possible; report the matched operating point's metrics."""
    candidates = np.arange(40.0, 110.01, 0.5)
    best = None
    for thr in candidates:
        shifted = raw_pred - thr + LOW_MAX  # shifted < LOW_MAX  <=>  raw_pred < thr
        report = event_based_hypoglycemia_detection(y_true, shifted, subj_ids, threshold=LOW_MAX)
        diff = abs(report.false_alarms_per_day - target_fa_per_day)
        if best is None or diff < best[0]:
            best = (diff, thr, report)
    diff, thr, report = best
    print(f"\n[{label}] matched decision threshold = {thr:.1f} mg/dL "
          f"(target {target_fa_per_day:.2f} FA/day, achieved {report.false_alarms_per_day:.3f})")
    print(f"  event sensitivity: {report.detected_episodes}/{report.total_episodes} "
          f"= {report.event_sensitivity:.4f}")
    print(f"  specificity: {report.specificity:.4f}")
    print(f"  median lead time: {report.median_lead_time_min:.1f} min, "
          f"mean lead time: {report.mean_lead_time_min:.1f} min")
    print(f"  monitored patient-days: {report.monitored_days:.1f}")
    return {
        "baseline": label,
        "decision_threshold_mgdl": thr,
        "target_fa_per_day": target_fa_per_day,
        "achieved_fa_per_day": report.false_alarms_per_day,
        "total_episodes": report.total_episodes,
        "detected_episodes": report.detected_episodes,
        "event_sensitivity": report.event_sensitivity,
        "specificity": report.specificity,
        "median_lead_time_min": report.median_lead_time_min,
        "mean_lead_time_min": report.mean_lead_time_min,
        "monitored_patient_days": report.monitored_days,
    }


def run_hypo_alarm_baseline(corpus, fold, subjects):
    y_true, persist_pred, extrap_pred, subj_ids = build_hypo_alarm_inputs(corpus, fold, subjects)
    # Paper's own A7 q=0.10 alarm operating point (main.tex ~line 1006/1031): 2.21 FA/day.
    target_fa = 2.21
    rows = [
        sweep_to_match_fa_rate(y_true, persist_pred, subj_ids, target_fa, "Persistence alarm"),
        sweep_to_match_fa_rate(y_true, extrap_pred, subj_ids, target_fa, "Linear-extrapolation alarm"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/hypo_alarm_baseline.csv", index=False)
    print("\nWrote results/tables/hypo_alarm_baseline.csv")
    print(df.to_string(index=False))
    print("\nPaper's own A7 quantile (q=0.10) alarm at 30 min (main.tex line ~1031): "
          "sensitivity 67.9%, specificity 95.4%, 2.21 FA/day, median lead time 15.0 min.")
    return df


def run_probabilistic_baseline(corpus, fold, subjects):
    quantiles = [0.1, 0.5, 0.9]
    per_h_pb = {h: {q: [] for q in quantiles} for h in HORIZONS_MIN}
    per_h_cov10, per_h_cov90, per_h_width = (
        {h: [] for h in HORIZONS_MIN},
        {h: [] for h in HORIZONS_MIN},
        {h: [] for h in HORIZONS_MIN},
    )

    for s in subjects:
        train_data = corpus["train"][s]
        test_data = corpus["test"][s]
        train_sel = next(sel for sel in fold.train if sel.subject_id == s)
        test_sel = next(sel for sel in fold.test if sel.subject_id == s)

        anchors_tr = train_data.windows.anchors[train_sel.indices]
        X_train_raw = train_data.features.values[anchors_tr]
        Y_train = train_data.windows.targets[train_sel.indices]

        anchors_te = test_data.windows.anchors[test_sel.indices]
        X_test_raw = test_data.features.values[anchors_te]
        Y_test = test_data.windows.targets[test_sel.indices]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(np.nan_to_num(X_train_raw))
        X_test = scaler.transform(np.nan_to_num(X_test_raw))

        for h_idx, h in enumerate(HORIZONS_MIN):
            preds = {}
            for q in quantiles:
                model = QuantileRegressor(quantile=q, alpha=1e-4, solver="highs")
                model.fit(X_train, Y_train[:, h_idx])
                pred = np.clip(model.predict(X_test), 40.0, 400.0)
                preds[q] = pred
                pb = pinball_loss(q, pred, Y_test[:, h_idx])
                per_h_pb[h][q].append(pb)
            # enforce monotonicity q10<=q50<=q90 for coverage/width sanity
            q10, q90 = np.minimum(preds[0.1], preds[0.9]), np.maximum(preds[0.1], preds[0.9])
            cov10 = float(np.mean(Y_test[:, h_idx] < q10)) * 100.0
            cov90 = float(np.mean(Y_test[:, h_idx] <= q90)) * 100.0
            width = float(np.mean(q90 - q10))
            per_h_cov10[h].append(cov10)
            per_h_cov90[h].append(cov90)
            per_h_width[h].append(width)
        print(f"  fit subject {s} done")

    rows = []
    for h in HORIZONS_MIN:
        pb10 = float(np.mean(per_h_pb[h][0.1]))
        pb50 = float(np.mean(per_h_pb[h][0.5]))
        pb90 = float(np.mean(per_h_pb[h][0.9]))
        crps = crps_from_three_quantiles(pb10, pb50, pb90)
        rows.append({
            "model": "Linear quantile regression (35 feat)",
            "horizon_min": h,
            "lower_q10_coverage_pct": float(np.mean(per_h_cov10[h])),
            "upper_q90_coverage_pct": float(np.mean(per_h_cov90[h])),
            "mean_interval_width": float(np.mean(per_h_width[h])),
            "pinball_q10": pb10,
            "pinball_q50": pb50,
            "pinball_q90": pb90,
            "crps": crps,
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/probabilistic_baseline.csv", index=False)
    print("\nWrote results/tables/probabilistic_baseline.csv")
    print(df.to_string(index=False))
    return df


def main():
    config, corpus, fold = load_fold()
    subjects = sorted(corpus["test"].keys())
    n_test = sum(len(sel.indices) for sel in fold.test)
    print(f"Loaded official split: {n_test} test windows across {len(subjects)} subjects "
          f"(expected 26,498 to match Table V/VII/X).")
    assert n_test == 26498, f"test window count mismatch: {n_test} != 26498"

    run_hypo_alarm_baseline(corpus, fold, subjects)
    run_probabilistic_baseline(corpus, fold, subjects)


if __name__ == "__main__":
    main()
