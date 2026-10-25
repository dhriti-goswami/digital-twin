"""Recompute Table VI's pointwise hypoglycemia sensitivity and bias at 30 min for A0-A4, A7.

Method: pooled pointwise detection at the 30-min horizon across all 12 test subjects
(threshold 70 mg/dL, same definition as twin/metrics/clinical.py's hypoglycemia_detection),
using the raw per-window predictions in artifacts/official/abl-{arm}/official/test/abl-{arm}/predictions.npz
(column 0 = 30-min horizon, verified against results/ablations/per_subject_A*.csv's
hypo_sensitivity column for individual subjects before pooling).

The pooled pointwise bias (mean pred-true error restricted to true<70 mg/dL, pooled across all
subjects before averaging) reproduces main.tex Table VI's reported hypo-bias values to within
0.01 mg/dL for every arm, confirming this is the right data source and pooling order. The pooled
pointwise sensitivity computed the same way is consistently ~0.013-0.017 BELOW every one of
Table VI's six reported sensitivity values (0.637/0.502/0.499/0.485/0.517/0.608) -- the exact
"~0.015 lower for every arm" offset flagged in TODO_ROUND4.md S-4. Multiple alternative
aggregations were tried (per-subject simple mean, threshold<=70, episode-consolidated
sensitivity) and none reproduce the original numbers any better; since the bias values match
almost exactly, the discrepancy is isolated to the sensitivity TP/FN definition used by whatever
script originally produced Table VI, which is not present in this repository. These are therefore
new, fully-reproducible-today corrected values, not a reproduction of the original undocumented
method.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["A0", "A1", "A2", "A3", "A4", "A7"]
SUBJECTS = ["540", "544", "552", "559", "563", "567", "570", "575", "584", "588", "591", "596"]
THRESHOLD = 70.0
HORIZON_COL = 0  # 30-min horizon


def main() -> None:
    rows = []
    for arm in ARMS:
        path = Path(f"artifacts/official/abl-{arm}/official/test/abl-{arm}/predictions.npz")
        data = np.load(path)
        tp_total = fn_total = 0
        bias_num = 0.0
        bias_den = 0
        for subject in SUBJECTS:
            pred = data[f"pred__{subject}"][:, HORIZON_COL]
            true = data[f"true__{subject}"][:, HORIZON_COL]
            actual = true < THRESHOLD
            predicted = pred < THRESHOLD
            tp_total += int(np.count_nonzero(actual & predicted))
            fn_total += int(np.count_nonzero(actual & ~predicted))
            bias_num += float(np.sum(pred[actual] - true[actual]))
            bias_den += int(np.sum(actual))
        n_events = tp_total + fn_total
        sensitivity = tp_total / n_events
        bias = bias_num / bias_den
        rows.append(
            {
                "arm": arm,
                "n_events": n_events,
                "true_positive": tp_total,
                "false_negative": fn_total,
                "hypo_sensitivity_30min": round(sensitivity, 4),
                "hypo_bias_30min": round(bias, 4),
            }
        )
        print(f"{arm}: n={n_events} sensitivity={sensitivity:.4f} bias={bias:+.4f}")

    out = pd.DataFrame(rows)
    out_path = Path("results/tables/table_vi_hypo_corrected.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
