"""Regenerate results/tables/hypoglycemia_a0_vs_a7.csv using the fairly-retrained
quantile-A0-fair checkpoint (S-15, TODO_ROUND5.md) in place of the original
quantile-A0 run (which had an 8-epoch selection handicap relative to A7)."""
import numpy as np
import pandas as pd

from twin.metrics.clinical import event_based_hypoglycemia_detection, hypoglycaemia_detection, LOW_MAX

HORIZON_IDX = 0  # 30 min
Q10_IDX, Q50_IDX = 0, 1

RUNS = {
    "A0 (No physics)": "artifacts/official/quantile-A0-fair/official/test_diagnostics.npz",
    "A7 (Physics hybrid)": "artifacts/official/quantile/official/test_diagnostics.npz",
}


def main():
    rows = []
    for arm, path in RUNS.items():
        d = np.load(path, allow_pickle=True)
        targets = d["targets"][:, HORIZON_IDX]
        subj = d["subject_index"]
        q_pred = d["quantile_predictions"]

        median_pred = q_pred[:, Q50_IDX, HORIZON_IDX]
        q10_pred = q_pred[:, Q10_IDX, HORIZON_IDX]

        pointwise_median = hypoglycaemia_detection(targets, median_pred, threshold=LOW_MAX)
        pointwise_q10 = hypoglycaemia_detection(targets, q10_pred, threshold=LOW_MAX)

        event_median = event_based_hypoglycemia_detection(targets, median_pred, subj, threshold=LOW_MAX)
        event_q10 = event_based_hypoglycemia_detection(targets, q10_pred, subj, threshold=LOW_MAX)

        rows.append({
            "arm": arm, "head": "Median (point forecast)",
            "sensitivity": pointwise_median["sensitivity"],
            "specificity": pointwise_median["specificity"],
            "precision": pointwise_median["precision"],
            "false_positives": pointwise_median["false_positive"],
            "event_sensitivity": event_median.event_sensitivity,
            "fa_per_day": event_median.false_alarms_per_day,
            "lead_time_min": event_median.median_lead_time_min,
        })
        rows.append({
            "arm": arm, "head": "Quantile band (q=0.10)",
            "sensitivity": pointwise_q10["sensitivity"],
            "specificity": pointwise_q10["specificity"],
            "precision": pointwise_q10["precision"],
            "false_positives": pointwise_q10["false_positive"],
            "event_sensitivity": event_q10.event_sensitivity,
            "fa_per_day": event_q10.false_alarms_per_day,
            "lead_time_min": event_q10.median_lead_time_min,
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("results/tables/hypoglycemia_a0_vs_a7.csv", index=False)
    print("wrote results/tables/hypoglycemia_a0_vs_a7.csv")


if __name__ == "__main__":
    main()
