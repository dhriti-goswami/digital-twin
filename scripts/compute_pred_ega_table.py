"""Regenerate the PRED-EGA summary table (Table VIII) from the final quantile-head
model's test predictions, with correct per-subject-aggregated means and SDs for
every category (Accurate/Benign/Erroneous/Clinically-Acceptable, and each
stratified by hypo/eugly/hyper) at every horizon.

Fixes TODO_ROUND5.md S-16: the previous `results/tables/pred_ega_summary.csv`
over-pooled points before taking an SD (giving a pooled-point SD, not the
across-subject SD the paper's Table VIII caption claims), and never computed
SDs for the hypo/eugly/hyper-stratified accurate/benign/erroneous percentages
at all. This script aggregates per-subject first, matching every other
per-subject +/- SD table in the paper (e.g. Table IV/V).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from twin.metrics.errorgrid import pred_ega_summary

PRED_PATH = Path("artifacts/official/quantile/official/test/quantile/predictions.npz")
HORIZONS_MIN = (30, 60, 90, 120)
OUT_PATH = Path("results/tables/pred_ega_summary.csv")


def subject_ids(data: np.lib.npyio.NpzFile) -> list[str]:
    return sorted({k.split("__", 1)[1] for k in data.files if k.startswith("pred__")})


def main() -> None:
    data = np.load(PRED_PATH)
    subjects = subject_ids(data)

    rows = []
    for h_idx, h_min in enumerate(HORIZONS_MIN):
        per_subject = {
            "accurate": [], "benign": [], "erroneous": [], "acceptable": [],
            "hypo_acc": [], "hypo_ben": [], "hypo_err": [],
            "eugly_acc": [], "eugly_ben": [], "eugly_err": [],
            "hyper_acc": [], "hyper_ben": [], "hyper_err": [],
        }
        pooled_ref = []
        pooled_pred = []
        for sid in subjects:
            ref = data[f"true__{sid}"][:, h_idx]
            pred = data[f"pred__{sid}"][:, h_idx]
            pooled_ref.append(ref)
            pooled_pred.append(pred)
            s = pred_ega_summary(ref, pred)
            per_subject["accurate"].append(s.accurate_pct)
            per_subject["benign"].append(s.benign_pct)
            per_subject["erroneous"].append(s.erroneous_pct)
            per_subject["acceptable"].append(s.clinically_acceptable_pct)
            per_subject["hypo_acc"].append(s.hypo_accurate_pct)
            per_subject["hypo_ben"].append(s.hypo_benign_pct)
            per_subject["hypo_err"].append(s.hypo_erroneous_pct)
            per_subject["eugly_acc"].append(s.eugly_accurate_pct)
            per_subject["eugly_ben"].append(s.eugly_benign_pct)
            per_subject["eugly_err"].append(s.eugly_erroneous_pct)
            per_subject["hyper_acc"].append(s.hyper_accurate_pct)
            per_subject["hyper_ben"].append(s.hyper_benign_pct)
            per_subject["hyper_err"].append(s.hyper_erroneous_pct)

        pooled = pred_ega_summary(np.concatenate(pooled_ref), np.concatenate(pooled_pred))

        def mean_sd(key: str) -> tuple[float, float]:
            arr = np.asarray(per_subject[key])
            return float(np.mean(arr)), float(np.std(arr, ddof=1))

        row = {"horizon_min": h_min}
        for key in per_subject:
            m, sd = mean_sd(key)
            row[f"{key}_mean"] = m
            row[f"{key}_sd"] = sd
        row["pooled_accurate"] = pooled.accurate_pct
        row["pooled_benign"] = pooled.benign_pct
        row["pooled_erroneous"] = pooled.erroneous_pct
        row["pooled_acceptable"] = pooled.clinically_acceptable_pct
        rows.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
