#!/usr/bin/env python3
"""Statistical comparison of A7 vs ARX(6) baseline across all 12 OhioT1DM subjects.

Evaluates paired two-sided Wilcoxon signed-rank tests for MAE and RMSE at
horizons 30, 60, 90, 120 minutes with Holm-Bonferroni correction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from twin.config import Config
from twin.data.dataset import load_corpus
from twin.data.splits import official_split
from twin.metrics import holm_bonferroni
from scripts.run_ar_arx import extract_arx


def main():
    print("=== A7 vs ARX(6) Wilcoxon Signed-Rank Significance Test ===")
    config = Config()
    corpus = load_corpus(config)
    train_sets = {key: value.windows for key, value in corpus["train"].items()}
    test_sets = {key: value.windows for key, value in corpus["test"].items()}
    fold = official_split(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=config.split.val_fraction,
        purge_steps=config.split.purge_steps,
    )
    subjects = sorted(list(corpus["test"].keys()))

    # Run / extract per-subject ARX(6) errors
    arx_subj_mae = {30: [], 60: [], 90: [], 120: []}
    arx_subj_rmse = {30: [], 60: [], 90: [], 120: []}

    for s in subjects:
        train_data = corpus["train"][s]
        test_data = corpus["test"][s]
        train_sel = next(sel for sel in fold.train if sel.subject_id == s)
        test_sel = next(sel for sel in fold.test if sel.subject_id == s)

        X_train, Y_train = extract_arx(train_data, train_sel.indices)
        X_test, Y_test = extract_arx(test_data, test_sel.indices)

        W = np.linalg.solve(
            X_train.T @ X_train + 1.0 * np.eye(X_train.shape[1]),
            X_train.T @ Y_train,
        )
        Y_pred = np.clip(X_test @ W, 40.0, 400.0)

        err = Y_pred - Y_test
        mae = np.mean(np.abs(err), axis=0)
        rmse = np.sqrt(np.mean(err**2, axis=0))

        for idx, h in enumerate([30, 60, 90, 120]):
            arx_subj_mae[h].append(mae[idx])
            arx_subj_rmse[h].append(rmse[idx])

    # Load official Arm A7 per-subject metrics
    df_a7 = pd.read_csv("results/ablations/per_subject_A7.csv")

    horizons = [30, 60, 90, 120]
    comparison_rows = []

    for metric in ["mae", "rmse"]:
        p_raw_dict = {}
        for h in horizons:
            a7_vals = (
                df_a7[df_a7["horizon_min"] == h]
                .sort_values("subject_id")[metric]
                .to_numpy()
            )
            arx_vals = np.array(
                arx_subj_mae[h] if metric == "mae" else arx_subj_rmse[h]
            )
            diff = a7_vals - arx_vals
            w_stat, p_val = wilcoxon(a7_vals, arx_vals, alternative="two-sided")
            better = int(np.sum(diff < 0))

            p_raw_dict[f"{metric.upper()}_{h}"] = p_val
            comparison_rows.append({
                "metric": metric.upper(),
                "horizon_min": h,
                "a7_mean": np.mean(a7_vals),
                "a7_sd": np.std(a7_vals, ddof=1),
                "arx_mean": np.mean(arx_vals),
                "arx_sd": np.std(arx_vals, ddof=1),
                "diff_mean": np.mean(diff),
                "a7_better_n": f"{better}/12",
                "wilcoxon_W": w_stat,
                "p_raw": p_val,
            })

    # Apply Holm-Bonferroni correction across the 4 horizons for each metric
    df_comp = pd.DataFrame(comparison_rows)
    for metric in ["MAE", "RMSE"]:
        p_dict = {
            f"{r['horizon_min']}": r["p_raw"]
            for _, r in df_comp[df_comp["metric"] == metric].iterrows()
        }
        holm = holm_bonferroni(p_dict)
        for h in horizons:
            mask = (df_comp["metric"] == metric) & (df_comp["horizon_min"] == h)
            df_comp.loc[mask, "p_holm"] = holm[f"{h}"]["p_adjusted"]
            df_comp.loc[mask, "significant"] = holm[f"{h}"]["reject"]

    print("\nResults Summary:")
    print(df_comp.to_string(index=False))

    out_path = Path("results/tables/a7_vs_arx_wilcoxon.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(out_path, index=False)
    print(f"\nSaved statistical test results to {out_path}")


if __name__ == "__main__":
    main()
