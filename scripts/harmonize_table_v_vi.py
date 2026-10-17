"""Harmonize Table V (comprehensive baselines) and Table VI (ablation matrix)
so their A0/A7 rows draw from the same run family as the tables they are
compared against, instead of mixing single-seed ablation numbers with
final-model / quantile-head numbers. No retraining -- everything here
aggregates or re-points to result files that already exist on disk.

Table V (tab:baselines_identical, comprehensive_baselines.csv):
    The A7 row previously used the single-seed ablation run (13.25 mg/dL MAE
    at 30 min), which differs from the actual final model reported in
    Table IV (13.08). This script replaces the A0 ("Transformer (all
    features, A0)") and A7 ("Digital Twin (Hybrid, A7)") rows with the
    5-seed means from results/tables/multiseed_a0_vs_a7.csv (same source as
    Table VII, the post hoc 5-seed replication), so Table V's headline A0/A7
    numbers are the same ones used for the significance claims elsewhere.

Table VI (tab:ablation) hypoglycemia columns:
    main.tex currently hardcodes A0 sens=0.637, A7 sens=0.608, A0 bias=+7.29,
    A7 bias=+8.61 for Table VI, sourced from the single-seed ablation run
    WITHOUT a quantile head. Table IX (tab:hypo_event) reports A0=0.413,
    A7=0.551 from the retrained quantile-head models -- the opposite
    ranking. Attempted reproduction of 0.637/0.608 from
    results/ablations/per_subject_A{0,7}.csv's hypo_sensitivity column (both
    simple per-subject mean and event-count-weighted pooled mean) does NOT
    reproduce those two numbers (see printed diagnostics below) -- the exact
    aggregation originally used for Table VI's hypo columns could not be
    located in this repo, same unreproducible-provenance issue as the old
    CRPS numbers. Given that, and per the explicit ask to make Table VI's
    hypoglycemia columns use the SAME quantile-head model family as Table IX,
    this script recomputes Table VI's hypo sens/bias columns directly from
    the quantile-head models' own saved test predictions (median head,
    30-min horizon, readings below 70 mg/dL) -- i.e. the same models and
    methodology as Table IX, not the single-seed ablation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MULTISEED_CSV = "results/tables/multiseed_a0_vs_a7.csv"
MULTISEED_CSV_V5 = "paper/digital-twin-v5/data/multiseed_a0_vs_a7.csv"
COMPREHENSIVE_CSVS = [
    "results/tables/comprehensive_baselines.csv",
    "paper/digital-twin-v5/data/comprehensive_baselines.csv",
]
A0_LABEL = "Transformer (all features, A0)"
A7_LABEL = "Digital Twin (Hybrid, A7)"

ABLATION_A0_CSV = "results/ablations/per_subject_A0.csv"
ABLATION_A7_CSV = "results/ablations/per_subject_A7.csv"
DIAG_A0 = "artifacts/official/quantile-A0/official/test_diagnostics.npz"
DIAG_A7 = "artifacts/official/quantile/official/test_diagnostics.npz"
HYPO_THRESHOLD = 70.0


def harmonize_table_v() -> None:
    seeds = pd.read_csv(MULTISEED_CSV)
    for csv_path in COMPREHENSIVE_CSVS:
        df = pd.read_csv(csv_path)
        for arm, label in [("A0", A0_LABEL), ("A7", A7_LABEL)]:
            for _, row in seeds.iterrows():
                h = int(row["horizon_min"])
                mask = (df["model"] == label) & (df["horizon_min"] == h)
                df.loc[mask, "mae_mean"] = row[f"{arm}_mae_mean_of_seeds"]
                df.loc[mask, "mae_sd"] = row[f"{arm}_mae_sd_of_seeds"]
                df.loc[mask, "rmse_mean"] = row[f"{arm}_rmse_mean_of_seeds"]
                df.loc[mask, "rmse_sd"] = row[f"{arm}_rmse_sd_of_seeds"]
        df.to_csv(csv_path, index=False)
        print(f"Table V harmonized (5-seed means) -> {csv_path}")
        print(df[df["model"].isin([A0_LABEL, A7_LABEL])].to_string(index=False))


def reproduce_old_table_vi_hypo() -> None:
    """Diagnostic only: show that the old 0.637/0.608 can't be reproduced."""
    a0 = pd.read_csv(ABLATION_A0_CSV)
    a7 = pd.read_csv(ABLATION_A7_CSV)
    for name, df in [("A0", a0), ("A7", a7)]:
        d = df[df.horizon_min == 30]
        simple_mean = d["hypo_sensitivity"].mean()
        pooled = (d["hypo_sensitivity"] * d["hypo_n_events"]).sum() / d["hypo_n_events"].sum()
        print(f"[diagnostic] ablation {name} 30min hypo_sensitivity: "
              f"simple subject mean={simple_mean:.4f}, event-weighted pooled={pooled:.4f} "
              f"(main.tex currently states {'0.637' if name == 'A0' else '0.608'} -- "
              f"neither reproduces it; original aggregation not found in repo)")


def compute_quantile_head_hypo() -> dict:
    out = {}
    for name, path in [("A0", DIAG_A0), ("A7", DIAG_A7)]:
        d = np.load(path, allow_pickle=True)
        qp = d["quantile_predictions"].astype(np.float64)
        y = d["targets"].astype(np.float64)
        median_30min = qp[:, 1, 0]
        true_30min = y[:, 0]
        hypo_mask = true_30min < HYPO_THRESHOLD
        detected = median_30min[hypo_mask] < HYPO_THRESHOLD
        sensitivity = float(detected.mean())
        bias = float((median_30min[hypo_mask] - true_30min[hypo_mask]).mean())
        out[name] = {"n_hypo_readings": int(hypo_mask.sum()), "sensitivity": sensitivity, "bias": bias}
    return out


def main() -> None:
    harmonize_table_v()
    print()
    reproduce_old_table_vi_hypo()
    print()
    hypo = compute_quantile_head_hypo()
    for name, vals in hypo.items():
        print(f"[Table VI replacement, quantile-head model, same family as Table IX] "
              f"{name}: n={vals['n_hypo_readings']}, median-forecast pointwise "
              f"sensitivity={vals['sensitivity']:.3f}, hypo bias={vals['bias']:+.2f} mg/dL")
    rows = [{"arm": k, **v} for k, v in hypo.items()]
    out_df = pd.DataFrame(rows)
    for out_path in [
        "results/tables/table_vi_hypo_quantile_head.csv",
        "paper/digital-twin-v5/data/table_vi_hypo_quantile_head.csv",
    ]:
        out_df.to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
