#!/usr/bin/env python3
"""Subject-bootstrap 95% CIs for the A7-A0 MAE difference at each horizon, 5-seed runs.

Reuses scripts/run_seeds_a0_a7.py's model-loading/eval path (all 5-seed checkpoints already
exist under artifacts/seeds/, so this only evaluates, it does not train). For each horizon,
takes the seed-averaged per-subject MAE vectors for A0 and A7 (12 subjects), then resamples
subjects with replacement 5000 times and reports the 95% percentile CI of the A7-A0 mean
difference.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.splits import official_split, verify_no_leakage
from twin.config import Config

import scripts.run_seeds_a0_a7 as rs

SEEDS = rs.SEEDS
HORIZONS = rs.HORIZONS
N_BOOT = 5000
RNG_SEED = 42


def main() -> None:
    base_config = Config.from_yaml("configs/official-small.yaml")
    base_config.train.batch_size = 256
    base_config.train.num_workers = 3
    corpus = load_corpus(base_config)
    train_sets = {key: value.windows for key, value in corpus["train"].items()}
    test_sets = {key: value.windows for key, value in corpus["test"].items()}
    fold = official_split(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=base_config.split.val_fraction,
        purge_steps=base_config.split.purge_steps,
    )
    verify_no_leakage(fold, train_sets, test_sets)
    scaler = fit_scaler(fold, corpus)

    loaders = {
        part: build_loader(
            build_dataset(fold, part, corpus, scaler, base_config),
            base_config,
            shuffle=(part == "train"),
        )
        for part in ("train", "val", "test")
    }
    config = base_config
    device = base_config.resolve_device()

    results = {"A0": {}, "A7": {}}
    for arm in ["A0", "A7"]:
        for seed in SEEDS:
            results[arm][seed] = rs.run_or_load_seed(arm, seed, fold, corpus, scaler, loaders, device)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for h in HORIZONS:
        a0_subj_mae = np.mean(
            [results["A0"][s].metric_by_subject("mae", h).to_numpy() for s in SEEDS], axis=0
        )
        a7_subj_mae = np.mean(
            [results["A7"][s].metric_by_subject("mae", h).to_numpy() for s in SEEDS], axis=0
        )
        n_subj = len(a0_subj_mae)
        diffs = a7_subj_mae - a0_subj_mae
        point_estimate = float(np.mean(diffs))

        boot_means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n_subj, size=n_subj)
            boot_means[b] = np.mean(diffs[idx])
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        rows.append(
            {
                "horizon_min": h,
                "n_subjects": n_subj,
                "mae_diff_mean": round(point_estimate, 4),
                "ci_lower_95": round(float(ci_lo), 4),
                "ci_upper_95": round(float(ci_hi), 4),
                "crosses_zero": bool(ci_lo <= 0 <= ci_hi),
            }
        )
        print(f"{h}min: diff={point_estimate:+.4f} 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    out = pd.DataFrame(rows)
    out_path = Path("results/tables/physics_effect_cis.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
