#!/usr/bin/env python3
"""Run 5 seeds for A0 and A7, report mean ± SD across seeds, and run Wilcoxon test."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch


from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.features import N_FEATURES
from twin.data.splits import official_split, verify_no_leakage
from twin.eval.runner import evaluate_predictions, write_result
from twin.metrics import holm_bonferroni
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.seeding import set_seed
from twin.train.loop import predict_loader, train_model

SEEDS = [42, 101, 202, 303, 404]
HORIZONS = [30, 60, 90, 120]


def get_model_config(arm: str, seed: int) -> Config:
    config = Config.from_yaml("configs/official-small.yaml")
    config.run.seed = seed
    config.run.deterministic = True
    config.train.batch_size = 256
    config.train.early_stopping_patience = 10
    config.train.num_workers = 3

    if arm == "A0":
        config.physics.enabled = False
        config.model.hybrid_residual = False
        config.model.per_patient_params = False
        config.run.name = f"abl-A0-s{seed}"
    elif arm == "A7":
        config.physics.enabled = True
        config.physics.weighting = "kendall"
        config.model.hybrid_residual = True
        config.model.per_patient_params = True
        config.physics.param_warmup_epochs = 0
        config.physics.ramp_start_epoch = 0
        config.physics.ramp_end_epoch = 0
        config.run.name = f"abl-A7-s{seed}"
    else:
        raise ValueError(f"Unknown arm: {arm}")

    return config


def run_or_load_seed(arm: str, seed: int, fold, corpus, scaler, loaders, device):
    config = get_model_config(arm, seed)
    out_dir = Path(f"artifacts/seeds/{arm}_s{seed}/official")
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = loaders["test"].dataset.subject_ids

    # If seed 42, load existing official ablation predictions
    official_preds_path = Path(f"artifacts/official/abl-{arm}/official/test/abl-{arm}/predictions.npz")
    if seed == 42 and official_preds_path.is_file():
        print(f"[{arm} seed {seed}] Loading existing official predictions from {official_preds_path}")
        data = np.load(official_preds_path)
        preds = {s_id: data[f"pred__{s_id}"] for s_id in subject_ids}
        return evaluate_predictions(
            method=f"{arm}_s{seed}",
            fold=fold,
            part="test",
            corpus=corpus,
            predictions=preds,
            config=config,
        )

    if (out_dir / "best_model.pt").is_file():
        ckpt_path = out_dir / "best_model.pt"
        print(f"[{arm} seed {seed}] Loading existing checkpoint from {ckpt_path}")
        model = PhysicsGuidedForecaster(N_FEATURES, config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"\n[{arm} seed {seed}] Training from scratch...")
        set_seed(seed, deterministic=True)
        model = PhysicsGuidedForecaster(N_FEATURES, config)
        train_model(
            model,
            loaders["train"],
            loaders["val"],
            config,
            scaler=scaler,
            fold_name="official",
            out_dir=out_dir,
            verbose=True,
        )

    # Evaluate on test loader
    evaluation = predict_loader(model, loaders["test"], device)
    preds = {}
    for index, s_id in enumerate(subject_ids):
        mask = evaluation["subject_index"] == index
        preds[s_id] = evaluation["predictions"][mask]

    res = evaluate_predictions(
        method=f"{arm}_s{seed}",
        fold=fold,
        part="test",
        corpus=corpus,
        predictions=preds,
        config=config,
    )
    return res


def main():
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
    subject_ids = loaders["test"].dataset.subject_ids
    device = base_config.resolve_device()

    results = {"A0": {}, "A7": {}}

    for arm in ["A0", "A7"]:
        for seed in SEEDS:
            res = run_or_load_seed(arm, seed, fold, corpus, scaler, loaders, device)
            results[arm][seed] = res

    # Compute per-subject seed-averaged errors
    # per_subject_mae[arm][h] is a dict: subject_id -> list of 5 seed MAEs
    seed_table_rows = []
    comparison_rows = []

    for h in HORIZONS:
        # Collect seed summary
        a0_seed_maes = [results["A0"][s].summary[results["A0"][s].summary["horizon_min"] == h]["mae_mean"].iloc[0] for s in SEEDS]
        a0_seed_rmses = [results["A0"][s].summary[results["A0"][s].summary["horizon_min"] == h]["rmse_mean"].iloc[0] for s in SEEDS]

        a7_seed_maes = [results["A7"][s].summary[results["A7"][s].summary["horizon_min"] == h]["mae_mean"].iloc[0] for s in SEEDS]
        a7_seed_rmses = [results["A7"][s].summary[results["A7"][s].summary["horizon_min"] == h]["rmse_mean"].iloc[0] for s in SEEDS]

        seed_table_rows.append({
            "horizon_min": h,
            "A0_mae_mean_of_seeds": np.mean(a0_seed_maes),
            "A0_mae_sd_of_seeds": np.std(a0_seed_maes, ddof=1),
            "A0_rmse_mean_of_seeds": np.mean(a0_seed_rmses),
            "A0_rmse_sd_of_seeds": np.std(a0_seed_rmses, ddof=1),
            "A7_mae_mean_of_seeds": np.mean(a7_seed_maes),
            "A7_mae_sd_of_seeds": np.std(a7_seed_maes, ddof=1),
            "A7_rmse_mean_of_seeds": np.mean(a7_seed_rmses),
            "A7_rmse_sd_of_seeds": np.std(a7_seed_rmses, ddof=1),
        })

        # Subject-level seed averaging
        a0_subj_mae = np.mean([
            results["A0"][s].metric_by_subject("mae", h).to_numpy() for s in SEEDS
        ], axis=0)
        a7_subj_mae = np.mean([
            results["A7"][s].metric_by_subject("mae", h).to_numpy() for s in SEEDS
        ], axis=0)

        a0_subj_rmse = np.mean([
            results["A0"][s].metric_by_subject("rmse", h).to_numpy() for s in SEEDS
        ], axis=0)
        a7_subj_rmse = np.mean([
            results["A7"][s].metric_by_subject("rmse", h).to_numpy() for s in SEEDS
        ], axis=0)

        # Paired Wilcoxon signed-rank test on seed-averaged per-subject errors
        stat_mae, p_mae = wilcoxon(a7_subj_mae, a0_subj_mae, alternative="two-sided")
        stat_rmse, p_rmse = wilcoxon(a7_subj_rmse, a0_subj_rmse, alternative="two-sided")

        diff_mae = a7_subj_mae - a0_subj_mae
        diff_rmse = a7_subj_rmse - a0_subj_rmse

        comparison_rows.append({
            "horizon_min": h,
            "mae_diff_mean": np.mean(diff_mae),
            "mae_p_raw": p_mae,
            "rmse_diff_mean": np.mean(diff_rmse),
            "rmse_p_raw": p_rmse,
        })

    # Holm correction
    p_mae_dict = {f"MAE_{r['horizon_min']}": r["mae_p_raw"] for r in comparison_rows}
    p_rmse_dict = {f"RMSE_{r['horizon_min']}": r["rmse_p_raw"] for r in comparison_rows}
    holm_mae = holm_bonferroni(p_mae_dict)
    holm_rmse = holm_bonferroni(p_rmse_dict)

    for r in comparison_rows:
        h = r["horizon_min"]
        r["mae_p_holm"] = holm_mae[f"MAE_{h}"]["p_adjusted"]
        r["mae_significant"] = holm_mae[f"MAE_{h}"]["reject"]
        r["rmse_p_holm"] = holm_rmse[f"RMSE_{h}"]["p_adjusted"]
        r["rmse_significant"] = holm_rmse[f"RMSE_{h}"]["reject"]

    df_seeds = pd.DataFrame(seed_table_rows)
    df_comp = pd.DataFrame(comparison_rows)

    print("\n=== Multi-Seed (5 seeds: 42, 101, 202, 303, 404) Summary ===")
    print(df_seeds.to_string(index=False))
    print("\n=== Paired Wilcoxon Test on Seed-Averaged Per-Subject Errors ===")
    print(df_comp.to_string(index=False))

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("paper/digital-twin-v3-final/data").mkdir(parents=True, exist_ok=True)
    df_seeds.to_csv("results/tables/multiseed_a0_vs_a7.csv", index=False)
    df_seeds.to_csv("paper/digital-twin-v3-final/data/multiseed_a0_vs_a7.csv", index=False)
    df_comp.to_csv("results/tables/multiseed_wilcoxon_tests.csv", index=False)
    df_comp.to_csv("paper/digital-twin-v3-final/data/multiseed_wilcoxon_tests.csv", index=False)


if __name__ == "__main__":
    main()
