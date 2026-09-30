#!/usr/bin/env python3
"""Ablation: train A7 with lambda_temporal_consistency = 0.0 to evaluate S_I ICC."""

import json
from pathlib import Path
import numpy as np
import torch

from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.features import N_FEATURES
from twin.data.splits import official_split, verify_no_leakage
from twin.eval.runner import evaluate_predictions, write_result
from twin.eval.sensitivity import validate_sensitivity
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.seeding import set_seed
from twin.train.loop import predict_loader, train_model


def main() -> None:
    config = Config.from_yaml("configs/official.yaml")
    # Exact A7 settings
    config.train.batch_size = 256
    config.train.early_stopping_patience = 10
    config.train.num_workers = 3
    config.physics.enabled = True
    config.physics.weighting = "kendall"
    config.model.hybrid_residual = True
    config.model.per_patient_params = True
    config.physics.param_warmup_epochs = 0
    config.physics.ramp_start_epoch = 0
    config.physics.ramp_end_epoch = 0
    # The ablation under test: zero temporal consistency penalty
    config.physics.lambda_temporal_consistency = 0.0
    config.run.name = "abl-A7-notc"

    out_root = Path("artifacts/official/abl-A7-notc")
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(config.run.seed, deterministic=config.run.deterministic)
    corpus = load_corpus(config)
    train_sets = {key: value.windows for key, value in corpus["train"].items()}
    test_sets = {key: value.windows for key, value in corpus["test"].items()}

    fold = official_split(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=config.split.val_fraction,
        purge_steps=config.split.purge_steps,
    )
    verify_no_leakage(fold, train_sets, test_sets)
    scaler = fit_scaler(fold, corpus)

    loaders = {
        part: build_loader(
            build_dataset(fold, part, corpus, scaler, config),
            config,
            shuffle=(part == "train"),
        )
        for part in ("train", "val", "test")
    }

    fold_out = out_root / fold.name
    fold_out.mkdir(parents=True, exist_ok=True)

    print(f"=== Training {config.run.name} (lambda_tc = 0.0, bs=256) ===", flush=True)
    model = PhysicsGuidedForecaster(N_FEATURES, config)
    result = train_model(
        model,
        loaders["train"],
        loaders["val"],
        config,
        scaler=scaler,
        fold_name=fold.name,
        out_dir=fold_out,
        verbose=True,
    )

    evaluation = predict_loader(model, loaders["test"], config.resolve_device())
    test_ds = loaders["test"].dataset
    subject_ids = test_ds.subject_ids

    predictions = {}
    for index, subject_id in enumerate(subject_ids):
        mask = evaluation["subject_index"] == index
        predictions[subject_id] = evaluation["predictions"][mask]

    diagnostics = {
        "insulin_sensitivity": evaluation["insulin_sensitivity"],
        "subject_index": evaluation["subject_index"],
        "targets": evaluation["targets"],
    }
    np.savez_compressed(fold_out / "test_diagnostics.npz", **diagnostics)

    model_result = evaluate_predictions(
        method=config.run.name,
        fold=fold,
        part="test",
        corpus=corpus,
        predictions=predictions,
        config=config,
    )
    write_result(model_result, out_root)

    s_i_by_subj = {
        s: evaluation["insulin_sensitivity"][evaluation["subject_index"] == i]
        for i, s in enumerate(subject_ids)
    }
    report = validate_sensitivity(s_i_by_subj, corpus)
    print("\n=== Sensitivity Validation Report (lambda_tc = 0.0) ===")
    print(report.verdict())
    print("\nSummary Frame:")
    print(report.summary_frame())

    res_dict = {
        "icc": report.stability.get("icc", np.nan),
        "cv": report.degeneracy.get("cv", np.nan),
        "min_si": report.degeneracy.get("min", np.nan),
        "max_si": report.degeneracy.get("max", np.nan),
    }
    for c in report.correlations:
        res_dict[f"rho_{c.label}"] = c.rho
        res_dict[f"p_{c.label}"] = c.p_value
        res_dict[f"ci_low_{c.label}"] = c.ci_low
        res_dict[f"ci_high_{c.label}"] = c.ci_high

    with open(fold_out / "sensitivity_report_notc.json", "w") as f:
        json.dump(res_dict, f, indent=2)
    print(f"Saved report to {fold_out / 'sensitivity_report_notc.json'}")


if __name__ == "__main__":
    main()
