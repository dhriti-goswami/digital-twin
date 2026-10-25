#!/usr/bin/env python3
"""Retrain the quantile-head A0 run (Table IX) with a fair selection window.

The original artifacts/official/quantile-A0 run used physics.param_warmup_epochs=10,
ramp_end_epoch=25 (official-small.yaml defaults), so selection could not start before
epoch 11 (it picked epoch 13). The quantile-head A7 run selects from epoch 1 in every
case. This retrains A0 with the same early curriculum as the ablation A0 run
(warmup=2, ramp_start=2, ramp_end=6, selection from epoch 3), keeping everything else
(patience=12, quantile head, architecture) identical to the original quantile-A0 run.
"""
from pathlib import Path

import numpy as np
import torch

from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.features import N_FEATURES
from twin.data.splits import official_split, verify_no_leakage
from twin.eval.runner import evaluate_predictions
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.seeding import set_seed
from twin.train.loop import predict_loader, train_model


def main() -> None:
    config = Config.from_yaml("configs/official-small.yaml")
    config.run.seed = 42
    config.run.deterministic = True
    config.train.batch_size = 256
    config.train.num_workers = 3
    config.train.early_stopping_patience = 12

    config.physics.enabled = False
    config.model.hybrid_residual = False
    config.model.per_patient_params = False
    config.physics.param_warmup_epochs = 2
    config.physics.ramp_start_epoch = 2
    config.physics.ramp_end_epoch = 6
    config.run.name = "quantile-A0-fair"

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
    subject_ids = loaders["test"].dataset.subject_ids
    device = config.resolve_device()

    out_dir = Path("artifacts/official/quantile-A0-fair/official")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.run.seed, deterministic=True)
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

    evaluation = predict_loader(model, loaders["test"], device)
    preds = {}
    for index, s_id in enumerate(subject_ids):
        mask = evaluation["subject_index"] == index
        preds[s_id] = evaluation["predictions"][mask]
        if "quantile_predictions" in evaluation:
            pass

    res = evaluate_predictions(
        method="quantile-A0-fair",
        fold=fold,
        part="test",
        corpus=corpus,
        predictions=preds,
        config=config,
    )
    print(res.summary.to_string(index=False))

    np.savez(
        out_dir / "test" / "quantile-A0-fair" / "predictions.npz"
        if (out_dir / "test" / "quantile-A0-fair").exists()
        else out_dir / "predictions_fallback.npz",
        **{f"pred__{s_id}": preds[s_id] for s_id in subject_ids},
    )


if __name__ == "__main__":
    main()
