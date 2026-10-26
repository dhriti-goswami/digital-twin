#!/usr/bin/env python3
"""Round 6, P6 (pre-registered in paper/digital-twin-v5/PREREG_ROUND6.md): low-data regime.

Retrain fair A0 and A7 on 25% and 50% of the training SUBJECTS (seeds 42/101/202) and
evaluate on the unchanged full test set. Configs/curricula are exactly those of
scripts/run_seeds_a0_a7.py::get_model_config (A0 with the round-5 fair selection window).

Subset choice is deterministic by seed: rng = default_rng(seed); k = round(frac * n_subjects)
subjects drawn without replacement from the sorted training-subject ids. A0 and A7 with the
same (seed, frac) see the same subjects. The validation split is restricted to the same
subjects (it is a tail of their own training files). The scaler is refit on the subset's
training slots only. The test set is built from the FULL fold (so per-subject G_b/I_b for test
subjects are resolved exactly as in the full-data runs) but scaled with the subset scaler.

Outputs: results/tables/round6_p6_per_subject.csv (per run/subject/horizon MAE),
results/tables/round6_p6_runs.csv (chosen subjects, best epoch). Statistics + Holm are done in
scripts/round6_regime_analysis.py.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.features import N_FEATURES
from twin.data.splits import official_split, verify_no_leakage
from twin.eval.runner import evaluate_predictions
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.seeding import set_seed
from twin.train.loop import load_checkpoint, predict_loader, train_model

import scripts.run_seeds_a0_a7 as rs

FRACS = [0.25, 0.50]
SEEDS = [42, 101, 202]
HORIZONS = rs.HORIZONS


def main() -> None:
    base = Config.from_yaml("configs/official-small.yaml")
    base.train.batch_size = 256
    base.train.num_workers = 3
    corpus = load_corpus(base)
    train_sets = {k: v.windows for k, v in corpus["train"].items()}
    test_sets = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(
        list(train_sets.values()), list(test_sets.values()),
        val_fraction=base.split.val_fraction, purge_steps=base.split.purge_steps,
    )
    verify_no_leakage(fold, train_sets, test_sets)
    device = base.resolve_device()
    subj_all = sorted(s.subject_id for s in fold.train)

    rows, runs = [], []
    for frac in FRACS:
        k = int(round(frac * len(subj_all)))
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            chosen = sorted(rng.choice(subj_all, size=k, replace=False).tolist())
            fold_sub = replace(
                fold,
                train=[s for s in fold.train if s.subject_id in chosen],
                val=[s for s in fold.val if s.subject_id in chosen],
            )
            scaler = fit_scaler(fold_sub, corpus)
            loaders = {
                "train": build_loader(build_dataset(fold_sub, "train", corpus, scaler, base), base, shuffle=True),
                "val": build_loader(build_dataset(fold_sub, "val", corpus, scaler, base), base, shuffle=False),
                "test": build_loader(build_dataset(fold, "test", corpus, scaler, base), base, shuffle=False),
            }
            test_ids = loaders["test"].dataset.subject_ids
            for arm in ["A0", "A7"]:
                cfg = rs.get_model_config(arm, seed)
                out_dir = Path(f"artifacts/round6_lowdata/{arm}_f{int(frac*100)}_s{seed}/official")
                ckpt = out_dir / "best_model.pt"
                if not ckpt.is_file():
                    print(f"\n[{arm} frac {frac} seed {seed}] training on {chosen}", flush=True)
                    set_seed(seed, deterministic=True)
                    model = PhysicsGuidedForecaster(N_FEATURES, cfg)
                    train_model(model, loaders["train"], loaders["val"], cfg, scaler=scaler,
                                fold_name="official", out_dir=out_dir, verbose=True)
                model = PhysicsGuidedForecaster(N_FEATURES, cfg).to(device)
                state = load_checkpoint(ckpt)
                model.load_state_dict(state["model_state_dict"])
                ev = predict_loader(model, loaders["test"], device)
                preds = {sid: ev["predictions"][ev["subject_index"] == i] for i, sid in enumerate(test_ids)}
                res = evaluate_predictions(method=f"{arm}_f{frac}_s{seed}", fold=fold, part="test",
                                           corpus=corpus, predictions=preds, config=cfg)
                for h in HORIZONS:
                    for sid, v in res.metric_by_subject("mae", h).items():
                        rows.append(dict(arm=arm, frac=frac, seed=seed, subject_id=sid, horizon_min=h, mae=float(v)))
                runs.append(dict(arm=arm, frac=frac, seed=seed, n_train_subjects=k,
                                 train_subjects=";".join(chosen), best_epoch=state.get("best_epoch"),
                                 best_val_mae_30=state.get("best_val_mae_30"),
                                 test_mae_30=float(res.summary[res.summary.horizon_min == 30]["mae_mean"].iloc[0])))
                print(f"[{arm} f{frac} s{seed}] best_epoch={state.get('best_epoch')} test MAE@30={runs[-1]['test_mae_30']:.3f}", flush=True)
                del model
                torch.cuda.empty_cache()
                Path("results/tables").mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv("results/tables/round6_p6_per_subject.csv", index=False)
                pd.DataFrame(runs).to_csv("results/tables/round6_p6_runs.csv", index=False)


if __name__ == "__main__":
    main()
