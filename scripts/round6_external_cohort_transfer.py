#!/usr/bin/env python3
"""Round-6 P8: cross-cohort transfer, A0 (fair curriculum) vs A7, OhioT1DM 2018<->2020.

Train on one cohort's train files (validation = purged tail of those same subjects),
test on the other cohort's official test files. 5 seeds, fresh training for every seed.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seeds_a0_a7 import get_model_config, SEEDS, HORIZONS  # same curricula as round 5
from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.features import N_FEATURES
from twin.data.ohio import COHORT_SUBJECTS
from twin.data.splits import official_split, verify_no_leakage
from twin.eval.runner import evaluate_predictions
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.seeding import set_seed
from twin.train.loop import predict_loader, train_model

BATCH = 128
DIRECTIONS = [("2018", "2020"), ("2020", "2018")]
rng = np.random.default_rng(0)


def main():
    base = Config.from_yaml("configs/official-small.yaml")
    corpus = load_corpus(base)
    device = base.resolve_device()
    rows = []  # per direction/arm/seed/subject/horizon
    for src, tgt in DIRECTIONS:
        src_ids, tgt_ids = set(COHORT_SUBJECTS[src]), set(COHORT_SUBJECTS[tgt])
        c = {"train": {k: v for k, v in corpus["train"].items() if k in src_ids},
             "test": {k: v for k, v in corpus["test"].items() if k in tgt_ids}}
        assert len(c["train"]) == 6 and len(c["test"]) == 6, (len(c["train"]), len(c["test"]))
        tr = {k: v.windows for k, v in c["train"].items()}
        te = {k: v.windows for k, v in c["test"].items()}
        fold = official_split(list(tr.values()), list(te.values()),
                              val_fraction=base.split.val_fraction, purge_steps=base.split.purge_steps)
        verify_no_leakage(fold, tr, te)
        scaler = fit_scaler(fold, c)
        for arm in ("A0", "A7"):
            for seed in SEEDS:
                cfg = get_model_config(arm, seed)
                cfg.train.batch_size = BATCH
                cfg.train.num_workers = 2
                loaders = {p: build_loader(build_dataset(fold, p, c, scaler, cfg), cfg, shuffle=(p == "train"))
                           for p in ("train", "val", "test")}
                sids = loaders["test"].dataset.subject_ids
                out = Path(f"artifacts/round6_external/{src}to{tgt}/{arm}_s{seed}")
                out.mkdir(parents=True, exist_ok=True)
                if not (out / "best_model.pt").is_file():
                    print(f"[{src}->{tgt} {arm} s{seed}] training", flush=True)
                    set_seed(seed, deterministic=True)
                    model = PhysicsGuidedForecaster(N_FEATURES, cfg)
                    train_model(model, loaders["train"], loaders["val"], cfg, scaler=scaler,
                                fold_name="official", out_dir=out, verbose=False)
                model = PhysicsGuidedForecaster(N_FEATURES, cfg).to(device)
                model.load_state_dict(torch.load(out / "best_model.pt", map_location=device)["model_state_dict"])
                ev = predict_loader(model, loaders["test"], device)
                preds = {s: ev["predictions"][ev["subject_index"] == i] for i, s in enumerate(sids)}
                res = evaluate_predictions(method=f"{arm}_s{seed}", fold=fold, part="test",
                                           corpus=c, predictions=preds, config=cfg)
                for h in HORIZONS:
                    mae = res.metric_by_subject("mae", h)
                    rmse = res.metric_by_subject("rmse", h)
                    for s in mae.index:
                        rows.append(dict(direction=f"{src}->{tgt}", arm=arm, seed=seed, subject=s,
                                         horizon_min=h, mae=float(mae[s]), rmse=float(rmse[s])))
                pd.DataFrame(rows).to_csv("results/tables/round6_external_per_seed_subject.csv", index=False)
                del model, loaders
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    subj = df.groupby(["direction", "arm", "subject", "horizon_min"])[["mae", "rmse"]].mean().reset_index()
    subj.to_csv("results/tables/round6_external_per_subject.csv", index=False)
    out = []
    for d in df.direction.unique():
        for h in HORIZONS:
            row = dict(direction=d, horizon_min=h)
            for m in ("mae", "rmse"):
                a0 = subj[(subj.direction == d) & (subj.arm == "A0") & (subj.horizon_min == h)].sort_values("subject")[m].to_numpy()
                a7 = subj[(subj.direction == d) & (subj.arm == "A7") & (subj.horizon_min == h)].sort_values("subject")[m].to_numpy()
                diff = a0 - a7  # positive = A7 better
                bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
                row.update({f"A0_{m}": a0.mean(), f"A7_{m}": a7.mean(), f"{m}_A0minusA7": diff.mean(),
                            f"{m}_ci_lo": np.percentile(bs, 2.5), f"{m}_ci_hi": np.percentile(bs, 97.5),
                            f"{m}_wilcoxon_p": wilcoxon(a0, a7).pvalue, f"{m}_n_subj_A7_better": int((diff > 0).sum())})
            out.append(row)
    res = pd.DataFrame(out)
    # Holm within P8 (all 2 dirs x 4 horizons x 2 metrics = 16 tests)
    for m in ("mae", "rmse"):
        pass
    ps = np.concatenate([res["mae_wilcoxon_p"], res["rmse_wilcoxon_p"]])
    order = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    res["mae_p_holm"], res["rmse_p_holm"] = adj[:len(res)], adj[len(res):]
    res.to_csv("results/tables/round6_external_summary.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
