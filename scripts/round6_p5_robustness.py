#!/usr/bin/env python3
"""P5 robustness (post hoc, NOT pre-registered): where in the horizon do |dG/dt|>4 violations occur?

Usage (repo root, .venv active):
  # 1) seed-42 checkpoints predate the quantile head -> dump their 121-pt collocation with the OLD code
  #    git worktree add --detach /path/wt 7eb9ad2; ln -s $PWD/OhioT1DM /path/wt/OhioT1DM
  python scripts/round6_p5_robustness.py dump42 /path/wt
  # 2) analysis
  python scripts/round6_p5_robustness.py run

Variants of the per-window "steep" flag (threshold 4 mg/dL/min), from the 121-pt, 1-min collocation trajectory:
  full         first differences over all 120 one-minute intervals (reproduces P5)
  a_excl_last  intervals ending at minute <= 110
  b_excl_both  intervals within minutes [10, 110]
  c_5min       25-point subsample every 5 min, finite difference / 5
  c_30min      the 5 points 0,30,60,90,120 (anchor + native horizons), finite difference / 30
Same unit of inference/bootstrap as P5: per-subject fraction, seed-averaged, paired Wilcoxon, subject bootstrap
(5000, rng 42) via scripts.round6_regime_analysis.paired_test. No Holm (post hoc, descriptive).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THR = 4.0
SEEDS = (42, 101, 202, 303, 404)
OUT = Path("results/tables")
REPO = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))


def dump42(wt):
    sys.path.insert(0, wt)
    os.chdir(wt)
    import torch
    from twin.config import Config
    from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
    from twin.data.features import N_FEATURES
    from twin.data.splits import official_split
    from twin.models.forecaster import PhysicsGuidedForecaster
    base = Config.from_yaml("configs/official-small.yaml"); base.train.batch_size = 256; base.train.num_workers = 0
    corpus = load_corpus(base, cache_dir=None)
    tr = {k: v.windows for k, v in corpus["train"].items()}; te = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(list(tr.values()), list(te.values()), val_fraction=base.split.val_fraction, purge_steps=base.split.purge_steps)
    loader = build_loader(build_dataset(fold, "test", corpus, fit_scaler(fold, corpus), base), base, shuffle=False)
    for arm in ("A0", "A7"):
        ck = torch.load(REPO / f"artifacts/official/abl-{arm}/official/best_model.pt", map_location="cpu", weights_only=False)
        model = PhysicsGuidedForecaster(N_FEATURES, Config.from_dict(ck["config"])).cuda()
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        coll = []
        with torch.no_grad():
            for b in loader:
                coll.append(model({k: v.cuda() for k, v in b.items()}).collocation.float().cpu().numpy())
        coll = np.concatenate(coll)
        ref = np.load(REPO / f"artifacts/round6_seed42_traj_{arm}.npz")
        print(arm, coll.shape, "max |coll slope| vs P5 stored", np.abs(np.abs(np.diff(coll, axis=1)).max(1) - ref["slope"]).max())
        np.savez(REPO / f"artifacts/round6_p5rob_seed42_coll_{arm}.npz", coll=coll)


def variants(c):
    """c: (N,121) trajectory. Returns dict name -> (N,) bool, and (N,120) abs slope."""
    s = np.abs(np.diff(c, axis=1))  # per-minute |dG/dt|, interval j = [j, j+1]
    v = {
        "full": (s > THR).any(1),
        "a_excl_last10": (s[:, :110] > THR).any(1),
        "b_excl_first_last10": (s[:, 10:110] > THR).any(1),
        "c_5min_points": (np.abs(np.diff(c[:, ::5], axis=1)) / 5 > THR).any(1),
        "c_30min_points": (np.abs(np.diff(c[:, ::30], axis=1)) / 30 > THR).any(1),
    }
    return v, s


def run():
    sys.path.insert(0, str(REPO)); os.chdir(REPO)
    import torch
    import scripts.round6_regime_analysis as ra
    from twin.data.features import N_FEATURES
    from twin.models.forecaster import PhysicsGuidedForecaster
    from twin.train.loop import load_checkpoint
    import scripts.run_seeds_a0_a7 as rs

    base, corpus, loader = ra.build_test()
    device = base.resolve_device()
    ids, subj, targets, post, roc = ra.window_meta(corpus, loader, base.data.grid_minutes)
    n_subj, N = len(ids), len(subj)

    flags, hist, meta = {}, {}, {}
    for arm in ("A0", "A7"):
        for s in SEEDS:
            if s == 42:
                coll = np.load(REPO / f"artifacts/round6_p5rob_seed42_coll_{arm}.npz")["coll"]
            else:
                cfg = rs.get_model_config(arm, s)
                model = PhysicsGuidedForecaster(N_FEATURES, cfg).to(device)
                model.load_state_dict(load_checkpoint(ra.ckpt_path(arm, s))["model_state_dict"]); model.eval()
                outs = []
                with torch.no_grad():
                    for b in loader:
                        o = model({k: v.to(device) for k, v in b.items()})
                        outs.append(o.collocation.float().cpu().numpy())
                    if s == SEEDS[1]:
                        meta[arm] = dict(hybrid=model.hybrid, grid=model.spline.collocation_min.cpu().numpy()[[0, 1, -1]].tolist(),
                                         n_coll=len(model.spline.collocation_min), gate=o.prior_gate)
                coll = np.concatenate(outs)
            assert coll.shape == (N, 121), coll.shape
            v, sl = variants(coll)
            flags[(arm, s)] = v
            # per-minute-interval violation rate and location of the max-slope interval
            hist[(arm, s)] = dict(rate=(sl > THR).mean(0), argmax=np.bincount(sl.argmax(1)[sl.max(1) > THR], minlength=120),
                                  nviol=(sl > THR).sum(1))
            print(arm, s, {k: round(float(x.mean()), 4) for k, x in v.items()}, flush=True)
    print("model meta", meta)

    rows = []
    names = list(next(iter(flags.values())).keys())
    for nm in names:
        per_arm = {}
        for arm in ("A0", "A7"):
            per_arm[arm] = np.mean([ra.subject_mean(flags[(arm, s)][nm].astype(float), subj, np.ones(N, bool), n_subj) for s in SEEDS], axis=0)
        r = ra.paired_test(per_arm["A0"], per_arm["A7"], in_holm=False)
        r.update(variant=nm, **{f"{a}_seed{s}": float(flags[(a, s)][nm].mean()) for a in ("A0", "A7") for s in SEEDS})
        rows.append(r)
    df = pd.DataFrame(rows)
    front = ["variant", "n_subjects", "a0_mean", "a7_mean", "gap", "ci_lo", "ci_hi", "p_raw"]
    df = df[front + [c for c in df.columns if c not in front and c != "in_holm"]]
    df["section"] = "summary"

    # histogram by minute (mean over seeds of per-interval violation rate), and 10-min bins
    h_rows = []
    for arm in ("A0", "A7"):
        rate = np.mean([hist[(arm, s)]["rate"] for s in SEEDS], axis=0)
        am = np.sum([hist[(arm, s)]["argmax"] for s in SEEDS], axis=0)
        for m in range(120):
            h_rows.append(dict(section="by_minute", arm=arm, minute_start=m, rate_windows_with_violation_in_interval=rate[m],
                               n_windows_argmax_here_all_seeds=int(am[m])))
    hd = pd.DataFrame(h_rows)
    full = pd.concat([df, hd], ignore_index=True)
    full.to_csv(OUT / "round6_p5_robustness.csv", index=False)

    # extras for the write-up
    print(df.drop(columns=["section"]).to_string())
    for arm in ("A0", "A7"):
        rate = np.mean([hist[(arm, s)]["rate"] for s in SEEDS], axis=0)
        am = np.sum([hist[(arm, s)]["argmax"] for s in SEEDS], axis=0)
        print(arm, "rate by 10-min bin (mean over seeds):", [round(float(rate[i:i+10].mean()), 4) for i in range(0, 120, 10)])
        print(arm, "share of violating windows (all seeds) whose max-slope interval falls in 10-min bin:",
              [round(float(am[i:i+10].sum() / max(am.sum(), 1)), 3) for i in range(0, 120, 10)])
        print(arm, "last 5 min rates", [round(float(x), 3) for x in rate[-5:]], "peak minute", int(rate.argmax()))
        nv = np.mean([hist[(arm, s)]["nviol"][hist[(arm, s)]["nviol"] > 0].mean() if (hist[(arm, s)]["nviol"] > 0).any() else 0 for s in SEEDS])
        print(arm, "mean # violating minutes per violating window:", round(float(nv), 2))
    # which windows violate only in the last 10 min
    for arm in ("A0",):
        only_last = np.mean([(flags[(arm, s)]["full"] & ~flags[(arm, s)]["a_excl_last10"]).sum() / max(flags[(arm, s)]["full"].sum(), 1) for s in SEEDS])
        print("A0 share of violating windows that violate ONLY in last 10 min:", round(float(only_last), 3))


if __name__ == "__main__":
    if sys.argv[1] == "dump42":
        dump42(sys.argv[2])
    else:
        run()
