#!/usr/bin/env python3
"""Round-6 P9: multi-seed q10<70 hypoglycemia alarm, A0 vs A7 (5 seeds, matched configs).

Seeds 101-404 reuse artifacts/seeds/{arm}_s{seed}. Seed 42 is retrained for both arms with
the exact config from run_seeds_a0_a7.get_model_config into artifacts/seeds_round6/{arm}_s42
(new dirs; nothing existing is overwritten). Usage: python scripts/round6_hypo_multiseed.py [train|eval]
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch

from run_seeds_a0_a7 import (Config, build_dataset, build_loader, fit_scaler, load_corpus,
                             official_split, verify_no_leakage, evaluate_predictions,
                             get_model_config, PhysicsGuidedForecaster, N_FEATURES, set_seed,
                             predict_loader, train_model)
from twin.metrics.clinical import event_based_hypoglycemia_detection as ev

SEEDS = [42, 101, 202, 303, 404]
HORIZONS = [30, 60, 90, 120]
NEW42 = "artifacts/seeds_round6/{arm}_s42/official"


def ckpt_dir(arm, seed):
    return NEW42.format(arm=arm) if seed == 42 else f"artifacts/seeds/{arm}_s{seed}/official"


def setup():
    base = Config.from_yaml("configs/official-small.yaml")
    base.train.batch_size = 256
    base.train.num_workers = 3
    corpus = load_corpus(base)
    tr = {k: v.windows for k, v in corpus["train"].items()}
    te = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(list(tr.values()), list(te.values()),
                          val_fraction=base.split.val_fraction, purge_steps=base.split.purge_steps)
    verify_no_leakage(fold, tr, te)
    sc = fit_scaler(fold, corpus)
    loaders = {p: build_loader(build_dataset(fold, p, corpus, sc, base), base, shuffle=(p == "train"))
               for p in ("train", "val", "test")}
    return base, corpus, fold, sc, loaders, base.resolve_device()


def train_s42():
    base, corpus, fold, sc, loaders, dev = setup()
    for arm in ["A0", "A7"]:  # sequential
        out = Path(NEW42.format(arm=arm))
        if (out / "best_model.pt").is_file():
            print("exists, skip", out); continue
        out.mkdir(parents=True, exist_ok=False)
        cfg = get_model_config(arm, 42)
        set_seed(42, deterministic=True)
        model = PhysicsGuidedForecaster(N_FEATURES, cfg)
        train_model(model, loaders["train"], loaders["val"], cfg, scaler=sc,
                    fold_name="official", out_dir=out, verbose=True)
        torch.cuda.empty_cache()


def evaluate():
    base, corpus, fold, sc, loaders, dev = setup()
    sids = loaders["test"].dataset.subject_ids
    rows, mae_rows, persubj = [], [], {}
    for arm in ["A0", "A7"]:
        for seed in SEEDS:
            cfg = get_model_config(arm, seed)
            m = PhysicsGuidedForecaster(N_FEATURES, cfg).to(dev)
            ck = torch.load(Path(ckpt_dir(arm, seed)) / "best_model.pt", map_location=dev, weights_only=False)
            m.load_state_dict(ck["model_state_dict"])
            e = predict_loader(m, loaders["test"], dev)
            t = e["targets"][:, 0]; s = e["subject_index"]; q10 = e["quantile_predictions"][:, 0, 0]
            r = ev(t, q10, s, threshold=70.0)
            per = []
            for i in np.unique(s):
                mk = s == i
                rr = ev(t[mk], q10[mk], threshold=70.0)
                per.append((rr.detected_episodes, rr.total_episodes))
            persubj[(arm, seed)] = np.array(per)
            assert persubj[(arm, seed)].sum(0)[1] == r.total_episodes
            spec = getattr(r, "specificity", np.nan)
            if np.isnan(spec):
                neg = t >= 70.0
                spec = float((q10[neg] >= 70.0).mean())  # fallback: window-level
            rows.append(dict(arm=arm, seed=seed, n_windows=len(t), episodes=r.total_episodes,
                             detected=r.detected_episodes, event_sensitivity=r.event_sensitivity,
                             specificity=spec, fa_per_day=r.false_alarms_per_day,
                             median_lead_min=r.median_lead_time_min,
                             sum_persubject_detected=int(persubj[(arm, seed)].sum(0)[0])))
            preds = {sid: e["predictions"][s == k] for k, sid in enumerate(sids)}
            res = evaluate_predictions(method=f"{arm}_s{seed}_r6", fold=fold, part="test",
                                       corpus=corpus, predictions=preds, config=cfg)
            for h in HORIZONS:
                v = res.summary[res.summary.horizon_min == h]
                mae_rows.append(dict(arm=arm, seed=seed, horizon_min=h,
                                     mae=float(v.mae_mean.iloc[0]), rmse=float(v.rmse_mean.iloc[0])))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    mae = pd.DataFrame(mae_rows)

    # persistence baseline (validation-tuned, round 5; not retuned)
    pb = pd.read_csv("results/tables/hypo_alarm_baseline_round5.csv")
    pb = pb[pb.baseline == "Persistence alarm"].iloc[0]

    # two-level bootstrap on A7-A0 sensitivity
    rng = np.random.default_rng(20260626)
    nsub = persubj[("A0", 42)].shape[0]
    A = {arm: np.stack([persubj[(arm, s)] for s in SEEDS]) for arm in ["A0", "A7"]}  # seed,subj,2
    diffs = np.empty(5000)
    for b in range(5000):
        si = rng.integers(0, nsub, nsub)
        sens = {}
        for arm in ["A0", "A7"]:
            sd = rng.integers(0, len(SEEDS), len(SEEDS))
            x = A[arm][sd][:, si, :].sum(1)  # seeds, 2
            sens[arm] = np.mean(x[:, 0] / np.maximum(x[:, 1], 1))
        diffs[b] = sens["A7"] - sens["A0"]
    ci = np.percentile(diffs, [2.5, 97.5])
    obs = df[df.arm == "A7"].event_sensitivity.mean() - df[df.arm == "A0"].event_sensitivity.mean()

    out = []
    for arm in ["A0", "A7"]:
        d = df[df.arm == arm]
        for _, r in d.iterrows():
            out.append(dict(row=f"{arm}_seed{int(r.seed)}", **{k: r[k] for k in
                       ["episodes", "detected", "event_sensitivity", "specificity", "fa_per_day", "median_lead_min"]}))
        row = dict(row=f"{arm}_mean", episodes=d.episodes.iloc[0])
        row2 = dict(row=f"{arm}_sd", episodes=d.episodes.iloc[0])
        for k in ["detected", "event_sensitivity", "specificity", "fa_per_day", "median_lead_min"]:
            row[k] = d[k].mean(); row2[k] = d[k].std(ddof=1)
        out += [row, row2]
    out.append(dict(row="A7_minus_A0_sensitivity_obs", event_sensitivity=obs))
    out.append(dict(row="A7_minus_A0_sensitivity_boot_ci95_lo", event_sensitivity=ci[0]))
    out.append(dict(row="A7_minus_A0_sensitivity_boot_ci95_hi", event_sensitivity=ci[1]))
    out.append(dict(row="boot_frac_diff_gt0", event_sensitivity=float((diffs > 0).mean())))
    out.append(dict(row="persistence_val_tuned", episodes=pb.test_total_episodes, detected=pb.test_detected_episodes,
                    event_sensitivity=pb.test_event_sensitivity, specificity=pb.test_specificity,
                    fa_per_day=pb.test_fa_per_day, median_lead_min=pb.test_median_lead_time_min))
    o = pd.DataFrame(out)
    Path("results/tables").mkdir(exist_ok=True)
    o.to_csv("results/tables/round6_hypo_multiseed.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.to_string(index=False))

    old = pd.read_csv("results/tables/multiseed_a0_vs_a7.csv")
    print("\n5-seed MAE/RMSE with retrained seed 42 (new) vs existing table")
    for arm in ["A0", "A7"]:
        for h in HORIZONS:
            d = mae[(mae.arm == arm) & (mae.horizon_min == h)].set_index("seed")
            o_ = old[old.horizon_min == h].iloc[0]
            print(arm, h, "s42 new MAE %.3f" % d.mae[42], "| others", np.round(d.mae.drop(42).values, 3),
                  "| new 5-seed %.3f±%.3f" % (d.mae.mean(), d.mae.std(ddof=1)),
                  "| table %.3f±%.3f" % (o_[f"{arm}_mae_mean_of_seeds"], o_[f"{arm}_mae_sd_of_seeds"]))
    mae.to_csv("results/tables/round6_hypo_multiseed_mae.csv", index=False)


if __name__ == "__main__":
    train_s42() if (len(sys.argv) > 1 and sys.argv[1] == "train") else evaluate()
