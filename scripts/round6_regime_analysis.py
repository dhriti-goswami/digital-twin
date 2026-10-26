#!/usr/bin/env python3
"""Round 6 pre-registered analyses P1-P5 and P7 (plus the P6 statistics) for A7 vs fair A0.

Pre-registration: paper/digital-twin-v5/PREREG_ROUND6.md (commit c84348f).
Run scripts/round6_low_data.py first (P6 training); this script then reads
results/tables/round6_p6_per_subject.csv for the P6 statistics.

Conventions
-----------
* Models: A0 (fair, round-5 selection window) and A7, seeds 42/101/202/303/404, loaded exactly as
  scripts/run_seeds_a0_a7.py does (seed-42 from artifacts/official/abl-*, others artifacts/seeds).
  All 10 checkpoints are re-run on the Table-VII test loader; the overall seed-averaged
  per-subject MAE is asserted to reproduce Table VII (results/tables/multiseed_a0_vs_a7.csv).
* Unit of inference: subject. Per-subject, per-stratum MAE is averaged over seeds, then
  A0 - A7 is formed per subject ("gap": positive = A7 better). Paired two-sided Wilcoxon and a
  subject-level percentile bootstrap (5000 resamples, rng seed 42) for the 95% CI.
* One Holm-Bonferroni family over every test with in_holm=True, across all analyses.
* Stratum definitions (fixed before looking at data; see ROUND6_RESULTS.md for interpretation
  choices where the pre-registration was silent):
    P2 post-meal: anchor lies 0-180 min after the most recent grid slot with carbs_mg_per_min>0.
    P3 range: by the TARGET glucose at the evaluated horizon: <70, 70-180 (inclusive), >180.
    P4 ROC: |roc_15min| at the anchor (feature roc_15min, mg/dL/min): <1, [1,2], >2.
    P5: from the 121-point 1-min collocation trajectory (output.collocation); dG/dt is the
        first difference of that trajectory per minute.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import levene, spearmanr, wilcoxon

from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.features import FEATURE_NAMES, N_FEATURES
from twin.data.splits import official_split, verify_no_leakage
from twin.metrics import holm_bonferroni
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.train.loop import load_checkpoint

import scripts.run_seeds_a0_a7 as rs

SEEDS = rs.SEEDS
HORIZONS = rs.HORIZONS
N_BOOT = 5000
RNG = np.random.default_rng(42)
OUT = Path("results/tables")
ROC_COL = FEATURE_NAMES.index("roc_15min")
ROC_STEP_ABS = 4.0
POSTMEAL_MIN = 180


def ckpt_path(arm, seed):
    if seed == 42:
        return Path(f"artifacts/official/abl-{arm}/official/best_model.pt")
    return Path(f"artifacts/seeds/{arm}_s{seed}/official/best_model.pt")


# --------------------------------------------------------------------------- statistics
def paired_test(a0, a7, in_holm=True):
    """a0, a7: per-subject values (aligned). gap = mean(a0 - a7)."""
    a0, a7 = np.asarray(a0, float), np.asarray(a7, float)
    ok = ~(np.isnan(a0) | np.isnan(a7))
    a0, a7 = a0[ok], a7[ok]
    d = a0 - a7
    n = len(d)
    if n < 3:
        return dict(n_subjects=n, a0_mean=np.nan, a7_mean=np.nan, gap=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, p_raw=np.nan, in_holm=False)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(a0, a7, alternative="two-sided").pvalue)
    return dict(n_subjects=n, a0_mean=a0.mean(), a7_mean=a7.mean(), gap=d.mean(), ci_lo=lo,
                ci_hi=hi, p_raw=p, in_holm=in_holm)


def subject_mean(values, subj, mask, n_subj):
    """Per-subject mean of values[mask]; NaN if the subject has no such windows."""
    out = np.full(n_subj, np.nan)
    for i in range(n_subj):
        m = mask & (subj == i)
        if m.any():
            out[i] = values[m].mean()
    return out


# --------------------------------------------------------------------------- setup / inference
def build_test():
    base = Config.from_yaml("configs/official-small.yaml")
    base.train.batch_size = 256
    base.train.num_workers = 3
    corpus = load_corpus(base)
    train_sets = {k: v.windows for k, v in corpus["train"].items()}
    test_sets = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(list(train_sets.values()), list(test_sets.values()),
                          val_fraction=base.split.val_fraction, purge_steps=base.split.purge_steps)
    verify_no_leakage(fold, train_sets, test_sets)
    scaler = fit_scaler(fold, corpus)
    loader = build_loader(build_dataset(fold, "test", corpus, scaler, base), base, shuffle=False)
    return base, corpus, loader


@torch.no_grad()
def run_model(arm, seed, loader, device):
    if seed == 42:
        # pre-quantile-head checkpoints: produced by scripts/round6_seed42_trajectories.py (old code;
        # predictions verified bit-identical to artifacts/official/abl-*/.../predictions.npz)
        z = np.load(f"artifacts/round6_seed42_traj_{arm}.npz")
        return dict(pred=z["pred"], cmax=z["cmax"], cmin=z["cmin"], slope=z["slope"])
    cfg = rs.get_model_config(arm, seed)
    model = PhysicsGuidedForecaster(N_FEATURES, cfg).to(device)
    model.load_state_dict(load_checkpoint(ckpt_path(arm, seed))["model_state_dict"])
    model.eval()
    hz, mx, mn, rmax = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        coll = out.collocation.float()
        dt = float(model.spline.collocation_min[1] - model.spline.collocation_min[0])
        slope = (coll[:, 1:] - coll[:, :-1]).abs() / dt
        hz.append(out.horizons.float().cpu().numpy())
        mx.append(coll.max(1).values.cpu().numpy())
        mn.append(coll.min(1).values.cpu().numpy())
        rmax.append(slope.max(1).values.cpu().numpy())
    return dict(pred=np.concatenate(hz), cmax=np.concatenate(mx), cmin=np.concatenate(mn),
                slope=np.concatenate(rmax))


def window_meta(corpus, loader, grid_min):
    ids = loader.dataset.subject_ids
    subj, tgt, post, roc = [], [], [], []
    for i, sid in enumerate(ids):
        d = corpus["test"][sid]
        anchors = d.windows.anchors
        carbs = d.frame["carbs_mg_per_min"].to_numpy()
        cs = np.flatnonzero(carbs > 0)
        pos = np.searchsorted(cs, anchors, side="right") - 1
        delta = np.where(pos >= 0, (anchors - cs[np.clip(pos, 0, None)]) * grid_min, np.inf) if cs.size else np.full(len(anchors), np.inf)
        subj.append(np.full(len(anchors), i))
        tgt.append(d.windows.targets)
        post.append(delta <= POSTMEAL_MIN)
        roc.append(np.abs(d.features.values[anchors, ROC_COL]))
    return (ids, np.concatenate(subj), np.concatenate(tgt), np.concatenate(post), np.concatenate(roc))


def fmt_p(p):
    return "nan" if p is None or np.isnan(p) else (f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}")


def verdict(r):
    if np.isnan(r["gap"]):
        return "n/a"
    sig = r["ci_lo"] > 0 or r["ci_hi"] < 0
    if sig and r["p_holm"] < 0.05:
        return "helps" if r["gap"] > 0 else "hurts"
    return "no difference detected"


def main():
    base, corpus, loader = build_test()
    device = base.resolve_device()
    ids, subj, targets, post, roc = window_meta(corpus, loader, base.data.grid_minutes)
    n_subj = len(ids)
    N = len(subj)
    print(f"{N} test windows, {n_subj} subjects; post-meal windows {post.sum()} ({post.mean():.1%}); "
          f"postmeal windows per subject {[int(post[subj==i].sum()) for i in range(n_subj)]}")

    R = {arm: {s: run_model(arm, s, loader, device) for s in SEEDS} for arm in ("A0", "A7")}

    # ---- Table-VII reproduction check (overall seed-averaged per-subject MAE, mean over subjects)
    t7 = pd.read_csv(OUT / "multiseed_a0_vs_a7.csv").set_index("horizon_min")
    for hi, h in enumerate(HORIZONS):
        for arm in ("A0", "A7"):
            per_seed = [subject_mean(np.abs(R[arm][s]["pred"][:, hi] - targets[:, hi]), subj,
                                     np.ones(N, bool), n_subj).mean() for s in SEEDS]
            assert abs(np.mean(per_seed) - t7.loc[h, f"{arm}_mae_mean_of_seeds"]) < 5e-3, (arm, h, np.mean(per_seed))
            assert abs(np.std(per_seed, ddof=1) - t7.loc[h, f"{arm}_mae_sd_of_seeds"]) < 5e-3, (arm, h)
    print("Table VII reproduction check passed (means and SDs, all arms/horizons).")

    def err(arm, s, hi):
        return np.abs(R[arm][s]["pred"][:, hi] - targets[:, hi])

    def strat_gap(hi, mask, in_holm=True):
        a0 = np.mean([subject_mean(err("A0", s, hi), subj, mask, n_subj) for s in SEEDS], axis=0)
        a7 = np.mean([subject_mean(err("A7", s, hi), subj, mask, n_subj) for s in SEEDS], axis=0)
        return a0, a7, paired_test(a0, a7, in_holm)

    tests = []  # all rows for Holm

    def add(analysis, label, horizon, res, extra=None, window_n=None):
        row = dict(analysis=analysis, stratum=label, horizon_min=horizon, **res)
        if window_n is not None:
            row["n_windows"] = window_n
        if extra:
            row.update(extra)
        tests.append(row)
        return row

    allm = np.ones(N, bool)

    # ---- P1 horizon trend
    gaps = np.zeros((n_subj, len(HORIZONS)))
    for hi, h in enumerate(HORIZONS):
        a0, a7, res = strat_gap(hi, allm)
        gaps[:, hi] = a0 - a7
        add("P1", "all", h, res, dict(rel_gap_pct=100 * res["gap"] / res["a0_mean"]), N)
    rhos = np.array([spearmanr(HORIZONS, gaps[i]).statistic for i in range(n_subj)])
    rho_ok = rhos[~np.isnan(rhos)]
    idx = RNG.integers(0, len(rho_ok), size=(N_BOOT, len(rho_ok)))
    lo, hi_ = np.percentile(rho_ok[idx].mean(axis=1), [2.5, 97.5])
    p_trend = 1.0 if np.allclose(rho_ok, 0) else float(wilcoxon(rho_ok, alternative="two-sided").pvalue)
    pooled = spearmanr(np.tile(HORIZONS, n_subj), gaps.ravel())
    p1_trend = dict(mean_subject_rho=rho_ok.mean(), ci_lo=lo, ci_hi=hi_, p_raw=p_trend,
                    n_subjects=len(rho_ok), pooled_rho_descriptive=pooled.statistic,
                    pooled_p_descriptive=pooled.pvalue,
                    n_subjects_gap_increasing=int((rhos > 0).sum()))
    tests.append(dict(analysis="P1", stratum="trend (per-subject Spearman gap~horizon, Wilcoxon vs 0)",
                      horizon_min=-1, gap=rho_ok.mean(), ci_lo=lo, ci_hi=hi_, p_raw=p_trend,
                      n_subjects=len(rho_ok), in_holm=True))
    pd.DataFrame([p1_trend]).to_csv(OUT / "round6_p1_trend.csv", index=False)

    # ---- P2 post-meal
    for hi, h in enumerate(HORIZONS):
        ga = {}
        for name, mask in (("post-meal (0-180 min)", post), ("other", ~post)):
            a0, a7, res = strat_gap(hi, mask)
            ga[name] = a0 - a7
            add("P2", name, h, res, window_n=int(mask.sum()))
        d_pm, d_ot = ga["post-meal (0-180 min)"], ga["other"]
        ok = ~(np.isnan(d_pm) | np.isnan(d_ot))
        # difference of gaps, paired by subject: reuse paired_test with a0=d_pm, a7=d_ot
        res = paired_test(d_pm[ok], d_ot[ok])
        res.update(a0_mean=np.nan, a7_mean=np.nan)
        add("P2", "difference of gaps (post-meal minus other)", h, res)

    # ---- P3 glycemic range of target
    for hi, h in enumerate(HORIZONS):
        t = targets[:, hi]
        for name, mask in (("target <70", t < 70), ("target 70-180", (t >= 70) & (t <= 180)), ("target >180", t > 180)):
            a0, a7, res = strat_gap(hi, mask)
            add("P3", name, h, res, window_n=int(mask.sum()))

    # ---- P4 rate of change
    for hi, h in enumerate(HORIZONS):
        for name, mask in (("|ROC15|<1", roc < 1), ("|ROC15| 1-2", (roc >= 1) & (roc <= 2)), ("|ROC15|>2", roc > 2)):
            a0, a7, res = strat_gap(hi, mask)
            add("P4", name, h, res, window_n=int(mask.sum()))

    # ---- P5 plausibility
    for name, fn in (("frac trajectories with |dG/dt|>4 mg/dL/min", lambda r: (r["slope"] > ROC_STEP_ABS).astype(float)),
                     ("frac trajectories <40 or >400 mg/dL", lambda r: ((r["cmin"] < 40) | (r["cmax"] > 400)).astype(float))):
        a0 = np.mean([subject_mean(fn(R["A0"][s]), subj, allm, n_subj) for s in SEEDS], axis=0)
        a7 = np.mean([subject_mean(fn(R["A7"][s]), subj, allm, n_subj) for s in SEEDS], axis=0)
        add("P5", name, -1, paired_test(a0, a7), window_n=N)

    # ---- P7 seed variance
    p7 = []
    for hi, h in enumerate(HORIZONS):
        v = {arm: np.array([subject_mean(err(arm, s, hi), subj, allm, n_subj).mean() for s in SEEDS]) for arm in ("A0", "A7")}
        sd0, sd7 = v["A0"].std(ddof=1), v["A7"].std(ddof=1)
        b0 = RNG.integers(0, len(SEEDS), size=(N_BOOT, len(SEEDS)))
        b7 = RNG.integers(0, len(SEEDS), size=(N_BOOT, len(SEEDS)))
        diffs = v["A0"][b0].std(axis=1, ddof=1) - v["A7"][b7].std(axis=1, ddof=1)
        lo, hi_ = np.percentile(diffs, [2.5, 97.5])
        p = float(levene(v["A0"], v["A7"], center="median").pvalue)
        row = dict(analysis="P7", stratum="across-seed SD of mean MAE (gap = SD_A0 - SD_A7)", horizon_min=h,
                   n_subjects=len(SEEDS), a0_mean=sd0, a7_mean=sd7, gap=sd0 - sd7, ci_lo=lo, ci_hi=hi_,
                   p_raw=p, in_holm=True, sd_ratio_a7_over_a0=sd7 / sd0)
        tests.append(row)
        p7.append(row)

    # ---- P6 low data (statistics)
    p6path = OUT / "round6_p6_per_subject.csv"
    p6_note = None
    if p6path.is_file():
        d = pd.read_csv(p6path)
        for frac in sorted(d.frac.unique()):
            for h in HORIZONS:
                sub = d[(d.frac == frac) & (d.horizon_min == h)]
                piv = sub.groupby(["arm", "subject_id"]).mae.mean().unstack(0).sort_index()
                nseed = sub.seed.nunique()
                add("P6", f"train on {int(frac*100)}% of subjects ({nseed} seeds)", h, paired_test(piv["A0"], piv["A7"]))
        # descriptive reference, same 3 seeds on full data (not in Holm)
        for hi, h in enumerate(HORIZONS):
            ss = [42, 101, 202]
            a0 = np.mean([subject_mean(err("A0", s, hi), subj, allm, n_subj) for s in ss], axis=0)
            a7 = np.mean([subject_mean(err("A7", s, hi), subj, allm, n_subj) for s in ss], axis=0)
            add("P6", "100% of subjects (3 seeds; descriptive reference, not in Holm)", h, paired_test(a0, a7, in_holm=False))
    else:
        p6_note = "round6_p6_per_subject.csv not found; P6 statistics not computed."

    # ---- Holm across the whole file
    df = pd.DataFrame(tests)
    fam = {i: r.p_raw for i, r in df.iterrows() if r.in_holm and not np.isnan(r.p_raw)}
    holm = holm_bonferroni({str(i): p for i, p in fam.items()})
    df["p_holm"] = np.nan
    for i in fam:
        df.loc[i, "p_holm"] = holm[str(i)]["p_adjusted"]
    df["verdict"] = [verdict(r) if r.in_holm else ("descriptive" if r.analysis == "P6" else "n/a") for _, r in df.iterrows()]
    print(f"Holm family size: {len(fam)}")

    OUT.mkdir(parents=True, exist_ok=True)
    names = {"P1": "p1_horizon", "P2": "p2_postmeal", "P3": "p3_range", "P4": "p4_roc", "P5": "p5_plausibility",
             "P6": "p6_lowdata", "P7": "p7_seedvar"}
    for a, nm in names.items():
        sel = df[df.analysis == a]
        if len(sel):
            sel.to_csv(OUT / f"round6_{nm}.csv", index=False)
    df.to_csv(OUT / "round6_all_tests.csv", index=False)

    # P5 descriptive rates
    p5_desc = []
    for arm in ("A0", "A7"):
        for s in SEEDS:
            r = R[arm][s]
            p5_desc.append(dict(arm=arm, seed=s, frac_steep=float((r["slope"] > ROC_STEP_ABS).mean()),
                                frac_out_of_range=float(((r["cmin"] < 40) | (r["cmax"] > 400)).mean()),
                                max_slope=float(r["slope"].max()), min_traj=float(r["cmin"].min()),
                                max_traj=float(r["cmax"].max())))
    pd.DataFrame(p5_desc).to_csv(OUT / "round6_p5_per_model.csv", index=False)

    write_md(df, p1_trend, pd.DataFrame(p5_desc), p6_note, post, subj, n_subj, ids, len(fam))


def md_table(sub, cols, heads):
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for _, r in sub.iterrows():
        cells = []
        for c in cols:
            v = None if c == "ci" else r[c]
            if c in ("p_raw", "p_holm"):
                cells.append(fmt_p(v))
            elif c == "ci":
                cells.append(f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:+.3f}" if c == "gap" else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_md(df, p1_trend, p5_desc, p6_note, post, subj, n_subj, ids, fam_size):
    path = OUT / "round6_tables.md"
    cols = ["stratum", "horizon_min", "n_subjects", "n_windows", "a0_mean", "a7_mean", "gap", "ci", "p_raw", "p_holm", "verdict"]
    heads = ["stratum", "h (min)", "n subj", "n win", "A0", "A7", "gap A0-A7", "95% CI", "raw p", "Holm p", "verdict"]
    out = [f"# Auto-generated tables (Holm family size {fam_size})\n"]
    for a in ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]:
        sub = df[df.analysis == a].copy()
        if "n_windows" not in sub:
            sub["n_windows"] = np.nan
        sub["n_windows"] = sub["n_windows"].apply(lambda v: "" if pd.isna(v) else int(v))
        sub["horizon_min"] = sub["horizon_min"].apply(lambda v: "-" if v == -1 else int(v))
        out.append(f"## {a}\n\n" + md_table(sub, cols, heads) + "\n")
        if a == "P1":
            out.append("P1 trend detail: " + ", ".join(f"{k}={v:.4g}" for k, v in p1_trend.items()) + "\n")
        if a == "P6" and p6_note:
            out.append(p6_note + "\n")
    out.append("## P5 per-model rates\n\n" + p5_desc.round(5).to_csv(index=False) + "\n")
    out.append("Post-meal windows per subject: " + ", ".join(f"{ids[i]}:{int(post[subj==i].sum())}" for i in range(n_subj)) + "\n")
    path.write_text("\n".join(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
