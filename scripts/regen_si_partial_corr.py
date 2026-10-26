"""Regenerate Sec. III-H partial correlations of per-subject S_I vs TDD.

Method (as in main.tex): per-subject S_I = median over that subject's test-window
S_I estimates (twin.eval.sensitivity.validate_sensitivity); TDD = total daily dose
(basal+bolus U / record days, twin.eval.sensitivity.therapy_summary, whole record).
Partial correlation of S_I and TDD given z = mean glucose: residualise both on z
(OLS with intercept), Pearson of residuals; equals the standard first-order formula
(r_xy - r_xz r_yz)/sqrt((1-r_xz^2)(1-r_yz^2)). Partial rank: same on ranks.
p-value: t = r sqrt((n-3)/(1-r^2)), n-3 df, two-sided. n=12.
Mean glucose: (a) whole-study = pooled observed CGM readings (glucose_mg_dl, NaN
dropped) over train+test files; (b) test-interval = same over the test file only.
S_I source: test_diagnostics.npz of each model (point A7 = abl-A7 primary, plus the
quantile A7 for reference); nothing is tuned.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin.config import Config
from twin.data.dataset import build_dataset, load_corpus, FittedScaler
from twin.data.splits import official_split
from twin.eval.sensitivity import therapy_summary
from twin.train.loop import load_checkpoint

MODELS = {"A7_point(abl-A7)": "artifacts/official/abl-A7", "A7_quantile": "artifacts/official/quantile"}


def partial_corr(x, y, z):
    def resid(a):
        A = np.column_stack([np.ones_like(z), z])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    r = float(np.corrcoef(resid(x), resid(y))[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), n - 3))


def main():
    rows = []
    for name, root in MODELS.items():
        config = Config.from_yaml(f"{root}/resolved_config.yaml")
        state = load_checkpoint(f"{root}/official/best_model.pt")
        scaler = FittedScaler.from_state_dict(state["scaler"])
        corpus = load_corpus(config)
        fold = official_split(
            [v.windows for v in corpus["train"].values()], [v.windows for v in corpus["test"].values()],
            val_fraction=config.split.val_fraction, purge_steps=config.split.purge_steps)
        ds = build_dataset(fold, "test", corpus, scaler, config)
        d = np.load(f"{root}/official/test_diagnostics.npz")
        si, idx = d["insulin_sensitivity"], d["subject_index"]
        subs = list(ds.subject_ids)
        df = []
        for i, s in enumerate(subs):
            v = si[idx == i]; v = v[np.isfinite(v)]
            ts = therapy_summary(corpus["test"][s]) if s in corpus["test"] else therapy_summary(corpus["train"][s])
            # TDD over the subject's whole record is what validate_sensitivity uses
            tg = corpus["test"][s].frame["glucose_mg_dl"].to_numpy(float)
            trg = corpus["train"][s].frame["glucose_mg_dl"].to_numpy(float)
            df.append(dict(subject=s, si=float(np.median(v)), tdd=ts.total_daily_dose_u,
                           g_test=np.nanmean(tg), g_all=np.nanmean(np.concatenate([tg, trg]))))
        df = pd.DataFrame(df); n = len(df)
        rho, prho = stats.spearmanr(df.si, df.tdd)
        pr, pp = stats.pearsonr(df.si, df.tdd)
        rows.append(dict(model=name, control="none", n=n, rank_r=rho, rank_p=prho, pearson_r=pr, pearson_p=pp))
        for ctl, col in [("test_interval_mean_glucose", "g_test"), ("whole_study_mean_glucose", "g_all")]:
            rr, rp = partial_corr(stats.rankdata(df.si), stats.rankdata(df.tdd), stats.rankdata(df[col]))
            qr, qp = partial_corr(df.si.to_numpy(), df.tdd.to_numpy(), df[col].to_numpy())
            rows.append(dict(model=name, control=ctl, n=n, rank_r=rr, rank_p=rp, pearson_r=qr, pearson_p=qp))
        print(name); print(df.to_string())
    out = pd.DataFrame(rows)
    out.to_csv("results/tables/si_partial_corr_regen.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
