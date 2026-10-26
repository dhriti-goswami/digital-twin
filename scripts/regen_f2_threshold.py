"""Regenerate Sec. III-G3 validation-selected F2 threshold for the q0.10 alarm @30 min.

Method: re-run artifacts/official/quantile checkpoint on the VALIDATION loader
(same technique as run_extra_baselines_round5.py); alarm = q0.10(30 min) < T.
Grid: T in 60.0..100.0 step 0.5 mg/dL. F2 = 5PR/(4P+R), pointwise over windows.
The paper does not say which target is used for selection, so two protocols are run:
  A) target G<70 fixed during selection (primary)
  B) target G<T (target moves with the threshold)
T* (argmax F2 on validation; ties -> lowest T) is then evaluated once on TEST with
targets G<70 and G<T*, plus the plain 70 mg/dL alarm (G<70).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_extra_baselines_round5 import load_fold_and_model, HORIZON_IDX_30MIN
from twin.data.dataset import build_dataset, build_loader
from twin.train.loop import predict_loader

GRID = np.arange(60.0, 100.0 + 1e-9, 0.5)


def metrics(y, q, alarm_thr, target_thr):
    a, p = q < alarm_thr, y < target_thr
    tp, fp = int((a & p).sum()), int((a & ~p).sum())
    fn, tn = int((~a & p).sum()), int((~a & ~p).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f2 = 5 * prec * sens / (4 * prec + sens) if tp else 0.0
    return dict(sensitivity=sens, specificity=spec, precision=prec, f2=f2, tp=tp, fp=fp, fn=fn, tn=tn, n=len(y))


def get_q10(split, corpus, fold, model, scaler, config, device):
    ds = build_dataset(fold, split, corpus, scaler, config)
    ev = predict_loader(model, build_loader(ds, config, shuffle=False), device)
    return ev["targets"][:, HORIZON_IDX_30MIN], ev["quantile_predictions"][:, 0, HORIZON_IDX_30MIN]


def main():
    config, corpus, fold, model, scaler, device = load_fold_and_model()
    yv, qv = get_q10("val", corpus, fold, model, scaler, config, device)
    yt, qt = get_q10("test", corpus, fold, model, scaler, config, device)
    print(f"val n={len(yv)}, test n={len(yt)}")
    rows = []
    for proto in ("A_target_G<70", "B_target_G<T"):
        f2s = [metrics(yv, qv, T, 70.0 if proto[0] == "A" else T)["f2"] for T in GRID]
        Ts = GRID[int(np.argmax(f2s))]
        print(f"{proto}: T*={Ts} val F2={max(f2s):.4f}")
        rows.append(dict(protocol=proto, split="val", alarm="q10<T*", T=Ts, target="G<70" if proto[0] == "A" else "G<T*",
                         **metrics(yv, qv, Ts, 70.0 if proto[0] == "A" else Ts)))
        for tgt_name, tgt in (("G<70", 70.0), ("G<T*", Ts)):
            rows.append(dict(protocol=proto, split="test", alarm="q10<T*", T=Ts, target=tgt_name, **metrics(yt, qt, Ts, tgt)))
        rows.append(dict(protocol=proto, split="test", alarm="q10<70", T=70.0, target="G<70", **metrics(yt, qt, 70.0, 70.0)))
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/f2_threshold_regen.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
