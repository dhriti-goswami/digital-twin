# Samyak's answers — TODO_ROUND4.md S-1, S-4, S-9, S-10, S-11

## S-1. Training/selection details — fully resolved, no bug anywhere

Each run's `best_model.pt` embeds its own training-time config (saved by `torch.save` in
`twin/train/loop.py`, never touched by the `report`-stage manifest bug fixed two rounds ago) — this
is the authoritative source, not `manifest.json`. Verified across all 20 checkpoints:
**`stop_epoch = best_epoch + patience` exactly, every time**, once you use each run's own embedded
warmup/ramp_end rather than assuming the default curriculum.

| run | warmup | ramp_end | physics | patience | best_epoch | stop_epoch |
|---|---|---|---|---|---|---|
| A0_no_mech | 10 | 25 | off | 10 | 14 | 24 |
| LSTM | 10 | 25 | off | 10 | 11 | 21 |
| abl-A0 | 2 | 6 | off | 10 | 5 | 15 |
| abl-A1 | 2 | 6 | on | 10 | 10 | 20 |
| abl-A2 | 2 | 6 | on | 10 | 7 | 17 |
| abl-A3 | 2 | 6 | on | 10 | 7 | 17 |
| abl-A4 | 2 | 6 | on | 10 | 9 | 19 |
| abl-A7 | 0 | 0 | on | 10 | 5 | 15 |
| abl-A7-notc | 0 | 0 | on | 10 | 3 | 13 |
| quantile-A0 | 10 | 25 | off | 12 | 13 | 25 |
| **quantile (final)** | 0 | 0 | on | **8** | 3 | 11 |
| small-fast | 2 | 6 | on | 10 | 7 | 17 |
| A0 5-seed (101/202/303/404) | 10 | 25 | off | 10 | 13/11/11/11 | 23/21/21/21 |
| A7 5-seed (101/202/303/404) | 0 | 0 | on | 10 | 6/7/4/5 | 16/17/14/15 |

**Your reviewer's "A0/A1-A4 couldn't stop before epoch 21/36" rests on a wrong assumption**: they
applied the default curriculum (warmup=10, ramp_end=25) to the ablation arms, but A0-A4 actually
use a shorter, ablation-specific curriculum (warmup=2, ramp_end=6), confirmed directly in each
checkpoint. Under their real curriculum, selection starts at epoch 3 (A0) or 7 (A1-A4) — every stop
epoch matches `best_epoch + patience(10)` exactly. No unexplained gap, no bug.

**Quantile run's patience=8 is now certain** (directly in checkpoint config: `early_stopping_patience=8,
ramp_start_epoch=0, ramp_end_epoch=0, param_warmup_epochs=0, best_epoch=3`) — last round's arithmetic
inference is confirmed exactly, no longer needs hedging.

**Validation split / LOSO protocol** (`twin/data/splits.py`): time-ordered tail split per subject —
final 15% of each subject's chronologically-ordered windows become validation, with a purge gap of
`seq_len + max_horizon_steps` windows discarded immediately before the validation tail (prevents
window-overlap leakage). No purge needed at train/test boundary (physically separate recording
periods). In LOSO, each fold holds out one subject entirely; every remaining subject gets its own
purged time-ordered 15% validation tail (same mechanism); the held-out subject's data is only
touched for final test scoring.

## S-4. Table VI hypoglycemia values — bias confirmed exact, sensitivity method lost (again)

Row order confirmed A0/A1/A2/A3/A4/A7. The **hypo bias values reproduce exactly** (to 0.01 mg/dL)
from `artifacts/official/abl-{arm}/official/test/abl-{arm}/predictions.npz` via pooled pointwise
mean(pred-true) for true<70mg/dL — confirms correct data/model source. But **sensitivity** from the
same data/method is consistently ~0.013-0.017 below every paper value — tried per-subject mean,
threshold ≤70, episode-consolidation; none closed the gap. The original sensitivity definition is
unreproducible (same issue as the hypoglycemia numbers two rounds ago) — but at least now isolated
to sensitivity alone, not bias.

**New, fully-reproducible corrected values** (`scripts/recompute_table_vi_hypo.py`,
`results/tables/table_vi_hypo_corrected.csv`):

| arm | sensitivity | bias |
|---|---|---|
| A0 | 0.6209 | +7.2867 |
| A1 | 0.4876 | +11.2871 |
| A2 | 0.4863 | +11.0287 |
| A3 | 0.4712 | +10.9498 |
| A4 | 0.5041 | +10.1810 |
| A7 | 0.5907 | +8.6096 |

**Discussion's "1.3 mg/dL" claim**: recomputed (8.6096-7.2867=1.3229) — **still correctly rounds to
1.3 mg/dL, no change needed.**

## S-9. Physics effect uncertainty — significant at every horizon

Subject-level bootstrap (12 subjects, 5000 resamples) on 5-seed per-subject MAE
(`scripts/bootstrap_physics_effect.py`, `results/tables/physics_effect_cis.csv`):

| horizon | A7−A0 MAE diff | 95% CI |
|---|---|---|
| 30 min | −0.509 | [−0.696, −0.362] |
| 60 min | −1.184 | [−1.597, −0.873] |
| 90 min | −1.929 | [−2.560, −1.416] |
| 120 min | −2.380 | [−3.152, −1.690] |

None cross zero — physics effect is bootstrap-significant at every horizon, consistent with the
existing Wilcoxon/Holm results.

**A3/A4 5-seed replication ("Should"): not attempted.** No cached checkpoints exist for A3/A4 at
seeds 101-404 (only A0/A7 have 5-seed runs) — would require real new training, out of session scope.

## S-10. Low-glucose calibration — conformal recalibration tried, honest negative result

Reproduced your reviewer's exact numbers first (confirms correct data): 30min interval coverage
76.35% (<80mg/dL subset), 120min 53.26%, q10-above-truth 22.41% at 30min — all match to the decimal.

Implemented split-conformal recalibration (`scripts/conformal_recalibration.py`): per-horizon
additive correction on q10/q90, fit on validation, applied to test.

- **Marginal/global coverage fixes well**: 30min interval coverage 79.4%→82.3% overall, q90 coverage
  88.9%→91.2% — both near nominal.
- **Conditional coverage on the <80mg/dL subset barely moves and gets WORSE at longer horizons**:
  interval coverage 76.4%→78.1% (30min, marginal gain), 53.3%→51.9% (120min, actually worse);
  q10-above-truth 22.4%→21.2% (30min, marginal), 46.7%→48.1% (120min, worse).
- Hypoglycemia alarm (q10<70 @ 30min): sensitivity 91.8%→92.4%, precision 34.7%→33.9% — negligible.

**This is a genuine heteroscedastic miscalibration** — predictive uncertainty doesn't widen enough
specifically in the low-glucose regime, so a single global per-horizon shift can't fix it (it
corrects the marginal distribution, leaves the conditional shape broken). Recommended limitations
text:
> "We attempted split-conformal recalibration (per-horizon additive correction fit on the
> validation split); while this restores near-nominal marginal coverage, conditional coverage in
> the clinically critical <80 mg/dL range improves only marginally at short horizons and does not
> improve at longer horizons, indicating the miscalibration is heteroscedastic and would require
> quantile-regression recalibration conditional on glucose level (e.g. local/binned conformal
> calibration), which we leave to future work."

## S-11. Missing baselines — both run; one surprising/uncomfortable finding

Both on the identical 26,498-window test split (script asserts `n_test==26498` at runtime).

**Hypoglycemia alarm baseline** (`results/tables/hypo_alarm_baseline.csv`), both tuned to match the
paper's own ~2.21 false-alarms/day operating point:
- **Persistence alarm** (threshold 90.5 mg/dL): sensitivity **67.9% (38/56)**, specificity 92.0%,
  median lead time **17.5 min**.
- **Linear-extrapolation alarm** (threshold 65.5 mg/dL): sensitivity 62.5% (35/56), specificity
  97.0%, median lead time 5.0 min.

**Flagging honestly — this needs to go in the paper, not be hidden**: the trivial persistence alarm
reaches essentially the SAME event sensitivity as the quantile-head model (67.9% both) at the same
false-alarm rate, with a LONGER lead time (17.5 vs the model's reported 15.0 min). At this specific
operating point, the model's hypoglycemia-alarm advantage over doing nothing clever looks weak.
This should be stated plainly in the results/limitations, not glossed over — it's exactly the kind
of finding a careful reviewer would want reported.

**Probabilistic baseline** (linear quantile regression, 35 features, per-subject fits,
`results/tables/probabilistic_baseline.csv`), CRPS via the same formula as `scripts/compute_crps.py`:

| horizon | q10 cov% | q90 cov% | width | CRPS |
|---|---|---|---|---|
| 30 min | 9.24 | 89.94 | 43.80 | 8.54 |
| 60 min | 9.71 | 88.91 | 74.92 | 14.77 |
| 90 min | 9.88 | 88.37 | 98.34 | 19.45 |
| 120 min | 10.23 | 87.84 | 114.02 | 22.75 |

Compared to the quantile-head model's Table X CRPS (8.434/13.989/18.017/21.009): linear baseline is
close but consistently slightly worse at every horizon (+0.1 to +1.7 mg/dL CRPS), with wider
intervals — a legitimate, modest probabilistic-modeling gain for the neural model.

**Skipped ("Should", not Must)**: N-BEATS/TFT point-forecast baseline — needs real training
infrastructure, out of session scope.
