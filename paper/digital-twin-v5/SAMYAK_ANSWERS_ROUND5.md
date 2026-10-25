# Samyak's answers — TODO_ROUND5.md S-15, S-11, S-16, S-17

## S-15. A0 fair selection window — physics gain shrinks ~3x; hypoglycemia-alarm advantage reverses

**Root cause**: `scripts/run_seeds_a0_a7.py` left A0 at `configs/official-small.yaml`'s
defaults (`param_warmup_epochs=10` -> selection epoch 11), while A0 seed 42 (reused from
`artifacts/official/abl-A0`) actually used `param_warmup_epochs=2, ramp_start_epoch=2,
ramp_end_epoch=6` (selection epoch 3, confirmed from the checkpoint's embedded config) and
every A7 seed selected from epoch 1. Fixed by giving A0 seeds 101-404 the same curriculum as
seed 42, then retraining (~45s/run, 3-18 epochs each).

**New Table VII (5-seed mean ± SD)**:

| horizon | A0 (fair) | A7 |
|---|---|---|
| 30 min | 13.543 ± 0.097 | 13.370 ± 0.114 |
| 60 min | 23.007 ± 0.160 | 22.599 ± 0.320 |
| 90 min | 30.006 ± 0.152 | 29.283 ± 0.527 |
| 120 min | 35.073 ± 0.101 | 34.080 ± 0.637 |

The physics MAE gain at 30 min shrinks from **0.51 to 0.17 mg/dL** (A7 unchanged; only A0
moved) — about a third of the original headline gain was the unfair selection handicap, not
physics. Paired Wilcoxon (Holm-corrected) on seed-averaged per-subject error: MAE difference
stays significant at all four horizons (p_holm 0.0098/0.0098/0.0073/0.0059); RMSE is now
significant only at 90/120 min (p_holm 0.027/0.028), not 30/60 min.

**Bootstrap CIs** (`results/tables/physics_effect_cis.csv`, subject-level, 5000 resamples):
30 min −0.174 [−0.304, −0.055]; 60 min −0.408 [−0.665, −0.180]; 90 min −0.723 [−1.113, −0.369];
120 min −0.993 [−1.493, −0.533] — still bootstrap-significant at every horizon (none cross
zero), but roughly a third the magnitude of round 4's values (which used the handicapped A0).

**Quantile-head A0 (Table IX) — retrained the same way, and the result reverses**: with fair
selection (`artifacts/official/quantile-A0-fair`, best epoch 5), A0's hypoglycemia event
sensitivity is **0.786**, now *beating* A7's 0.679 (`results/tables/hypoglycemia_a0_vs_a7.csv`).
Pointwise median-forecast sensitivity also edges ahead: A0 0.563 vs A7 0.551. Subject-level
bootstrap on the event-sensitivity difference: A7−A0 95% CI **[−0.220, −0.041]** — entirely
negative, i.e. A7 is now significantly *worse* than A0 at hypoglycemia event detection, the
opposite of what Table IX currently claims. (A0's false-alarm rate is a bit higher: 2.53 vs
2.21/day, specificity 0.943 vs 0.951 — a real trade-off, but the sensitivity story flips.)

**This needs honest treatment, not quiet correction**: the original quantile-A0 run's 8-epoch
selection handicap was propping up both the headline MAE gain and the entire hypoglycemia-alarm
argument for physics guidance. With a fair comparison, the MAE gain is real but much smaller
(0.17-0.99 mg/dL depending on horizon, still significant), and the hypoglycemia-detection
advantage disappears — A0 is now the better alarm. Recommend: update Table VII, the
Wilcoxon/Holm table, Fig. 4, the abstract/Discussion "0.51-2.38 mg/dL" and "4%/7%" sentences,
Table II's outcome column, and Table IX with these new numbers; the hypoglycemia paragraph
(Sec. III-G1 / Discussion / Conclusion) needs to drop or substantially qualify the "physics
improves hypoglycemia alarm sensitivity" claim.

Files: `scripts/run_seeds_a0_a7.py` (fix), `scripts/retrain_quantile_a0_fair.py`,
`scripts/recompute_hypo_a0_vs_a7_fair.py` (new), `results/tables/multiseed_a0_vs_a7.csv`,
`multiseed_wilcoxon_tests.csv`, `physics_effect_cis.csv`, `hypoglycemia_a0_vs_a7.csv`
(regenerated, copied to `paper/digital-twin-v5/data/`). Old A0 seed checkpoints moved to
`artifacts/seeds_old_round5/`, not deleted.

## S-11. Hypoglycemia alarm baseline — validation-tuned thresholds, finding confirmed unchanged

Round 4's baseline tuned decision thresholds on the TEST set to match the model's 2.21 FA/day
(test-set leak); the model's own threshold (70 mg/dL) was fixed in advance. Fixed version
(`scripts/run_extra_baselines_round5.py` → `results/tables/hypo_alarm_baseline_round5.csv`):
tune each baseline's threshold on the VALIDATION split to match the model's own q10<70 alarm's
false-alarm rate **on validation** (re-ran the checkpoint on the validation loader — predictions
were never cached). Model's validation alarm: 2.229 FA/day (sensitivity 63.6%, specificity
95.97%, 28/44 events, 59.7 patient-days).

| baseline | threshold (val-tuned) | test sensitivity | test specificity | test FA/day | test median lead time |
|---|---|---|---|---|---|
| Persistence alarm | 90.5 mg/dL | **67.9% (38/56)** | 92.0% | 2.206 | **17.5 min** |
| Linear-extrapolation alarm | 63.0 mg/dL | 44.6% (25/56) | 97.4% | 1.956 | 15.0 min |

**Persistence alarm numbers are identical to round 4's leaked ones** (same 90.5 mg/dL
threshold, same 67.9%/92.0%/17.5min) — test-set tuning wasn't driving that result; it's a
stable property of the data. The round-4 flag stands: persistence matches the quantile-head
model's event sensitivity exactly (67.9% both) at a comparable false-alarm rate, with a
*longer* lead time (17.5 vs model's 15.0 min). Confirmed honest with validation-only tuning.
Linear-extrapolation alarm moved down (44.6% vs round 4's leaked 62.5%) — that one *was*
inflated by the leak.

**Recommend for Sec. III-G1 / Discussion**: state plainly that a trivial persistence alarm,
honestly tuned on validation, reaches the same event sensitivity as the model's quantile alarm
at this operating point (with longer lead time) — do not soften this, both leak-free and
leaked versions agree. (Note: with S-15's A0 retrain, A0 now beats both A7 and the persistence
baseline on this metric — see above.)

## S-16. Table VIII (PRED-EGA) regenerated — new script, means confirmed, most SDs were stale/wrong

New script `scripts/compute_pred_ega_table.py` computes the full table from
`artifacts/official/quantile/official/test/quantile/predictions.npz`, aggregating
per-subject (12 subjects), for *every* category including the previously-missing Hypo/Hyper
Accurate and Erroneous SDs. Output: `results/tables/pred_ega_summary.csv` (copied to
`paper/digital-twin-v5/data/`).

All means are unchanged — they already matched. Most SDs the paper reported were stale: the
Benign-row and Erroneous/Acceptable-row SDs at 60/90/120 min don't match a direct per-subject
recomputation, and the Hypoglycemia-Erroneous "SD" values in the paper (0.13/0.24/1.59) turn
out to be the *mean* column copy-pasted in by mistake (they equal the per-subject means
0.1366/0.1377/0.8815 almost exactly, not any SD).

Corrected Table VIII (bold = changed from the current paper value):

| Metric | 30 min | 60 min | 90 min | 120 min |
|---|---|---|---|---|
| Overall Accurate (%) | 90.8 ± 3.3 | 77.8 ± 7.0 | 68.6 ± 8.1 | 61.9 ± 8.3 |
| Overall Benign (%) | 9.2 ± 3.3 | 21.9 ± **6.8** | 30.9 ± **7.8** | 37.5 ± **8.1** |
| Overall Erroneous (%) | 0.04 ± 0.07 | 0.28 ± **0.34** | 0.50 ± **0.65** | 0.60 ± **0.63** |
| **Clinically Acceptable (%)** | **99.96 ± 0.07** | **99.72 ± 0.34** | **99.50 ± 0.65** | **99.40 ± 0.63** |
| Hypoglycemia Accurate (%) | 65.9 ± 28.4 | 33.8 ± 25.5 | 18.3 ± 21.3 | 10.9 ± 16.2 |
| Hypoglycemia Erroneous (%) | 0.00 ± 0.00 | 0.14 ± **0.47** | 0.14 ± **0.32** | 0.88 ± **1.61** |
| Hyperglycemia Accurate (%) | 96.2 ± 2.0 | 86.1 ± 6.0 | 75.6 ± 8.6 | 66.1 ± 10.4 |
| Hyperglycemia Erroneous (%) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

File now lives at `results/tables/pred_ega_summary.csv` and
`paper/digital-twin-v5/data/pred_ega_summary.csv`; generator is
`scripts/compute_pred_ega_table.py`.

## S-17. Numbers with no surviving generator script — one confirmed drift, three undocumented

**1. Sec. III-G3 threshold analysis (T*=84.0, F2=0.771, etc.):** No script in `scripts/` or
`twin/` computes an F2-maximizing threshold search or produces these exact numbers (3,397/1,257
false positives, 0.988/0.868/0.175). Not reproducible from current repo state — no generator
found, nothing deleted in git history either. Needs a new script to regenerate from scratch —
flagging as unreproducible, not regenerating blind (out of scope this pass).

**2. Table IX A0 event-level specificity (99.4%, 95.9%):** Already appear verbatim in
`main.tex` lines 1025–1026 (A0 median-forecast specificity 99.4%, A0 q=0.10-alarm specificity
95.9%) — self-consistent within Table IX itself, not sourced from an external file. No action
needed unless S-15's A0 retrain changes these — it does: see S-15 above, A0-fair's specificity
is now 0.943 (median) and needs updating alongside the sensitivity numbers.

**3. Sec. III-H partial correlations for whole-study mean glucose (r=−0.843, p=0.0011;
r=−0.875, p=0.0004):** No `partial_corr`/`pcorr` code anywhere in the repo — only plain
`spearmanr` exists (`twin/eval/sensitivity.py:171`). No generator script found. Unreproducible
from current repo state; flagging, not regenerating blind.

**4. IG–permutation agreement — drift explained, confirmed current value:** Recomputed directly
from `results/attribution/integrated_gradients_per_feature_30min.csv` and
`permutation_importance_30min.csv` (35 features, inner join): Spearman ρ = 0.7397
(p = 3.83e-7) on mean_abs_attribution vs. mae_increase. The paper's in-text 0.731/6.05e-7 is
stale — written against an earlier version of these two CSVs that was later regenerated (no
code bug, single deterministic computation, no non-determinism). **Fix: replace ρ=0.731
(p=6.05e-7) with ρ=0.740 (p=3.83e-7) in main.tex line 1223.**
