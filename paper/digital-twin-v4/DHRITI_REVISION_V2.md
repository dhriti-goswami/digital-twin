# Revision round 2 — for Dhriti (text/literature only, Monday deadline)

Samyak did all code/numbers verification below. Every number here is machine-verified against
scripts in `scripts/` and `artifacts/official/`. Do NOT touch any code/scripts — just edit
`main.tex` (and `supplementary.tex` where noted). Nothing in this file requires rerunning anything.

New scripts added this round (reference only, already run, outputs saved):
`scripts/summarize_training_time.py`, `scripts/compute_quantile_extrapolation.py`,
`scripts/compute_crps.py`, `scripts/bootstrap_hypo_sensitivity.py`, `scripts/run_extra_baselines.py`,
plus an addition to `scripts/compute_autocorrelation.py` (test-set n_eff block).

---

## 1. Test-set n_eff (vs cohort-wide ~2,800)

`main.tex:146-151` (Table III accounting) and `main.tex:1330` (Limitations, ~2,800 n_eff) describe
the **whole cohort** (135,000 slots / 48 ≈ 2,800). The hypoglycemia analysis (56 events) actually
rests on the **test set only**: 26,498 test windows.

Add near the hypoglycemia analysis (wherever the 56-event number is introduced, e.g. Sec III-F/G):
> "The test-set effective sample size is ≈552 ($26{,}498/48$), or ≈490 using the empirical
> autocorrelation time ($26{,}498/53.6$)."

Exact verified numbers: naive 552, autocorrelation-adjusted 494 (script: `scripts/compute_autocorrelation.py`, new test-set block).

## 2. PRED-EGA wording — "at most" / "at least"

`main.tex` (wherever "stayed below 0.6%" and "exceeded 99.4%" appear, Sec III-?/Table VIII prose):
Table VIII gives exactly 0.60% (Erroneous) and 99.40% (Acceptable) at 120 min — those ARE the
boundary values, not values "below"/"exceeding" them. Change wording to:
- "Erroneous classifications remained **at most** 0.6% across all horizons"
- "Acceptable classifications remained **at least** 99.4% across all horizons"

## 3. Training time / epochs

`main.tex:465` currently says "about 23 s per epoch (about 4.5 min per run)". **Wrong on both counts.**
Measured directly from all 8 `training_history.csv` runs under `artifacts/official/`:
- Mean epochs per run: **16.38** (patience=12 per `configs/official-small.yaml` means ≥13 epochs is a floor, not the actual mean — actual mean is higher)
- Mean wall-clock: **20.43 s/epoch**
- Mean time per run: **5.47 min** (not 4.5 min)

Replace the sentence with: "training required a mean of 16.4 epochs per run (early-stopping
patience of 12, so a floor of 13 epochs), at a mean 20.4 s/epoch, i.e. ≈5.5 min per run."
(Source: `results/tables/training_time_summary.csv`.)

## 4. Quantile levels q∈{0.05,0.10,0.20} vs Eq.16's q∈{0.1,0.5,0.9}

Sec III-G3 uses q=0.05/0.10/0.20 but the model is only trained on q=0.1/0.5/0.9 (Eq.16). These
extra levels are **not** separately trained models — they're a parametric left-tail extrapolation:
treat the lower half of the predictive distribution as Gaussian, anchor scale on the trained
q10/q50 gap, then extrapolate:
$$\sigma_L = \frac{q_{50}-q_{10}}{-\Phi^{-1}(0.10)}, \qquad q_p = q_{50} + \sigma_L\,\Phi^{-1}(p)\ \ (p\in\{0.05,0.20\})$$
Verified to exactly reproduce the already-committed `sensitivity_precision_curves.csv` (max abs
error 0.002 mg/dL, i.e. float rounding only). Add a sentence in III-G3 stating this explicitly —
this is an approximation under a Gaussian left-tail assumption, not an independently trained
quantile. Cite `scripts/compute_quantile_extrapolation.py`.

## 5. CRPS — how computed, and upper-tail under-coverage

State explicitly that with only 3 trained quantiles, true CRPS can't be computed exactly; we use
the quantile/pinball-based approximation restricted to the q10–q90 bracket:
$$\mathrm{CRPS} \approx 0.5\cdot\mathrm{pinball}_{0.1} + \mathrm{pinball}_{0.5} + 0.5\cdot\mathrm{pinball}_{0.9}$$
This exactly reproduces the committed `calibration_by_horizon.csv` CRPS column (residual <1e-6).
Values: 9.74 / 16.18 / 20.85 / 24.32 mg/dL at 30/60/90/120 min.

Also add: the upper q90 quantile is **under-covered** relative to its nominal 90% level at every
horizon — empirical coverage 88.88% / 88.12% / 87.38% / 86.89% (30→120 min) — whereas the lower
q10 tail is roughly calibrated (miss rates 9.47–10.22%, close to nominal 10%). Currently the text
only discusses the lower (hypoglycemia-relevant) tail — add a sentence noting the upper-tail
under-coverage too, since it affects hyperglycemia-alarm calibration symmetrically.

## 6. Three "A7" numbers at 30 min (13.08 / 13.25 / 13.37)

Not an error, but needs explaining — these are three *different trained artifacts*, not
cherry-picking:
- **13.08 MAE** = the actual final model (quantile-head architecture), Table IV
- **13.25 MAE** = seed-42 single run of the ablation-A7 architecture (no quantile head), Table V/VI
- **13.37 MAE** = 5-seed mean of the same ablation-A7 architecture, Table VII

Add a sentence (e.g. in the Table IV caption or just before it) stating plainly that the "final
model" (quantile-head) and the "A7" ablation arm are architecturally distinct runs, and that
headline numbers should lead with the 5-seed mean ± SD (13.37 mg/dL) for the ablation comparison,
reserving the single final-model number (13.08) only for describing the deployed/reported system.
No new number to compute here — just state the provenance clearly, once, and be consistent
everywhere else in the text.

## 7. Abstract LOSO claim — add horizon

"All 12 still beat persistence by MAE" is true only at 30 min; by RMSE it's 10/12. Reword to:
"At the 30-minute horizon, all 12 held-out subjects beat persistence by MAE (10/12 by RMSE)."
Apply the same horizon qualifier anywhere else this claim is repeated (check Table XI/limitations).

## 8. Clarke A mixup (89.8% paired with final-model 0.551 sensitivity)

`main.tex:924` wrongly pairs Table VI's ablation-A7 Clarke A (89.8%) with the final model's hypo
sensitivity (0.551, from Table IX). **Verified final model's own Clarke-A-alone value at 30 min:
89.86%** (23,811/26,498 windows; computed directly from `artifacts/official/quantile/figures/clarke_quantile_30min.csv`,
the final model's own output — not the ablation table). Essentially identical to 89.8% by
coincidence, but now correctly sourced. Replace main.tex:924's citation/value with 89.86% sourced
to the final model, and use final-model numbers (Clarke A, A+B=99.0%, sensitivity 0.551) together,
consistently, throughout III-F — never mix a Table VI ablation number with a Table IX final-model number again.

## 9. Eq. 17 vs Eq. 22 symbol harmonization

Pick one symbol per concept and use it in both equations and surrounding text:
- λ_phys (not w_phys)
- L_phys (not L_res)
- λ_prior (not λ_pr)
Mechanical find-and-replace across `main.tex` — no numbers change.

## 10. Ref [8] volume mismatch

Ref [8] is cited as 2024 but the reference entry says "vol. 20" — check the actual publication
year/volume on the journal site and correct whichever field (year or volume) is wrong in the
bibliography entry.

## 11. "Pre-specified" vs "preregistered"

Abstract says "pre-specified"; body, Table XI, and Limitations say "preregistered". There is no
timestamped public registration — only a git commit. **Use "pre-specified" everywhere**, and cite
the exact commit: `3baf636fce11ed5780d3e79aeee6ed8b4a5e02d4`, dated **2026-07-25 17:49:43 +0530**
(already correctly cited at `main.tex:455` — just make sure every OTHER "preregistered" instance
is changed to "pre-specified" and points to this same commit/date, not a registry).

## 12. Title wording — "Validated"/"Audited"

"Validated" insulin sensitivity conflicts with failed LOSO validity, no clamp data, and only
indirect checks (concurrent TDD consistency, not external validity — see item 13). "Audited" can
read as an independent third-party audit, which this isn't. Suggested replacements per Dr.
Yugnanda: **"Stable, Dose-Consistent"** or **"Integrity-Checked"** in the title.

## 13. SI vs TDD — "partly", not fully, explained by temporal-consistency loss

The parameter head sees insulin inputs and TDD comes from the same records — so the SI-vs-TDD
agreement is **concurrent consistency**, not external/independent validity. Without the temporal-
consistency loss: ICC drops 0.89→**0.645**, CV changes 35.8%→**22.2%** (both numbers verified
exactly against `artifacts/official/abl-A7-notc/official/sensitivity_report_notc.json`).
Note the direction carefully: ICC gets *worse* (less within-subject-stable) without the loss, but
CV gets *numerically smaller* (more homogeneous across subjects) without it — so the loss trades
some between-subject spread for better within-subject stability. The current text (`main.tex:1162`)
already says "not produced solely by the penalty" — just tighten the wording to "**partly**
attributable to the temporal-consistency penalty" to match the reviewer's exact phrasing.

**Missing methods detail to add** (Sec III or Methods, SI/ICC paragraph):
- **Body weight**: VG/VI (per kg) and Ra (needs weight) use a **fixed 70 kg population-average
  constant for every subject** (`NOMINAL_WEIGHT_KG=70.0`, used uniformly — OhioT1DM has no real
  per-subject body weight field). State this explicitly as a limitation/assumption — don't leave
  weight as merely "unavailable"; say what was substituted.
- **Window→subject SI aggregation**: each subject's window-level SI time series is split into
  **4 disjoint, time-ordered blocks** (`np.array_split`), each block collapsed to its median,
  giving a 4-point block-median vector per subject; ICC(1) is computed across subjects on those
  block medians.
- **TDD computation**: total daily dose = sum of recorded insulin delivery over a subject's
  recorded days, divided by number of days (U/day); `tdd_per_kg` divides that by the same 70 kg constant.

## 14. Comparator selection (Table I vs Table XIII)

Table I in the current draft only lists weaker 2020-era comparators, making this work look best;
Table XIII (presumably a supplementary/appendix table) has stronger, more recent comparators where
this work is mid-field. **Harmonize**: either merge the strong comparators from Table XIII into
Table I, or explicitly note in Table I's caption that stronger recent baselines are deferred to
Table XIII and give the reader a pointer. Also add or explicitly explain the omission of
physics-informed OhioT1DM works [11] and [68] — if they're not included because of methodology
mismatch, state why; otherwise add them to the comparison.

## 15. Ablation confound claim (A3 vs A4) — pushback, not a fix

Reviewer says A3 vs A4 mixes per-patient parameters with the curriculum — **verified this is NOT
actually true in the code**: `twin/train/ablations.py:126-144` shows A3 and A4 differ in exactly
ONE override (`model.per_patient_params`: True vs False); `physics.enabled`, `physics.weighting`,
and `model.hybrid_residual` are identical between them. This is a clean single-factor ablation.
Write a short clarifying footnote/sentence in the ablation table caption citing this (you can just
state "A3 and A4 differ only in per-patient parameterization; physics weighting and curriculum
settings are held fixed" — no need to cite code line numbers in the paper itself).

Also state explicitly (already true, just needs saying): the Holm correction family size is **4**
(one per horizon — 30/60/90/120 min), applied separately for MAE and RMSE; and A7 was **selected
on validation data**, not test data — state this plainly wherever A7 is first singled out as the
chosen architecture.

## 16. PRED-EGA hypoglycemia accuracy near "99.96% acceptable" claim

Add, right next to wherever "99.96% acceptable" is stated: PRED-EGA **Accurate** classification
(a stricter sub-category than "Acceptable") for hypoglycemic-range readings specifically falls to
**65.9% at 30 min and 10.9% at 120 min** — the 99.96% acceptable figure is reassuring at an
aggregate level but masks substantially worse accurate-classification performance in the
clinically critical hypoglycemic range at longer horizons.

## 17. Baselines — AR(6) beats non-mechanistic LSTM/Transformer; add richer linear baselines

Reviewer is right that "competitive" undersells AR(6) (14.39/20.11 MAE/RMSE at 30min beats LSTM
15.41/21.46 and Transformer-A0-nomech 14.96/21.08 at all 4 horizons). **New baselines added and run**
on the identical 26,498-window pipeline (`scripts/run_extra_baselines.py`,
`results/tables/extra_baselines.csv`):

| model | 30 min MAE/RMSE | 60 min | 90 min | 120 min |
|---|---|---|---|---|
| Ridge, full 35-feature set | 13.56±1.55 / 19.10±2.60 | 23.63±3.07 / 31.80±3.81 | 31.11±3.89 / 40.94±4.65 | 36.47±4.18 / 47.04±5.17 |
| ARX(6) + IOB/COB | 13.51±1.50 / 19.09±2.69 | 23.71±2.86 / 31.86±3.77 | 31.39±3.51 / 41.04±4.25 | 36.84±3.74 / 47.22±4.50 |

Both land between ARX(6) (13.97) and A7 (13.25) at 30 min — i.e. **most of A7's gain over the
non-mechanistic neural baselines comes from richer input features** (mechanistic state variables
used even as plain linear regressors close most of the gap), **not** from the neural architecture
or hybrid-physics training per se; only ~0.26–0.31 mg/dL MAE gap remains between the best linear
baseline and A7 at 30 min, and adding IOB/COB explicitly barely helps beyond the full feature set
(13.51 vs 13.56) since the mechanistic states already carry that information.

**Rewrite the baseline-comparison paragraph** to (a) drop "competitive" in favor of a precise
statement that AR(6)/ARX(6) beat the non-mechanistic neural baselines at every horizon, (b) add the
two new rows to Table V/Table I, (c) state the above conclusion about feature-richness vs
architecture explaining most of the gain, with the small residual (~0.3 mg/dL) attributed to the
hybrid-physics training itself.

## 18. Hypoglycemia sensitivity, Table VI vs Table IX + bootstrap CIs

Both Table VI (ablation A7 sensitivity 0.608 < A0's 0.637, worse hypo bias +8.61 vs +7.29) and
Table IX (final model A7 67.9% vs A0 58.9%, i.e. 38/56 vs 33/56 events) are each internally
consistent **but are two different runs/architectures** — already footnoted elsewhere in the
draft; make sure a cross-reference note ties them together explicitly so readers don't think it's
a contradiction (A7 looking worse in one table and better in the other), e.g.: "Table VI reports
the single-seed ablation architecture (no quantile head); Table IX reports the final quantile-head
model — the two are not directly comparable architectures."

Also flag A0's own instability: single-seed A0 sensitivity swings from 0.637 (ablation, no
quantile head) to 0.413 (A0 with quantile head added) — this is high run-to-run/architecture
variance and should be mentioned as a caveat on any single-run A0 sensitivity claim.

**Bootstrap CIs computed** (subject-level/cluster bootstrap over the 12 test subjects, 5,000
resamples, `scripts/bootstrap_hypo_sensitivity.py`):
- A0 sensitivity 95% CI: **[0.4528, 0.7258]** (point estimate 0.5893 = 33/56)
- A7 sensitivity 95% CI: **[0.5854, 0.7500]** (point estimate 0.6786 = 38/56)
- Paired A7−A0 difference 95% CI: **[−0.0500, 0.2353]** (mean +0.0848) — **crosses zero**

Replace any claim that A7 is "better" at hypoglycemia detection with: A7 is **numerically higher**
(67.9% vs 58.9%, a 5-event difference out of 56), but the subject-bootstrap 95% CI on the
difference crosses zero (**[−0.050, 0.235]**), so this is **not a statistically significant**
improvement. State the CIs for both arms and for the difference explicitly in the text/table.

---

## Summary table of what changed where (for your own tracking)

| # | Section | Action | New number to insert |
|---|---|---|---|
| 1 | Sec II/III, near 56-event hypo analysis | add sentence | n_eff(test) ≈552 / ≈494 |
| 2 | Wherever PRED-EGA wording appears | reword | "at most 0.6%", "at least 99.4%" |
| 3 | main.tex:465 | replace sentence | 16.4 epochs, 20.4 s/epoch, 5.5 min/run |
| 4 | Sec III-G3 | add explanatory sentence | Gaussian-extrapolation formula |
| 5 | Sec III-G3/calibration section | add sentences | CRPS formula + values; upper-tail 86.9–88.9% |
| 6 | Table IV caption / before it | add clarifying sentence | none (provenance only) |
| 7 | Abstract | reword | "at 30 min... (10/12 by RMSE)" |
| 8 | main.tex:924 | fix number+citation | 89.86% (final model, not ablation) |
| 9 | Eq. 17 & 22 + surrounding text | find/replace symbols | none |
| 10 | Bibliography, Ref [8] | fix vol/year | check source |
| 11 | Abstract/body/Table XI/Limitations | find/replace | "pre-specified" everywhere, cite commit 3baf636 (2026-07-25) |
| 12 | Title | reword | "Stable, Dose-Consistent" or "Integrity-Checked" |
| 13 | SI/ICC methods paragraph, main.tex:1162 | reword + add methods detail | "partly"; 70kg; 4 blocks; TDD method |
| 14 | Table I / Table XIII | harmonize | merge or cross-reference |
| 15 | Ablation table caption | add footnote | A3/A4 single-factor; Holm family=4; A7 selected on val |
| 16 | Near "99.96% acceptable" | add sentence | Accurate 65.9%@30min, 10.9%@120min |
| 17 | Baseline comparison paragraph + Table V | add 2 rows + reword | Ridge/ARX+IOB/COB numbers above |
| 18 | Hypoglycemia section, Table VI/IX | add cross-ref + CIs | bootstrap CIs above |

All source numbers are reproducible — if anyone (reviewer/ma'am) asks "where does this number come
from," point to the script named in each section above.
