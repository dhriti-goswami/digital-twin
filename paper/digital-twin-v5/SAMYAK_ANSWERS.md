# Samyak's answers to todo.md S1-S15

Full detail, exact main.tex line pointers, and sentence-ready wording for every D-item is in
`paper/digital-twin-v4/DHRITI_REVISION_V2.md` (main.tex still lives in v4, not yet copied to v5).
This file is just the direct S1-S15 answers, numbered to match `todo.md`.

- **S1.** 135,000 is the whole-cohort retained 5-min slot count (not a window count); 141,100 is the
  retained *window* count, each with its own anchor slot — different units, both correct. The
  hypoglycemia analysis (56 events) rests on the test set only: 26,498 windows → n_eff ≈552 naive,
  ≈494 autocorrelation-adjusted (τ=53.6). See DHRITI_REVISION_V2.md item 1.

- **S2.** Mean 16.38 epochs/run, mean 20.43 s/epoch, mean 5.47 min/run (not 23s/4.5min).
  `scripts/summarize_training_time.py`, `results/tables/training_time_summary.csv`. Item 3.

- **S3.** Not separately trained. Gaussian left-tail extrapolation anchored on trained q10/q50:
  σ_L=(q50-q10)/(-Φ⁻¹(0.10)), q_p=q50+σ_L·Φ⁻¹(p). Verified to reproduce existing numbers to float
  rounding. `scripts/compute_quantile_extrapolation.py`. Item 4.

- **S4.** CRPS ≈ 0.5·pinball_q10 + pinball_q50 + 0.5·pinball_q90 (exact fit to existing committed
  values, residual <1e-6). `scripts/compute_crps.py`. Item 5.

- **S5.** Not test-set cherry-picking — three different trained artifacts (final quantile-head model
  13.08; ablation-A7-seed42 13.25; ablation-A7 5-seed-mean 13.37). Recommend leading with 5-seed
  mean ± SD for ablation comparisons, final-model number only for the deployed system. Item 6.

- **S6.** Final model's own Clarke A (zone A alone) at 30 min = **89.86%** (23,811/26,498), from the
  final model's own output, not the ablation table (coincidentally ≈ablation's 89.8%). Item 8.

- **S7.** Keep λ_phys, L_phys, λ_prior (drop w_phys, L_res, λ_pr) — pure terminology unification,
  no numeric change. Item 9.

- **S8.** Confirmed: commit `3baf636fce11ed5780d3e79aeee6ed8b4a5e02d4`, dated **2026-07-25 17:49:43
  +0530**. Already correctly cited at main.tex:455. Use "pre-specified" everywhere (no public
  registry exists). Item 11.

- **S9.** Fixed 70 kg population-average constant for every subject (`NOMINAL_WEIGHT_KG=70.0` in
  `twin/data/ohio.py:56`), used uniformly — OhioT1DM has no real per-subject weight field. Item 13.

- **S10.** Window-level SI per subject split into 4 disjoint time-ordered blocks (`np.array_split`),
  each collapsed to its median; ICC(1) computed across subjects on the 4-point block-median vectors
  (`twin/eval/sensitivity.py:278-308`). TDD = total recorded insulin delivery / days recorded
  (U/day) (`twin/eval/sensitivity.py:84-112`); `tdd_per_kg` divides by the same 70kg constant. Item 13.

- **S11.** Holm family size = 4 (one per horizon: 30/60/90/120 min), applied separately for MAE and
  RMSE. A7 was selected on validation data, not test. Item 15.

- **S12.** Checked in code (`twin/train/ablations.py:126-144`): A3 and A4 differ in exactly ONE
  override, `model.per_patient_params` (True/False) — physics weighting, physics.enabled, and
  hybrid_residual are identical. This is a single-factor ablation, not confounded — the reviewer's
  premise is incorrect. No isolating run needed; it's already isolated. Item 15.

- **S13.** Done. Ridge on full 35-feature set and ARX(6)+IOB/COB, same 26,498-window pipeline.
  Both land between ARX(6) and A7 at every horizon (30min MAE: ARX(6) 13.97 > Ridge 13.56 ≈
  ARX+IOB/COB 13.51 > A7 13.25) — most of A7's gain over non-mechanistic neural baselines (LSTM/
  Transformer) is from feature richness, not architecture; only ~0.26-0.31 mg/dL MAE gap remains
  for A7 at 30min. `scripts/run_extra_baselines.py`, `results/tables/extra_baselines.csv`. Item 17.

- **S14.** Subject-level (12-cluster) bootstrap, 5000 resamples: A0 95% CI [0.4528, 0.7258], A7 95%
  CI [0.5854, 0.7500], paired A7-A0 difference 95% CI **[-0.0500, 0.2353]** (crosses zero — not
  significant). Say "numerically higher," not "better." `scripts/bootstrap_hypo_sensitivity.py`.
  Item 18.

- **S15.** Table VI (ablation-A7, no quantile head) and Table IX (final quantile-head model) are
  different architectures/runs — not a contradiction, needs a cross-reference sentence. A0's
  0.637→0.413 swing when a quantile head is added is real run-to-run/architecture variance, flag it
  as a caveat on single-run A0 claims. Item 18.
