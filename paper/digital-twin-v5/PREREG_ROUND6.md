# Round 6 pre-registered analyses (written 2026-10-26, before any of these were run)

Goal: find out honestly where (if anywhere) physics guidance (A7) helps over the fair A0.
Every analysis below is reported in full regardless of direction. No analysis is added or
dropped after seeing results without being listed under "Deviations" with a reason.

Models: A7 vs fair A0 (5 seeds each: 42, 101, 202, 303, 404), same test split as Table VII.
Unit of inference: subject (seed-averaged per-subject error). Paired Wilcoxon + subject-level
bootstrap 95% CI (5000 resamples). Holm correction across all tests in this file together.

## Analyses

P1. Horizon trend: is the A0-A7 MAE gap increasing with horizon (30/60/90/120)? Spearman of
    per-subject gap vs horizon; report gap and relative gap (%) per horizon.
P2. Post-meal windows: MAE in 0-180 min after a carbohydrate input vs all other times, per
    horizon. Report gap in each stratum and the difference between strata.
P3. Glycemic range of the target: error in target <70, 70-180, >180 mg/dL, per horizon.
P4. Rate of change at prediction time: |ROC| < 1, 1-2, > 2 mg/dL/min strata.
P5. Physiological plausibility of forecasts: fraction of predicted trajectories with
    |dG/dt| > 4 mg/dL/min anywhere, and fraction predicting < 40 or > 400 mg/dL.
P6. Low-data regime: retrain A0 and A7 on 25% and 50% of training subjects (seeds 42/101/202),
    evaluate on the same test set. Report gap per fraction.
P7. Seed variance: compare across-seed SD of A0 vs A7 (stability claim).

P8. Cross-cohort transfer (added 2026-10-26 before running; no external real dataset in repo):
    train A0 and A7 (fair curricula, seeds 42/101/202/303/404) on OhioT1DM 2018 subjects only,
    test on 2020 test files; then the reverse. Per-subject seed-averaged MAE/RMSE at 30/60/90/
    120 min, paired Wilcoxon + subject bootstrap CI (n=6 per direction, low power stated).
    Labelled cross-cohort transfer, not external validation. Holm-corrected separately from P1-P7.

## Deviations

(none yet)
