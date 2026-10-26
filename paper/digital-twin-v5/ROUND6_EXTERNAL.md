# Round 6 P8: cross-cohort transfer (OhioT1DM 2018 <-> 2020), A0 vs A7

**Label: cross-cohort transfer, not external validation.** Both cohorts are the same dataset family. Pre-registered as P8 in PREREG_ROUND6.md.

## Design (as pre-registered)
- Train on one cohort's 6 subjects' train files; validation = purged time-ordered tail (15%) of those same subjects only; scaler fit on training windows only; test = the *other* cohort's official test files (2020 warm-up exclusion applied by the existing loader). No test-cohort subject is seen in training, validation, or scaling.
- A0 (fair curriculum: warmup 2, ramp 2-6) and A7 (Kendall weighting, warmup 0), configs from `scripts/run_seeds_a0_a7.py::get_model_config`; seeds 42/101/202/303/404, every model trained fresh here (no reuse of round-5 checkpoints). Checkpoint selection = the existing `train_model` validation rule, identical for both arms.
- Only deviation: batch size 128 and 2 dataloader workers (GPU shared), identical for both arms (round 5 used 256).
- Metrics: per-subject MAE/RMSE (mg/dL), averaged over the 5 seeds, then paired Wilcoxon (two-sided) and subject-bootstrap (10,000 resamples, seed 0) CI of A0 minus A7 over n=6 test subjects. Positive difference = A7 better. Holm across all 16 tests (2 directions x 4 horizons x 2 metrics), separate from P1-P7.

## Results (MAE in mg/dL, seed- and subject-averaged)
| direction | h | A0 MAE | A7 MAE | A0-A7 MAE [95% boot CI] | A0-A7 RMSE [95% CI] | Wilcoxon p (MAE/RMSE) | Holm p |
|---|---|---|---|---|---|---|---|
| 2018->2020 | 30 | 18.38 | 16.64 | 1.74 [1.21, 2.30] | 1.80 [1.22, 2.43] | 0.031/0.031 | 0.50 |
| 2018->2020 | 60 | 31.48 | 27.80 | 3.68 [2.37, 5.08] | 3.82 [2.31, 5.44] | 0.031/0.031 | 0.50 |
| 2018->2020 | 90 | 40.57 | 35.77 | 4.80 [2.78, 6.94] | 5.08 [2.68, 7.66] | 0.031/0.031 | 0.50 |
| 2018->2020 | 120 | 46.39 | 41.57 | 4.82 [2.26, 7.47] | 5.34 [2.35, 8.51] | 0.031/0.031 | 0.50 |
| 2020->2018 | 30 | 18.71 | 16.14 | 2.56 [2.24, 2.85] | 2.42 [2.18, 2.69] | 0.031/0.031 | 0.50 |
| 2020->2018 | 60 | 31.83 | 26.65 | 5.17 [4.59, 5.68] | 5.40 [4.99, 5.79] | 0.031/0.031 | 0.50 |
| 2020->2018 | 90 | 41.30 | 34.45 | 6.85 [5.95, 7.58] | 7.50 [7.01, 8.06] | 0.031/0.031 | 0.50 |
| 2020->2018 | 120 | 47.78 | 40.39 | 7.40 [6.31, 8.24] | 8.38 [7.73, 9.11] | 0.031/0.031 | 0.50 |

RMSE means and per-subject numbers: `results/tables/round6_external_summary.csv`, `round6_external_per_subject.csv`, `round6_external_per_seed_subject.csv`.

## Reading
- A7 beat A0 on all 6 of 6 test subjects at every horizon, in both directions, for both MAE and RMSE; bootstrap CIs exclude zero. Gap grows with horizon (MAE 1.7-2.6 at 30 min to 4.8-7.4 at 120 min), far larger than the in-cohort 5-seed gap (~0.2-0.7 mg/dL).
- **Power is low.** With n=6 the smallest attainable two-sided exact Wilcoxon p is 0.031 (all six same sign), which is what every test hits; after Holm over 16 tests all adjusted p = 0.5, so none is significant by the pre-registered correction. The bootstrap CI over 6 subjects is also optimistic about subject-level variability. Seed variation is averaged out, not modelled.
- Absolute errors are much worse than in-cohort (30-min MAE 16-19 here; in-cohort 5-seed 30-min MAE is 13.54 (A0) and 13.37 (A7), Table VII; verified to be MAE), as expected for zero-subject transfer from 6 training subjects; both models are weak here, so this says A7 degrades less, not that either forecaster is clinically usable.
- Caveat: A7 gets population-level physics with amortised per-patient parameters (no per-subject lookup), so unseen subjects are handled; a possible reason for A7's advantage is regularisation under small, shifted training data rather than physiological fidelity. This experiment does not distinguish those.
