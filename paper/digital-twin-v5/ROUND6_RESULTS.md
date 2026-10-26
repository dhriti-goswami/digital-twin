# Round 6 results (pre-registered analyses P1-P7)

Pre-registration: `PREREG_ROUND6.md` (commit c84348f), executed unchanged. Models: A7 vs fair A0 (round-5
selection window), seeds 42/101/202/303/404, same test split as Table VII (12 subjects, 26,498 windows).
Generators: `scripts/round6_regime_analysis.py` (P1-P5, P7, and the P6 statistics), `scripts/round6_low_data.py`
(P6 training), `scripts/round6_seed42_trajectories.py` (helper, see Deviations 1). CSVs: `results/tables/round6_*.csv`
(`round6_all_tests.csv` has every test; `round6_tables.md` is the raw table dump below).
The script asserts that the seed-averaged per-subject MAE reproduces Table VII (means and SDs, all arms and
horizons) before doing anything else; it passed.

**Conventions.** gap = A0 - A7 (positive = A7 has lower error / fewer violations). Unit of inference = subject
(seed-averaged per-subject value). Paired two-sided Wilcoxon, subject-level percentile bootstrap 95% CI
(5000 resamples). **One Holm family of 55 tests** over everything marked in the tables (P1: 5, P2: 12, P3: 12,
P4: 12, P5: 2, P6: 8, P7: 4; the 100%-data P6 rows are descriptive and excluded). With n = 12 subjects the smallest
attainable Wilcoxon p is 0.00049, so after Holm over 55 tests the best attainable adjusted p is 0.027; the family
is power-limited by construction. Verdict column rule (stated here, applied mechanically): "helps"/"hurts" only if
the CI excludes 0 AND Holm p < 0.05; otherwise "no difference detected" (which includes cases where the raw CI
excludes 0 but Holm does not).

## Summary

| Analysis | Result | Verdict |
|---|---|---|
| P1 horizon trend | Gap grows with horizon: +0.17 / +0.41 / +0.72 / +0.99 mg/dL (rel. 1.3 / 1.8 / 2.4 / 2.8%). Per-subject Spearman(gap, horizon): mean rho +0.68 [0.33, 0.97], 10/12 subjects increasing; raw p 0.0039, Holm 0.17. Each horizon gap has raw p 0.0015-0.0093, Holm 0.069-0.35 | consistent positive trend, not Holm-significant |
| P2 post-meal | Post-meal gaps +0.16/+0.19/+0.30/+0.54 (all CIs include 0, n=11 subjects); non-post-meal gaps +0.15/+0.47/+0.97/+1.36 (Holm-significant at 30, 60, 90 min). Gap(post-meal) minus gap(other) is NEGATIVE at 60-120 min (-0.30/-0.73/-0.94; raw p 0.58/0.024/0.054, Holm 1.0) | physics does NOT help more after meals; if anything the gain is concentrated outside meal windows (not Holm-significant) |
| P3 glycemic range | Gain sits in high targets: >180 mg/dL +0.30/+0.85/+1.58/+1.73 (Holm-significant at 60 and 90 min); 70-180 about +0.1-0.3 (n.s.); <70 CIs span 0 with width of several mg/dL (n.s., only ~720 windows) | helps for hyperglycemic targets; no detectable difference elsewhere (hypo: underpowered) |
| P4 rate of change | Gain concentrated in the calm stratum (|ROC15|<1): +0.15/+0.43/+0.78/+1.12 (Holm-significant at 90, 120); 1-2 and >2 positive but CIs include 0 | helps when glucose is steady; no detectable difference when it is changing fast |
| P5 plausibility | Trajectories with |dG/dt|>4 mg/dL/min anywhere: A0 36.8% vs A7 0.0% of windows (gap +0.368 [0.345, 0.390], Holm 0.027). Trajectories <40 or >400: both about 0 (gap 0.000, n.s.) | helps (steep-slope); no difference (range) |
| P6 low data | A7 better in every fraction x horizon, gap grows as data shrink: 25% data +0.49/+1.25/+1.94/+2.35; 50% data +0.15/+0.55/+0.99/+1.41 (100%, same 3 seeds, descriptive: +0.14/+0.32/+0.61/+0.95). Raw p 0.0049-0.27; none Holm-significant (3 seeds only) | directionally helps, larger when data are scarce; not Holm-significant |
| P7 seed variance | Across-seed SD of mean MAE is HIGHER for A7 at every horizon (A7/A0: 1.2x, 2.0x, 3.5x, 6.3x at 30/60/90/120 min). Gap in SD (A0-A7) -0.02/-0.16/-0.38/-0.54; Brown-Forsythe raw p 0.43/0.10/0.033/0.019, Holm >= 0.64 | no stability advantage; A7 is directionally LESS stable (not Holm-significant) |

Plain-language reading: physics guidance gives a small, consistent, horizon-increasing MAE gain (about 1-3%),
located in steady / high-glucose / non-meal conditions rather than where mechanistic insulin and carbohydrate dynamics
would be expected to matter most (post-meal, fast-changing); it removes physiologically implausible steep
forecast slopes; the gain looks larger with less training data; and it does not improve (it may worsen) seed-to-seed
stability. Only a minority of these effects survive Holm correction over 55 tests, and several of the more
"interesting" patterns (P2 difference-of-gaps, P6, P7) do not.

## Full tables

## P1

| stratum | h (min) | n subj | n win | A0 | A7 | gap A0-A7 | 95% CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 30 | 12 | 26498 | 13.543 | 13.370 | +0.174 | [+0.055, +0.304] | 0.0093 | 0.3525 | no difference detected |
| all | 60 | 12 | 26498 | 23.007 | 22.599 | +0.408 | [+0.180, +0.665] | 0.0049 | 0.2051 | no difference detected |
| all | 90 | 12 | 26498 | 30.006 | 29.283 | +0.723 | [+0.369, +1.113] | 0.0024 | 0.1099 | no difference detected |
| all | 120 | 12 | 26498 | 35.073 | 34.080 | +0.993 | [+0.533, +1.493] | 0.0015 | 0.0688 | no difference detected |
| trend (per-subject Spearman gap~horizon, Wilcoxon vs 0) | - | 12 |  |  |  | +0.683 | [+0.333, +0.967] | 0.0039 | 0.1680 | no difference detected |

P1 trend detail: mean_subject_rho=0.6833, ci_lo=0.3333, ci_hi=0.9667, p_raw=0.003906, n_subjects=12, pooled_rho_descriptive=0.4641, pooled_p_descriptive=0.000893, n_subjects_gap_increasing=10

## P2

| stratum | h (min) | n subj | n win | A0 | A7 | gap A0-A7 | 95% CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| post-meal (0-180 min) | 30 | 11 | 10066 | 16.376 | 16.215 | +0.161 | [-0.083, +0.441] | 0.4648 | 1.0000 | no difference detected |
| other | 30 | 12 | 16432 | 11.738 | 11.586 | +0.152 | [+0.079, +0.229] | 0.0010 | 0.0488 | helps |
| difference of gaps (post-meal minus other) | 30 | 11 |  |  |  | +0.013 | [-0.214, +0.262] | 0.8984 | 1.0000 | no difference detected |
| post-meal (0-180 min) | 60 | 11 | 10066 | 26.794 | 26.605 | +0.190 | [-0.427, +0.801] | 0.6377 | 1.0000 | no difference detected |
| other | 60 | 12 | 16432 | 20.490 | 20.022 | +0.467 | [+0.315, +0.651] | 0.0005 | 0.0269 | helps |
| difference of gaps (post-meal minus other) | 60 | 11 |  |  |  | -0.302 | [-0.889, +0.261] | 0.5771 | 1.0000 | no difference detected |
| post-meal (0-180 min) | 90 | 11 | 10066 | 33.772 | 33.468 | +0.304 | [-0.331, +0.888] | 0.4648 | 1.0000 | no difference detected |
| other | 90 | 12 | 16432 | 27.562 | 26.594 | +0.968 | [+0.611, +1.361] | 0.0005 | 0.0269 | helps |
| difference of gaps (post-meal minus other) | 90 | 11 |  |  |  | -0.732 | [-1.282, -0.229] | 0.0244 | 0.7812 | no difference detected |
| post-meal (0-180 min) | 120 | 11 | 10066 | 38.068 | 37.531 | +0.537 | [-0.040, +1.099] | 0.1230 | 1.0000 | no difference detected |
| other | 120 | 12 | 16432 | 33.054 | 31.693 | +1.361 | [+0.721, +2.066] | 0.0024 | 0.1099 | no difference detected |
| difference of gaps (post-meal minus other) | 120 | 11 |  |  |  | -0.937 | [-1.811, -0.114] | 0.0537 | 1.0000 | no difference detected |

## P3

| stratum | h (min) | n subj | n win | A0 | A7 | gap A0-A7 | 95% CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| target <70 | 30 | 12 | 728 | 13.442 | 13.188 | +0.254 | [-0.865, +1.337] | 0.5693 | 1.0000 | no difference detected |
| target 70-180 | 30 | 12 | 16434 | 11.982 | 11.893 | +0.089 | [-0.032, +0.209] | 0.2334 | 1.0000 | no difference detected |
| target >180 | 30 | 12 | 9336 | 16.967 | 16.672 | +0.296 | [+0.119, +0.468] | 0.0122 | 0.4395 | no difference detected |
| target <70 | 60 | 12 | 721 | 30.167 | 29.377 | +0.790 | [-1.771, +3.112] | 0.3804 | 1.0000 | no difference detected |
| target 70-180 | 60 | 12 | 16402 | 19.583 | 19.513 | +0.070 | [-0.249, +0.397] | 0.9697 | 1.0000 | no difference detected |
| target >180 | 60 | 12 | 9375 | 29.853 | 28.999 | +0.854 | [+0.505, +1.237] | 0.0005 | 0.0269 | helps |
| target <70 | 90 | 12 | 714 | 43.181 | 43.108 | +0.073 | [-3.562, +3.077] | 0.6221 | 1.0000 | no difference detected |
| target 70-180 | 90 | 12 | 16392 | 24.709 | 24.606 | +0.103 | [-0.390, +0.544] | 0.5693 | 1.0000 | no difference detected |
| target >180 | 90 | 12 | 9392 | 40.178 | 38.596 | +1.582 | [+0.895, +2.318] | 0.0010 | 0.0488 | helps |
| target <70 | 120 | 12 | 706 | 52.567 | 52.689 | -0.123 | [-4.339, +3.373] | 0.6221 | 1.0000 | no difference detected |
| target 70-180 | 120 | 12 | 16408 | 28.277 | 27.948 | +0.329 | [-0.235, +0.816] | 0.1763 | 1.0000 | no difference detected |
| target >180 | 120 | 12 | 9384 | 47.730 | 45.999 | +1.731 | [+0.586, +2.995] | 0.0210 | 0.6929 | no difference detected |

## P4

| stratum | h (min) | n subj | n win | A0 | A7 | gap A0-A7 | 95% CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| |ROC15|<1 | 30 | 12 | 20824 | 11.862 | 11.708 | +0.154 | [+0.078, +0.232] | 0.0068 | 0.2734 | no difference detected |
| |ROC15| 1-2 | 30 | 12 | 4582 | 18.259 | 18.052 | +0.207 | [-0.115, +0.589] | 0.4697 | 1.0000 | no difference detected |
| |ROC15|>2 | 30 | 12 | 1092 | 25.085 | 24.873 | +0.212 | [-0.335, +0.744] | 0.5693 | 1.0000 | no difference detected |
| |ROC15|<1 | 60 | 12 | 20824 | 20.958 | 20.525 | +0.433 | [+0.243, +0.633] | 0.0015 | 0.0688 | no difference detected |
| |ROC15| 1-2 | 60 | 12 | 4582 | 29.226 | 28.892 | +0.334 | [-0.281, +1.031] | 0.4697 | 1.0000 | no difference detected |
| |ROC15|>2 | 60 | 12 | 1092 | 35.838 | 35.710 | +0.128 | [-0.844, +1.057] | 0.7334 | 1.0000 | no difference detected |
| |ROC15|<1 | 90 | 12 | 20824 | 27.983 | 27.206 | +0.777 | [+0.433, +1.137] | 0.0005 | 0.0269 | helps |
| |ROC15| 1-2 | 90 | 12 | 4582 | 36.130 | 35.533 | +0.597 | [-0.087, +1.252] | 0.2036 | 1.0000 | no difference detected |
| |ROC15|>2 | 90 | 12 | 1092 | 43.022 | 42.730 | +0.292 | [-1.197, +1.780] | 0.7910 | 1.0000 | no difference detected |
| |ROC15|<1 | 120 | 12 | 20824 | 33.093 | 31.976 | +1.116 | [+0.655, +1.614] | 0.0010 | 0.0488 | helps |
| |ROC15| 1-2 | 120 | 12 | 4582 | 41.004 | 40.416 | +0.588 | [-0.155, +1.348] | 0.2036 | 1.0000 | no difference detected |
| |ROC15|>2 | 120 | 12 | 1092 | 47.298 | 47.057 | +0.241 | [-1.250, +1.720] | 0.9697 | 1.0000 | no difference detected |

## P5

| stratum | h (min) | n subj | n win | A0 | A7 | gap A0-A7 | 95% CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| frac trajectories with |dG/dt|>4 mg/dL/min | - | 12 | 26498 | 0.368 | 0.000 | +0.368 | [+0.345, +0.390] | 0.0005 | 0.0269 | helps |
| frac trajectories <40 or >400 mg/dL | - | 12 | 26498 | 0.000 | 0.000 | +0.000 | [-0.000, +0.001] | 0.6250 | 1.0000 | no difference detected |

## P6

| stratum | h (min) | n subj | n win | A0 | A7 | gap A0-A7 | 95% CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| train on 25% of subjects (3 seeds) | 30 | 12 |  | 16.180 | 15.693 | +0.487 | [-0.086, +0.924] | 0.0425 | 1.0000 | no difference detected |
| train on 25% of subjects (3 seeds) | 60 | 12 |  | 26.793 | 25.543 | +1.250 | [+0.382, +1.993] | 0.0269 | 0.8325 | no difference detected |
| train on 25% of subjects (3 seeds) | 90 | 12 |  | 34.293 | 32.352 | +1.941 | [+0.880, +2.928] | 0.0122 | 0.4395 | no difference detected |
| train on 25% of subjects (3 seeds) | 120 | 12 |  | 39.723 | 37.374 | +2.349 | [+1.215, +3.464] | 0.0049 | 0.2051 | no difference detected |
| train on 50% of subjects (3 seeds) | 30 | 12 |  | 14.873 | 14.719 | +0.154 | [-0.009, +0.321] | 0.2661 | 1.0000 | no difference detected |
| train on 50% of subjects (3 seeds) | 60 | 12 |  | 24.953 | 24.401 | +0.553 | [+0.156, +0.972] | 0.0342 | 0.9912 | no difference detected |
| train on 50% of subjects (3 seeds) | 90 | 12 |  | 32.435 | 31.447 | +0.988 | [+0.414, +1.528] | 0.0093 | 0.3525 | no difference detected |
| train on 50% of subjects (3 seeds) | 120 | 12 |  | 37.959 | 36.550 | +1.409 | [+0.639, +2.129] | 0.0068 | 0.2734 | no difference detected |
| 100% of subjects (3 seeds; descriptive reference, not in Holm) | 30 | 12 |  | 13.496 | 13.357 | +0.139 | [-0.006, +0.287] | 0.0522 | nan | descriptive |
| 100% of subjects (3 seeds; descriptive reference, not in Holm) | 60 | 12 |  | 22.927 | 22.610 | +0.317 | [+0.052, +0.624] | 0.0640 | nan | descriptive |
| 100% of subjects (3 seeds; descriptive reference, not in Holm) | 90 | 12 |  | 29.952 | 29.341 | +0.611 | [+0.205, +1.028] | 0.0161 | nan | descriptive |
| 100% of subjects (3 seeds; descriptive reference, not in Holm) | 120 | 12 |  | 35.068 | 34.124 | +0.945 | [+0.481, +1.408] | 0.0034 | nan | descriptive |

## P7

| stratum | h (min) | n subj | n win | A0 | A7 | gap A0-A7 | 95% CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| across-seed SD of mean MAE (gap = SD_A0 - SD_A7) | 30 | 5 |  | 0.097 | 0.114 | -0.017 | [-0.114, +0.072] | 0.4262 | 1.0000 | no difference detected |
| across-seed SD of mean MAE (gap = SD_A0 - SD_A7) | 60 | 5 |  | 0.160 | 0.320 | -0.160 | [-0.289, +0.060] | 0.1038 | 1.0000 | no difference detected |
| across-seed SD of mean MAE (gap = SD_A0 - SD_A7) | 90 | 5 |  | 0.152 | 0.527 | -0.376 | [-0.571, -0.045] | 0.0326 | 0.9781 | no difference detected |
| across-seed SD of mean MAE (gap = SD_A0 - SD_A7) | 120 | 5 |  | 0.101 | 0.637 | -0.536 | [-0.673, -0.097] | 0.0187 | 0.6366 | no difference detected |

## P5 per-model rates

arm,seed,frac_steep,frac_out_of_range,max_slope,min_traj,max_traj
A0,42,0.35089,0.00019,16.74161,30.29111,400.0
A0,101,0.38075,0.0,17.80316,40.0,400.0
A0,202,0.35354,0.00068,17.91338,29.15358,400.0
A0,303,0.3544,0.00045,16.48416,34.04096,400.0
A0,404,0.39222,8e-05,18.46074,37.97666,400.0
A7,42,4e-05,0.00011,4.3324,38.49751,400.0
A7,101,0.0,4e-05,3.06354,36.58388,400.0
A7,202,0.0,0.00019,3.52798,7.0059,400.0
A7,303,4e-05,4e-05,4.09314,38.38069,400.0
A7,404,0.0,4e-05,3.34052,39.24401,400.0


Post-meal windows per subject: 540:647, 544:1078, 552:354, 559:753, 563:825, 567:0, 570:934, 575:1067, 584:604, 588:1135, 591:1284, 596:1385

## Deviations and interpretation choices

None of these were chosen after looking at the sign or size of any result; items marked (spec gap) are places where
PREREG_ROUND6.md was silent and a choice had to be made when writing the code.

1. **Seed-42 checkpoints required the original code.** `artifacts/official/abl-A0` and `abl-A7` (the seed-42 runs
   that Table VII reuses) were trained at commit 7eb9ad2, before the quantile head was added (1a275e8); the current
   `PhysicsGuidedForecaster` cannot load them (state-dict key mismatch). Table VII itself reuses their saved
   predictions. For P1-P5/P7 I re-ran them with the original code (a temporary git worktree of 7eb9ad2, deleted
   afterwards; helper `scripts/round6_seed42_trajectories.py`) and verified the resulting horizon predictions are
   **bit-identical** (max abs diff 0.0) to the saved `predictions.npz`, for both arms. The 1-minute trajectory
   statistics for P5 (seed 42) come from the same pass. No model was retrained for P1-P5/P7.
2. (spec gap) **P1 trend test.** "Spearman of per-subject gap vs horizon" is ambiguous. The tested quantity is the
   per-subject Spearman rho between horizon and gap (4 points per subject), tested across the 12 subjects by
   Wilcoxon vs 0 (subject = unit of inference), CI by subject bootstrap. A pooled 48-point Spearman (rho 0.464,
   p 0.0009; pseudo-replicated, so anti-conservative) is reported descriptively only and is not in the Holm family.
   Relative gap = gap / mean A0 MAE.
3. (spec gap) **Post-meal definition (P2).** "Carbohydrate input" = grid slot with `carbs_mg_per_min > 0` in the test
   file; a window is post-meal if its anchor is 0-180 min after the latest such slot at or before the anchor. Meals
   before the start of the test file are unobservable, so early-file windows may be misclassified as "other".
   Subject 567 has zero carbohydrate entries in the test data, hence no post-meal windows: post-meal strata and the
   difference-of-gaps use 11 subjects. The "difference between strata" is tested paired by subject on the per-subject
   gaps.
4. (spec gap) **ROC variable (P4).** `|roc_15min|` (feature at the anchor, mg/dL/min). roc_5min and roc_30min were not
   analysed. Strata: <1, [1, 2], >2.
5. (spec gap) **Range strata (P3)** are defined by the target glucose at the evaluated horizon: <70, [70, 180], >180.
   Only ~710-730 windows are <70 per horizon, so that stratum is underpowered.
6. (spec gap) **P5 definition.** Trajectory = the 121-point 1-minute collocation forecast (`output.collocation`) over
   0-120 min; slope = first difference per minute; "anywhere" = max over the trajectory. The model output is
   saturated at 400 mg/dL, so ">400" cannot occur by construction (observed max exactly 400.0 for all models); the
   <40 / >400 criterion therefore effectively measures <40 only. **Context not pre-registered, from a one-off check
   [Superseded by "P5 robustness" section below: violations are at BOTH ends of the horizon.]    on A0 seed 101 (first 22 batches):** the A0 steep slopes are located almost entirely in the last ~10 minutes of the
   horizon (spline end-boundary behaviour), not along the whole trajectory; I did not check A7's slope location or
   the other seeds. The 36.8% vs 0.0% contrast should be read as "A0's free spline has large end-of-horizon slopes
   that the physics-regularised A7 does not", not as evidence about the 30-120 min forecast values (MAE) themselves.
7. (spec gap) **P7 tests.** Across-seed SD of the 12-subject-mean MAE (the Table VII SDs, ddof=1, n=5 seeds per arm);
   CI for SD_A0 - SD_A7 by resampling seeds within each arm (5000x); p by the Brown-Forsythe (median-centred Levene)
   test. With 5 seeds per arm this has very little power. Only MAE was analysed.
8. **P6 implementation.** Same configs/curricula as the fair A0 and A7 runs (`get_model_config`; A0 with the
   round-5 curriculum, batch size 256), seeds 42/101/202, 25% = 3 and 50% = 6 of the 12 training subjects. Subset =
   `default_rng(seed).choice(sorted ids, k)` (identical for A0 and A7 at a given seed and fraction; differs across
   seeds; listed in `round6_p6_runs.csv`). Validation split restricted to the same subjects; scaler refit on the
   subset's training slots; test set built from the full fold (so each test subject's G_b / I_b are resolved from its
   own training period exactly as in the full-data runs) but scaled with the subset scaler. DataLoader shuffle seed
   follows the original script (base seed 42 for all runs). Tests: per fraction x horizon, paired Wilcoxon on
   seed-averaged (3 seeds) per-subject MAE, 8 tests in the Holm family. The 100%-data rows (same 3 seeds from the
   existing checkpoints) are shown only as a descriptive reference. Because only 3 seeds and 3 different subject
   subsets per fraction were run, between-subset variability is confounded with the seed effect.
9. **Holm family.** All 55 tests in the tables above (listed under Conventions), computed once across the whole file,
   including P6 and P7. P5's two metrics and P1's trend test are single tests in the family.
10. Nothing infeasible was dropped: every pre-registered analysis P1-P7 was run. No existing results, tables, or paper
    files were modified; new files are `scripts/round6_*.py`, `results/tables/round6_*`, this file, and
    `artifacts/round6_*` (seed-42 trajectory stats and the 12 low-data checkpoints).

## P5 robustness (post hoc, not pre-registered)

Script: `scripts/round6_p5_robustness.py`; table: `results/tables/round6_p5_robustness.csv` (summary rows plus a per-minute
histogram). Same models, test windows, subject-level unit of inference, seed-averaging, paired Wilcoxon and 5000x subject
bootstrap (rng 42) as P5 (`paired_test` imported from `round6_regime_analysis.py`). No Holm correction (post hoc,
descriptive). Seed-42 trajectories re-generated with the pre-quantile-head code (7eb9ad2); slopes match the stored P5
values exactly (max abs diff 0.0). The "full" row reproduces P5 (0.368 vs 0.000).

**1. Is the comparison like-for-like?** Partly. Same: both arms use the same `SplineEvaluator` grid (121 points, 1 min,
0-120 min), the same quantile/spline head, and the same slope computation (first difference of `output.collocation`,
per minute, max over the trajectory). Different: the A0 trajectory is the free cubic B-spline alone (`hybrid_residual=False`).
The A7 trajectory is `anchor + gate*(Bergman ODE forecast - anchor) + (learned spline - anchor)` (`hybrid_residual=True`,
learned prior gate about 0.16 on seed 101), so it contains an ODE-integrated component, and its learned spline is
trained with the physics residual on the collocation grid. A7's smoothness is therefore partly built into the output
path, and the 36.8% vs 0.0% contrast compares a free spline with an ODE-prior-plus-spline, not two identical interpolants
that differ only in a loss term. It is not an artifact of different grids or differencing.

**2. Fraction of windows with |dG/dt| > 4 mg/dL/min** (mean of per-subject fractions, 5 seeds, 12 subjects):

| variant | A0 | A7 | gap A0-A7 | 95% CI | raw p |
|---|---|---|---|---|---|
| full, 1-min diffs (P5) | 0.368 | 0.000 | +0.368 | [+0.345, +0.391] | 0.0005 |
| (a) excluding last 10 min | 0.194 | 0.000 | +0.194 | [+0.175, +0.211] | 0.0005 |
| (b) excluding first and last 10 min | 0.0010 | 0.000 | +0.001 | [+0.0004, +0.0016] | 0.0039 |
| (c1) 5-min points (25 pts), diff/5 | 0.204 | 0.000 | +0.204 | [+0.187, +0.218] | 0.0005 |
| (c2) native points 0/30/60/90/120, diff/30 | 0.000 | 0.000 | 0 | [0, 0] | 1.0 |

Per-seed A0 full rates 0.351-0.392; A7 non-zero in one seed only (0.004%, one window). (c2) is uninformative: a 4 mg/dL/min
slope over 30 min means a 120 mg/dL jump, which neither arm produces.

**3. Where A0 violations occur.** Not only at the end. Per-minute violation rate of A0 (share of windows whose
1-min slope in that minute exceeds 4): first 5 minutes 19.2%, 13.5%, 8.3%, 4.1%, 1.0%; minutes 10-109 at most 0.09%
(about zero); last 5 minutes 1.3%, 9.2%, 18.6%, 27.8%, 36.2% (peak at the final minute). Of A0 windows with any violation,
47.6% violate only in the last 10 min. So the earlier spot check (last ~10 min only) was incomplete: the free spline is steep at
**both** ends of the horizon (clamped-knot boundary behaviour; the start is pinned to the anchor, so the first minutes
show the spline departing it), and the interior 10-110 min is essentially clean (0.1%). The 120-min forecast value is the final trajectory point, so
the end segment is adjacent to a reported horizon; P5 itself says nothing about MAE.

**Suggested wording.** "In an exploratory, post hoc check, 36.8% of A0 forecast trajectories exceeded 4 mg/dL/min
versus 0.0% for A7. These excursions are boundary effects of the unconstrained spline: they occur almost entirely in the
first and last 5 minutes of the 120-min horizon (interior 10-110 min: 0.1% for A0, 0% for A7); the A0 rate is still
20% when the trajectory is sampled at 5-min steps (which include the end segments) and 0% at 30-min steps. A7's
trajectory additionally includes an ODE-integrated prior, so this is evidence that the physics-guided output is smoother at
the horizon boundaries, not that the learned forecasts are more physiologically plausible in the interior."
