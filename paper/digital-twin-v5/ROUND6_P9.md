
## P9 — Multi-seed hypoglycemia alarm (q10<70, 30 min), A0 vs A7

Script: `scripts/round6_hypo_multiseed.py`; table: `results/tables/round6_hypo_multiseed.csv` (MAE: `round6_hypo_multiseed_mae.csv`).
Seeds 101-404 reused unchanged (`artifacts/seeds/`). Seed 42 retrained for both arms with the identical
`get_model_config` config (patience 10, 70 epochs, quantile head 0.1/0.5/0.9) into `artifacts/seeds_round6/{A0,A7}_s42`.
Same 26,498 test windows, 56 episodes, identical alarm/event logic as `recompute_hypo_a0_vs_a7_fair.py`.

| seed | A0 detected/56 (sens) | A0 FA/day | A0 lead | A7 detected/56 (sens) | A7 FA/day | A7 lead |
|---|---|---|---|---|---|---|
| 42 | 44 (0.786) | 2.53 | 17.5 | 37 (0.661) | 2.36 | 15.0 |
| 101 | 38 (0.679) | 2.15 | 15.0 | 39 (0.696) | 2.27 | 10.0 |
| 202 | 37 (0.661) | 2.23 | 20.0 | 45 (0.804) | 2.27 | 15.0 |
| 303 | 34 (0.607) | 2.17 | 17.5 | 39 (0.696) | 1.97 | 15.0 |
| 404 | 36 (0.643) | 2.01 | 15.0 | 45 (0.804) | 2.33 | 15.0 |
| mean ± SD | 0.675 ± 0.067 | 2.22 ± 0.19 | 17.0 ± 2.1 | 0.732 ± 0.067 | 2.24 ± 0.16 | 14.0 ± 2.2 |

Specificity: A0 0.954 ± 0.004, A7 0.953 ± 0.004. A7 − A0 sensitivity = +0.057; two-level bootstrap
(subjects, then seeds; 5000 reps; seed 20260626) 95% CI [−0.057, +0.153]; P(diff>0) = 0.852. CI includes zero.
Validation-tuned persistence alarm (round 5, not retuned): sens 0.679 (38/56), spec 0.920, FA/day 2.21, lead 17.5 min.
Both arm means are within noise of persistence on sensitivity; the NN arms have higher specificity.

Verdict: the round-5 single-seed A0 > A7 result (0.786 vs 0.679) is not reproduced as a general effect; seed 42 A0 is the high outlier
for A0 (0.786 again) and seed 42 A7 the lowest for A7. Over 5 matched seeds A7 is nominally higher but not significant.

Seed-42 MAE effect (retrained vs existing table; 5-seed mean ± SD of per-subject-mean MAE):
A0 30/60/90/120 min: new 13.547/23.001/29.991/35.050 vs table 13.543/23.007/30.006/35.073.
A7: new 13.392/22.605/29.282/34.087 vs table 13.370/22.599/29.283/34.080. Seed-42 MAE change is negligible (<=0.03 min-level on 5-seed means).
(Retrained seed-42 MAE: A0 13.53/22.79/29.66/34.82; A7 13.36/22.32/28.77/33.42.) `multiseed_*.csv` untouched.
