# Figure-by-Figure Validation Report
Paper: digital-twin final draft · Repo checked: github.com/sammyyakk/digital-twin (results/ folder)

| Fig | Paper label | Source CSV(s) | Result |
|---|---|---|---|
| 1 | fig:pipeline | — (native TikZ diagram, not a data plot) | N/A |
| 2 | fig:summary | mae_by_horizon.csv, rmse_by_horizon.csv | MAE & RMSE panels match. Skill panel needed to be DERIVED (see below) — do not use skill_vs_persistence.csv directly. |
| 3 | fig:matched | mae_by_horizon.csv + loso/leaderboard_mae.csv | Matching. |
| 4 | fig:ablationfig | ablations/leaderboard_mae_{A0,A1,A2,A3,A4,A7}.csv | Matching exactly: 13.51 / 13.68 / 13.58 / 13.65 / 13.58 / 13.25 |
| 5 | fig:clarke | tables/summary_model.csv | Matching, but ONLY if summary_model.csv (subject-mean) is used, not figures/clarke_quantile_*min.csv (pooled-window; gives slightly different numbers, e.g. 89.86% vs 89.80% at 30 min zone A). |
| 6 | fig:hypo | **none found** | No CSV exists anywhere in the repo. The six numbers (0.566/0.928 sensitivity, 0.993/0.951 specificity, 0.699/0.347 precision) only appear as a markdown table in docs/RESULTS.md (\u00a75.2\u20135.3) — not as machine-readable data. |
| 7 | fig:excursion | tables/summary_model.csv | Matching (TIR, TBR, CV ratio columns). |
| 8 | fig:attribution | attribution/integrated_gradients_by_group_30min.csv, permutation_importance_30min.csv | Left panel matching. Right panel (permutation importance) had a **real ordering/selection bug** — see below. |

## Details

### Figure 2 — RMSE skill panel
`results/figures/skill_vs_persistence.csv` computes skill as the mean of each subject's own skill ratio (a "per-subject-mean" method), giving 19.3% / 19.8% / 20.5% / 21.0%. The paper's Table II uses a different, pooled-ratio method: `1 - mean(RMSE_model) / mean(RMSE_persistence)`, computed directly from `rmse_by_horizon.csv`, giving **19.40% / 20.00% / 20.71% / 21.21%**. These match Table II. Using `skill_vs_persistence.csv` as-is in the figure would silently contradict the table. The corrected code derives skill arithmetically from `rmse_by_horizon.csv` instead.

### Figure 5 — Clarke zones
The repo has two different Clarke computations:
- `results/tables/summary_model.csv` (subject-mean) — 89.80/9.20/0.01/0.99/0.00 at 30 min. **This is what the paper's Table matches.**
- `results/figures/clarke_quantile_30min.csv` (pooled across all windows) — 89.86/9.13/0.01/1.00/0.00 at 30 min. Close but not identical.

Use `summary_model.csv` for consistency with the rest of the paper.

### Figure 6 — Hypoglycemia alarm trade-off
No standalone CSV exists for this comparison anywhere in the repository (checked `results/`, `data/`, and all subfolders). The six values are transcribed from a markdown table in `docs/RESULTS.md` (lines ~195, 229, 477, 484) and `docs/METHODOLOGY.md` (lines ~838, 845, 878). If strict CSV-only sourcing is a hard requirement, this figure cannot be produced until a raw CSV is added to the repo — the values used here are the exact figures documented, just not from a CSV.

### Figure 8 — Permutation importance (real bug, corrected)
The original figure's right panel showed 9 bars: `roc_5, roc_30, bolus_time, night, glucose_mean, glucose_std, roc_15, glucose_level, glucose_max`.

The true top-9 rows in `permutation_importance_30min.csv`, sorted by `mae_increase` (this is the file's own row order, rows 1–9 of the data):

| Rank | Feature | mae_increase |
|---|---|---|
| 1 | roc_5min | 3.113 |
| 2 | roc_30min | 0.428 |
| 3 | minutes_since_bolus | 0.402 |
| 4 | is_night | 0.340 |
| 5 | glucose_mean_1h | 0.321 |
| 6 | **glucose_mean_2h** | 0.296 |
| 7 | glucose_std_1h | 0.252 |
| 8 | roc_15min | 0.233 |
| 9 | **hour_sin** | 0.227 |

The original figure **dropped** `glucose_mean_2h` (rank 6) and `hour_sin` (rank 9), substituting `glucose_max_1h` (0.2268, actual rank 10) and `glucose_mg_dl`/"glucose level" (0.2222, actual rank 11) — two features that don't belong in the top 9. It also plotted them out of order (the smaller-valued "glucose level" appeared above the larger-valued "glucose max"). `figures.tex` fixes this using the CSV's true top-9, in the CSV's own order.

### Minor, non-blocking note
Figure 7's caption text ("predicted TIR climbs toward 74%") is a loose approximation — the actual 120-min value in `summary_model.csv` is 73.16%. The plotted graph itself is correct; only the prose wording is imprecise.
