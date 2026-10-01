# Corrections applied (final check against github.com/sammyyakk/digital-twin results/ and docs/)

## Data / claim corrections
1. Fig. 3 (matched-protocol): LOSO points at 90/120 min corrected 32.40/37.90 -> 32.28/37.56 (results/loso/leaderboard_mae.csv).
2. Sec. III-H: removed the retracted claim that 60-min RMSE 30.52 / MAE 21.98 beat the comparison set (RESULTS.md §8.4). Now uses the matched 2020 cohort: RMSE 31.16 tied with best (31.10); MAE 22.55 lower than all entries except Pavan [60].
3. Table VII: Current Work / LOSO / Persistence rows switched to the 2020-cohort subset (same 6 subjects as the published entries), computed from per-subject CSVs:
   - Current Work: 18.74 13.24 31.16 22.55 39.52 29.14 45.39 34.03
   - LOSO:         20.93 14.98 34.55 25.45 43.50 32.69 49.36 37.76
   - Persistence:  24.22 17.45 40.11 29.51 51.15 38.34 58.98 44.83
   Caption now says "2020 Challenge Cohort"; footnote explains and points to Table IV for all-12 results.
4. Sec. III-A: 2018 persistence validation updated 22.60±2.50 / 36.34±3.14 -> 22.52±2.56 / 36.19±3.26 (tables/per_subject_persistence.csv); agreement now "within 0.02 mg/dL (30 min) and 0.41 mg/dL (60 min)".
5. Fig. 2 caption: removed "Error bars are subject-level SD" (no error bars are plotted).
6. Sec. III-C: "10 of 12 subjects" -> "all 12 subjects by MAE (10 of 12 by RMSE)".
7. Fig. 3 caption: "increasingly approaches persistence" -> constant ~12% MAE advantage over persistence; gap to personalized widens 1.6 -> 4.3 mg/dL.
8. Sec. III-E: added Parkes result (pooled over windows, results/figures/parkes_quantile_*min.csv): A+B 99.8% (30 min) -> 95.9% (120 min), zone E 0% at all horizons.
9. Sec. II-G: skill definition now states it is computed from cohort-mean RMSEs (matches 19.4-21.2%).

## References / formatting
10. Removed verification notes from refs [12], [13], [14], [17], [28].
11. Previously uncited refs now cited: [65], [70] (other 2020 entries, Sec. III-H), [75] (GARNN all-12 comparison, Sec. III-H), [79] (preregistration, Sec. II-H).
12. Added missing \label to 46 bibitems (fixed undefined hyperlinks to [48], [49]).
13. Removed 11 stray "\\" line breaks after citation ranges (e.g. "[58-59]\n, this ablation").
14. Paper-roadmap paragraph rewritten to match actual sections I-VI (old text had an unfinished sentence).

## NOT changed - authors to decide
- Background paragraph citations appear shifted vs Literature Review: [3-4] (Transformers), [6-10] (PINN), [7-8] (simulator-only), [14] (RL). Verify intended refs.
- Ref [13] (Colmegna): old note said content match "not fully certain" - confirm it supports the citing sentence.
- Citation order: Table I (in Introduction) cites [58], [66]-[74] before [30]; IEEE prefers first-appearance order.
- Eq. (1) "135,000 slots" has no source in the repo (appears back-calculated from 2,800 x 48).
- MISMATCH_REPORT.md (earlier report) lists Fig. 3 as matching; it was not (see item 1).