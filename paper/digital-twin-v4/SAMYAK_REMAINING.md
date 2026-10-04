# For Samyak: remaining checks after the checklist answers

## Must answer (number mismatches a reviewer could spot)
1. A7 vs ARX Wilcoxon test: your test values differ from Table V except MAE at 30/60 min.
   - ARX MAE 90/120: 32.24 / 37.68 (test) vs 32.41 / 38.00 (Table V)
   - A7 RMSE 30/60/90/120: 19.06 / 30.89 / 39.04 / 44.78 vs 19.16 / 31.13 / 39.46 / 45.14
   - ARX RMSE 90/120: 41.82 / 48.00 vs 42.23 / 48.87
   Why do they differ (windows, run, aggregation)? Are the p-values Holm-corrected?
   RMSE at 30 min is not significant (p = 0.077). P-values stay out of the paper until resolved.
2. Eq. (1): your autocorrelation N is 166,520 but the paper says 166,443 CGM observations. Which is right?
   Is "135,000 slots" an exact count? Commit the autocorrelation script (tau_int = 53.63) so
   n_eff = 2,517-3,105 is reproducible; it also needs a citation (Bayley & Hammersley, 1946).

## Confirm (already written into main.tex using your numbers)
3. Sec. III-G3 now reports the strict G<70 target for the 84 mg/dL alarm:
   sensitivity 0.988, specificity 0.868, precision 0.175, 3,397 FP; and 1,257 FP for the 70 mg/dL alarm.
   The G<84 numbers (0.943 / 0.908 / 0.451) are kept as the redefined-target case. Confirm all final.
4. "Post hoc" added for the 5-seed replication: abstract, Sec. III-E, Sec. III-E1, Table VII caption,
   Limitations, Conclusion. (The Table VII caption did not say this before.)
5. Table VI caption now says Table IX's A7 row is the final model of Table IV (your item 7).
6. ICR item now states g/U, so a positive correlation is expected (your item 13).
7. Eq. (1) now explains 48 = 24 input + 24 forecast steps and calls n_eff a coarse approximation.

## Review requests removed from the checklist (do, or give a reason for the response letter)
8. Concern 11: PatchTST or TFT baseline.
9. Concern 13: Zenodo DOI for the code.
10. Table IV: per-subject skill +/- SD (no data in data/).
11. Sec. II-A: citations of published OhioT1DM studies showing the integrity pitfalls (optional).
