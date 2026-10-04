# For Samyak: open items after SAMYAK_REMAINING.md (commit 45017eb)

Already applied in this branch (from your answers; please confirm):
- Table V + `data/comprehensive_baselines.csv`: A7 RMSE -> 19.06±2.51 / 30.89±3.89 / 39.04±4.83 / 44.78±5.46;
  ARX 90/120 -> MAE 32.24±3.47 / 37.68±3.69, RMSE 41.82±4.47 / 48.00±4.77 (from `results/tables/a7_vs_arx_wilcoxon.csv`).
- Sec. II-B, after Eq. (1): autocorrelation-based n_eff ≈ 2,500–3,100 (mean tau_int = 53.6 steps), citing
  Bayley & Hammersley (1946) — new reference [40]; later references and supplement citations renumbered.
- A7 vs ARX p-values NOT added to the paper (RMSE at 30 min not significant).

## To do
1. **Regenerate the whole Table V from scripts.** Two rows (A7 RMSE, ARX 90/120) and earlier the persistence SDs
   turned out to be stale draft values. Please re-run every row (persistence, both Bergman ODEs, AR(6), ARX(6),
   LSTM, A0no-mech, A0, A7) from the scripts and confirm Table V / `comprehensive_baselines.csv` match exactly.
2. **ARX fit:** you wrote the old values came from "an earlier unregularized **or** unclipped OLS run".
   Confirm the final ARX in Table V is the Ridge (λ = 1.0) + clipped [40, 400] fit from `scripts/run_ar_arx.py`,
   and that the 30/60-min ARX values (13.97 / 24.48 MAE, 19.61 / 32.67 RMSE) come from the same fit.
3. **166,463 − 20 = 166,443:** excluding the first test hour of the 2020 cohort should remove up to
   6 subjects × 12 readings = 72 observations, not 20. Please explain the 20 (missing readings in that hour?).
4. **Broken link** in `SAMYAK_REMAINING.md`: `file:///home/sammyyakk/...compute_autocorrelation.py` is a local
   path; replace it with the repo path `scripts/compute_autocorrelation.py`.
5. **Confirm the n_eff sentence** and the Bayley & Hammersley reference details
   (J. R. Stat. Soc. Suppl., 8(2), 184–197, 1946).
