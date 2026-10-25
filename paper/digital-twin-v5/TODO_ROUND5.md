# Round 5 to-do (26 Oct 2026)

Only the open items. Everything else is done (see `CHANGES.md`).
**Must** = needed before submission; **Should** = do if time allows. Samyak's items come first, because Dhriti's
D-3 depends on them. Do S-15 first.

---

## Samyak — to be answered

- [ ] **S-15. Rerun A0 with a fair selection window (Must).**
  In `scripts/run_seeds_a0_a7.py`, seed 42 of A0 reuses the ablation run (selection can start at epoch 3), but
  seeds 101–404 use the `configs/official-small.yaml` defaults (warmup 10, ramp 25), so A0 cannot select before
  epoch 11. Those seeds chose epochs 13, 11, 11, 11, mostly the first epoch allowed. A7 could select from epoch 1
  in every seed. A0's four retrained seeds average about 13.97 (seed 42: 13.51) vs A7's about 13.40 (seed 42: 13.25),
  so the restriction may explain part of the 0.51 mg/dL physics gain.
  - Retrain A0 seeds 101–404 with selection allowed from the same epoch as seed 42 (or from epoch 1); about
    3 min per run.
  - Redo Table VII, the Wilcoxon/Holm tests and the bootstrap CIs (`physics_effect_cis.csv`).
  - The quantile-head A0 in Table IX has the same restriction (warmup 10, ramp 25, best epoch 13): retrain it the
    same way and redo `hypoglycemia_a0_vs_a7.csv`.

- [ ] **S-11. Rerun the hypoglycemia alarm baseline with validation-tuned thresholds (Must).**
  `scripts/run_extra_baselines_round4.py` tunes the persistence and linear-extrapolation thresholds on the
  **test set** to match 2.21 false alarms/day, while the model's threshold (70 mg/dL) was fixed in advance.
  Choose each baseline's threshold on the **validation** split (same false alarms/day there as the model's alarm
  on validation), then report test event sensitivity, specificity, false alarms/day and median lead time.

- [ ] **S-16. Regenerate Table VIII (PRED-EGA) (Must).** The overall means match `pred_ega_summary.csv`, but:
  - Benign SD at 60/90/120 min: 7.0/8.1/8.2 in the paper vs 6.8/7.8/8.1 in the file.
  - Erroneous and Clinically Acceptable SD at 60/90/120 min: 0.28/0.39/0.44 vs 0.34/0.65/0.63.
  - Hypoglycemia Erroneous at 60/90/120 min: 0.13/0.24/1.59 vs 0.14/0.14/0.88.
  - Hypo/Hyper Accurate SDs are not in the file.
  Regenerate the whole table from the script and give the file.

- [ ] **S-17. Numbers with no result file (Should).** Point to the output for each, or regenerate:
  - Sec. III-G3 threshold analysis: T* = 84.0 mg/dL, F2 = 0.771, sensitivity 0.988, specificity 0.868,
    precision 0.175, 3,397 false positives; for G < 84: 0.943/0.908/0.451.
  - Table IX A0 event-level specificity (99.4%, 95.9%).
  - Sec. III-H partial correlations for whole-study mean glucose (r = −0.843, p = 0.0011; r = −0.875, p = 0.0004).
  - IG–permutation agreement: the paper says ρ = 0.731 (p = 6.05e−7); the current attribution CSVs give
    ρ = 0.740 (p = 3.8e−7).

## Dhriti — to be answered

- [ ] **D-3. Update the paper after Samyak's answers (Must).**
  - S-15: every 5-seed number (abstract, Sec. III-E1, Table VII, CIs, Discussion, Table II outcome) and Table IX.
  - S-11: add the alarm-baseline result to Sec. III-G1, and reword the hypoglycemia claims to match it (abstract
    alarm sentence and conclusion, contribution bullet 4, Discussion "Second, ...", Conclusion). If a simple
    alarm still matches the model at the same false-alarm rate, say so plainly.
  - S-16: Table VIII and the "0.13% at 60 min" sentence in Sec. III-F.
  - S-17: any numbers that change.
