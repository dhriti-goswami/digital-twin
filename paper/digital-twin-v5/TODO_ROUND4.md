# Round 4 to-do (25 Oct 2026)

Combines the open round-3 items and the strict-reviewer report (M1–M8). Already done: S-5, S-6, S-7,
D-4, D-5, D-6, D-8 (see `CHANGES.md` Parts 10–11).

Priority: **Must** = needed before submission; **Should** = strengthens the paper, do if time allows.
Samyak's items come first, because several of Dhriti's depend on them.

---

## Samyak — to be answered

Updated 26 Oct 2026 after `SAMYAK_ANSWERS_ROUND4.md`. S-1, S-4, S-9 and S-10 are done and in `main.tex`;
S-11 needs one rerun; S-15 to S-17 come from a full cross-check of every number against the result files.

- [x] **S-1. Training and selection details.** Per-run patience, validation split and LOSO selection are now in Sec. II-I.
- [x] **S-4. Table VI hypoglycemia values.** Recomputed with `scripts/recompute_table_vi_hypo.py`; Table VI and
  Sec. III-G2 updated. The "1.3 mg/dL" sentence is unchanged (8.61 − 7.29).
- [x] **S-9. Physics effect CIs.** In Sec. III-E1. (A3/A4 5-seed replication not done; it was a Should.)
- [x] **S-10. Low-glucose calibration.** Conformal result and the linear quantile-regression baseline are in Sec. III-G4.

- [ ] **S-11. Hypoglycemia alarm baseline: rerun with validation-tuned thresholds (Must).**
  `scripts/run_extra_baselines_round4.py` tunes the persistence and linear-extrapolation thresholds on the
  **test set** to match 2.21 false alarms/day, while the model's alarm threshold (70 mg/dL) was fixed in
  advance. That favours the baselines. Rerun with each baseline's threshold chosen on the **validation** split
  (to give the same false alarms/day there as the model's alarm does on validation), then report test event
  sensitivity, specificity, false alarms/day and median lead time. The current result (persistence alarm
  67.9%, 17.5 min) is not in the paper yet, because its framing depends on this rerun.

- [ ] **S-15. A0 was handicapped in the 5-seed replication (Must, affects the main physics result).**
  `scripts/run_seeds_a0_a7.py` reuses the ablation run for seed 42 (A0 curriculum: warmup 2, ramp 6, so
  selection can start at epoch 3), but trains seeds 101–404 from `configs/official-small.yaml` defaults
  (warmup 10, ramp 25), so A0 cannot select before epoch 11. Your checkpoint table shows A0 seeds 101–404 chose
  epochs 13, 11, 11, 11, mostly the first epoch allowed, which suggests their best epoch came earlier.
  A7 could select from epoch 1 in every seed. From the means, A0's four retrained seeds average about 13.97
  (seed 42: 13.51), while A7's average about 13.40 (seed 42: 13.25), so the handicap may explain part of the
  0.51 mg/dL gain. A0 has no physics and no per-patient parameters, so its curriculum only delays selection.
  - Retrain A0 seeds 101–404 with selection allowed from the same epoch as seed 42 (or from epoch 1), about
    3 min per run, then redo Table VII, the Wilcoxon/Holm tests and the bootstrap CIs (`physics_effect_cis.csv`).
  - The quantile-head A0 in Table IX has the same issue (warmup 10, ramp 25, best epoch 13): retrain it the
    same way and redo `hypoglycemia_a0_vs_a7.csv`.

- [ ] **S-16. Table VIII (PRED-EGA) does not match `pred_ega_summary.csv` (Must).** The means of the overall rows
  match, but:
  - Benign SD at 60/90/120 min is 7.0/8.1/8.2 in the paper vs 6.8/7.8/8.1 in the file (the paper's values equal
    the Accurate SDs).
  - Erroneous and Clinically Acceptable SD at 60/90/120 min: 0.28/0.39/0.44 in the paper vs 0.34/0.65/0.63.
  - Hypoglycemia Erroneous at 60/90/120 min: 0.13/0.24/1.59 in the paper vs 0.14/0.14/0.88.
  - The Hypo/Hyper Accurate SDs are not in the file, so they cannot be checked.
  Regenerate the whole table from the script and give the file.

- [ ] **S-17. Numbers with no result file (Should).** Point to the output for each, or regenerate:
  - Sec. III-G3 threshold analysis: T* = 84.0 mg/dL, F2 = 0.771, sensitivity 0.988, specificity 0.868,
    precision 0.175, 3,397 false positives, and the G < 84 values 0.943/0.908/0.451.
  - Table IX A0 event-level specificity (99.4%, 95.9%); only A7's is in `event_hypoglycemia_summary.csv`.
  - Sec. III-H partial correlations for whole-study mean glucose (r = −0.843, p = 0.0011; r = −0.875, p = 0.0004).
  - IG–permutation agreement: paper and `docs/RESULTS.md` say ρ = 0.731 (p = 6.05e−7), but the current
    attribution CSVs give ρ = 0.740 (p = 3.8e−7). Which run is right?

## Dhriti — to be answered

- [ ] **D-3. After S-11, S-15 and S-16 (Must).** Add the alarm-baseline result to Sec. III-G1, and reframe the hypoglycemia
  claims to match it: abstract (alarm sentence and conclusion), contribution bullet 4, Discussion ("Second, ...")
  and Conclusion. If a simple persistence alarm still matches the model's event sensitivity at the same
  false-alarm rate, say so plainly. After S-15, update every 5-seed number (abstract, Sec. III-E1,
  Table VII, Discussion, Table II outcome) and Table IX; after S-16, Table VIII and the "0.13% at 60 min" sentence. Already done: S-1, S-4, S-9, S-10 integrated; "calibrated" softened.

- [x] **D-9. Novelty and framing.** Done: novelty paragraph at the end of Sec. I-B; digital-twin, RCT and
  "parallel tracks" material cut to one sentence (all citations kept).
- [x] **D-10. Overclaims.** Done: abstract (post hoc, same test set; "small accuracy gains in a post hoc
  replication"), "calibrated" softened, "Mean glucose", attribution softened, clinical meaning of the gain in the
  Discussion, S_I evidence stated as internal in Limitations, Conclusion limited to this cohort.
- [x] **D-11. Table II outcomes.** Done: "Outcome" column.
- [x] **D-12. Minor text and layout.** Done: Sec. II-B clause, Table XIII removed (its LOSO and 2018 points
  are now in Sec. III-J), captions of Tables IV, VI, VII and IX shortened, workshop-paper note in the Table I
  footnote. Not done: moving Tables III and XIV to the supplement (not needed for space).

- [ ] **D-13. Ref [11] year (Must).** Completed from the publisher page (sciopen.com): Tsinghua Sci. Technol.,
  vol. 32, no. 1, pp. 549–561, doi 10.26599/TST.2025.9010140. The page's citation says 2027 but its listed
  publication dates are 2026, so the reference reads "2027 (published online 2026)". Confirm the year.
