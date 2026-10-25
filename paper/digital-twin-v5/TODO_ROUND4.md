# Round 4 to-do (25 Oct 2026)

Combines the open round-3 items and the strict-reviewer report (M1–M8). Already done: S-5, S-6, S-7,
D-4, D-5, D-6, D-8 (see `CHANGES.md` Parts 10–11).

Priority: **Must** = needed before submission; **Should** = strengthens the paper, do if time allows.
Samyak's items come first, because several of Dhriti's depend on them.

---

## Samyak — to be answered

- [ ] **S-1. Training and selection details (Must, M4).**
  - Load the quantile run's `best_model.pt` and give `config["train"]["early_stopping_patience"]` and
    `best_epoch` (the saved config there was not overwritten by `report`).
  - For A0–A4, A7 and the 5-seed runs, give: warmup, ramp end, patience, best epoch, stop epoch. With the
    default curriculum, A0 and A1–A4 could not stop before epochs 21 and 36, but they stopped at 15–20.
  - Describe how the validation set is formed in the official protocol (chronological? purge gap?), and how
    early stopping and selection work inside each LOSO fold.

- [ ] **S-4. Table VI hypoglycemia values (Must, M5).** None of the six hypo-sensitivity values (0.637,
  0.502, 0.499, 0.485, 0.517, 0.608) can be reproduced from `results/ablations/per_subject_A*.csv` (about
  0.015 lower for every arm). Point to the script, or recompute all six plus the hypo bias with one
  documented method. The Discussion's "1.3 mg/dL" depends on it.

- [ ] **S-9. Physics effect: uncertainty (Must, M2).** Give 95% CIs (subject bootstrap) for the A7 − A0 MAE
  difference at each horizon from the 5-seed runs. Should: replicate A3 and A4 over the same 5 seeds.

- [ ] **S-10. Low-glucose calibration (Must, M6).** Below 80 mg/dL, the 80% interval covers 76.4% (30 min)
  to 53.3% (120 min), and q0.10 is above the truth 22.4% of the time. Try conformal recalibration of the
  quantiles on the validation set, and report conditional coverage and alarm sensitivity/precision after it.

- [ ] **S-11. Missing baselines (Must, M7).** On the same 26,498 test windows:
  - **Hypoglycemia alarm baseline:** an alarm from persistence or linear extrapolation of the recent glucose
    trend, tuned to the same false alarms per day (about 2.2). Report event sensitivity and lead time.
  - **Probabilistic baseline:** linear quantile regression on the 35 features (or the LSTM with a quantile
    head). Report coverage, interval width and CRPS as in Table X.
  - Should: one modern point-forecast model (N-BEATS or Temporal Fusion Transformer) on the same windows.

## Dhriti — to be answered

- [ ] **D-3. After Samyak answers (Must).**
  - Sec. II-I: early-stopping sentence and a new sentence on the validation split and LOSO selection (S-1).
  - Table VI and the Discussion's "1.3 mg/dL" sentence (S-4).
  - Add the CIs (S-9), recalibration results (S-10) and new baselines (S-11).

- [ ] **D-9. Novelty and framing (Must, M1).**
  - Add a short paragraph (or a feature table) stating exactly what this work does that [9]–[13] do not.
  - Cut the digital-twin and RCT material in Sec. I to one or two sentences; the paper is a forecaster.

- [ ] **D-10. Overclaims (Must).**
  - Abstract: say the 5-seed replication was post hoc and reused the test set.
  - Abstract conclusion: "small but consistent accuracy gains" → "small accuracy gains in a post hoc replication".
  - "Calibrated predictive tails": soften unless S-10 fixes low-glucose calibration.
  - Sec. III-H: "Ambient glucose does not explain" → "Mean glucose does not explain".
  - Sec. III-I: soften "informative in their own right" (permutation importance is unreliable with correlated features).
  - Discussion: say what a 0.5 mg/dL gain at 30 min means clinically (M2).
  - Sec. III-H or V: state plainly that the evidence for S_I is internal only (stability and dose
    consistency within this cohort, with a weak external reference) and that its usefulness, e.g. for
    insulin dosing, has not been tested (M3).
  - Check the Discussion and Conclusion keep every claim to this cohort (M8).

- [ ] **D-11. Reporting (Must).** Add a "Met / Not met" column to Table II (e.g. beats persistence: met, 12/12;
  field-competitive: met; physics improves accuracy: not met; S_I checks: met under the official protocol).

- [ ] **D-12. Minor text and layout (Should).**
  - Sec. II-B: one clause on how excluding the 2020 cohort's first test hour affects comparison with the 2018 rows.
  - Keep q = 0.05 and 0.20 labelled as approximate wherever they appear.
  - Table IV caption: one line, once the headline number is settled.
  - Consider dropping Table XIII (move its LOSO row into Table I or the text), and moving Tables III and XIV
    to the supplement.
  - Shorten long captions (Tables IV, VI, VII, IX) by moving explanations into the text.
  - Say once that most OhioT1DM comparators are BGLP workshop papers.

- [ ] **D-13. Ref [11] year (Must).** Completed from the publisher page (sciopen.com): Tsinghua Sci. Technol.,
  vol. 32, no. 1, pp. 549–561, doi 10.26599/TST.2025.9010140. The page's citation says 2027 but its listed
  publication dates are 2026, so the reference reads "2027 (published online 2026)". Confirm the year.
