# Round 3 to-do (20 Oct 2026)

From a CMPB-style review of the current `main.tex`. Done so far: see `CHANGES.md`.
Order: Samyak's items first, because several of Dhriti's depend on them.

---

## Samyak — to be answered

- [ ] **S-1. Early stopping (major).** Sec. II-I says patience 12, but the final run's best epoch is 3 and it
  stopped at 11 (8 epochs later). The final model uses the A7 settings, so selection ran from epoch 1
  (`training_history.csv` shows `physics_ramp = 1.0` at epoch 1). Still unexplained:
  - The run stopped after 8 non-improving epochs, not 12.
  - A0–A4 also stopped too early for patience 12 to have fired.

- [ ] **S-4. Tables V and VI (major).** The harmonized values are not applied, because:
  1. Table V's new A0/A7 "±" is across seeds; every other row is across subjects.
  2. Table V (A7 = 13.37) would contradict Table VI (A7 = 13.25).
  3. Table VI's hypoglycemia column would mix quantile-head values (A0, A7) with old values (A1–A4),
     and none of the six old values can be reproduced from `per_subject_A*.csv`.

  Note: the Discussion's "adds approximately 1.3 mg/dL" hypoglycemic bias is 8.61 − 7.29 from Table VI,
  so it changes with this fix. Give the new value or say it should be removed.

- [ ] **S-5. Headline number (major).** The paper reports three 30-min MAEs for the same model: 13.08
  (final model, Table IV), 13.25 (seed 42, Tables V–VI) and 13.37 (5-seed mean, Table VII). After S-1,
  say which one the abstract and conclusion should lead with.

- [ ] **S-6. S_I reference fits (minor).** S_I is checked against per-subject Bergman NLS/MAP fits
  (ρ = 0.79), but the same fits forecast poorly (30-min MAE 25.80). Give their fit quality on the
  calibration intervals, or a reason they are still a valid S_I reference.

- [ ] **S-7. Final hyperparameters (major).** Sec. II-I lists the search space but not the chosen values.
  Confirm the final model's values (`configs/official-small.yaml` has d_model 64, 2 layers, 4 heads,
  d_ff 128, dropout 0.2, attention pooling) and that the search used validation data only.

## Dhriti — to be answered

- [ ] **D-3. After Samyak answers:**
  - Update the Sec. II-I early-stopping sentence (S-1).
  - Update Tables V and VI, their captions, the Sec. III-C baseline paragraph, Fig. 4, the
    Table VI vs IX paragraph and the Discussion's "1.3 mg/dL" sentence (S-4).
  - Use one headline number in the abstract and conclusion (S-5).

- [ ] **D-4. Title (major).** "Stable, Dose-Consistent ... Insulin Sensitivity" overclaims: the dose link fails
  under LOSO and is concurrent, not external. Suggested: "...with a Stable Patient-Specific
  Insulin-Sensitivity Estimate".

- [ ] **D-5. Comparison section (major).**
  - Sec. III-J cites a 2020 entry with RMSE@30 18.34 (Zhu et al., GAN) that beats the current work but is not in
    Table I or XIII. Add it from the paper itself (to keep Table I's length, drop the weakest row).
  - Sec. III-J still describes Tables I and XIII as 2020-only; Table I now has 2018 rows.
  - Reword the remark that one entry's "MAE-to-RMSE ratio is far outside the range" neutrally, or remove it.
  - Decide whether Table XIII is still needed, now that Table I has the same entries (keep only for the LOSO row?).

- [ ] **D-6. Wording (minor).**
  - Fig. 4 caption: the 5-seed test confirms the 5-seed difference (13.37 vs 13.88), not 13.25 vs 13.51.
  - Abstract conclusion: "Calibrated predictive tails ... determine hypoglycemia detection" → add "in this cohort".
  - Hypoglycemia alarm (precision 0.347, 2.2 false alarms/day): add one sentence on clinical acceptability,
    with a citation.

- [ ] **D-7. Submission requirements (required).**
  - Check whether CMPB accepts the IEEE template at first submission; if not, move to Elsevier's template.
  - Add the declarations: author contributions (CRediT), competing interests, funding, data availability,
    ethics statement for a public de-identified dataset, generative AI use.
  - Write 3–5 highlights (each at most 85 characters).
  - Confirm Ma'am's name: author line says "Yugananda", email says "yugnanda".
  - Check CMPB's abstract word limit; the abstract is about 375 words.
  - Reference style: if moving to the Elsevier template, reformat references (11 entries use "et al."
    instead of full author lists).
  - Write the cover letter, and the response letter to Ma'am's round-2 comments (from `CHANGES.md`).
  - Supplement: its title was still the old one ("Audited ... Validated"); now updated. If the title
    changes again (D-4), change it in both files.

- [ ] **D-8. Feature names in Fig. 7 (minor).** The y-axis uses code names (`roc_5min`, `minutes_since_bolus`,
  `is_night`). Replace them with readable names (the `yticklabels` line in `main.tex`), e.g. "Glucose rate
  of change, 5 min".
