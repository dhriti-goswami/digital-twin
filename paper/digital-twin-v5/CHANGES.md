# Changes for revision round 2 (Yugnanda Ma'am's comments, 13 Oct 2026)

Base: `sammyyakk/digital-twin` main @ `9da054d` (16 Oct 2026), `paper/digital-twin-v4/main.tex`, copied into `paper/digital-twin-v5/` with the edits below.
Every number added comes from Samyak's answers (`paper/digital-twin-v5/SAMYAK_ANSWERS.md`,
`paper/digital-twin-v4/DHRITI_REVISION_V2.md`) or from his result files in `results/tables/`.
Nothing was rerun or estimated. The paper compiles with no errors and no overfull lines (17 pages).

---

## Part 1 — Changes made in `main.tex` and `supplementary.tex`

### D7. Title and "audited" / "validated"
- **Title:** "An Audited Multi-Horizon Evaluation with Validated Patient-Specific Insulin Sensitivity"
  → "An Integrity-Checked Multi-Horizon Evaluation with Stable, Dose-Consistent Patient-Specific Insulin Sensitivity".
- "audited" → "integrity-checked" in the abstract (Background), Contributions bullet 2, Fig. 1 caption and Conclusion.
- Fig. 1 final box: "Validated Patient Parameters" → "Patient-Specific Parameters".
- Sec. III-H heading: "…and Physiological Validation" → "…and Validity Checks".
- "Validated persistence baseline" kept (it means something different).

### D4 and D15. Abstract Results
- LOSO claim now gives the horizon: "under LOSO all 12 still beat persistence at 30 min by MAE (10 of 12 by RMSE)".
- Physics result now leads with the 5-seed mean ± SD: 30-min MAE 13.37 ± 0.11 vs 13.88 ± 0.25 mg/dL, 2.38 mg/dL lower at 120 min,
  Holm p = 0.002; the pre-specified single-seed test (13.25 vs 13.51, p = 0.105) follows. Source: Table VII.
- Abstract Methods: "checks tested the validity of S_I" → "checks tested whether S_I was subject-specific, stable, and dose-consistent".
- The provenance of 13.08 / 13.25 / 13.37 was already stated in the Table IV, V and VII captions; unchanged.

### D1 and D12. Effective sample size
- Sec. II-B: the 135,000 slots are now described as whole-cohort slots, noting that the 141,100 retained windows overlap and share them.
- Sec. II-B: added the test-set figures, 26,498/48 ≈ 552, or ≈ 494 with the 53.6-step autocorrelation time.
- Table III: row renamed "Effective independent windows, whole cohort (≈ 2,800)"; new row "Effective independent test windows ≈ 552 (≈ 494)".
- Sec. III-G1 (56 events): added that 56 events on ≈ 552 (494) effective test windows carry substantial uncertainty.
- Limitations: "…about 2,800 effectively independent observations, of which only about 550 fall in the test set; the hypoglycemia analysis rests on 56 events".

### D6. "Preregistered" → "pre-specified" (all 8 places)
- Sec. II-H heading, Table II caption, Sec. III-H text, Table XI caption, Sec. III-J (comparison), Limitations (two places), Appendix.
- Sec. II-H now opens: "The outcomes and criteria below were fixed on 25 July 2026, before any test-set evaluation."
  (Date from S8. The repo link and commit hash are not in the paper, as decided earlier; give them in the response letter.)

### D8 and D18. Insulin sensitivity wording and missing methods
- Table II row "S_I is externally consistent" → "S_I is dose-consistent".
- Sec. III-H: the dose relation is now called concurrent consistency, not external validity, with the reason
  (the parameter head sees the insulin inputs and TDD comes from the same records).
- Sec. III-H: added the ICC method (4 disjoint time-ordered blocks per subject, block medians, ICC(1) across subjects)
  and the TDD method (total recorded insulin ÷ recorded days, U/day). Source: S10.
- Circularity item 1: "not produced solely by the penalty" → "partly, but not solely, attributable to the penalty:
  removing it lowered ICC(1) from 0.890 to 0.645 and CV from 35.8% to 22.2%".
- Table XII group header "External Validation against Classic Minimal Model" → "Agreement with Classic Minimal Model".
- Body weight (Sec. III-H and Limitations): a fixed 70 kg is used for every subject for V_G, V_I and R_a. Source: S9.
- Limitations: "LOSO external validity fails" → "the dose relation fails under LOSO"; "external validity of one
  physiological quantity" → "stability and dose consistency of one physiological quantity".

### D13. Training time (Sec. II-I)
- "about 23 s per epoch (about 4.5 min per run)" → "Across the seven training runs reported here (A0–A4, A7 and the
  final quantile-head model), training stopped after 11–20 epochs (mean 16.3), at 10.7–23.8 s per epoch, or 2.7–6.4 min per run."
- Source: `results/tables/training_time_summary.csv`, **excluding** the `small-fast` run (see flag 1).

### D14. Quantile levels and CRPS
- Sec. III-G3: states that only q = 0.1, 0.5, 0.9 are trained, and that q = 0.05 and 0.20 come from a Gaussian
  left-tail extrapolation, with the formula. Source: S3.
- Sec. III-G4: states that CRPS is approximated from three quantiles, with the formula Samyak gave (see flag 2).

### D3. Upper quantile coverage (Sec. III-G4)
- Added: the q90 quantile is under-covered at every horizon (88.9%, 88.1%, 87.4%, 86.9% vs nominal 90%). Source: Table X.

### D2 and D16. PRED-EGA and Clarke A
- "exceeded 99.4%" → "at least 99.4%"; "stayed below 0.6%" → "at most 0.6%, reaching that value at 120 min".
- Next to the 99.96% acceptable claim: hypoglycemia-range Accurate is only 65.9% at 30 min, falling to 10.9% at 120 min (Table VIII).
- Clarke A 89.8% kept, now labelled as the final model's per-subject mean (Fig. 5). See flag 3.

### D11 and D21. Hypoglycemia, Tables VI vs IX
- "A7 had higher / more sensitive" → "numerically higher / numerically more sensitive", with 38 vs 33 of 56 events.
- Added subject-bootstrap 95% CIs (5,000 resamples): A7 [0.585, 0.750], A0 [0.453, 0.726], difference [−0.050, 0.235],
  which includes zero, so not significant. Source: S14.
- "physics does not degrade hypoglycemia detection" → "no evidence that physics degrades … but no significant improvement either".
- Table IX caption: same three CIs added.
- New paragraph: in the ablation, A7 was worse than A0 (0.608 vs 0.637; bias +8.61 vs +7.29); A0 swings 0.637 → 0.413
  once a quantile head is added; Tables VI and IX are different runs and should not be compared directly. Source: S15.

### D9 and D20. Baselines (Sec. III-C, Table V)
- Two rows added to Table V: ARX(6) + IOB/COB and Ridge (35 features), all horizons, MAE and RMSE ± SD.
  Source: `results/tables/extra_baselines.csv`. A7 is still lowest in every column.
- Model list: "nine models" → "eleven models", with the two new models described.
- "Linear models were competitive" replaced: AR(6) beats both non-mechanistic neural baselines at every horizon;
  the two new linear baselines match A0 at 30 min (13.56, 13.51), so most of the 30-min gain comes from the richer inputs;
  A7 is 0.26–0.31 mg/dL lower at 30 min, and 1.34 / 2.34 / 3.09 mg/dL lower than Ridge at 60 / 90 / 120 min;
  these differences were not tested for significance. See flag 6.

### D19. Ablation
- Sec. II-F: A7 "selected on validation data rather than the test set". Source: S11.
- Table VI: A4 label "Hybrid, fixed population parameters" → "Hybrid prior, curriculum, population parameters";
  caption adds "A3 and A4 differ only in per-patient parameterization; physics weighting and curriculum are held fixed". Source: S12.
- Sec. III-E1: Holm correction "across the four horizons, applied separately to MAE and RMSE". Source: S11.

### D5. Reference [8] (Cichosz et al.)
- "vol. 20, no. 3, pp. 836–847, 2024" → "vol. 20, no. 3, pp. 836–847, 2026 (published online 8 Oct. 2024), doi: 10.1177/19322968241286907".
- Volume, issue, pages, DOI and online date are from the article's ResearchGate record. See flag 9 for the print year.

### D17. Equation symbols
- Eq. 22: w_phys L_res → λ_phys L_phys; λ_pr → λ_prior, matching Eq. 17.
- Supplement loss table: L_res → L_phys, w_phys → λ_phys.

### D10. Comparators
- Table I footnote now points to Table XIII: "Stronger 2020 challenge entries (RMSE@30 of 17.45–19.05 mg/dL) are
  compared in Table XIII, where the current work is mid-field." (Numbers from Table XIII.)

### Fixes settled from Samyak's code (no new numbers)
- **Early stopping (Sec. II-I):** "validation MAE averaged across horizons" → "30-min validation MAE". The code
  (`twin/train/loop.py`) selects and stops on `val_mae_30`.
- **Final run (Sec. II-I):** added "The final quantile-head run stopped earlier, after 11 epochs, with its best
  validation MAE at epoch 3." Source: `results/tables/training_history.csv`. The reason is open (TODO_ROUND2.md S-1).
- **Subject-level S_I (Sec. III-H, flag 7 resolved):** "Each subject's S_I is the median of its window-level estimates;
  the between-subject CV and the dose correlation are computed across these 12 subject medians."
  Source: `twin/eval/sensitivity.py` (`s_i_median`).
- **Ablation Holm family (Physics Ablations, flag 5 resolved):** "Holm correction was applied across the five
  arm-versus-A0 comparisons (A1–A4 and A7) on 30-min MAE." Source: `twin/eval/results_doc.py`; check: 0.021 × 5 = 0.105.
- **A4 curriculum (flag 8 resolved):** `twin/train/ablations.py` shows A3 and A4 differ only in `per_patient_params`;
  both keep the default curriculum (only A7 removes it). The A4 label is correct.

### Layout
- The appendix still starts on a new page (page 15) with Table XIV on top; the conclusion stays on page 14.
  The page break now works whichever column the conclusion ends in, and the space around floats was reduced
  slightly (10 pt) so the conclusion fits on its page. 17 pages, no errors, no overfull lines.

---

## Part 2 — Flags: what is wrong or still open

**For Samyak**

1. **Training time (S2). Partly resolved.** His mean (16.4 epochs, 20.4 s/epoch, 5.5 min/run) includes `small-fast`, a run that is not
   in the paper (41 s/epoch, 11.7 min). The paper now uses only the seven paper runs. Also, the final quantile-head run
   stopped after **11 epochs**, but Sec. II-I says early stopping uses a patience of 12, which implies at least 13 epochs.
   Both his answer ("floor of 13") and Ma'am's premise conflict with this run. The results file shows the best epoch was 3,
   so it stopped after 8 epochs without improvement. The paper now states this fact; the reason is in TODO_ROUND2.md S-1.
2. **CRPS formula (S4). Open, TODO_ROUND2.md S-2.** His script says the formula was back-fitted to the existing numbers. 0.5·pinball_0.1 + pinball_0.5 + 0.5·pinball_0.9 reproduces the paper's values, but it is not
   the standard quantile approximation: integrating the pinball loss over quantile levels with the trapezoid rule gives
   weights 0.5 / 0.8 / 0.5 (8.43 instead of 9.74 at 30 min). The paper states his formula as used. He should confirm it is
   intentional, or recompute CRPS and Table X.
3. **Clarke A (S6).** His 89.86% is pooled across windows. Fig. 5 and the text use the per-subject mean, which for the
   final model is 89.80% (`summary_model.csv`). So the paper's 89.8% was already the final model's value; Ma'am's premise
   was mistaken. The paper keeps 89.8% and now labels it. Use this in the response letter.
4. **S8.** He said the commit is "already cited at main.tex:455". It was not: that paragraph was removed earlier. The paper
   now gives only the date (25 July 2026).
5. **Holm family for the ablation (S11). Resolved from code: 5 comparisons.** "4 horizons" fits the 5-seed test (0.00049 × 4 = 0.00195). The single-seed
   ablation result (p = 0.105) compares several physics arms with A0, so its family may differ. Not added to the paper; he must state it.
6. **New baselines (S13).** His conclusion "only ~0.3 mg/dL gap remains" holds at 30 min only. At 60–120 min A7 is
   1.3–3.1 mg/dL better than Ridge. The paper states this horizon by horizon. No significance test exists for A7 vs Ridge or ARX+IOB/COB.
7. **Subject-level S_I (S10). Resolved from code: median of window estimates.** He described the ICC blocks, but not how the single per-subject S_I used for CV and the
   dose correlation is obtained (median of all windows?). Only the ICC method was added; he must confirm the rest.
8. **A3 vs A4 (S12). Resolved from code: same curriculum.** His code check says they differ only in per-patient parameters, so the A4 label now says "curriculum".
   He should confirm A4 really uses the same curriculum as A3, because the label now states it.

**For Dhriti**

9. **Reference [8] print year.** The record gives vol. 20, no. 3 and online publication on 8 Oct 2024, but not the print year.
   2026 was inferred from the journal's volume numbering (vol. 1 = 2007). Check the issue year on the journal site; if you
   would rather not rely on it, cite the online-first form instead ("2024, doi: …" without vol./no./pp.).
10. **Comparators [11] and [68] (D10, partly done).** Only the pointer from Table I to Table XIII was added.
    - [68] Gu et al. (EMBC 2020): their lab page reports OhioT1DM, 6 subjects, 30-min RMSE 17.80 mg/dL (no MAE, no 60 min),
      but not which cohort (2018 or 2020). Check the paper's dataset section, then add the row to Table I.
    - [11] Wang et al. (DDCLS 2025): paywalled; dataset unknown. Check via college IEEE access; add a row if it reports
      official-split OhioT1DM results, otherwise add a footnote saying why it is omitted.
11. **Response letter.** Explain the points where the reviewer's premise was not quite right: Clarke A (flag 3) and A3/A4 (flag 8).
