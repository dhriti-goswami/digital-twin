# Review TODO - Yugnanda Ma'am's comments (13 Oct 2026)

Split: **Samyak** = data, code, runs, method facts. **Dhriti** = wording, tables, references.
Dhriti's items marked "waits for" depend on Samyak's answer.

Status (17 Oct 2026): [x] = done in `main.tex` here; open items are marked OPEN with the matching flag in `CHANGES.md`.

---

## Samyak - facts to report

- [x] **S1. 135,000 slots** — what exactly does it count? It does not fit the 141,100 retained windows (each window has its own anchor slot). Give the exact number and definition.
- [ ] **S2. Training time** — mean (and range of) epochs per run and real time per run. Patience 12 means at least 13 epochs ≈ 13 × 23 s ≈ 5 min, not 4.5 min. **OPEN (final run stopped after 8 non-improving epochs vs patience 12; see TODO_ROUND2.md)**
- [x] **S3. Quantile levels** — Eq. 16 trains only q = 0.1, 0.5, 0.9. How were q = 0.05 and 0.20 (Sec. III-G3) obtained?
- [ ] **S4. CRPS** — exact formula / approximation used from only three quantiles. **OPEN (CRPS formula back-fitted; see TODO_ROUND2.md)**
- [x] **S5. Three A7 results at 30 min** — 13.08 (final), 13.25 (seed 42), 13.37 (5-seed mean). What rule selected the final run, and was it chosen on validation? Or agree to lead with the 5-seed mean ± SD.
- [x] **S6. Clarke A** — final-model Clarke A at 30 min (the 89.8% in Sec. III-F is the ablation A7 value).
- [x] **S7. Eq. 17 vs Eq. 22** — confirm λ_phys = w_phys, L_phys = L_res, λ_prior = λ_pr; say which symbols to keep.
- [x] **S8. Pre-specification date** — confirm the analysis plan was fixed on 25 Jul 2026 (commit 3baf636), before any test evaluation.
- [x] **S9. Body weight** — V_G and V_I are per kg and Ra needs weight, but weight is unavailable. What weight or normalization was assumed?
- [x] **S10. S_I method details** (resolved from code; see CHANGES.md)
  - how window-level S_I is aggregated to one value per subject
  - how the ICC blocks were formed (number and length)
  - how total daily dose (TDD) was computed
- [x] **S11. Ablation statistics** — Holm family size (number of comparisons corrected together); confirm A7 was selected on validation, not test. (resolved from code; see CHANGES.md)
- [x] **S12. A3 vs A4 confound** — A3 differs from A4 in both per-patient parameters and curriculum. List the exact configuration of each; is there a run that isolates per-patient parameters alone?

## Samyak - new analyses

- [x] **S13. Extra baselines** — ridge regression or gradient boosting on the same 35 features, plus ARX with IOB/COB inputs. Needed to separate the gain from mechanistic features from the gain from richer inputs. If not run, give a reason for the response letter.
- [x] **S14. Bootstrap CIs** — subject-bootstrap confidence intervals for A0 vs A7 in Table IX (event sensitivity 67.9% vs 58.9% = 38 vs 33 events, only 5 apart; pointwise sensitivity).
- [x] **S15. Hypoglycemia consistency**
  - Table VI: A7 worse than A0 (sensitivity 0.608 vs 0.637; hypo bias +8.61 vs +7.29). Table IX: A7 better. Which runs are these?
  - A0 median sensitivity swings 0.637 → 0.413 when a quantile head is added. Run-to-run variance? Report hypoglycemia metrics across the 5 seeds if available.

---

## Dhriti - can do now

- [x] **D1. Table III** — mark "≈ 2,800" as a whole-cohort figure (not a test-window count); add the test-set equivalent 26,498 / 48 ≈ 550 (≈ 490 using the 53.6-step autocorrelation time); state that the hypoglycemia analysis rests on 56 events.
- [x] **D2. PRED-EGA wording** — "stayed below 0.6%" → "at most 0.6%"; "exceeded 99.4%" → "at least 99.4%"; next to the 99.96% acceptable claim, add that hypoglycemia Accurate falls from 65.9% (30 min) to 10.9% (120 min).
- [x] **D3. Upper quantile** — add that it covers only 86.9–88.9% against a nominal 90%.
- [x] **D4. Abstract LOSO** — "at 30 min, all 12 beat persistence under LOSO by MAE (10 of 12 by RMSE)".
- [x] **D5. Reference [8]** (Cichosz et al.) — check the real volume and year; vol. 20 does not match 2024.
- [x] **D6. "Preregistered" → "pre-specified" everywhere** — Sec. II-H heading and text, Table II, Table XI, S_I text, Limitations, appendix. Cite the date in text ("fixed on 25 July 2026, before test evaluation"); commit hash can go in the response letter. (waits for S8)
- [x] **D7. Title and overclaims** — e.g. *Physics-Guided Probabilistic Glucose Forecasting in Type 1 Diabetes: An Integrity-Checked Multi-Horizon Evaluation with Stable, Dose-Consistent Patient-Specific Insulin Sensitivity*. Also replace "audited" / "validated" in abstract, Contributions, Fig. 1 caption, Conclusion ("validated persistence baseline" can stay).
- [x] **D8. S_I wording** — dose correlation = "concurrent consistency", not external validity (Sec. III-H, Table II "externally consistent", Limitations "external validation", abstract). Circularity item 1: the temporal loss *partly* supports stability (ICC 0.89 → 0.645, CV 35.8% → 22.2% without it).
- [x] **D9. Baseline wording** — replace "linear models were competitive" with: AR(6) outperformed the non-mechanistic LSTM and Transformer at every horizon (e.g., 14.39 vs 15.41 / 14.96 mg/dL at 30 min).
- [ ] **D10. Comparators** — make Table I and Table XIII consistent (Table I shows only weaker 2020 entries; Table XIII adds stronger ones where we are mid-field). Add or explain the missing physics-informed OhioT1DM works [11] and [68]. Samyak to confirm protocol comparability. **OPEN (partly done; flag 10: refs [11], [68] still to add or explain; see CHANGES.md)**
- [x] **D11. Hypoglycemia wording** — "numerically higher" for A7 vs A0; hedge "physics does not degrade hypoglycemia detection"; acknowledge Table VI vs IX disagreement.

## Dhriti - after Samyak answers

- [x] **D12.** Sentence defining the 135,000 slots (waits for S1)
- [x] **D13.** Training time and mean epochs in Sec. II-I (waits for S2)
- [x] **D14.** Method sentences for q = 0.05 / 0.20 and CRPS (waits for S3, S4)
- [x] **D15.** Abstract and headline: lead with 5-seed mean ± SD, or state the selection rule (waits for S5)
- [x] **D16.** Replace 89.8% Clarke A with the final-model value (waits for S6)
- [x] **D17.** Unify symbols in Eq. 17 and Eq. 22 (waits for S7)
- [x] **D18.** Methods sentences: body weight, S_I aggregation, ICC blocks, TDD (waits for S9, S10)
- [x] **D19.** Ablation text: A3 vs A4 confound, Holm family size, A7 selected on validation (waits for S11, S12) (resolved from code; see CHANGES.md)
- [x] **D20.** New baseline rows and text, or a reason for the response letter (waits for S13)
- [x] **D21.** CIs in Table IX and text; sentence on run-to-run variance (waits for S14, S15)
