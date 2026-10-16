# Round 2 — remaining to-do (17 Oct 2026)

Everything else from `todo.md` is done in `main.tex` (see `CHANGES.md`). Only the items below are open.

---

## Samyak — to be answered

- [ ] **S-1. Final model stopped early.** `results/tables/training_history.csv` (the quantile-head run, 11 epochs):
  best 30-min validation MAE at **epoch 3** (14.33 mg/dL), training stopped at **epoch 11**, i.e. after **8**
  epochs without improvement. `configs/official-small.yaml` sets `early_stopping_patience: 12`, and the code
  default (`twin/config.py`) is 15. Why did this run stop after 8? Different patience, a different config,
  a manual stop, or something else?
  The paper currently states only the fact: "The final quantile-head run stopped earlier, after 11 epochs,
  with its best validation MAE at epoch 3."

- [ ] **S-2. CRPS.** `scripts/compute_crps.py` says the formula `0.5·pinball_q10 + pinball_q50 + 0.5·pinball_q90`
  was found by least-squares fitting to the existing `crps` column, so the code that originally produced the
  CRPS values is not in the repo. The derivation in that docstring also does not reduce to the stated formula:
  `2·[0.5·p10 + 0.5·(p10+p50)/2 + 0.5·(p50+p90)/2]` = `1.5·p10 + p50 + 0.5·p90`.
  Point to the original CRPS code, or recompute CRPS (Table X) with a documented approximation.
  The paper currently states the fitted formula.

- [ ] **S-3. Outdated docs.** `docs/TRAINING_METHODOLOGY.md` says early-stopping patience is 15 and reports
  30-min MAE 5.53 mg/dL, neither of which matches the paper. Delete it or mark it as legacy.

- [ ] **S-4. Tables V and VI use different runs from the rest of the paper, so their captions have grown too long.**
  Both captions now carry long explanations instead of a short description (Table V ~300 characters, Table VI ~540):
  - **Table V:** the A7 row is the single-seed ablation run (13.25 mg/dL at 30 min), not the final model reported in
    Table IV (13.08). Readers see two different "A7" results.
  - **Table VI:** its hypoglycemia columns (A0 0.637, A7 0.608) come from the ablation runs without a quantile head,
    while Table IX reports A0 0.413 and A7 0.551 from the quantile-head models, so A7 looks worse than A0 in one
    table and better in the other.
  Please fix this at the data level so the captions can be cut back to one line each, for example by reporting
  the A0 and A7 rows of Tables V and VI from the same runs used elsewhere (the 5-seed means of Table VII, or the
  final model with a clearly labelled row), and taking Table VI's hypoglycemia columns from the same models as
  Table IX. Say which runs you use, and Dhriti will shorten the captions to match.

---

## Dhriti — to be answered

- [ ] **D-1. Refs [11] and [68] in Table I.** Table I now points to Table XIII; these two are still missing.
  - [68] Gu, Dang & Prioleau, "Neural physiological model", EMBC 2020 (doi 10.1109/EMBC44109.2020.9176004).
    Their lab page reports OhioT1DM, 6 subjects, 30-min RMSE 17.80 mg/dL (no MAE, no 60 min). Check which
    cohort (2018 or 2020) in the paper, then add a row to the physics-informed group of Table I; if 2018, add
    [68] to the footnote's 2018-cohort exception next to [35].
  - [11] Wang, Zhang & Song, DDCLS 2025 (IEEE document 11064966). Paywalled; open via college IEEE access.
    If it reports official-split OhioT1DM results, add a row; if not, add to the Table I footnote:
    "Physics-informed studies that do not report results on the official OhioT1DM split (e.g., [11]) are
    discussed in Section I-B but omitted here because their metrics are not comparable."

- [ ] **D-2. Ref [8] print year.** Now "vol. 20, no. 3, pp. 836–847, 2026 (published online 8 Oct. 2024), doi …".
  2026 is inferred from the journal's volume numbering; confirm the issue year on the journal site.

- [ ] **D-3. After Samyak answers:** update the Sec. II-I early-stopping sentence (S-1) and, if CRPS is
  recomputed, the CRPS sentence in Sec. III-G4 and the CRPS column of Table X (S-2).
