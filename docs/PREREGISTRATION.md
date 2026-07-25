# Pre-registration

Written **before** the two experiment branches are run and before any test-set
number from either is inspected. Its purpose is to remove the discretion that turns
two experiments into one cherry-picked result.

Git commit at time of writing: see `git log` for the commit that added this file.
Any change to this document after a branch's test metrics have been seen must be
recorded as an amendment at the bottom, with the reason.

---

## 1. Why this exists

The plan runs two tuning targets from the identical pipeline:

- **`exp/physio-fidelity`** — optimise physiological fidelity: insulin-sensitivity
  identifiability and trajectory plausibility.
- **`exp/accuracy-60min`** — optimise 60-minute point accuracy against the best
  published value.

These genuinely trade off: a heavier physics weight buys `S_I` stability and may
cost point accuracy. Running both is legitimate. **Running both and publishing
whichever wins is not** — with two branches and four horizons there are enough
degrees of freedom to find a favourable comparison by chance, and the reader has no
way to know how many were tried.

So the commitments below are fixed now.

---

## 2. Primary outcome, declared in advance

**Primary:** 30-minute MAE on the OhioT1DM test set under the `official` protocol,
reported as mean ± SD across the 12 subjects, from branch **`exp/physio-fidelity`**.

`physio-fidelity` is primary because the paper's contribution is the physiology and
the patient-specific parameter estimate. Point accuracy at 30 minutes is *not* a
novel contribution — 15 of 17 published entries already clear MAE < 15 mg/dL, and
persistence alone reaches 16.36. Making accuracy primary would elevate the least
novel part of the work.

**Secondary, all reported regardless of outcome:**

1. MAE and RMSE at 60, 90 and 120 minutes, both protocols, both branches.
2. Skill score against persistence at every horizon.
3. Clarke and Parkes zone distributions.
4. The three `S_I` validation checks (§4).
5. The ablation matrix A0–A4.
6. The learned mechanistic-prior gate `g`.

---

## 3. What counts as success, stated numerically

Declared now so it cannot be redefined after the fact.

| Claim | Threshold | Basis |
|---|---|---|
| Beats the naive baseline | 30-min MAE below persistence's 16.36 mg/dL, in **≥ 10 of 12** subjects | persistence measured by this pipeline, validated against two published RMSE values to within 0.3 mg/dL |
| Meets the field standard | 30-min MAE < 15 mg/dL | the field median; explicitly **not** claimed as novel |
| Competitive with published work | 30-min MAE ≤ 13.5 mg/dL | best credible published is 12.83 (Rubin-Falcone et al.) |
| Physics earns its place | A3 beats A0 on 30-min MAE, and the difference survives a paired Wilcoxon test across subjects with Holm correction | the ablation matrix |
| `S_I` is subject-specific | between-subject CV of median `S_I` > 10%, **and** test-retest ICC(1) > 0.5 | a collapsed parameter head would otherwise pass the correlation checks trivially |
| `S_I` is externally valid | Spearman ρ with total daily dose negative, 95% bootstrap CI excluding zero | n = 12, underpowered by construction |

**Anything not met is reported as not met.** In particular, if A3 does not beat A0,
the paper states that the physics term did not improve point accuracy on this
dataset. That is a publishable negative result about physics-informed forecasting,
and it does not invalidate the physiological contribution — the mechanistic
trajectory and `S_I` estimate stand or fall on §4, independently.

---

## 4. Insulin sensitivity: what would falsify the claim

`S_I` is reported as a patient-specific physiological estimate **only if all three**
hold:

1. **Not degenerate.** Between-subject CV of median `S_I` > 10%. If the parameter
   head has collapsed to one value for every subject, `S_I` is not subject-specific
   and will not be described as such, regardless of any correlation.
2. **Stable.** Test-retest ICC(1) > 0.5 across disjoint time blocks within subject.
   An estimate that varies window to window is absorbing prediction error, not
   measuring a parameter.
3. **Externally consistent.** Spearman ρ against total daily dose is negative with a
   bootstrap CI excluding zero.

If (1) or (2) fails, `S_I` is reported as **not validated** and the corresponding
claim is removed from the paper, not softened.

Known limitations, fixed now so they are not discovered conveniently late:

- **The carbohydrate-ratio correlation is exploratory, n = 6.** Bolus-wizard
  carbohydrate entries exist only for the 2018 cohort; all six 2020 subjects have
  none. Spearman on six points needs |ρ| > 0.83 for p < 0.05. It will be labelled
  exploratory whatever it shows.
- **`tdd_per_kg` is not an independent variable.** Body weight is unidentifiable
  from OhioT1DM (every file records the placeholder 99 kg), so dividing by a
  constant nominal weight is a rescaling of TDD. It will not be presented as a
  second correlation.
- **`S_I` is partly range-enforced.** It is bounded by construction through `p3` and
  `p2`, so lying in a plausible range is partly imposed rather than learned. Stated
  wherever the range is reported.
- **Subject 567's test period contains no carbohydrate records**, so COB is
  identically zero there and the physics is structurally degraded for that subject.
  It stays in the cohort; its per-subject row is footnoted.

---

## 5. Protocol commitments

- **The test set is touched once per branch**, after training and validation are
  complete. No hyperparameter, architecture, or stopping decision is made on test
  data.
- **Hyperparameters are selected on validation only.** Dropout was raised from 0.1
  to 0.2 on the basis of validation overfitting before this document was written; any
  further change is recorded as an amendment.
- **Both protocols are reported for both branches** — `official` (temporal holdout,
  personalised) and `loso` (subject-disjoint). Neither is omitted, and `official` is
  never described as cross-subject generalisation.
- **No published-comparison table at 90 or 120 minutes.** No protocol-matched
  published MAE exists there; only own-baseline comparison is reported.
- **Per-subject metrics aggregate to mean ± SD across subjects** as the headline;
  pooled numbers appear only as clearly labelled secondary.
- **No asymmetric hypo/hyper training penalty**, ever. It inflates error-grid zone A
  by construction and would make the safety table dependent on the objective.
- **Seeds are fixed at 42.** If a result depends materially on the seed, that
  instability is reported rather than a favourable seed selected.

---

## 6. Claims that are already withdrawn

Recorded here because they appear in the current Introduction draft and must not
survive into submission:

1. **"MAE < 15 mg/dL across all evaluated horizons."** Not attainable; no published
   work attains it. Best published 120-minute MAE is ~35 mg/dL.
2. **"< 15 mg/dL is generally regarded as clinically acceptable."** Not citable for
   forecasting. The figure derives from ISO 15197:2013 — a per-reading tolerance
   under 95% coverage, only below 100 mg/dL, for an in-vitro capillary meter
   measuring the present. No standard, error-grid paper, or consensus statement
   defines an MAE threshold for prediction.
3. **The RAG / guideline-grounding contribution.** Out of scope, and the existing
   corpus is author-paraphrased text rather than ingested guidelines.
4. **"Validated on data spanning CGM, insulin, clinical profiles, and EHR-derived
   glucose."** The EHR and cross-sectional datasets are real but unrelated to the
   CGM subjects and cannot be joined.
5. **Any PINN claim attached to the previous results.** Every legacy checkpoint was
   trained with `use_pinn=False`. Ablation A1 will report what that declared method
   would actually have produced.

---

## 7. Amendments

Any change after test metrics are seen goes here, dated, with the reason.

*(none)*
