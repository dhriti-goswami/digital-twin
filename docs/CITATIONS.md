# Citations index

Phase 0 verification. Every external fact used by this project traces to one of the
two files below, each of which tags facts `VERIFIED-PRIMARY` / `SECOND-HAND` /
`UNVERIFIED` and ends with an explicit list of what could not be confirmed.

| File | Covers |
|---|---|
| [`CITATIONS_benchmarks.md`](CITATIONS_benchmarks.md) | OhioT1DM published benchmarks, the BGLP challenge protocols, persistence baselines, and whether "<15 mg/dL MAE is clinically acceptable" is a citable threshold |
| [`CITATIONS_methods.md`](CITATIONS_methods.md) | Clarke and Parkes error-grid boundaries, Bergman minimal model, Hovorka SC insulin, Lehmann–Deutsch / Dalla Man gut absorption, Kovatchev risk indices, PINN loss-weighting methods |

## Findings that changed the design

1. **`MAE < 15 mg/dL at 30 min` is the field median, not a frontier.** 15 of 17
   tabulated OhioT1DM entries clear it, including a *non-personalised* LSTM at
   14.37. Best credible published: 12.83 MAE / 18.22 RMSE (Rubin-Falcone et al.).
   Consequence: point accuracy cannot be the headline contribution.
2. **The "clinically acceptable" threshold claim is not citable.** The figure leaks
   in from ISO 15197:2013 — a per-reading tolerance, 95% coverage, only below
   100 mg/dL, for an in-vitro capillary meter measuring *the present*. No standard,
   error-grid paper, or consensus states an MAE threshold for *prediction*.
   Replacements: skill score vs persistence; PRED-EGA (Sivananthan et al. 2011),
   an error grid built for predictors; horizon-stratified field-relative targets.
3. **Persistence RMSE@30 = 22.5 ± 2.2, RMSE@60 = 36.6 ± 3.0**, corroborated by two
   independent 2018 sources agreeing to 0.05 mg/dL. Persistence *MAE* is
   unpublished — we compute it, and validate our implementation against the
   published RMSE. This confirms the legacy 30.44 and 78.5 both lose to naive.
4. **The two cohorts use different protocols.** The 2020 BGLP challenge excludes
   the first hour (12 points) of each test file; 2018 does not. Must be implemented
   per-cohort. Implemented in `twin/data/ohio.py`.
5. **No protocol-matched published MAE exists at 90 or 120 min**, so no
   published-comparison table can be built there — own-baseline comparison only.
6. **Clarke 1987 publishes no inequalities**, only a figure and prose. Boundaries
   come from the two canonical reference implementations; an exhaustive
   302,500-point lattice diff isolated exactly one substantive disagreement (the
   upper-C cap at `r ≤ 290`, an artefact of the figure's 0–400 axes, which we
   drop). Regression-tested against the reference implementation in
   `tests/test_metrics.py`.
7. **Pfützner 2013 does not correct Parkes 2000 — it first publishes the
   coordinates.** Cite Parkes for the grid, Pfützner for the vertices.
8. **`p1 = 0` for type 1 diabetes is a modelling simplification, not an empirical
   finding.** Ward et al. 1991 measured `S_G = 1.0–1.6e-2 /min` in IDDM subjects —
   reduced but clearly non-zero. Population default changed accordingly;
   `p1 = 0` remains reachable for the ablation.

## Known gaps carried forward

These are unresolved and must be disclosed rather than papered over:

- **Wilinska et al. 2005 numeric parameters** — paywalled, no reproducing source
  found. Structure only; no value from it is used.
- **Stöckl et al. 2000** letter on the Clarke upper A-line — paywalled and
  unreadable. If a reviewer challenges the A-band construction this is a live gap.
- **Dalla Man 2006/2007 values** are `SECOND-HAND` (via a peer-reviewed survey).
  Surfaced at runtime by `twin.physio.params.second_hand_bounds()`.
- **Kovatchev 2004 page range** and the **PRED-EGA full text** unread.
- ~~**Persistence MAE** — must be computed by us; no published value to check against.~~
  **RESOLVED — computed, see below.**

## Persistence baseline, computed here

No published persistence *MAE* exists on OhioT1DM, so it is computed by this
pipeline. The *RMSE* is published, which lets the implementation be validated
before the MAE is trusted.

**Validation.** Our 2018-cohort test persistence reproduces the two independent
published RMSE values to within 0.3 mg/dL, with matching standard deviations:

| Horizon | Ours (2018 cohort, n=6) | Published (2018 challenge) | Δ |
|---|---|---|---|
| 30 min RMSE | 22.60 ± 2.50 | 22.5 ± 2.2 | 0.10 |
| 60 min RMSE | 36.34 ± 3.14 | 36.6 ± 3.0 | 0.26 |

This is the project's end-to-end correctness check: it simultaneously validates the
XML parser, the 5-minute grid snapping, gap-aware sequencing, horizon integrity,
the metrics implementation, and per-subject aggregation. A defect in any of them
would break the agreement.

**Full persistence baseline** (per-subject metrics, then mean ± SD across subjects):

| Horizon | 2018 (n=6) RMSE | 2018 MAE | 2020 (n=6) RMSE | 2020 MAE | All 12 RMSE | All 12 MAE |
|---|---|---|---|---|---|---|
| 30 min | 22.60 ± 2.50 | **16.36 ± 1.46** | 24.22 ± 3.21 | 17.47 ± 2.36 | 23.41 ± 2.87 | 16.92 ± 1.96 |
| 60 min | 36.34 ± 3.14 | 27.05 ± 1.88 | 40.14 ± 5.26 | 29.57 ± 4.03 | 38.24 ± 4.58 | 28.31 ± 3.28 |
| 90 min | 46.40 ± 3.69 | 34.92 ± 2.35 | 51.05 ± 6.42 | 38.33 ± 4.81 | 48.73 ± 5.56 | 36.62 ± 4.03 |
| 120 min | 54.04 ± 4.48 | 41.11 ± 2.88 | 58.83 ± 7.03 | 44.84 ± 5.27 | 56.44 ± 6.15 | 42.98 ± 4.50 |

The 2020 cohort is consistently harder at every horizon — worth stating, since
pooling the cohorts moves every number.

**Consequence for the paper's original claim.** Persistence alone achieves
**MAE 16.36 mg/dL at 30 minutes**. The "< 15 mg/dL clinically acceptable" target
therefore represents roughly a **9% improvement over predicting no change at
all** — before accounting for the fact that 15 of 17 published entries already
clear it. Presenting it as an achievement would not survive review.

**Window accounting.** Enforcing horizon integrity keeps 144,266 of 187,780
candidate windows (76.8%); test-split windows total 27,145. Rejections are
dominated by incomplete input spans, with target-not-observed second. Per-subject
counts are produced by `twin.data.sequencing.window_report_table`.
