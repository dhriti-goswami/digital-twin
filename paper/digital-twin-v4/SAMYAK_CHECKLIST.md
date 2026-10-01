# For Samyak: verify / cross-check before submission

## A. Numbers that may be wrong
1. Sec. III-G3: precision 0.451 at T*=84 mg/dL is HIGHER than 0.347 at 70 mg/dL while specificity drops (0.951->0.908). Correct? Is hypo still defined as <70 for the 84 mg/dL evaluation?
2. Sec. III-G4: "lower quantile covered 22.4%" for G<80 is not in Table X. Source?
3. Eq. (1): where do "135,000 slots" and "48 steps" come from? Explain, or replace with an autocorrelation-based n_eff.
4. Table V vs data/: persistence RMSE SD 2.87/4.39/5.56/6.13 (comprehensive_baselines.csv) vs 2.85/4.58/5.58/6.21 (rmse_by_horizon.csv). Which is right? Are other rows of Table V affected?

## B. Captions written from data-file labels - confirm they are true
5. Table IV = final quantile-head model run; Tables V and VI A7 = separate single-seed ablation run (13.08 vs 13.25, 18.84 vs 19.16 ...).
6. Table VI = one run per arm, seed 42.
7. Table IX: A0 AND A7 are separately trained quantile-head models (not the Table VI runs). Which run is Table IX's A7 - the final model of Table IV?
8. Table VII: seed 42 run is one of the five seeds and matches Tables V/VI.
9. Fig. 2, 3, 5, 6, 7: which run/model and pooled vs per-subject mean?

## C. Claims to confirm
10. Git commit 3baf636 (25 Jul 2026) was made BEFORE any test-set evaluation.
11. Was the 5-seed replication pre-specified? If not, say "post hoc" in the paper.
12. A7 vs ARX (13.25 vs 13.97): any significance test? Text says A7 lowest at every horizon.
13. ICR convention (units/g or g/unit) - decides whether rho = +0.371 has the expected sign.
14. PRED-EGA hypoglycemia region uses <=70 while the rest of the paper uses <70: intended (PRED-EGA definition)?
15. Table XIV: anchor for f is 0.90 (check nothing else is mistyped in the parameter table).
16. Eq. (17) loss term was renamed to lambda_tc L_temporal to match Eq. (22) - confirm they are the same term.

## D. Still missing vs the review
17. Concern 11: no PatchTST/TFT baseline - add one or give a reason for the response letter.
18. Concern 13: Zenodo DOI for code (GitHub link only now).
19. Minor: per-subject skill +/- SD for Table IV (no data in data/).
20. Citations of OhioT1DM papers showing the integrity pitfalls (Sec. II-A), if any can be named.
