# Round 6: instructions for Dhriti (main.tex edits)

Written 26 Oct 2026 by Samyak. All numbers below are final and reproducible from the scripts named.
Line numbers refer to `paper/digital-twin-v5/main.tex` at commit 247c1b8 (before any round-6
edit). **Edit from the bottom of the file upward**, or line numbers will shift; or search for the
quoted text instead of trusting the line number.

---

## 0. Read this first: what actually changed (and what did not)

Your text message to me was too pessimistic. After the A0 rerun I had every result checked by an
independent audit and ran pre-registered follow-ups. The real situation:

| Claim | Status | Evidence |
|---|---|---|
| Model beats persistence by ~20% RMSE, 12/12 subjects | **Unchanged** | Table V, Fig. 2 |
| $S_I$ subject-specific, stable, dose-consistent | **Unchanged** | Tables X, XI |
| Physics improves in-cohort accuracy | **Holds, but smaller**: 0.17 mg/dL (30 min) to 0.99 mg/dL (120 min), significant at all 4 horizons (Holm p ≤ 0.010) | Table VII (new numbers) |
| Physics gain grows with horizon / concentrates in high-glucose and steady periods | **New, supportive** | ROUND6_RESULTS.md P1, P3, P4 |
| Physics helps under distribution shift (train on one Ohio cohort, test on the other) | **New, strongest new result**: A7 better in 12/12 test subjects at every horizon, 1.7–7.4 mg/dL (≈9–16%) | ROUND6_EXTERNAL.md (P8) |
| Physics helps more with less training data | **New, directional only** (not significant, 3 seeds) | P6 |
| Physics reduces seed variance | **Now false** — A7 is *more* variable. Delete this claim. | P7 |
| "A0 beats A7 on hypo alarm" (round-5 single-seed result) | **Was an artifact** (one seed + unequal patience). Over 5 matched seeds: A7 0.732 vs A0 0.675, CI includes zero. Same conclusion the paper already had: no significant difference. | P9 |
| Quantile alarm beats a simple alarm | **False**: validation-tuned persistence alarm catches the same 38/56 events at the same FA/day. Must be stated. | S-11 |
| Physics improves post-meal accuracy | **Not supported** (no extra gain after meals). Don't claim it. | P2 |

**So the paper keeps its story** ("physics gives modest accuracy, a stable physiological parameter,
and honest evaluation"), and gains one new positive result (cross-cohort robustness). The paper
already said "physics improves accuracy only modestly" and "Table II: physics criterion Not met" —
those stay. Do **not** strengthen any claim beyond what is written below.

**Rule for all edits:** use the numbers in this file exactly. Don't round differently, don't drop
the "post hoc"/"exploratory" qualifiers, don't drop a negative result.

Statistics vocabulary to keep consistent:
- *Primary* physics test = pre-specified single-seed ablation (Holm p = 0.105, not met). Unchanged.
- *Post hoc 5-seed replication* = Table VII (Holm over 4 horizons).
- *Secondary analyses* = round-6 P1–P9, specified in a time-stamped document
  (`PREREG_ROUND6.md`) **before** they were run, but after the main results were known — call them
  "secondary analyses specified before they were run", never "pre-specified" (that word is reserved
  for the 25 July 2026 protocol).

---

## 1. Abstract (lines 45–53)

### 1a. Methods (line 45) — append one sentence at the end of the paragraph
Add after "...whether insulin sensitivity ($S_I$) was subject-specific, stable, and dose-consistent.":

```latex
Secondary analyses, specified before they were run, replicated the physics comparison over five seeds, stratified it by glycemic context, and tested transfer between the two OhioT1DM cohorts.
```

### 1b. Results (line 47) — replace the physics sentence
**Replace** from "In a post hoc five-seed replication on the same test set, ..." up to "...Holm $p=0.105$)." with:

```latex
In a post hoc five-seed replication on the same test set, physics reduced MAE modestly at every horizon (30-min MAE $13.37 \pm 0.11$ vs.\ $13.54 \pm 0.10$~mg/dL; 0.99~mg/dL lower at 120~min; Holm $p \le 0.010$), although the pre-specified single-seed test was not significant (13.25 vs.\ 13.51~mg/dL; Holm $p=0.105$). When trained on one OhioT1DM cohort and tested on the other, physics lowered MAE in all 12 test subjects at every horizon (by 1.7--7.4~mg/dL), although with six subjects per direction no test survived Holm correction.
```

**Same paragraph, last sentence** — keep "At 30~min, the median forecast detected 55.1\% ... precision of 0.347." and **append**:

```latex
A persistence alarm with a validation-tuned threshold detected as many hypoglycemic events (38 of 56) at a similar false-alarm rate.
```

### 1c. Conclusion (line 53) — replace whole paragraph body
```latex
\textbf{Conclusion:} Physiological structure yielded a stable insulin-sensitivity estimate, consistent with insulin dose under personalized evaluation, and small accuracy gains that grew with horizon and were larger under cross-cohort transfer. Physics did not change hypoglycemia detection, and in this cohort a lower-quantile alarm did not detect more events than a tuned persistence alarm, although coverage below 80~mg/dL remained poor.
```

Check the abstract word limit of the venue after this; if over, shorten the Background paragraph, not the results.

---

## 2. Objectives and Contributions (lines 128–136)

- **Bullet 3 (line 133)** — replace with:
```latex
\item To separate the value of physics from accuracy gain through a pre-specified ablation, a five-seed replication, and a cross-cohort transfer test, rather than reporting accuracy alone.
```
- **Bullet 4 ($S_I$, line 134)** — no change.
- **Bullet 5 (line 135, last bullet)** — replace with:
```latex
\item To quantify forecast uncertainty and evaluate lower-quantile hypoglycemia alarms against a validation-tuned persistence alarm.
```
- **Line 78** (related-work list "(iv) quantile forecasts evaluated for hypoglycemia") — no change.

---

## 3. Methods

### 3a. Sec. "Pre-specified Outcomes and Falsification Criteria" (after line 467)
Add a new paragraph at the end of the subsection:

```latex
After the primary results were known, nine secondary analyses of the physics contribution were written down and time-stamped before they were run (\texttt{PREREG\_ROUND6.md} in the code repository): the horizon trend of the A0--A7 gap; strata by post-meal period, target glycemic range, and glucose rate of change; physiological plausibility of the forecast trajectory; reduced training data (25\% and 50\% of subjects, three seeds); seed variance; transfer between the 2018 and 2020 OhioT1DM cohorts; and five-seed hypoglycemia alarm performance. They are reported in full, regardless of direction, with Holm correction within each family; they are not part of the 25 July 2026 protocol.
```

### 3b. Table II, `tab:falsification` (lines 485–502)
- Row "Physics improves accuracy" (line 496), Outcome cell — replace
  `Not met: Holm $p = 0.105$ (significant only post hoc, 5 seeds)` with
```latex
Not met: Holm $p = 0.105$ (post hoc, 5 seeds: $-0.17$~mg/dL, Holm $p = 0.010$)
```
- Every other row: **no change** (13.37 is still the A7 5-seed mean).

---

## 4. Results

### 4a. Sec. III-B baselines paragraph (line 704) and Table `tab:baselines_identical` (lines 706–731)
**No change.** These use seed 42 (13.51 / 13.25), which the audit confirmed was trained fairly.

### 4b. NEW subsection: Cross-Cohort Transfer — insert after Sec. "Cross-Subject Generalization" (i.e. right before `\subsection{Physics Ablations}`, line 800)

Text:
```latex
\subsection{Cross-Cohort Transfer}
\label{sec:cross_cohort}
OhioT1DM contains two cohorts recorded with different sensors and pumps (2018: subjects 559, 563, 570, 575, 588, 591; 2020: 540, 544, 552, 567, 584, 596). As a secondary analysis, A0 and A7 were each trained with five seeds on one cohort only, with validation data from the same cohort, and tested on the official test files of the other cohort. This is cross-cohort transfer within one dataset, not external validation. Both models were markedly less accurate than under the personalized protocol (30-min MAE 16.1--18.7 vs.\ 13.4--13.5~mg/dL), but A7 degraded less: it had lower MAE than A0 in all six test subjects at every horizon in both directions, by 1.7--4.8~mg/dL for 2018$\rightarrow$2020 and 2.6--7.4~mg/dL for 2020$\rightarrow$2018 (Table~\ref{tab:cross_cohort}, Fig.~\ref{fig:cross_cohort}); RMSE showed the same pattern. With six subjects per direction the smallest attainable Wilcoxon $p$ is 0.031, and no test survived Holm correction across the 16 comparisons ($p_{\text{Holm}} = 0.5$), so the result is consistent in sign but statistically weak. It suggests that the physiological prior acts as a regularizer under distribution shift, although this design cannot separate that from improved physiological fidelity.
```

NEW Table `tab:cross_cohort` (place right after the paragraph). Data source:
`results/tables/round6_external_summary.csv` (copy into `paper/digital-twin-v5/data/`).
```latex
\begin{table}[t]
\centering
\caption{Cross-cohort transfer: MAE (mg/dL), mean over five seeds and six test subjects. Gap = A0 $-$ A7 with subject-bootstrap 95\% CI; A7 was better in 6 of 6 subjects in every row (Wilcoxon $p = 0.031$, Holm $p = 0.5$).}
\label{tab:cross_cohort}
\footnotesize
\begin{tabular}{llrrr}
\toprule
Direction & Horizon & A0 & A7 & Gap [95\% CI] \\
\midrule
2018$\rightarrow$2020 & 30 min & 18.38 & 16.64 & 1.74 [1.21, 2.30] \\
 & 60 min & 31.48 & 27.80 & 3.68 [2.37, 5.08] \\
 & 90 min & 40.57 & 35.77 & 4.80 [2.78, 6.94] \\
 & 120 min & 46.39 & 41.57 & 4.82 [2.26, 7.47] \\
\midrule
2020$\rightarrow$2018 & 30 min & 18.71 & 16.14 & 2.56 [2.24, 2.85] \\
 & 60 min & 31.83 & 26.65 & 5.17 [4.59, 5.68] \\
 & 90 min & 41.30 & 34.45 & 6.85 [5.95, 7.58] \\
 & 120 min & 47.78 & 40.39 & 7.40 [6.31, 8.24] \\
\bottomrule
\end{tabular}
\end{table}
```

NEW Figure `fig:cross_cohort`: grouped bar chart, x = horizon (30/60/90/120), two panels
(2018→2020, 2020→2018), bars A0 vs A7 MAE, use the same pgfplots style/colors as Fig. 4
(`fig:ablationfig`, lines 846–878) so it looks consistent. Single-column (`figure`, not `figure*`).
Read values from `data/round6_external_summary.csv` columns `A0_mae`, `A7_mae` — or hardcode the
numbers from the table above. Caption:
```latex
\caption{\small\textit{Cross-cohort transfer. MAE of A0 (no physics) and A7 (physics-guided) trained on one OhioT1DM cohort and tested on the other (five seeds, six test subjects per panel). A7 is lower in every subject and horizon, but no comparison survives Holm correction.}}
```

### 4c. Sec. "Physics Ablations" paragraph (line 801)
Replace "...found that, on seed-averaged per-subject errors, A7 has lower MAE at every horizon, by 0.51~mg/dL at 30~min and 2.38~mg/dL at 120~min (Holm $p = 0.002$ at each horizon)." with
```latex
found that, on seed-averaged per-subject errors, A7 has lower MAE at every horizon, by 0.17~mg/dL at 30~min and 0.99~mg/dL at 120~min (Holm $p \le 0.010$ at each horizon).
```
Rest of line 801 (A3 vs A4, 0.07–0.76) and line 803: **no change**.

### 4d. Table `tab:ablation` (lines 812–831) — **no change** (single-seed, pre-specified).

### 4e. Fig. 4 `fig:ablationfig` (lines 846–878)
- Bars (line 870): **no change** (seed 42).
- Caption (line 876): replace the last sentence "A post hoc 5-seed replication (30-min MAE 13.37 vs.\ 13.88~mg/dL) found A7 significantly better at all horizons under paired Wilcoxon testing (Table~\ref{tab:multiseed})." with
```latex
A post hoc 5-seed replication (30-min MAE 13.37 vs.\ 13.54~mg/dL) found A7 modestly but significantly better in MAE at all horizons under paired Wilcoxon testing (Table~\ref{tab:multiseed}).
```

### 4f. Table VII `tab:multiseed` (lines 885–903) — replace the four data rows and the caption
Source: `data/multiseed_a0_vs_a7.csv`, `data/multiseed_wilcoxon_tests.csv` (already in `data/`).
The single p column is for MAE; RMSE p goes in the caption.

Caption:
```latex
\caption{Post hoc multi-seed replication (5 seeds: 42, 101, 202, 303, 404) comparing A0 (no physics) and A7 (hybrid physics-guided), with paired Wilcoxon signed-rank tests on seed-averaged per-subject MAE and Holm correction over the four horizons. Means $\pm$ SD over the five seeds. For RMSE, the difference was significant only at 90 and 120~min ($p_{\text{Holm}} = 0.027$, $0.028$; 30 and 60~min: $0.154$).}
```
Rows:
```latex
30 min & $13.54 \pm 0.10$ & $19.32 \pm 0.13$ & $\mathbf{13.37 \pm 0.11}$ & $\mathbf{19.17 \pm 0.16}$ & $-0.17$ & 0.0093 & 0.0098 & \textbf{Yes} ($p < 0.01$) \\
60 min & $23.01 \pm 0.16$ & $31.64 \pm 0.15$ & $\mathbf{22.60 \pm 0.32}$ & $\mathbf{31.23 \pm 0.44}$ & $-0.41$ & 0.0049 & 0.0098 & \textbf{Yes} ($p < 0.01$) \\
90 min & $30.01 \pm 0.15$ & $40.43 \pm 0.20$ & $\mathbf{29.28 \pm 0.53}$ & $\mathbf{39.61 \pm 0.70}$ & $-0.72$ & 0.0024 & 0.0073 & \textbf{Yes} ($p < 0.01$) \\
120 min & $35.07 \pm 0.10$ & $46.54 \pm 0.26$ & $\mathbf{34.08 \pm 0.64}$ & $\mathbf{45.47 \pm 0.86}$ & $-0.99$ & 0.0015 & 0.0059 & \textbf{Yes} ($p < 0.01$) \\
```
Table header line ("Paired Wilcoxon Test (Seed-Averaged)") → change to "Paired Wilcoxon Test, MAE (Seed-Averaged)".

### 4g. Paragraph after Table VII (line 905) — replace entirely
```latex
A7 had significantly lower MAE than A0 at every horizon ($p_{\text{Holm}} \le 0.010$), by 0.17~mg/dL at 30~min (1.3\% of A0's MAE) rising to 0.99~mg/dL at 120~min (2.8\%); subject-bootstrap 95\% CIs (5,000 resamples) excluded zero at every horizon ($[-0.30, -0.06]$ at 30~min, $[-0.67, -0.18]$ at 60~min, $[-1.11, -0.37]$ at 90~min, $[-1.49, -0.53]$ at 120~min). The gap grew with horizon in 10 of 12 subjects. Physics did not reduce variability across seeds: the seed SD of A7 was similar to A0's at 30~min (0.11 vs.\ 0.10~mg/dL) and larger at 120~min (0.64 vs.\ 0.10~mg/dL).

\textit{Where the gain arises.} In secondary analyses (Section~\ref{sec:prespec}; full tables in the Supplementary Material, Table~S1), the in-cohort gain concentrated in hyperglycemic targets ($>180$~mg/dL: 0.85~mg/dL at 60~min and 1.58~mg/dL at 90~min, both Holm-significant in a family of 55 tests) and in periods of stable glucose ($|\text{ROC}| < 1$~mg/dL/min: 0.78 and 1.12~mg/dL at 90 and 120~min), with no difference in the 70--180~mg/dL range or during rapid change. Contrary to expectation, the gain was not larger in the 3~h after meals (it was, if anything, smaller). With 25\% of training subjects, A7's advantage rose to 0.49--2.35~mg/dL across horizons (50\%: 0.15--1.41~mg/dL), but with three seeds no low-data comparison survived Holm correction. Forecast trajectories of A0 more often exceeded 4~mg/dL/min (36.8\% of windows vs.\ $<0.01\%$ for A7), but almost all of these excursions lay in the first and last 10~min of the 2-h spline (0.1\% in between), so this reflects smoother horizon boundaries rather than more plausible interior dynamics.
```
⚠ `\label{sec:prespec}` — put that label on the "Pre-specified Outcomes and Falsification Criteria" subsection heading (line 463) if it does not already have one.

### 4h. Sec. III-F Clinical Accuracy, PRED-EGA (S-16)
- Line 911: "Clinically acceptable ... at least 99.4\% ... at most 0.6\%" — **no change** (means unchanged).
- Line 911 last sentence: change "0.13\% at 60~min" → **"0.14\% at 60~min"**.
- Table VIII `tab:predega` (lines 913–933): replace rows with (source `data/pred_ega_summary.csv`, script `scripts/compute_pred_ega_table.py`):
```latex
Overall Accurate (\%) & $90.8 \pm 3.3$ & $77.8 \pm 7.0$ & $68.6 \pm 8.1$ & $61.9 \pm 8.3$ \\
Overall Benign (\%) & $9.2 \pm 3.3$ & $21.9 \pm 6.8$ & $30.9 \pm 7.8$ & $37.5 \pm 8.1$ \\
Overall Erroneous (\%) & $0.04 \pm 0.07$ & $0.28 \pm 0.34$ & $0.50 \pm 0.65$ & $0.60 \pm 0.63$ \\
\textbf{Clinically Acceptable (\%)} & $\mathbf{99.96 \pm 0.07}$ & $\mathbf{99.72 \pm 0.34}$ & $\mathbf{99.50 \pm 0.65}$ & $\mathbf{99.40 \pm 0.63}$ \\
\midrule
Hypoglycemia Accurate (\%) & $65.9 \pm 28.4$ & $33.8 \pm 25.5$ & $18.3 \pm 21.3$ & $10.9 \pm 16.2$ \\
Hypoglycemia Erroneous (\%) & $0.00 \pm 0.00$ & $0.14 \pm 0.47$ & $0.14 \pm 0.32$ & $0.88 \pm 1.61$ \\
Hyperglycemia Accurate (\%) & $96.2 \pm 2.0$ & $86.1 \pm 6.0$ & $75.6 \pm 8.6$ & $66.1 \pm 10.4$ \\
Hyperglycemia Erroneous (\%) & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ \\
```
- Line ~935 paragraph: "65.9\% ... 10.9\%" — no change.

### 4i. Sec. III-G Hypoglycemia (lines 999–1033)

**Line 1001 (intro sentence)**: "four ways" — keep, but change "a head-to-head comparison of A0 and A7 with quantile heads" → "a five-seed comparison of A0 and A7 with quantile heads, a comparison with simple threshold alarms".  → "five ways".

**III-G1 (line 1004)** — after "...extended the median lead time to 25.0~min." and **before** "At about two false alarms per day...", insert:
```latex
To test whether the quantile alarm adds value over a trivial rule, persistence and linear-extrapolation alarms were given thresholds tuned on the validation split to match the model alarm's validation false-alarm rate (2.23/day), then evaluated once on test. The persistence alarm ($G(t) < 90.5$~mg/dL) detected 38 of 56 events (67.9\%, the same as the $q=0.10$ alarm) at 2.21 false alarms/day with a longer median lead time (17.5~min), but lower window-level specificity (92.0\% vs.\ 95.4\%); linear extrapolation (threshold 63.0~mg/dL) detected 25 of 56 (44.6\%; specificity 97.4\%, 1.96 false alarms/day). At this operating point the quantile alarm therefore does not detect more events than a tuned persistence rule; its advantages are higher specificity, an adjustable operating curve, and the multi-horizon alarm, for which no persistence analogue was evaluated.
```
Source: `data/hypo_alarm_baseline_round5.csv`, script `scripts/run_extra_baselines_round5.py`.

**III-G2 `sec:quantile_a0_a7` (line 1008)** — replace the whole paragraph with:
```latex
To test whether physics changes hypoglycemia detection when both models have quantile heads, A0 and A7 were each trained with the same non-crossing quantile head and identical configuration under five seeds, and the $q = 0.10$ alarm was evaluated on the same 56 test episodes (Table~\ref{tab:hypo_multiseed}). Event-level sensitivity was $73.2 \pm 6.7\%$ for A7 and $67.5 \pm 6.7\%$ for A0, at the same specificity ($95.3\%$ vs.\ $95.4\%$) and false-alarm rate (2.24 vs.\ 2.22 per day); A0's median lead time was longer (17.0 vs.\ 14.0~min). The difference in sensitivity ($+5.7$ percentage points) had a 95\% CI of $[-5.7, +15.3]$ from a two-level bootstrap over subjects and seeds, so physics did not significantly change hypoglycemia detection. Single seeds varied widely (60.7--80.4\%), and a single-seed comparison could point in either direction; an earlier single-seed comparison is superseded by this replication.
```

**Line 1010 (single-run paragraph)** — keep the first sentence ("Single-run hypoglycemia estimates are also sensitive to training. In the single-seed ablation ... (0.591 vs.\ 0.621) ... (+8.61 vs.\ +7.29~mg/dL)"), **delete** "and A0's median-forecast sensitivity fell from 0.621 there to 0.413 once a quantile head was added" (that was the old handicapped A0 run). Keep the last sentence.

**III-G3 (lines 1013–1014)** — see Section 6 below (S-17). Do not touch until that section is filled in.

**Table IX `tab:hypo_event` (lines 1016–1033)** — the A0 rows came from the old handicapped run and must go. New content (A7 rows are the final model, unchanged):
- Caption → `Consensus event-level and pointwise hypoglycemia detection at 30 minutes for the final model and validation-tuned threshold alarms (92.0 patient-days, 56 episodes). The five-seed A0 vs.\ A7 comparison is in Table~\ref{tab:hypo_multiseed}.`
- **Delete** the two A0 rows (lines 1025–1026) and the `\midrule` after them.
- Keep the three A7 rows (1028–1030) but **un-bold** 67.9\% (persistence equals it). Keep bold on 82.1\% and 25.0 min.
- **Add** after the A7 rows:
```latex
\midrule
Persistence & $G(t) < 90.5$ (val.-tuned) & -- & -- & -- & 67.9\% & 92.0\% & 2.21 & 17.5 min \\
Linear extrapolation & $\hat{G} < 63.0$ (val.-tuned) & -- & -- & -- & 44.6\% & 97.4\% & 1.96 & 15.0 min \\
```

**NEW Table `tab:hypo_multiseed`** — place right after Table IX. Source `results/tables/round6_hypo_multiseed.csv` (copy into `data/`), script `scripts/round6_hypo_multiseed.py`.
```latex
\begin{table}[t]
\centering
\caption{Five-seed hypoglycemia alarm ($\hat{G}_{0.10} < 70$~mg/dL, 30~min) for A0 and A7 with identical quantile heads and training configuration; mean $\pm$ SD over seeds 42, 101, 202, 303, 404 (56 episodes). Difference CI from a two-level bootstrap (subjects, then seeds; 5,000 resamples).}
\label{tab:hypo_multiseed}
\footnotesize
\begin{tabular}{lrrrr}
\toprule
Model & Event Sens. & Specificity & FA/day & Lead (min) \\
\midrule
A0 & $67.5 \pm 6.7\%$ & $95.4 \pm 0.4\%$ & $2.22 \pm 0.19$ & $17.0 \pm 2.1$ \\
A7 & $73.2 \pm 6.7\%$ & $95.3 \pm 0.4\%$ & $2.24 \pm 0.16$ & $14.0 \pm 2.2$ \\
\midrule
A7 $-$ A0 & \multicolumn{4}{l}{$+5.7$ pp, 95\% CI $[-5.7, +15.3]$ (not significant)} \\
\bottomrule
\end{tabular}
\end{table}
```
Per-seed values (if a reviewer asks; do not put in main text): A0 44/38/37/34/36 of 56, A7 37/39/45/39/45 of 56 for seeds 42/101/202/303/404.

### 4j. Sec. III-H $S_I$ (lines 1147–1216) — see Section 6 (partial correlations). Everything else: **no change**.

### 4k. Sec. III-I Attribution (line 1223)
Replace "Spearman $\rho = 0.731$ ($p = 6.05 \times 10^{-7}$, $n=35$)" → "Spearman $\rho = 0.740$ ($p = 3.83 \times 10^{-7}$, $n=35$)". (CSVs were regenerated; recomputed value.)

---

## 5. Discussion, Limitations, Conclusion

### 5a. Discussion "First, ..." (line 1313)
Replace "Here it improves accuracy only modestly (0.51--2.38~mg/dL MAE across horizons over five seeds): about 4\% of MAE at 30~min, unlikely to change a clinical decision on its own, and about 7\% at 120~min, where it matters more for planning." with:
```latex
Here it improves in-cohort accuracy only modestly (0.17--0.99~mg/dL MAE across horizons over five seeds, 1.3--2.8\%), too little to change a clinical decision on its own. The gain is larger where a physiological prior has most leverage: at longer horizons, in hyperglycemia, and, most clearly, under distribution shift, where models trained on one OhioT1DM cohort and tested on the other were 1.7--7.4~mg/dL (about 9--16\%) more accurate with physics in every test subject. This pattern is consistent with physics acting as a regularizer when training data do not match the test population, which is the situation a deployed forecaster faces; confirming it requires a genuinely external cohort.
```
Keep "It also supplies interpretable state variables and a stable $S_I$ estimate, ..." unchanged.

**"Second, ..."** same line — replace "a lower-quantile alarm detects more events, but its calibration below 80~mg/dL is poor." with:
```latex
a lower-quantile alarm detects more events than the median forecast, but no more than a persistence alarm tuned to the same false-alarm rate, and its calibration below 80~mg/dL is poor. Physics did not change alarm performance. Forecast-based alarms in this cohort should therefore be justified by specificity, adjustable operating points, and multi-horizon warning, not by event sensitivity alone.
```

### 5b. Limitations (line 1320)
After "...only A0 and A7 were replicated over five seeds, post hoc (Section~\ref{sec:multiseed_replication})." insert:
```latex
The secondary analyses were specified after the main results were known; the low-data comparison used three seeds and the cross-cohort comparison six subjects per direction, so neither reached significance after correction, and the cross-cohort test is transfer within one dataset rather than external validation. A7 was more variable across seeds than A0 at longer horizons.
```
After "...the hypoglycemia analysis rests on 56 events." append: " A tuned persistence alarm matched the quantile alarm's event sensitivity."

### 5c. Future work (line 1326) — no change (already lists Shanghai, DiaTrend external validation).

### 5d. Conclusion (line 1353)
Replace "Physics improves point accuracy only modestly, and significantly only in a post hoc five-seed replication. The strongest evidence is instead that physiology can provide stable patient-specific structure alongside this small accuracy gain, and that lower predictive quantiles can recover hypoglycemic events missed by an otherwise accurate median forecast." with:
```latex
Physics improves in-cohort point accuracy only modestly, and significantly only in a post hoc five-seed replication, but it degraded less when models were transferred between cohorts. The strongest evidence is that physiology can provide stable patient-specific structure alongside this small accuracy gain. Lower predictive quantiles recovered hypoglycemic events missed by the median forecast, but no more than a tuned persistence alarm, and physics did not change alarm performance.
```

### 5e. Stray `\begin{table*}[t]` at line 1352
Directly after `\section{Conclusion}` there is `\begin{table*}[t]` that opens `tab:params`. Check it compiles and the table floats where you want; consider moving `tab:params` to the Notation section (line 1358) so the Conclusion text isn't interrupted in the source.

---

## 6. S-17: the two "no generator" results — DONE

I wrote generators that follow the method exactly as the paper describes it, and ran them once
(no tuning toward the old numbers).

### 6a. Partial correlations (III-H) — REPRODUCE, keep
Script `scripts/regen_si_partial_corr.py`, output `results/tables/si_partial_corr_regen.csv`
(copy to `data/`). Every value matches except one rounding:
- Line ~1178 (item 2) and `tab:si_circularity` row "Partial Pearson $r(S_I, \text{TDD} \mid \bar{G}_{\text{test}})$": **−0.886 → −0.885** (p = 0.0003 unchanged).
- All others (−0.838/0.0013, −0.843/0.0011, −0.875/0.0004, unadjusted −0.839) unchanged.
- Add to item 2, end: "(partial correlations computed from OLS residuals, $p$ from a $t$ test with $n-3$ df)".

### 6b. F2 threshold search (III-G3) — DOES NOT REPRODUCE, remove
Re-running the method in the text gives T* = 66.5 mg/dL (target G<70) or 99.5 mg/dL (target G<T),
never 84.0, and none of 0.988/0.868/0.175/3,397 FP/0.943/0.908/0.451 come back. The original
script does not exist, so these numbers cannot stay. The 70 mg/dL reference values
(0.918/0.951/0.347, 1,257 FP) **do** reproduce.

Edit III-G3 (`sec:validation_threshold_selection`, lines 1013–1014):
- Rename subsection to `\subsubsection{Alarm Operating Curves}` (keep the label).
- **Delete** from "To avoid test-set reuse, the alarm threshold was chosen..." through "...The higher threshold therefore trades precision for sensitivity."
- **Insert** instead:
```latex
The 70~mg/dL alarm threshold was fixed in advance from the consensus definition~\cite{ref52} rather than tuned; on test it gave pointwise sensitivity 0.918, specificity 0.951, and precision 0.347 (1,257 false positives).
```
- Keep the rest of the paragraph (the $q = 0.05/0.20$ Gaussian approximation and the
  sensitivity–precision curves sentence) unchanged.
- Line 1001 intro: "validation-based threshold selection" → "operating curves".
- Limitations (line 1320): replace "The alarm operating point was selected on validation data without test-set reuse (Section~\ref{sec:validation_threshold_selection}), but lacks an elicited clinical cost ratio." with "The alarm threshold was fixed at the consensus 70~mg/dL rather than chosen from an elicited clinical cost ratio (Section~\ref{sec:validation_threshold_selection})."
- Checklist: search for `84.0`, `0.771`, `3,397`, `F_2` — none should remain.

## 7. Supplementary (`supplementary.tex`)

Add **Table S1: Secondary analyses of the physics contribution (P1–P7)**. Copy the P1–P5 and P7
tables from `results/tables/round6_tables.md` (already formatted: stratum, horizon, n, A0, A7,
gap, 95% CI, raw p, Holm p). Keep every row, including "no difference" and "hurts" rows. Caption:

```latex
Secondary analyses of the A0--A7 MAE gap (mg/dL; positive = A7 better), seed-averaged per subject over five seeds. Holm correction over all 55 tests in this family; with 12 subjects the smallest attainable Holm-adjusted $p$ is about 0.027. Specified in \texttt{PREREG\_ROUND6.md} before being run; deviations are listed in \texttt{ROUND6\_RESULTS.md}.
```
Add **Table S2: P6 low-data** from `results/tables/round6_p6_lowdata.csv` (25%, 50%, 100% rows; mark 100% as descriptive).
Add one sentence on P5 robustness: values from `results/tables/round6_p5_robustness.csv` (excluding first and last 10 min: A0 0.10% vs A7 0%).

---

## 8. Things NOT to change

- Persistence/skill numbers (19.4–21.2%), Fig. 2 `fig:summary`, LOSO section and Fig. `fig:matched_protocol_comparison`.
- Table `tab:ablation`, Fig. 4 bars, Table `tab:baselines_identical` (single seed 42, pre-specified).
- All $S_I$ results (CV 35.8%, ICC 0.890, ρ = −0.839, abl-A7-notc), except the S-17 partial correlations.
- Abstract alarm numbers 55.1% / 91.8% / 0.347 and Table IX A7 rows (final model, unchanged).
- Calibration table `tab:calibration`.
- Table II rows other than "Physics improves accuracy".

## 9. Data files
Already copied into `paper/digital-twin-v5/data/` by Samyak: `round6_external_summary.csv`,
`round6_hypo_multiseed.csv`, `round6_p6_lowdata.csv`, `round6_p5_robustness.csv`,
`round6_tables.md`, `si_partial_corr_regen.csv`, `f2_threshold_regen.csv`. Full analysis write-ups:
`ROUND6_RESULTS.md` (P1–P7), `ROUND6_EXTERNAL.md` (P8), `ROUND6_P9.md` (P9).

## 10. Checklist when done
- [ ] Search main.tex for `84.0`, `0.771`, `3,397`, `0.886`, `0.731`, `0.51`, `2.38`, `13.88`, `0.00195`, `4\%`, `7\%`, `58.9`, `0.413`, `0.861`, `reduced variability` — none of these should remain except 0.51 if used elsewhere for a different quantity (check context).
- [ ] Every `\ref` to a new label (`tab:cross_cohort`, `fig:cross_cohort`, `tab:hypo_multiseed`, `sec:prespec`, `sec:cross_cohort`) resolves.
- [ ] Compiles with no `??`.
- [ ] Add a round-6 entry to `CHANGES.md` listing the above.
