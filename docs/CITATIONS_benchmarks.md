# Citations & Benchmarks: OhioT1DM glucose forecasting, and the "clinically acceptable error" question

Compiled 2026-07-25. Every numeric claim below carries a confidence tag:

- **VERIFIED-PRIMARY** — I read the number in the primary paper's own results table/abstract (PDF fetched and text-extracted).
- **SECOND-HAND** — number comes from another paper's comparison table, a publisher abstract, or a search snippet; not read in the originating source.
- **UNVERIFIED** — could not confirm; do not cite.

> **Reading warning.** Section A.2 is the only table that is internally comparable. Section A.3 contains
> numbers obtained under *different* protocols and **must not be merged** into the same table without an
> explicit protocol column. Section A.6 lists specific papers whose numbers are non-comparable or implausible.

---

## PART A — OhioT1DM published benchmarks

### A.1 The dataset and the official challenge protocol

**Source (primary, read in full):** Marling, C. & Bunescu, R. *The OhioT1DM Dataset for Blood Glucose Level
Prediction: Update 2020.* KDH@ECAI 2020, CEUR-WS Vol-2675, pp. 71–74. <https://ceur-ws.org/Vol-2675/paper11.pdf>

Confirmed verbatim from that paper (**VERIFIED-PRIMARY**):

| Fact | Value |
|---|---|
| Cohort released 2018 (1st BGLP Challenge, KDH@IJCAI-ECAI 2018, Stockholm) | **559, 563, 570, 575, 588, 591** (all wore Basis Peak bands) |
| Cohort released 2020 (2nd BGLP Challenge, KDH@ECAI 2020, Santiago de Compostela) | **540, 544, 552, 567, 584, 596** (all wore Empatica Embrace) |
| Total | 12 subjects, 8 weeks each, CGM every 5 min |
| Split | **Fixed, provided by the organisers**: one XML file per subject for train+dev, a separate XML file per subject for test. 24 files total. Not a random or user-chosen split. |
| Test-set sizes (2020 cohort) | 540: 2884, 544: 2704, 552: 2352, 567: 2377, 584: 2653, 596: 2731 scored points |
| Test-set sizes (2018 cohort) | 559: 2514, 563: 2570, 570: 2745, 575: 2590, 588: 2791, 591: 2760 scored points |
| **2018 vs 2020 scoring difference** | In 2018 every point in the test XML was scored. In **2020 the first hour (12 points) of each test file is excluded** from evaluation, "to allow unbiased comparison of prediction models using all training data to predict each test point". So 2018 and 2020 numbers are *not* strictly identical protocols even on the same data. |

**Your prior is confirmed.** The 2020 challenge scored 540/544/552/567/584/596 and the 2018 challenge used
559/563/570/575/588/591. (**VERIFIED-PRIMARY**, Marling & Bunescu 2020, Tables 1 and 2.)

**Horizons.** The KDH-2020 challenge page states participants "report results for 30 and 60-minute prediction
horizons, as specified in The BGLP Challenge Rules" (**VERIFIED-PRIMARY** for the page text;
<https://sites.google.com/view/kdh-2020/bglp-challenge>). The rules PDF itself and the 2018 rules page are no
longer publicly reachable (2018 page is now behind a Google login) — see UNVERIFIED section. In the 2018
proceedings, entries report **30-min RMSE** essentially universally, with 60-min reported only by some teams
(Martinsson, Bertachi, Contreras); this is consistent with 30-min being the required 2018 horizon, but that
inference is **SECOND-HAND**.

**There is no official leaderboard paper.** I enumerated all 27 papers in CEUR Vol-2675 (KDH 2020) and all
papers in CEUR Vol-2148 (KDH 2018). Papers 11–27 of Vol-2675 are the challenge entries; paper 11 is the
dataset paper. **No summary/ranking/leaderboard paper exists in either volume** — Marling & Bunescu's paper
contains no results at all. Any "challenge winner" claim in the literature is therefore an informal
reconstruction from the individual entry papers. (**VERIFIED-PRIMARY**: I read the Vol-2675 and Vol-2148
tables of contents and the Marling & Bunescu paper end-to-end.)

Metrics used by entrants: RMSE and MAE (mg/dL) are near-universal; several add Clarke/Parkes EGA zones,
gRMSE, COD, MARD, and time-gain/delay.

---

### A.2 Comparable table — 2nd BGLP Challenge (2020), official split, subjects 540/544/552/567/584/596

All rows below: **official organiser-provided train/test XML split, 6 subjects of the 2020 cohort, per-subject
models averaged across subjects (i.e. mean of per-patient errors, not pooled), scored at the horizon
endpoint.** All values mg/dL. All rows are **VERIFIED-PRIMARY** — I extracted them from each paper's own
results table or abstract.

| Model | RMSE@30 | MAE@30 | RMSE@60 | MAE@60 | Source (CEUR Vol-2675) |
|---|---|---|---|---|---|
| **Deep residual (N-BEATS + BiLSTM blocks + aux losses)** — Rubin-Falcone, Fox & Wiens | **18.22** | **12.83** | **31.66** | **23.60** | paper18 † |
| Shallow NN + error-imputation module (NN-EIM) — Pavan et al. | 18.63 ± 2.22 | **10.08 ± 2.10** ‡ | 32.27 ± 4.25 | 17.69 ± 3.87 ‡ | paper16 |
| Non-personalised LSTM (128 units, 30-min history) — Bevan & Coenen | 18.23 ± 2.36 | 14.37 ± 1.83 | 31.10 ± 4.05 | 25.75 ± 3.43 | paper17 |
| GAN (GRU generator + 1D-CNN discriminator) — Zhu et al. | 18.34 ± 0.17 | 13.37 ± 0.18 | 32.31 ± 0.46 | 24.20 ± 0.42 | paper15 |
| Stacked regression, CGM+activity fusion (Method 1) — Nemat et al. | 18.99 ± 0.08 | 13.73 ± 0.07 | 33.39 ± 0.12 | 25.04 ± 0.11 | paper21 |
| Multi-lag stacking (System 2) — Khadem et al. | 19.21 ± 0.14 | 13.93 ± 0.12 | 33.65 ± 0.19 | 25.31 ± 0.19 | paper26 |
| Multi-scale LSTM, multi-lag (MS-LSTM) — Yang et al. | 19.048 | 13.503 | 32.029 | 23.833 | paper24 |
| Latent-variable statistical model — Sun et al. | 19.37 | 13.76 | 32.59 | 24.64 | paper20 |
| Multitask CRNN (MTCRNN) — Daniels, Herrero & Georgiou | 19.79 ± 0.06 | 13.62 ± 0.05 | 33.73 ± 0.24 | 24.54 ± 0.15 | paper19 |
| Single-task CRNN (same paper's ablation) | 20.67 ± 0.32 | 14.28 ± 0.19 | 34.40 ± 0.14 | 24.67 ± 0.14 | paper19 |
| Multi-class LSTM classifier (risk-domain bins), personalised | 19.8 | 14.4 | 34.0 | 25.8 | paper13, Tables 1–2 § |
| …same, + 2018 cohort as extra training data | 19.4 | 13.9 | 33.4 | 25.0 | paper13, Tables 3–4 § |
| Personalised interpretable LSTM — Cappon et al. | 20.20 | 14.74 | 34.19 | 25.98 | paper12 |
| Online ARMA + residual-compensation net (best-of-runs) — Ma et al. | 20.03 ¶ | 14.52 ¶ | 34.89 ¶ | 24.61 ¶ | paper27 |
| Genetic programming, best variant — Joedicke et al. | 19.60 | 14.25 | 32.04 | 23.58 | paper25 ‖ |
| Seq2Seq BiLSTM — Bhimireddy et al. | 21.8 | 15.0 | 35.0 | 25.0 | paper22 |
| CNN/LSTM ("best model") — Freiburghaus, Rizzotti-Kaddouri & Albertetti | **17.45** ‡ | **11.22** ‡ | 33.67 | 23.25 | paper23 ‡ |

Footnotes:
- **†** Rubin-Falcone et al. pre-trained on **external Tidepool data plus the 2018 OhioT1DM cohort** before
  fitting the 2020 subjects. Test protocol is the official 2020 split, but the training corpus is larger than
  OhioT1DM alone. Their own baseline (plain N-BEATS, 10 blocks) is RMSE@30 = 21.2 — a *model* baseline, **not**
  a persistence baseline.
- **‡** *Anomaly flags.* Pavan et al. report MAE/RMSE = 10.08/18.63 = 0.54 overall, and 6.51/16.51 = 0.39 for
  subject 552. For a roughly symmetric error distribution MAE/RMSE ≈ 0.7–0.8; a ratio of 0.39 implies an
  extremely heavy-tailed error distribution and is out of family with every other entry (all 0.68–0.75). Their
  paper also states no missing-value imputation was applied to the test set and reports how many of the
  available CGM samples each model actually predicted — i.e. **the MAE may be computed over a subset of test
  points**. Freiburghaus et al. likewise report the single best model configuration and elsewhere quote
  RMSE = 13.34 / MAE = 9.08 for selected curves. **Treat both MAE@30 figures as not-safely-comparable.**
- **§** Mayo & Koutny discard all test examples containing any gap or non-exactly-5-min spacing, so their
  scored test set is smaller than the official one (they state 2,743 / 2,579 / 2,177 / 2,185 / 2,393 / 2,624
  examples at 30 min). Slightly non-standard.
- **¶** Ma et al. report the *mean of the best* RMSE/MAE across runs, which is optimistically biased.
- **‖** Joedicke et al.'s table reports many GP variants with comma decimal separators; I took the best column
  per metric. Lower confidence on which variant these belong to — verify before quoting.

For the same paper13 entry on the **2018** cohort (official 2018 protocol): RMSE@30 20.7, MAE@30 14.3,
RMSE@60 32.8, MAE@60 24.2 (**VERIFIED-PRIMARY**).

---

### A.3 1st BGLP Challenge (2018), official split, subjects 559/563/570/575/588/591

MAE was rarely reported in 2018. All **VERIFIED-PRIMARY** unless noted.

| Model | RMSE@30 | MAE@30 | RMSE@60 | MAE@60 | Source (CEUR Vol-2148) |
|---|---|---|---|---|---|
| Dilated RNN — Chen, Li, Herrero, Zhu, Georgiou | **18.91** (best-of-10) / 19.04 (mean-of-10) | — | — | — | paper11 |
| Physiological model + ANN — Bertachi et al. | 19.33 | — | 31.72 | — | paper14 |
| SVR-RBF, recursive (best of 13 models) — Xie & Wang | 19.53 | — | — | — | paper16 |
| Ridge / Linear regression, recursive — Xie & Wang | 19.62 | — | — | — | paper16 |
| LSTM w/ MSE loss — Martinsson et al. | 20.1 ± 2.5 | — | 33.2 ± 3.1 | — | paper10 |
| LSTM w/ NLL loss (variance estimation) — Martinsson et al. | 20.7 ± 3.2 | — | 33.6 ± 3.2 | — | paper10 |
| XGBoost — Midroni et al. | 20.377 | — | — | — | paper13 |
| Grammatical evolution — Contreras et al. | 21.19 | — | 31.34 | — | paper15 (also RMSE@90 = **36.26**) |
| WaveNet-style CNN — Zhu, Li, Herrero, Chen, Georgiou | 21.73 ± 2.52 (best-avg) | — | — | — | paper12 |
| **ZOH / persistence baseline** — Xie & Wang | **22.54 ± 2.25** | — | — | — | paper16, Table 4 |
| **t₀ "predict last value" baseline** — Martinsson et al. | **22.5 ± 2.2** | — | **36.6 ± 3.0** | — | paper10, Table 1 |

Best credible 2018-challenge RMSE@30 is therefore **≈18.9–19.0 mg/dL** (Chen et al.), and best 2018 RMSE@60
is **≈31.3–31.7** (Contreras, Bertachi). Note Chen et al. quote a best-of-10-seeds figure; their honest mean
is 19.04.

---

### A.4 Post-challenge papers on OhioT1DM (protocols differ — separate table)

| Model | RMSE@30 | MAE@30 | RMSE@60 | MAE@60 | Subjects | Protocol | Source | Confidence |
|---|---|---|---|---|---|---|---|---|
| **GARNN** (GATv2+GRU) — Piao et al. 2025 | 18.97 ± 0.06 | **13.34 ± 0.02** | — | — | 12 (all) | Official OhioT1DM train/test split; train further split 80/20 train/val; T=48 (4 h) history, H=6 | Neural Networks 185:107229 | VERIFIED-PRIMARY |
| NHiTS (their baseline) | 20.14 | 14.07 | — | — | 12 | same | ibid. | VERIFIED-PRIMARY |
| N-BEATS (their baseline) | 20.15 | 14.11 | — | — | 12 | same | ibid. | VERIFIED-PRIMARY |
| RETAIN / IMV-TENSOR (baselines) | 20.30 / 20.15 | 14.41 / 14.00 | — | — | 12 | same | ibid. | VERIFIED-PRIMARY |
| Linear regression (their baseline) | 22.19 | 15.92 | — | — | 12 | same | ibid. | VERIFIED-PRIMARY |
| **NPE + LSTM** (physiology-informed conv encoder) — Gu, Dang & Prioleau 2020 | **17.80** | — | — | — | 6 (2018 cohort, ages 40–60) | OhioT1DM, T=12 (60 min) history; per-subject | IEEE EMBC 2020 | VERIFIED-PRIMARY (RMSE read from PMC full text; no MAE reported) |
| **Zero-order hold (persistence)** — Prioleau et al. 2025 (Glucose-ML) | **23.27 ± 2.92** | — | — | — | 12 (all) | **Own protocol** — the whole dataset, not the official test split | arXiv:2507.14077, Table 2 | VERIFIED-PRIMARY |
| Simple linear-regression extrapolation baseline — ibid. | 28.08 ± 4.80 | — | — | — | 12 | same | ibid. | VERIFIED-PRIMARY |

The **closest thing to a physics-informed / physiologically-informed model on OhioT1DM with a citable number**
is Gu, Dang & Prioleau (2020) NPE+LSTM at RMSE@30 = 17.80 mg/dL on the 2018 cohort, and Bertachi et al. (2018)
physiological-model + ANN at RMSE@30 = 19.33 / RMSE@60 = 31.72. I found **no** paper on OhioT1DM that both
(a) calls itself a PINN with a physiology ODE embedded in the loss and (b) reports MAE on the official split.
That is a genuine gap you can claim.

---

### A.5 Long horizons (90 / 120 min)

| Horizon | Best value found | Source | Confidence |
|---|---|---|---|
| RMSE@90 | 36.26 mg/dL | Contreras et al., KDH 2018, CEUR Vol-2148 paper15 (grammatical evolution, 2018 cohort, official split) | VERIFIED-PRIMARY |
| MAE@90 | — | **no value found under the official protocol** | UNVERIFIED |
| RMSE@120 | 36.10 mg/dL, MAE@120 25.47 mg/dL (PatchTST) | Karagoz, Breton & El Fathi, IFAC Diabetes Technology Conf. 2025, arXiv:2505.08821 | VERIFIED-PRIMARY value, but **protocol is not comparable** — see A.6 |
| RMSE@240 | 46.49 / MAE 34.06 (PatchTST) | ibid. | same caveat |

**Do not put a 90- or 120-min OhioT1DM MAE in a comparison table.** There is no established protocol-matched
number. If you report those horizons, report them as your own results against your own persistence baseline
and say explicitly that no protocol-matched published comparator exists.

---

### A.6 ⚠️ Protocol-mismatch warnings — papers you must NOT table alongside A.2/A.3

1. **Karagoz, Breton & El Fathi 2025 (arXiv:2505.08821)** — reports RMSE@30 = 15.81, MAE@30 = 9.67 on
   OhioT1DM, better than every challenge entry. Three reasons it is not comparable, all read directly from
   the paper: (i) models are **trained on DCLP3 (n=112)** and OhioT1DM is used only as an **external test
   set**; (ii) they state "the complete OhioT1DM dataset was used for testing" — i.e. **not** the official
   test XMLs; (iii) critically, their RMSE/MAE definitions (their Eqs. 4–5) **sum over the whole predicted
   sequence index j**, so the reported "30-minute" error is the average error across *all* steps from 5 min to
   30 min, not the error *at* 30 min. That alone explains the apparent ~2.5 mg/dL advantage. **VERIFIED-PRIMARY.**
2. **"AWD-stacking" ensemble** reported in search results at RMSE@30 = 1.425 / MAE@30 = 0.721 mg/dL and
   RMSE@60 = 6.346. These are physically impossible for a 30-min forecast (they are below CGM sensor noise).
   Almost certainly leakage or a mislabelled horizon. **Do not cite.** (SECOND-HAND, not verified in source —
   and deliberately not chased.)
3. **"Ls-Encoder" transformer+LSTM** reported at RMSE@120 = 13.986 / MAE@120 = 6.986. A 2-hour-ahead MAE of
   7 mg/dL is below the persistence error at *5 minutes*. Broken protocol. **Do not cite.** (SECOND-HAND.)
4. **U-Net CNN** reported at MAE@30 = 8.4761 / MAE@60 = 14.0170, and a **"CRNN at RMSE@30 = 9.38 ± 0.71"**.
   Both are far outside the challenge distribution. Almost certainly self-defined random splits with
   train/test contamination across the 5-min sampling grid. **Do not cite without reading the split.**
   (SECOND-HAND.)
5. **Pooled vs per-patient reporting.** Every A.2/A.3 number is a *mean of per-subject errors*. A paper that
   pools all subjects' test points into one error pool gets a different (usually slightly different, sometimes
   materially different) number. Several post-challenge papers do not say which they did. Always check.
6. **Best-of-N-seeds reporting.** Chen et al. 2018 (18.91 vs 19.04) and Ma et al. 2020 explicitly report
   best-of-runs. Compare to mean-of-runs numbers only with a note.
7. **Missing-data handling.** Mayo & Koutny drop test windows with gaps; Rubin-Falcone ignore missing values in
   evaluation; Pavan et al. do not impute the test set and predict fewer than all available samples;
   Freiburghaus et al. report selected curves. The scored point set therefore varies between entries even on
   the "same" split.
8. **2018 ≠ 2020 scoring rule.** The 2020 protocol excludes the first hour of each test file; 2018 does not.
   Do not average across cohorts.
9. **Extra training data.** Rubin-Falcone et al. pre-train on Tidepool + 2018 cohort. Bevan & Coenen train
   non-personalised models on other patients. Mayo & Koutny's second round adds the 2018 cohort. All legal
   under challenge rules, but they are not "OhioT1DM-only" models.

---

### A.7 Direct answers

**Q1. What is the lowest credibly-published MAE at 30 min on OhioT1DM, and by whom?**

Three tiers:

- *Lowest number that exists in a peer-reviewed challenge paper on the official split:* **10.08 ± 2.10 mg/dL**,
  Pavan, Prendin, Meneghetti, Cappon, Sparacino, Facchinetti & Del Favero, NN-EIM, KDH@ECAI 2020, CEUR Vol-2675
  paper16. **VERIFIED-PRIMARY** — but see the ‡ anomaly flag: MAE/RMSE = 0.54 overall and 0.39 for one subject
  is out of family with all other entries, and the paper states the test set was not imputed and reports a
  reduced number of predicted samples. I would not use this as "the record" without replicating it.
- *Second lowest:* **11.22 mg/dL**, Freiburghaus, Rizzotti-Kaddouri & Albertetti, CEUR Vol-2675 paper23
  (paired with RMSE@30 = 17.45, also the lowest RMSE in the challenge). Same caution: single best
  configuration, and elsewhere in the paper they quote curve-selected figures.
- **The number I would actually cite as the credible state of the art: 12.83 mg/dL MAE@30 (with RMSE@30 =
  18.22), Rubin-Falcone, Fox & Wiens, "Deep Residual Time-Series Forecasting: Application to Blood Glucose
  Prediction", KDH@ECAI 2020, CEUR-WS Vol-2675 paper18.** Its MAE/RMSE ratio (0.70) is in family, it is the
  best RMSE among entries with clean protocol descriptions, it comes from a group with a strong methods
  record, and it is the most frequently treated-as-winner entry of the 2020 challenge. On the
  post-challenge/12-subject protocol, **GARNN (Piao et al. 2025, Neural Networks) at MAE@30 = 13.34** is the
  best I verified.

**Q2. Is MAE < 15 mg/dL at 30 min actually attained in the published literature? Yes/no + citation.**

**Yes — routinely, and by a large margin.** MAE@30 < 15 mg/dL is not a frontier; it is roughly the *median* of
the 2020 BGLP Challenge field. Of the 17 entries I tabulated in A.2, **15 report MAE@30 < 15 mg/dL**; only
Bhimireddy et al. (15.0) sits at the line. Concrete citations, all VERIFIED-PRIMARY:

- Rubin-Falcone, Fox & Wiens 2020 — 12.83 mg/dL (CEUR-WS Vol-2675, paper18)
- Zhu, Yao, Li, Herrero & Georgiou 2020 — 13.37 ± 0.18 mg/dL (Vol-2675, paper15)
- Piao et al. 2025 — 13.34 ± 0.02 mg/dL (Neural Networks 185:107229)
- Daniels, Herrero & Georgiou 2020 — 13.62 ± 0.05 mg/dL (Vol-2675, paper19)
- Bevan & Coenen 2020 — 14.37 mg/dL, with a *non-personalised* model (Vol-2675, paper17)
- Cappon et al. 2020 — 14.74 mg/dL (Vol-2675, paper12)

**Implication for your draft:** a paper that (a) asserts "<15 mg/dL MAE is clinically acceptable" and (b)
reports MAE@30 < 15 mg/dL as evidence of clinical usefulness is claiming a bar that even a *non-personalised*
LSTM from 2020 clears. Reviewers who know this field will flag it. Worse, Part B shows the threshold itself is
not citable. Drop the framing.

**Q3. Realistic best MAE at 60, 90, 120 min?**

| Horizon | Realistic best published MAE (official protocol) | Basis | Confidence |
|---|---|---|---|
| 60 min | **≈23.3–23.6 mg/dL** | Freiburghaus et al. 23.25; Rubin-Falcone et al. 23.60; Joedicke et al. 23.58; Yang et al. 23.833. Typical field value 24–26. | VERIFIED-PRIMARY |
| 90 min | **no protocol-matched MAE exists** (only RMSE@90 = 36.26, Contreras et al. 2018). If you must estimate a band from the RMSE and the field's MAE/RMSE ratio of ~0.72, that is *your* extrapolation, not a citation. | — | UNVERIFIED |
| 120 min | **no protocol-matched MAE exists.** The only value I found, 25.47 mg/dL (Karagoz et al. 2025), is under a sequence-averaged, external-test-set protocol and is not comparable — see A.6.1. | — | UNVERIFIED |

**Q4. Persistence-baseline RMSE and MAE at 30 and 60 min?**

| Baseline | RMSE@30 | MAE@30 | RMSE@60 | MAE@60 | Subjects / protocol | Source | Confidence |
|---|---|---|---|---|---|---|---|
| t₀ = predict last value | **22.5 ± 2.2** | — | **36.6 ± 3.0** | — | 2018 cohort, official split | Martinsson et al., KDH 2018, CEUR Vol-2148 paper10, Table 1 | VERIFIED-PRIMARY |
| Zero-order hold (ZOH) | **22.54 ± 2.25** | — | — | — | 2018 cohort, official split | Xie & Wang, KDH 2018, CEUR Vol-2148 paper16, Table 4 | VERIFIED-PRIMARY |
| Zero-order hold | **23.27 ± 2.92** | — | — | — | all 12 subjects, **own protocol** (whole dataset) | Prioleau et al. 2025, arXiv:2507.14077, Table 2 | VERIFIED-PRIMARY |
| Simple linear extrapolation | 28.08 ± 4.80 | — | — | — | all 12, own protocol | ibid. | VERIFIED-PRIMARY |

So **persistence RMSE@30 ≈ 22.5 mg/dL and RMSE@60 ≈ 36.6 mg/dL** are solidly citable, and two fully
independent 2018 sources agree on the 30-min value to within 0.05 mg/dL — that is unusually strong
corroboration and you should lean on it.

**Persistence MAE at 30 and 60 min: UNVERIFIED — I could not find it published anywhere.** Martinsson's
journal extension (Martinsson et al., *J. Healthcare Informatics Research* 4:1–18, 2020,
doi:10.1007/s41666-019-00059-y) also reports RMSE only for the t₀ baseline; I checked. Glucose-ML reports RMSE
only. **Recommendation:** compute it yourself on the official test XMLs (a five-line computation) and report it
as "computed by the authors on the official BGLP test split", citing Marling & Bunescu 2020 for the split and
Martinsson et al. 2018 / Xie & Wang 2018 for the RMSE cross-check that validates your implementation. Do not
guess the MAE from the RMSE.

**Q5. Papers whose protocol differs from the official split.** See §A.6 — nine distinct failure modes,
itemised, with the four "do not cite" papers named.

---

## PART B — Is "<15 mg/dL MAE is clinically acceptable" a citable threshold for forecasting?

### B.1 ISO 15197:2013 — what it actually says, and what it governs

**Scope: self-monitoring blood glucose (SMBG) *meters* — i.e. capillary blood glucose measurement systems for
in-vitro self-testing. It is a measurement-accuracy standard. It says nothing about prediction.**

Minimum system accuracy criteria (**SECOND-HAND** — consistently and identically quoted across multiple
peer-reviewed ISO-15197 evaluation papers; I did not purchase the standard itself, which is paywalled):

- ≥ 95 % of measured results within **± 15 mg/dL** of the reference method at BG **< 100 mg/dL**, and within
  **± 15 %** at BG **≥ 100 mg/dL**;
- ≥ 99 % of individual results within **zones A and B of the Consensus (Parkes) Error Grid**;
- assessed over > 100 samples, in duplicate, across 3 test-strip lots.

Note carefully what the ±15 mg/dL is: a **per-measurement tolerance with a 95 % coverage requirement, applied
only below 100 mg/dL, against a laboratory reference, for a device measuring glucose *now*.** It is not a mean
error, not an MAE, and not a statement about any horizon. Converting "95 % of readings within ±15 mg/dL below
100 mg/dL" into "MAE < 15 mg/dL" is a category error twice over: it changes a quantile bound into a mean, and
it changes measurement into forecasting.

(The earlier ISO 15197:2003 criteria were the looser ±15 mg/dL below 75 mg/dL / ±20 % above — you will see
this quoted in older papers; do not mix them up.)

### B.2 FDA iCGM special controls (2018) — 21 CFR 862.1355

**Scope: integrated continuous glucose monitoring *sensors* whose output may be used by connected devices
(e.g. automated insulin dosing). Again a measurement standard; no prediction requirement.** Established
2018-03-27 with the de novo authorisation of Dexcom G6 (DEN170088).

Accuracy special controls, quoted from 21 CFR 862.1355(b) (**VERIFIED-PRIMARY**, read from the CFR text):

| Stratum | Criterion | Lower one-sided 95 % confidence bound must exceed |
|---|---|---|
| BG < 70 mg/dL | within ± 15 mg/dL | 85 % |
| BG < 70 mg/dL | within ± 40 mg/dL | 98 % |
| BG 70–180 mg/dL | within ± 15 % | 70 % |
| BG 70–180 mg/dL | within ± 40 % | 99 % |
| BG > 180 mg/dL | within ± 15 % | 80 % |
| BG > 180 mg/dL | within ± 40 % | 99 % |
| Whole measuring range | within ± 20 % | 87 % |

No MARD requirement is stated in the regulation. Note how *loose* these are relative to your draft's claim:
even a cleared iCGM sensor need only get **70 %** of its in-range readings within ±15 % — and it is measuring
glucose at the present instant. Any forecast necessarily inherits this sensor error *plus* physiological
uncertainty over the horizon.

### B.3 Clarke / Kovatchev / prediction-specific error grids

These are the correct literature for "clinically acceptable", and none of them is an MAE threshold.

- **Clarke EGA (1987).** Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. "Evaluating clinical
  accuracy of systems for self-monitoring of blood glucose." *Diabetes Care* 10(5):622–628.
  doi:10.2337/diacare.10.5.622. Defines zones A–E; **zones A + B = "clinically acceptable"**, i.e.
  acceptability is defined by whether the error would lead to a wrong or dangerous *treatment decision*, as a
  function of the true glucose value — deliberately *not* by a single error magnitude. **VERIFIED-PRIMARY**
  (bibliographic details and DOI confirmed at the ADA journal record).
- **Continuous glucose-EGA (2004).** Kovatchev BP, Gonder-Frederick LA, Cox DJ, Clarke WL. "Evaluating the
  accuracy of continuous glucose-monitoring sensors: continuous glucose-error grid analysis used with clinical
  distinction of hypo- and hyperglycemia." *Diabetes Care* 27(8):1922–1928. Extends EGA to *rates of change* /
  trends, still for sensors. **SECOND-HAND** (PubMed record; not read in full).
- **PRED-EGA (2011) — the one that is actually about predictors.** Sivananthan S, Naumova V, Dalla Man C,
  Facchinetti A, Renard E, Cobelli C, Pereverzyev SV. "Assessment of blood glucose predictors: the
  prediction-error grid analysis." *Diabetes Technology & Therapeutics* 13(8):787–796. Purpose-built to assess
  *forecasts* rather than measurements, and explicitly motivated by the fact that CG-EGA misclassifies
  predictor output. **SECOND-HAND** (bibliographic details confirmed via Semantic Scholar / Wikidata; I did
  not read the full text). **This is the reference you want if you need a "clinical acceptability" criterion
  for a forecaster.**

### B.4 ADA / ATTD consensus on prediction error tolerance

**None exists, as far as I can determine.** The ATTD 2019 international consensus (Battelino et al., "Clinical
Targets for Continuous Glucose Monitoring Data Interpretation: Recommendations From the International Consensus
on Time in Range", *Diabetes Care* 42(8):1593–1603, 2019) standardises *glycaemic* metrics — time in range,
GMI, variability — not algorithm error tolerances. ADA *Standards of Care* §6 (Glycemic Goals and Hypoglycemia)
likewise sets glycaemic targets, not predictor accuracy targets. I searched specifically for an ADA/ATTD
prediction-accuracy consensus and found nothing. Confidence: **SECOND-HAND absence** — I am confident enough to
assert it, but if a reviewer produces one I would not be shocked.

What does exist, and is directly on point for your framing, is a 2025 position/analysis paper:

**Wolff MK, Schaathun HG, Gros S, Volden R, Steinert M, Fougner AL. "Blood Glucose Prediction Algorithms
Require Clinically Relevant Performance Criteria Beyond Accuracy." *Diabetes Technology & Therapeutics*
27(10), Oct 2025; online 2025-04-29. doi:10.1089/dia.2025.0074, PMID 40300777.** Its thesis: RMSE (and by
extension MAE) rewards models that predict the target range, because target-range values dominate the data,
and therefore *systematically* under-rewards detection of the clinically important events — rapid excursions,
hypoglycaemia, hyperglycaemia. Conclusion: aggregate accuracy is the wrong acceptance criterion for a glucose
predictor. **SECOND-HAND** (abstract + PubMed record; full text not read).

### B.5 ⚖️ VERDICT

**No. "<15 mg/dL MAE is generally regarded as clinically acceptable" is not a citable claim for glucose
forecasting, and I could not find any source that supports it. It is a misapplied measurement-accuracy
figure.** Specifically, the "15" almost certainly leaks in from ISO 15197:2013's ±15 mg/dL — which is (i) a
per-reading tolerance under a 95 % coverage requirement, not a mean; (ii) applicable only below 100 mg/dL;
(iii) a requirement on an *in-vitro capillary blood glucose meter measuring the present*, not on a forecast at
any horizon. No standard (ISO 15197, FDA 21 CFR 862.1355), no error-grid paper (Clarke 1987, Kovatchev 2004,
Sivananthan 2011), and no ADA/ATTD consensus states an MAE threshold for prediction. Compounding the problem,
MAE@30 < 15 mg/dL is met by ~15 of 17 entries in the 2020 BGLP Challenge (§A.7 Q2), including a
non-personalised LSTM — so even if the threshold were citable it would carry no discriminative weight.

**If you keep any variant of it, the only honest sentence is:** *"ISO 15197:2013 requires that 95 % of SMBG
meter readings fall within ±15 mg/dL of reference below 100 mg/dL; we note this is a measurement-accuracy
requirement for a device reporting current glucose and is not a validated acceptance criterion for
forecasts."* That is defensible as context and indefensible as a target.

**Proposed defensible replacement framing — use all three, they are cheap and each is independently
citable:**

1. **Skill relative to persistence (primary claim).** Report a skill score
   `1 − MAE_model / MAE_persistence` at each horizon, with persistence computed by you on the official test
   split. Persistence is the standard naive baseline for CGM forecasting (Prioleau et al. 2025 explicitly call
   including a zero-order-hold predictor a *best-practice guideline*; Martinsson et al. 2018 and Xie & Wang
   2018 both use it), and its RMSE@30 = 22.5 / RMSE@60 = 36.6 mg/dL are independently corroborated
   (§A.7 Q4). This makes your claim protocol-invariant and reviewer-proof: "we reduce 30-min MAE by X % versus
   persistence" survives any disagreement about split conventions.
2. **Clinical acceptability via an error grid designed for predictors, not for meters.** Report
   **PRED-EGA** (Sivananthan et al. 2011, *Diabetes Technol Ther* 13(8):787–796) — accurate / benign /
   erroneous rates stratified by hypo / euglycaemia / hyper. If you prefer the more familiar grid, report
   Clarke EGA zone A+B % and cite Clarke et al. 1987 for the fact that zones A+B *are* the definition of
   clinically acceptable, rather than inventing a magnitude threshold. Most 2020 BGLP entries report EGA
   zones, so this is also directly comparable to the field.
3. **Horizon-stratified, field-relative targets instead of an absolute threshold.** State targets as
   "at or below the best published error on the official BGLP protocol at each horizon": MAE@30 ≤ 12.8,
   MAE@60 ≤ 23.3 mg/dL (§A.7 Q1, Q3), and for 90/120 min state plainly that no protocol-matched published
   comparator exists. Cite Wolff et al. 2025 (doi:10.1089/dia.2025.0074) for *why* you are additionally
   reporting event-level metrics (hypo/hyper detection sensitivity, time-gain) rather than resting on an
   aggregate error threshold — this pre-empts the exact reviewer objection your current framing invites.

---

## BibTeX — verified sources only

```bibtex
@inproceedings{marling2020ohiot1dm,
  author    = {Marling, Cindy and Bunescu, Razvan},
  title     = {The {OhioT1DM} Dataset for Blood Glucose Level Prediction: Update 2020},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  pages     = {71--74},
  year      = {2020},
  address   = {Santiago de Compostela, Spain},
  url       = {https://ceur-ws.org/Vol-2675/paper11.pdf},
  note      = {VERIFIED-PRIMARY}
}

@inproceedings{marling2018ohiot1dm,
  author    = {Marling, Cindy and Bunescu, Razvan},
  title     = {The {OhioT1DM} Dataset for Blood Glucose Level Prediction},
  booktitle = {Proc. 3rd Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2148},
  year      = {2018},
  address   = {Stockholm, Sweden},
  url       = {https://ceur-ws.org/Vol-2148/paper09.pdf},
  note      = {VERIFIED-PRIMARY (superseded by the 2020 update paper)}
}

@inproceedings{rubinfalcone2020residual,
  author    = {Rubin-Falcone, Harry and Fox, Ian and Wiens, Jenna},
  title     = {Deep Residual Time-Series Forecasting: Application to Blood Glucose Prediction},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper18.pdf},
  note      = {rMSE 18.22/31.66, MAE 12.83/23.60 mg/dL at 30/60 min, 2020 cohort. VERIFIED-PRIMARY}
}

@inproceedings{zhu2020gan,
  author    = {Zhu, Taiyu and Yao, Xi and Li, Kezhi and Herrero, Pau and Georgiou, Pantelis},
  title     = {Blood Glucose Prediction for Type 1 Diabetes Using Generative Adversarial Networks},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper15.pdf},
  note      = {RMSE 18.34/32.31, MAE 13.37/24.20 mg/dL. VERIFIED-PRIMARY}
}

@inproceedings{bevan2020nonpersonalized,
  author    = {Bevan, Robert and Coenen, Frans},
  title     = {Experiments in Non-Personalized Future Blood Glucose Level Prediction},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper17.pdf},
  note      = {RMSE 18.23/31.10, MAE 14.37/25.75 mg/dL; non-personalized. VERIFIED-PRIMARY}
}

@inproceedings{cappon2020interpretable,
  author    = {Cappon, Giacomo and Meneghetti, Lorenzo and Prendin, Francesco and Pavan, Jacopo
               and Sparacino, Giovanni and Del Favero, Simone and Facchinetti, Andrea},
  title     = {A Personalized and Interpretable Deep Learning Based Approach to Predict Blood
               Glucose Concentration in Type 1 Diabetes},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper12.pdf},
  note      = {RMSE 20.20/34.19, MAE 14.74/25.98 mg/dL, TG 9.17/18.33 min. VERIFIED-PRIMARY}
}

@inproceedings{pavan2020shallow,
  author    = {Pavan, Jacopo and Prendin, Francesco and Meneghetti, Lorenzo and Cappon, Giacomo
               and Sparacino, Giovanni and Facchinetti, Andrea and Del Favero, Simone},
  title     = {Personalized Machine Learning Algorithm based on Shallow Network and Error
               Imputation Module for an Improved Blood Glucose Prediction},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper16.pdf},
  note      = {RMSE 18.63/32.27, MAE 10.08/17.69 mg/dL. VERIFIED-PRIMARY but MAE/RMSE ratio anomalous}
}

@inproceedings{mayo2020multiclass,
  author    = {Mayo, Michael and Koutny, Tomas},
  title     = {Neural Multi-class Classification Approach to Blood Glucose Level Forecasting
               with Prediction Uncertainty Visualisation},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper13.pdf},
  note      = {2018 cohort MAE 14.3/24.2, RMSE 20.7/32.8; 2020 cohort MAE 14.4/25.8, RMSE 19.8/34.0.
               Drops test windows with gaps. VERIFIED-PRIMARY}
}

@inproceedings{daniels2020multitask,
  author    = {Daniels, John and Herrero, Pau and Georgiou, Pantelis},
  title     = {Personalised Glucose Prediction via Deep Multitask Networks},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper19.pdf},
  note      = {MTCRNN: RMSE 19.79/33.73, MAE 13.62/24.54 mg/dL, 2020 cohort. VERIFIED-PRIMARY}
}

@inproceedings{nemat2020datafusion,
  author    = {Nemat, Hoda and Khadem, Heydar and Elliott, Jackie and Benaissa, Mohammed},
  title     = {Data Fusion of Activity and {CGM} for Predicting Blood Glucose Levels},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper21.pdf},
  note      = {Method 1: RMSE 18.99/33.39, MAE 13.73/25.04 mg/dL. VERIFIED-PRIMARY}
}

@inproceedings{khadem2020multilag,
  author    = {Khadem, Heydar and Nemat, Hoda and Elliott, Jackie and Benaissa, Mohammed},
  title     = {Multi-lag Stacking for Blood Glucose Level Prediction},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper26.pdf},
  note      = {System 2: RMSE 19.21/33.65, MAE 13.93/25.31 mg/dL. VERIFIED-PRIMARY}
}

@inproceedings{yang2020mslstm,
  author    = {Yang, Tao and Wu, Ruikun and Tao, Rui and Wen, Shuang and Ma, Ning and Zhao, Yuhang
               and Yu, Xia and Li, Hongru},
  title     = {Multi-Scale Long Short-Term Memory Network with Multi-Lag Structure for Blood
               Glucose Prediction},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper24.pdf},
  note      = {RMSE 19.048/32.029, MAE 13.503/23.833 mg/dL. VERIFIED-PRIMARY}
}

@inproceedings{sun2020latent,
  author    = {Sun, Xiaoyu and Rashid, Mudassir and Sevil, Mert and Hobbs, Nicole and Brandt, Rachel
               and Askari, Mohammad Reza and Shahidehpour, Andrew and Cinar, Ali},
  title     = {Prediction of Blood Glucose Levels for People with Type 1 Diabetes using
               Latent-Variable-based Model},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper20.pdf},
  note      = {RMSE 19.37/32.59, MAE 13.76/24.64 mg/dL. VERIFIED-PRIMARY}
}

@inproceedings{freiburghaus2020deep,
  author    = {Freiburghaus, Jonas and Rizzotti-Kaddouri, A{\"i}cha and Albertetti, Fabrizio},
  title     = {A Deep Learning Approach for Blood Glucose Prediction of Type 1 Diabetes},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper23.pdf},
  note      = {RMSE 17.45/33.67, MAE 11.22/23.25 mg/dL. Lowest challenge RMSE@30; single best
               configuration. VERIFIED-PRIMARY}
}

@inproceedings{bhimireddy2020seq2seq,
  author    = {Bhimireddy, Ananth Reddy and Sinha, Priyanshu and Oluwalade, Bolu
               and Gichoya, Judy Wawira and Purkayastha, Saptarshi},
  title     = {Blood Glucose Level Prediction as Time-Series Modeling using Sequence-to-Sequence
               Neural Networks},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper22.pdf},
  note      = {BiLSTM: RMSE 21.8/35.0, MAE 15.0/25.0 mg/dL. VERIFIED-PRIMARY}
}

@inproceedings{ma2020arma,
  author    = {Ma, Ning and Zhao, Yuhang and Wen, Shuang and Yang, Tao and Wu, Ruikun and Tao, Rui
               and Yu, Xia and Li, Hongru},
  title     = {Online Blood Glucose Prediction Using Autoregressive Moving Average Model with
               Residual Compensation Network},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper27.pdf},
  note      = {Best RMSE 20.03/34.89, best MAE 14.52/24.61 mg/dL (best-of-runs). VERIFIED-PRIMARY}
}

@inproceedings{joedicke2020gp,
  author    = {Joedicke, David and Garnica, Oscar and Kronberger, Gabriel and Colmenar, J. Manuel
               and Winkler, Stephan and Velasco, Jose Manuel and Contador, Sergio and Hidalgo, J. Ignacio},
  title     = {Analysis of the Performance of Genetic Programming on the Blood Glucose Level
               Prediction Challenge 2020},
  booktitle = {Proc. 5th Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@ECAI 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2675},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2675/paper25.pdf},
  note      = {Best variant approx. RMSE 19.60/32.04, MAE 14.25/23.58 mg/dL; multi-variant table,
               lower confidence on variant attribution}
}

@inproceedings{martinsson2018rnn,
  author    = {Martinsson, John and Schliep, Alexander and Eliasson, Bj{\"o}rn and Meijner, Christian
               and Persson, Simon and Mogren, Olof},
  title     = {Automatic Blood Glucose Prediction with Confidence Using Recurrent Neural Networks},
  booktitle = {Proc. 3rd Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2148},
  year      = {2018},
  url       = {https://ceur-ws.org/Vol-2148/paper10.pdf},
  note      = {*** PERSISTENCE BASELINE SOURCE: t0 (predict last value) RMSE 22.5 +/- 2.2 at 30 min,
               36.6 +/- 3.0 at 60 min, 2018 cohort, official split. Model RMSE 20.1/33.2 (MSE loss).
               No MAE reported. VERIFIED-PRIMARY}
}

@article{martinsson2020variance,
  author  = {Martinsson, John and Schliep, Alexander and Eliasson, Bj{\"o}rn and Mogren, Olof},
  title   = {Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks},
  journal = {Journal of Healthcare Informatics Research},
  volume  = {4},
  pages   = {1--18},
  year    = {2020},
  doi     = {10.1007/s41666-019-00059-y},
  note    = {Journal extension of martinsson2018rnn; also reports RMSE only for the t0 baseline}
}

@inproceedings{xie2018benchmark,
  author    = {Xie, Jinyu and Wang, Qian},
  title     = {Benchmark Machine Learning Approaches with Classical Time Series Approaches on the
               Blood Glucose Level Prediction Challenge},
  booktitle = {Proc. 3rd Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2148},
  year      = {2018},
  url       = {https://ceur-ws.org/Vol-2148/paper16.pdf},
  note      = {*** PERSISTENCE BASELINE SOURCE: ZOH RMSE@30 = 22.54 +/- 2.25 mg/dL; best model
               (SVR-RBF, recursive) 19.53. 2018 cohort, official split. No MAE. VERIFIED-PRIMARY}
}

@inproceedings{chen2018dilated,
  author    = {Chen, Jianwei and Li, Kezhi and Herrero, Pau and Zhu, Taiyu and Georgiou, Pantelis},
  title     = {Dilated Recurrent Neural Network for Short-Time Prediction of Glucose Concentration},
  booktitle = {Proc. 3rd Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2148},
  year      = {2018},
  url       = {https://ceur-ws.org/Vol-2148/paper11.pdf},
  note      = {Best RMSE@30 = 18.9066 (best-of-10 seeds); mean-of-10 = 19.0417. Lowest 2018-challenge
               RMSE@30 found. VERIFIED-PRIMARY}
}

@inproceedings{bertachi2018physiological,
  author    = {Bertachi, Arthur and Biagi, Lyvia and Contreras, Iv{\'a}n and Luo, Ningsu and Veh{\'i}, Josep},
  title     = {Prediction of Blood Glucose Levels and Nocturnal Hypoglycemia Using Physiological
               Models and Artificial Neural Networks},
  booktitle = {Proc. 3rd Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2148},
  year      = {2018},
  url       = {https://ceur-ws.org/Vol-2148/paper14.pdf},
  note      = {RMSE 19.33/31.72 mg/dL at 30/60 min. Physiology-informed. VERIFIED-PRIMARY}
}

@inproceedings{contreras2018grammatical,
  author    = {Contreras, Iv{\'a}n and Bertachi, Arthur and Biagi, Lyvia and Veh{\'i}, Josep
               and Oviedo, Silvia},
  title     = {Using Grammatical Evolution to Generate Short-Term Blood Glucose Prediction Models},
  booktitle = {Proc. 3rd Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2148},
  year      = {2018},
  url       = {https://ceur-ws.org/Vol-2148/paper15.pdf},
  note      = {RMSE 21.19/31.34/36.26 mg/dL at 30/60/90 min (gRMSE 24.83/32.39/40.23). Only
               official-protocol 90-min value found. VERIFIED-PRIMARY}
}

@inproceedings{midroni2018xgboost,
  author    = {Midroni, Cooper and Leimbigler, Peter J. and Baruah, Gaurav and Kolla, Maheedhar
               and Whitehead, Alfred and Fossat, Yan},
  title     = {Predicting Glycemia in Type 1 Diabetes Patients: Experiments with {XGBoost}},
  booktitle = {Proc. 3rd Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2148},
  year      = {2018},
  url       = {https://ceur-ws.org/Vol-2148/paper13.pdf},
  note      = {RMSE@30 = 20.377 mg/dL. VERIFIED-PRIMARY}
}

@inproceedings{zhu2018wavenet,
  author    = {Zhu, Taiyu and Li, Kezhi and Herrero, Pau and Chen, Jianwei and Georgiou, Pantelis},
  title     = {A Deep Learning Algorithm for Personalized Blood Glucose Prediction},
  booktitle = {Proc. 3rd Int. Workshop on Knowledge Discovery in Healthcare Data (KDH@IJCAI-ECAI 2018)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2148},
  year      = {2018},
  url       = {https://ceur-ws.org/Vol-2148/paper12.pdf},
  note      = {Best-average RMSE@30 = 21.73 +/- 2.52 mg/dL. VERIFIED-PRIMARY}
}

@article{piao2025garnn,
  author  = {Piao, Chengzhe and Zhu, Taiyu and Baldeweg, Stephanie E. and Taylor, Paul
             and Georgiou, Pantelis and Sun, Jiahao and Wang, Jun and Li, Kezhi},
  title   = {{GARNN}: An Interpretable Graph Attentive Recurrent Neural Network for Predicting
             Blood Glucose Levels via Multivariate Time Series},
  journal = {Neural Networks},
  volume  = {185},
  pages   = {107229},
  year    = {2025},
  doi     = {10.1016/j.neunet.2025.107229},
  note    = {arXiv:2402.16230. OhioT1DM, official split, all 12 subjects, H=30 min:
             GATv2+GRU RMSE 18.97, MAE 13.34, gRMSE 23.65 mg/dL; 12 baselines tabulated.
             VERIFIED-PRIMARY (from arXiv version)}
}

@inproceedings{gu2020npe,
  author    = {Gu, Kang and Dang, Ruoqi and Prioleau, Temiloluwa},
  title     = {Neural Physiological Model: A Simple Module for Blood Glucose Prediction},
  booktitle = {42nd Annual Int. Conf. of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  pages     = {5476--5479},
  year      = {2020},
  doi       = {10.1109/EMBC44109.2020.9176004},
  note      = {PMID 33019219. NPE+LSTM RMSE@30 = 17.80 mg/dL on OhioT1DM (6 subjects, 2018 cohort).
               No MAE reported. VERIFIED-PRIMARY}
}

@article{prioleau2025glucoseml,
  author  = {Prioleau, Temiloluwa and Lu, Baiying and Cui, Yanjun},
  title   = {Glucose-{ML}: A Collection of Longitudinal Diabetes Datasets for Development of
             Robust {AI} Solutions},
  journal = {arXiv preprint},
  volume  = {arXiv:2507.14077},
  year    = {2025},
  note    = {*** PERSISTENCE BASELINE SOURCE (non-official protocol): zero-order-hold RMSE@30 =
             23.27 +/- 2.92 mg/dL on OhioT1DM (all 12 subjects, whole dataset). Also states that
             including a ZOH naive baseline is a best-practice guideline. VERIFIED-PRIMARY}
}

@inproceedings{karagoz2025transformers,
  author    = {Karagoz, Meryem Altin and Breton, Marc D. and El Fathi, Anas},
  title     = {A Comparative Study of Transformer-Based Models for Multi-Horizon Blood Glucose
               Prediction},
  booktitle = {IFAC Diabetes Technology Conference},
  year      = {2025},
  note      = {arXiv:2505.08821. DO NOT TABLE ALONGSIDE BGLP RESULTS: trained on DCLP3, OhioT1DM
               used as external test set over the complete dataset, and errors are averaged over the
               whole predicted sequence rather than at the horizon endpoint. Values:
               Crossformer 30 min RMSE 15.81 / MAE 9.67; PatchTST 60 min 24.62/16.38,
               120 min 36.10/25.47, 240 min 46.49/34.06. VERIFIED-PRIMARY values, invalid comparison}
}

@article{clarke1987ega,
  author  = {Clarke, William L. and Cox, Daniel and Gonder-Frederick, Linda A. and Carter, William
             and Pohl, Stephen L.},
  title   = {Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose},
  journal = {Diabetes Care},
  volume  = {10},
  number  = {5},
  pages   = {622--628},
  year    = {1987},
  doi     = {10.2337/diacare.10.5.622},
  note    = {Clarke Error Grid. Zones A+B = clinically acceptable. Defines acceptability by treatment
             consequence, not by an error magnitude. VERIFIED-PRIMARY (bibliographic record + DOI)}
}

@article{kovatchev2004cgega,
  author  = {Kovatchev, Boris P. and Gonder-Frederick, Linda A. and Cox, Daniel J. and Clarke, William L.},
  title   = {Evaluating the Accuracy of Continuous Glucose-Monitoring Sensors: Continuous
             Glucose-Error Grid Analysis Used with Clinical Distinction of Hypo- and Hyperglycemia},
  journal = {Diabetes Care},
  volume  = {27},
  number  = {8},
  pages   = {1922--1928},
  year    = {2004},
  note    = {SECOND-HAND: bibliographic record confirmed, full text not read. Verify page range and
             DOI before submission}
}

@article{sivananthan2011predega,
  author  = {Sivananthan, Sivananthan and Naumova, Valeriya and Dalla Man, Chiara
             and Facchinetti, Andrea and Renard, Eric and Cobelli, Claudio
             and Pereverzyev, Sergei V.},
  title   = {Assessment of Blood Glucose Predictors: The Prediction-Error Grid Analysis},
  journal = {Diabetes Technology \& Therapeutics},
  volume  = {13},
  number  = {8},
  pages   = {787--796},
  year    = {2011},
  note    = {PRED-EGA. The error grid designed specifically for evaluating PREDICTORS rather than
             sensors. SECOND-HAND: bibliographic record confirmed via Semantic Scholar/Wikidata,
             full text not read. RECOMMENDED as the clinical-acceptability citation for forecasting}
}

@article{wolff2025beyondaccuracy,
  author  = {Wolff, Miriam K. and Schaathun, Hans Georg and Gros, Sebastien and Volden, Rune
             and Steinert, Martin and Fougner, Anders L.},
  title   = {Blood Glucose Prediction Algorithms Require Clinically Relevant Performance Criteria
             Beyond Accuracy},
  journal = {Diabetes Technology \& Therapeutics},
  volume  = {27},
  number  = {10},
  year    = {2025},
  doi     = {10.1089/dia.2025.0074},
  note    = {PMID 40300777, online 2025-04-29. Argues RMSE/MAE systematically reward target-range
             prediction and under-reward detection of clinically critical events. SECOND-HAND:
             abstract and PubMed record only}
}

@misc{cfr862_1355,
  title        = {21 {CFR} 862.1355 --- Integrated Continuous Glucose Monitoring System},
  author       = {{U.S. Food and Drug Administration}},
  year         = {2018},
  howpublished = {Code of Federal Regulations, Title 21},
  note         = {Special controls established 2018-03-27 with de novo DEN170088 (Dexcom G6).
                  Accuracy criteria quoted in Part B.2. VERIFIED-PRIMARY (CFR text read)}
}

@misc{iso15197_2013,
  title        = {{ISO} 15197:2013 --- In Vitro Diagnostic Test Systems: Requirements for Blood-Glucose
                  Monitoring Systems for Self-Testing in Managing Diabetes Mellitus},
  author       = {{International Organization for Standardization}},
  year         = {2013},
  address      = {Geneva},
  note         = {Governs SMBG METERS, not CGM sensors and not predictions. 95% of results within
                  +/-15 mg/dL below 100 mg/dL and +/-15% at or above 100 mg/dL; 99% in Consensus
                  Error Grid zones A+B. SECOND-HAND: standard is paywalled; criteria confirmed via
                  multiple peer-reviewed ISO-15197 evaluation papers}
}

@article{battelino2019tir,
  author  = {Battelino, Tadej and Danne, Thomas and Bergenstal, Richard M. and others},
  title   = {Clinical Targets for Continuous Glucose Monitoring Data Interpretation:
             Recommendations From the International Consensus on Time in Range},
  journal = {Diabetes Care},
  volume  = {42},
  number  = {8},
  pages   = {1593--1603},
  year    = {2019},
  note    = {ATTD 2019 consensus. Standardises GLYCAEMIC metrics (TIR, GMI, variability).
             Contains NO predictor-accuracy criterion. SECOND-HAND; cite only to support the
             statement that no prediction-error consensus exists}
}
```

---

## UNVERIFIED / COULD NOT CONFIRM

1. **Official BGLP Challenge rules documents.** The "BGLP Challenge Rules" PDF linked from the KDH-2020 site
   was not retrievable, and the KDH-2018 challenge page is now behind a Google account login. The 30/60-min
   requirement for 2020 is confirmed from the KDH-2020 page text; **the 2018 required horizon set is inferred
   from what entrants reported (30 min universally, 60 min by some) and is UNVERIFIED.**
2. **Official leaderboards / winner declarations.** No ranking document exists in CEUR Vol-2148 or Vol-2675.
   Any claim of the form "the winning entry achieved X" is a reconstruction. I have deliberately written
   "best published entry" rather than "winner" throughout.
3. **Persistence / zero-order-hold MAE on OhioT1DM at any horizon.** Not published anywhere I could find.
   Martinsson 2018, Martinsson 2020 (journal), Xie & Wang 2018, and Prioleau 2025 all report persistence RMSE
   only. **Compute it yourself; do not cite a number for it.**
4. **Persistence RMSE at 90 and 120 min.** Not found. Only 30 min (three sources) and 60 min (one source,
   Martinsson).
5. **MAE at 90 min and at 120 min under the official BGLP protocol.** Not found (see A.5).
6. **RMSE at 120 min under the official BGLP protocol.** Not found.
7. **Exact page range / DOI for Kovatchev et al. 2004 CG-EGA.** Bibliographic record confirmed via PubMed
   listing only; verify before submission.
8. **PRED-EGA full text (Sivananthan et al. 2011).** Bibliographic details confirmed; the actual acceptability
   bands and the exact classification procedure were not read. Read it before you build a metric on it.
9. **ISO 15197:2013 verbatim clause text.** Paywalled; criteria taken from multiple concordant peer-reviewed
   evaluation papers rather than the standard itself. The concordance is strong (identical wording across
   independent groups) but it is formally SECOND-HAND.
10. **Absence of an ADA/ATTD/EASD consensus on prediction-error tolerance.** Asserted after targeted searching,
    but absence-of-evidence. If a reviewer demands a citation for the absence, cite Wolff et al. 2025, whose
    entire premise is that no adequate criteria are established.
11. **Wolff et al. 2025 full text.** Abstract only. Read before relying on it for anything beyond the general
    thesis.
12. **Joedicke et al. 2020 (genetic programming) variant attribution.** Their tables use comma decimal
    separators and many method columns; I extracted the best value per metric but am not certain which named
    variant each belongs to.
13. **Daniels et al. 2020 cohort scope.** Their abstract describes OhioT1DM as "12 participants", but their
    results table lists exactly 540/544/552/567/584/596. I have tabled them as 2020-cohort (6 subjects), which
    matches the table; the abstract wording is loose.
14. **"AWD-stacking", "Ls-Encoder", "U-Net MAE 8.48", "CRNN RMSE 9.38".** Not verified in primary sources by
    design — the reported magnitudes are physically implausible for the stated horizons. Listed here so you
    know they exist in the literature and will appear in other papers' comparison tables. Do not propagate them.
