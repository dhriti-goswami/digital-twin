# For Samyak: open items after SAMYAK_REMAINING.md (commit 45017eb)

Already applied in this branch (from your answers; please confirm):
- Table V + `data/comprehensive_baselines.csv`: A7 RMSE -> 19.06±2.51 / 30.89±3.89 / 39.04±4.83 / 44.78±5.46;
  ARX 90/120 -> MAE 32.24±3.47 / 37.68±3.69, RMSE 41.82±4.47 / 48.00±4.77 (from `results/tables/a7_vs_arx_wilcoxon.csv`).
- Sec. II-B, after Eq. (1): autocorrelation-based n_eff ≈ 2,500–3,100 (mean tau_int = 53.6 steps), citing
  Bayley & Hammersley (1946) — new reference [40]; later references and supplement citations renumbered.
- A7 vs ARX p-values NOT added to the paper (RMSE at 30 min not significant).

## Status & Completed Answers

### 1. Regenerate the whole Table V from scripts
**Confirmed and Completed.**
A master execution script `scripts/generate_comprehensive_baselines.py` evaluates all 9 models on the exact identical 26,498 test evaluation windows. Both `paper/digital-twin-v4/data/comprehensive_baselines.csv` and `results/tables/comprehensive_baselines.csv` have been updated and match Table V in `main.tex` to every digit.

**Exact numbers across all 9 models:**
- **Persistence:**
  - 30 min: MAE 16.87 ± 1.92, RMSE 23.37 ± 2.85
  - 60 min: MAE 28.22 ± 3.24, RMSE 38.15 ± 4.58
  - 90 min: MAE 36.55 ± 4.00, RMSE 48.69 ± 5.58
  - 120 min: MAE 42.90 ± 4.46, RMSE 56.43 ± 6.21
- **Bergman ODE (population):** (from `scripts/run_pure_bergman.py`)
  - 30 min: MAE 29.70 ± 9.67, RMSE 39.67 ± 12.67
  - 60 min: MAE 43.71 ± 12.84, RMSE 57.08 ± 16.10
  - 90 min: MAE 50.72 ± 13.66, RMSE 65.26 ± 16.79 *(stale draft had 52.79, 67.92)*
  - 120 min: MAE 54.18 ± 13.78, RMSE 69.20 ± 16.77 *(stale draft had 58.91, 74.96)*
  *(Note: Sec. III prose line 693 was also updated from 58.91 to 54.18 mg/dL).*
- **Bergman ODE (per-subject NLS):** (from `scripts/fit_bergman_per_subject.py`)
  - 30 min: MAE 25.80 ± 8.36, RMSE 34.72 ± 10.45
  - 60 min: MAE 38.09 ± 9.56, RMSE 50.41 ± 11.78
  - 90 min: MAE 44.70 ± 8.99, RMSE 58.47 ± 11.01 *(stale draft had 46.46, 60.59)*
  - 120 min: MAE 48.43 ± 8.35, RMSE 63.03 ± 10.29 *(stale draft had 52.48, 67.62)*
  *(Note: Sec. III prose line 693 was also updated from 52.48 to 48.43 mg/dL).*
- **AR(6):** (from `scripts/run_ar_arx.py`)
  - 30 min: MAE 14.39 ± 1.52, RMSE 20.11 ± 2.65
  - 60 min: MAE 25.52 ± 2.90, RMSE 33.88 ± 3.90
  - 90 min: MAE 33.18 ± 3.66, RMSE 42.92 ± 4.62 *(stale draft had 33.91, 44.02)*
  - 120 min: MAE 38.30 ± 3.90, RMSE 48.64 ± 4.92 *(stale draft had 39.81, 50.98)*
- **ARX(6):** (from `scripts/run_ar_arx.py`)
  - 30 min: MAE 13.97 ± 1.51, RMSE 19.61 ± 2.76
  - 60 min: MAE 24.48 ± 2.86, RMSE 32.67 ± 3.98
  - 90 min: MAE 32.24 ± 3.47, RMSE 41.82 ± 4.47
  - 120 min: MAE 37.68 ± 3.69, RMSE 48.00 ± 4.77
- **LSTM (non-mechanistic):** (from `results/tables/baselines_lstm_a0nomech.csv`)
  - 30 min: MAE 15.41 ± 2.16, RMSE 21.46 ± 2.83
  - 60 min: MAE 26.08 ± 3.87, RMSE 35.24 ± 4.56
  - 90 min: MAE 33.59 ± 4.77, RMSE 44.97 ± 5.76
  - 120 min: MAE 38.81 ± 5.30, RMSE 51.54 ± 6.63
- **Transformer ($A0_{\text{no-mech}}$):** (from `results/tables/baselines_lstm_a0nomech.csv`)
  - 30 min: MAE 14.96 ± 1.82, RMSE 21.08 ± 2.55
  - 60 min: MAE 26.06 ± 3.15, RMSE 35.56 ± 4.15
  - 90 min: MAE 34.46 ± 3.81, RMSE 46.08 ± 5.11
  - 120 min: MAE 40.06 ± 4.09, RMSE 53.03 ± 5.63
- **Transformer (all features, A0):** (from `results/ablations/per_subject_A0.csv`)
  - 30 min: MAE 13.51 ± 1.63, RMSE 19.38 ± 2.41
  - 60 min: MAE 22.82 ± 2.88, RMSE 31.61 ± 3.71
  - 90 min: MAE 29.74 ± 3.74, RMSE 40.32 ± 4.71
  - 120 min: MAE 34.93 ± 4.20, RMSE 46.70 ± 5.24
- **Hybrid physics-guided (A7):** (from `results/ablations/per_subject_A7.csv`)
  - 30 min: MAE 13.25 ± 1.69, RMSE 19.06 ± 2.51
  - 60 min: MAE 22.29 ± 2.93, RMSE 30.89 ± 3.89
  - 90 min: MAE 28.77 ± 3.75, RMSE 39.04 ± 4.83
  - 120 min: MAE 33.38 ± 4.21, RMSE 44.78 ± 5.46

---

### 2. ARX Fit Confirmation
**Confirmed.**
All four horizons of ARX(6) (30, 60, 90, 120 min) are produced by the identical multi-horizon Ridge ($\lambda = 1.0$) regression with physiological clipping $[40, 400]$ mg/dL in `scripts/run_ar_arx.py`.
The coefficient matrix $W = (X_{\text{train}}^T X_{\text{train}} + 1.0 \cdot I)^{-1} X_{\text{train}}^T Y_{\text{train}}$ is solved simultaneously for all 4 horizons at once. The 30/60-min ARX values (13.97 / 24.48 MAE, 19.61 / 32.67 RMSE) and 90/120-min ARX values (32.24 / 37.68 MAE, 41.82 / 48.00 RMSE) originate from the exact same fit and identical model weights.

---

### 3. Explanation of "166,463 − 20 = 166,443" and the 2020 First-Hour Exclusion
**Clarified with exact event counts:**
1. **Raw XML files count:**
   - 2018 cohort (12 XML files): exactly **85,225** glucose readings.
   - 2020 cohort (12 XML files): exactly **81,308** glucose readings.
   - Total raw XML events across all 24 files = **166,533**.
2. **First test hour exclusion (2020 cohort):**
   - The first 60 minutes of test data are excluded for all six 2020 test subjects ($6 \times 12 = 72$ potential 5-minute slots).
   - In the raw device XML files, **3 slots were already missing data**:
     - Subject 552: 11 readings in first hour (1 missing).
     - Subject 567: 11 readings in first hour (1 missing).
     - Subject 596: 11 readings in first hour (1 missing).
     - Subjects 540, 544, 584: 12 readings each.
     - Total real readings in the first test hour = $11 + 11 + 11 + 12 + 12 + 12 = \mathbf{69}$.
   - In addition, `540-ws-testing.xml` contains **1 duplicate timestamp** (`04-07-2027 00:01:44` repeated), so 2,896 XML events reduce to 2,895 unique timestamp readings.
   - Total dropped observations = $69 + 1 = \mathbf{70}$.
   - Total real CGM readings loaded into the 5-minute resampled grid = $166,533 - 70 = \mathbf{166,463}$ (2018: 85,225; 2020: 81,238).
3. **The 20-observation difference ($166,463 - 20 = 166,443$):**
   - The difference of **20 observations** is **not** missing data in the first test hour; it is caused by **sequence edge boundary trimming**.
   - Each evaluation window requires a 48-step historical input sequence (4 hours) and a 24-step forecast horizon (2 hours).
   - At the trailing boundary of each subject's time-series recording, any readings after the final valid anchor window cannot form a full 24-step horizon. Summing the unwindowed trailing readings across all subject streams equals exactly **20 grid observations**, yielding $166,463 - 20 = 166,443$ windowed observations.

---

### 4. Broken Link in `SAMYAK_REMAINING.md`
**Fixed.**
Replaced the local path `file:///home/sammyyakk/...` with the repository path `scripts/compute_autocorrelation.py` in `paper/digital-twin-v4/SAMYAK_REMAINING.md`.

---

### 5. Confirm $n_{\text{eff}}$ Sentence and Bayley & Hammersley Citation
**Confirmed.**
- **In Section II-B:** Text reads:
  > "Because consecutive 5-minute CGM errors are autocorrelated (empirical integrated autocorrelation time $\bar{\tau}_{\text{int}} \approx 53.6$ steps across subjects, yielding an effective sample size $n_{\text{eff}} = N / \bar{\tau}_{\text{int}} \approx 2{,}500\text{--}3{,}100$ per subject)~\cite{refBH}..."
- **In References:** `\bibitem{refBH}` matches:
  > G.~V.~Bayley and J.~M.~Hammersley, "The effective number of independent observations in an autocorrelated time series," *J. R. Stat. Soc. Suppl.*, vol.~8, no.~2, pp.~184--197, 1946.
- The document compiles cleanly to `main.pdf` without errors or missing citation warnings.
