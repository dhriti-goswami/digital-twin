# Notation and Abbreviations

Consolidated index of every symbol, abbreviation and feature name used across
[`METHODOLOGY.md`](METHODOLOGY.md), [`RESULTS.md`](RESULTS.md) and the `twin` package.
Intended to sit as an appendix at the end of the paper.

Ordering within each table follows the order the quantity is introduced, not the
alphabet, so that the tables also read as a derivation summary. Every entry names the
code identifier where one exists, so a reviewer can move from a paper symbol to the
line that computes it.

---

## 1. State variables

The six-state linear compartment vector plus the glucose state. All rate constants are
**per minute**; this was a source of a legacy defect, where 30-minute finite differences
were compared against per-minute constants.

| Symbol | Meaning | Unit | Code |
|---|---|---|---|
| $G$ | Plasma glucose concentration | mg/dL | `glucose_mg_dl` |
| $G_b$ | Basal (fasting) glucose, subject-specific | mg/dL | `resolve_basal_glucose` |
| $G_0$ | Last observed glucose at the forecast anchor $t=0$ | mg/dL | — |
| $X$ | Remote (interstitial) insulin **action** | min⁻¹ | `insulin_action_per_min` |
| $I$ | Plasma insulin concentration | µU/mL | `insulin_plasma_uU_mL` |
| $I_b$ | Basal plasma insulin, from the subject's basal rate | µU/mL | `basal_insulin_concentration` |
| $S_1$ | Subcutaneous insulin, compartment 1 | U | `insulin_sc_1_u` |
| $S_2$ | Subcutaneous insulin, compartment 2 | U | `insulin_sc_2_u` |
| $Q_{\text{sto}}$ | Carbohydrate in stomach | g | `carbs_stomach_g` |
| $Q_{\text{gut}}$ | Carbohydrate in gut | g | `carbs_gut_g` |
| $R_a(t)$ | Rate of glucose appearance from the meal | mg/min | `glucose_appearance_mgdl_per_min` |
| $\mathbf{z}$ | State vector $[S_1, S_2, I, Q_{\text{sto}}, Q_{\text{gut}}, X]^\top$ | — | `compartments.py` |
| $\mathbf{u}$ | Input vector $[u_{\text{ins}}, u_{\text{carb}}, I_b]^\top$ | — | `N_INPUTS = 3` |

Two identities are **derived rather than assumed**, which is the point of §2.3:

$$\text{IOB} = S_1 + S_2 \qquad\qquad \text{COB} = Q_{\text{sto}} + Q_{\text{gut}}$$

The third input channel carries the constant $I_b$ so that the affine term $-p_3 I_b$
fits inside a linear system — without it the basal equilibrium breaks (§4.1).

---

## 2. Estimated physiological parameters

The nine-element vector $\theta_p$ the encoder produces per window, each squashed into a
published interval by $\theta_i = \ell_i + (u_i - \ell_i)\sigma(z_i)$ so **no network
output can leave the admissible range**. Bounds and provenance in `twin/physio/params.py`;
confidence tags in [`CITATIONS.md`](CITATIONS.md).

| Symbol | Meaning | Range | Unit | Pop. mean | Code |
|---|---|---|---|---|---|
| $p_1$ | Insulin-independent glucose disposal | 0 – 0.030 | min⁻¹ | 0.013 | `p1` |
| $p_2$ | Decay rate of remote insulin action | 0.005 – 0.10 | min⁻¹ | 0.025 | `p2` |
| $p_3$ | Build rate of action from plasma insulin excess | $10^{-6}$ – $3{\times}10^{-5}$ | mL·µU⁻¹·min⁻² | $6.25{\times}10^{-6}$ | `p3` |
| $n$ | Plasma insulin clearance ($k_e$ in Hovorka) | 0.08 – 0.25 | min⁻¹ | 0.138 | `n` |
| $V_G$ | Glucose distribution volume | 1.4 – 2.4 | dL/kg | 1.88 | `V_G_per_kg` |
| $V_I$ | Insulin distribution volume | 0.08 – 0.18 | L/kg | 0.12 | `V_I_per_kg` |
| $t_{\max,I}$ | Time-to-peak of subcutaneous absorption | 30 – 90 | min | 55 | `tmax_I` |
| $k_{\text{gri}}$ | Gastric emptying rate | 0.008 – 0.10 | min⁻¹ | 0.035 | `k_gri` |
| $k_{\text{abs}}$ | Intestinal absorption rate | 0.005 – 0.10 | min⁻¹ | 0.0167 | `k_abs` |
| $f$ | Carbohydrate bioavailability | 0.70 – 1.00 | dimensionless | 0.90 | `f` |

### The reported derived quantity

$$\boxed{\;S_I = \frac{p_3}{p_2}\;}\qquad[\mathrm{mL}\,\mu\mathrm{U}^{-1}\,\mathrm{min}^{-1}]$$

**$S_I$ — insulin sensitivity index.** The **steady-state gain of the remote insulin
compartment**: setting $dX/dt = -p_2X + p_3(I-I_b) = 0$ gives
$X_{ss} = (p_3/p_2)(I - I_b) = S_I(I-I_b)$, so $S_I$ is *insulin action delivered per
unit of plasma insulin above basal*. Because $X$ enters the glucose equation additively
on the disposal constant $p_1$, a high $S_I$ means the same insulin excess drives glucose
down harder (insulin sensitive) and a low $S_I$ means it barely moves disposal (insulin
resistant). Units resolve as (mL·µU⁻¹·min⁻²)/(min⁻¹) = mL·µU⁻¹·min⁻¹, which multiplied by
an insulin excess in µU/mL yields min⁻¹ — the same units as $p_1$, as it must be to sit
beside it. Population anchor $S_I^{\text{IDDM}} = 2.5{\times}10^{-4}$
(`S_I_IDDM_MEAN`, Ward et al. 1991); $p_3$'s bound is *derived* as $S_I p_2$ rather than
sourced independently. Full derivation, the three pre-registered validation checks and the
exact scope of the claim: **METHODOLOGY §2.3**, **RESULTS §6**.

---

## 3. Model and architecture

| Symbol | Meaning | Value here |
|---|---|---|
| $t$ | Time from the forecast anchor | 0 – 120 min |
| $h$ | A forecast horizon | 30, 60, 90, 120 min |
| $\Delta$ | Grid step | 5 min |
| $T$ | Non-dimensionalisation timescale | 60 min |
| $B$ | Batch size | — |
| $H$ | Number of horizons | 4 |
| $K$ | Number of cubic B-spline basis functions | 12 |
| $B_k(t)$ | The $k$-th B-spline basis function | clamped uniform knots |
| $c_k$ | Spline coefficient emitted by the head | — |
| $G_\theta(t)$ | Learned spline trajectory | $G_0 + \sum_k c_k(B_k(t)-B_k(0))$ |
| $A, B$ | Continuous state and input matrices | $6{\times}6$, $6{\times}3$ |
| $A_d, B_d$ | Exact zero-order-hold discretisations | from one $9{\times}9$ `matrix_exp` |
| $M$ | Augmented matrix exponential | $A_d = M_{1:6,1:6}$, $B_d = M_{1:6,7:9}$ |
| $\theta_p$ | Estimated physiological parameter vector | 9 elements (§2) |
| $z_i$ | Unconstrained pre-sigmoid parameter output | — |
| $g = \sigma(\gamma)$ | Learned **trust gate** on the mechanistic prior | init $\sigma(-2.2)\approx0.10$ |
| $\hat{G}_{0.5}$ | Median (point) forecast | mg/dL |
| $\hat{G}_q$ | The $q$-th quantile forecast | mg/dL |
| $\delta_q(h)$ | Quantile band offset, in **glucose units** | $\mp\,\mathrm{softplus}(\cdot)$ |
| $q$ | Quantile level | 0.1, 0.5, 0.9 |
| $r(t)$ | Non-dimensionalised physics residual | evaluated at 121 points |
| $\sigma(\cdot)$ | Logistic sigmoid | — |
| $\varphi(x)$ | $(1-e^{-x})/x$, with a Taylor branch near 0 | — |
| $n_{\text{eff}}$ | Effective independent sample count | $\approx N/48$ |

Offsets live in glucose units, **not** coefficient space: because the basis is a
partition of unity, $\sum_k\delta(B_k(t)-B_k(0)) = \delta(1-1) = 0$, so a uniform
coefficient shift produces a band of exactly zero width (§2.8).

---

## 4. Loss terms

$$
\mathcal{L} = \mathcal{L}_{\text{Huber}}
+ \lambda_q\mathcal{L}_{\text{pinball}}
+ w_{\text{phys}}\mathcal{L}_{\text{res}}
+ \lambda_{\text{pr}}\mathcal{L}_{\text{prior}}
+ \lambda_{\text{tc}}\mathcal{L}_{\text{temporal}}
$$

| Symbol | Meaning |
|---|---|
| $\mathcal{L}_{\text{Huber}}$ | Horizon-weighted Huber point loss — robust to CGM artefacts where MSE is not |
| $\mathcal{L}_{\text{pinball}}$ | $\max\big(q(y-\hat y_q),\,(q-1)(y-\hat y_q)\big)$; proper scoring rule for a quantile |
| $\mathcal{L}_{\text{res}}$ | Collocation residual $\lVert r(t)\rVert^2$ over the 1-min grid |
| $\mathcal{L}_{\text{prior}}$ | Weak penalty pulling $\theta_p$ toward population means (anti-collapse) |
| $\mathcal{L}_{\text{temporal}}$ | Penalty on $\theta_p$ disagreement between adjacent windows of one subject |
| $w_{\text{phys}}$ | Physics weight: adaptive $\times$ curriculum ramp |
| $\lambda_q,\lambda_{\text{pr}},\lambda_{\text{tc}}$ | Fixed weights on band, prior, temporal-consistency terms |
| $s_i = \log\sigma_i^2$ | Kendall log-variance parameterisation, $\mathcal{L} = \sum_i\frac{1}{2\sigma_i^2}\mathcal{L}_i + \frac12\log\sigma_i^2$ |

**Deliberately absent:** any asymmetric hypo/hyper penalty. Such a term inflates
error-grid zone A by construction, making a safety table computed afterwards no longer
independent evidence. A test asserts the option cannot be configured.

---

## 5. Metrics

| Symbol / name | Definition | Unit |
|---|---|---|
| MAE | $\frac1n\sum\lvert \hat y - y\rvert$ | mg/dL |
| RMSE | $\sqrt{\frac1n\sum(\hat y - y)^2}$ | mg/dL |
| $R^2$ | $1 - \mathrm{SS}_{\text{res}}/\mathrm{SS}_{\text{tot}}$ | — |
| MARD | $\frac{100}{n}\sum\lvert\hat y - y\rvert/y$, **reference in the denominator** | % |
| Skill | $1 - \mathrm{RMSE}_{\text{model}}/\mathrm{RMSE}_{\text{persistence}}$ | % |
| $f(G)$ | Kovatchev risk transform $1.509\big((\ln G)^{1.084} - 5.381\big)$ | — |
| $r$ | Risk value $10f(G)^2$ | — |
| LBGI / HBGI | Low / High Blood Glucose Index; mean of $r$ over the **total** $n$ | — |
| TIR / TAR / TBR | Time in / above / below range | % |
| CV | Coefficient of variation, $\mathrm{SD}/\text{mean}$ | % |
| `cv_ratio` | $\mathrm{CV}_{\text{pred}}/\mathrm{CV}_{\text{actual}}$ — excursion compression | — |
| Prediction lag | Cross-correlation delay; a lag at the full horizon means the model is replaying its input | min |
| Coverage | Fraction of observations below the nominal quantile | — |
| Hypo sensitivity | Recall of $G < 70$ mg/dL events | — |
| ICC(1) | One-way intraclass correlation; test–retest stability of $S_I$ | — |
| $\rho$ | Spearman rank correlation | — |
| IG$_i$ | Integrated gradient of feature $i$: $(x_i-x_i')\int_0^1 \partial_i f(x'+\alpha(x-x'))\,d\alpha$ | mg/dL |

Kovatchev symmetric endpoints are **20 and 600** mg/dL, not 40 and 400:
$f(20) = -3.1634$ and $f(600) = +3.1629$, mapping $[20,600]$ onto
$[-\sqrt{10},+\sqrt{10}]$.

Consensus bands, exhaustive and disjoint: very low $<54$, low 54–69, in-range 70–180,
high 181–250, very high $>250$ mg/dL.

Reporting rule throughout: **per subject first, then mean ± SD across subjects.** Pooled
figures are secondary and labelled, because subjects contribute wildly different window
counts.

---

## 6. Abbreviations

### Clinical and physiological

| Short | Expansion | Note |
|---|---|---|
| T1D | Type 1 diabetes | |
| IDDM | Insulin-dependent diabetes mellitus | older term, retained in `S_I_IDDM_MEAN` |
| CGM | Continuous glucose monitor(ing) | 5-min sampling in OhioT1DM |
| SC | Subcutaneous | route of insulin delivery |
| IOB | Insulin on board | here **derived** as $S_1 + S_2$ |
| COB | Carbohydrate on board | here **derived** as $Q_{\text{sto}} + Q_{\text{gut}}$ |
| TDD | Total daily dose (of insulin) | U/day; per kg for the $S_I$ check |
| ICR | Insulin-to-carbohydrate ratio | from Ohio `bwz_carb_input` |
| TIR / TAR / TBR | Time in / above / below range | |
| LBGI / HBGI | Low / High Blood Glucose Index | Kovatchev |
| EGA | Error grid analysis | Clarke and Parkes |
| PRED-EGA | Prediction error grid analysis | Sivananthan 2011; **not implemented** — boundaries unverifiable |
| ISO 15197 | Standard for blood glucose *meters* | a per-reading tolerance, **not** a forecasting threshold |
| iCGM | FDA integrated CGM criteria | measures the present, forecasts nothing |
| ADA | American Diabetes Association | |

### Datasets and protocols

| Short | Expansion | Note |
|---|---|---|
| OhioT1DM | Ohio University T1D dataset | 12 subjects, 2018 + 2020 cohorts |
| BGLP | Blood Glucose Level Prediction challenge | 2018 and 2020 protocols **differ** |
| LOSO | Leave-one-subject-out | 12 folds, genuinely subject-disjoint |
| Official | The dataset's own temporal holdout | personalised; same subjects, later ~10 days |
| ShanghaiT1DM | External-validation dataset | 15-min CGM; **not obtained** |
| DiaTrend | External-validation dataset | Synapse DUA; **not obtained** |
| UVA/Padova | 13-state simulator used for pretraining | never in reported results |
| DUA | Data use agreement | |

The 2020 cohort **excludes the first hour (12 points) of each test file**; 2018 does not
(`TEST_WARMUP_EXCLUSION`). Cohort sensor availability also differs — 2018 has heart
rate and steps, 2020 has acceleration with those channels zeroed — which is why every
sensor feature ships with an explicit availability mask rather than silent zero-fill.

### Methods and architecture

| Short | Expansion |
|---|---|
| PINN | Physics-informed neural network |
| ODE | Ordinary differential equation |
| ZOH | Zero-order hold (the exact discretisation of §2.4) |
| RK4 | Fourth-order Runge–Kutta (considered, rejected) |
| MLP | Multilayer perceptron |
| LSTM / BiLSTM | (Bidirectional) long short-term memory |
| CNN | Convolutional neural network |
| NN | Neural network |
| IG | Integrated gradients (Sundararajan et al. 2017) |
| SHAP | SHapley Additive exPlanations — the *legacy* explainer, **replaced** by IG |
| SD | Standard deviation |
| CI | Confidence interval |
| ICC | Intraclass correlation coefficient |
| A0 – A7 | Ablation arms; see RESULTS §5 |

### Infrastructure

| Short | Expansion |
|---|---|
| VRAM | Video RAM (4 GB budget — the constraint behind §2.4) |
| SHA | Git commit hash recorded in every run manifest |
| CSV / XML | Artifact and Ohio source formats |
| NaN | Not-a-number (§4.2: three simultaneous causes) |

---

## 7. Feature names

The 35-element feature contract, enforced by assertion in `twin/data/features.py`. One
list, one order — the legacy pipeline wrote by magic index (`feat_matrix[:, 31+i]`),
which corrupts silently if the list is reordered. Attribution shares are from RESULTS §7.

### Glucose — 42.7% of attribution

| Name | Meaning | Unit |
|---|---|---|
| `glucose_mg_dl` | CGM reading | mg/dL |
| `roc_5min` | Rate of change over 5 min — **the dominant feature** | mg/dL/min |
| `roc_15min` | Rate of change over 15 min | mg/dL/min |
| `roc_30min` | Rate of change over 30 min | mg/dL/min |
| `glucose_mean_1h`, `glucose_std_1h` | 1-hour rolling mean, SD | mg/dL |
| `glucose_min_1h`, `glucose_max_1h` | 1-hour rolling extremes | mg/dL |
| `glucose_mean_2h` | 2-hour rolling mean | mg/dL |

Rates are **true rates on real timestamps**, not index differences. A feature-validity
gate extends each window's exclusion zone by the feature's look-back, so a rate
differenced across a gap cannot enter a window; within emitted windows the range is
$-5.8$ to $+11.0$ mg/dL/min.

### Mechanistic — 27.1% of attribution

All six states plus the two derived sums, from the **same** parameterisation that
supplies the physics residual. This shared parameterisation is the methodological point:
features and constraint cannot disagree.

| Name | Symbol | Unit |
|---|---|---|
| `iob_u` | $S_1 + S_2$ | U |
| `insulin_sc_1_u`, `insulin_sc_2_u` | $S_1$, $S_2$ | U |
| `insulin_plasma_uU_mL` | $I$ | µU/mL |
| `insulin_action_per_min` | $X$ | min⁻¹ |
| `cob_g` | $Q_{\text{sto}} + Q_{\text{gut}}$ | g |
| `carbs_stomach_g`, `carbs_gut_g` | $Q_{\text{sto}}$, $Q_{\text{gut}}$ | g |
| `glucose_appearance_mgdl_per_min` | $R_a/V_G$ | mg/dL/min |

### Therapy — 13.5% · Time — 8.1% · Sensor — 7.0% · Context — 1.7%

| Group | Names |
|---|---|
| therapy | `basal_u_per_min` (temp-basal aware), `bolus_u_per_min`, `minutes_since_bolus`, `minutes_since_meal` |
| time | `hour_sin`, `hour_cos`, `is_night` |
| sensor | `basis_heart_rate`, `basis_gsr`, `basis_skin_temperature`, each with a `_available` mask |
| context | `exercise_intensity`, `sleeping`, `working`, `glucose_interpolated` |

Every sensor channel is paired with an availability mask. Silent zero-fill would be
mapped by the fitted scaler to a large negative z-score — a fabricated extreme rather
than a missing value. `glucose_interpolated` exposes input interpolation as a feature;
**targets are never interpolated**.

---

## 8. Conventions

- **Rate constants are per minute.** Mixing per-minute constants with 30-minute
  differences was legacy defect 1.
- **Subscript $b$** denotes a basal (fasting) value: $G_b$, $I_b$.
- **Hat** denotes a prediction: $\hat{G}$, $\hat{y}$.
- **Overbar** denotes a step-average: $\bar{k} = \frac12(k_j + k_{j+1})$, giving the
  midpoint rule and $O(\Delta^2)$ accuracy.
- **Confidence tags** on every citation: `VERIFIED-PRIMARY`, `SECOND-HAND`, `UNVERIFIED`.
  Where a source could not be read, it says so rather than being cited on faith.
- **Purge gap** $= \text{seq\_len} + \max(\text{horizon steps}) = 48$ windows, the
  minimum guaranteeing no training window shares a grid slot with a validation window.
- **Unit of statistical analysis is the subject**, never the window: windows overlap
  23/24 and are strongly autocorrelated.
