# Methodology

Complete mathematical specification of the physics-guided digital twin, with the
derivation of each choice, the primary source it rests on, and the file that
implements it.

Every citation tag below is one of `VERIFIED-PRIMARY`, `SECOND-HAND`, or
`UNVERIFIED`, matching [`CITATIONS.md`](CITATIONS.md). Nothing in this document is
asserted from memory: where a source could not be read, it says so.

**Notation.** Glucose `G` [mg/dL], plasma insulin `I` [µU/mL], remote insulin
action `X` [1/min], time `t` [min]. Subscript `b` denotes a basal (fasting)
value. `B` is batch size, `H` the number of forecast horizons.

---

## 1. Problem statement

Given 2 hours of history — CGM, insulin delivery, carbohydrate intake, and
context — predict glucose at 30, 60, 90 and 120 minutes ahead, while
simultaneously estimating the subject's insulin sensitivity.

Formally, with an input window ending at anchor time `t₀`:

$$
\hat{G}(t_0 + h) = f_\theta\big(\mathbf{x}_{t_0-23:t_0}\big), \quad h \in \{30, 60, 90, 120\}\ \text{min}
$$

subject to the constraint that `Ĝ(t)` satisfy the Bergman minimal model on
`t ∈ [t₀, t₀+120]`.

The second requirement is what makes this more than a regression: the forecast must
be a physiologically realisable trajectory, not merely four numbers.

---

## 2. Mechanistic model

### 2.1 Bergman minimal model

`VERIFIED-PRIMARY` — Bergman RN, Phillips LS, Cobelli C. *Physiologic evaluation of
factors controlling glucose tolerance in man.* J Clin Invest 1981;68(6):1456–1467.

$$
\begin{aligned}
\frac{dG}{dt} &= -(p_1 + X)\,G + p_1 G_b + \frac{R_a(t)}{V_G} \\[4pt]
\frac{dX}{dt} &= -p_2 X + p_3 (I - I_b) \\[4pt]
\frac{dI}{dt} &= -n I + \frac{1000}{V_I}\,\frac{S_2}{t_{\max,I}}
\end{aligned}
$$

Implemented in `twin/physio/bergman.py` and `twin/physio/compartments.py`.

**Note on the published form.** The 1981 paper's figure caption writes the second
equation with `p₃·I(t)` rather than `p₃·(I − I_b)`. The modern form with the basal
subtraction is used here, and it is not cosmetic — see §2.5.

**Insulin sensitivity**, the quantity this study reports per subject:

$$
S_I = \frac{p_3}{p_2} \quad [\mathrm{mL}\,\mu\mathrm{U}^{-1}\,\mathrm{min}^{-1}]
$$

### 2.2 Subcutaneous insulin absorption

`VERIFIED-PRIMARY` — Hovorka R, Canonico V, Chassin LJ, et al. *Nonlinear model
predictive control of glucose concentration in subjects with type 1 diabetes.*
Physiol Meas 2004;25:905–920. Two-compartment absorption, `t_max,I = 55 min`,
`k_e = 0.138 min⁻¹`, `V_I = 0.12 L/kg`.

$$
\frac{dS_1}{dt} = u_{\text{ins}}(t) - \frac{S_1}{t_{\max,I}}, \qquad
\frac{dS_2}{dt} = \frac{S_1}{t_{\max,I}} - \frac{S_2}{t_{\max,I}}
$$

`u_ins(t)` [U/min] is the pump record: basal rate, temporary-basal overrides, and
boluses distributed over their delivery interval.

**Why this matters for the features.** For an impulse dose `D` at `t = 0`:

$$
S_1 + S_2 = D\,e^{-t/t_{\max,I}}\left(1 + \frac{t}{t_{\max,I}}\right)
$$

which equals `D` at `t = 0` and decreases monotonically to zero
(`d/dt[e^{-x}(1+x)] = -x e^{-x} < 0`). **This is insulin-on-board**, derived rather
than assumed. The legacy pipeline instead convolved the bolus train with a
*time-reversed activity curve*, giving a feature that was ≈0 at the moment of the
bolus and peaked 145 minutes later.

`UNVERIFIED` — Wilinska et al. 2005 numeric parameters. Paywalled, no reproducing
source found. No value from it is used.

### 2.3 Gut carbohydrate absorption

`VERIFIED-PRIMARY` — Lehmann ED, Deutsch T. *A physiological model of
glucose–insulin interaction in type 1 diabetes mellitus.* J Biomed Eng
1992;14(3):235–242. `k_gabs = 1 h⁻¹ = 0.0167 min⁻¹`.
`SECOND-HAND` — Dalla Man C, Rizza RA, Cobelli C. IEEE TBME 2007;54(10):1740–1749,
for `k_gri` and bioavailability `f` (taken from a peer-reviewed survey, not the
paper itself).

$$
\frac{dQ_{\text{sto}}}{dt} = -k_{\text{gri}} Q_{\text{sto}} + u_{\text{carb}}(t), \qquad
\frac{dQ_{\text{gut}}}{dt} = k_{\text{gri}} Q_{\text{sto}} - k_{\text{abs}} Q_{\text{gut}}
$$

$$
R_a(t) = f\,k_{\text{abs}}\,Q_{\text{gut}}(t) \quad [\mathrm{mg/min}]
$$

**Mass conservation.** Integrating, `∫₀^∞ R_a dt = f·D` exactly. **Carbohydrate-on-board
is `Q_sto + Q_gut`.** The legacy `_compute_cob` used a bare `exp(-k/36)` kernel,
which jumped instantaneously at ingestion and never reached zero inside the
window — it conserved nothing.

A structural departure is disclosed: Lehmann–Deutsch specifies *trapezoidal*
gastric emptying (`T_asc = T_des = 30 min`, `V_max,ge = 120 mmol/h`). A first-order
`k_gri` is used instead, as in Dalla Man, because it keeps the system linear and
therefore exactly integrable (§2.4). This is an approximation, and it is the reason
`k_gri` is tagged `SECOND-HAND`.

### 2.4 Exact linear propagation

**This is the design decision that makes the physics affordable.**

Only the glucose equation is nonlinear, through the `X·G` product — and `G` is
supplied by the network's spline head rather than integrated. Every other state
forms a **linear system with known inputs**:

$$
\mathbf{z} = [S_1, S_2, I, Q_{\text{sto}}, Q_{\text{gut}}, X]^\top, \qquad
\dot{\mathbf{z}} = A(\theta_p)\,\mathbf{z} + B(\theta_p)\,\mathbf{u}(t)
$$

with `u = [u_ins, u_carb, I_b]ᵀ`. For piecewise-constant inputs the exact
zero-order-hold solution is

$$
\mathbf{z}_{k+1} = A_d \mathbf{z}_k + B_d \mathbf{u}_k, \qquad
A_d = e^{A\Delta}, \quad B_d = \left(\int_0^\Delta e^{As}\,ds\right) B
$$

Both blocks come from a **single** matrix exponential of the augmented system:

$$
M = \exp\!\left(\begin{bmatrix} A & B \\ 0 & 0\end{bmatrix}\Delta\right)
\;\Longrightarrow\;
A_d = M_{1:6,\,1:6}, \quad B_d = M_{1:6,\,7:9}
$$

Since `Δ` is fixed, this is **one 9×9 `matrix_exp` per patient for an entire
trajectory**. Pump and meal records genuinely are piecewise-constant on the
5-minute grid, so the discretisation is *exact*, not approximate. No ODE solver
runs inside the training loop, no adjoint is needed, and it fits comfortably in
4 GB of VRAM.

Verified in `tests/test_physio.py::test_discretisation_is_exact_for_scalar_decay`
and `::test_insulin_mass_conserved`.

### 2.5 The basal equilibrium — a correctness requirement

In type 1 diabetes all plasma insulin is exogenous, so state `I` is the *total*
concentration. Remote insulin action must nonetheless be driven by insulin **above
basal**, i.e. `p₃(I − I_b)` with

$$
I_b = \frac{1000}{V_I}\cdot\frac{u_{\text{basal}}}{n}
$$

the steady-state concentration implied by the subject's own basal rate.

**Why.** At basal, `I = I_b`, hence `X* = 0`, hence

$$
\left.\frac{dG}{dt}\right|_{G=G_b} = -(p_1 + 0)G_b + p_1 G_b = 0
$$

so `G = G_b` is an equilibrium: with no stimulus, glucose does not move. Driving
`X` by total insulin instead makes `X* > 0` at basal, and glucose is then pulled
toward `p₁G_b/(p₁+X*)`. With population parameters that is ≈68 mg/dL, and a
two-hour forecast from 300 mg/dL collapsed by over 200 mg/dL **with no meal or
bolus present at all**. This was a real defect during development, found because
the untrained model's deviation from persistence was implausibly large.

`I_b` enters as a constant third input channel, so the affine `−p₃I_b` term is
absorbed into the exact ZOH machinery of §2.4 at no cost.

Guarded by `test_basal_is_a_glucose_equilibrium` and
`test_inconsistent_basal_insulin_breaks_the_equilibrium` — the latter asserts the
failure mode reappears if the fix is reverted.

### 2.6 Closed-form glucose solution

Once `X(t)` is known, the glucose equation is **linear** and non-autonomous. With
`k(t) = p₁ + X(t)` and `c(t) = p₁G_b + R_a(t)/V_G`:

$$
G(t+\Delta) = G(t)\,e^{-\bar{k}\Delta} + \frac{\bar{c}}{\bar{k}}\left(1 - e^{-\bar{k}\Delta}\right)
$$

Coefficients are **averaged across the step** (`k̄ = ½(kₖ + kₖ₊₁)`), giving the
midpoint rule and `O(Δ²)` accuracy. Sampling at the step start would leave an
`O(Δ)` bias, visible as a sustained residual after a meal; correcting it reduced
the median residual 58-fold.

The quotient is evaluated as `c·Δ·(1−e^{−x})/x` with a Taylor branch near `x = 0`,
because `p₁ → 0` is a normal configuration, not an edge case.

**Order verified empirically**, not asserted: `test_glucose_integrator_is_second_order`
requires the residual to fall ≈4× per halving of `Δ`. Measured: 2.17e−3 → 2.17e−5 →
2.17e−7 across 10× refinements.

### 2.7 Parameter ranges and reparameterisation

Estimated parameters are mapped into published intervals by a scaled sigmoid, so
**no network output can leave the admissible range**:

$$
\theta_i = \ell_i + (u_i - \ell_i)\,\sigma(z_i)
$$

A zero pre-activation gives the interval midpoint; the output bias is initialised
so the head starts at the population mean.

| Parameter | Range | Unit | Confidence | Source |
|---|---|---|---|---|
| `p₁` | 0 – 0.030 | 1/min | `VERIFIED-PRIMARY` | Ward et al. 1991 |
| `p₂` | 0.005 – 0.10 | 1/min | `SECOND-HAND` | Bergman 1981 (range spans reported fits) |
| `p₃` | 1e−6 – 3e−5 | mL/(µU·min²) | `VERIFIED-PRIMARY` | derived as `S_I·p₂` from Ward |
| `n` | 0.08 – 0.25 | 1/min | `VERIFIED-PRIMARY` | Hovorka 2004 (`k_e = 0.138`) |
| `V_G` | 1.4 – 2.4 | dL/kg | `SECOND-HAND` | Dalla Man 2007 |
| `V_I` | 0.08 – 0.18 | L/kg | `VERIFIED-PRIMARY` | Hovorka 2004 (0.12) |
| `t_max,I` | 30 – 90 | min | `VERIFIED-PRIMARY` | Hovorka 2004 (55) |
| `k_gri` | 0.008 – 0.10 | 1/min | `SECOND-HAND` | Dalla Man 2007 |
| `k_abs` | 0.005 – 0.10 | 1/min | `VERIFIED-PRIMARY` | Lehmann & Deutsch 1992 |
| `f` | 0.70 – 1.00 | — | `SECOND-HAND` | Dalla Man 2007 |

**On `p₁ = 0`.** Fixing `p₁ = 0` for T1D is common in control-oriented work, on the
argument that without endogenous insulin there is no glucose-mediated
self-regulation. **That is a modelling simplification, not an empirical finding.**
`VERIFIED-PRIMARY` — Ward GM, Walters JM, Aitken PM, Best JD, Alford FP.
*Metabolism* 1991;40(1):4–9 measured glucose effectiveness directly in IDDM
subjects: `S_G = 1.0–1.6 × 10⁻² min⁻¹` — reduced relative to controls but clearly
non-zero, and `S_I = 2.5 ± 0.6 × 10⁻⁴`. The population default is therefore the
measured value; `p₁ = 0` remains reachable and is an ablation.

`assert_bounds_sourced()` refuses to produce reportable output while any bound is
still provisional. `second_hand_bounds()` lists those resting on a secondary
source, for disclosure.

---

## 3. Model architecture

### 3.1 Encoder

`features (B, 24, 35) → Linear(35→128) → sinusoidal PE → 4 × TransformerEncoderLayer
(d_model 128, 8 heads, FF 512, GELU, pre-norm) → attention pooling → context (B, 128)`

Total 816,023 parameters.

**Attention pooling** rather than the mean: mean pooling weights a reading two
hours old identically to the most recent one, in a problem where recency dominates.

No causal mask. Every input timestep is history relative to the forecast, so
attending across the full window leaks nothing.

### 3.2 Continuous-time head — why B-splines

The physics residual needs `dG/dt`. Four options were considered:

| Option | Cost | Verdict |
|---|---|---|
| `(context, t) → G(t)` MLP, autodiff in `t` | one backward pass per collocation point | slow; gradient noise destabilises training |
| **cubic B-spline coefficients** | one matrix multiply | **chosen** |
| Neural ODE / latent ODE | adjoint over 120 min | most elegant, too slow and memory-hungry at 4 GB |
| RK4 on a 1-min grid | 120 unrolled steps | slowest, offers nothing the spline does not |

The head emits `K = 12` coefficients over `t ∈ [0, 120]` min with a clamped uniform
knot vector:

$$
G_\theta(t) = G_0 + \sum_{k=1}^{K} c_k\big(B_k(t) - B_k(0)\big)
$$

Three consequences, each of which fixes a specific legacy defect:

1. **`G(0) = G₀` identically**, for any coefficients — anchoring by construction, not
   by penalty.
2. **`dG/dt` is exact and analytic**, from the spline derivative basis. The legacy
   residual finite-differenced across 30-minute steps against per-minute rate
   constants.
3. **The reported horizons are evaluations of the constrained function.**
   `G(30), G(60), G(90), G(120)` are read off the same trajectory the residual acts
   on, so there is no train/report mismatch.

The basis is verified to be a partition of unity, its derivative to sum to zero,
and value/derivative to be mutually consistent under the fundamental theorem of
calculus — the last again by *convergence order* rather than a chosen tolerance.

Design matrices are stored in `float64`: they are exact constants, and `float32`
storage caps the achievable derivative precision at ≈1e−7 relative.

### 3.3 Patient-specific parameters

An amortised head maps the context to `θ_p` (10 parameters, §2.7). `G_b` and `I_b`
are **not** estimated: `G_b` is the subject's overnight fasting median (an
observable), and `I_b` is fixed by §2.5. Estimating a latent for something already
measured would be inventing precision.

Anti-collapse measures, all necessary for `S_I` to mean anything:

- hard reparameterisation into physiological ranges (§2.7);
- a prior penalty toward population means, **normalised by interval width** so a
  parameter spanning `1e-6..3e-5` is not ignored beside one spanning `30..90`;
- a temporal-consistency penalty: windows from the same subject in one batch must
  give near-identical estimates. Without it the head absorbs per-window prediction
  error into the parameters, yielding an excellent fit and a meaningless `S_I`;
- a warmup during which parameters are frozen at population values, because an
  unconstrained estimate fitted to initial noise collapses.

### 3.4 Hybrid physics-guided prediction

The title says physics-**guided**, and the implementation reflects that: the
mechanistic forecast is a *prior the network corrects*.

$$
\hat{G}(t) = G_0 + g\cdot\big(G_{\text{Bergman}}(t) - G_0\big) + \big(G_\theta(t) - G_0\big)
$$

Both terms are anchored at `G₀`, so each contributes only its deviation and the
anchor is not double-counted.

`g = σ(γ)` is a **learned trust gate** on the prior, initialised at `σ(−2.2) ≈ 0.10`.
Rationale: with *population* parameters the unfitted mechanistic forecast is
substantially worse than persistence (MAE 29.4 vs 16.9 mg/dL at 30 min), so
starting at `g = 1` would force the learned part to spend capacity undoing the
prior. Starting low means the model begins near persistence — the strongest naive
comparator on this data — and learns how much physics to trust.

At initialisation, with `g ≈ 0.10` and an untrained encoder, the model already
attains MAE 16.3 / 28.3 / 39.0 / 42.8 versus persistence 16.9 / 30.0 / 41.9 / 45.6
— the physics contributes before any training at all.

**The converged `g` is reported.** It quantifies how much of the forecast the
physics actually carries, which is precisely the question the ablation asks.

---

## 4. Objective

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + w_{\text{phys}}\mathcal{L}_{\text{res}}
+ \lambda_{\text{prior}}\mathcal{L}_{\text{prior}}
+ \lambda_{\text{temp}}\mathcal{L}_{\text{temp}}
$$

### 4.1 Data term

Huber loss, `δ = 10 mg/dL`, uniform across horizons. Huber because CGM contains
sensor artefacts and compression lows that a squared loss lets dominate the
gradient. Horizon weights are uniform deliberately: down-weighting long horizons
would improve the headline 30-minute number at the cost of the long-horizon table,
which is a presentational choice disguised as an optimisation one.

### 4.2 Physics residual

$$
r(t) = \frac{T}{G_b}\left[\frac{d\hat{G}}{dt} + (p_1 + X(t))\hat{G}(t) - p_1 G_b - \frac{R_a(t)}{V_G}\right]
$$

evaluated at 121 collocation points on a 1-minute grid, with
`L_res = mean(r²)`.

**Non-dimensionalisation is mandatory.** With `Ĝ = (G−G_b)/G_b` and `τ = t/T`
(`T = 60 min`), residuals are `O(1)` and the weight is comparable across subjects.
The legacy residual mixed 30-minute differences with per-minute constants and was
never scaled, which is why its `0.1` multiplier was meaningless.

`X(t)` and `R_a(t)` come from advancing the compartments with the **estimated**
parameters over a **12-hour burn-in of the subject's actual insulin and meal
history**, initialised at the analytic basal steady state. 144 steps at 5 minutes is
several multiples of the slowest admissible time constant (`k_abs = 0.005 min⁻¹`
gives a 200-minute gut constant), so a bolus or meal before the burn-in cannot
leave a trace.

### 4.3 Adaptive weighting

`kendall` — Kendall A, Gal Y, Cipolla R. *Multi-task learning using uncertainty to
weigh losses for scene geometry and semantics.* CVPR 2018.

$$
\mathcal{L} = \sum_i \frac{1}{2\sigma_i^2}\mathcal{L}_i + \frac{1}{2}\log\sigma_i^2
$$

parameterised as `sᵢ = log σᵢ²` for stability. The log term is what prevents the
optimiser from driving every weight to zero.

A **data-first cosine ramp** scales `w_phys` from 0 to 1 over epochs 5→25: the
residual constrains a trajectory that is initially meaningless, so enforcing it
from step one fights the data term while the encoder is random.

A fixed `λ = 0.1` is available for **ablation A1 — the naive PINN the original
paper claimed** — so adaptive weighting is measured, not assumed. GradNorm (Chen et
al. 2018) and ReLoBRaLo (Bischof & Kraus 2021) are the further comparators.

### 4.4 What is deliberately absent

**No asymmetric hypo/hyper penalty.** The legacy `clinical_penalty_loss` weighted a
missed hyperglycaemia at 6.0 and a missed hypoglycaemia at 2.0 — backwards, since
hypoglycaemia is the acute risk. More fundamentally, *any* asymmetric training loss
inflates error-grid zone A **by construction**, so a clinical-safety table computed
afterwards is no longer independent evidence. Safety is measured and reported
(§6.3), not optimised into the objective.
`test_no_clinical_asymmetry_knob_exists` asserts the option cannot be configured.

---

## 5. Data pipeline

### 5.1 Source

OhioT1DM (Marling & Bunescu): 12 subjects, 2018 cohort {559, 563, 570, 575, 588,
591} and 2020 cohort {540, 544, 552, 567, 584, 596}. 166,463 CGM observations.

**The two cohorts are not the same protocol.** `VERIFIED-PRIMARY`: the 2020 BGLP
challenge excludes the first hour (12 samples) of each test file; the 2018 edition
does not. Applied per cohort in `twin/data/ohio.py`; ignoring it makes results
incomparable to published work.

**Body weight is not identifiable.** Every file reports `weight="99"` — a
de-identification placeholder. A nominal 70 kg is used, and because distribution
volumes are estimated per kilogram, the unknown true weight is absorbed into the
estimated absolute volumes. Treating 99 kg as real would be fabricated precision.

### 5.2 Parsing fixes

| Defect | Consequence | Fix |
|---|---|---|
| `<temp_basal>` never read | basal wrong exactly around exercise and hypoglycaemia | parsed and applied as an override; empty `value` = suspension |
| boluses used only `ts_begin` | extended/dual-wave doses collapsed to an instant | distributed over `[ts_begin, ts_end]`, mass-conserving |
| tag read as `stress` | channel silently empty | correct tag is `stressors` |
| cohort sensors zero-filled | 2020 lacks Basis channels; a fitted scaler maps 0 to a large negative z-score, usable as a cohort indicator | explicit `*_available` masks; value zeroed *and* flagged |

### 5.3 Gap-aware sequencing

**The defect this replaces.** `evaluate_ohio.py:254-256` dropped rows with missing
CGM, then called `reset_index(drop=True)` and sliced windows from the compacted
array. For any window spanning a gap, the value labelled "+30 minutes" was not 30
minutes ahead. Every Ohio number in the legacy repository was computed against
mislabelled targets.

A window at anchor `t₀` is emitted **only if**:

1. its input span is `seq_len = 24` consecutive slots on the exact time grid —
   guaranteed structurally, since no row is ever removed;
2. every input slot has glucose, after interpolating **whole runs** of at most
   `max_interp_gap = 2` missing slots;
3. the interpolated fraction does not exceed `1 − 0.9`;
4. **every target is a real observation at exactly the nominal horizon.**

`WindowSet.verify` re-derives (4) from the frame at runtime, so the central
correctness claim is a check rather than a comment.

**On (2):** `Series.interpolate(limit=n)` fills the *first* `n` values of every run,
so a three-hour outage would receive two fabricated values on its leading edge. Run
length is therefore tested explicitly.

**On (4):** the legacy code forward-filled gaps up to 15 minutes and used the filled
values *as targets*, so a "30-minute prediction" could be scored against a copy of a
reading the model had already seen — which persistence predicts perfectly, inflating
short-horizon accuracy by construction.

**Feature-validity gate.** Rates propagate `NaN` rather than being filled. Carrying
the last observation forward and differencing produced rates above 50 mg/dL/min at
gap edges; within emitted windows the range is now −5.8 to +11.0 mg/dL/min.
`valid_row_mask` extends the exclusion zone around a gap by each feature's
look-back (`roc_30min` reads 6 slots back), so a rate differenced against a value
from the far side of a gap cannot enter a window. A glucose-only check would admit
it.

**Accounting.** 141,100 of 187,780 candidate windows survive (75.1%). Rejections are
dominated by incomplete input spans, target-not-observed second. Reported per
subject, because the reduction relative to legacy counts must be visible as
rigour rather than data loss.

### 5.4 Feature contract

35 features in a fixed, named, asserted order (`twin/data/features.py`). Groups:
glucose level and rates (9), **mechanistic states (9)**, therapy context (4), time
of day (3), activity and context (4), masked sensors (6).

**Mechanistic features come from the same model as the physics loss.** `IOB = S₁+S₂`,
`COB = Q_sto+Q_gut`, plus `I`, `X`, `R_a`. They are mass-conserving by construction
and share one parameterisation with the residual. This is the methodological point:
features and constraint are not two separate approximations of the same physiology.

They are computed with **population** parameters, deliberately — features must not
drift under the scaler while the network's per-patient estimates move during
training. The estimated parameters are used in the physics loss, where varying them
is the point.

Replaces: magic index writes (`feat_matrix[:, 31+i]`, which would silently reassign
columns on any reordering); a duplicated feature (`time_frac_day ≡ day_frac`, so 35
declared features were 34); and differences labelled as rates.

### 5.5 Split protocols

Both are evaluated on the **identical test windows** (27,092), so their difference
isolates one variable: whether the test subject's own earlier data was available
during training.

**Design A — `official`.** Train on all 12 subjects' training files, test on their
test files (the next contiguous ~10 days). This is what every published Ohio number
uses. It measures **personalised** forecasting and must not be described as
cross-subject generalisation.

**Design B — `loso`.** 12 subject-disjoint folds; train on 11 subjects, test on the
held-out subject's test file.

**Inner validation** is the time-ordered **tail** of each training subject's record,
with a **purge gap** of `seq_len + max_horizon_steps = 48` windows at the boundary —
the minimum that guarantees no training window shares a grid slot with a validation
window. `verify_no_leakage` checks slot overlap directly and rejects a smaller purge.

Replaces `finetune_ohio.py:232-239`, which applied `torch.randperm` to the pooled
window set and took 15% as validation. Consecutive windows share 23 of 24 input
timesteps, so validation was a near-duplicate of training — and it drove both early
stopping and the reported headline metric.

No purge at the train/test boundary: windows are built only within a single file's
frame, so a training window's targets always lie inside the training period. There
is no overlap to purge, and adding one would deviate from the published protocol.

**Leakage-free `G_b`.** Under `official`, each subject's own training period supplies
it. Under `loso`, the held-out subject's data is off-limits *including for a summary
statistic*, so the value is the mean across training subjects.

### 5.6 Scaling

Fitted on the **set of grid slots** appearing as an input in at least one training
window, each counted **once**. Stacking windows would count a slot up to 24 times,
weighting statistics by how many windows happen to cover it — an artefact of gap
structure. Fitted on 105,592 unique slots.

The scaler is **bound to the fold that fitted it** and serialised into the
checkpoint. `evaluate.py:629-646` fed two different scalers into one reported table.

---

## 6. Evaluation

### 6.1 Reporting rule

**Metrics per subject, then mean ± SD across subjects.** Pooled figures are computed
and reported as clearly-labelled secondary. Pooling let a subject with 2,137 windows
outweigh one with 356, so the legacy tables described the best-instrumented
subjects rather than the cohort. Per-subject-then-aggregate is also the standard
OhioT1DM format, without which numbers are not comparable to published work.

### 6.2 Baselines

Persistence, linear ROC extrapolation, ARIMA, plain LSTM, plain Transformer — all on
exactly the windows the model sees.

**Persistence is the reference point.** The legacy pipeline computed none, and both
its headline numbers (30.44 and 78.5 mg/dL RMSE at 30 min) lose to it.

Measured here, per-subject then averaged:

| Horizon | 2018 (n=6) RMSE | 2018 MAE | All 12 RMSE | All 12 MAE |
|---|---|---|---|---|
| 30 min | **22.60 ± 2.50** | **16.36 ± 1.46** | 23.37 ± 2.85 | 16.87 ± 1.92 |
| 60 min | **36.34 ± 3.14** | 27.05 ± 1.88 | 38.15 ± 4.58 | 28.22 ± 3.24 |
| 90 min | 46.40 ± 3.69 | 34.92 ± 2.35 | 48.69 ± 5.58 | 36.55 ± 4.00 |
| 120 min | 54.04 ± 4.48 | 41.11 ± 2.88 | 56.43 ± 6.21 | 42.90 ± 4.46 |

**This is the project's end-to-end validation.** Two independent published sources
report 2018-cohort persistence RMSE of 22.5 ± 2.2 at 30 min and 36.6 ± 3.0 at 60 min.
Reproducing both to within 0.3 mg/dL, with matching SDs, simultaneously validates
parsing, grid snapping, sequencing, horizon integrity, the metrics implementation,
and per-subject aggregation. A defect in any one would break the agreement.
Persistence **MAE** is unpublished anywhere; it is computed here.

**Finding: ROC extrapolation is worse than persistence at every horizon**, for 0 of
12 subjects (skill −0.13 at 30 min to −0.57 at 120 min). CGM trends mean-revert, so
persistence is a genuinely strong comparator, not a trivial one.

### 6.3 Error grids

**Clarke** — `VERIFIED-PRIMARY` Clarke WL et al. *Diabetes Care* 1987;10(5):622–628.

The 1987 paper publishes **no inequalities** — only a figure and prose. (The
existence of Stöckl et al. 2000, a letter clarifying the construction of the upper
A-line alone, is itself evidence of the ambiguity.) Boundaries here are recovered
from the two canonical reference implementations. Evaluating both over the integer
lattice `r, p ∈ [1, 550]` (302,500 points) gives 12,029 disagreements which reduce
to **exactly one substantive difference** plus 854 measure-zero open/closed points:

- the MATLAB lineage caps upper-C at `r ≤ 290`. That is an artefact of the original
  figure's 0–400 axes — `p = r + 110` leaves the vertex `(70, 180)` and exits a
  400-limit plot at `(290, 400)`. **We drop the cap**, following CRAN `ega`, which is
  the defensible extrapolation for CGM data exceeding 400 mg/dL.

Evaluation order is the canonical **A → E → C → D → B**, retained even though a point
can score A on the ±20% rule while sitting on a dangerous-zone edge. Reordering to
prioritise safety would change the numbers and make them incomparable to every
published result — which is the entire reason for computing them.

The `58.33` constant sometimes quoted for lower-D is **not a primary boundary**: it is
`70/1.2`, where the upper A-line crosses `p = 70`. Because A is tested first, writing
lower-D as `r ≤ 70` is provably equivalent.

Legacy defects, each now a named regression test: any pair with both values in
70–180 scored zone A (so ref 75 / pred 180, a 140% error, counted as accurate); zone
D relabelled as B; zone E **unreachable** because its branch duplicated D's; and an
invented `r ≥ 290` branch that scored ref 400 / pred 210 as A.

**Parkes** — `VERIFIED-PRIMARY` Parkes JL et al. *Diabetes Care* 2000;23(8):1143–1148
for the grid; **Pfützner A et al.** *J Diabetes Sci Technol* 2013;7(5):1275–1281 for
the coordinates. The 2013 paper does **not correct** the 2000 one — it *first
publishes* coordinates that had never appeared in print. Cite accordingly. Type 1
vertices verified vertex-by-vertex against CRAN `ega`, including its
slope-extrapolated terminal points. Type 1 has **no lower E zone**; extreme
under-prediction saturates at D. That is a property of the published grid, and it is
stated rather than silently absorbed.

**Figure/table consistency by construction.** `zone_field` shades a figure by
evaluating *the same classifier* used for the table on a dense mesh. The legacy plots
drew decorative segments unrelated to the counting logic and did not colour points
by zone, so a figure could contradict its own table indefinitely.

### 6.4 Clinical metrics

Consensus bands (very low <54, low 54–69, in-range 70–180, high 181–250, very high
>250 mg/dL), exhaustive and disjoint. Reported **actual versus predicted**, with the
signed gap: a model that under-predicts hypoglycaemia shows a strongly negative
`below_range_delta` even when RMSE looks acceptable. The legacy code computed these
on predictions only, so compression was invisible.

**Kovatchev LBGI/HBGI.** Risk transform
`f(G) = 1.509(ln(G)^{1.084} − 5.381)`, `r = 10f²`, averaged over the **total** `n`
(not the branch count), which is what makes the indices comparable between subjects
who spend different amounts of time low.

The symmetric endpoints are **20 and 600 mg/dL**, not 40 and 400: `f(20) = −3.1634`,
`f(600) = +3.1629`, i.e. the transform maps `[20, 600]` onto `[−√10, +√10]`. This is
unit-tested as the strongest available check on the three constants short of the
primary source.

**Excursion compression.** `cv_ratio = CV_pred / CV_actual` is reported per subject.
A model that flattens toward the mean earns a flattering RMSE and a visibly wrong CV.

**Prediction lag.** Cross-correlation delay between predicted and actual traces. A
model whose optimal alignment sits at the full horizon is replaying its input, not
anticipating the future — two models with identical RMSE can differ entirely here.

### 6.5 Statistics

The unit of analysis is the **subject**. Windows overlap by 23 of 24 timesteps and
are strongly autocorrelated, so a per-window test on ~27,000 correlated windows
would call a 0.2 mg/dL difference significant.

- **Paired Wilcoxon signed-rank** across subjects, exact null distribution (`n = 12`
  makes the normal approximation inappropriate).
- **Holm–Bonferroni** over the whole family of comparisons — several methods × four
  horizons. Holm rather than plain Bonferroni (uniformly more powerful, same FWER
  control) and rather than Benjamini–Hochberg (family-wise control is the right
  standard when each comparison is reported as an individual claim).
- **Percentile bootstrap CIs resampling subjects**, not windows. Resampling windows
  would give a hopelessly narrow interval describing only sampling within this
  cohort.
- With `n = 12` a paired test has limited power. `describe_comparison` reports
  direction and magnitude first, p-value last, and flags underpowered comparisons
  explicitly, so a sentence cannot read as stronger evidence than it is.

### 6.6 Skill score

$$
\text{skill} = 1 - \frac{\text{metric}_{\text{method}}}{\text{metric}_{\text{persistence}}}
$$

computed **per subject** then averaged.

Reported against persistence because it is the one comparator immune to protocol
mismatch: no training data, no hyperparameters, no assumptions about how another
paper built its windows.

**This replaces a claim that cannot be made.** The draft asserted that
"MAE < 15 mg/dL is generally regarded as clinically acceptable". That figure derives
from ISO 15197:2013 — a **per-reading** tolerance under 95% coverage, only **below
100 mg/dL**, for an **in-vitro capillary meter measuring the present**. No standard,
error-grid paper, or consensus statement defines an MAE threshold for *prediction*.
(FDA iCGM special controls require only 70% of in-range readings within ±15%, and
forecast nothing.) Citing it for forecasting is a category error.

Two further points make the original framing untenable:

1. **15 of 17 published OhioT1DM entries already clear MAE < 15 at 30 min**,
   including a *non-personalised* LSTM at 14.37. It is the field median, not a
   frontier. Best credible published: 12.83 MAE / 18.22 RMSE.
2. **Persistence alone attains MAE 16.36 mg/dL at 30 min**, so the target represents
   roughly a **9% improvement over predicting no change at all**.

Citable replacements: the skill score above; **PRED-EGA** (Sivananthan et al. 2011),
an error grid built for *predictors* rather than sensors; and horizon-stratified,
field-relative targets.

**No protocol-matched published MAE exists at 90 or 120 minutes**, so no
published-comparison table is built there — own-baseline comparison only, stated
explicitly.

### 6.7 Insulin sensitivity validation

`S_I` is only a contribution if it is shown to measure something. There is no clamp
study, so three orthogonal checks are used, all computable from Ohio's own records:

1. **Correlation with the subject's own therapy.** Ohio `bolus` events carry
   `bwz_carb_input`, so an empirical carb ratio and total daily dose per kg are
   derivable. Estimated `S_I` should correlate negatively with insulin requirement.
   Reported as Spearman ρ across the 12 subjects with a bootstrap CI.
2. **Test–retest stability.** `S_I` estimated independently on disjoint time windows
   for the same subject; reported as within-subject ICC. An unstable estimate is not
   a physiological parameter.
3. **Physiological ordering.** Ranking across subjects should be stable and
   consistent with insulin requirement.

Reported honestly if any fails.

---

## 7. Ablations

| # | Configuration | Isolates |
|---|---|---|
| A0 | Transformer, no physics | baseline |
| A1 | + collocation residual, fixed `λ = 0.1` | the naive PINN the original draft claimed |
| A2 | + learned log-variance weighting | adaptive weighting |
| A3 | + mechanistic prior and residual correction | hybrid versus pure penalty |
| A4 | A3, population-fixed `θ_p` | value of per-patient parameters |
| A5 | A3 + simulation pretraining | transfer value |
| A6 | A3, mechanistic IOB/COB → hand-rolled kernels | value of §5.4 |

**If the physics term does not improve accuracy, that is reported.** It is a
publishable negative result about physics-informed forecasting, and far better than
a fabricated PINN. The hybrid formulation hedges by making physics useful as a prior
even if the residual penalty proves neutral, and the ablation makes either outcome
reportable.

---

## 8. Reproducibility

- `set_seed` covers Python, NumPy, torch (CPU and all CUDA devices), cuDNN
  determinism, `use_deterministic_algorithms`, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`
  (set at import, before any CUDA context exists), and DataLoader `worker_init_fn` —
  which PyTorch leaves unseeded for NumPy and `random`.
- Configs are typed and **reject unknown keys**: a silently-ignored typo makes a
  recorded setting differ from the one that ran.
- Every run writes a manifest with the git commit and dirty-file list, resolved
  config, SHA-256 of every input file, package versions, and hardware.
- Checkpoints carry the config, the fitted scaler, the feature-name contract, and
  the fold identity. `load_checkpoint` **refuses** a checkpoint whose feature contract
  differs from the current one — the legacy deployment silently zero-filled six
  missing features and fed them through a scaler that mapped the zeros to large
  negative z-scores.
- 223 tests, each named for the defect it guards.

**Hardware.** NVIDIA RTX 2050 (4 GB), 12 CPU cores, 15 GB RAM, PyTorch 2.10 + CUDA
12.8. The closed-form physics (§2.4) is what makes this sufficient: no solver in the
loop and no adjoint.

---

## 9. Model capacity and the effective sample size

A measured result, recorded because it constrains what this dataset can support.

**The window count overstates the sample size by two orders of magnitude.** The
pipeline emits ~97,000 training windows, but consecutive windows share 23 of their
24 input timesteps. The number of *independent* windows is closer to
`135,000 slots / 48 steps ≈ 2,800` — one per non-overlapping window span. Against
the default 816,024-parameter model that is a ratio near **300:1**.

The consequence was unambiguous. Two full-size runs (`artifacts/diagnostics/`)
showed training loss falling from 108 to 7.6 while validation degraded monotonically:

| epoch | train loss | val MAE@30 | val mean | prior gate |
|---|---|---|---|---|
| 1 | 108.5 | **15.25** | 26.6 | 0.106 |
| 5 | 55.5 | 15.42 | 27.3 | 0.120 |
| 10 | 27.2 | 15.62 | 29.0 | 0.138 |
| 15 | 14.0 | 15.72 | 29.6 | 0.167 |
| 20 | 7.6 | **16.85** | 30.7 | 0.214 |

By epoch 20 the 30-minute validation MAE had degraded to the level of persistence
(16.87). Raising dropout from 0.1 to 0.2 changed the magnitude slightly and the
*shape* not at all, which is what distinguishes a capacity problem from insufficient
regularisation.

`configs/official-small.yaml` therefore reduces the model to **79,192 parameters**
(`d_model` 64, 2 layers, weight decay 0.03), bringing the ratio to roughly 28:1.

Two points for the paper:

- **This is a property of the dataset, not of the architecture.** Any model with
  ~10⁵-10⁶ parameters trained on 12 subjects will meet the same ceiling. It is the
  main reason a 12-subject benchmark cannot separate methods finely, and it is worth
  stating rather than absorbing into a hyperparameter footnote.
- **The learned prior gate rose monotonically throughout** (0.106 → 0.214) even as
  the data fit overfitted. The optimiser increased its reliance on the mechanistic
  forecast rather than discarding it, which is a genuine — if preliminary — signal
  that the physics term carries information the data term was not supplying. The
  A0-versus-A3 ablation is what will confirm or refute it; this observation is
  suggestive only, and is reported as such.

---

## 10. Known gaps

Disclosed rather than papered over:

- **Wilinska et al. 2005** numeric parameters — paywalled, no reproducing source. No
  value from it is used.
- **Stöckl et al. 2000** letter on the Clarke upper A-line — paywalled and unreadable.
  A live gap if a reviewer challenges the A-band construction.
- **Dalla Man 2006/2007** values are `SECOND-HAND`, via a peer-reviewed survey.
  Surfaced at runtime by `second_hand_bounds()`.
- **Kovatchev 2004** page range and the **PRED-EGA** full text unread.
- **Gastric emptying** is first-order here, where Lehmann–Deutsch specifies
  trapezoidal (§2.3).
- **Subject 567's test period contains no carbohydrate records at all**, so COB is
  identically zero there and a physics-informed model is structurally degraded on
  that subject.
- **Subject 552 test coverage is 59.7%**, yielding far fewer windows than the others.
- **Body weight is not identifiable** from OhioT1DM (§5.1).
