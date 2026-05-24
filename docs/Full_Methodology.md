# Mathematical Methodology of the Physics-Guided Personalized Digital Twin for T1D

This document presents the complete mathematical methodology of the **Explainable Physics-Guided Personalized Digital Twin for T1D** framework. All derivations and formulations are detailed in standard mathematical notation and mapped directly onto the ingestion, training, optimization, explainability, and personalization segments of the production codebase.

---

## 3.1 Data Representation

The temporal-clinical history of a patient with Type 1 Diabetes is formally modeled as a multivariate sequence window. The input tensor is represented as:

$$\mathbf{X} \in \mathbb{R}^{N \times T \times F}$$

where:
* $N$ represents the batch size of active window sequences.
* $T = 24$ is the historical lookback sequence length, corresponding to 2 hours of continuous observation sampled at $\Delta t = 5\text{ minute}$ steps.
* $F = 43$ represents the unified feature dimension.

At each timestep $t$, the feature vector $\mathbf{x}_t \in \mathbb{R}^{F}$ is decomposed into five distinct physical and demographic subvectors:

$$\mathbf{x}_t = \left[ \mathbf{x}_t^{\text{CGM}} \mathbin{\Vert} \mathbf{x}_t^{\text{Insulin}} \mathbin{\Vert} \mathbf{x}_t^{\text{Meal}} \mathbin{\Vert} \mathbf{x}_t^{\text{Temporal}} \mathbin{\Vert} \mathbf{x}_t^{\text{Static}} \right]$$

where $\mathbin{\Vert}$ represents vector concatenation:

1. **Continuous Glucose Monitor (CGM) Features**:
   $$\mathbf{x}_t^{\text{CGM}} \in \mathbb{R}^6$$
   $$\mathbf{x}_t^{\text{CGM}} = \left[ G(t), \Delta_5 G(t), \Delta_{15} G(t), \Delta_{30} G(t), \mu_{1\text{h}}(t), \sigma_{1\text{h}}(t) \right]^T$$
   tracking continuous systemic glucose, three multiscale rates of change, and rolling statistical moments.

2. **Insulin Pharmacokinetics**:
   $$\mathbf{x}_t^{\text{Insulin}} \in \mathbb{R}^3$$
   $$\mathbf{x}_t^{\text{Insulin}} = \left[ I_{\text{basal}}(t), I_{\text{bolus}}(t), \text{IOB}(t) \right]^T$$
   tracking background basal delivery, rapid bolus administration, and active Insulin-on-Board.

3. **Meal Ingestion Dynamics**:
   $$\mathbf{x}_t^{\text{Meal}} \in \mathbb{R}^2$$
   $$\mathbf{x}_t^{\text{Meal}} = \left[ C_{\text{ingested}}(t), \text{COB}(t) \right]^T$$
   tracking carbohydrate ingestion and gastrointestinal Carbs-on-Board.

4. **Temporal Cyclical Features**:
   $$\mathbf{x}_t^{\text{Temporal}} \in \mathbb{R}^2$$
   $$\mathbf{x}_t^{\text{Temporal}} = \left[ \sin\left(\frac{2\pi \cdot h(t)}{24}\right), \cos\left(\frac{2\pi \cdot h(t)}{24}\right) \right]^T$$
   mapping the circadian hour $h(t) \in [0, 24)$ to orthogonal coordinates.

5. **Static Clinical Covariates**:
   $$\mathbf{x}_t^{\text{Static}} \in \mathbb{R}^{30}$$
   $$\mathbf{x}_t^{\text{Static}} = \left[ \text{Age}, \text{BMI}, \text{HbA}_{1\text{c}}, \text{Medications}, \mathbf{p}_{\text{harmonized}} \right]^T$$
   representing demographics aligned across the multi-dataset fusion pipeline (UCI, PIMA, 130-Hospitals).

---

## 3.2 Feature Engineering Mathematics

Dynamic indicators are computed at each time step $t$ using the following physiological formulations:

### 1. Multiscale Rates of Change (ROC)
Glucose velocity is estimated using backward finite differences over horizons of $k \in \{1, 3, 6\}$ steps (5, 15, and 30 minutes):

$$\Delta_k G(t) = \frac{G(t) - G(t - k\Delta t)}{k \cdot \Delta t} \quad \text{mg/dL/min}$$

### 2. Rolling Statistical Windows
Historical glucose volatility and distribution are captured over a 1-hour window ($W = 12$ steps):

$$\mu_{1\text{h}}(t) = \frac{1}{W} \sum_{i=0}^{W-1} G(t - i\Delta t)$$

$$\sigma_{1\text{h}}(t) = \sqrt{\frac{1}{W-1} \sum_{i=0}^{W-1} \left(G(t - i\Delta t) - \mu_{1\text{h}}(t)\right)^2}$$

$$\text{CV}(t) = \left( \frac{\sigma_{1\text{h}}(t)}{\mu_{1\text{h}}(t)} \right) \times 100\%$$

### 3. Cyclical Time Encoding
To prevent boundary discontinuities at midnight, diurnal time is projected into orthogonal trigonometric coordinates:

$$\phi(t) = \frac{2\pi \cdot \text{hour}(t)}{24}$$

$$\mathbf{x}_t^{\text{Temporal}} = \left[ \sin(\phi(t)), \cos(\phi(t)) \right]^T$$

### 4. Insulin-on-Board (IOB) Decay Model
Systemic insulin bioavailability of rapid-acting analogues is modeled using a bilinear-exponential decay profile driven by previous bolus events:

$$\text{IOB}(t) = \sum_{t_i \le t} d_i \cdot k_{\text{decay}}(t - t_i)$$

where $d_i$ represents the bolus dose in units (U) administered at time $t_i$. The decay function $k_{\text{decay}}(\tau)$ is defined programmatically as:

$$k_{\text{decay}}(\tau) = \max\left(0, 1 - \frac{\tau}{\tau_{\text{dur}}}\right)^{\alpha_{\text{decay}}}$$

[ASSUMPTION: Rapid insulin active duration $\tau_{\text{dur}} = 240\text{ min}$ and exponential decay coefficient $\alpha_{\text{decay}} = 1.5$]

### 5. Carbs-on-Board (COB) Absorption Model
The active gastrointestinal carbohydrate absorption curve is derived using a continuous exponential absorption profile:

$$\text{COB}(t) = \sum_{t_j \le t} c_j \cdot \exp\left(-\kappa \cdot (t - t_j)\right)$$

where $c_j$ is the meal size in grams at time $t_j$, and $\kappa$ is the gastrointestinal clearance rate constant:

$$\kappa = 0.0116 \text{ min}^{-1} \quad \text{(nominal clearing rate constant for 90-minute absorption half-life)}$$

---

## 3.3 Transformer Architecture — Full Mathematical Derivation

The deep forecasting block maps historical multivariate sequences $\mathbf{X}_n \in \mathbb{R}^{T \times F}$ to future prediction horizons using an attention-based sequence encoder.

### 1. Input Embedding and Projection Layer
The raw sequence is linearly projected into the model dimension $d_{\text{model}} = 128$:

$$\mathbf{Z}^{(0)} = \mathbf{X}_n \mathbf{W}_{\text{proj}} + \mathbf{b}_{\text{proj}}$$

where the projection weights and biases reside in:

$$\mathbf{W}_{\text{proj}} \in \mathbb{R}^{F \times d_{\text{model}}}$$
$$\mathbf{b}_{\text{proj}} \in \mathbb{R}^{d_{\text{model}}}$$

### 2. Sinusoidal Positional Encoding
To preserve sequence order information without relying on recurrence, sinusoidal positional encodings are added element-wise:

$$\mathbf{Z}^{(0)} \leftarrow \mathbf{Z}^{(0)} + \mathbf{PE}$$

$$\mathbf{PE}(t, 2i) = \sin\left(\frac{t}{10000^{\frac{2i}{d_{\text{model}}}}}\right), \quad \mathbf{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

where $t \in \{1, \dots, T\}$ and $i \in \{0, \dots, \frac{d_{\text{model}}}{2} - 1\}$.

### 3. Multi-Head Self-Attention Derivation
For each attention layer $l \in \{1, \dots, L\}$, with $h = 8$ parallel heads, the input representation $\mathbf{Z}^{(l-1)}$ is projected into Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) spaces:

$$\mathbf{Q}_j = \mathbf{Z}^{(l-1)}\mathbf{W}_j^Q, \quad \mathbf{K}_j = \mathbf{Z}^{(l-1)}\mathbf{W}_j^K, \quad \mathbf{V}_j = \mathbf{Z}^{(l-1)}\mathbf{W}_j^V$$

where:
$$\mathbf{W}_j^Q, \mathbf{W}_j^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$$
$$\mathbf{W}_j^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$
* $d_k = d_v = \frac{d_{\text{model}}}{h} = 16$.

The self-attention matrix for head $j$ is derived using a scaled dot-product:

$$\mathbf{A}_j = \text{Attention}(\mathbf{Q}_j, \mathbf{K}_j, \mathbf{V}_j) = \text{softmax}\left(\frac{\mathbf{Q}_j\mathbf{K}_j^T}{\sqrt{d_k}}\right)\mathbf{V}_j$$

where $\sqrt{d_k} = 4$ represents the scaling factor that prevents gradient vanishing in the softmax function.

### 4. Concatenation and Linear Mixing
The output from all $h = 8$ heads is concatenated and linearly transformed:

$$\mathbf{Z}_{\text{attn}}^{(l)} = \text{Concat}\left(\mathbf{A}_1, \mathbf{A}_2, \dots, \mathbf{A}_h\right)\mathbf{W}^O$$

where $\mathbf{W}^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

### 5. Feed-Forward Sublayer and Regularization
A residual connection and Layer Normalization ($\text{LN}$) are applied:

$$\mathbf{Z}_{\text{norm}}^{(l)} = \text{LN}\left(\mathbf{Z}^{(l-1)} + \mathbf{Z}_{\text{attn}}^{(l)}\right)$$

The representation is then passed through a Position-Wise Feed-Forward Network ($\text{FFN}$) using the Gaussian Error Linear Unit ($\text{GELU}$) activation:

$$\text{FFN}\left(\mathbf{Z}_{\text{norm}}^{(l)}\right) = \text{GELU}\left(\mathbf{Z}_{\text{norm}}^{(l)}\mathbf{W}_1^{(l)} + \mathbf{b}_1^{(l)}\right)\mathbf{W}_2^{(l)} + \mathbf{b}_2^{(l)}$$

$$\mathbf{Z}^{(l)} = \text{LN}\left(\mathbf{Z}_{\text{norm}}^{(l)} + \text{FFN}\left(\mathbf{Z}_{\text{norm}}^{(l)}\right)\right)$$

where $\mathbf{W}_1^{(l)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $\mathbf{W}_2^{(l)} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and $d_{\text{ff}} = 512$.

### 6. Sequence Pooling and Horizon Projection
The output sequence tensor of the final encoder block is aggregated over the time dimension using global average pooling:

$$\mathbf{Z}^{(L)} \in \mathbb{R}^{T \times d_{\text{model}}}$$
$$\bar{\mathbf{z}} \in \mathbb{R}^{d_{\text{model}}}$$
$$\bar{\mathbf{z}} = \frac{1}{T} \sum_{t=1}^T \mathbf{z}_t^{(L)}$$

This sequence embedding vector $\bar{\mathbf{z}}$ is projected to the final forecast dimension (4 horizons):

$$\hat{\mathbf{y}} = \bar{\mathbf{z}}\mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \quad \in \mathbb{R}^4$$

where $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times 4}$ and $\mathbf{b}_{\text{out}} \in \mathbb{R}^4$.

---

## 3.4 LSTM Architecture — Full Mathematical Derivation

When selected, the recurrent baseline structures sequence dynamics through a gated memory process. For a given hidden state dimension $d_{\text{hidden}} = 128$:

### 1. Gated Equations at Timestep $t$
$$\mathbf{i}_t = \sigma\left(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i\right) \quad \text{(Input Gate)}$$

$$\mathbf{f}_t = \sigma\left(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f\right) \quad \text{(Forget Gate)}$$

$$\tilde{\mathbf{g}}_t = \tanh\left(\mathbf{W}_g \mathbf{x}_t + \mathbf{U}_g \mathbf{h}_{t-1} + \mathbf{b}_g\right) \quad \text{(Cell Input Gate)}$$

$$\mathbf{o}_t = \sigma\left(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o\right) \quad \text{(Output Gate)}$$

where $\sigma(z) = (1 + e^{-z})^{-1}$ represents the logistic sigmoid function, and $\odot$ represents the Hadamard element-wise product.

### 2. Cell and Hidden State Update
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{g}}_t$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh\left(\mathbf{c}_t\right)$$

### 3. Bidirectional Combination
The bidirectional layer processes the sequence in both forward ($\vec{\mathbf{h}}_t$) and backward ($\overleftarrow{\mathbf{h}}_t$) directions, concatenating the final hidden representations at each time step:

$$\tilde{\mathbf{h}}_t = \left[ \vec{\mathbf{h}}_t \mathbin{\Vert} \overleftarrow{\mathbf{h}}_t \right] \quad \in \mathbb{R}^{2 \cdot d_{\text{hidden}}}$$

---

## 3.5 Physics-Informed Loss — Full Bergman Minimal Model

To guarantee clinical safety and physiological plausibility, predictions are regularized using ordinary differential equations (ODEs) derived from the **Bergman Minimal Model** of glucose-insulin dynamics.

### 1. Dynamic ODE Formulations
$$\frac{dG(t)}{dt} = -p_1 \left(G(t) - G_b\right) - X(t) \cdot G(t) + Ra(t)$$

$$\frac{dX(t)}{dt} = -p_2 X(t) + p_3 \left(I(t) - I_b\right)$$

$$\frac{dI(t)}{dt} = -p_4 I(t) + R_i(t)$$

where:
* $G(t)$ is plasma glucose concentration.
* $X(t)$ is remote compartment insulin action.
* $I(t)$ is plasma insulin concentration.
* $G_b, I_b$ are basal (target equilibrium) glucose and insulin levels.
* $Ra(t)$ is the glucose appearance rate from meal carbohydrates.
* $R_i(t)$ is external rapid-acting insulin infusion.

### 2. Model Parameters and Code Mapping
[ASSUMPTION: Intracellular clearing $p_1 = 0.028 \text{ min}^{-1}$]

[ASSUMPTION: Active insulin decay $p_2 = 0.025 \text{ min}^{-1}$]

[ASSUMPTION: Insulin sensitivity $p_3 = 5.0 \times 10^{-5} \text{ min}^{-1}/(\mu\text{U/mL})$]

[ASSUMPTION: Basal glucose $G_b = 110.0 \text{ mg/dL, and step size }\Delta t = 30.0\text{ min}$]

### 3. Numerical ODE Residual Approximations
The continuous glucose derivative is approximated across the forecast horizons $k \in \{1, 2, 3, 4\}$ (30, 60, 90, 120 minutes) using finite differences:

$$\frac{dG(t_k)}{dt} \approx \frac{G(t_k) - G(t_{k-1})}{\Delta t}$$

where $G(t_0) = G(t)$ is the current glucose value. 

Remote insulin action $X(t_k)$ and carbohydrate rate of appearance $Ra(t_k)$ are estimated dynamically from the input parameters:

$$X(t_k) = p_3 \cdot \text{IOB}_0 \cdot \max\left(0, 1 - \frac{t_k}{240}\right)^{1.5}$$

$$Ra(t_k) = 0.15 \cdot \left[ \frac{\text{COB}(t_{k-1}) - \text{COB}(t_k)}{\Delta t} \right]$$

where $\text{COB}(t_k) = \text{COB}_0 \cdot \max\left(0, 1 - \frac{t_k}{180}\right)^{1.2}$.

The Bergman glucose ODE residual $\mathcal{R}_{\text{Bergman}}$ is defined as:

$$\mathcal{R}_{\text{Bergman}}(t_k) = \frac{G(t_k) - G(t_{k-1})}{\Delta t} - \left[ -p_1\left(G(t_k) - G_b\right) - X(t_k)G(t_k) + Ra(t_k) \right]$$

### 4. Intracellular and Physiological Violation Penalties
The structural loss enforces physiological boundaries:

$$\mathcal{L}_{\text{bounds}} = \frac{1}{4} \sum_{k=1}^4 \left[ \text{ReLU}\left(40.0 - \hat{G}(t_k)\right) + \text{ReLU}\left(\hat{G}(t_k) - 400.0\right) \right]$$

The total physics loss is formulated as:

$$\mathcal{L}_{\text{phys}} = \left( \frac{1}{4} \sum_{k=1}^4 \mathcal{R}_{\text{Bergman}}(t_k)^2 \right) + 0.01 \cdot \mathcal{L}_{\text{bounds}}$$

### 5. Total Loss Formulation
The network optimizes a multi-objective loss function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \cdot \mathcal{L}_{\text{phys}}$$

[ASSUMPTION: Physics loss regularization weight $\lambda_{\text{phys}} = 0.1$]

$$\mathcal{L}_{\text{data}} = \frac{1}{4} \sum_{k=1}^4 \left(\hat{G}(t_k) - G(t_k)\right)^2 \quad \text{(Standard MSE)}$$

---

## 3.6 Multi-Task Multi-Horizon Loss

The forecasting objective is treated as a multi-task learning problem where predicting each future time horizon $h \in \{30, 60, 90, 120\text{ min}\}$ constitutes a distinct task:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{4 \cdot B} \sum_{h=1}^4 \sum_{n=1}^B w_h \left(\hat{y}_{n,h} - y_{n,h}\right)^2$$

where:
* $B$ is the batch size.
* $w_h$ represents the task weight for horizon $h$. In this implementation, tasks are weighted equally ($w_h = 1.0, \forall h$), ensuring stable training across both short-term (30 min) and long-term (120 min) horizons.

---

## 3.7 Training Optimization

Parameters are optimized using advanced weight-decay methods and cyclical learning rates.

### 1. AdamW Update Equations
The learnable parameters $\theta$ are updated using decoupled weight decay to prevent regularization leakage into the gradient moments:

$$\mathbf{g}_t = \nabla_{\theta} \mathcal{L}_{\text{total}}(\theta_t)$$

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t \quad \text{(First Moment Vector)}$$

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2 \quad \text{(Second Moment Vector)}$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + w_{\text{decay}}\theta_t \right)$$

where:
* $\beta_1 = 0.9, \beta_2 = 0.999$.
* $\epsilon = 10^{-8}$.
* $w_{\text{decay}} = 10^{-4}$ is the decoupled weight decay factor.
* $\eta_t$ represents the dynamic learning rate.

### 2. Cosine Annealing with Warm Restarts
The learning rate $\eta_t$ is modulated cyclically using a cosine function:

$$\eta_t = \eta_{\text{min}} + \frac{1}{2}\left(\eta_{\text{max}} - \eta_{\text{min}}\right)\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)$$

where:
* $\eta_{\text{max}} = 10^{-3}$, $\eta_{\text{min}} = 10^{-6}$.
* $T_{\text{cur}}$ is the number of epochs since the last restart.
* $T_i = 10$ is the restart epoch interval.

### 3. Adaptive Gradient Clipping
To prevent exploding gradients during large physiological transitions:

$$\mathbf{g}_{\text{clipped}} = \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 \le \gamma \\ \gamma \frac{\mathbf{g}}{\...

[Output truncated for brevity]
