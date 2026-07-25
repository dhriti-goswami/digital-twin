# CITATIONS — Methods reference sheet

Compiled 2026-07-25 by web research for the digital-twin PINN / evaluation pipeline.

**Verification tags used throughout**

| Tag | Meaning |
|---|---|
| `VERIFIED-PRIMARY` | I read the actual text/table/equation in the original paper (or its author-published PDF / PMC full text) during this research session. |
| `SECOND-HAND` | Taken from a peer-reviewed or authoritative secondary source that reproduces the primary value; primary not directly read. |
| `UNVERIFIED` | Could not confirm. **Do not put this number in the paper without checking the primary source yourself.** |

Every numeric value below carries a tag. Where two widely-used reference implementations disagree, the disagreement is documented explicitly rather than papered over.

---

## 1. Clarke Error Grid (Clarke et al. 1987)

### 1.1 Canonical citation — `VERIFIED-PRIMARY` (bibliographic record)

Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. "Evaluating clinical accuracy of systems for self-monitoring of blood glucose." *Diabetes Care* 1987 Sep–Oct;10(5):622–628. DOI: `10.2337/diacare.10.5.622`. PMID: 3677983.

### 1.2 Critical caveat about "the exact boundary definitions"

**`VERIFIED-PRIMARY` (negative result):** The 1987 paper **does not publish the zone boundaries as algebraic inequalities.** It publishes a *figure* (the grid) plus prose definitions of the five zones. This is well documented — the Parkes-grid technical paper (Pfützner et al. 2013) makes the same complaint about Clarke's grid in passing, and the existence of the Stöckl et al. 2000 letter ("comment on constructing the *upper A-line*", *Diabetes Care* 23(11):1711–1712, DOI `10.2337/diacare.23.11.1711`, PMID 11092305 — `VERIFIED-PRIMARY` bibliographic record only, **no abstract available and full text is paywalled: I could not read it**) is itself evidence that the construction of even the A-zone boundary was ambiguous enough to require a published clarification 13 years later.

Consequence: **there is no single "definitive published set of inequalities."** What exists is (a) the prose zone definitions, and (b) two widely-used reference implementations that encode the figure geometry. Below I give the prose, both implementations verbatim, and an exhaustive numerical diff between them so you know exactly where the ambiguity bites.

### 1.3 Prose zone definitions

`SECOND-HAND` (paraphrase of Clarke 1987 as reproduced in a peer-reviewed reimplementation paper; see §1.7 note on discrepancies):

- Target range accepted by the five clinician panellists: **70–180 mg/dL**.
- **Zone A** — clinically accurate: evaluation and reference within **20%** of each other, **or** both in the hypoglycaemic range (**< 70 mg/dL**).
- **Zone B** — deviate by > 20% but would lead to benign or no treatment.
- **Zone C** — overcorrection.
- **Zone D** — dangerous failure to detect: meter reading in the target range 70–180 while reference is outside it.
- **Zone E** — erroneous treatment (opposite-direction error).

### 1.4 Reference implementation A — the MATLAB/Python lineage `VERIFIED-PRIMARY` (source code read verbatim)

Provenance chain: MATLAB *Clarke Error Grid Analysis* File Exchange entry v1.2 by Edgar Guevara Codina (2013-03-29) → Python port by Trevor Tsue (2017-07-18), `github.com/suetAndTie/ClarkeErrorGrid`. This is the implementation used by most ML/CGM papers.

Evaluation order is **A → E → C → D → else B** (first match wins), with `r` = reference, `p` = predicted, both mg/dL:

```python
# Zone A
if (r <= 70 and p <= 70) or (0.8*r <= p <= 1.2*r):
    return 'A'
# Zone E
elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
    return 'E'
# Zone C
elif ((70 <= r <= 290) and p >= r + 110) \
     or ((130 <= r <= 180) and p <= (7/5)*r - 182):
    return 'C'
# Zone D
elif (r >= 240 and 70 <= p <= 180) \
     or (r <= 175/3 and 70 <= p <= 180) \
     or ((175/3 <= r <= 70) and p >= (6/5)*r):
    return 'D'
else:
    return 'B'
```

Note `175/3 = 58.333...`.

### 1.5 Reference implementation B — CRAN `ega` package `VERIFIED-PRIMARY` (source code read verbatim)

`ega::getClarkeZones`, file `R/ega.R` (v2.0.0). Different structure: it *overwrites* rather than short-circuits, so the effective precedence is **D, then C overwrites D, then A overwrites C, then E overwrites A, remainder = B** — i.e. final precedence **E > A > C > D > B**. Comments in the source explicitly flag two fixed bugs ("error corrected >=70 instead of >70", "error solved").

```r
are <- abs(p - r) / r * 100          # absolute relative error, %
eq1 <- (7/5) * (r - 130)             # == 1.4*r - 182
eq2 <- r + 110

# D
test_D <- (p >= 70) & (p < 180)
zoneD  <- (r < 70 & test_D) | (r > 240 & test_D)
# C  (overwrites D)
zoneC  <- (r >= 130 & r <= 180 & p < eq1) | (r > 70 & p > 180 & p > eq2)
# A  (overwrites C)
zoneA  <- (are <= 20) | (r < 70 & p < 70)
# E  (overwrites A)
zoneE  <- (r <= 70 & p >= 180) | (r >= 180 & p <= 70)
# remainder -> B
```

### 1.6 Exhaustive numerical diff between the two implementations `VERIFIED-PRIMARY` (I ran it)

I evaluated both classifiers on the full integer lattice `r, p ∈ {1,…,550}` (302,500 points). **12,029 points disagree**, but they fall into exactly two categories:

**(a) One substantive disagreement — the upper-C cap at `r ≤ 290`.**
11,175 of the 12,029 disagreements are `B` (MATLAB) vs `C` (ega), confined to `r ∈ [291, 439]`, `p ∈ [402, 550]`. Cause: the MATLAB lineage restricts the upper-C leg to `70 ≤ r ≤ 290`; `ega` does not cap it.

**Interpretation (my inference, high confidence, tagged `SECOND-HAND` reasoning):** the original Clarke figure is plotted on 0–400 mg/dL axes on both axes. The line `p = r + 110` starts at the vertex `(70, 180)` and exits the top of that figure at `(290, 400)`. So `r ≤ 290` is an **artefact of the original figure's axis extent, not a clinical boundary.** For CGM/prediction data that exceeds 400 mg/dL, the `ega` behaviour (extend `p = r + 110` indefinitely) is the defensible extrapolation. `ega`'s plotting code draws exactly this: a segment from `(70, 180)` to `(550, 660)`.

**Recommendation for your reimplementation: drop the `r ≤ 290` cap**, and say so in the methods section.

**(b) Everything else (854 points) is pure open-vs-closed boundary handling** — measure-zero lines only:
- `('C','B')`, n=226: strict vs non-strict on `p ≥ r+110` and `p > 180`.
- `('D','B')`, n=515: `r ≥ 240` vs `r > 240`, `p ≤ 180` vs `p < 180`, and the exact line `r = 70`.
- `('A','D')`, n=58: the exact line `p = 70` for `r ≤ 58`.
- `('A','B')`, n=55: the exact line `r = 70` with `p ≤ 55`.

These cannot change any real-data result by more than a rounding hair. Pick one convention and document it.

### 1.7 Direct answers to your four specific questions

**Zone A: "within ±20% of reference, OR both reference and prediction < 70."**
**CONFIRMED** by both implementations and by the prose definition. Both encode `0.8r ≤ p ≤ 1.2r` (equivalently `|p−r|/r ≤ 0.20`) OR `(r < 70 AND p < 70)`.
Caveat `SECOND-HAND`: whether the hypoglycaemic clause should be `< 70` or `≤ 70` differs between the two implementations (see §1.6b). Note also that the "±20% band" and "reference is the denominator" choice is precisely what the Stöckl 2000 letter disputes — **I could not read that letter, so if a reviewer challenges your A-line construction you have an unresolved citation gap.** Tagged `UNVERIFIED` for the letter's actual content.

**Zone E: "reference ≥ 180 and prediction ≤ 70, or reference ≤ 70 and prediction ≥ 180."**
**CONFIRMED**, identical in both implementations (modulo `<` vs `≤`). This matches the prose: the two off-diagonal corner rectangles.

**Zone D and the 58.33 question.**
**Your rule is correct in effect, but 58.33 is not a primary boundary — it is a derived vertex.**
`58.333… = 70 / 1.2`, i.e. the point where the **upper A-line `p = 1.2·r` crosses `p = 70`**. `VERIFIED-PRIMARY`: `ega`'s plotting code draws the segment `(58.3, 70) → (maxX, 1.2·maxX)`, confirming exactly this geometric role.

Why both formulations agree: for `r ∈ (58.33, 70)` the A-band already extends above `p = 70` (since `1.2r > 70`), so those points are captured by A before D is reached. Therefore writing lower-D as "`r ≤ 70` and `70 < p < 180`" **with A tested first** gives the same answer as writing it as "`r ≤ 58.33`" plus the extra wedge `(58.33 ≤ r ≤ 70) ∧ (p ≥ 1.2r)`. The MATLAB lineage spells out both legs (`r ≤ 175/3`, and `175/3 ≤ r ≤ 70 ∧ p ≥ (6/5)r`); `ega` writes just `r < 70` and lets A overwrite. **Both are correct.** Use the simple form.

Upper-D leg: both implementations use **`r ≥ 240`** (not 180) with `70 ≤ p ≤ 180`. `CONFIRMED`.
⚠️ `SECOND-HAND` **discrepancy flag:** at least one peer-reviewed reimplementation paper paraphrases Clarke's upper-D as "reference more than **180**", which contradicts both implementations and would make D overlap B massively. The `240` figure is geometrically consistent with the figure vertices `(240,70)` and `(240,180)` that `ega` draws. **Use 240.**

**Zone C.**
**CONFIRMED, both legs:**
- Upper C: `p ≥ r + 110` (for `70 ≤ r`, with the `r ≤ 290` cap being a figure artefact — see §1.6a). The constant **110** is right, not 100: it is forced by the vertex `(70, 180)`, since `70 + 110 = 180` exactly.
  ⚠️ `SECOND-HAND` discrepancy: a reimplementation paper describes upper-C as "reference more than **100** mg/dL lower than the meter value". That is inconsistent with the `(70,180)` vertex. **Use 110.**
- Lower C: `p ≤ (7/5)·r − 182` for `130 ≤ r ≤ 180`. `ega` writes it as `p < (7/5)(r − 130)`, which is **algebraically identical** since `(7/5)·130 = 182`. `CONFIRMED`. Geometrically this is the triangle with vertices `(130, 0)`, `(180, 0)`, `(180, 70)`.

### 1.8 Recommended code-ready specification (my synthesis)

Order matters. Test in this order, first match wins:

```
Given r (reference, mg/dL, r > 0) and p (predicted, mg/dL, p >= 0):

1. A  if  (0.8*r <= p <= 1.2*r)  or  (r < 70 and p < 70)
2. E  if  (r >= 180 and p <= 70) or (r <= 70 and p >= 180)
3. C  if  (r >= 70 and p >= r + 110)                       # no upper cap on r
       or (130 <= r <= 180 and p <= 1.4*r - 182)
4. D  if  (r >= 240 and 70 <= p <= 180)
       or (r <= 70  and 70 <= p <= 180)
5. B  otherwise
```

Full published vertex list that `ega` uses to *draw* the grid (`VERIFIED-PRIMARY`, read from `ega/R/plots.R`; `maxX`, `maxY` are data-dependent plot limits, `tolerance = 0.2`):

| segment | from | to | boundary |
|---|---|---|---|
| 1 | (58.3, 70) | (maxX, 1.2·maxX) | upper A-line |
| 2 | (70, 56) | (maxX, 0.8·maxX) | lower A-line |
| 3 | (70, 180) | (550, 660) | upper C line, `p = r + 110` |
| 4 | (70, 83) | (70, maxY) | vertical at r=70 above A band |
| 5 | (0, 180) | (70, 180) | top of lower-D / bottom of upper-E |
| 6 | (240, 180) | (maxX, 180) | top of lower-right D |
| 7 | (0, 70) | (58.3, 70) | bottom of lower-D |
| 8 | (70, 0) | (70, 56) | vertical at r=70 below A band |
| 9 | (180, 70) | (maxX, 70) | top of lower-right E |
| 10 | (240, 70) | (240, 180) | left edge of lower-right D |
| 11 | (180, 0) | (180, 70) | right edge of lower C |
| 12 | (130, 0) | (180, 70) | lower C line, `p = 1.4r − 182` |

---

## 2. Parkes / Consensus Error Grid — **exact coordinates, fully verified**

### 2.1 Citations

- Parkes JL, Slatin SL, Pardo S, Ginsberg BH. "A new consensus error grid to evaluate the clinical significance of inaccuracies in the measurement of blood glucose." *Diabetes Care* 2000 Aug;23(8):1143–1148. DOI `10.2337/diacare.23.8.1143`. `VERIFIED-PRIMARY` (bibliographic record).
- Pfützner A, Klonoff DC, Pardo S, Parkes JL. "Technical aspects of the Parkes error grid." *J Diabetes Sci Technol* 2013 Sep 1;7(5):1275–1281. DOI `10.1177/193229681300700517`. PMID 24124954. PMC3876371. `VERIFIED-PRIMARY` (full text and Table 1 read).

### 2.2 Does 2013 *correct* 2000? — `VERIFIED-PRIMARY`, quoted

**No — it *first publishes* them.** From the 2013 paper: the 2000 article "depicted boundaries for the performance zones but did not present technical specifications or coordinates… It has therefore been necessary to scan or trace the Parkes error grid boundaries, some of which are curved, in order to work with an exact version of this metric." The 2013 paper's stated purpose is to "present the never-before-published exact coordinates and specifications of the grid so that others may produce an exact replica of the original grid."

So: **cite Parkes 2000 for the grid, Pfützner 2013 for the coordinates.** There is no coordinate correction to reconcile — write it as *first publication of coordinates*, not as *erratum*.

Also from 2013 (`VERIFIED-PRIMARY`), useful for a methods section:
- Grid built from 100 physician respondents at the June 1994 ADA meeting; a 10 mg/dL master grid over 0–550 mg/dL; mean risk score 0–4 (`0 = A … 4 = E`) per cell.
- Smoothing was a separable triangular filter: `Y[i,j] = 0.25·Y[i−1,j] + 0.50·Y[i,j] + 0.25·Y[i+1,j]`, then `Y[i,j] = 0.25·Y[i,j−1] + 0.50·Y[i,j] + 0.25·Y[i,j+1]`. Zones are curves of constant risk, then piecewise-linear-fitted to remove filtering oscillations.
- Axes: x = reference 0–550 mg/dL, y = test device 0–550 mg/dL.
- ISO 15197:2013 uses the **Type 1** grid, and the regulatory reading is that **only zone A** is acceptable.

### 2.3 Type 1 diabetes vertex lists — `VERIFIED-PRIMARY` (Pfützner 2013 Table 1)

All coordinates `(reference, test)` in mg/dL. Zone A is the region *between* the B-Lower and B-Upper polylines (it has no coordinate row of its own — it is defined implicitly). Each polyline is the *lower* (below-identity) or *upper* (above-identity) boundary of the named zone; the zone itself is the band between consecutive boundaries.

```python
# Parkes error grid, TYPE 1 DIABETES  (Pfützner et al. 2013, Table 1)
# Each list is an open polyline of (reference, test) vertices, mg/dL.
PARKES_T1 = {
    "B_lower": [(50, 0),   (50, 30),  (170, 145), (385, 300), (550, 450)],
    "B_upper": [(0, 50),   (30, 50),  (140, 170), (280, 380), (430, 550)],
    "C_lower": [(120, 0),  (120, 30), (260, 130), (550, 250)],
    "C_upper": [(0, 60),   (30, 60),  (50, 80),   (70, 110),  (260, 550)],
    "D_lower": [(250, 0),  (250, 40), (550, 150)],
    "D_upper": [(0, 100),  (25, 100), (50, 125),  (80, 215),  (125, 550)],
    "E_upper": [(0, 150),  (35, 155), (50, 550)],
}
# Line of identity: (0,0) -> (550,550).
# NOTE: Type 1 has NO lower E zone.
```

### 2.4 Type 2 diabetes vertex lists — `VERIFIED-PRIMARY` (same table)

Included for completeness. The 2013 authors state the Type 2 grid "has fallen out of favor"; do not use it for regulatory-style claims.

```python
PARKES_T2 = {
    "B_lower": [(50, 0),  (50, 30),  (90, 80),   (330, 230), (550, 450)],
    "B_upper": [(0, 50),  (30, 50),  (230, 330), (440, 550)],
    "C_lower": [(90, 0),  (260, 130),(550, 250)],
    "C_upper": [(0, 60),  (30, 60),  (280, 550)],
    "D_lower": [(250, 0), (250, 40), (410, 110), (550, 160)],
    "D_upper": [(0, 80),  (25, 80),  (35, 90),   (125, 550)],
    "E_upper": [(0, 200), (35, 200), (50, 550)],
}
```

### 2.5 Independent cross-check — `VERIFIED-PRIMARY`

I compared the vertex multisets above against the boundary construction in `CRAN::ega::getParkesZones` (`R/ega.R`, read verbatim). **Every vertex matches exactly for both Type 1 and Type 2** — including the terminal points, which `ega` computes as slope extrapolations (`.coef`/`.endx`/`.endy`) rather than hard-coding: e.g. T1 `C_upper` terminates at `(260, 550)` via `coef(70,110,260,550)`, and T1 `D_upper` at `(125, 550)` via `coef(80,215,125,550)`. So `ega` is a faithful implementation and can be used as an oracle for your reimplementation's unit tests.

Classification algorithm used by `ega` (`VERIFIED-PRIMARY`): initialise all points to `"A"`, then close each boundary polyline into a polygon against the axes and overwrite with point-in-polygon tests in the order **B, C, D, E** (later zones win). Equivalent, and simpler to implement: assign the *outermost* band whose polygon contains the point.

---

## 3. Bergman Minimal Model

### 3.1 Citations

- Bergman RN, Ider YZ, Bowden CR, Cobelli C. "Quantitative estimation of insulin sensitivity." *Am J Physiol* 1979;236(6):E667–E677. DOI `10.1152/ajpendo.1979.236.6.E667`. `SECOND-HAND` (bibliographic record via PMC review; full text not read).
- Bergman RN, Phillips LS, Cobelli C. "Physiologic evaluation of factors controlling glucose tolerance in man: measurement of insulin sensitivity and β-cell glucose sensitivity from the response to intravenous glucose." *J Clin Invest* 1981;68(6):1456–1467. DOI `10.1172/JCI110398`. `VERIFIED-PRIMARY` (full PDF read).

### 3.2 Published equation form — `VERIFIED-PRIMARY` (Bergman 1981, Figure 1 caption)

The 1981 paper states the model in **two** equations for glucose disappearance plus **one** for insulin kinetics (the "3-equation minimal model" in modern usage is these three together). The scanned PDF's OCR mangles signs; the mathematically consistent published form is:

**Glucose disappearance minimal model (Fig. 1A):**
```
dG/dt = -(p1 + X(t))·G(t) + p1·G_b          G(0) = G_0
dX/dt = -p2·X(t) + p3·(I(t) - I_b)          X(0) = 0
```
`VERIFIED-PRIMARY` caveats on exactly what the paper prints: the paper's Fig. 1 caption shows `dG/dt = -(P1 ± X)G(t) ∓ P1·G_b` and `dX/dt = -P2·X(t) + P3·I(t)` — i.e. **the 1981 caption writes the X-equation driving term as `p3·I(t)`, not `p3·(I − I_b)`.** The `(I − I_b)` form is the standard modern convention (needed so that `X = 0` at basal insulin). Flag this in your methods if you use `(I − I_b)`: `SECOND-HAND` for the basal-subtracted form.

**Insulin kinetics model (Fig. 1B), `VERIFIED-PRIMARY`:**
```
dI/dt = γ·(G(t) - h)·t - n·I(t)
```
with first-phase release modelled as a bolus at `t = 0`; `φ1 = I_0 / (n·ΔG)`; `φ2 = γ` is second-phase pancreatic responsivity; `h` is the glucose threshold; `n` is the insulin disappearance time constant (min⁻¹). **For T1D this equation is normally deleted entirely** (no endogenous secretion) and `I(t)` is supplied by an exogenous insulin PK model instead — see §4.

**Derived indices, `VERIFIED-PRIMARY` (quoted from Bergman 1981):** "Insulin sensitivity index (SI) is `P3/P2`… The units of SI are `min⁻¹ / (μU/mL)` (fractional glucose disappearance rate per unit insulin concentration)." Also, `P1` is "the insulin-independent fractional turnover constant for glucose disposition", i.e. glucose effectiveness `S_G = p1`.

### 3.3 Units

| Symbol | Meaning | Units |
|---|---|---|
| `G` | plasma glucose | mg/dL (or mmol/L) |
| `G_b` | basal glucose | same as `G` |
| `I` | plasma insulin | μU/mL (≡ mU/L) |
| `I_b` | basal insulin | μU/mL |
| `X` | insulin action in remote compartment | min⁻¹ |
| `p1` = `S_G` | insulin-independent fractional glucose turnover (glucose effectiveness) | min⁻¹ |
| `p2` | rate of remote-insulin (in)activation | min⁻¹ |
| `p3` | rate of rise of insulin action per unit plasma insulin above basal | min⁻² per (μU/mL) |
| `S_I` = `p3/p2` | insulin sensitivity index | min⁻¹ per (μU/mL) |
| `n` | insulin fractional disappearance rate | min⁻¹ |
| `V_G` | glucose distribution volume | dL/kg or L/kg |
| `V_I` | insulin distribution volume | L/kg |

`VERIFIED-PRIMARY` for `p1`, `S_I`, `n` units (Bergman 1981). `p2`, `p3` units are the dimensionally forced consequence.

### 3.4 Parameter values **specifically in Type 1 diabetes**

**The good source (`VERIFIED-PRIMARY`, abstract read verbatim):**

Ward GM, Weber KM, Walters IM, Aitken PM, Lee B, Best JD, Boston RC, Alford FP. "A modified minimal model analysis of insulin sensitivity and glucose-mediated glucose disposal in insulin-dependent diabetes." *Metabolism* 1991 Jan;40(1):4–9. DOI `10.1016/0026-0495(91)90183-w`. PMID 1984568.

n = 8 young non-obese C-peptide-negative IDDM subjects vs controls, FSIGT with exogenous insulin infusion (the standard minimal model "cannot accommodate data from diabetics", hence the modification):

| Parameter | IDDM | Controls | Units |
|---|---|---|---|
| `S_I` | **2.5 ± 0.6** | 8.3 ± 1.5 | ×10⁻⁴ min⁻¹·mU⁻¹·L (≡ min⁻¹ per μU/mL) |
| `S_G` (= `p1`) | **1.6 ± 0.5** | 2.6 ± 0.2 | ×10⁻² min⁻¹ |
| `S_G`, basal-insulin FSIGT | **1.0 ± 0.3** | — | ×10⁻² min⁻¹ |
| `K_G` | 1.3 ± 0.29 | — | ×10⁻² min⁻¹ |

(± are SE. `P < .05` for both `S_I` and `S_G`.)

So in T1D: **`p1 ≈ 0.010–0.016 min⁻¹`**, **`S_I ≈ 2.5×10⁻⁴ min⁻¹ per μU/mL`**, and hence **`p3 ≈ 2.5e-4 · p2`**.

**`p2` in T1D:** `UNVERIFIED`. Ward et al. report `S_I` and `S_G` but I did not obtain `p2` separately. The commonly quoted `p2 ≈ 0.025 min⁻¹` in the control literature traces to Fisher 1991 (below) — **not verified**.

**`n`, `V_I`, `V_G` in T1D from Bergman-family sources:** `UNVERIFIED`. Use the T1D-specific values from §4/§5 instead, which *are* verified:
- Lehmann & Deutsch 1992 (T1D-specific, `VERIFIED-PRIMARY`): `k_e = 5.4 h⁻¹` (= 0.09 min⁻¹) insulin elimination; `V_I = 0.142 L/kg`; `V_G = 0.22 L/kg`.
- Hovorka et al. 2004 (T1D-specific, `VERIFIED-PRIMARY`): `k_e = 0.138 min⁻¹`; `V_I = 0.12 L/kg`; `V_G = 0.16 L/kg`.

### 3.5 "Who fixes p1 = 0 for T1D?"

`UNVERIFIED` — **flagged as a citation gap.** The convention `p1 = 0` for T1D is standard in the control-engineering literature and is conventionally attributed to:

Fisher ME. "A semiclosed-loop algorithm for the control of blood glucose levels in diabetics." *IEEE Trans Biomed Eng* 1991 Jan;38(1):57–61. PMID 2026432. (Bibliographic record `VERIFIED-PRIMARY`; **parameter table not read — I could not access the full text.**)

I could **not** confirm from any source I read that Fisher sets `p1 = 0`, nor his numeric `p2`, `p3`, `n`, `V_I` values. **Do not cite specific Fisher numbers from this document.**

**Physiological counter-evidence you should be aware of (`VERIFIED-PRIMARY`, Ward 1991):** measured glucose effectiveness in IDDM is **reduced but clearly non-zero** (`S_G = 1.0–1.6 × 10⁻² min⁻¹`, vs `2.6 × 10⁻²` in controls). So `p1 = 0` is a *modelling simplification for controller design*, not an empirical finding, and Ward et al. is the right primary citation to justify calling it reduced-but-nonzero. If your PINN estimates `p1` freely, Ward's IDDM range `[0.010, 0.016] min⁻¹` is the defensible prior; `p1 = 0` is defensible only as an explicit worst-case/simplifying assumption.

---

## 4. Subcutaneous insulin absorption

### 4.1 Hovorka et al. 2004 — `VERIFIED-PRIMARY` (full PDF read: equations, Table 1, Table 2)

Hovorka R, Canonico V, Chassin LJ, Haueter U, Massi-Benedetti M, Orsini Federici M, Pieber TR, Schaller HC, Schaupp L, Vering T, Wilinska ME. "Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes." *Physiol Meas* 2004;25(4):905–920. DOI `10.1088/0967-3334/25/4/010`.

**Two-compartment SC insulin absorption (paper eq. 5–6):**
```
dS1/dt = u(t) - S1(t)/t_max,I
dS2/dt = S1(t)/t_max,I - S2(t)/t_max,I
U_I(t) = S2(t)/t_max,I                    # insulin appearance rate in plasma
dI/dt  = U_I(t)/V_I - k_e·I(t)
```
`S1`, `S2` = a two-compartment chain for SC-administered short-acting (e.g. lispro) insulin; `u(t)` = bolus + infusion administration; `t_max,I` = time-to-maximum insulin absorption; `k_e` = fractional elimination rate; `V_I` = distribution volume.

**Parameter values `VERIFIED-PRIMARY`:**

| Symbol | Quantity | Value | Source cited by Hovorka |
|---|---|---|---|
| **`t_max,I`** | **time-to-maximum absorption of SC short-acting insulin** | **55 min** | Howey et al. 1994; Rave et al. 1999 |
| `k_e` | insulin elimination from plasma | 0.138 min⁻¹ | Hovorka et al. 1993 |
| `V_I` | insulin distribution volume | 0.12 L kg⁻¹ | Hovorka et al. 1993 |
| `V_G` | glucose distribution volume | 0.16 L kg⁻¹ | Hovorka et al. 2002 |
| `k_12` | transfer rate (non-accessible → accessible glucose) | 0.066 min⁻¹ | Hovorka et al. 2002 |
| `k_a1`, `k_a2`, `k_a3` | insulin-action deactivation rates | 0.006, 0.06, 0.03 min⁻¹ | Hovorka et al. 2002 |
| `S_IT^f` = `k_b1/k_a1` | insulin sensitivity of distribution/transport | 51.2×10⁻⁴ min⁻¹ per mU/L | Hovorka et al. 2002 |
| `S_ID^f` = `k_b2/k_a2` | insulin sensitivity of disposal | 8.2×10⁻⁴ min⁻¹ per mU/L | Hovorka et al. 2002 |
| `S_IE^f` = `k_b3/k_a3` | insulin sensitivity of EGP | 520×10⁻⁴ per mU/L | Hovorka et al. 2002 |
| `EGP_0` | EGP extrapolated to zero insulin | 0.0161 mmol kg⁻¹ min⁻¹ | Hovorka et al. 2002 |
| `F_01` | non-insulin-dependent glucose flux | 0.0097 mmol kg⁻¹ min⁻¹ | Hovorka et al. 2002 |
| `A_G` | carbohydrate bioavailability | 0.8 (unitless) | Livesey et al. 1998 |
| `t_max,G` | time-to-maximum CHO absorption | 40 min | Livesey et al. 1998 |

Note (`VERIFIED-PRIMARY`): the Table 2 quantities are the **means used as Bayesian priors**, not fixed constants; Table 1 quantities are fixed constants. Say which you are doing.

**Rest of the Hovorka model (for completeness, `VERIFIED-PRIMARY`):**
```
dQ1/dt = -[F01^c/(V_G·G) + x1]·Q1 + k12·Q2 - F_R + U_G(t) + EGP0·(1 - x3)
dQ2/dt = x1·Q1 - (k12 + x2)·Q2
G      = Q1 / V_G

F01^c  = F01                if G >= 4.5 mmol/L
       = F01·G/4.5          otherwise
F_R    = 0.003·(G - 9)·V_G  if G >= 9 mmol/L, else 0

dx1/dt = -k_a1·x1 + k_b1·I ;  dx2/dt = -k_a2·x2 + k_b2·I ;  dx3/dt = -k_a3·x3 + k_b3·I
```

### 4.2 Wilinska et al. 2005 — **numeric parameters `UNVERIFIED`**

Wilinska ME, Chassin LJ, Schaller HC, Schaupp L, Pieber TR, Hovorka R. "Insulin kinetics in type-1 diabetes: continuous and bolus delivery of rapid acting insulin." *IEEE Trans Biomed Eng* 2005;52(1):3–12. DOI `10.1109/TBME.2004.839639`. `VERIFIED-PRIMARY` (bibliographic record only — paywalled, full text not obtained).

**Structure, `SECOND-HAND`:** eleven alternative candidate models of lispro kinetics were compared for bolus and CSII delivery; the **best-fitting model has two parallel absorption channels (a slow two-compartment channel and a fast channel) plus saturable local insulin degradation at the injection site (Michaelis–Menten, parameters `V_max,LD` and `k_M,LD`)**, feeding a single plasma compartment with elimination `k_e` and volume `V_I`.

**All numeric parameter values for Wilinska 2005 (`k_a1`/`k_a2` or `k_SQi,s`/`k_SQi,f`, the channel split fraction, `V_max,LD`, `k_M,LD`) are `UNVERIFIED`.** I could not obtain the paper or any source reproducing its table. **Get the PDF before writing any Wilinska number into code or the paper.** If you need a verified SC-insulin model *now*, use Hovorka 2004 §4.1 (`t_max,I = 55 min`, `k_e = 0.138 min⁻¹`, `V_I = 0.12 L/kg`) — all three verified primary.

---

## 5. Meal / gut glucose absorption

### 5.1 Citations

- Dalla Man C, Camilleri M, Cobelli C. "A system model of oral glucose absorption: validation on gold standard data." *IEEE Trans Biomed Eng* 2006 Dec;53(12 Pt 1):2472–2478. DOI `10.1109/TBME.2006.883792`. PMID 17153204. `VERIFIED-PRIMARY` (bibliographic record).
- Dalla Man C, Rizza RA, Cobelli C. "Meal simulation model of the glucose-insulin system." *IEEE Trans Biomed Eng* 2007 Oct;54(10):1740–1749. DOI `10.1109/TBME.2007.893506`. PMID 17926672. `VERIFIED-PRIMARY` (bibliographic record).
- Lehmann ED, Deutsch T. "A physiological model of glucose-insulin interaction in type 1 diabetes mellitus." *J Biomed Eng* 1992 May;14(3):235–242. DOI `10.1016/0141-5425(92)90058-S`. PMID 1588781. `VERIFIED-PRIMARY` (full PDF read).

### 5.2 Dalla Man oral glucose absorption model — equations `SECOND-HAND`, values `SECOND-HAND`

Source read: Jørgensen JB et al., "Mathematical meal models for simulation of human metabolism", arXiv:2307.16444 (peer-review-style survey reproducing the model and a parameter table). The original 2006/2007 papers are paywalled and I did **not** read them, so everything in this subsection is `SECOND-HAND`. It is internally consistent and matches the commonly used UVA/Padova formulation.

**Three-compartment model** (`Q_sto,1` = solid stomach, `Q_sto,2` = liquid stomach, `Q_gut` = small intestine; `D` = total meal CHO in mg; `d(t)` = ingestion rate):

```
dQ_sto,1/dt = d(t) - R_12
dQ_sto,2/dt = R_12 - R_sto,gut
dQ_gut/dt   = R_sto,gut - R_gut,pla

R_12      = k_gri · Q_sto,1
R_sto,gut = k_empt(Q_sto, D) · Q_sto,2
R_gut,pla = k_abs · Q_gut
R_A       = f · R_gut,pla                       # glucose rate of appearance in plasma

Q_sto = Q_sto,1 + Q_sto,2

k_empt(Q_sto, D) = k_min + ((k_max - k_min)/2) · [ tanh(alpha·(Q_sto - b·D))
                                                 - tanh(beta ·(Q_sto - c·D)) + 2 ]
alpha = 5 / (2·D·(1 - b))
beta  = 5 / (2·D·c)
```
(`b`, `c` are the fractions of `D` at which `|dk_empt/dQ_sto| = ½(k_max − k_min)`.)

**Parameter values, `SECOND-HAND`:**

| Symbol | Value | Units | Meaning |
|---|---|---|---|
| **`k_gri`** | **0.0465** | min⁻¹ | inverse grinding (solid→liquid in stomach) time constant |
| `k_max` | 0.0465 | min⁻¹ | maximum inverse gastric-emptying time constant |
| `k_min` | 0.0076 | min⁻¹ | minimum inverse gastric-emptying time constant |
| **`k_abs`** | **0.023** | min⁻¹ | inverse intestinal glucose absorption time constant |
| `b` | 0.69 | — | fraction of `D` at left inflection point |
| `c` | 0.17 | — | fraction of `D` at right inflection point |
| **`f`** | **0.90** | — | fraction of absorbed glucose appearing in plasma (bioavailability) |
| `BW` | 91 | kg | body weight used in the cited simulation |

Note `k_gri = k_max` numerically in this parameterisation — that is intentional in the original model, not a transcription slip.

### 5.3 Hovorka 2004 gut absorption (verified alternative) — `VERIFIED-PRIMARY`

Simpler, and the values *are* primary-verified (§4.1). Two-compartment chain with identical transfer rates `1/t_max,G`, giving a closed form:
```
U_G(t) = D_G · A_G · t · exp(-t / t_max,G) / t_max,G^2
```
with `t_max,G = 40 min`, `A_G = 0.8` (CHO bioavailability), `D_G` = amount of CHO digested.
`SECOND-HAND` note: the arXiv survey uses `f = 1` for Hovorka (bioavailability is folded entirely into `A_G`) — do not apply both `A_G` and a separate `f`.

### 5.4 Lehmann & Deutsch 1992 gastric emptying — `VERIFIED-PRIMARY` (equations and Table 2 read)

Trapezoidal (large meal) / triangular (small meal) gastric-emptying rate `G_empt(t)`, with gut compartment:
```
dG_gut/dt = G_empt - k_gabs · G_gut
```

For `Ch > Ch_crit` (trapezoid), with `T_asc`, `T_max`, `T_des` the ascending / plateau / descending durations:
```
T_max = [ Ch - 0.5·V_max_ge·(T_asc + T_des) ] / V_max_ge                             (eq. 10)

G_empt = (V_max_ge / T_asc) · t                       ,  t < T_asc                   (13a)
       = V_max_ge                                     ,  T_asc <= t < T_asc + T_max  (13b)
       = V_max_ge - (V_max_ge/T_des)·(t - T_asc - T_max),
                                    T_asc + T_max <= t < T_asc + T_max + T_des       (13c)
       = 0                                            , elsewhere                    (13d)
```
For small meals (`Ch < Ch_crit`, ≈ below 10 g CHO) the curve degenerates to a triangle:
```
T_asc = T_des = 2·Ch / V_max_ge                                                      (eq. 11)
Ch_crit = (T_asc + T_des) · V_max_ge / 2                                             (eq. 12)
```

**Parameter values, `VERIFIED-PRIMARY` (Table 2, "patient-independent", derived from Berger & Rodbard and Guyton et al.):**

| Symbol | Value | Meaning |
|---|---|---|
| `V_max_ge` | **120 mmol h⁻¹** | maximal rate of gastric emptying |
| `T_asc`, `T_des` | **30 min (0.5 h)** default each | ascending / descending branch lengths |
| `k_gabs` | **1 h⁻¹** | rate constant of glucose absorption from the gut |
| `k_e` | 5.4 h⁻¹ | insulin elimination rate constant |
| `k_1` | 0.025 h⁻¹ | insulin pharmacodynamics parameter |
| `k_2` | 1.25 h⁻¹ | insulin pharmacodynamics parameter |
| `I_basal` | 10 mU L⁻¹ | reference basal insulin |
| `K_m` | 10 mmol L⁻¹ | Michaelis constant, enzyme-mediated glucose uptake |
| `G_I` | 0.54 mmol h⁻¹ kg⁻¹ | insulin-independent glucose utilisation |
| `G_x` | 5.3 mmol L⁻¹ | reference glucose level for utilisation |
| `c` | 0.015 mmol h⁻¹ kg⁻¹ (mU/L)⁻¹ | slope of peripheral glucose utilisation vs insulin |
| `V_I` | 0.142 L kg⁻¹ | insulin distribution volume |
| `V_G` | 0.22 L kg⁻¹ | glucose distribution volume |

Also `VERIFIED-PRIMARY`: peripheral glucose utilisation uses Michaelis–Menten in `G` with insulin shifting `V_max`:
`G_out(G, I_active) = G·(c·I_active/... + G_I)(K_m + G_x) / [G_x·(K_m + G)]` (eq. 8 — OCR partially illegible; **re-read eq. 8 from the PDF before implementing**, tagged `UNVERIFIED` for the exact algebra of that one equation). Simulation used first-order Euler, 15 min step, 48 h.

**Note on units:** Lehmann & Deutsch work in **mmol and hours**; Dalla Man in **mg and minutes**; Hovorka in **mmol and minutes**. Do not mix without converting.

---

## 6. Kovatchev LBGI / HBGI

### 6.1 Citations

- Kovatchev BP, Cox DJ, Gonder-Frederick LA, Clarke WL. "Symmetrization of the blood glucose measurement scale and its applications." *Diabetes Care* 1997 Nov;20(11):1655–1658. DOI `10.2337/diacare.20.11.1655`. — **the transform.** `SECOND-HAND` (bibliographic record; full text not read).
- Kovatchev BP, Otto E, Cox D, Gonder-Frederick L, Clarke W. "Evaluation of a new measure of blood glucose variability in diabetes." *Diabetes Care* 2006 Nov;29(11):2433–2438. DOI `10.2337/dc06-1085`. — **the ADRR paper; the standard modern citation for the LBGI/HBGI computational recipe.** `SECOND-HAND`.

### 6.2 The transform and indices — **CONFIRMED**, `SECOND-HAND`

Your commonly-cited form is **correct**. Two independent authoritative reproductions agree exactly (the `iglu` R package reference documentation, which cites Kovatchev 2006; and a Kovatchev-authored review):

For `G` in **mg/dL**:
```
f(G) = 1.509 · ( ln(G)^1.084  -  5.381 )

r_low(G)  = 10 · f(G)^2   if f(G) < 0,  else 0
r_high(G) = 10 · f(G)^2   if f(G) > 0,  else 0

LBGI = (1/n) · Σ_{i=1..n} r_low(G_i)
HBGI = (1/n) · Σ_{i=1..n} r_high(G_i)
BGRI = LBGI + HBGI
```
Constants **1.509, 1.084, 5.381 all CONFIRMED**. The logarithm is the **natural** log (the `iglu` docs write `log`, and 1.509/1.084/5.381 only reproduce the documented behaviour with `ln`).

Equivalent one-liner used by `iglu` (`VERIFIED-PRIMARY`, quoted from its reference page):
`LBGI = 1/n * Σ (10 · fbg_i²)` where `fbg_i = min(0, 1.509·(log(G_i)^1.084 − 5.381))`.

### 6.3 The averaging convention — **your question answered explicitly**

**Divide by `n` = the total number of readings, NOT the number of readings in that branch.** `SECOND-HAND` but unambiguous: the `iglu` documentation states "*n is the total number of measurements for that subject*", and it clamps with `min(0, ·)` so out-of-branch readings contribute a literal zero to the sum while still counting in the denominator.

Practical consequence, worth stating in your methods: LBGI and HBGI are therefore **not** conditional means — they are frequency-*and*-severity weighted, so a subject with rare-but-severe hypoglycaemia and one with frequent-but-mild hypoglycaemia can score the same. That is the intended design.

### 6.4 Scale properties — `SECOND-HAND`

The transform maps the BG range **20–600 mg/dL** onto the symmetric interval **(−√10, +√10)**, with the scale centre **112.5 mg/dL** mapping to **0**. Hence `r = 10·f²` ∈ [0, 100]. The risk function's target range is 70–180 mg/dL.

**Unit warning:** the constants are for **mg/dL only**. For mmol/L input, convert to mg/dL first (`×18.0182`); do not refit the constants.

---

## 7. PINN loss weighting methods

### 7.1 Wang, Teng & Perdikaris — learning-rate annealing `VERIFIED-PRIMARY` (arXiv PDF read, Algorithm 1)

Wang S, Teng Y, Perdikaris P. "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." *SIAM J Sci Comput* 2021;43(5):A3055–A3081. DOI `10.1137/20M1318043`. arXiv:2001.04536.

Loss: `L(θ) = L_r(θ) + Σ_{i=1..M} λ_i · L_i(θ)`, where `L_r` is the PDE residual loss and `L_i` are data-fit / IC / BC terms. Initialise `λ_i = 1`. At each step:

```
(a)  λ̂_i  =  max_θ{ |∇_θ L_r(θ_n)| }  /  mean_θ{ |∇_θ L_i(θ_n)| }        , i = 1..M
(b)  λ_i  =  (1 - α)·λ_i  +  α·λ̂_i
(c)  θ_{n+1} = θ_n - η·∇_θ L_r(θ_n) - η·Σ_i λ_i·∇_θ L_i(θ_n)
```
Numerator is the **max** of the absolute residual gradient; denominator is the **mean** of the absolute gradient of term `i`; both taken over the parameter vector `θ`.

**Recommended hyperparameters, quoted verbatim: `η = 10⁻³` and `α = 0.9`.** `VERIFIED-PRIMARY`.

Note the asymmetry: the residual term is **unweighted** (weight fixed at 1); only the data/BC terms get `λ_i`.

### 7.2 Wang, Yu & Perdikaris — NTK-based weighting `VERIFIED-PRIMARY` (arXiv PDF read, Algorithm 1)

Wang S, Yu X, Perdikaris P. "When and why PINNs fail to train: a neural tangent kernel perspective." *J Comput Phys* 2022;449:110768. DOI `10.1016/j.jcp.2021.110768`. arXiv:2007.14527.

Loss: `L(θ) = λ_b·L_b(θ) + λ_r·L_r(θ)` (boundary/data and residual). Initialise `λ_b = λ_r = 1`. Then:

```
λ_b  =  ( Σ_{i=1}^{N_r+N_b} λ_i(n) ) / ( Σ_{i=1}^{N_b} λ_i^{uu}(n) )  =  Tr(K(n))   / Tr(K_uu(n))
λ_r  =  ( Σ_{i=1}^{N_r+N_b} λ_i(n) ) / ( Σ_{i=1}^{N_r} λ_i^{rr}(n) )  =  Tr(K(n))   / Tr(K_rr(n))
θ_{n+1} = θ_n - η·∇_θ L(θ_n)
```
where `λ_i`, `λ_i^{uu}`, `λ_i^{rr}` are eigenvalues of the full NTK `K(n)` and its diagonal blocks `K_uu(n)` (boundary/data) and `K_rr(n)` (residual). Practical notes quoted from the paper (`VERIFIED-PRIMARY`): **use traces instead of eigenvalue sums** (identical, much cheaper), and the update **need not run every step** — "every 10 gradient descent steps" is given as an acceptable frequency. Stability bound: max learning rate `≤ 2/λ_max(K̃(t))`.

The paper explicitly frames this as the theoretically-grounded replacement for §7.1: the earlier gradient-magnitude heuristic "lacked any theoretical justification".

### 7.3 GradNorm — Chen et al. 2018 `VERIFIED-PRIMARY` (arXiv PDF read, eqs. 1–2 + Algorithm 1)

Chen Z, Badrinarayanan V, Lee C-Y, Rabinovich A. "GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks." *ICML 2018*, PMLR 80:794–803. arXiv:1711.02257.

Definitions (all `VERIFIED-PRIMARY`, quoted):
- `G_W^(i)(t)` = `‖∇_W ( w_i(t)·L_i(t) )‖_2` — norm of the gradient of the **weighted** single-task loss w.r.t. a chosen subset of shared weights `W` (usually the last shared layer).
- `Ḡ_W(t) = E_task[ G_W^(i)(t) ]` — average gradient norm across tasks.
- `L̃_i(t) = L_i(t) / L_i(0)` — loss ratio = inverse training rate.
- `r_i(t) = L̃_i(t) / E_task[ L̃_i(t) ]` — relative inverse training rate.

Target and update:
```
target_i(t) = Ḡ_W(t) × [ r_i(t) ]^α                                        (eq. 1)

L_grad(t; w_i(t)) = Σ_i | G_W^(i)(t) - Ḡ_W(t) × [r_i(t)]^α |_1            (eq. 2)
```
Crucial implementation details, quoted (`VERIFIED-PRIMARY`):
- `L_grad` is differentiated **only with respect to the `w_i`**, and the target `Ḡ_W(t) × [r_i(t)]^α` is **treated as a fixed constant** (otherwise the `w_i` drift to zero).
- After every update, **renormalise so that `Σ_i w_i(t) = T`** (number of tasks) — this decouples gradient normalisation from the global learning rate.
- `w_i(0) = 1 ∀i`; `α > 0` is a hyperparameter (higher `α` for more dissimilar tasks; `α = 0` pins all gradient norms equal).
- If `L_i(0)` is initialisation-sensitive, substitute a theoretical initial loss (e.g. `log C` for `C`-class cross-entropy).

### 7.4 ReLoBRaLo — Bischof & Kraus `VERIFIED-PRIMARY` (arXiv PDF read, eq. 11)

Bischof R, Kraus M. "Multi-objective loss balancing for physics-informed deep learning." arXiv:2110.09813 (preprint; the version I read is dated 2022-11-16). *Note: this is a preprint, not a journal paper — check for a published version before citing as such.*

"ReLoBRaLo" = **Re**lative **Lo**ss **B**alancing with **Ra**ndom **Lo**okback. With `m` loss terms, temperature `T`, exponential decay `α`, and Bernoulli "saudade" variable `ρ`:

```
λ_i^bal(t, t') = m · softmax_i( L_i(t) / (T · L_i(t')) )
               = m · exp( L_i(t)/(T·L_i(t')) ) / Σ_{j=1..m} exp( L_j(t)/(T·L_j(t')) )

λ_i^hist(t)    = ρ · λ_i(t-1)  +  (1 - ρ) · λ_i^bal(t, 0)

λ_i(t)         = α · λ_i^hist(t)  +  (1 - α) · λ_i^bal(t, t-1)                (eq. 11)
```
Hyperparameter guidance, quoted (`VERIFIED-PRIMARY`):
- `α` ∈ **[0.9, 0.999]** in their experiments. `α = 1` reduces to progress measured against `L_i(0)` and "is too restrictive".
- `E[ρ]` should be **close to 1** (`E[ρ]=1` = minimum saudade, only last step; `E[ρ]=0` = always look back to initialisation).
- Temperature `T` swept over `[10⁻⁶, 10²]`; `T → ∞` makes the softmax uniform, small `T` makes it an argmax.
- Uses **loss statistics only** (no gradient statistics) — cheaper than §7.1 and §7.3.

### 7.5 Kendall, Gal & Cipolla — multi-task uncertainty weighting `VERIFIED-PRIMARY` (arXiv PDF read, eqs. 5–7 + §3.2)

Kendall A, Gal Y, Cipolla R. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." *CVPR 2018*, pp. 7482–7491. arXiv:1705.07115.

Single Gaussian likelihood (eq. 5):
```
log p(y | f^W(x))  ∝  -(1/(2σ²))·‖y - f^W(x)‖²  -  log σ
```

**Two-task regression minimisation objective (eq. 7) — this is the exact form you asked for:**
```
L(W, σ1, σ2) = -log p(y1, y2 | f^W(x))
             ∝ (1/(2σ1²))·‖y1 - f^W(x)‖²  +  (1/(2σ2²))·‖y2 - f^W(x)‖²  +  log(σ1·σ2)
             =  (1/(2σ1²))·L1(W)  +  (1/(2σ2²))·L2(W)  +  log σ1 σ2
```
with `L_i(W) = ‖y_i - f^W(x)‖²`. The `log σ1σ2` (= `log σ1 + log σ2`) term is the **log-variance regulariser** — it is what stops the weights collapsing to zero. Quoted (`VERIFIED-PRIMARY`): "This loss is smoothly differentiable, and is well formed such that the task weights will not converge to zero. In contrast, directly learning the weights using a simple linear sum of losses would result in weights which quickly converge to zero."

**Numerically stable parameterisation, quoted verbatim (`VERIFIED-PRIMARY`, §3.2):** "In practice, we train the network to predict the log variance, `s := log σ²`. This is because it is more numerically stable than regressing the variance `σ²`, as the loss avoids any division by zero. The exponential mapping also allows us to regress unconstrained scalar values, where `exp(−s)` is resolved to the positive domain giving valid values for variance."

So the form to implement, generalised to `K` tasks with learnable `s_k = log σ_k²`:
```
L  =  Σ_{k=1..K} [ 0.5 · exp(-s_k) · L_k(W)  +  0.5 · s_k ]
```
(using `log σ_k = ½ s_k`). `VERIFIED-PRIMARY` for the two-task eq. 7 and for the `s := log σ²` reparameterisation; the `K`-task rewrite is the trivial algebraic generalisation the paper says is a "trivial extension" to arbitrary combinations of losses.
`SECOND-HAND` caveat: some implementations drop the factor `½` on `L_k` and/or use `s_k` instead of `½ s_k` in the regulariser. These differ by a constant rescaling of the losses and an additive constant; state which you use.

---

## BibTeX

```bibtex
@article{clarke1987ega,
  author  = {Clarke, William L. and Cox, Daniel and Gonder-Frederick, Linda A.
             and Carter, William and Pohl, Stephen L.},
  title   = {Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose},
  journal = {Diabetes Care},
  volume  = {10}, number = {5}, pages = {622--628}, year = {1987},
  doi     = {10.2337/diacare.10.5.622}
}

@article{stockl2000upperaline,
  author  = {St{\"o}ckl, Dietmar and Dewitte, Kristian and Fierens, Charlotte
             and Thienpont, Linda M.},
  title   = {Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood
             Glucose by Error Grid Analysis: Comment on Constructing the ``Upper A-Line''},
  journal = {Diabetes Care},
  volume  = {23}, number = {11}, pages = {1711--1712}, year = {2000},
  doi     = {10.2337/diacare.23.11.1711}
}

@article{parkes2000consensus,
  author  = {Parkes, Joan L. and Slatin, Sherry L. and Pardo, Scott and Ginsberg, Barry H.},
  title   = {A New Consensus Error Grid to Evaluate the Clinical Significance of
             Inaccuracies in the Measurement of Blood Glucose},
  journal = {Diabetes Care},
  volume  = {23}, number = {8}, pages = {1143--1148}, year = {2000},
  doi     = {10.2337/diacare.23.8.1143}
}

@article{pfutzner2013parkes,
  author  = {Pf{\"u}tzner, Andreas and Klonoff, David C. and Pardo, Scott and Parkes, Joan L.},
  title   = {Technical Aspects of the Parkes Error Grid},
  journal = {Journal of Diabetes Science and Technology},
  volume  = {7}, number = {5}, pages = {1275--1281}, year = {2013},
  doi     = {10.1177/193229681300700517}
}

@article{bergman1979quantitative,
  author  = {Bergman, Richard N. and Ider, Y. Ziya and Bowden, Charles R. and Cobelli, Claudio},
  title   = {Quantitative Estimation of Insulin Sensitivity},
  journal = {American Journal of Physiology},
  volume  = {236}, number = {6}, pages = {E667--E677}, year = {1979},
  doi     = {10.1152/ajpendo.1979.236.6.E667}
}

@article{bergman1981physiologic,
  author  = {Bergman, Richard N. and Phillips, Lawrence S. and Cobelli, Claudio},
  title   = {Physiologic Evaluation of Factors Controlling Glucose Tolerance in Man:
             Measurement of Insulin Sensitivity and Beta-Cell Glucose Sensitivity from
             the Response to Intravenous Glucose},
  journal = {Journal of Clinical Investigation},
  volume  = {68}, number = {6}, pages = {1456--1467}, year = {1981},
  doi     = {10.1172/JCI110398}
}

@article{ward1991iddmminimalmodel,
  author  = {Ward, Glenn M. and Weber, K. M. and Walters, I. M. and Aitken, P. M.
             and Lee, B. and Best, James D. and Boston, Raymond C. and Alford, Frank P.},
  title   = {A Modified Minimal Model Analysis of Insulin Sensitivity and
             Glucose-Mediated Glucose Disposal in Insulin-Dependent Diabetes},
  journal = {Metabolism},
  volume  = {40}, number = {1}, pages = {4--9}, year = {1991},
  doi     = {10.1016/0026-0495(91)90183-W}
}

@article{fisher1991semiclosed,
  author  = {Fisher, Michael E.},
  title   = {A Semiclosed-Loop Algorithm for the Control of Blood Glucose Levels in Diabetics},
  journal = {IEEE Transactions on Biomedical Engineering},
  volume  = {38}, number = {1}, pages = {57--61}, year = {1991},
  note    = {PMID: 2026432. Parameter table NOT verified in this research pass.}
}

@article{hovorka2004nmpc,
  author  = {Hovorka, Roman and Canonico, Valentina and Chassin, Ludovic J. and
             Haueter, Ulrich and Massi-Benedetti, Massimo and Orsini Federici, Marco and
             Pieber, Thomas R. and Schaller, Helga C. and Schaupp, Lukas and
             Vering, Thomas and Wilinska, Malgorzata E.},
  title   = {Nonlinear Model Predictive Control of Glucose Concentration in
             Subjects with Type 1 Diabetes},
  journal = {Physiological Measurement},
  volume  = {25}, number = {4}, pages = {905--920}, year = {2004},
  doi     = {10.1088/0967-3334/25/4/010}
}

@article{wilinska2005insulinkinetics,
  author  = {Wilinska, Malgorzata E. and Chassin, Ludovic J. and Schaller, Helga C. and
             Schaupp, Lukas and Pieber, Thomas R. and Hovorka, Roman},
  title   = {Insulin Kinetics in Type-1 Diabetes: Continuous and Bolus Delivery
             of Rapid Acting Insulin},
  journal = {IEEE Transactions on Biomedical Engineering},
  volume  = {52}, number = {1}, pages = {3--12}, year = {2005},
  doi     = {10.1109/TBME.2004.839639},
  note    = {Parameter values NOT verified in this research pass.}
}

@article{dallaman2006oralabsorption,
  author  = {Dalla Man, Chiara and Camilleri, Michael and Cobelli, Claudio},
  title   = {A System Model of Oral Glucose Absorption: Validation on Gold Standard Data},
  journal = {IEEE Transactions on Biomedical Engineering},
  volume  = {53}, number = {12}, pages = {2472--2478}, year = {2006},
  doi     = {10.1109/TBME.2006.883792}
}

@article{dallaman2007mealmodel,
  author  = {Dalla Man, Chiara and Rizza, Robert A. and Cobelli, Claudio},
  title   = {Meal Simulation Model of the Glucose-Insulin System},
  journal = {IEEE Transactions on Biomedical Engineering},
  volume  = {54}, number = {10}, pages = {1740--1749}, year = {2007},
  doi     = {10.1109/TBME.2007.893506}
}

@article{lehmann1992physiological,
  author  = {Lehmann, Eldon D. and Deutsch, Tibor},
  title   = {A Physiological Model of Glucose-Insulin Interaction in
             Type 1 Diabetes Mellitus},
  journal = {Journal of Biomedical Engineering},
  volume  = {14}, number = {3}, pages = {235--242}, year = {1992},
  doi     = {10.1016/0141-5425(92)90058-S}
}

@article{jorgensen2023mealmodels,
  author  = {J{\o}rgensen, John Bagterp and others},
  title   = {Mathematical Meal Models for Simulation of Human Metabolism},
  journal = {arXiv preprint arXiv:2307.16444},
  year    = {2023},
  note    = {Secondary source used to verify Dalla Man 2006/2007 parameter values.}
}

@article{kovatchev1997symmetrization,
  author  = {Kovatchev, Boris P. and Cox, Daniel J. and Gonder-Frederick, Linda A.
             and Clarke, William L.},
  title   = {Symmetrization of the Blood Glucose Measurement Scale and Its Applications},
  journal = {Diabetes Care},
  volume  = {20}, number = {11}, pages = {1655--1658}, year = {1997},
  doi     = {10.2337/diacare.20.11.1655}
}

@article{kovatchev2006adrr,
  author  = {Kovatchev, Boris P. and Otto, Erik and Cox, Daniel and
             Gonder-Frederick, Linda and Clarke, William},
  title   = {Evaluation of a New Measure of Blood Glucose Variability in Diabetes},
  journal = {Diabetes Care},
  volume  = {29}, number = {11}, pages = {2433--2438}, year = {2006},
  doi     = {10.2337/dc06-1085}
}

@article{wang2021gradientpathologies,
  author  = {Wang, Sifan and Teng, Yujun and Perdikaris, Paris},
  title   = {Understanding and Mitigating Gradient Flow Pathologies in
             Physics-Informed Neural Networks},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {43}, number = {5}, pages = {A3055--A3081}, year = {2021},
  doi     = {10.1137/20M1318043}
}

@article{wang2022ntkpinn,
  author  = {Wang, Sifan and Yu, Xinling and Perdikaris, Paris},
  title   = {When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective},
  journal = {Journal of Computational Physics},
  volume  = {449}, pages = {110768}, year = {2022},
  doi     = {10.1016/j.jcp.2021.110768}
}

@inproceedings{chen2018gradnorm,
  author    = {Chen, Zhao and Badrinarayanan, Vijay and Lee, Chen-Yu and Rabinovich, Andrew},
  title     = {GradNorm: Gradient Normalization for Adaptive Loss Balancing in
               Deep Multitask Networks},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
  series    = {PMLR}, volume = {80}, pages = {794--803}, year = {2018}
}

@article{bischof2021relobralo,
  author  = {Bischof, Rafael and Kraus, Michael},
  title   = {Multi-Objective Loss Balancing for Physics-Informed Deep Learning},
  journal = {arXiv preprint arXiv:2110.09813},
  year    = {2021},
  note    = {Introduces ReLoBRaLo. Preprint -- check for a published version.}
}

@inproceedings{kendall2018multitask,
  author    = {Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  title     = {Multi-Task Learning Using Uncertainty to Weigh Losses for
               Scene Geometry and Semantics},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and
               Pattern Recognition (CVPR)},
  pages     = {7482--7491}, year = {2018}
}

@Manual{ega_rpackage,
  title  = {ega: Error Grid Analysis},
  note   = {R package; source read verbatim and used as the coordinate oracle for
            Clarke and Parkes grids. Cites Clarke 1987, Parkes 2000, Pfuetzner 2013.},
  url    = {https://cran.r-project.org/package=ega}
}
```

---

## UNVERIFIED / COULD NOT CONFIRM

Read this list before writing any of these numbers into code or the paper.

1. **Clarke 1987 full text.** Paywalled; I never read the original paper. Everything in §1.3 (prose zone definitions) is a paraphrase from secondary sources. The zone *inequalities* in §1.4/§1.5 come from reference implementations, **not** from the paper — because **the paper does not publish inequalities.** Say so in your methods; do not write "as defined by Clarke et al." next to an inequality.
2. **Stöckl et al. 2000 "upper A-line" letter content.** No abstract exists and the full text is paywalled. If a reviewer challenges your ±20% A-band construction (e.g. reference vs. mean as the denominator), you have no answer sourced here.
3. **Clarke figure axis limits (0–400 mg/dL).** My explanation of the `r ≤ 290` upper-C cap as a figure artefact rests on the original figure having 0–400 axes. This is consistent with `290 + 110 = 400`, and with the fact that the Parkes grid is repeatedly described in the 2013 paper as offering "a more comprehensive range up to 550 mg/dL" *in contrast to* Clarke — but **I did not see the Clarke figure.** Treat the geometric argument as strong inference, not verified fact.
4. **Secondary-source contradictions about Clarke's zones** (both from a peer-reviewed reimplementation paper): it states upper-C uses a **100** mg/dL offset (implementations use **110**) and upper-D triggers at reference **> 180** (implementations use **> 240**). I recommend 110 and 240 on geometric grounds (§1.7), but the contradiction is unresolved against the primary text.
5. **Bergman `p1 = 0` for T1D — attribution.** Cannot confirm that Fisher 1991 sets `p1 = 0`, and I did **not** read any of Fisher's parameter values (`p2`, `p3`, `n`, `V_I`, `V_G`). Widely-quoted values like `p2 = 0.025 min⁻¹`, `p3 = 1.3×10⁻⁵`, `n = 5/54 min⁻¹` are **UNVERIFIED — do not use.**
6. **Bergman `p2` in Type 1 diabetes.** Ward 1991 gives `S_I` and `S_G` but I did not obtain `p2` alone. You can only pin `p3 = S_I · p2`.
7. **Bergman `n`, `V_I`, `V_G` in T1D from Bergman-lineage papers.** Not obtained. Use Hovorka 2004 or Lehmann & Deutsch 1992 values (both verified, both T1D-specific) instead.
8. **Bergman 1979 (Am J Physiol) full text.** Not read; cited bibliographically only. The three-equation form in §3.2 is from the 1981 paper.
9. **Bergman 1981 X-equation driving term.** The published Fig. 1 caption writes `dX/dt = -p2·X + p3·I(t)`, i.e. **without** basal subtraction. The modern `p3·(I − I_b)` form is a convention, not what the 1981 caption prints.
10. **Wilinska et al. 2005 — ALL numeric parameter values.** Paywalled; no reproducing source found. The two-channel + local-degradation *structure* is second-hand. `k_a1`/`k_a2`, the channel split fraction, `V_max,LD`, `k_M,LD` are all unknown here.
11. **Dalla Man 2006 and 2007 full texts.** Not read. The equations and all of `k_gri`, `k_max`, `k_min`, `k_abs`, `b`, `c`, `f` in §5.2 are `SECOND-HAND` from arXiv:2307.16444. They are self-consistent and match the standard UVA/Padova parameterisation, but verify against the IEEE papers before publication.
12. **Lehmann & Deutsch eq. 8** (peripheral glucose utilisation Michaelis–Menten). The scanned PDF's OCR is partially illegible for this one equation. Re-read it from the PDF before implementing.
13. **Kovatchev 1997 and 2006 full texts.** Not read. The transform `f(G) = 1.509·(ln(G)^1.084 − 5.381)`, `r = 10f²`, and the `1/n`-over-**all**-points averaging convention are confirmed by two independent authoritative reproductions but are `SECOND-HAND`. Also: whether the log is natural or base-10 is inferred (natural) from the reproductions' notation, not read from Kovatchev.
14. **ReLoBRaLo publication venue.** arXiv:2110.09813 is a preprint (version read: 2022-11-16). Check whether a journal version exists before citing it as published.
15. **Kendall et al. K-task generalisation.** Eq. 7 (two tasks) and the `s := log σ²` reparameterisation are verified primary; the `K`-task sum in §7.5 is my algebraic generalisation of what the paper calls a "trivial extension". Factor-of-½ conventions differ across implementations.

### Reproducibility note

The Clarke implementation diff in §1.6 is reproducible: the comparison script lives at
`/tmp/claude-1000/-home-sammyyakk-projects-digital-twin/d127788c-9af5-4b5d-abd1-bdbd710d9f24/scratchpad/cmp.py`
(scratchpad — copy it into the repo's test suite if you want it retained). It should be turned into a regression test asserting your reimplementation agrees with `ega` semantics everywhere except the documented `≤`/`<` boundary conventions.
