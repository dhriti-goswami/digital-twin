"""Clarke and Parkes (Consensus) error grids.

Both legacy implementations were wrong in ways that inflated the reported
clinical-safety numbers:

* ``evaluate_ohio.py:317-342`` scored *any* pair with both values inside
  70-180 mg/dL as zone A -- so reference 75 with prediction 180, a 140% error,
  counted as clinically accurate. It relabelled Clarke's zone D as B, and its
  zone-E branch was a byte-identical duplicate of the zone-D branch, making
  **zone E unreachable**. Anything unmatched fell through to C.
* ``evaluate.py:186-207`` invented a ``reference >= 290`` branch that scored
  reference 400 with prediction 210 -- a 47% error -- as zone A.

Provenance of the boundaries used here
--------------------------------------
**Clarke.** Clarke et al. (1987) *Diabetes Care* 10(5):622-628 publishes the grid
as a *figure plus prose* and **never states the zone boundaries as inequalities**
-- the ambiguity was sufficient that Stoeckl et al. (2000) published a letter
clarifying the construction of the upper A-line alone. What exists instead are two
widely-used reference implementations: the MATLAB (Guevara Codina) -> Python
(Tsue) lineage used by most ML/CGM papers, and CRAN ``ega::getClarkeZones``.
Evaluating both over the integer lattice ``r, p in [1, 550]`` (302,500 points)
gives 12,029 disagreements which reduce to exactly one substantive difference plus
854 measure-zero open/closed boundary points:

* The MATLAB lineage caps the upper-C leg at ``r <= 290``. That cap is an artefact
  of the original figure's 0-400 mg/dL axes -- the line ``p = r + 110`` starts at
  the vertex ``(70, 180)`` and exits the top of a 400-limit plot at ``(290, 400)``.
  It is not a clinical boundary. **This implementation drops the cap**, following
  ``ega``, which is the defensible extrapolation for CGM data exceeding 400 mg/dL.

The evaluation order below (**A, E, C, D, B**) is the canonical order shared by
both reference implementations, and is deliberately kept even though it means a
point can be scored A on the +/-20% rule while also sitting on a dangerous-zone
edge. Reordering to put safety zones first would change the numbers and make them
incomparable to every published OhioT1DM result, which is the entire reason for
computing them.

The ``58.33`` constant sometimes quoted for the lower-D leg is not a primary
boundary: it is ``70 / 1.2``, the point where the upper A-line ``p = 1.2 r``
crosses ``p = 70``. Because A is tested first, writing lower-D as ``r <= 70`` is
provably equivalent to spelling out the ``58.33`` wedge.

**Parkes.** Parkes et al. (2000) *Diabetes Care* 23(8):1143-1148 defines the grid;
it too published no coordinates. Pfuetzner et al. (2013) *J Diabetes Sci Technol*
7(5):1275-1281 **first publishes** the exact coordinates (it is not an erratum).
The vertex lists here are Table 1 of that paper, cross-checked vertex-by-vertex
against ``ega::getParkesZones`` including its slope-extrapolated terminal points.

Figure/table consistency
------------------------
:func:`zone_field` shades a figure by evaluating *the same* classifier used for the
table on a dense mesh. The legacy plots drew decorative segments unrelated to the
counting logic and did not colour points by zone, so a figure could contradict its
own table indefinitely. Here it cannot.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twin.metrics.accuracy import MetricError

Array = NDArray[np.floating]

CLARKE_ZONES: tuple[str, ...] = ("A", "B", "C", "D", "E")
PARKES_ZONES: tuple[str, ...] = ("A", "B", "C", "D", "E")

#: Boundaries verified against primary sources; see the module docstring and
#: ``docs/CITATIONS_methods.md``.
VERIFICATION_STATUS: dict[str, bool] = {"clarke": True, "parkes": True}

CITATIONS: dict[str, str] = {
    "clarke": (
        "Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. Evaluating "
        "clinical accuracy of systems for self-monitoring of blood glucose. "
        "Diabetes Care 1987;10(5):622-628. doi:10.2337/diacare.10.5.622. "
        "Boundaries reconstructed from the figure via the two canonical reference "
        "implementations; upper-C cap at r<=290 dropped as a figure-axis artefact."
    ),
    "parkes": (
        "Parkes JL, Slatin SL, Pardo S, Ginsberg BH. A new consensus error grid to "
        "evaluate the clinical significance of inaccuracies in the measurement of "
        "blood glucose. Diabetes Care 2000;23(8):1143-1148. "
        "doi:10.2337/diacare.23.8.1143. Coordinates from Pfuetzner A, Klonoff DC, "
        "Pardo S, Parkes JL. Technical aspects of the Parkes error grid. "
        "J Diabetes Sci Technol 2013;7(5):1275-1281. doi:10.1177/193229681300700517."
    ),
}


class UnverifiedBoundaryError(RuntimeError):
    """Raised when reportable output is requested before boundaries are verified."""


def assert_verified(grid: str = "clarke") -> None:
    """Refuse to generate reportable output from unverified boundaries."""
    if not VERIFICATION_STATUS.get(grid, False):
        raise UnverifiedBoundaryError(
            f"The {grid} error-grid boundaries have not been verified against the "
            "primary source. Confirm every boundary line, record the citation in "
            f"docs/CITATIONS_methods.md, then set VERIFICATION_STATUS['{grid}'] = "
            "True. Refusing to produce a clinical-safety table from unverified "
            "boundaries."
        )


# --------------------------------------------------------------------------- #
# Clarke error grid
# --------------------------------------------------------------------------- #

#: Relative-agreement half-width defining zone A.
CLARKE_A_TOLERANCE = 0.20
#: Clinical action thresholds bounding the target range.
HYPO_THRESHOLD = 70.0
HYPER_THRESHOLD = 180.0
#: Reference above which failure to detect hyperglycaemia is zone D. Both
#: reference implementations use 240, consistent with the figure vertices
#: (240, 70) and (240, 180). At least one peer-reviewed reimplementation
#: paraphrases this as 180, which would make D overlap B massively -- do not.
CLARKE_D_HYPER_REFERENCE = 240.0
#: Zone C overcorrection offset. 110, not 100: forced by the vertex (70, 180),
#: since 70 + 110 = 180 exactly.
CLARKE_C_OFFSET = 110.0
#: Lower-C leg: p <= 1.4 r - 182 on 130 <= r <= 180. Identical to ega's
#: 1.4 (r - 130) since 1.4 * 130 = 182. Triangle (130,0), (180,0), (180,70).
CLARKE_C_LOWER_SLOPE = 1.4
CLARKE_C_LOWER_INTERCEPT = -182.0

#: Vertex list ``ega`` uses to draw the Clarke grid, for reference. Segments with
#: data-dependent endpoints are expressed against ``max_glucose``.
CLARKE_DRAW_SEGMENTS: tuple[tuple[str, tuple[float, float], tuple[float, float]], ...] = (
    ("upper A-line", (58.3, 70.0), (float("nan"), float("nan"))),
    ("lower A-line", (70.0, 56.0), (float("nan"), float("nan"))),
    ("upper C line p=r+110", (70.0, 180.0), (550.0, 660.0)),
    ("lower C line p=1.4r-182", (130.0, 0.0), (180.0, 70.0)),
)


def clarke_zone(reference: Array, predicted: Array) -> NDArray[np.str_]:
    """Classify (reference, predicted) pairs into Clarke zones A-E.

    Evaluation order is **A, E, C, D, B**, first match wins -- the canonical order
    of both reference implementations. See the module docstring for why this order
    is kept rather than reordered to prioritise the dangerous zones.

    Boundaries
    ----------
    ``A`` clinically accurate
        ``0.8 r <= p <= 1.2 r``, or both below 70 mg/dL
    ``E`` erroneous treatment
        ``r >= 180 and p <= 70``, or ``r <= 70 and p >= 180``
    ``C`` unnecessary overcorrection
        ``r >= 70 and p >= r + 110`` (no upper cap on ``r``), or
        ``130 <= r <= 180 and p <= 1.4 r - 182``
    ``D`` dangerous failure to detect
        ``r >= 240 and 70 <= p <= 180``, or ``r <= 70 and 70 <= p <= 180``
    ``B``
        everything else -- benign deviation
    """
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if ref.shape != pred.shape:
        raise MetricError(f"shape mismatch: {ref.shape} vs {pred.shape}")
    if ref.size and (np.any(ref <= 0) or np.any(pred < 0)):
        raise MetricError("error grids require positive reference and non-negative prediction")

    zones = np.full(ref.shape, "B", dtype="<U1")
    assigned = np.zeros(ref.shape, dtype=bool)

    def claim(mask: NDArray[np.bool_], label: str) -> None:
        nonlocal assigned
        target = mask & ~assigned
        zones[target] = label
        assigned |= target

    # A -- clinically accurate. Tested first, per both reference implementations.
    within_tolerance = (pred >= (1.0 - CLARKE_A_TOLERANCE) * ref) & (
        pred <= (1.0 + CLARKE_A_TOLERANCE) * ref
    )
    both_low = (ref < HYPO_THRESHOLD) & (pred < HYPO_THRESHOLD)
    claim(within_tolerance | both_low, "A")

    # E -- erroneous treatment: the prediction prompts the opposite action.
    claim(
        ((ref >= HYPER_THRESHOLD) & (pred <= HYPO_THRESHOLD))
        | ((ref <= HYPO_THRESHOLD) & (pred >= HYPER_THRESHOLD)),
        "E",
    )

    # C -- unnecessary overcorrection of a value needing no treatment.
    upper_c = (ref >= HYPO_THRESHOLD) & (pred >= ref + CLARKE_C_OFFSET)
    lower_c = (
        (ref >= 130.0)
        & (ref <= HYPER_THRESHOLD)
        & (pred <= CLARKE_C_LOWER_SLOPE * ref + CLARKE_C_LOWER_INTERCEPT)
    )
    claim(upper_c | lower_c, "C")

    # D -- failure to detect: a true excursion predicted as in-range.
    in_range_prediction = (pred >= HYPO_THRESHOLD) & (pred <= HYPER_THRESHOLD)
    claim(
        ((ref >= CLARKE_D_HYPER_REFERENCE) & in_range_prediction)
        | ((ref <= HYPO_THRESHOLD) & in_range_prediction),
        "D",
    )

    return zones


# --------------------------------------------------------------------------- #
# Parkes / Consensus error grid
# --------------------------------------------------------------------------- #

#: Parkes error grid, TYPE 1 DIABETES. Pfuetzner et al. 2013, Table 1.
#: Each list is an open polyline of ``(reference, test)`` vertices in mg/dL.
#: Zone A has no row of its own: it is the region between ``B_lower`` and
#: ``B_upper``. Type 1 has **no lower E zone**.
PARKES_T1: dict[str, list[tuple[float, float]]] = {
    "B_lower": [(50, 0), (50, 30), (170, 145), (385, 300), (550, 450)],
    "B_upper": [(0, 50), (30, 50), (140, 170), (280, 380), (430, 550)],
    "C_lower": [(120, 0), (120, 30), (260, 130), (550, 250)],
    "C_upper": [(0, 60), (30, 60), (50, 80), (70, 110), (260, 550)],
    "D_lower": [(250, 0), (250, 40), (550, 150)],
    "D_upper": [(0, 100), (25, 100), (50, 125), (80, 215), (125, 550)],
    "E_upper": [(0, 150), (35, 155), (50, 550)],
}

#: Type 2 grid, included for completeness. The 2013 authors note it "has fallen
#: out of favor"; not used for any claim in this project.
PARKES_T2: dict[str, list[tuple[float, float]]] = {
    "B_lower": [(50, 0), (50, 30), (90, 80), (330, 230), (550, 450)],
    "B_upper": [(0, 50), (30, 50), (230, 330), (440, 550)],
    "C_lower": [(90, 0), (260, 130), (550, 250)],
    "C_upper": [(0, 60), (30, 60), (280, 550)],
    "D_lower": [(250, 0), (250, 40), (410, 110), (550, 160)],
    "D_upper": [(0, 80), (25, 80), (35, 90), (125, 550)],
    "E_upper": [(0, 200), (35, 200), (50, 550)],
}


def _polyline(x: Array, vertices: list[tuple[float, float]], *, extrapolate_high: bool) -> Array:
    """Piecewise-linear evaluation of a boundary polyline.

    Below the first vertex the value is clamped, which is correct for both
    boundary families: upper boundaries begin with a flat segment, and lower
    boundaries clamp to their minimum so no point can fall below them outside
    their support.

    Above the last vertex the terminal-segment slope is extrapolated rather than
    clamped, matching ``ega``'s treatment. Clamping would place the boundary at
    550 mg/dL and misclassify extreme points.
    """
    xs = np.array([vertex[0] for vertex in vertices], dtype=np.float64)
    ys = np.array([vertex[1] for vertex in vertices], dtype=np.float64)
    values = np.interp(x, xs, ys)

    if extrapolate_high:
        # Use the last pair of distinct x values, so a terminal vertical segment
        # does not produce a division by zero.
        distinct = np.flatnonzero(np.diff(xs) > 0)
        if distinct.size:
            index = int(distinct[-1])
            slope = (ys[index + 1] - ys[index]) / (xs[index + 1] - xs[index])
            values = np.where(x > xs[-1], ys[-1] + slope * (x - xs[-1]), values)
    return values


def parkes_zone(
    reference: Array, predicted: Array, *, diabetes_type: int = 1
) -> NDArray[np.str_]:
    """Classify pairs into Parkes (Consensus) error-grid zones.

    Points start in zone A and are overwritten by successively more severe bands,
    so each point ends in the outermost band containing it -- the same procedure
    ``ega`` uses. Type 1 has no lower E zone, so an extreme under-prediction
    saturates at D on that side; that is a property of the published grid, not an
    omission here.

    ISO 15197:2013 uses the Type 1 grid, and the regulatory reading is that only
    zone A is acceptable -- worth stating when reporting these numbers.
    """
    assert_verified("parkes")
    grid = {1: PARKES_T1, 2: PARKES_T2}.get(diabetes_type)
    if grid is None:
        raise MetricError(f"diabetes_type must be 1 or 2, got {diabetes_type}")

    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if ref.shape != pred.shape:
        raise MetricError(f"shape mismatch: {ref.shape} vs {pred.shape}")
    if ref.size and (np.any(ref < 0) or np.any(pred < 0)):
        raise MetricError("error grids require non-negative glucose values")

    zones = np.full(ref.shape, "A", dtype="<U1")

    # Severity increases down each list; later assignments win.
    for label in ("B", "C", "D", "E"):
        upper_key = f"{label}_upper"
        if upper_key in grid:
            boundary = _polyline(ref, grid[upper_key], extrapolate_high=True)
            zones[pred >= boundary] = label
        lower_key = f"{label}_lower"
        if lower_key in grid:
            boundary = _polyline(ref, grid[lower_key], extrapolate_high=True)
            zones[pred <= boundary] = label

    return zones


# --------------------------------------------------------------------------- #
# Summaries and plotting support
# --------------------------------------------------------------------------- #

_CLASSIFIERS = {"clarke": clarke_zone, "parkes": parkes_zone}


@dataclass(frozen=True)
class ZoneSummary:
    """Percentage of points in each zone, plus the aggregate readings."""

    grid: str
    counts: dict[str, int]
    percentages: dict[str, float]
    n: int

    @property
    def clinically_acceptable(self) -> float:
        """A + B: the conventional 'clinically acceptable' aggregate [%]."""
        return self.percentages.get("A", 0.0) + self.percentages.get("B", 0.0)

    @property
    def dangerous(self) -> float:
        """D + E: failure to detect plus erroneous treatment [%]."""
        return self.percentages.get("D", 0.0) + self.percentages.get("E", 0.0)

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {
            f"{self.grid}_zone_{zone}_pct": value for zone, value in self.percentages.items()
        }
        out.update(
            {f"{self.grid}_zone_{zone}_n": float(count) for zone, count in self.counts.items()}
        )
        out[f"{self.grid}_clinically_acceptable_pct"] = self.clinically_acceptable
        out[f"{self.grid}_dangerous_pct"] = self.dangerous
        out[f"{self.grid}_n"] = float(self.n)
        return out


def zone_summary(reference: Array, predicted: Array, *, grid: str = "clarke") -> ZoneSummary:
    """Zone occupancy for a set of predictions.

    Every zone appears with an explicit zero when empty. The legacy metrics writer
    emitted only A-D columns, so zone E could not have been reported even had it
    been reachable -- and an absent column reads as "no dangerous predictions" when
    it actually means "never measured".
    """
    if grid not in _CLASSIFIERS:
        raise MetricError(f"unknown grid {grid!r}; expected one of {sorted(_CLASSIFIERS)}")
    labels = _CLASSIFIERS[grid](reference, predicted)
    flat = labels.ravel()
    tally = Counter(flat.tolist())
    zones = CLARKE_ZONES if grid == "clarke" else PARKES_ZONES

    total = int(flat.size)
    counts = {zone: int(tally.get(zone, 0)) for zone in zones}
    if sum(counts.values()) != total:
        unexpected = set(tally) - set(zones)
        raise MetricError(f"unexpected zone labels produced: {sorted(unexpected)}")

    return ZoneSummary(
        grid=grid,
        counts=counts,
        percentages={zone: 100.0 * count / total for zone, count in counts.items()},
        n=total,
    )


def zone_field(
    *,
    grid: str = "clarke",
    glucose_max: float = 400.0,
    resolution: int = 400,
) -> tuple[Array, Array, NDArray[np.int_], tuple[str, ...]]:
    """Dense zone map for shading an error-grid figure.

    Evaluates the *same* classifier used for the table on a regular mesh, so the
    shaded regions in a figure are by construction the regions being counted.
    This makes a figure/table contradiction impossible rather than merely
    unlikely.

    Returns ``(reference_axis, predicted_axis, zone_indices, zone_order)`` where
    ``zone_indices`` indexes into ``zone_order``, suitable for ``pcolormesh``.
    """
    if grid not in _CLASSIFIERS:
        raise MetricError(f"unknown grid {grid!r}")
    zones = CLARKE_ZONES if grid == "clarke" else PARKES_ZONES

    # Start slightly above zero: Clarke's relative-agreement rule is undefined at
    # zero reference glucose.
    axis = np.linspace(1.0, glucose_max, resolution)
    reference_mesh, predicted_mesh = np.meshgrid(axis, axis, indexing="ij")
    labels = _CLASSIFIERS[grid](reference_mesh, predicted_mesh)

    lookup = {zone: index for index, zone in enumerate(zones)}
    indices = np.vectorize(lookup.__getitem__, otypes=[np.int_])(labels)
    return axis, axis, indices, zones


__all__ = [
    "CITATIONS",
    "CLARKE_A_TOLERANCE",
    "CLARKE_C_LOWER_INTERCEPT",
    "CLARKE_C_LOWER_SLOPE",
    "CLARKE_C_OFFSET",
    "CLARKE_D_HYPER_REFERENCE",
    "CLARKE_DRAW_SEGMENTS",
    "CLARKE_ZONES",
    "HYPER_THRESHOLD",
    "HYPO_THRESHOLD",
    "PARKES_T1",
    "PARKES_T2",
    "PARKES_ZONES",
    "VERIFICATION_STATUS",
    "UnverifiedBoundaryError",
    "ZoneSummary",
    "assert_verified",
    "clarke_zone",
    "parkes_zone",
    "zone_field",
    "zone_summary",
]
