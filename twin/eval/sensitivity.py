"""Validation of the estimated insulin sensitivity.

An estimated `S_I` is only a contribution if it can be shown to measure something.
There is no clamp study here, so three orthogonal checks are used, all computable
from OhioT1DM's own records:

1. **External correlation.** `S_I` should be inversely related to how much insulin
   the subject actually needs. Insulin requirement is observable from the pump
   record, so this tests the estimate against a quantity it never saw.
2. **Test-retest stability.** `S_I` estimated on disjoint time windows for the same
   subject should agree. Physiology drifts over days, not minutes; an estimate that
   varies window to window is absorbing prediction error, not measuring a parameter.
3. **Physiological plausibility and spread.** The estimates must lie inside the
   published IDDM range and must actually vary between subjects. A parameter head
   that collapses to the population mean for everyone would pass checks 1 and 2
   trivially, so degeneracy is tested for explicitly.

All three are reported whether or not they support the claim. A failure here is a
real finding about amortised parameter estimation, and is far more useful than a
number presented without evidence that it means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from twin.data.dataset import SubjectData
from twin.physio.params import BOUNDS, S_I_IDDM_MEAN, S_I_IDDM_SD

Array = NDArray[np.floating]


# --------------------------------------------------------------------------- #
# Observable therapy summaries
# --------------------------------------------------------------------------- #


@dataclass
class TherapySummary:
    """What the pump record says about a subject's insulin requirement."""

    subject_id: str
    days: float
    total_daily_dose_u: float
    basal_fraction: float
    total_daily_carbs_g: float
    #: Empirical carbohydrate ratio [g/U] from the pump's own bolus wizard entries.
    carb_ratio_g_per_u: float | None
    n_wizard_boluses: int

    @property
    def tdd_per_kg(self) -> float:
        """Total daily dose per kilogram, at the nominal weight.

        Body weight is not identifiable from OhioT1DM (every file reports the
        placeholder 99 kg), so this is TDD divided by the same nominal weight for
        every subject. It is therefore a *rescaling* of TDD, not an independent
        quantity -- stated so it is not over-read.
        """
        from twin.data.ohio import NOMINAL_WEIGHT_KG

        return self.total_daily_dose_u / NOMINAL_WEIGHT_KG

    def as_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "days": round(self.days, 2),
            "total_daily_dose_u": round(self.total_daily_dose_u, 2),
            "tdd_per_kg": round(self.tdd_per_kg, 4),
            "basal_fraction": round(self.basal_fraction, 3),
            "total_daily_carbs_g": round(self.total_daily_carbs_g, 1),
            "carb_ratio_g_per_u": (
                round(self.carb_ratio_g_per_u, 2) if self.carb_ratio_g_per_u else None
            ),
            "n_wizard_boluses": self.n_wizard_boluses,
        }


def therapy_summary(data: SubjectData) -> TherapySummary:
    """Summarise a subject's insulin and carbohydrate requirement from the record."""
    frame = data.frame
    grid = data.subject.grid_minutes
    span_minutes = float(len(frame) * grid)
    days = span_minutes / 1440.0

    basal_u = float(frame["basal_u_per_min"].sum() * grid)
    bolus_u = float(frame["bolus_u_per_min"].sum() * grid)
    carbs_g = float(frame["carbs_mg_per_min"].sum() * grid / 1000.0)
    total = basal_u + bolus_u

    # The pump's bolus wizard records the carbohydrate the user entered alongside
    # the dose, which gives an empirical carb ratio independent of the CGM trace.
    wizard = data.subject_wizard_boluses() if hasattr(data, "subject_wizard_boluses") else None
    carb_ratio: float | None = None
    n_wizard = 0
    if wizard is not None and len(wizard):
        usable = wizard[(wizard["dose"] > 0) & (wizard["bwz_carb_input"] > 0)]
        n_wizard = int(len(usable))
        if n_wizard >= 5:
            carb_ratio = float(np.median(usable["bwz_carb_input"] / usable["dose"]))

    return TherapySummary(
        subject_id=data.subject_id,
        days=days,
        total_daily_dose_u=total / days if days > 0 else float("nan"),
        basal_fraction=basal_u / total if total > 0 else float("nan"),
        total_daily_carbs_g=carbs_g / days if days > 0 else float("nan"),
        carb_ratio_g_per_u=carb_ratio,
        n_wizard_boluses=n_wizard,
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


@dataclass
class Correlation:
    """A rank correlation with a bootstrap interval."""

    label: str
    rho: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int
    underpowered: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "comparison": self.label,
            "spearman_rho": round(self.rho, 4),
            "p_value": round(self.p_value, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "n_subjects": self.n,
            "underpowered": self.underpowered,
        }

    def __str__(self) -> str:
        caveat = "; UNDERPOWERED" if self.underpowered else ""
        return (
            f"{self.label}: rho={self.rho:+.3f} "
            f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"p={self.p_value:.4f}, n={self.n}{caveat}"
        )


def spearman_with_ci(
    x: Array, y: Array, *, label: str = "", n_resamples: int = 10_000, seed: int = 42
) -> Correlation:
    """Spearman correlation with a percentile bootstrap CI over subjects.

    With 12 subjects the point estimate is very uncertain, so the interval is the
    honest summary and is reported alongside. A correlation from 12 points should
    never be presented as a bare number.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return Correlation(label, np.nan, np.nan, np.nan, np.nan, int(x.size), True)

    result = stats.spearmanr(x, y)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, x.size, size=(n_resamples, x.size))
    samples = np.array(
        [
            stats.spearmanr(x[index], y[index]).statistic
            if np.ptp(x[index]) > 0 and np.ptp(y[index]) > 0
            else np.nan
            for index in draws
        ]
    )
    samples = samples[np.isfinite(samples)]
    low, high = (
        (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))
        if samples.size
        else (np.nan, np.nan)
    )
    return Correlation(
        label=label,
        rho=float(result.statistic),
        p_value=float(result.pvalue),
        ci_low=low,
        ci_high=high,
        n=int(x.size),
        underpowered=bool(x.size < 20),
    )


def intraclass_correlation(blocks: list[Array]) -> dict[str, float]:
    """ICC(1) from a one-way random-effects model.

    ``blocks[i]`` holds repeated estimates for subject ``i``.

    .. math::

        \\mathrm{ICC}(1) = \\frac{MS_b - MS_w}{MS_b + (k-1) MS_w}

    Interpreted as the fraction of total variance attributable to genuine
    between-subject differences. Near zero means the estimate is noise around a
    common value; near one means it is a stable subject-level property. This is the
    check that distinguishes a measured parameter from a per-window nuisance term.
    """
    usable = [np.asarray(block, dtype=np.float64) for block in blocks]
    usable = [block[np.isfinite(block)] for block in usable]
    usable = [block for block in usable if block.size >= 2]
    if len(usable) < 2:
        return {"icc": float("nan"), "n_subjects": len(usable), "n_per_subject": 0.0}

    counts = np.array([block.size for block in usable])
    k = float(counts.mean())
    n = len(usable)
    grand_mean = float(np.concatenate(usable).mean())
    subject_means = np.array([block.mean() for block in usable])

    ss_between = float((counts * (subject_means - grand_mean) ** 2).sum())
    ss_within = float(sum(((block - block.mean()) ** 2).sum() for block in usable))
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / max(counts.sum() - n, 1)

    denominator = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator > 0 else float("nan")
    return {
        "icc": float(icc),
        "n_subjects": n,
        "n_per_subject": k,
        "ms_between": ms_between,
        "ms_within": ms_within,
    }


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


@dataclass
class SensitivityReport:
    """Everything needed to judge whether the estimated ``S_I`` means anything."""

    per_subject: pd.DataFrame
    correlations: list[Correlation]
    stability: dict[str, float]
    degeneracy: dict[str, float]
    in_range_fraction: float
    notes: list[str] = field(default_factory=list)

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame([correlation.as_dict() for correlation in self.correlations])

    def verdict(self) -> str:
        """A plain statement of what the checks support, without overclaiming."""
        lines = []
        for correlation in self.correlations:
            lines.append(f"  {correlation}")
        icc = self.stability.get("icc", float("nan"))
        lines.append(f"  test-retest ICC(1) = {icc:.3f}")
        lines.append(
            f"  between-subject spread: CV = {self.degeneracy.get('cv', float('nan')):.1%}"
            f" (range {self.degeneracy.get('min', float('nan')):.3e}"
            f" to {self.degeneracy.get('max', float('nan')):.3e})"
        )
        lines.append(f"  within published IDDM range: {self.in_range_fraction:.0%} of subjects")
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


def validate_sensitivity(
    sensitivity_by_subject: dict[str, Array],
    corpus: dict[str, dict[str, SubjectData]],
    *,
    n_blocks: int = 4,
    seed: int = 42,
) -> SensitivityReport:
    """Run all three validation checks.

    ``sensitivity_by_subject`` maps subject id to the per-window ``S_I`` estimates,
    in time order, for that subject's evaluation windows.
    """
    rows: list[dict[str, object]] = []
    blocks: list[Array] = []
    notes: list[str] = []

    for subject_id, estimates in sorted(sensitivity_by_subject.items()):
        values = np.asarray(estimates, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        data = corpus["test"].get(subject_id) or corpus["train"].get(subject_id)
        therapy = therapy_summary(data) if data is not None else None

        # Time-ordered blocks for the test-retest check: disjoint stretches of the
        # subject's own record, which is the closest available analogue of repeating
        # a measurement on the same person.
        split_points = np.array_split(values, min(n_blocks, max(values.size // 2, 1)))
        block_medians = np.array([np.median(part) for part in split_points if part.size])
        blocks.append(block_medians)

        row: dict[str, object] = {
            "subject_id": subject_id,
            "n_windows": int(values.size),
            "s_i_median": float(np.median(values)),
            "s_i_iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
            "s_i_min": float(values.min()),
            "s_i_max": float(values.max()),
            "n_blocks": int(block_medians.size),
            "block_spread": float(np.ptp(block_medians)) if block_medians.size > 1 else 0.0,
        }
        if therapy is not None:
            row.update(therapy.as_dict())
        rows.append(row)

    per_subject = pd.DataFrame(rows)
    if per_subject.empty:
        raise ValueError("no sensitivity estimates supplied")

    correlations: list[Correlation] = []
    if "total_daily_dose_u" in per_subject:
        correlations.append(
            spearman_with_ci(
                per_subject["s_i_median"].to_numpy(),
                per_subject["total_daily_dose_u"].to_numpy(),
                label="S_I vs total daily dose (expected negative)",
                seed=seed,
            )
        )
    if "carb_ratio_g_per_u" in per_subject and per_subject["carb_ratio_g_per_u"].notna().sum() >= 3:
        correlations.append(
            spearman_with_ci(
                per_subject["s_i_median"].to_numpy(),
                per_subject["carb_ratio_g_per_u"].to_numpy(),
                label="S_I vs empirical carb ratio (expected positive)",
                seed=seed,
            )
        )
        notes.append(
            "the carbohydrate-ratio correlation is EXPLORATORY: bolus-wizard entries "
            "exist only for the 2018 cohort (n=6), and Spearman on six points needs "
            "|rho| > 0.83 to reach p < 0.05"
        )
    else:
        notes.append(
            "empirical carb ratio unavailable for enough subjects; "
            "the bolus-wizard correlation is not reported"
        )

    stability = intraclass_correlation(blocks)

    medians = per_subject["s_i_median"].to_numpy()
    degeneracy = {
        "cv": float(medians.std(ddof=1) / medians.mean()) if medians.size > 1 else 0.0,
        "min": float(medians.min()),
        "max": float(medians.max()),
        "spread_ratio": float(medians.max() / medians.min()) if medians.min() > 0 else float("nan"),
    }
    if degeneracy["cv"] < 0.02:
        notes.append(
            "between-subject CV below 2%: the parameter head has effectively "
            "collapsed to a single value, so S_I is NOT subject-specific and must "
            "not be reported as such"
        )

    # Published IDDM range, Ward et al. 1991: 2.5 +/- 0.6e-4. Two SD is used as a
    # generous plausibility band rather than a hard criterion.
    low = S_I_IDDM_MEAN - 2 * S_I_IDDM_SD
    high = S_I_IDDM_MEAN + 2 * S_I_IDDM_SD
    in_range = float(np.mean((medians >= low) & (medians <= high)))

    bound = BOUNDS["p3"]
    notes.append(
        f"S_I is bounded by construction via p3 in [{bound.low:.1e}, {bound.high:.1e}] "
        "and p2, so lying in a plausible range is partly enforced rather than learned"
    )

    return SensitivityReport(
        per_subject=per_subject,
        correlations=correlations,
        stability=stability,
        degeneracy=degeneracy,
        in_range_fraction=in_range,
        notes=notes,
    )


__all__ = [
    "Correlation",
    "SensitivityReport",
    "TherapySummary",
    "intraclass_correlation",
    "spearman_with_ci",
    "therapy_summary",
    "validate_sensitivity",
]
