"""Inferential statistics over subjects.

The unit of analysis is the **subject**, not the window. Windows overlap by 23 of
24 timesteps and are strongly autocorrelated, so treating them as independent
samples inflates any test statistic enormously -- a per-window test on ~10,000
correlated windows will call a 0.2 mg/dL difference significant. Every function
here therefore takes one value per subject and works with ``n`` in the low tens.

That small ``n`` is a real limitation, not a formality: with 12 subjects a paired
Wilcoxon test has limited power and cannot detect small effects. The honest
response is to report exact p-values, effect sizes, and confidence intervals
side by side, and to say plainly when a comparison is underpowered --
:func:`describe_comparison` does that rather than reducing everything to a
significance star.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.floating]

#: Below this many subjects a paired test is reported but flagged underpowered.
MIN_SUBJECTS_FOR_INFERENCE = 6


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap interval over subjects."""

    point: float
    low: float
    high: float
    level: float
    n_subjects: int
    n_resamples: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.point:.2f} [{self.low:.2f}, {self.high:.2f}]"


def bootstrap_ci(
    per_subject: Array,
    *,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
    statistic: str = "mean",
) -> BootstrapCI:
    """Percentile bootstrap CI by resampling **subjects** with replacement.

    Resampling subjects (not windows) is what makes the interval reflect
    between-subject generalisation, which is the quantity a reader cares about.
    Resampling windows would produce a hopelessly narrow interval that describes
    only sampling within this particular cohort.
    """
    values = np.asarray(per_subject, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite per-subject values")

    reduce = {"mean": np.mean, "median": np.median}[statistic]
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(n_resamples, values.size))
    distribution = reduce(values[indices], axis=1)

    tail = (1.0 - level) / 2.0
    return BootstrapCI(
        point=float(reduce(values)),
        low=float(np.quantile(distribution, tail)),
        high=float(np.quantile(distribution, 1.0 - tail)),
        level=level,
        n_subjects=int(values.size),
        n_resamples=n_resamples,
    )


@dataclass(frozen=True)
class PairedComparison:
    """One paired comparison between two methods across subjects."""

    label: str
    n_subjects: int
    mean_difference: float
    median_difference: float
    #: Wilcoxon signed-rank statistic and its exact two-sided p-value.
    statistic: float
    p_value: float
    #: Matched-pairs rank-biserial correlation: effect size in [-1, 1].
    effect_size: float
    #: Number of subjects for which the first method was better (lower).
    n_favouring_first: int
    difference_ci: BootstrapCI
    underpowered: bool

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["difference_ci"] = self.difference_ci.as_dict()
        return out


def paired_comparison(
    first: Array,
    second: Array,
    *,
    label: str = "",
    lower_is_better: bool = True,
    seed: int = 42,
) -> PairedComparison:
    """Paired Wilcoxon signed-rank test between two methods, across subjects.

    ``first`` and ``second`` hold one value per subject, in the same subject
    order. For an error metric (``lower_is_better``) a negative
    ``mean_difference`` means ``first`` wins.

    The exact null distribution is used rather than the normal approximation,
    which matters at ``n = 12``.
    """
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align: {a.shape} vs {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size == 0:
        raise ValueError("no paired finite observations")

    difference = a - b
    non_zero = difference[difference != 0]
    if non_zero.size == 0:
        statistic, p_value, effect = 0.0, 1.0, 0.0
    else:
        # ``exact`` is only available while there are no zero differences and n is
        # small; scipy picks the exact method automatically for n <= 25 when zeros
        # are dropped, which is the default ``zero_method="wilcox"``.
        result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
        ranks = stats.rankdata(np.abs(non_zero))
        positive = float(ranks[non_zero > 0].sum())
        negative = float(ranks[non_zero < 0].sum())
        total = positive + negative
        effect = (positive - negative) / total if total else 0.0

    better = int(np.count_nonzero(difference < 0 if lower_is_better else difference > 0))

    return PairedComparison(
        label=label,
        n_subjects=int(a.size),
        mean_difference=float(difference.mean()),
        median_difference=float(np.median(difference)),
        statistic=statistic,
        p_value=p_value,
        effect_size=float(effect),
        n_favouring_first=better,
        difference_ci=bootstrap_ci(difference, seed=seed),
        underpowered=bool(a.size < MIN_SUBJECTS_FOR_INFERENCE),
    )


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, object]]:
    """Holm-Bonferroni step-down correction across a family of comparisons.

    Every results table compares several models at four horizons, so the family is
    large and uncorrected p-values would be misleading. Holm is used rather than
    plain Bonferroni because it is uniformly more powerful while controlling the
    same family-wise error rate, and rather than Benjamini-Hochberg because
    family-wise control is the appropriate standard when each comparison is
    reported as an individual claim.

    Returns, per key, the raw and adjusted p-value and the reject decision.
    """
    if not p_values:
        return {}

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)

    adjusted: list[float] = []
    running_max = 0.0
    for index, (_, raw) in enumerate(ordered):
        candidate = (total - index) * raw
        running_max = max(running_max, candidate)
        adjusted.append(min(1.0, running_max))

    return {
        key: {
            "p_raw": float(raw),
            "p_adjusted": float(adjusted_value),
            "reject": bool(adjusted_value < alpha),
            "rank": rank + 1,
        }
        for rank, ((key, raw), adjusted_value) in enumerate(zip(ordered, adjusted, strict=True))
    }


def describe_comparison(comparison: PairedComparison, *, adjusted_p: float | None = None) -> str:
    """A one-line, non-overclaiming summary of a paired comparison.

    Deliberately reports the direction and magnitude first and the p-value last,
    and states explicitly when the comparison is underpowered, so the sentence
    cannot be read as stronger evidence than it is.
    """
    p_text = f"p={comparison.p_value:.4f}"
    if adjusted_p is not None:
        p_text += f" (Holm-adjusted {adjusted_p:.4f})"
    direction = "lower" if comparison.mean_difference < 0 else "higher"
    caveat = "; UNDERPOWERED" if comparison.underpowered else ""
    return (
        f"{comparison.label}: {abs(comparison.mean_difference):.2f} {direction} "
        f"on average, better in {comparison.n_favouring_first}/{comparison.n_subjects} "
        f"subjects, difference CI {comparison.difference_ci}, "
        f"effect size {comparison.effect_size:+.2f}, {p_text}{caveat}"
    )


__all__ = [
    "MIN_SUBJECTS_FOR_INFERENCE",
    "BootstrapCI",
    "PairedComparison",
    "bootstrap_ci",
    "describe_comparison",
    "holm_bonferroni",
    "paired_comparison",
]
