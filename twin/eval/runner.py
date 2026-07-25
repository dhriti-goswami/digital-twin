"""Evaluation: predictions in, reported tables out.

Every table this project publishes comes from here, so the reporting rules are
enforced in one place rather than restated per script:

* metrics per subject, then **mean ± SD across subjects** as the headline;
* pooled metrics computed too, and reported as clearly-labelled secondary;
* the persistence baseline attached to every result, with a skill score;
* nothing hand-typed -- the legacy ``train_ohio.py`` baked prior results in as
  literals (``sim_r = (10.94, 4.99, 99.4)``) which silently went stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from twin.config import Config
from twin.data.dataset import SubjectData
from twin.data.splits import Fold
from twin.metrics import (
    SubjectPredictions,
    across_subject_summary,
    holm_bonferroni,
    paired_comparison,
    per_subject_table,
    pooled_metrics,
)
from twin.metrics.stats import PairedComparison, describe_comparison

Array = NDArray[np.floating]


@dataclass
class EvaluationResult:
    """One method evaluated on one fold part."""

    method: str
    protocol: str
    part: str
    horizons_min: tuple[int, ...]
    per_subject: pd.DataFrame
    summary: pd.DataFrame
    pooled: pd.DataFrame
    predictions: dict[str, Array] = field(default_factory=dict)
    targets: dict[str, Array] = field(default_factory=dict)

    def metric_by_subject(self, metric: str, horizon_min: int) -> pd.Series:
        """One value per subject, indexed by subject id.

        This is the vector every statistical comparison operates on: subjects are
        the independent units, not windows.
        """
        rows = self.per_subject[self.per_subject["horizon_min"] == horizon_min]
        return rows.set_index("subject_id")[metric].sort_index()

    def headline(self, metric: str = "mae") -> pd.DataFrame:
        columns = ["horizon_min", "n_subjects", "n_windows_total"]
        columns += [f"{metric}_{suffix}" for suffix in ("mean", "sd", "median", "min", "max")]
        available = [column for column in columns if column in self.summary.columns]
        return self.summary[available]


def evaluate_predictions(
    *,
    method: str,
    fold: Fold,
    part: str,
    corpus: dict[str, dict[str, SubjectData]],
    predictions: dict[str, Array],
    config: Config,
    include_error_grid: bool = True,
) -> EvaluationResult:
    """Score a method's per-subject predictions.

    ``predictions`` maps subject id to an ``(n_windows, n_horizons)`` array aligned
    with that subject's selection in ``fold``. Alignment is checked rather than
    assumed: a silent misalignment would produce a plausible, wrong table.
    """
    source_split = "test" if part == "test" else "train"
    horizons = config.data.horizons_min

    subjects: list[SubjectPredictions] = []
    targets: dict[str, Array] = {}
    for selection in getattr(fold, part):
        subject_id = selection.subject_id
        if selection.indices.size == 0:
            continue
        if subject_id not in predictions:
            raise ValueError(f"{method}: no predictions for subject {subject_id}")

        data = corpus[source_split][subject_id]
        expected = data.windows.targets[selection.indices]
        given = np.asarray(predictions[subject_id], dtype=np.float64)
        if given.shape != expected.shape:
            raise ValueError(
                f"{method}/{subject_id}: predictions have shape {given.shape}, "
                f"expected {expected.shape}"
            )
        targets[subject_id] = expected
        subjects.append(
            SubjectPredictions(
                subject_id=subject_id,
                y_true=expected,
                y_pred=given,
                cohort=data.subject.cohort,
            )
        )

    if not subjects:
        raise ValueError(f"{method}: no subjects with windows in fold part {part!r}")

    table = per_subject_table(subjects, horizons, include_error_grid=include_error_grid)
    return EvaluationResult(
        method=method,
        protocol=fold.protocol,
        part=part,
        horizons_min=horizons,
        per_subject=table,
        summary=across_subject_summary(table, seed=config.run.seed),
        pooled=pooled_metrics(subjects, horizons),
        predictions={key: np.asarray(value) for key, value in predictions.items()},
        targets=targets,
    )


def evaluate_baseline(
    name: str,
    *,
    fold: Fold,
    part: str,
    corpus: dict[str, dict[str, SubjectData]],
    config: Config,
    include_error_grid: bool = True,
    **kwargs: object,
) -> EvaluationResult:
    """Run and score one non-learned baseline."""
    from twin.models.baselines import run_baseline

    source_split = "test" if part == "test" else "train"
    predictions: dict[str, Array] = {}
    for selection in getattr(fold, part):
        if selection.indices.size == 0:
            continue
        data = corpus[source_split][selection.subject_id]
        extra = dict(kwargs)
        if name == "arima":
            # Fit on the subject's training period so the baseline never estimates
            # parameters on the evaluation window.
            train_data = corpus["train"].get(selection.subject_id)
            if train_data is not None:
                extra["train_glucose"] = train_data.frame["glucose_filled"].to_numpy(
                    dtype=np.float64
                )
        result = run_baseline(name, data, selection.indices, **extra)
        predictions[selection.subject_id] = result.predictions

    return evaluate_predictions(
        method=name,
        fold=fold,
        part=part,
        corpus=corpus,
        predictions=predictions,
        config=config,
        include_error_grid=include_error_grid,
    )


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def skill_score(result: EvaluationResult, reference: EvaluationResult, *, metric: str = "rmse") -> pd.DataFrame:
    """Fractional improvement over a reference method, per horizon.

    ``1 - metric_method / metric_reference``, computed **per subject** and then
    averaged, so a subject with many windows cannot dominate. Positive means the
    method beats the reference.

    Reported against persistence because it is the one comparator immune to
    protocol mismatch: it needs no training data, no hyperparameters, and no
    assumptions about how another paper built its windows. That makes it the honest
    replacement for the non-citable "<15 mg/dL is clinically acceptable" framing.
    """
    rows = []
    for horizon in result.horizons_min:
        method_values = result.metric_by_subject(metric, horizon)
        reference_values = reference.metric_by_subject(metric, horizon)
        shared = method_values.index.intersection(reference_values.index)
        if shared.empty:
            continue
        skill = 1.0 - (method_values[shared] / reference_values[shared])
        rows.append(
            {
                "horizon_min": horizon,
                "metric": metric,
                "method": result.method,
                "reference": reference.method,
                "skill_mean": float(skill.mean()),
                "skill_sd": float(skill.std(ddof=1)) if skill.size > 1 else np.nan,
                "n_subjects_better": int((skill > 0).sum()),
                "n_subjects": int(skill.size),
            }
        )
    return pd.DataFrame(rows)


def compare_methods(
    results: list[EvaluationResult], *, metric: str = "rmse", reference: str | None = None
) -> tuple[pd.DataFrame, dict[str, PairedComparison]]:
    """Paired comparisons of every method against a reference, across subjects.

    Uses the paired Wilcoxon signed-rank test on per-subject values with
    Holm-Bonferroni correction over the whole family of comparisons, because a
    results table compares several methods at four horizons and uncorrected
    p-values across that family would overstate the evidence.
    """
    if not results:
        raise ValueError("no results to compare")
    by_name = {result.method: result for result in results}
    reference_name = reference or results[0].method
    if reference_name not in by_name:
        raise ValueError(f"reference {reference_name!r} not among {sorted(by_name)}")
    baseline = by_name[reference_name]

    comparisons: dict[str, PairedComparison] = {}
    for result in results:
        if result.method == reference_name:
            continue
        for horizon in result.horizons_min:
            method_values = result.metric_by_subject(metric, horizon)
            reference_values = baseline.metric_by_subject(metric, horizon)
            shared = method_values.index.intersection(reference_values.index)
            label = f"{result.method} vs {reference_name} @{horizon}min ({metric})"
            comparisons[label] = paired_comparison(
                method_values[shared].to_numpy(),
                reference_values[shared].to_numpy(),
                label=label,
                lower_is_better=True,
            )

    corrected = holm_bonferroni({key: value.p_value for key, value in comparisons.items()})
    rows = []
    for label, comparison in comparisons.items():
        adjustment = corrected.get(label, {})
        rows.append(
            {
                "comparison": label,
                "mean_difference": comparison.mean_difference,
                "ci_low": comparison.difference_ci.low,
                "ci_high": comparison.difference_ci.high,
                "effect_size": comparison.effect_size,
                "n_favouring_method": comparison.n_favouring_first,
                "n_subjects": comparison.n_subjects,
                "p_raw": comparison.p_value,
                "p_holm": adjustment.get("p_adjusted", np.nan),
                "significant": adjustment.get("reject", False),
                "underpowered": comparison.underpowered,
                "summary": describe_comparison(
                    comparison, adjusted_p=adjustment.get("p_adjusted")
                ),
            }
        )
    return pd.DataFrame(rows), comparisons


def leaderboard(
    results: list[EvaluationResult], *, metric: str = "rmse"
) -> pd.DataFrame:
    """One row per method and horizon: mean ± SD across subjects, plus pooled.

    Pooled appears beside the headline so the gap between them is visible. Where
    they diverge substantially the divergence is itself a finding about cohort
    imbalance.
    """
    rows = []
    for result in results:
        pooled_by_horizon = result.pooled.set_index("horizon_min")[metric]
        for horizon in result.horizons_min:
            summary_row = result.summary[result.summary["horizon_min"] == horizon]
            if summary_row.empty:
                continue
            record = summary_row.iloc[0]
            rows.append(
                {
                    "method": result.method,
                    "protocol": result.protocol,
                    "horizon_min": horizon,
                    f"{metric}_mean": record.get(f"{metric}_mean", np.nan),
                    f"{metric}_sd": record.get(f"{metric}_sd", np.nan),
                    f"{metric}_pooled": float(pooled_by_horizon.get(horizon, np.nan)),
                    "n_subjects": record.get("n_subjects", np.nan),
                    "n_windows": record.get("n_windows_total", np.nan),
                }
            )
    return pd.DataFrame(rows)


def write_result(result: EvaluationResult, out_dir: str | Path) -> dict[str, Path]:
    """Persist a result's tables and raw predictions.

    Raw predictions are saved so every figure and table can be regenerated without
    re-running a model, which is what makes a single regeneration command possible.
    """
    directory = Path(out_dir) / result.protocol / result.part / result.method
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for label, frame in (
        ("per_subject", result.per_subject),
        ("summary", result.summary),
        ("pooled", result.pooled),
    ):
        path = directory / f"{label}.csv"
        frame.to_csv(path, index=False)
        written[label] = path

    predictions_path = directory / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        **{f"pred__{key}": value for key, value in result.predictions.items()},
        **{f"true__{key}": value for key, value in result.targets.items()},
    )
    written["predictions"] = predictions_path
    return written


__all__ = [
    "EvaluationResult",
    "compare_methods",
    "evaluate_baseline",
    "evaluate_predictions",
    "leaderboard",
    "skill_score",
    "write_result",
]
