"""Publication figures.

Every figure is generated from saved predictions and writes a CSV of the exact
numbers it draws beside it, so a figure can always be checked against its own data
and the paper never contains a hand-typed value.

The error-grid figure is the important one. The legacy plots drew decorative line
segments unrelated to the counting logic and did not colour points by zone, so a
figure could contradict its own table indefinitely -- one of them shaded a grid
whose boundaries did not match the classifier at all. Here the background comes from
:func:`~twin.metrics.errorgrid.zone_field`, which evaluates *the same classifier*
used for the table, and each point is coloured by its *computed* zone. Disagreement
is impossible by construction rather than merely unlikely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from twin.eval.style import (
    REFERENCE_INK,
    TEXT_MUTED,
    TEXT_SECONDARY,
    ZONE_COLOURS,
    ZONE_EDGE,
    annotate_source,
    figure,
    needs_direct_label,
    series_colour,
)
from twin.metrics.errorgrid import zone_field, zone_summary

Array = NDArray[np.floating]


def _write_data(frame: pd.DataFrame, path: Path) -> None:
    """Persist the numbers a figure draws, beside the figure."""
    frame.to_csv(path.with_suffix(".csv"), index=False)


# --------------------------------------------------------------------------- #
# Error grids
# --------------------------------------------------------------------------- #


def plot_error_grid(
    reference: Array,
    predicted: Array,
    path: str | Path,
    *,
    grid: str = "clarke",
    horizon_min: int | None = None,
    protocol: str = "",
    glucose_max: float = 400.0,
    max_points: int = 6000,
    seed: int = 42,
) -> pd.DataFrame:
    """Error-grid scatter over the classifier's own shaded zones.

    Returns the zone percentages actually drawn, which are the same numbers the
    results table reports.
    """
    path = Path(path)
    reference = np.asarray(reference, dtype=np.float64).ravel()
    predicted = np.asarray(predicted, dtype=np.float64).ravel()

    summary = zone_summary(reference, predicted, grid=grid)
    axis_x, axis_y, indices, order = zone_field(
        grid=grid, glucose_max=glucose_max, resolution=420
    )
    from twin.metrics.errorgrid import clarke_zone, parkes_zone

    classify = {"clarke": clarke_zone, "parkes": parkes_zone}[grid]
    zones = classify(reference, predicted)

    # Thin the scatter for legibility only. Reported percentages always come from
    # the full set, so thinning can never change a number.
    generator = np.random.default_rng(seed)
    if reference.size > max_points:
        keep = generator.choice(reference.size, size=max_points, replace=False)
    else:
        keep = np.arange(reference.size)

    with figure(width=5.0, height=4.8, path=path) as (_, ax):
        colours = ListedColormap([ZONE_COLOURS[zone] for zone in order])
        ax.pcolormesh(
            axis_x, axis_y, indices.T, cmap=colours, shading="nearest", rasterized=True
        )
        ax.contour(
            axis_x, axis_y, indices.T, levels=len(order) - 1, colors=[ZONE_EDGE], linewidths=0.6
        )

        for zone in order:
            mask = zones[keep] == zone
            if not mask.any():
                continue
            ax.scatter(
                reference[keep][mask],
                predicted[keep][mask],
                s=3.5,
                linewidths=0.0,
                alpha=0.55,
                color="#0b0b0b" if zone == "A" else "#26251f",
                rasterized=True,
            )

        ax.plot([0, glucose_max], [0, glucose_max], color=TEXT_MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, glucose_max)
        ax.set_ylim(0, glucose_max)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("Reference glucose (mg/dL)")
        ax.set_ylabel("Predicted glucose (mg/dL)")

        title = f"{grid.title()} error grid"
        if horizon_min is not None:
            title += f" — {horizon_min} min horizon"
        ax.set_title(title, loc="left")

        # Zone percentages as a fixed swatch legend, not as labels positioned at each
        # zone's centroid. A centroid is not inside its own zone for the sparse zones:
        # the mean of the B points sits on the diagonal, so a label placed there
        # appears to annotate zone A. Each entry pairs the zone's fill colour with its
        # letter, so identity never rests on colour alone.
        entries = [
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=7,
                markerfacecolor=ZONE_COLOURS[zone],
                markeredgecolor=ZONE_EDGE,
                markeredgewidth=0.6,
                label=f"{zone}  {summary.percentages[zone]:5.1f}%  (n={summary.counts[zone]:,})",
            )
            for zone in order
        ]
        ax.legend(
            handles=entries,
            loc="upper left",
            bbox_to_anchor=(0.015, 0.985),
            handletextpad=0.6,
            labelspacing=0.35,
            fontsize=8,
            facecolor="#fcfcfb",
            framealpha=0.88,
            frameon=True,
            edgecolor=ZONE_EDGE,
        )
        ax.annotate(
            f"A+B {summary.clinically_acceptable:.1f}%   D+E {summary.dangerous:.1f}%",
            xy=(0.985, 0.02),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            color=TEXT_SECONDARY,
            bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
        )

        footnote = f"n = {summary.n:,} windows"
        if protocol:
            footnote += f" · {protocol} protocol"
        footnote += " · zones shaded by the same classifier that produced the percentages"
        annotate_source(ax, footnote)

    frame = pd.DataFrame(
        [
            {
                "grid": grid,
                "horizon_min": horizon_min,
                "zone": zone,
                "percent": summary.percentages[zone],
                "n": summary.counts[zone],
            }
            for zone in order
        ]
    )
    _write_data(frame, path)
    return frame


# --------------------------------------------------------------------------- #
# Accuracy by horizon
# --------------------------------------------------------------------------- #


def plot_horizon_metric(
    summaries: dict[str, pd.DataFrame],
    path: str | Path,
    *,
    metric: str = "rmse",
    reference_method: str = "persistence",
    protocol: str = "",
) -> pd.DataFrame:
    """Metric versus horizon, mean +/- SD across subjects, with the naive reference.

    The reference method is drawn in neutral ink as a dashed line rather than given a
    categorical hue: it is the bar every result must clear, not a competing series.
    Error bars are the across-subject SD, which is the quantity a reader should judge
    a difference against.
    """
    path = Path(path)
    rows: list[dict[str, object]] = []
    mean_column, sd_column = f"{metric}_mean", f"{metric}_sd"

    with figure(width=6.2, height=3.8, path=path) as (_, ax):
        methods = [name for name in summaries if name != reference_method]
        if reference_method in summaries:
            frame = summaries[reference_method].sort_values("horizon_min")
            ax.plot(
                frame["horizon_min"],
                frame[mean_column],
                color=REFERENCE_INK,
                lw=1.6,
                ls="--",
                marker="o",
                markersize=4,
                label=f"{reference_method} (naive)",
                zorder=2,
            )
            ax.annotate(
                reference_method,
                xy=(frame["horizon_min"].iloc[-1], frame[mean_column].iloc[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=REFERENCE_INK,
                va="center",
            )
            for _, record in frame.iterrows():
                rows.append(
                    {
                        "method": reference_method,
                        "horizon_min": record["horizon_min"],
                        "metric": metric,
                        "mean": record[mean_column],
                        "sd": record.get(sd_column, np.nan),
                    }
                )

        for index, name in enumerate(methods):
            frame = summaries[name].sort_values("horizon_min")
            colour = series_colour(index, form="bar")
            ax.errorbar(
                frame["horizon_min"],
                frame[mean_column],
                yerr=frame.get(sd_column),
                color=colour,
                lw=2.0,
                marker="o",
                markersize=5,
                elinewidth=1.0,
                label=name,
                zorder=3,
            )
            # Relief rule: low-contrast slots must carry a visible direct label.
            if needs_direct_label(colour):
                ax.annotate(
                    name,
                    xy=(frame["horizon_min"].iloc[-1], frame[mean_column].iloc[-1]),
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color=TEXT_SECONDARY,
                )
            for _, record in frame.iterrows():
                rows.append(
                    {
                        "method": name,
                        "horizon_min": record["horizon_min"],
                        "metric": metric,
                        "mean": record[mean_column],
                        "sd": record.get(sd_column, np.nan),
                    }
                )

        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_ylabel(f"{metric.upper()} (mg/dL)")
        ax.set_title(f"{metric.upper()} by horizon", loc="left")
        ax.set_xticks(sorted(summaries[next(iter(summaries))]["horizon_min"].unique()))
        if len(summaries) >= 2:
            ax.legend(loc="upper left")
        annotate_source(
            ax,
            f"mean ± SD across subjects{' · ' + protocol + ' protocol' if protocol else ''}"
            " · error bars are between-subject SD, not confidence intervals",
        )

    frame = pd.DataFrame(rows)
    _write_data(frame, path)
    return frame


def plot_skill_score(
    skill: pd.DataFrame,
    path: str | Path,
    *,
    protocol: str = "",
) -> pd.DataFrame:
    """Fractional improvement over the naive baseline, per horizon.

    Zero is drawn explicitly: a bar below it means the method is worse than
    predicting no change, which is the state both legacy headline numbers were in.
    """
    path = Path(path)
    with figure(width=5.6, height=3.4, path=path) as (_, ax):
        ordered = skill.sort_values("horizon_min")
        positions = np.arange(len(ordered))
        values = ordered["skill_mean"].to_numpy()
        colour = series_colour(0, form="bar")
        ax.bar(
            positions,
            values,
            width=0.55,
            color=colour,
            yerr=ordered.get("skill_sd"),
            error_kw={"elinewidth": 1.0, "ecolor": TEXT_MUTED},
        )
        ax.axhline(0.0, color=TEXT_SECONDARY, lw=1.0)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{int(value)} min" for value in ordered["horizon_min"]])
        ax.set_ylabel("Skill vs persistence")
        ax.set_title("Improvement over predicting no change", loc="left")

        for position, value, better, total in zip(
            positions,
            values,
            ordered["n_subjects_better"],
            ordered["n_subjects"],
            strict=False,
        ):
            ax.annotate(
                f"{value:+.1%}\n{int(better)}/{int(total)}",
                xy=(position, value),
                xytext=(0, 6 if value >= 0 else -18),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )

        annotate_source(
            ax,
            "positive is better than persistence · second line is subjects improved"
            f"{' · ' + protocol + ' protocol' if protocol else ''}",
        )
    _write_data(ordered, path)
    return ordered


# --------------------------------------------------------------------------- #
# Traces and learning curves
# --------------------------------------------------------------------------- #


def plot_prediction_trace(
    timestamps: pd.DatetimeIndex,
    actual: Array,
    predictions: dict[str, Array],
    path: str | Path,
    *,
    horizon_min: int,
    subject_id: str = "",
) -> None:
    """A single subject's trace: actual glucose with each method overlaid.

    Capped at three overlaid series because overlapping marks put every pair on
    screen at once, which is the all-pairs case.
    """
    path = Path(path)
    if len(predictions) > 3:
        raise ValueError(
            f"{len(predictions)} overlaid series exceeds the all-pairs colour cap of 3; "
            "use small multiples instead"
        )

    with figure(width=7.2, height=3.4, path=path) as (_, ax):
        ax.axhspan(70, 180, color="#eef1ea", zorder=0)
        ax.plot(timestamps, actual, color="#0b0b0b", lw=1.8, label="actual", zorder=3)
        for index, (name, values) in enumerate(predictions.items()):
            ax.plot(
                timestamps,
                values,
                color=series_colour(index, form="overlap"),
                lw=1.6,
                alpha=0.9,
                label=name,
                zorder=2,
            )
        ax.axhline(70, color=TEXT_MUTED, lw=0.7, ls=":")
        ax.axhline(180, color=TEXT_MUTED, lw=0.7, ls=":")
        ax.set_ylabel("Glucose (mg/dL)")
        title = f"{horizon_min}-minute forecast"
        if subject_id:
            title += f" — subject {subject_id}"
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right", ncol=len(predictions) + 1)
        annotate_source(ax, "shaded band is the 70–180 mg/dL target range")


def plot_learning_curves(history: pd.DataFrame, path: str | Path) -> None:
    """Training loss and validation MAE, with the curriculum boundaries marked.

    The boundaries matter: validation scores before the curriculum completes were
    produced by a different objective and are not comparable to later ones, which is
    why model selection ignores them. Marking them stops the early part of the curve
    being read as a fair comparison.
    """
    path = Path(path)
    with figure(width=6.4, height=3.6, path=path) as (fig, ax):
        ax.plot(
            history["epoch"],
            history["val_mae_30"],
            color=series_colour(0, form="bar"),
            label="validation MAE @30 min",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE (mg/dL)")

        twin_ax = ax.twinx()
        twin_ax.plot(
            history["epoch"],
            history["train_loss"],
            color=REFERENCE_INK,
            lw=1.4,
            ls="--",
            label="training loss",
        )
        twin_ax.set_ylabel("Training loss", color=TEXT_MUTED)
        twin_ax.grid(False)
        twin_ax.spines["top"].set_visible(False)

        if "physics_ramp" in history:
            ramped = history.loc[history["physics_ramp"] >= 1.0, "epoch"]
            if len(ramped):
                boundary = float(ramped.min())
                ax.axvline(boundary, color=TEXT_MUTED, lw=0.9, ls=":")
                ax.annotate(
                    "curriculum complete;\nselection begins",
                    xy=(boundary, ax.get_ylim()[1]),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=7,
                    color=TEXT_MUTED,
                    va="top",
                )

        handles = [
            Line2D([], [], color=series_colour(0, form="bar"), label="validation MAE @30 min"),
            Line2D([], [], color=REFERENCE_INK, ls="--", label="training loss"),
        ]
        ax.legend(handles=handles, loc="upper right")
        ax.set_title("Training history", loc="left")
        annotate_source(
            ax,
            "two measures on separate axes are shown here only because they share an "
            "epoch index and are not compared to each other",
        )


def plot_range_agreement(
    per_subject: pd.DataFrame, path: str | Path, *, horizon_min: int = 30
) -> pd.DataFrame:
    """Predicted versus actual time-in-range, per subject.

    Exposes excursion compression, which RMSE hides: a model that flattens toward
    the mean shows systematically higher predicted time-in-range than actual.
    """
    path = Path(path)
    rows = per_subject[per_subject["horizon_min"] == horizon_min].sort_values("subject_id")
    with figure(width=5.2, height=4.4, path=path) as (_, ax):
        ax.plot([0, 100], [0, 100], color=TEXT_MUTED, lw=0.8, ls=":")
        ax.scatter(
            rows["actual_in_range"],
            rows["predicted_in_range"],
            s=42,
            color=series_colour(0, form="scatter"),
            zorder=3,
        )
        for _, record in rows.iterrows():
            ax.annotate(
                str(record["subject_id"]),
                xy=(record["actual_in_range"], record["predicted_in_range"]),
                xytext=(5, -3),
                textcoords="offset points",
                fontsize=7,
                color=TEXT_SECONDARY,
            )
        ax.set_xlabel("Actual time in range (%)")
        ax.set_ylabel("Predicted time in range (%)")
        ax.set_title(f"Time-in-range agreement — {horizon_min} min", loc="left")
        ax.set_aspect("equal")
        annotate_source(
            ax, "points above the diagonal indicate excursion compression toward the mean"
        )
    _write_data(rows, path)
    return rows


__all__ = [
    "plot_error_grid",
    "plot_horizon_metric",
    "plot_learning_curves",
    "plot_prediction_trace",
    "plot_range_agreement",
    "plot_skill_score",
]
