"""Shared figure style.

One style module for every figure, so the paper reads as one system rather than as
a collection of scripts. The legacy pipeline had two figure families with different
fonts, different grids, and inline hex colours repeated per function; nothing
enforced consistency and nothing was validated.

Colour
------
The categorical palette is used **in fixed slot order, never cycled**, and the
subset is chosen by chart form because the two forms have different requirements:

* **Bars and lines** compare *adjacent* series, so the adjacent-pair criterion
  applies and five slots are usable. Verified: worst adjacent colour-vision-
  deficiency separation dE 9.1 (target >= 8), worst normal-vision dE 19.6
  (floor 15).
* **Scatter and overlapping marks** put every pair on screen simultaneously, so the
  all-pairs criterion applies and only **three** slots clear it. Verified: worst
  all-pairs CVD dE 9.2, normal-vision dE 24.0. A fourth series in a scatter must
  fold into "other" or become a small multiple -- it may not be given a new hue.

Three slots sit below 3:1 contrast on the light surface, so the **relief rule**
applies: those series carry visible direct labels, and every figure has a CSV
counterpart written beside it.

Error-grid zones are **not** categorical. They are ordered by clinical severity, so
they take a status ramp (benign to critical) rather than identity hues, and severity
is additionally encoded by position in the grid itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Chart surface the palette was validated against.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#78776f"
GRID = "#e4e3dd"

#: Categorical slots, in fixed order. Never cycle; never generate a new hue.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
)

#: Slots that clear the all-pairs criterion. Scatter and overlapping marks are
#: capped at this many series.
SCATTER_SLOTS: tuple[str, ...] = CATEGORICAL[:3]

#: Slots below 3:1 contrast on the light surface; these require direct labels.
NEEDS_RELIEF: frozenset[str] = frozenset({"#1baf7a", "#eda100", "#e87ba4"})

#: The naive comparator is always drawn in neutral ink, never given a hue: it is a
#: reference line, not a competing series, and every results figure carries it.
REFERENCE_INK = "#78776f"

#: Error-grid zones, ordered by clinical severity. A status ramp, not identity
#: hues, with a neutral opening rather than a rainbow.
ZONE_COLOURS: dict[str, str] = {
    "A": "#d4e8d4",
    "B": "#cfe0f2",
    "C": "#fde9c2",
    "D": "#f9cdb4",
    "E": "#f3b5b5",
}
ZONE_EDGE = "#b8b7ae"


def apply_style() -> None:
    """Install the project style on the global matplotlib state."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "medium",
            "axes.labelsize": 9,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.0,
            "lines.solid_capstyle": "round",
            "errorbar.capsize": 0,
        }
    )


def series_colour(index: int, *, form: str = "bar") -> str:
    """Colour for series ``index``, refusing to cycle.

    Raises rather than wrapping around: a ninth series is a signal to fold into
    "other" or facet, not to reuse a hue and make two entities look identical.
    """
    slots = SCATTER_SLOTS if form in {"scatter", "overlap"} else CATEGORICAL
    if index >= len(slots):
        raise ValueError(
            f"{form} charts support {len(slots)} series; series {index} would have to "
            "reuse a hue. Fold the extra series into 'other' or use small multiples."
        )
    return slots[index]


def needs_direct_label(colour: str) -> bool:
    """Whether this slot's contrast obliges a visible direct label."""
    return colour in NEEDS_RELIEF


@contextmanager
def figure(
    *,
    width: float = 6.0,
    height: float = 3.6,
    path: str | Path | None = None,
    **subplot_kwargs: object,
) -> Iterator[tuple[plt.Figure, plt.Axes]]:
    """A styled figure that saves and closes itself.

    Saving inside the context manager means a figure cannot be left open to leak
    into the next one, which is how the legacy scripts ended up with state bleeding
    between plots.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(width, height), **subplot_kwargs)  # type: ignore[arg-type]
    try:
        yield fig, ax
    finally:
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target)
        plt.close(fig)


def annotate_source(ax: plt.Axes, text: str) -> None:
    """Footnote a figure with what produced it.

    Every figure states its protocol and subject count, so a figure lifted out of
    the paper still says what it is.
    """
    ax.text(
        0.0,
        -0.22,
        text,
        transform=ax.transAxes,
        fontsize=7,
        color=TEXT_MUTED,
        va="top",
    )


__all__ = [
    "CATEGORICAL",
    "GRID",
    "NEEDS_RELIEF",
    "REFERENCE_INK",
    "SCATTER_SLOTS",
    "SURFACE",
    "TEXT_MUTED",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "ZONE_COLOURS",
    "ZONE_EDGE",
    "annotate_source",
    "apply_style",
    "figure",
    "needs_direct_label",
    "series_colour",
]
