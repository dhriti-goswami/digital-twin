"""OhioT1DM XML parsing and resampling onto an exact time grid.

Defects in the legacy parser (``scripts/evaluate_ohio.py:124-247``) that are fixed
here:

* **``<temp_basal>`` was never read**, so ``basal_u_h`` was wrong for every period
  during which a temporary basal rate was active -- which is exactly the periods
  around exercise and hypoglycaemia where the insulin signal matters most.
* **Boluses used only ``ts_begin``**, so an extended (square-wave or dual-wave)
  bolus delivered over 30-120 minutes was recorded as an instantaneous dose.
* **Records were compacted with ``reset_index(drop=True)`` after dropping NaN
  glucose**, so downstream windows straddled multi-hour gaps and the nominal
  forecast horizons were not the actual ones. Fixed structurally: this module
  resamples onto a *dense, gap-preserving* grid indexed by true wall-clock time,
  and never removes rows.
* **The channel tag is ``stressors``, not ``stress``**, so the legacy code silently
  read nothing.
* **Cohort sensor differences were zero-filled.** 2018 subjects carry
  ``basis_heart_rate`` / ``basis_steps`` / ``basis_air_temperature`` and no
  ``acceleration``; 2020 subjects carry ``acceleration`` with the Basis channels
  present but identically zero. Filling absent channels with 0.0 *before* scaling
  maps them to large negative z-scores, which a model will happily use as a
  cohort indicator. Here every optional channel gets an explicit availability
  mask instead.

Protocol note
-------------
The two BGLP challenges are **not the same protocol on the same data**: the 2020
edition excludes the first hour (12 samples) of each test file, the 2018 edition
does not. :func:`load_subject` applies this per cohort. Ignoring it makes results
incomparable to published work.

Body weight
-----------
Every OhioT1DM file reports ``weight="99"`` -- a de-identification placeholder, not
a measurement. Body weight is therefore **not identifiable from this dataset**. A
nominal 70 kg is used, and because the distribution volumes are estimated per
kilogram, the unknown true weight is absorbed into the estimated absolute volumes.
Treating 99 kg as real would be fabricated precision.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: OhioT1DM timestamp format: day-month-year, 24-hour clock.
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"

#: Placeholder weight present in every file; see the module docstring.
PLACEHOLDER_WEIGHT_KG = 99.0
NOMINAL_WEIGHT_KG = 70.0

#: Samples dropped from the start of each test file, per cohort.
TEST_WARMUP_EXCLUSION: dict[str, int] = {"2018": 0, "2020": 12}

COHORT_SUBJECTS: dict[str, tuple[str, ...]] = {
    "2018": ("559", "563", "570", "575", "588", "591"),
    "2020": ("540", "544", "552", "567", "584", "596"),
}

#: Channels present in every file, parsed as (timestamp, value) point events.
POINT_CHANNELS: tuple[str, ...] = ("glucose_level", "finger_stick", "basal")

#: Optional physiological-sensor channels. Availability differs by cohort, so each
#: one gets a companion ``*_available`` mask rather than a silent zero fill.
SENSOR_CHANNELS: tuple[str, ...] = (
    "basis_heart_rate",
    "basis_gsr",
    "basis_skin_temperature",
    "basis_air_temperature",
    "basis_steps",
)


class OhioParseError(ValueError):
    """Raised when a file does not match the expected OhioT1DM structure."""


def _parse_timestamp(text: str) -> pd.Timestamp:
    """Single-timestamp parse. Prefer :func:`_parse_timestamps` in bulk.

    ``pd.to_datetime`` carries a large per-call overhead, so calling it once per
    row costs ~27 s per subject file. Only use this for one-off values.
    """
    return pd.to_datetime(text.strip(), format=TIMESTAMP_FORMAT)


def _parse_timestamps(texts: list[str]) -> pd.DatetimeIndex:
    """Vectorised timestamp parse for a whole channel at once."""
    return pd.to_datetime(
        [text.strip() for text in texts], format=TIMESTAMP_FORMAT
    )


def _float_or_none(text: str | None) -> float | None:
    """Ohio uses empty attributes for 'no value', e.g. a suspended temp basal."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _events(root: ElementTree.Element, channel: str) -> list[dict[str, str]]:
    """All ``<event>`` attribute dicts for a channel, or ``[]`` if absent.

    A missing channel tag and a present-but-empty one are both normal in this
    dataset and must be distinguishable from a parse failure.
    """
    node = root.find(channel)
    if node is None:
        return []
    return [dict(event.attrib) for event in node.findall("event")]


@dataclass
class OhioSubject:
    """One parsed OhioT1DM file, still in event form."""

    subject_id: str
    cohort: str
    split: str
    insulin_type: str
    source_path: Path
    #: Point series, indexed by timestamp.
    glucose: pd.Series
    finger_stick: pd.Series
    #: Step series: a basal rate that holds until the next event [U/h].
    basal: pd.Series
    #: Interval events with ``ts_begin`` / ``ts_end``.
    temp_basal: pd.DataFrame
    bolus: pd.DataFrame
    meal: pd.DataFrame
    exercise: pd.DataFrame
    sleep: pd.DataFrame
    work: pd.DataFrame
    hypo_event: pd.DataFrame
    illness: pd.DataFrame
    stressors: pd.DataFrame
    sensors: dict[str, pd.Series] = field(default_factory=dict)
    acceleration: pd.Series | None = None

    @property
    def available_sensors(self) -> tuple[str, ...]:
        """Sensor channels that are present *and* carry a non-constant signal.

        The 2020 files include the Basis channels but fill them with zeros, so
        presence of the tag is not evidence of data.
        """
        available = []
        for name, series in self.sensors.items():
            if series.empty:
                continue
            if float(series.abs().max()) == 0.0:
                continue
            available.append(name)
        return tuple(sorted(available))

    def summary(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "cohort": self.cohort,
            "split": self.split,
            "insulin_type": self.insulin_type,
            "n_glucose": int(self.glucose.size),
            "n_bolus": int(len(self.bolus)),
            "n_meal": int(len(self.meal)),
            "n_temp_basal": int(len(self.temp_basal)),
            "n_exercise": int(len(self.exercise)),
            "glucose_start": self.glucose.index.min() if self.glucose.size else None,
            "glucose_end": self.glucose.index.max() if self.glucose.size else None,
            "available_sensors": self.available_sensors,
        }


def _point_series(root: ElementTree.Element, channel: str) -> pd.Series:
    rows = _events(root, channel)
    if not rows:
        return pd.Series(dtype=np.float64, index=pd.DatetimeIndex([], name="timestamp"))
    stamps, values = [], []
    for row in rows:
        value = _float_or_none(row.get("value"))
        if value is None:
            continue
        stamps.append(row["ts"])
        values.append(value)
    if not stamps:
        return pd.Series(dtype=np.float64, index=pd.DatetimeIndex([], name="timestamp"))
    index = _parse_timestamps(stamps).rename("timestamp")
    series = pd.Series(values, index=index, dtype=np.float64)
    # Duplicate timestamps do occur; keep the first and record nothing silently.
    return series[~series.index.duplicated(keep="first")].sort_index()


def _interval_frame(
    root: ElementTree.Element, channel: str, numeric: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Interval events as a frame with parsed ``ts_begin`` / ``ts_end``."""
    rows = _events(root, channel)
    if not rows:
        columns = ["ts_begin", "ts_end", *numeric]
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})

    # Collect the raw strings first and parse both timestamp columns in one call
    # each: per-row ``pd.to_datetime`` dominates runtime otherwise.
    begin_texts: list[str] = []
    end_texts: list[str] = []
    records: list[dict[str, object]] = []
    for row in rows:
        begin_text = row.get("ts_begin") or row.get("ts")
        if not begin_text:
            continue
        end_text = row.get("ts_end")
        begin_texts.append(begin_text)
        end_texts.append(end_text if end_text and end_text.strip() else begin_text)

        record: dict[str, object] = {}
        for key in numeric:
            record[key] = _float_or_none(row.get(key))
        for key, value in row.items():
            if key not in record and key not in {"ts", "ts_begin", "ts_end"}:
                record.setdefault(key, value)
        records.append(record)

    if not records:
        columns = ["ts_begin", "ts_end", *numeric]
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})

    frame = pd.DataFrame(records)
    frame["ts_begin"] = _parse_timestamps(begin_texts)
    frame["ts_end"] = _parse_timestamps(end_texts)
    ordered = ["ts_begin", "ts_end", *[c for c in frame.columns if c not in {"ts_begin", "ts_end"}]]
    return frame[ordered].sort_values("ts_begin").reset_index(drop=True)


def parse_ohio_xml(path: str | Path) -> OhioSubject:
    """Parse one OhioT1DM XML file into event series.

    Cohort and split are inferred from the path (``.../2018/train/559-ws-training.xml``)
    and cross-checked against :data:`COHORT_SUBJECTS`, so a file in the wrong
    directory cannot be silently mislabelled.
    """
    path = Path(path)
    tree = ElementTree.parse(path)
    root = tree.getroot()
    if root.tag != "patient":
        raise OhioParseError(f"{path}: expected root tag 'patient', got {root.tag!r}")

    subject_id = (root.get("id") or "").strip()
    if not subject_id:
        raise OhioParseError(f"{path}: missing patient id")

    parts = {part.lower() for part in path.parts}
    cohort = next((name for name in COHORT_SUBJECTS if name in parts), "")
    if not cohort:
        raise OhioParseError(f"{path}: cannot infer cohort (expected a 2018/ or 2020/ component)")
    if subject_id not in COHORT_SUBJECTS[cohort]:
        raise OhioParseError(
            f"{path}: subject {subject_id} is not a {cohort} cohort member "
            f"{COHORT_SUBJECTS[cohort]}"
        )
    if "train" in parts:
        split = "train"
    elif "test" in parts:
        split = "test"
    else:
        raise OhioParseError(f"{path}: cannot infer split (expected a train/ or test/ component)")

    weight = _float_or_none(root.get("weight"))
    if weight is not None and weight != PLACEHOLDER_WEIGHT_KG:
        # Informational: if a future release carries real weights, the nominal
        # substitution documented above should be revisited.
        pass

    sensors = {name: _point_series(root, name) for name in SENSOR_CHANNELS}
    acceleration_series = _point_series(root, "acceleration")

    return OhioSubject(
        subject_id=subject_id,
        cohort=cohort,
        split=split,
        insulin_type=(root.get("insulin_type") or "").strip(),
        source_path=path,
        glucose=_point_series(root, "glucose_level"),
        finger_stick=_point_series(root, "finger_stick"),
        basal=_point_series(root, "basal"),
        temp_basal=_interval_frame(root, "temp_basal", numeric=("value",)),
        bolus=_interval_frame(root, "bolus", numeric=("dose", "bwz_carb_input")),
        meal=_interval_frame(root, "meal", numeric=("carbs",)),
        exercise=_interval_frame(root, "exercise", numeric=("intensity", "duration")),
        sleep=_interval_frame(root, "sleep", numeric=("quality",)),
        work=_interval_frame(root, "work", numeric=("intensity",)),
        hypo_event=_interval_frame(root, "hypo_event"),
        illness=_interval_frame(root, "illness"),
        stressors=_interval_frame(root, "stressors"),
        sensors=sensors,
        acceleration=acceleration_series if acceleration_series.size else None,
    )


# --------------------------------------------------------------------------- #
# Resampling onto an exact grid
# --------------------------------------------------------------------------- #

#: Columns produced by :func:`to_grid`, in a fixed order. Any change here is a
#: breaking change to the feature contract and must be made deliberately.
GRID_COLUMNS: tuple[str, ...] = (
    "glucose_mg_dl",
    "glucose_observed",
    "bolus_u_per_min",
    "basal_u_per_min",
    "carbs_mg_per_min",
    "exercise_intensity",
    "sleeping",
    "working",
)


def _grid_index(start: pd.Timestamp, end: pd.Timestamp, grid_minutes: int) -> pd.DatetimeIndex:
    """Exact uniform index covering ``[start, end]``, aligned to the clock.

    Flooring the start to a grid multiple keeps every subject's grid on the same
    absolute phase, so timestamps are comparable across subjects and splits.
    """
    frequency = f"{grid_minutes}min"
    return pd.date_range(start.floor(frequency), end.ceil(frequency), freq=frequency)


def _snap_points(series: pd.Series, index: pd.DatetimeIndex, grid_minutes: int) -> pd.Series:
    """Snap point observations onto the nearest grid slot.

    CGM samples are nominally every 5 minutes but not exactly on the clock, so each
    reading is assigned to its nearest slot. Where two readings land in the same
    slot the mean is taken -- silently dropping one would bias the series.
    """
    if series.empty:
        return pd.Series(np.nan, index=index, dtype=np.float64)
    frequency = f"{grid_minutes}min"
    snapped = series.copy()
    snapped.index = series.index.round(frequency)
    aggregated = snapped.groupby(level=0).mean()
    return aggregated.reindex(index)


def _rate_from_intervals(
    frame: pd.DataFrame,
    index: pd.DatetimeIndex,
    grid_minutes: int,
    *,
    amount_column: str,
    scale: float = 1.0,
) -> pd.Series:
    """Distribute interval-delivered amounts into per-minute rates.

    Mass-conserving by construction: an amount delivered over ``[begin, end]`` is
    spread uniformly across the slots it covers, so summing ``rate * grid_minutes``
    over the grid recovers the total. This is what fixes extended (square-wave and
    dual-wave) boluses, which the legacy parser collapsed to an instant.
    """
    out = pd.Series(0.0, index=index, dtype=np.float64)
    if frame.empty:
        return out
    slot_minutes = float(grid_minutes)

    for row in frame.itertuples(index=False):
        amount = getattr(row, amount_column, None)
        if amount is None or not np.isfinite(amount) or amount == 0.0:
            continue
        amount = float(amount) * scale
        begin = pd.Timestamp(row.ts_begin)
        end = pd.Timestamp(row.ts_end)

        frequency = f"{grid_minutes}min"
        first = begin.round(frequency)
        last = end.round(frequency)
        if last < first:
            first, last = last, first

        slots = index[(index >= first) & (index <= last)]
        if slots.empty:
            # Outside the grid entirely (can happen at a file's edges).
            continue
        # Rate per minute such that total delivered == amount.
        rate = amount / (len(slots) * slot_minutes)
        out.loc[slots] += rate
    return out


def _basal_rate(subject: OhioSubject, index: pd.DatetimeIndex) -> pd.Series:
    """Basal insulin rate [U/min] with temporary-basal overrides applied.

    ``basal`` events are step changes in U/h that hold until the next event.
    ``temp_basal`` intervals override that rate for their duration; an empty
    ``value`` attribute means the pump was suspended, i.e. a rate of zero.
    """
    if subject.basal.empty:
        rate = pd.Series(0.0, index=index, dtype=np.float64)
    else:
        steps = subject.basal.copy()
        # Forward-fill the step function onto the grid, then convert U/h -> U/min.
        combined = steps.reindex(steps.index.union(index)).sort_index().ffill()
        rate = combined.reindex(index).ffill().fillna(0.0) / 60.0

    for row in subject.temp_basal.itertuples(index=False):
        begin = pd.Timestamp(row.ts_begin)
        end = pd.Timestamp(row.ts_end)
        if end <= begin:
            continue
        override = getattr(row, "value", None)
        # An absent value means suspension, not "leave the basal alone".
        override_rate = 0.0 if override is None or not np.isfinite(override) else float(override) / 60.0
        mask = (index >= begin) & (index < end)
        rate.loc[mask] = override_rate

    return rate.astype(np.float64)


def _interval_flag(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """1.0 while any interval in ``frame`` is active, else 0.0."""
    out = pd.Series(0.0, index=index, dtype=np.float64)
    for row in frame.itertuples(index=False):
        begin, end = pd.Timestamp(row.ts_begin), pd.Timestamp(row.ts_end)
        if end <= begin:
            continue
        out.loc[(index >= begin) & (index < end)] = 1.0
    return out


def _exercise_intensity(subject: OhioSubject, index: pd.DatetimeIndex) -> pd.Series:
    """Self-reported exercise intensity held over its stated duration [minutes]."""
    out = pd.Series(0.0, index=index, dtype=np.float64)
    for row in subject.exercise.itertuples(index=False):
        begin = pd.Timestamp(row.ts_begin)
        intensity = getattr(row, "intensity", None)
        duration = getattr(row, "duration", None)
        if intensity is None or not np.isfinite(intensity):
            continue
        minutes = float(duration) if duration is not None and np.isfinite(duration) else 0.0
        end = begin + pd.Timedelta(minutes=max(minutes, 1.0))
        out.loc[(index >= begin) & (index < end)] = float(intensity)
    return out


@dataclass
class GriddedSubject:
    """One subject resampled onto an exact uniform grid.

    ``frame`` is indexed by true wall-clock time with no rows removed, so a gap in
    the CGM record appears as ``NaN`` in ``glucose_mg_dl`` with
    ``glucose_observed == False``. Downstream sequencing relies on that: it is what
    makes "the target at t+30 min is a real observation" a checkable statement
    rather than an assumption.
    """

    subject_id: str
    cohort: str
    split: str
    grid_minutes: int
    frame: pd.DataFrame
    body_weight_kg: float
    available_sensors: tuple[str, ...]
    excluded_warmup_samples: int
    source_path: Path

    @property
    def n_slots(self) -> int:
        return int(len(self.frame))

    @property
    def n_observed(self) -> int:
        return int(self.frame["glucose_observed"].sum())

    @property
    def coverage(self) -> float:
        """Fraction of grid slots backed by a real CGM observation."""
        return self.n_observed / self.n_slots if self.n_slots else 0.0

    def summary(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "cohort": self.cohort,
            "split": self.split,
            "n_slots": self.n_slots,
            "n_observed": self.n_observed,
            "coverage": round(self.coverage, 4),
            "span_start": self.frame.index.min(),
            "span_end": self.frame.index.max(),
            "excluded_warmup_samples": self.excluded_warmup_samples,
            "available_sensors": self.available_sensors,
            "total_bolus_u": float(
                self.frame["bolus_u_per_min"].sum() * self.grid_minutes
            ),
            "total_carbs_g": float(
                self.frame["carbs_mg_per_min"].sum() * self.grid_minutes / 1000.0
            ),
        }


def to_grid(subject: OhioSubject, *, grid_minutes: int = 5) -> pd.DataFrame:
    """Resample a parsed subject onto an exact uniform grid.

    No row is ever dropped. Missing CGM appears as ``NaN`` alongside
    ``glucose_observed = False``.
    """
    if subject.glucose.empty:
        raise OhioParseError(f"{subject.source_path}: no glucose observations")

    index = _grid_index(subject.glucose.index.min(), subject.glucose.index.max(), grid_minutes)
    glucose = _snap_points(subject.glucose, index, grid_minutes)

    frame = pd.DataFrame(index=index)
    frame.index.name = "timestamp"
    frame["glucose_mg_dl"] = glucose
    frame["glucose_observed"] = glucose.notna()
    frame["bolus_u_per_min"] = _rate_from_intervals(
        subject.bolus, index, grid_minutes, amount_column="dose"
    )
    frame["basal_u_per_min"] = _basal_rate(subject, index)
    frame["carbs_mg_per_min"] = _rate_from_intervals(
        subject.meal, index, grid_minutes, amount_column="carbs", scale=1000.0
    )
    frame["exercise_intensity"] = _exercise_intensity(subject, index)
    frame["sleeping"] = _interval_flag(subject.sleep, index)
    frame["working"] = _interval_flag(subject.work, index)

    # Optional sensor channels, each with an explicit availability mask so an
    # absent channel is never confused with a measured zero.
    for name in SENSOR_CHANNELS:
        series = subject.sensors.get(name)
        if name in subject.available_sensors and series is not None:
            snapped = _snap_points(series, index, grid_minutes)
            frame[name] = snapped.ffill().bfill()
            frame[f"{name}_available"] = 1.0
        else:
            frame[name] = 0.0
            frame[f"{name}_available"] = 0.0

    missing = [column for column in GRID_COLUMNS if column not in frame.columns]
    if missing:
        raise OhioParseError(f"grid is missing required columns: {missing}")
    return frame


def load_subject(
    path: str | Path,
    *,
    grid_minutes: int = 5,
    apply_protocol_exclusion: bool = True,
) -> GriddedSubject:
    """Parse and grid one subject, applying the cohort's test-file protocol.

    The 2020 BGLP challenge excludes the first hour (12 samples) of each test
    file; the 2018 edition does not. ``apply_protocol_exclusion=False`` is provided
    for diagnostics only -- reported results must leave it on, or the two cohorts
    are being scored under different rules from the published work.
    """
    subject = parse_ohio_xml(path)
    frame = to_grid(subject, grid_minutes=grid_minutes)

    excluded = 0
    if apply_protocol_exclusion and subject.split == "test":
        excluded = TEST_WARMUP_EXCLUSION.get(subject.cohort, 0)
        if excluded:
            frame = frame.iloc[excluded:]

    return GriddedSubject(
        subject_id=subject.subject_id,
        cohort=subject.cohort,
        split=subject.split,
        grid_minutes=grid_minutes,
        frame=frame,
        body_weight_kg=NOMINAL_WEIGHT_KG,
        available_sensors=subject.available_sensors,
        excluded_warmup_samples=excluded,
        source_path=Path(path),
    )


def discover_files(root: str | Path, *, split: str | None = None) -> list[Path]:
    """All OhioT1DM XML files under ``root``, sorted by cohort then subject.

    Sorted deterministically so that any downstream split derived from file order
    is reproducible.
    """
    root = Path(root)
    if not root.is_dir():
        raise OhioParseError(f"{root} is not a directory")
    pattern = f"*/{split}/*.xml" if split else "*/*/*.xml"
    files = sorted(root.glob(pattern), key=lambda path: (path.parts[-3], path.parts[-1]))
    if not files:
        raise OhioParseError(f"no XML files found under {root} matching {pattern!r}")
    return files


def load_split(
    root: str | Path,
    split: str,
    *,
    grid_minutes: int = 5,
    apply_protocol_exclusion: bool = True,
) -> list[GriddedSubject]:
    """Load every subject in a split."""
    return [
        load_subject(
            path,
            grid_minutes=grid_minutes,
            apply_protocol_exclusion=apply_protocol_exclusion,
        )
        for path in discover_files(root, split=split)
    ]


__all__ = [
    "COHORT_SUBJECTS",
    "GRID_COLUMNS",
    "NOMINAL_WEIGHT_KG",
    "PLACEHOLDER_WEIGHT_KG",
    "SENSOR_CHANNELS",
    "TEST_WARMUP_EXCLUSION",
    "TIMESTAMP_FORMAT",
    "GriddedSubject",
    "OhioParseError",
    "OhioSubject",
    "discover_files",
    "load_split",
    "load_subject",
    "parse_ohio_xml",
    "to_grid",
]
