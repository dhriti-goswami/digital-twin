"""The feature contract.

One ordered, named list. Every consumer builds and reads features through this
module, and :func:`build_features` asserts the produced matrix matches
:data:`FEATURE_NAMES` exactly.

Defects this replaces
---------------------
* **Magic index writes.** ``evaluate_ohio.py:271`` and ``finetune_ohio.py:116``
  wrote activity channels at ``feat_matrix[:, 31 + i]``. Reordering the feature
  list would have silently reassigned columns with nothing raising.
* **A time-reversed insulin kernel.** ``ode_features._compute_iob`` convolved the
  bolus train with a reversed *activity* curve, so the feature was ~0 at the moment
  a bolus was given and peaked 145 minutes later. It was neither
  insulin-remaining nor insulin-activity.
* **An unnormalised carbohydrate kernel.** ``_compute_cob`` used a bare
  ``exp(-k/36)``, which jumped instantaneously and never decayed inside the window,
  so it did not conserve mass.
* **A duplicated feature.** ``time_frac_day`` was numerically identical to
  ``day_frac``, giving 34 effective features where 35 were claimed.
* **Rates computed as index differences.** Rate-of-change was ``g[i] - g[i-k]``,
  a *difference* labelled as a rate, computed over array positions that could span
  a gap. Here rates are in mg/dL/min on the true time grid, and gap-spanning
  windows have already been removed by :mod:`twin.data.sequencing`.

Mechanistic features
--------------------
``IOB``, ``COB``, plasma insulin, remote insulin action and glucose appearance all
come from :mod:`twin.physio.compartments` -- the same model that supplies the
physics residual. They are mass-conserving by construction.

These are computed with **population parameters**, deliberately. Features must not
change while the network's per-patient parameter estimates move during training,
or the input distribution would drift under the scaler. The estimated,
patient-specific parameters are used in the physics loss, where varying them is
the entire point. The separation is the same one that distinguishes a fixed input
transform from a learned one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from twin.data.ohio import GriddedSubject
from twin.data.sequencing import FILLED_COLUMN, INTERPOLATED_COLUMN, interpolate_bounded
from twin.physio import basal_steady_state, population_params, simulate_compartments

Array = NDArray[np.float64]

#: Cap on "time since" features [min]. Beyond this the exact value carries no
#: information and an uncapped counter would dominate the scaler's range.
TIME_SINCE_CAP_MIN = 480.0

#: Optional sensor channels carried with an availability mask.
MASKED_SENSORS: tuple[str, ...] = (
    "basis_heart_rate",
    "basis_gsr",
    "basis_skin_temperature",
)

#: The definitive, ordered feature list. Changing this is a breaking change.
FEATURE_NAMES: tuple[str, ...] = (
    # Glucose level and dynamics. Rates are mg/dL/min on the true time grid.
    "glucose_mg_dl",
    "roc_5min",
    "roc_15min",
    "roc_30min",
    "glucose_mean_1h",
    "glucose_std_1h",
    "glucose_min_1h",
    "glucose_max_1h",
    "glucose_mean_2h",
    # Mechanistic states, from the same model as the physics residual.
    "iob_u",
    "insulin_sc_1_u",
    "insulin_sc_2_u",
    "insulin_plasma_uU_mL",
    "insulin_action_per_min",
    "cob_g",
    "carbs_stomach_g",
    "carbs_gut_g",
    "glucose_appearance_mgdl_per_min",
    # Therapy context.
    "basal_u_per_min",
    "bolus_u_per_min",
    "minutes_since_bolus",
    "minutes_since_meal",
    # Time of day.
    "hour_sin",
    "hour_cos",
    "is_night",
    # Activity and context.
    "exercise_intensity",
    "sleeping",
    "working",
    # Data quality: makes an interpolated input distinguishable from a measured one.
    "glucose_interpolated",
    # Masked sensors: value is zeroed when unavailable, and the mask says so.
    "basis_heart_rate",
    "basis_heart_rate_available",
    "basis_gsr",
    "basis_gsr_available",
    "basis_skin_temperature",
    "basis_skin_temperature_available",
)

N_FEATURES = len(FEATURE_NAMES)

#: Features computed from the glucose series, and therefore the only ones allowed
#: to be ``NaN`` where the record has an un-bridgeable gap.
GLUCOSE_DERIVED: frozenset[str] = frozenset(
    {
        "glucose_mg_dl",
        "roc_5min",
        "roc_15min",
        "roc_30min",
        "glucose_mean_1h",
        "glucose_std_1h",
        "glucose_min_1h",
        "glucose_max_1h",
        "glucose_mean_2h",
    }
)


class FeatureError(ValueError):
    """Raised when the produced matrix violates the contract."""


def _rate(series: pd.Series, steps: int, grid_minutes: int) -> Array:
    """Rate of change over ``steps`` grid steps, in units per minute.

    A *rate*, not a difference: the legacy features stored ``g[i] - g[i-k]`` under a
    name implying a rate, which made the physics term dimensionally wrong.

    ``NaN`` is propagated rather than filled. If the look-back slot is inside an
    un-bridgeable gap the rate is genuinely unknown, and inventing a value there is
    how a fabricated number reaches a model input: carrying the last observation
    forward and differencing against it produces rates of tens of mg/dL/min at gap
    edges. :func:`valid_row_mask` then excludes those rows from any window.
    """
    values = series.to_numpy(dtype=np.float64)
    minutes = steps * grid_minutes
    out = np.full(values.size, np.nan)
    out[steps:] = (values[steps:] - values[:-steps]) / minutes
    return out


def _mechanistic_states(
    subject: GriddedSubject, frame: pd.DataFrame
) -> dict[str, Array]:
    """Run the compartment model across the whole record.

    Initialised at the analytic basal steady state for the record's first basal
    rate, so there is no start-up transient in the insulin limb. The gut starts
    empty, which is correct.
    """
    grid = float(subject.grid_minutes)
    params = population_params(
        batch_size=1, body_weight_kg=subject.body_weight_kg, dtype=torch.float64
    )

    basal = frame["basal_u_per_min"].to_numpy(dtype=np.float64)
    bolus = frame["bolus_u_per_min"].to_numpy(dtype=np.float64)
    carbs = frame["carbs_mg_per_min"].to_numpy(dtype=np.float64)

    u_ins = torch.tensor((basal + bolus)[None, :], dtype=torch.float64)
    u_carb = torch.tensor(carbs[None, :], dtype=torch.float64)
    x0 = basal_steady_state(params, torch.tensor([basal[0]], dtype=torch.float64))

    trajectory = simulate_compartments(params, u_ins, u_carb, dt=grid, x0=x0)
    # simulate_compartments returns T+1 states (initial condition first); align to
    # the input grid by taking the state at the start of each slot.
    take = slice(0, len(frame))
    return {
        "iob_u": trajectory.iob_u[0, take].numpy(),
        "insulin_sc_1_u": trajectory.S1[0, take].numpy(),
        "insulin_sc_2_u": trajectory.S2[0, take].numpy(),
        "insulin_plasma_uU_mL": trajectory.plasma_insulin[0, take].numpy(),
        "insulin_action_per_min": trajectory.X[0, take].numpy(),
        "cob_g": trajectory.cob_g[0, take].numpy(),
        "carbs_stomach_g": (trajectory.Qsto[0, take] / 1000.0).numpy(),
        "carbs_gut_g": (trajectory.Qgut[0, take] / 1000.0).numpy(),
        "glucose_appearance_mgdl_per_min": trajectory.Ra_mgdl_per_min[0, take].numpy(),
    }


def _minutes_since(events: Array, grid_minutes: int) -> Array:
    """Minutes since the last non-zero entry, capped.

    Before the first event the value is the cap, which is the honest encoding: "no
    recent event", not "an event just happened".
    """
    out = np.full(events.size, TIME_SINCE_CAP_MIN)
    elapsed = TIME_SINCE_CAP_MIN
    for index in range(events.size):
        if events[index] > 0:
            elapsed = 0.0
        out[index] = min(elapsed, TIME_SINCE_CAP_MIN)
        elapsed += grid_minutes
    return out


@dataclass
class FeatureMatrix:
    """Features for one subject's full record, aligned to its grid."""

    subject_id: str
    values: Array  # (n_slots, N_FEATURES)
    names: tuple[str, ...]
    index: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if self.values.shape[1] != len(self.names):
            raise FeatureError(
                f"{self.subject_id}: {self.values.shape[1]} columns for {len(self.names)} names"
            )
        if self.names != FEATURE_NAMES:
            raise FeatureError(f"{self.subject_id}: feature names deviate from the contract")
        # NaN is permitted only in glucose-derived columns, where it marks a genuine
        # gap. Anywhere else it is a bug: the mechanistic, therapy, time and context
        # features depend on inputs that are always present.
        offenders = [
            name
            for column, name in enumerate(self.names)
            if name not in GLUCOSE_DERIVED and not np.isfinite(self.values[:, column]).all()
        ]
        if offenders:
            raise FeatureError(f"{self.subject_id}: unexpected non-finite values in {offenders}")

    def column(self, name: str) -> Array:
        return self.values[:, self.names.index(name)]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=list(self.names), index=self.index)

    def valid_row_mask(self) -> NDArray[np.bool_]:
        """Rows whose every feature is finite, and therefore usable in a window.

        This is the general form of the horizon-integrity rule applied to *inputs*:
        a window may only be emitted if every feature of every input slot was
        computable from real data. Because ``roc_30min`` looks back six slots, the
        exclusion zone around a gap automatically extends six slots earlier -- which
        a glucose-only check would miss, letting a rate differenced against a
        carried value into the window.
        """
        return np.isfinite(self.values).all(axis=1)


def build_features(
    subject: GriddedSubject,
    *,
    frame: pd.DataFrame | None = None,
    max_interp_gap: int = 2,
) -> FeatureMatrix:
    """Build the full feature matrix for one subject.

    The matrix is aligned one-to-one with the subject's grid, which is what makes
    positional window slicing safe. Rows inside an un-bridgeable gap carry ``NaN`` in
    the glucose-derived columns; :meth:`FeatureMatrix.valid_row_mask` identifies
    them, and window construction excludes them. Nothing is carried forward, so no
    fabricated value can reach a model input.
    """
    work = interpolate_bounded(subject.frame, max_interp_gap=max_interp_gap) if frame is None else frame
    grid = subject.grid_minutes

    # Short interior gaps are interpolated; longer ones stay NaN and propagate
    # through every glucose-derived feature.
    glucose = work[FILLED_COLUMN]
    if glucose.isna().all():
        raise FeatureError(f"{subject.subject_id}: no glucose data at all")

    columns: dict[str, Array] = {}
    columns["glucose_mg_dl"] = glucose.to_numpy(dtype=np.float64)
    for name, steps in (
        ("roc_5min", 1),
        ("roc_15min", 3),
        ("roc_30min", 6),
    ):
        columns[name] = _rate(glucose, steps, grid)

    hour_window = max(1, 60 // grid)
    two_hour_window = max(1, 120 // grid)
    rolling_1h = glucose.rolling(hour_window, min_periods=1)
    columns["glucose_mean_1h"] = rolling_1h.mean().to_numpy(dtype=np.float64)
    # ddof=0: a population SD over the window, so a single-sample window is 0
    # rather than NaN.
    columns["glucose_std_1h"] = rolling_1h.std(ddof=0).fillna(0.0).to_numpy(dtype=np.float64)
    columns["glucose_min_1h"] = rolling_1h.min().to_numpy(dtype=np.float64)
    columns["glucose_max_1h"] = rolling_1h.max().to_numpy(dtype=np.float64)
    columns["glucose_mean_2h"] = (
        glucose.rolling(two_hour_window, min_periods=1).mean().to_numpy(dtype=np.float64)
    )

    columns.update(_mechanistic_states(subject, work))

    basal = work["basal_u_per_min"].to_numpy(dtype=np.float64)
    bolus = work["bolus_u_per_min"].to_numpy(dtype=np.float64)
    carbs = work["carbs_mg_per_min"].to_numpy(dtype=np.float64)
    columns["basal_u_per_min"] = basal
    columns["bolus_u_per_min"] = bolus
    columns["minutes_since_bolus"] = _minutes_since(bolus, grid)
    columns["minutes_since_meal"] = _minutes_since(carbs, grid)

    minutes_of_day = (
        work.index.hour.to_numpy() * 60 + work.index.minute.to_numpy()
    ).astype(np.float64)
    angle = 2.0 * np.pi * minutes_of_day / 1440.0
    columns["hour_sin"] = np.sin(angle)
    columns["hour_cos"] = np.cos(angle)
    hour = work.index.hour.to_numpy()
    columns["is_night"] = ((hour < 6) | (hour >= 22)).astype(np.float64)

    columns["exercise_intensity"] = work["exercise_intensity"].to_numpy(dtype=np.float64)
    columns["sleeping"] = work["sleeping"].to_numpy(dtype=np.float64)
    columns["working"] = work["working"].to_numpy(dtype=np.float64)
    columns["glucose_interpolated"] = work[INTERPOLATED_COLUMN].to_numpy(dtype=np.float64)

    for sensor in MASKED_SENSORS:
        available = work.get(f"{sensor}_available")
        mask = (
            available.to_numpy(dtype=np.float64)
            if available is not None
            else np.zeros(len(work), dtype=np.float64)
        )
        raw = (
            work[sensor].to_numpy(dtype=np.float64)
            if sensor in work.columns
            else np.zeros(len(work), dtype=np.float64)
        )
        # Zero the value where unavailable and state that in the mask. The zero is
        # then unambiguous, unlike the legacy silent fill that a scaler turned into
        # a large negative z-score indistinguishable from a real extreme reading.
        columns[sensor] = np.nan_to_num(raw) * mask
        columns[f"{sensor}_available"] = mask

    missing = [name for name in FEATURE_NAMES if name not in columns]
    if missing:
        raise FeatureError(f"feature contract not satisfied; missing {missing}")
    extra = [name for name in columns if name not in FEATURE_NAMES]
    if extra:
        raise FeatureError(f"features produced but not declared in the contract: {extra}")

    values = np.column_stack([columns[name] for name in FEATURE_NAMES])
    return FeatureMatrix(
        subject_id=subject.subject_id,
        values=values,
        names=FEATURE_NAMES,
        index=work.index,
    )


def feature_provenance() -> pd.DataFrame:
    """Each feature with its group and unit, for the paper's methods section."""
    groups = {
        **{
            name: ("glucose", unit)
            for name, unit in (
                ("glucose_mg_dl", "mg/dL"),
                ("roc_5min", "mg/dL/min"),
                ("roc_15min", "mg/dL/min"),
                ("roc_30min", "mg/dL/min"),
                ("glucose_mean_1h", "mg/dL"),
                ("glucose_std_1h", "mg/dL"),
                ("glucose_min_1h", "mg/dL"),
                ("glucose_max_1h", "mg/dL"),
                ("glucose_mean_2h", "mg/dL"),
            )
        },
        **{
            name: ("mechanistic", unit)
            for name, unit in (
                ("iob_u", "U"),
                ("insulin_sc_1_u", "U"),
                ("insulin_sc_2_u", "U"),
                ("insulin_plasma_uU_mL", "uU/mL"),
                ("insulin_action_per_min", "1/min"),
                ("cob_g", "g"),
                ("carbs_stomach_g", "g"),
                ("carbs_gut_g", "g"),
                ("glucose_appearance_mgdl_per_min", "mg/dL/min"),
            )
        },
        **{
            name: ("therapy", unit)
            for name, unit in (
                ("basal_u_per_min", "U/min"),
                ("bolus_u_per_min", "U/min"),
                ("minutes_since_bolus", "min"),
                ("minutes_since_meal", "min"),
            )
        },
        **{
            name: ("time", "dimensionless")
            for name in ("hour_sin", "hour_cos", "is_night")
        },
        **{
            name: ("context", "dimensionless")
            for name in ("exercise_intensity", "sleeping", "working", "glucose_interpolated")
        },
    }
    rows = []
    for name in FEATURE_NAMES:
        group, unit = groups.get(name, ("sensor", "device units"))
        rows.append({"feature": name, "group": group, "unit": unit})
    return pd.DataFrame(rows)


__all__ = [
    "FEATURE_NAMES",
    "MASKED_SENSORS",
    "N_FEATURES",
    "TIME_SINCE_CAP_MIN",
    "FeatureError",
    "FeatureMatrix",
    "build_features",
    "feature_provenance",
]
