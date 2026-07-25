"""Physiological parameters, their admissible ranges, and reparameterisation.

The model estimates patient-specific parameters, so two things must hold:

1. **No parameter may leave a physiologically plausible range.** Enforced by
   construction with a scaled sigmoid, not by a penalty — a penalty can be
   overwhelmed by the data term and lets the optimiser park a parameter at an
   absurd value that happens to fit.
2. **The ranges must be sourced.** Every bound below carries a provenance tag.

.. warning::

   Bounds marked ``PROVISIONAL`` are placeholders pending the Phase 0 literature
   verification (see ``docs/CITATIONS.md``). They are order-of-magnitude
   correct and adequate for development, but **no reported result may be
   produced with a PROVISIONAL bound still in place.** ``assert_bounds_sourced``
   fails the run if any remain.

Units are stated for every quantity and are consistent throughout ``twin.physio``:

======================  ==========================  ==========================
Symbol                  Unit                        Meaning
======================  ==========================  ==========================
``p1``                  1/min                       Glucose effectiveness
``p2``                  1/min                       Insulin-action decay
``p3``                  mL/(uU*min^2)               Insulin-action rise
``n``                   1/min                       Plasma insulin clearance
``V_G``                 dL                          Glucose distribution volume
``V_I``                 L                           Insulin distribution volume
``tmax_I``              min                         SC insulin transit time
``k_gri``               1/min                       Gastric emptying rate
``k_abs``               1/min                       Intestinal absorption rate
``f``                   dimensionless               Carbohydrate bioavailability
``G_b``                 mg/dL                       Basal (fasting) glucose
``I_b``                 uU/mL                       Basal plasma insulin
======================  ==========================  ==========================

Derived: insulin sensitivity ``S_I = p3 / p2`` [mL/(uU*min)].
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import torch
from torch import Tensor

#: Confidence tags. ``PROVISIONAL`` and ``UNVERIFIED`` block reporting;
#: ``SECOND_HAND`` is permitted but listed in the run manifest so it can be
#: disclosed in the paper.
PROVISIONAL = "PROVISIONAL"
UNVERIFIED = "UNVERIFIED"
SECOND_HAND = "SECOND-HAND"
VERIFIED_PRIMARY = "VERIFIED-PRIMARY"

_BLOCKING = frozenset({PROVISIONAL, UNVERIFIED})

#: Short keys into ``docs/CITATIONS_methods.md``.
SOURCES: dict[str, str] = {
    "ward1991": (
        "Ward GM, Walters JM, Aitken PM, Best JD, Alford FP. Effects of prolonged "
        "pulsatile hyperinsulinemia in humans. Metabolism 1991;40(1):4-9. "
        "IDDM values: S_I = 2.5 +/- 0.6e-4, S_G = 1.0-1.6e-2 /min."
    ),
    "hovorka2004": (
        "Hovorka R, Canonico V, Chassin LJ, et al. Nonlinear model predictive "
        "control of glucose concentration in subjects with type 1 diabetes. "
        "Physiol Meas 2004;25:905-920. t_max,I = 55 min, k_e = 0.138 /min, "
        "V_I = 0.12 L/kg."
    ),
    "lehmann1992": (
        "Lehmann ED, Deutsch T. A physiological model of glucose-insulin "
        "interaction in type 1 diabetes mellitus. J Biomed Eng 1992;14(3):235-242. "
        "k_gabs = 1 /h = 0.0167 /min; trapezoidal gastric emptying, "
        "T_asc = T_des = 30 min, V_max_ge = 120 mmol/h."
    ),
    "dallaman2007": (
        "Dalla Man C, Rizza RA, Cobelli C. Meal simulation model of the "
        "glucose-insulin system. IEEE Trans Biomed Eng 2007;54(10):1740-1749. "
        "Values taken from a peer-reviewed survey rather than the paper itself."
    ),
    "bergman1981": (
        "Bergman RN, Phillips LS, Cobelli C. Physiologic evaluation of factors "
        "controlling glucose tolerance in man. J Clin Invest 1981;68(6):1456-1467. "
        "NB the published caption writes p3*I(t), not the modern p3*(I - I_b)."
    ),
}


@dataclass(frozen=True)
class Bound:
    """An admissible interval with its provenance and confidence."""

    low: float
    high: float
    unit: str
    source: str
    confidence: str = PROVISIONAL
    note: str = ""

    @property
    def sourced(self) -> bool:
        """Whether this bound may be used to produce reported output."""
        return self.confidence not in _BLOCKING

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"bound low={self.low} must be < high={self.high}")


#: Admissible ranges for the estimated parameters.
#:
#: On ``p1``: it is common in control-oriented work to fix ``p1 = 0`` for type 1
#: diabetes, on the argument that without endogenous insulin there is no
#: glucose-mediated self-regulation. **That is a modelling simplification, not an
#: empirical finding.** Ward et al. 1991 measured glucose effectiveness directly in
#: IDDM subjects and found ``S_G = 1.0-1.6e-2 /min`` -- reduced relative to healthy
#: controls but clearly non-zero. The bound therefore admits zero (so the
#: simplification remains reachable and can be ablated) while the population mean
#: is set to the measured value.
BOUNDS: dict[str, Bound] = {
    "p1": Bound(
        0.0,
        0.030,
        "1/min",
        SOURCES["ward1991"],
        VERIFIED_PRIMARY,
        "S_G in IDDM is reduced but non-zero; zero admitted for the ablation",
    ),
    "p2": Bound(
        0.005,
        0.10,
        "1/min",
        SOURCES["bergman1981"],
        SECOND_HAND,
        "range spans reported minimal-model fits; not a single published interval",
    ),
    "p3": Bound(
        1e-6,
        3e-5,
        "mL/(uU*min^2)",
        SOURCES["ward1991"],
        VERIFIED_PRIMARY,
        "derived as p3 = S_I * p2 from the measured IDDM S_I",
    ),
    "n": Bound(
        0.08,
        0.25,
        "1/min",
        SOURCES["hovorka2004"],
        VERIFIED_PRIMARY,
        "insulin elimination k_e = 0.138 /min",
    ),
    "V_G_per_kg": Bound(
        1.4, 2.4, "dL/kg", SOURCES["dallaman2007"], SECOND_HAND
    ),
    "V_I_per_kg": Bound(
        0.08, 0.18, "L/kg", SOURCES["hovorka2004"], VERIFIED_PRIMARY, "V_I = 0.12 L/kg"
    ),
    "tmax_I": Bound(
        30.0,
        90.0,
        "min",
        SOURCES["hovorka2004"],
        VERIFIED_PRIMARY,
        "t_max,I = 55 min for a rapid-acting analogue",
    ),
    "k_gri": Bound(
        0.008,
        0.10,
        "1/min",
        SOURCES["dallaman2007"],
        SECOND_HAND,
        "first-order surrogate for Lehmann-Deutsch trapezoidal gastric emptying",
    ),
    "k_abs": Bound(
        0.005,
        0.10,
        "1/min",
        SOURCES["lehmann1992"],
        VERIFIED_PRIMARY,
        "k_gabs = 1 /h = 0.0167 /min",
    ),
    "f": Bound(0.70, 1.00, "dimensionless", SOURCES["dallaman2007"], SECOND_HAND),
}

#: Insulin sensitivity measured in IDDM subjects, ``S_I = p3 / p2``
#: [mL/(uU*min)]. Ward et al. 1991: 2.5 +/- 0.6e-4. Used to set the ``p3``
#: population mean and as the prior centre for the estimated value.
S_I_IDDM_MEAN = 2.5e-4
S_I_IDDM_SD = 0.6e-4

#: Population means for the parameter prior and the warmup phase during which the
#: encoder output is ignored. Must lie inside ``BOUNDS``.
POPULATION_MEANS: dict[str, float] = {
    # Ward et al. 1991 IDDM S_G midpoint of 1.0-1.6e-2 /min.
    "p1": 0.013,
    "p2": 0.025,
    # p3 = S_I * p2, so that S_I matches the measured IDDM value by construction.
    "p3": S_I_IDDM_MEAN * 0.025,
    # Hovorka 2004 insulin elimination rate.
    "n": 0.138,
    "V_G_per_kg": 1.88,
    "V_I_per_kg": 0.12,
    "tmax_I": 55.0,
    "k_gri": 0.035,
    # Lehmann & Deutsch 1992, converted from 1 /h.
    "k_abs": 0.0167,
    "f": 0.90,
}

#: The parameters the encoder estimates, in a fixed order. Fixing the order
#: matters: the encoder emits a flat vector and a silent reordering would
#: reassign every parameter without any error being raised.
ESTIMATED: tuple[str, ...] = (
    "p1",
    "p2",
    "p3",
    "n",
    "V_G_per_kg",
    "V_I_per_kg",
    "tmax_I",
    "k_gri",
    "k_abs",
    "f",
)

N_ESTIMATED: int = len(ESTIMATED)


def assert_bounds_sourced() -> None:
    """Fail loudly if any bound is unsourced.

    Called from the reporting entry point. This is the guard that stops a
    development placeholder reaching a paper table. ``SECOND-HAND`` bounds pass but
    are surfaced by :func:`second_hand_bounds` for disclosure.
    """
    unsourced = sorted(name for name, bound in BOUNDS.items() if not bound.sourced)
    if unsourced:
        raise RuntimeError(
            "Refusing to produce reportable output: the following parameter "
            f"bounds are still {PROVISIONAL}/{UNVERIFIED}: {unsourced}. "
            "Resolve them against primary sources and record the citation in "
            "docs/CITATIONS_methods.md before generating results."
        )


def second_hand_bounds() -> dict[str, str]:
    """Bounds resting on a secondary source, for disclosure in the manifest.

    These do not block a run, but a paper that depends on them should say so
    rather than implying every parameter range was read from its originating
    paper.
    """
    return {
        name: bound.note or bound.source
        for name, bound in BOUNDS.items()
        if bound.confidence == SECOND_HAND
    }


def provenance_table() -> list[dict[str, object]]:
    """Every bound with its range, unit, confidence, and source.

    Emitted alongside results so the methods section can be generated rather than
    hand-typed.
    """
    return [
        {
            "parameter": name,
            "low": bound.low,
            "high": bound.high,
            "unit": bound.unit,
            "population_mean": POPULATION_MEANS[name],
            "confidence": bound.confidence,
            "source": bound.source,
            "note": bound.note,
        }
        for name, bound in BOUNDS.items()
    ]


# --------------------------------------------------------------------------- #
# Reparameterisation
# --------------------------------------------------------------------------- #


def unconstrained_to_params(
    raw: Tensor, *, names: tuple[str, ...] = ESTIMATED
) -> dict[str, Tensor]:
    """Map an unconstrained network output into the admissible ranges.

    ``raw`` has shape ``(..., len(names))``. Each column is squashed with a
    sigmoid and affinely mapped onto its interval, so the result can never leave
    the range regardless of what the network emits.

    A zero input maps to the interval midpoint, which keeps initialisation
    physiologically sensible.
    """
    if raw.shape[-1] != len(names):
        raise ValueError(
            f"expected last dim {len(names)} (order: {names}), got {raw.shape[-1]}"
        )
    squashed = torch.sigmoid(raw)
    out: dict[str, Tensor] = {}
    for index, name in enumerate(names):
        bound = BOUNDS[name]
        out[name] = bound.low + (bound.high - bound.low) * squashed[..., index]
    return out


def params_to_unconstrained(
    values: dict[str, float], *, names: tuple[str, ...] = ESTIMATED
) -> Tensor:
    """Inverse of :func:`unconstrained_to_params`, for initialising a bias.

    Used to start the parameter head at the population mean rather than at an
    interval midpoint that may be far from physiological.
    """
    raw = torch.empty(len(names))
    for index, name in enumerate(names):
        bound = BOUNDS[name]
        span = bound.high - bound.low
        fraction = (values[name] - bound.low) / span
        # Clamp strictly inside (0, 1) so logit stays finite; p1's population
        # mean sits exactly on its lower bound.
        fraction = min(max(fraction, 1e-4), 1 - 1e-4)
        raw[index] = torch.logit(torch.tensor(fraction))
    return raw


def population_unconstrained() -> Tensor:
    """The unconstrained vector corresponding to :data:`POPULATION_MEANS`."""
    return params_to_unconstrained(POPULATION_MEANS)


# --------------------------------------------------------------------------- #
# Resolved parameter set
# --------------------------------------------------------------------------- #


@dataclass
class PatientParams:
    """A batched, resolved parameter set.

    Every field is a tensor broadcastable to shape ``(B,)``. Volumes are stored
    already multiplied by body weight, so downstream code never has to remember
    whether a volume is per-kg or absolute.
    """

    p1: Tensor
    p2: Tensor
    p3: Tensor
    n: Tensor
    V_G: Tensor  # dL, absolute
    V_I: Tensor  # L, absolute
    tmax_I: Tensor
    k_gri: Tensor
    k_abs: Tensor
    f: Tensor
    G_b: Tensor
    I_b: Tensor

    #: Fields that are volumes-per-kg in the estimated vector and must be
    #: scaled by body weight to become absolute.
    _PER_KG: ClassVar[dict[str, str]] = {"V_G_per_kg": "V_G", "V_I_per_kg": "V_I"}

    @classmethod
    def from_estimated(
        cls,
        estimated: dict[str, Tensor],
        *,
        body_weight_kg: Tensor,
        G_b: Tensor,
        I_b: Tensor,
    ) -> "PatientParams":
        """Assemble from the encoder output plus per-subject context.

        ``G_b`` and ``I_b`` are not estimated by the network. They are observable
        summaries of the subject's own record (fasting-window CGM median, and
        basal insulin rate translated to a steady-state concentration), so
        estimating them would be inventing a latent for something measured.
        """
        resolved: dict[str, Tensor] = {}
        for name, value in estimated.items():
            resolved[cls._PER_KG.get(name, name)] = value
        resolved["V_G"] = resolved["V_G"] * body_weight_kg
        resolved["V_I"] = resolved["V_I"] * body_weight_kg
        resolved["G_b"] = G_b
        resolved["I_b"] = I_b

        expected = {f.name for f in fields(cls)}
        missing = expected - set(resolved)
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")
        return cls(**{k: resolved[k] for k in expected})

    @property
    def S_I(self) -> Tensor:
        """Insulin sensitivity ``p3 / p2`` [mL/(uU*min)].

        This is the patient-specific quantity the study reports. Its validation
        (correlation with the subject's own insulin requirement, test-retest
        stability) lives in ``twin.eval.sensitivity``.
        """
        return self.p3 / self.p2

    def batch_size(self) -> int:
        return int(self.p2.reshape(-1).shape[0])

    def detach(self) -> "PatientParams":
        return PatientParams(
            **{f.name: getattr(self, f.name).detach() for f in fields(self)}
        )


def population_params(
    *,
    batch_size: int = 1,
    body_weight_kg: float = 70.0,
    G_b: float = 120.0,
    I_b: float = 10.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> PatientParams:
    """A batch of population-mean parameters.

    Used for the parameter-warmup phase, for the fixed-parameter ablation (A4),
    and throughout the test suite. ``float64`` by default because the
    verification tests compare against analytic solutions.
    """
    estimated = {
        name: torch.full((batch_size,), POPULATION_MEANS[name], device=device, dtype=dtype)
        for name in ESTIMATED
    }
    return PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.full((batch_size,), body_weight_kg, device=device, dtype=dtype),
        G_b=torch.full((batch_size,), G_b, device=device, dtype=dtype),
        I_b=torch.full((batch_size,), I_b, device=device, dtype=dtype),
    )


__all__ = [
    "BOUNDS",
    "ESTIMATED",
    "N_ESTIMATED",
    "POPULATION_MEANS",
    "PROVISIONAL",
    "Bound",
    "PatientParams",
    "assert_bounds_sourced",
    "params_to_unconstrained",
    "population_params",
    "population_unconstrained",
    "unconstrained_to_params",
]
