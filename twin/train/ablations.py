"""The ablation matrix.

Each configuration isolates one claim. Together they answer the question the
original draft asserted without evidence: **does the physics do anything?**

======  ===========================================  =========================================
id      configuration                                isolates
======  ===========================================  =========================================
A0      Transformer only, no physics                 the data-driven baseline
A1      + collocation residual, fixed lambda = 0.1   the naive PINN the original draft claimed
A2      + learned log-variance weighting             adaptive weighting
A3      + mechanistic prior and residual correction  hybrid versus pure penalty
A4      A3 with population-fixed parameters          the value of per-patient estimation
A5      A3 + simulation pretraining                  transfer value
A6      A3 with the legacy hand-rolled IOB/COB       the value of mechanistic features
======  ===========================================  =========================================

Every configuration shares the identical corpus, splits, scaler, seed, and epoch
budget, so the only difference between two rows is the thing being ablated.

A1 exists specifically so the paper can state what the previous approach *would*
have produced had it actually been trained. The legacy code declared a physics loss
with ``lambda = 0.1`` and then passed ``use_pinn=False`` in every script that
produced a checkpoint, so its claimed method was never run at all.

If the physics terms do not improve on A0, that is the finding and it is reported.
A negative result about physics-informed forecasting on real CGM data is publishable
and useful; a fabricated positive one is neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from twin.config import Config


class AblationError(ValueError):
    """Raised for an unknown or not-yet-runnable ablation."""


@dataclass(frozen=True)
class Ablation:
    """One ablation: a label, an override set, and what it isolates."""

    id: str
    label: str
    isolates: str
    overrides: dict[str, object] = field(default_factory=dict)
    #: Set when the configuration needs machinery that is not wired up yet, so it
    #: fails loudly instead of silently running as something else.
    requires: str | None = None

    def apply(self, base: Config) -> Config:
        """Return a copy of ``base`` with this ablation's overrides applied."""
        if self.requires is not None:
            raise AblationError(
                f"ablation {self.id} ({self.label}) requires {self.requires}, "
                "which is not implemented. It is declared here so it cannot be "
                "quietly dropped from the matrix."
            )
        raw = base.to_dict()
        for dotted, value in self.overrides.items():
            section, _, field_name = dotted.partition(".")
            if not field_name or section not in raw:
                raise AblationError(f"{self.id}: bad override key {dotted!r}")
            raw[section][field_name] = value
        raw["run"]["name"] = f"{base.run.name}-{self.id}"
        return Config.from_dict(raw)


ABLATIONS: tuple[Ablation, ...] = (
    Ablation(
        id="A0",
        label="Transformer only, no physics",
        isolates="the data-driven baseline every physics claim is measured against",
        overrides={
            "physics.enabled": False,
            "model.hybrid_residual": False,
            "model.per_patient_params": False,
        },
    ),
    Ablation(
        id="A1",
        label="naive PINN: collocation residual, fixed lambda = 0.1",
        isolates=(
            "what the original draft's declared method would have produced had it "
            "ever been trained"
        ),
        overrides={
            "physics.enabled": True,
            "physics.weighting": "fixed",
            "physics.lambda_phys": 0.1,
            "model.hybrid_residual": False,
            "model.per_patient_params": False,
        },
    ),
    Ablation(
        id="A2",
        label="residual with learned log-variance weighting",
        isolates="whether adaptive weighting beats a hand-picked constant",
        overrides={
            "physics.enabled": True,
            "physics.weighting": "kendall",
            "model.hybrid_residual": False,
            "model.per_patient_params": True,
        },
    ),
    Ablation(
        id="A3",
        label="hybrid: mechanistic prior plus learned residual correction",
        isolates="a physics prior the network corrects, versus a penalty term alone",
        overrides={
            "physics.enabled": True,
            "physics.weighting": "kendall",
            "model.hybrid_residual": True,
            "model.per_patient_params": True,
        },
    ),
    Ablation(
        id="A4",
        label="hybrid with population-fixed parameters",
        isolates="how much the per-patient parameter estimate is actually worth",
        overrides={
            "physics.enabled": True,
            "physics.weighting": "kendall",
            "model.hybrid_residual": True,
            "model.per_patient_params": False,
        },
    ),
    Ablation(
        id="A5",
        label="hybrid plus UVA/Padova simulation pretraining",
        isolates="whether synthetic pretraining transfers to real CGM",
        overrides={},
        requires=(
            "the simulation data pipeline, including the fix to the meal scheduler "
            "that made the existing synthetic corpus average ~0.6 meals per day "
            "instead of three"
        ),
    ),
    Ablation(
        id="A6",
        label="hybrid with the legacy hand-rolled IOB/COB kernels",
        isolates=(
            "the value of deriving IOB and COB from the same mechanistic model as "
            "the physics loss"
        ),
        overrides={},
        requires=(
            "a feature variant reproducing the legacy kernels (time-reversed insulin "
            "activity curve, unnormalised exponential COB) as an explicitly "
            "labelled comparison"
        ),
    ),
)

BY_ID: dict[str, Ablation] = {ablation.id: ablation for ablation in ABLATIONS}

#: Ablations that can be run today.
RUNNABLE: tuple[str, ...] = tuple(
    ablation.id for ablation in ABLATIONS if ablation.requires is None
)


def resolve(ids: list[str] | None) -> list[Ablation]:
    """Resolve ablation ids, defaulting to everything runnable."""
    if not ids:
        return [BY_ID[identifier] for identifier in RUNNABLE]
    unknown = [identifier for identifier in ids if identifier not in BY_ID]
    if unknown:
        raise AblationError(f"unknown ablation(s) {unknown}; known: {sorted(BY_ID)}")
    return [BY_ID[identifier] for identifier in ids]


def matrix_table() -> pd.DataFrame:
    """The matrix as a table, for the paper's ablation section."""
    return pd.DataFrame(
        [
            {
                "id": ablation.id,
                "configuration": ablation.label,
                "isolates": ablation.isolates,
                "runnable": ablation.requires is None,
                "blocked_on": ablation.requires or "",
                "overrides": ", ".join(
                    f"{key}={value}" for key, value in sorted(ablation.overrides.items())
                ),
            }
            for ablation in ABLATIONS
        ]
    )


__all__ = [
    "ABLATIONS",
    "BY_ID",
    "RUNNABLE",
    "Ablation",
    "AblationError",
    "matrix_table",
    "resolve",
]
