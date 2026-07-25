"""Loss terms and adaptive weighting.

Composition::

    L = L_data
      + w_phys  * L_residual
      + w_prior * L_param_prior
      + w_temporal * L_temporal_consistency

Deliberately absent: any asymmetric hypoglycaemia/hyperglycaemia penalty. The
legacy ``clinical_penalty_loss`` weighted a missed *hyper* at 6.0 and a missed
*hypo* at 2.0 -- backwards, since hypoglycaemia is the acute risk. More
fundamentally, *any* asymmetric training loss inflates error-grid zone A by
construction, so a clinical-safety table computed afterwards is no longer
independent evidence. Safety is measured and reported; it is not optimised into
the objective.

Weighting
---------
``kendall`` learns a log-variance per term (Kendall, Gal & Cipolla 2018):

.. math::

    L = \\sum_i \\frac{1}{2\\sigma_i^2} L_i + \\frac{1}{2}\\log \\sigma_i^2

parameterised as ``s_i = log sigma_i^2`` for stability, giving
``0.5 * exp(-s_i) * L_i + 0.5 * s_i``. The log term is what stops the optimiser
from driving every weight to zero.

A fixed weight is available for ablation A1 -- the naive PINN the original paper
claimed -- so the value of adaptive weighting is measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from twin.config import Config
from twin.physio import BOUNDS, ESTIMATED, POPULATION_MEANS, PatientParams


@dataclass
class LossBreakdown:
    """Every term, for logging and for the ablation tables."""

    total: Tensor
    data: Tensor
    physics: Tensor
    parameter_prior: Tensor
    temporal: Tensor
    weights: dict[str, float] = field(default_factory=dict)

    def items(self) -> dict[str, float]:
        out = {
            "loss": float(self.total.detach()),
            "loss_data": float(self.data.detach()),
            "loss_physics": float(self.physics.detach()),
            "loss_param_prior": float(self.parameter_prior.detach()),
            "loss_temporal": float(self.temporal.detach()),
        }
        out.update({f"weight_{key}": value for key, value in self.weights.items()})
        return out


def horizon_weights(n_horizons: int, device: torch.device | str = "cpu") -> Tensor:
    """Equal weight per horizon.

    Deliberately uniform. Down-weighting the long horizons would improve the
    headline 30-minute number at the cost of the long-horizon table, which is a
    presentational choice masquerading as an optimisation one.
    """
    return torch.ones(n_horizons, device=device)


def data_loss(
    predictions: Tensor, targets: Tensor, *, delta: float = 10.0, kind: str = "huber"
) -> Tensor:
    """Per-horizon-averaged regression loss.

    Huber by default: CGM contains sensor artefacts and compression-low episodes, and
    a squared loss lets a handful of them dominate the gradient. ``delta`` is in
    mg/dL, so it is directly interpretable as the error above which the penalty
    becomes linear.
    """
    if kind == "huber":
        elementwise = nn.functional.huber_loss(
            predictions, targets, reduction="none", delta=delta
        )
    elif kind == "mse":
        elementwise = (predictions - targets) ** 2
    elif kind == "mae":
        elementwise = (predictions - targets).abs()
    else:
        raise ValueError(f"unknown data loss {kind!r}")
    weights = horizon_weights(predictions.shape[-1], predictions.device)
    return (elementwise * weights).mean()


def physics_loss(residual: Tensor) -> Tensor:
    """Mean squared Bergman residual at the collocation points.

    The residual arrives already non-dimensionalised by ``G_b / T``, so this term is
    O(1) and its weight is comparable across subjects. The legacy residual mixed
    30-minute finite differences with per-minute rate constants and was never
    scaled, which made its 0.1 multiplier meaningless.
    """
    return residual.pow(2).mean()


def parameter_prior_loss(params: PatientParams) -> Tensor:
    """Keep estimated parameters near population values, in normalised units.

    Each parameter is scaled by the width of its admissible interval before being
    compared, so a parameter spanning 1e-6..3e-5 is not ignored relative to one
    spanning 30..90. Without this the optimiser can park a weakly-identified
    parameter at an extreme that happens to fit, and the resulting insulin
    sensitivity would not be a physiological estimate.
    """
    total = torch.zeros((), device=params.p2.device, dtype=params.p2.dtype)
    for name in ESTIMATED:
        bound = BOUNDS[name]
        width = bound.high - bound.low
        target = POPULATION_MEANS[name]
        if name == "V_G_per_kg":
            value = params.V_G / params.p2.new_tensor(70.0)
        elif name == "V_I_per_kg":
            value = params.V_I / params.p2.new_tensor(70.0)
        else:
            value = getattr(params, name)
        total = total + (((value - target) / width) ** 2).mean()
    return total / len(ESTIMATED)


def temporal_consistency_loss(
    params: PatientParams, subject_index: Tensor
) -> Tensor:
    """Penalise within-subject variation of the estimated parameters in a batch.

    Physiological parameters drift over days, not minutes. Windows from the same
    subject in one batch should therefore yield near-identical estimates. Without
    this the parameter head can absorb per-window prediction error into the
    parameters, which would produce an excellent fit and a meaningless insulin
    sensitivity -- the estimate has to be stable to be a measurement of anything.
    """
    device = params.p2.device
    dtype = params.p2.dtype
    total = torch.zeros((), device=device, dtype=dtype)
    count = 0
    for subject in torch.unique(subject_index):
        mask = subject_index == subject
        if int(mask.sum()) < 2:
            continue
        for name in ESTIMATED:
            bound = BOUNDS[name]
            width = bound.high - bound.low
            if name == "V_G_per_kg":
                value = params.V_G[mask] / params.p2.new_tensor(70.0)
            elif name == "V_I_per_kg":
                value = params.V_I[mask] / params.p2.new_tensor(70.0)
            else:
                value = getattr(params, name)[mask]
            total = total + (value.std(unbiased=False) / width) ** 2
        count += 1
    return total / max(count * len(ESTIMATED), 1)


class AdaptiveWeights(nn.Module):
    """Learned log-variance weights (Kendall et al. 2018), or fixed for ablation."""

    TERMS = ("data", "physics")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.scheme = config.physics.weighting
        self.fixed_lambda = config.physics.lambda_phys
        if self.scheme == "kendall":
            self.log_variance = nn.Parameter(torch.zeros(len(self.TERMS)))
        else:
            self.register_buffer("log_variance", torch.zeros(len(self.TERMS)))

    def combine(self, data: Tensor, physics: Tensor, *, ramp: float) -> tuple[Tensor, dict[str, float]]:
        """Combine the two main terms, applying the curriculum ramp to physics."""
        if self.scheme == "kendall":
            terms = torch.stack([data, physics * ramp])
            weighted = 0.5 * torch.exp(-self.log_variance) * terms + 0.5 * self.log_variance
            total = weighted.sum()
            weights = {
                name: float(0.5 * torch.exp(-self.log_variance[index]).detach())
                for index, name in enumerate(self.TERMS)
            }
            return total, weights
        if self.scheme == "fixed":
            weight = self.fixed_lambda * ramp
            return data + weight * physics, {"data": 1.0, "physics": weight}
        raise ValueError(f"unsupported weighting scheme {self.scheme!r}")


def physics_ramp(epoch: int, config: Config) -> float:
    """Cosine ramp of the physics weight over a data-first curriculum.

    The residual is a constraint on a trajectory that is initially meaningless, so
    enforcing it from step one fights the data term while the encoder is still
    random. The ramp lets the forecast become approximately right first, then tightens
    physical consistency.
    """
    start = config.physics.ramp_start_epoch
    end = config.physics.ramp_end_epoch
    if not config.physics.enabled:
        return 0.0
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return 1.0
    import math

    progress = (epoch - start) / max(end - start, 1)
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def compute_loss(
    output,
    batch: dict[str, Tensor],
    weights: AdaptiveWeights,
    config: Config,
    *,
    epoch: int,
) -> LossBreakdown:
    """Assemble the full objective for one batch."""
    ramp = physics_ramp(epoch, config)
    data = data_loss(
        output.horizons,
        batch["targets"],
        delta=config.train.huber_delta,
        kind=config.train.data_loss,
    )
    physics = (
        physics_loss(output.residual)
        if config.physics.enabled
        else torch.zeros((), device=data.device, dtype=data.dtype)
    )

    total, weight_values = weights.combine(data, physics, ramp=ramp)

    prior = torch.zeros((), device=data.device, dtype=data.dtype)
    temporal = torch.zeros((), device=data.device, dtype=data.dtype)
    if config.model.per_patient_params:
        prior = parameter_prior_loss(output.params).to(data.dtype)
        temporal = temporal_consistency_loss(output.params, batch["subject_index"]).to(data.dtype)
        total = (
            total
            + config.physics.lambda_param_prior * prior
            + config.physics.lambda_temporal_consistency * temporal
        )

    weight_values["ramp"] = ramp
    return LossBreakdown(
        total=total,
        data=data,
        physics=physics,
        parameter_prior=prior,
        temporal=temporal,
        weights=weight_values,
    )


__all__ = [
    "AdaptiveWeights",
    "LossBreakdown",
    "compute_loss",
    "data_loss",
    "horizon_weights",
    "parameter_prior_loss",
    "physics_loss",
    "physics_ramp",
    "temporal_consistency_loss",
]
