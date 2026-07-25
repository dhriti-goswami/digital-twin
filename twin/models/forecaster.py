"""The physics-guided glucose forecaster.

Architecture
------------
::

    features (B, 24, 35)
        |
        v
    Transformer encoder  ->  context (B, d_model)
        |                        |
        |                        +--> parameter head  -> theta_p  (Bergman parameters)
        |                        |
        |                        +--> spline head     -> c        (B-spline coefficients)
        v
    G_theta(t) = G_0 + sum_k c_k (B_k(t) - B_k(0))        [+ mechanistic prior]
    dG_theta/dt analytic from the derivative basis

Three properties distinguish this from the legacy ``PhysicsInformedLoss``:

1. **The physics is real.** All three Bergman states are integrated. ``X(t)`` comes
   from advancing the coupled insulin cascade with the *estimated* parameters over a
   12-hour burn-in of the subject's actual insulin and meal history, not from an
   algebraic stand-in.
2. **``dG/dt`` is exact.** The head emits spline coefficients, so the derivative is
   an analytic linear map. The legacy residual finite-differenced across 30-minute
   steps against per-minute rate constants.
3. **The reported horizons are the constrained function.** ``G(30), G(60), G(90),
   G(120)`` are evaluations of the same continuous trajectory the residual is
   evaluated on, so there is no train/report mismatch.

Hybrid mode
-----------
With ``hybrid_residual``, the mechanistic Bergman forecast is computed in closed
form and the network learns only the discrepancy from it. The title says
"physics-*guided*", and a prior the network corrects is both more accurate and
easier to defend than asking a penalty term to impose physiology on an otherwise
unconstrained output. Ablation A3 versus A1 quantifies the choice rather than
assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from twin.config import Config
from twin.data.dataset import PHYSICS_BURNIN_STEPS
from twin.physio import (
    N_ESTIMATED,
    PatientParams,
    advance_compartments,
    basal_steady_state,
    glucose_residual,
    integrate_glucose,
    population_params,
    population_unconstrained,
    residual_scale,
    simulate_compartments,
    unconstrained_to_params,
)
from twin.physio.spline import SplineEvaluator, SplineGrid

#: Initial logit for the mechanistic-prior gate. sigmoid(-2.2) ~ 0.10, so the
#: forecast starts close to persistence while leaving the physics path active and
#: differentiable from step one.
_PRIOR_LOGIT_INIT = -2.2


class PositionalEncoding(nn.Module):
    """Standard sinusoidal position encoding."""

    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.shape[1]]


class AttentionPooling(nn.Module):
    """Learned attention pooling over the sequence.

    Preferred to the mean pooling of the legacy encoder, which weighted a reading
    two hours old identically to the most recent one -- in a forecasting problem
    where recency dominates.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return (weights * x).sum(dim=1)


class GlucoseEncoder(nn.Module):
    """Transformer encoder over the feature window."""

    def __init__(self, n_features: int, config: Config) -> None:
        super().__init__()
        model = config.model
        self.embedding = nn.Linear(n_features, model.d_model)
        self.positional = PositionalEncoding(model.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=model.d_model,
            nhead=model.n_heads,
            dim_feedforward=model.d_ff,
            dropout=model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=model.n_layers)
        self.norm = nn.LayerNorm(model.d_model)

        if model.pooling == "attention":
            self.pool: nn.Module = AttentionPooling(model.d_model)
        elif model.pooling == "mean":
            self.pool = _Mean()
        elif model.pooling == "last":
            self.pool = _Last()
        else:
            raise ValueError(f"unknown pooling {model.pooling!r}")

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.positional(self.embedding(features))
        # No causal mask: every input timestep is history relative to the forecast,
        # so attending across the whole window leaks nothing.
        hidden = self.encoder(hidden)
        return self.norm(self.pool(hidden))


class _Mean(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)


class _Last(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x[:, -1]


class ParameterHead(nn.Module):
    """Amortised per-patient Bergman parameter estimation.

    Emits an unconstrained vector that
    :func:`~twin.physio.params.unconstrained_to_params` maps into published
    physiological ranges with a scaled sigmoid, so no output can leave the
    admissible interval regardless of what the network produces.

    The output bias is initialised so the head starts at the population means, and
    the final weight is *down-scaled* rather than zeroed. Zeroing it would make the
    head's output constant at initialisation, which is desirable -- but it also makes
    ``dL/d(hidden)`` exactly zero, so no gradient reaches the encoder on the first
    step. Scaling down keeps the head effectively at the population value while
    leaving the gradient path intact.
    """

    #: Down-scaling applied to the final layer's default initialisation.
    INIT_SCALE = 1e-2

    def __init__(self, d_model: int, *, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_ESTIMATED),
        )
        with torch.no_grad():
            final = self.net[-1]
            final.weight.mul_(self.INIT_SCALE)
            final.bias.copy_(population_unconstrained())

    def forward(self, context: Tensor) -> Tensor:
        return self.net(context)


class QuantileHead(nn.Module):
    """Median spline coefficients plus a non-crossing quantile band at each horizon.

    The band is emitted **per horizon in glucose units**, not as an offset to the
    spline coefficients. That is forced by the basis: cubic B-splines form a partition
    of unity, so the anchored trajectory

    .. math::

        G(t) = G_0 + \sum_k c_k\,(B_k(t) - B_k(0))

    is *invariant* to adding a constant to every coefficient --
    ``sum_k delta (B_k(t) - B_k(0)) = delta (1 - 1) = 0``. A first attempt put the
    quantile offsets in coefficient space and produced a band of exactly zero width;
    the property that makes the anchoring exact also makes uniform coefficient shifts
    invisible.

    Non-crossing is guaranteed by construction: offsets pass through ``softplus`` and
    accumulate outward from the median, so no network output can invert the order. A
    crossed quantile would make the lower band meaningless exactly where it is needed.

    The band is zero-width at ``t = 0`` by design -- the anchor is an observation, not
    a prediction -- and free to widen with horizon.

    Why quantiles at all: a point forecast trained on a mean-seeking loss regresses
    toward the centre, measured here at +7.29 mg/dL below 70 mg/dL even with the
    physics removed entirely, so it understates hypoglycaemia risk by construction.
    The remedy is distributional -- predict the lower tail and alarm on it -- rather
    than tilting the objective, which would inflate error-grid zone A and make the
    safety table depend on the loss.
    """

    INIT_SCALE = 1e-2
    #: Initial half-band width [mg/dL] between adjacent quantiles.
    INIT_SPREAD = 8.0

    def __init__(
        self,
        d_model: int,
        n_basis: int,
        quantiles: tuple[float, ...],
        n_horizons: int,
        *,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.quantiles = quantiles
        self.n_basis = n_basis
        self.n_horizons = n_horizons
        self.median_index = quantiles.index(0.5)
        self.n_offsets = len(quantiles) - 1

        self.median = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_basis)
        )
        self.band = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_offsets * n_horizons),
        )
        with torch.no_grad():
            self.median[-1].weight.mul_(self.INIT_SCALE)
            self.median[-1].bias.zero_()
            self.band[-1].weight.mul_(self.INIT_SCALE)
            # softplus(x) = INIT_SPREAD  ->  x = log(exp(spread) - 1)
            self.band[-1].bias.fill_(math.log(math.expm1(self.INIT_SPREAD)))

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(median_coefficients, band_offsets)``.

        ``median_coefficients`` is ``(B, n_basis)``; ``band_offsets`` is
        ``(B, n_quantiles, n_horizons)`` in mg/dL relative to the median, ascending
        across the quantile axis with an exact zero at the median.
        """
        coefficients = self.median(context)
        offsets = nn.functional.softplus(
            self.band(context).reshape(-1, self.n_offsets, self.n_horizons)
        )

        below = self.median_index
        zero = offsets.new_zeros(offsets.shape[0], 1, self.n_horizons)
        levels: list[Tensor] = [zero]
        current = zero
        for step in range(below):
            current = current - offsets[:, below - 1 - step : below - step]
            levels.insert(0, current)
        current = zero
        for step in range(self.n_offsets - below):
            current = current + offsets[:, below + step : below + step + 1]
            levels.append(current)
        return coefficients, torch.cat(levels, dim=1)


class SplineHead(nn.Module):
    """Emits cubic B-spline coefficients for the forecast trajectory.

    The final layer is down-scaled and its bias zeroed, so the model's initial
    prediction is very nearly a flat trajectory at the last observed value -- the
    persistence baseline, which on this data is a strong comparator. Training starts
    there rather than from noise.

    Down-scaled rather than exactly zeroed: a zero final weight makes
    ``dL/d(hidden)`` vanish, so the encoder would receive no gradient at all on the
    first optimiser step.
    """

    INIT_SCALE = 1e-2

    def __init__(self, d_model: int, n_basis: int, *, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_basis),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(self.INIT_SCALE)
            self.net[-1].bias.zero_()

    def forward(self, context: Tensor) -> Tensor:
        return self.net(context)


@dataclass
class ForecastOutput:
    """Everything a loss or an evaluation might need from one forward pass."""

    horizons: Tensor  # (B, H) mg/dL -- the reported predictions
    collocation: Tensor  # (B, n_coll) mg/dL
    derivative: Tensor  # (B, n_coll) mg/dL/min
    residual: Tensor  # (B, n_coll) dimensionless
    params: PatientParams
    coefficients: Tensor
    mechanistic: Tensor | None = None  # (B, n_coll) the prior, when hybrid
    insulin_action: Tensor | None = None  # X(t) at collocation points
    appearance: Tensor | None = None  # Ra(t) at collocation points
    #: Learned weight on the mechanistic prior, in (0, 1). Reported.
    prior_gate: float | None = None
    #: Forecast quantiles at each horizon, ``(B, n_quantiles, n_horizons)``,
    #: guaranteed non-crossing. The median column equals ``horizons``.
    quantile_horizons: Tensor | None = None
    quantiles: tuple[float, ...] = ()

    @property
    def insulin_sensitivity(self) -> Tensor:
        """``S_I = p3 / p2`` per sample -- the reported patient-specific parameter."""
        return self.params.S_I


class PhysicsGuidedForecaster(nn.Module):
    """Transformer encoder, Bergman-constrained spline head, patient parameters."""

    def __init__(self, n_features: int, config: Config) -> None:
        super().__init__()
        self.config = config
        self.grid_minutes = config.data.grid_minutes
        self.encoder = GlucoseEncoder(n_features, config)
        self.quantiles = config.train.quantiles
        self.median_index = config.median_index
        self.spline_head = QuantileHead(
            config.model.d_model,
            config.model.n_spline_basis,
            self.quantiles,
            len(config.data.horizons_min),
        )
        self.parameter_head = (
            ParameterHead(config.model.d_model) if config.model.per_patient_params else None
        )
        self.spline = SplineEvaluator(
            SplineGrid.build(
                horizons_min=config.data.horizons_min,
                n_collocation=config.physics.n_collocation,
            ),
            n_basis=config.model.n_spline_basis,
        )
        self.hybrid = config.model.hybrid_residual
        # Learned trust in the mechanistic prior, as a logit. Initialised low so the
        # model starts near persistence -- the strongest naive comparator on this data
        # -- rather than at an unfitted population-parameter forecast, which begins
        # substantially worse. The converged value is reported: it quantifies how much
        # of the forecast the physics actually carries, which is exactly the question
        # the ablation asks.
        self.prior_logit = nn.Parameter(torch.tensor(_PRIOR_LOGIT_INIT))

    # -- parameters --------------------------------------------------------- #

    def resolve_params(
        self,
        context: Tensor,
        *,
        basal_glucose: Tensor,
        body_weight_kg: Tensor,
        basal_insulin_rate: Tensor,
        use_population: bool = False,
    ) -> PatientParams:
        """Estimate patient parameters, or fall back to population values.

        ``use_population`` covers the warmup epochs -- during which the encoder has
        no signal yet and unconstrained parameter estimates would collapse -- and
        ablation A4, which fixes parameters to test how much the per-patient
        estimate is worth.

        ``I_b`` is not estimated: it is the basal steady-state plasma insulin implied
        by the subject's own basal rate and the resolved parameters, so it is
        self-consistent with the burn-in starting at that steady state.
        """
        batch = context.shape[0]
        dtype = context.dtype
        device = context.device

        if self.parameter_head is None or use_population:
            reference = population_params(
                batch_size=batch, device=device, dtype=dtype
            )
            estimated = {
                "p1": reference.p1,
                "p2": reference.p2,
                "p3": reference.p3,
                "n": reference.n,
                "V_G_per_kg": reference.V_G / 70.0,
                "V_I_per_kg": reference.V_I / 70.0,
                "tmax_I": reference.tmax_I,
                "k_gri": reference.k_gri,
                "k_abs": reference.k_abs,
                "f": reference.f,
            }
        else:
            estimated = unconstrained_to_params(self.parameter_head(context))

        weight = body_weight_kg.to(dtype)
        volume_insulin = estimated["V_I_per_kg"] * weight
        # I_b must be exactly the concentration implied by this basal rate and these
        # parameters, otherwise X is non-zero at basal and glucose drifts away from
        # G_b with no stimulus present. See
        # twin.physio.compartments.basal_insulin_concentration.
        basal_plasma_insulin = (
            (1000.0 / volume_insulin) * basal_insulin_rate.to(dtype) / estimated["n"]
        )

        return PatientParams.from_estimated(
            estimated,
            body_weight_kg=weight,
            G_b=basal_glucose.to(dtype),
            I_b=basal_plasma_insulin,
        )

    # -- mechanistic state -------------------------------------------------- #

    def mechanistic_state(
        self,
        params: PatientParams,
        insulin_rate: Tensor,
        carb_rate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Advance the compartments and return ``X`` and ``Ra`` at collocation times.

        The burn-in is advanced with :func:`advance_compartments`, which returns only
        its final state: the intermediate burn-in states are never used, and a
        step-by-step scan over 144 steps was 73% of the whole forward pass. The
        chunked form is exact, agreeing with the sequential scan to floating-point
        tolerance.

        Only the forecast interval is then integrated step by step and interpolated
        onto the 1-minute collocation grid. The states are smooth at that scale and
        every input discontinuity lands on a grid point, so linear interpolation
        introduces no error at the kinks.
        """
        dtype = params.p2.dtype
        insulin = insulin_rate.to(dtype)
        carbs = carb_rate.to(dtype)
        grid = float(self.grid_minutes)

        # Initial condition is the steady state of the subject's own basal rate,
        # matching the I_b used for the parameters so basal is a true equilibrium.
        basal_rate = (params.I_b * params.V_I * params.n) / 1000.0
        anchor_state = advance_compartments(
            params,
            insulin[:, :PHYSICS_BURNIN_STEPS],
            carbs[:, :PHYSICS_BURNIN_STEPS],
            dt=grid,
            x0=basal_steady_state(params, basal_rate),
        )
        trajectory = simulate_compartments(
            params,
            insulin[:, PHYSICS_BURNIN_STEPS:],
            carbs[:, PHYSICS_BURNIN_STEPS:],
            dt=grid,
            x0=anchor_state,
        )
        states = trajectory.interpolate(self.spline.collocation_min.to(dtype))

        from twin.physio.compartments import IDX_QGUT, IDX_X

        insulin_action = states[..., IDX_X]
        gut = states[..., IDX_QGUT]
        appearance = (
            params.f.unsqueeze(-1) * params.k_abs.unsqueeze(-1) * gut
        ) / params.V_G.unsqueeze(-1)
        return insulin_action, appearance

    # -- forward ------------------------------------------------------------ #

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        use_population_params: bool = False,
    ) -> ForecastOutput:
        features = batch["features"]
        anchor_glucose = batch["anchor_glucose"]
        context = self.encoder(features)

        params = self.resolve_params(
            context,
            basal_glucose=batch["basal_glucose"],
            body_weight_kg=batch["body_weight_kg"],
            # The subject's robust median basal rate, not the rate at one slot: that
            # slot carries basal plus any bolus delivered then, and is zero for an
            # anchor whose burn-in is padded.
            basal_insulin_rate=batch["basal_insulin_rate"],
            use_population=use_population_params,
        )
        insulin_action, appearance = self.mechanistic_state(
            params, batch["insulin_rate"], batch["carb_rate"]
        )

        # The physics constrains the median trajectory: it is the central estimate the
        # Bergman equation describes. The band carries predictive uncertainty, which
        # the ODE says nothing about.
        coefficients, band_offsets = self.spline_head(context)
        dtype = coefficients.dtype
        insulin_action = insulin_action.to(dtype)
        appearance = appearance.to(dtype)

        learned = self.spline.value(coefficients, anchor_glucose, at="collocation")
        learned_derivative = self.spline.derivative(coefficients)
        learned_horizons = self.spline.value(coefficients, anchor_glucose, at="horizon")

        mechanistic: Tensor | None = None
        if self.hybrid:
            # Closed-form Bergman forecast on the collocation grid, used as a prior
            # the network corrects. dt is the collocation spacing, not the data grid.
            collocation = self.spline.collocation_min.to(dtype)
            dt = float(collocation[1] - collocation[0])
            mechanistic = integrate_glucose(
                anchor_glucose.to(dtype),
                insulin_action,
                appearance,
                params,
                dt=dt,
            )
            # Both the prior and the learned spline start at the anchor value, so
            # adding them directly would double-count it. Each contributes only its
            # deviation from the anchor, the prior's scaled by the learned gate.
            gate = torch.sigmoid(self.prior_logit)
            anchor = anchor_glucose.unsqueeze(-1).to(dtype)
            prior_deviation = gate * (mechanistic - anchor)
            collocation_glucose = anchor + prior_deviation + (learned - anchor)
            horizon_indices = [
                int(torch.argmin((collocation - minutes).abs()))
                for minutes in self.spline.horizon_min.to(dtype)
            ]
            horizons = collocation_glucose[:, horizon_indices]
            derivative = learned_derivative + gate * _finite_difference(mechanistic, dt)
        else:
            collocation_glucose = learned
            horizons = learned_horizons
            derivative = learned_derivative

        # The band is centred on the final median forecast, so any prior shift is
        # inherited automatically and the band width is unaffected by it.
        quantile_horizons = horizons.unsqueeze(1) + band_offsets.to(dtype)

        residual = glucose_residual(
            collocation_glucose, derivative, insulin_action, appearance, params
        ) / residual_scale(params).to(dtype)

        return ForecastOutput(
            horizons=horizons,
            collocation=collocation_glucose,
            derivative=derivative,
            residual=residual,
            params=params,
            coefficients=coefficients,
            mechanistic=mechanistic,
            insulin_action=insulin_action,
            appearance=appearance,
            prior_gate=float(torch.sigmoid(self.prior_logit).detach()) if self.hybrid else None,
            quantile_horizons=quantile_horizons,
            quantiles=self.quantiles,
        )


def _finite_difference(values: Tensor, dt: float) -> Tensor:
    """Central-difference derivative of the mechanistic prior.

    The prior comes from a closed-form recursion rather than a differentiable
    functional form, so its derivative is differenced. Second-order accurate on the
    interior with one-sided ends; on a 1-minute grid the error is far below the
    residual's own scale.
    """
    derivative = torch.empty_like(values)
    derivative[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * dt)
    derivative[:, 0] = (values[:, 1] - values[:, 0]) / dt
    derivative[:, -1] = (values[:, -1] - values[:, -2]) / dt
    return derivative


__all__ = [
    "AttentionPooling",
    "QuantileHead",
    "ForecastOutput",
    "GlucoseEncoder",
    "ParameterHead",
    "PhysicsGuidedForecaster",
    "PositionalEncoding",
    "SplineHead",
]
