"""Typed configuration loaded from YAML.

Two properties matter here and neither was present before:

1. **Unknown keys are an error.** A silently-ignored typo in a config file is
   indistinguishable from a flag that does not work, and produces a run whose
   recorded settings do not match what executed.
2. **The resolved config is serialisable.** Whatever actually ran is written
   next to the artifacts, so a table can always be traced back to its settings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

HORIZON_MINUTES: tuple[int, ...] = (30, 60, 90, 120)


class ConfigError(ValueError):
    """Raised for an unknown key, a missing required key, or a bad value."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class RunConfig:
    name: str = "unnamed"
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"  # auto | cuda | cpu
    out_root: str = "artifacts"


@dataclass
class DataConfig:
    dataset: str = "ohio"  # ohio | shanghai | diatrend | sim
    root: str = "OhioT1DM"
    grid_minutes: int = 5
    seq_len: int = 24  # input steps; 24 x 5 min = 2 h context
    horizons_min: tuple[int, ...] = HORIZON_MINUTES
    #: Maximum run of consecutive missing *input* slots that may be linearly
    #: interpolated. Targets are never interpolated (see `sequencing`).
    max_interp_gap: int = 2
    #: Minimum fraction of input slots that must be real (not interpolated).
    min_input_coverage: float = 0.9

    def __post_init__(self) -> None:
        self.horizons_min = tuple(int(h) for h in self.horizons_min)
        if any(h % self.grid_minutes for h in self.horizons_min):
            raise ConfigError(
                f"horizons_min={self.horizons_min} must all be multiples of "
                f"grid_minutes={self.grid_minutes}"
            )
        if not 0.0 < self.min_input_coverage <= 1.0:
            raise ConfigError("min_input_coverage must be in (0, 1]")

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        return tuple(h // self.grid_minutes for h in self.horizons_min)

    @property
    def max_horizon_steps(self) -> int:
        return max(self.horizon_steps)


@dataclass
class SplitConfig:
    #: ``official`` = OhioT1DM temporal holdout (same subjects, later period).
    #: ``loso``     = leave-one-subject-out, truly subject-disjoint.
    protocol: str = "official"
    val_fraction: float = 0.15
    #: Steps discarded at every split boundary so no training window shares a
    #: timestep with a validation or test window. ``None`` resolves to
    #: ``seq_len + max_horizon_steps``, the minimum safe value.
    purge_steps: int | None = None

    def __post_init__(self) -> None:
        if self.protocol not in {"official", "loso"}:
            raise ConfigError(
                f"split.protocol must be 'official' or 'loso', got {self.protocol!r}"
            )
        if not 0.0 < self.val_fraction < 0.5:
            raise ConfigError("split.val_fraction must be in (0, 0.5)")

    def resolved_purge_steps(self, data: DataConfig) -> int:
        if self.purge_steps is not None:
            return int(self.purge_steps)
        return data.seq_len + data.max_horizon_steps


@dataclass
class ModelConfig:
    #: ``persistence``/``roc``/``arima`` are baselines with no learned weights.
    #: ``lstm``/``transformer`` are the data-driven baselines.
    #: ``pinn`` is the physics-guided model (spline head + parameter encoder).
    kind: str = "pinn"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1
    #: Number of cubic B-spline basis functions on the forecast interval. The
    #: head emits one coefficient per basis, giving an analytic dG/dt.
    n_spline_basis: int = 12
    #: Attention pooling over the encoder output rather than a mean, which
    #: discards recency.
    pooling: str = "attention"  # attention | mean | last
    #: Add the mechanistic Bergman forecast as a prior and learn the residual.
    hybrid_residual: bool = True
    #: Estimate per-patient Bergman parameters from the observation window.
    per_patient_params: bool = True


@dataclass
class PhysicsConfig:
    enabled: bool = True
    #: Collocation points on the forecast interval where the ODE residual is
    #: evaluated. 121 = every minute over 0..120 min inclusive.
    n_collocation: int = 121
    #: ``kendall`` = learnable log-variance weights (Kendall et al. 2018).
    #: ``fixed``   = constant lambda_phys (the naive PINN, for ablation A1).
    #: ``gradnorm``/``relobralo`` for ablation.
    weighting: str = "kendall"
    lambda_phys: float = 0.1  # used when weighting == "fixed"
    #: Data-first curriculum: physics weight ramps in over this epoch window.
    ramp_start_epoch: int = 5
    ramp_end_epoch: int = 25
    #: Freeze patient parameters at population values for this many epochs so
    #: they cannot collapse before the encoder has any signal.
    param_warmup_epochs: int = 10
    lambda_param_prior: float = 1e-3
    lambda_temporal_consistency: float = 1e-2


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    early_stopping_patience: int = 15
    num_workers: int = 4
    #: Huber is robust to CGM artifacts in a way MSE is not.
    data_loss: str = "huber"
    huber_delta: float = 10.0  # mg/dL
    #: Quantiles the head predicts. The median is the reported point forecast; the
    #: lower quantile provides a hypoglycaemia alarm without biasing it.
    #:
    #: A point forecast trained on a mean-seeking loss systematically regresses
    #: toward the centre -- measured here at +7.29 mg/dL below 70 mg/dL even with no
    #: physics at all -- so it understates hypoglycaemia risk by construction. The
    #: fix is distributional rather than a tilt in the objective: predict the lower
    #: tail explicitly and alarm on it, leaving the point forecast unbiased.
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    #: Weight on the pinball loss relative to the point loss.
    lambda_quantile: float = 1.0
    #: Deliberately absent: any hypo/hyper asymmetric penalty. An asymmetric
    #: training loss inflates error-grid zone A by construction. Clinical
    #: safety is reported through the error grids, not baked into the loss.


def _validate_quantiles(values: tuple[float, ...]) -> None:
    if not values:
        raise ConfigError("train.quantiles must not be empty")
    if sorted(values) != list(values):
        raise ConfigError(f"train.quantiles must be ascending, got {values}")
    if any(not 0.0 < q < 1.0 for q in values):
        raise ConfigError(f"train.quantiles must lie in (0, 1), got {values}")
    if 0.5 not in values:
        raise ConfigError(
            f"train.quantiles must include 0.5: the median is the reported point "
            f"forecast and the physics constrains it. Got {values}"
        )


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        return _build(cls, raw, path="")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open() as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        return cls.from_dict(raw)

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return _unstructure(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    # -- derived ------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self.train.quantiles = tuple(float(q) for q in self.train.quantiles)
        _validate_quantiles(self.train.quantiles)

    @property
    def median_index(self) -> int:
        return self.train.quantiles.index(0.5)

    def resolve_device(self) -> str:
        if self.run.device != "auto":
            return self.run.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def out_dir(self) -> Path:
        return Path(self.run.out_root) / self.split.protocol / self.run.name


# --------------------------------------------------------------------------- #
# Strict structuring helpers
# --------------------------------------------------------------------------- #


def _build(cls: type, raw: Any, path: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path or '<root>'}: expected a mapping, got {type(raw).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        where = path or "<root>"
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    # ``from __future__ import annotations`` makes ``Field.type`` a string, so the
    # annotations must be resolved before they can be introspected. Without this,
    # nested sections are passed through as raw dicts and never validated.
    hints = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in raw:
            continue
        child_path = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(hints[name], raw[name], child_path)
    return cls(**kwargs)


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    if is_dataclass(annotation):
        return _build(annotation, value, path)
    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list")
        elem = args[0] if args else Any
        return tuple(_coerce(elem, v, f"{path}[]") for v in value)
    return value


def _unstructure(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _unstructure(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


__all__ = [
    "HORIZON_MINUTES",
    "Config",
    "ConfigError",
    "DataConfig",
    "ModelConfig",
    "PhysicsConfig",
    "RunConfig",
    "SplitConfig",
    "TrainConfig",
]
