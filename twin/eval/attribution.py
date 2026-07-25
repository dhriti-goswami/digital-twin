"""Feature attribution over the whole input window.

Why not the legacy approach
---------------------------
``src/models/explainer.py`` ran ``shap.KernelExplainer`` after tiling a *single*
timestep across all 24 positions:

.. code-block:: python

    np.tile(X_np[:, np.newaxis, :], (1, seq_len, 1))   # legacy

That explains a counterfactual the model never sees — a window in which every
timestep is identical — so the attributions describe a different input distribution
from the one the model was trained on. It also keyed its feature descriptions to an
obsolete feature list, so roughly half of them fell back to raw column names.

What is used instead
--------------------
**Integrated gradients** (Sundararajan, Taly & Yan, ICML 2017) over the *full*
``(seq_len, n_features)`` window:

.. math::

    \\mathrm{IG}_i(x) = (x_i - x'_i)\\int_{\\alpha=0}^{1}
        \\frac{\\partial f\\big(x' + \\alpha (x - x')\\big)}{\\partial x_i}\\, d\\alpha

approximated with an ``m``-step Riemann sum. Two properties make it preferable here:

* **Completeness.** :math:`\\sum_i \\mathrm{IG}_i = f(x) - f(x')` exactly, so the
  attributions account for the entire prediction rather than an unexplained residue.
  This is checkable, and :func:`integrated_gradients` returns the check.
* **The baseline is meaningful.** After standardisation the all-zero window *is* the
  training mean, so :math:`f(x')` is "what the model predicts for an average
  window" — a real reference point, not an arbitrary one.

Attributions are reported per feature (summed over the 24 timesteps) and also as a
time profile, which answers a question a single number cannot: *when* in the two-hour
window does a feature matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from twin.data.features import FEATURE_NAMES

Array = NDArray[np.floating]


@dataclass
class Attribution:
    """Integrated-gradient attributions for one horizon."""

    horizon_min: int
    #: ``(n_windows, seq_len, n_features)`` signed attributions, mg/dL.
    values: Array
    #: ``f(x) - f(baseline)`` per window, the quantity completeness must reproduce.
    prediction_delta: Array
    #: Max relative completeness violation across windows.
    completeness_error: float
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def per_feature(self) -> pd.DataFrame:
        """Mean absolute attribution per feature, summed over time.

        Absolute values are aggregated because a feature that pushes the forecast up in
        some windows and down in others is still influential; the signed mean would
        cancel it to zero and hide it.
        """
        summed = self.values.sum(axis=1)  # (n_windows, n_features)
        frame = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_attribution": np.abs(summed).mean(axis=0),
                "mean_signed_attribution": summed.mean(axis=0),
                "sd_attribution": summed.std(axis=0, ddof=1),
            }
        )
        total = frame["mean_abs_attribution"].sum()
        frame["share_pct"] = 100.0 * frame["mean_abs_attribution"] / total
        return frame.sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)

    def time_profile(self, top: int = 6) -> pd.DataFrame:
        """Mean absolute attribution by timestep for the most influential features."""
        ranked = self.per_feature()["feature"].head(top).tolist()
        indices = [self.feature_names.index(name) for name in ranked]
        profile = np.abs(self.values[:, :, indices]).mean(axis=0)  # (seq_len, top)
        frame = pd.DataFrame(profile, columns=ranked)
        frame.insert(0, "minutes_before_forecast", -5 * np.arange(profile.shape[0])[::-1])
        return frame

    def by_group(self) -> pd.DataFrame:
        """Attribution aggregated into the feature-contract groups."""
        from twin.data.features import feature_provenance

        groups = feature_provenance().set_index("feature")["group"]
        frame = self.per_feature()
        frame["group"] = frame["feature"].map(groups)
        summary = (
            frame.groupby("group")["mean_abs_attribution"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "total_abs_attribution", "count": "n_features"})
        )
        summary["share_pct"] = 100.0 * summary["total_abs_attribution"] / summary["total_abs_attribution"].sum()
        return summary.sort_values("total_abs_attribution", ascending=False).reset_index()


def integrated_gradients(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    horizon_index: int = 0,
    steps: int = 64,
    device: str = "cpu",
) -> Attribution:
    """Integrated gradients of one horizon's forecast w.r.t. the feature window.

    The baseline is the all-zero (standardised) window, i.e. the training mean. Only
    ``features`` is interpolated: the insulin and carbohydrate rate channels drive the
    mechanistic state and are held at their actual values, so the attribution answers
    "what did the *observed history* contribute", not "what if the pump record were
    also averaged away".
    """
    model = model.to(device).eval()
    features = batch["features"].to(device)
    baseline = torch.zeros_like(features)

    others = {
        key: value.to(device) for key, value in batch.items() if key != "features"
    }

    total = torch.zeros_like(features)
    for step in range(steps):
        alpha = (step + 0.5) / steps  # midpoint rule
        point = (baseline + alpha * (features - baseline)).detach().requires_grad_(True)
        output = model({**others, "features": point})
        target = output.horizons[:, horizon_index].sum()
        (gradient,) = torch.autograd.grad(target, point)
        total = total + gradient.detach()

    attributions = (features - baseline) * total / steps

    with torch.no_grad():
        high = model({**others, "features": features}).horizons[:, horizon_index]
        low = model({**others, "features": baseline}).horizons[:, horizon_index]
    delta = (high - low).cpu().numpy()

    summed = attributions.sum(dim=(1, 2)).cpu().numpy()
    scale = np.maximum(np.abs(delta), 1e-6)
    completeness = float(np.abs(summed - delta).max() / scale.max())

    from twin.config import HORIZON_MINUTES

    return Attribution(
        horizon_min=HORIZON_MINUTES[horizon_index],
        values=attributions.cpu().numpy(),
        prediction_delta=delta,
        completeness_error=completeness,
    )


def attribution_report(
    model: torch.nn.Module,
    loader,
    *,
    horizon_index: int = 0,
    max_batches: int = 8,
    steps: int = 64,
    device: str = "cpu",
) -> Attribution:
    """Attributions accumulated over several batches."""
    collected: list[Attribution] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        collected.append(
            integrated_gradients(
                model, batch, horizon_index=horizon_index, steps=steps, device=device
            )
        )
    if not collected:
        raise ValueError("no batches supplied")
    return Attribution(
        horizon_min=collected[0].horizon_min,
        values=np.concatenate([item.values for item in collected], axis=0),
        prediction_delta=np.concatenate([item.prediction_delta for item in collected]),
        completeness_error=max(item.completeness_error for item in collected),
    )


def permutation_importance(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    horizon_index: int = 0,
    n_repeats: int = 3,
    seed: int = 42,
    device: str = "cpu",
) -> pd.DataFrame:
    """Model-agnostic importance: rise in MAE when one feature is shuffled.

    Included as an independent cross-check on the integrated-gradient ranking. It makes
    **no differentiability assumption** — it only calls the model forward — so it is
    valid even where the completeness axiom is violated. Where the two methods agree on
    an ordering, that ordering does not rest on either method's assumptions.

    A feature is permuted **across windows but coherently across time**, preserving its
    within-window temporal structure while destroying its association with the target.
    Shuffling each timestep independently would additionally destroy the autocorrelation
    that makes the feature a plausible input at all, and would overstate importance.
    """
    model = model.to(device).eval()
    generator = np.random.default_rng(seed)

    def mae(perturb: int | None) -> float:
        errors: list[float] = []
        with torch.no_grad():
            for batch in batches:
                features = batch["features"].to(device).clone()
                if perturb is not None:
                    order = torch.from_numpy(
                        generator.permutation(features.shape[0])
                    ).to(device)
                    features[:, :, perturb] = features[order][:, :, perturb]
                others = {k: v.to(device) for k, v in batch.items() if k != "features"}
                predicted = model({**others, "features": features}).horizons[:, horizon_index]
                errors.append(
                    (predicted - batch["targets"].to(device)[:, horizon_index]).abs().mean().item()
                )
        return float(np.mean(errors))

    reference = mae(None)
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        deltas = [mae(index) - reference for _ in range(n_repeats)]
        rows.append(
            {
                "feature": name,
                "mae_increase": float(np.mean(deltas)),
                "mae_increase_sd": float(np.std(deltas, ddof=1)) if n_repeats > 1 else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["baseline_mae"] = reference
    frame["share_pct"] = 100.0 * frame["mae_increase"].clip(lower=0) / max(
        frame["mae_increase"].clip(lower=0).sum(), 1e-9
    )
    return frame.sort_values("mae_increase", ascending=False).reset_index(drop=True)


__all__ = [
    "Attribution",
    "attribution_report",
    "integrated_gradients",
    "permutation_importance",
]
