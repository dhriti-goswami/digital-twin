"""Corpus assembly: the single supported path from raw XML to model tensors.

Everything the earlier modules guarantee individually is composed here, so that
using the pipeline correctly is the default and bypassing a guarantee requires
going out of one's way:

* windows are built with the feature-validity gate always applied,
* the scaler is fitted on **training-fold slots only** and travels with the split,
* the horizon guarantee is re-verified after assembly.

Scaler fitting
--------------
The scaler is fitted over the *set of grid slots* that appear as an input in at
least one training window, each counted once -- not over stacked windows. Stacking
would count a slot up to ``seq_len`` times (consecutive windows overlap by 23 of
24 slots), weighting the statistics by how many windows happen to cover a slot,
which is an artefact of gap structure rather than anything physiological.

The legacy pipeline also refitted a fresh scaler in ``personalize.py:97`` over an
entire patient series including the evaluation period. Here a
:class:`FittedScaler` is attached to the split that produced it and is the only way
to transform, so a mismatch is structurally impossible.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from twin.config import Config
from twin.data.features import FEATURE_NAMES, FeatureMatrix, build_features
from twin.data.ohio import GriddedSubject, discover_files, load_subject
from twin.data.sequencing import (
    FILLED_COLUMN,
    WindowSet,
    build_windows,
    interpolate_bounded,
    persistence_targets,
)
from twin.data.splits import Fold, Selection
from twin.seeding import make_dataloader_kwargs

Array = NDArray[np.float64]


@dataclass
class SubjectData:
    """Everything derived from one subject's file, kept aligned."""

    subject: GriddedSubject
    frame: pd.DataFrame
    features: FeatureMatrix
    windows: WindowSet

    @property
    def subject_id(self) -> str:
        return self.subject.subject_id

    @property
    def split(self) -> str:
        return self.subject.split

    def anchor_glucose(self) -> Array:
        """Glucose at each window's anchor -- the persistence prediction."""
        return persistence_targets(self.subject, self.windows, frame=self.frame)


def load_subject_data(path: str, config: Config) -> SubjectData:
    """Parse, grid, featurise, and window one subject."""
    subject = load_subject(path, grid_minutes=config.data.grid_minutes)
    frame = interpolate_bounded(subject.frame, max_interp_gap=config.data.max_interp_gap)
    features = build_features(subject, frame=frame, max_interp_gap=config.data.max_interp_gap)
    windows = build_windows(
        subject,
        seq_len=config.data.seq_len,
        horizons_min=config.data.horizons_min,
        min_input_coverage=config.data.min_input_coverage,
        max_interp_gap=config.data.max_interp_gap,
        frame=frame,
        # Always gated: a window is only emitted if every feature of every input
        # slot was computable from real data.
        input_valid=features.valid_row_mask(),
    )
    return SubjectData(subject=subject, frame=frame, features=features, windows=windows)


def load_corpus(
    config: Config, *, cache_dir: str | Path | None = "artifacts/cache"
) -> dict[str, dict[str, SubjectData]]:
    """Load every subject, keyed by split then subject id.

    Parsing and featurising the full corpus takes ~45 s, so the result is cached to
    disk. The cache key covers every config field that changes the output *and* the
    source-file checksums, so a stale cache cannot silently supply data built under
    different settings or from different files.
    """
    if cache_dir is None:
        return _load_corpus_uncached(config)

    from twin.manifest import sha256_file

    paths = {
        split: [str(path) for path in discover_files(config.data.root, split=split)]
        for split in ("train", "test")
    }
    key_material = {
        "data": config.data.to_dict() if hasattr(config.data, "to_dict") else str(config.data),
        "feature_names": list(FEATURE_NAMES),
        "files": {path: sha256_file(path)[:16] for split in paths for path in paths[split]},
    }
    digest = hashlib.sha256(
        json.dumps(key_material, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    cache_path = Path(cache_dir) / f"corpus-{digest}.pkl"
    if cache_path.is_file():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    corpus = _load_corpus_uncached(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return corpus


def _load_corpus_uncached(config: Config) -> dict[str, dict[str, SubjectData]]:
    corpus: dict[str, dict[str, SubjectData]] = {"train": {}, "test": {}}
    for split in ("train", "test"):
        for path in discover_files(config.data.root, split=split):
            data = load_subject_data(str(path), config)
            corpus[split][data.subject_id] = data
    return corpus


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #


@dataclass
class FittedScaler:
    """Per-feature standardisation, fitted once on a training fold.

    Bound to the fold that produced it and serialised into the checkpoint, so the
    transform applied at evaluation is provably the transform fitted at training.
    """

    mean: Array
    scale: Array
    feature_names: tuple[str, ...]
    n_slots_fitted: int
    fold_name: str

    def transform(self, values: Array) -> Array:
        if values.shape[-1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {values.shape[-1]}"
            )
        return (values - self.mean) / self.scale

    def state_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "feature_names": list(self.feature_names),
            "n_slots_fitted": self.n_slots_fitted,
            "fold_name": self.fold_name,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "FittedScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float64),
            scale=np.asarray(state["scale"], dtype=np.float64),
            feature_names=tuple(state["feature_names"]),  # type: ignore[arg-type]
            n_slots_fitted=int(state["n_slots_fitted"]),  # type: ignore[arg-type]
            fold_name=str(state["fold_name"]),
        )


def _input_slots(window_set: WindowSet, indices: NDArray[np.int64]) -> NDArray[np.int64]:
    """The set of grid slots read by a selection of windows, each once."""
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    anchors = window_set.anchors[indices]
    offsets = np.arange(-window_set.seq_len + 1, 1, dtype=np.int64)
    return np.unique((anchors[:, None] + offsets[None, :]).ravel())


def fit_scaler(
    fold: Fold, corpus: dict[str, dict[str, SubjectData]], *, epsilon: float = 1e-8
) -> FittedScaler:
    """Fit standardisation on the training fold's input slots only."""
    stacked: list[Array] = []
    for selection in fold.train:
        data = corpus["train"][selection.subject_id]
        slots = _input_slots(data.windows, selection.indices)
        stacked.append(data.features.values[slots])
    if not stacked:
        raise ValueError(f"{fold.name}: no training slots to fit the scaler on")

    values = np.concatenate(stacked, axis=0)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    # A constant feature (an availability mask that is uniformly 0 or 1 within a
    # fold) would otherwise divide by zero. Leaving its scale at 1 maps it to a
    # constant offset, which is harmless and honest.
    scale = np.where(scale < epsilon, 1.0, scale)
    return FittedScaler(
        mean=mean,
        scale=scale,
        feature_names=FEATURE_NAMES,
        n_slots_fitted=int(values.shape[0]),
        fold_name=fold.name,
    )


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #


#: Grid steps of insulin and carbohydrate history supplied before each anchor so
#: the mechanistic state is settled when the forecast interval begins.
#:
#: The compartments are initialised at the analytic basal steady state and then
#: replay the subject's real history, so the state entering the forecast is
#: data-driven rather than assumed. 144 steps at 5 minutes is 12 hours: several
#: multiples of the slowest time constant in the admissible parameter range
#: (``k_abs`` as low as 0.005/min gives a 200-minute gut time constant), so a meal
#: or bolus before the burn-in cannot leave a trace.
PHYSICS_BURNIN_STEPS = 144


@dataclass
class WindowBatchSource:
    """One subject's scaled features plus the windows selected from it."""

    subject_id: str
    scaled: NDArray[np.float32]
    anchors: NDArray[np.int64]
    targets: NDArray[np.float32]
    anchor_glucose: NDArray[np.float32]
    seq_len: int
    #: Insulin and carbohydrate input rates on the grid, padded at both ends, from
    #: which each window's physics span is sliced.
    insulin_rate: NDArray[np.float32]
    carb_rate: NDArray[np.float32]
    horizon_steps: tuple[int, ...]
    #: Observable basal glucose for this subject, resolved leakage-free per fold.
    basal_glucose: float = 120.0
    body_weight_kg: float = 70.0
    #: Offset applied to an anchor to index into the padded rate arrays.
    pad_offset: int = PHYSICS_BURNIN_STEPS


class WindowDataset(Dataset):
    """Windows drawn from several subjects, indexed globally.

    Feature slices are taken as views at access time rather than materialised up
    front: a stacked tensor for ~100k windows would be ~340 MB of largely duplicated
    data, since consecutive windows share 23 of 24 slots.

    Each item carries the mechanistic input rates over its forecast interval so the
    physics residual can be evaluated without re-reading the corpus.
    """

    def __init__(self, sources: list[WindowBatchSource], *, physics_span_steps: int) -> None:
        self.sources = sources
        self.physics_span_steps = physics_span_steps
        self._offsets = np.cumsum([0] + [source.anchors.size for source in sources])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        """Map a global index to ``(source_index, local_index)``."""
        source_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return source_index, index - int(self._offsets[source_index])

    @property
    def subject_ids(self) -> tuple[str, ...]:
        """Subject id per source index, so a prediction can be attributed back."""
        return tuple(source.subject_id for source in self.sources)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index, local = self._locate(index)
        source = self.sources[source_index]
        anchor = int(source.anchors[local])
        start = anchor - source.seq_len + 1

        features = source.scaled[start : anchor + 1]
        # The physics span runs from the burn-in start to the longest horizon. The
        # rate arrays are pre-padded by ``pad_offset`` at the front, so an anchor
        # early in the record still has a full burn-in (of zero input, i.e. only the
        # basal steady state, which is the correct assumption for unknown history).
        physics_start = anchor + source.pad_offset - PHYSICS_BURNIN_STEPS
        span = slice(physics_start, physics_start + PHYSICS_BURNIN_STEPS + self.physics_span_steps + 1)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(features)),
            "targets": torch.from_numpy(source.targets[local]),
            "anchor_glucose": torch.tensor(source.anchor_glucose[local]),
            "insulin_rate": torch.from_numpy(np.ascontiguousarray(source.insulin_rate[span])),
            "carb_rate": torch.from_numpy(np.ascontiguousarray(source.carb_rate[span])),
            # Resolved positionally. Searching the source list by value would
            # compare numpy arrays and only happen to work via an identity
            # short-circuit.
            "subject_index": torch.tensor(source_index, dtype=torch.long),
            "basal_glucose": torch.tensor(source.basal_glucose, dtype=torch.float32),
            "body_weight_kg": torch.tensor(source.body_weight_kg, dtype=torch.float32),
        }


#: Hours treated as the overnight fasting window when estimating basal glucose.
FASTING_HOURS = (0, 6)


def fasting_glucose(data: SubjectData) -> float:
    """Median overnight glucose -- an observable estimate of ``G_b``.

    ``G_b`` is not estimated by the network. It is a measurable summary of the
    subject's own record, so inventing a latent for it would be modelling something
    already observed.
    """
    frame = data.frame
    hour = frame.index.hour
    overnight = frame.loc[(hour >= FASTING_HOURS[0]) & (hour < FASTING_HOURS[1]), FILLED_COLUMN]
    value = float(overnight.median()) if overnight.notna().any() else float(
        frame[FILLED_COLUMN].median()
    )
    return value


def resolve_basal_glucose(
    fold: Fold, corpus: dict[str, dict[str, SubjectData]]
) -> dict[str, float]:
    """Per-subject ``G_b``, resolved so it never leaks across a split.

    Under the official protocol each subject's own **training** period supplies the
    value -- legitimate, since that subject's history is available by construction.

    Under LOSO the held-out subject's own data is off-limits, including for a
    summary statistic, so the value is the mean across the *training* subjects. A
    fasting median is a weak statistic, but taking it from the held-out subject
    would still be using that subject to configure the model that scores it.
    """
    training_values = {
        selection.subject_id: fasting_glucose(corpus["train"][selection.subject_id])
        for selection in fold.train
    }
    if not training_values:
        raise ValueError(f"{fold.name}: no training subjects to estimate basal glucose from")
    population = float(np.mean(list(training_values.values())))

    resolved = dict(training_values)
    if fold.protocol == "loso" and fold.held_out_subject is not None:
        resolved[fold.held_out_subject] = population
    for subject_id in corpus["test"]:
        resolved.setdefault(subject_id, population)
    return resolved


def build_dataset(
    fold: Fold,
    part: str,
    corpus: dict[str, dict[str, SubjectData]],
    scaler: FittedScaler,
    config: Config,
) -> WindowDataset:
    """Assemble the dataset for one part of a fold.

    ``part`` is ``"train"``, ``"val"``, or ``"test"``. Validation draws from the
    training files (it is a purged temporal tail of them), test from the test files.
    """
    if part not in {"train", "val", "test"}:
        raise ValueError(f"part must be train/val/test, got {part!r}")
    source_split = "test" if part == "test" else "train"
    selections: list[Selection] = getattr(fold, part)

    # The forecast interval plus one slot, so the mechanistic state can be advanced
    # across the whole horizon.
    physics_span = config.data.max_horizon_steps
    basal_glucose = resolve_basal_glucose(fold, corpus)

    sources: list[WindowBatchSource] = []
    for selection in selections:
        if selection.indices.size == 0:
            continue
        data = corpus[source_split][selection.subject_id]
        scaled = scaler.transform(data.features.values).astype(np.float32)

        frame = data.frame
        insulin = (
            frame["basal_u_per_min"].to_numpy(dtype=np.float64)
            + frame["bolus_u_per_min"].to_numpy(dtype=np.float64)
        )
        carbs = frame["carbs_mg_per_min"].to_numpy(dtype=np.float64)
        # Pad both ends. The tail lets a window whose horizon reaches the final slot
        # read a full span; the head gives every anchor a full burn-in. Zero input
        # is the right pad in both directions: nothing is known to be delivered
        # outside the record, and the compartments start from the basal steady state
        # regardless.
        head = np.zeros(PHYSICS_BURNIN_STEPS)
        tail = np.zeros(physics_span + 1)
        insulin = np.concatenate([head, insulin, tail])
        carbs = np.concatenate([head, carbs, tail])

        anchors = data.windows.anchors[selection.indices]
        sources.append(
            WindowBatchSource(
                subject_id=selection.subject_id,
                scaled=scaled,
                anchors=anchors,
                targets=data.windows.targets[selection.indices].astype(np.float32),
                anchor_glucose=data.frame[FILLED_COLUMN]
                .to_numpy(dtype=np.float32)[anchors],
                seq_len=config.data.seq_len,
                insulin_rate=insulin.astype(np.float32),
                carb_rate=carbs.astype(np.float32),
                horizon_steps=data.windows.horizon_steps,
                basal_glucose=float(basal_glucose[selection.subject_id]),
                body_weight_kg=float(data.subject.body_weight_kg),
            )
        )
    return WindowDataset(sources, physics_span_steps=physics_span)


def build_loader(
    dataset: WindowDataset,
    config: Config,
    *,
    shuffle: bool,
) -> DataLoader:
    """A deterministic DataLoader over a window dataset."""
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        drop_last=False,
        **make_dataloader_kwargs(config.run.seed, num_workers=config.train.num_workers),
    )


def collect_predictions(
    fold: Fold,
    part: str,
    corpus: dict[str, dict[str, SubjectData]],
) -> dict[str, tuple[Array, Array]]:
    """Per-subject ``(targets, persistence)`` for a fold part.

    The persistence column is returned alongside the targets everywhere, so a
    results table cannot be produced without its naive baseline sitting next to it.
    """
    source_split = "test" if part == "test" else "train"
    out: dict[str, tuple[Array, Array]] = {}
    for selection in getattr(fold, part):
        if selection.indices.size == 0:
            continue
        data = corpus[source_split][selection.subject_id]
        targets = data.windows.targets[selection.indices]
        baseline = data.anchor_glucose()[selection.indices]
        out[selection.subject_id] = (targets, baseline)
    return out


__all__ = [
    "FittedScaler",
    "SubjectData",
    "WindowBatchSource",
    "WindowDataset",
    "build_dataset",
    "build_loader",
    "collect_predictions",
    "fit_scaler",
    "load_corpus",
    "load_subject_data",
]
