"""Determinism control.

Every reported number must be reproducible from a seed. The legacy pipeline
seeded ``torch`` only, sometimes after the data had already been shuffled, and
never seeded the DataLoader workers or cuDNN. This module centralises all of it.

Usage::

    from twin.seeding import set_seed, make_dataloader_kwargs

    set_seed(42)
    loader = DataLoader(ds, batch_size=64, shuffle=True,
                        **make_dataloader_kwargs(42, num_workers=4))
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

# cuBLAS requires this to be set *before* the first CUDA context is created for
# deterministic matmul. Setting it here, at import time, is the only reliable
# place: by the time set_seed() is called a context may already exist.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_DEFAULT_SEED = 42


@dataclass(frozen=True)
class SeedState:
    """Record of what was seeded, for the run manifest."""

    seed: int
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    torch_version: str
    cuda_available: bool


def set_seed(
    seed: int = _DEFAULT_SEED,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
) -> SeedState:
    """Seed every source of randomness we depend on.

    Parameters
    ----------
    seed
        The seed applied to ``random``, ``numpy``, and ``torch`` (CPU and all
        CUDA devices).
    deterministic
        Enable ``torch.use_deterministic_algorithms`` and deterministic cuDNN.
        Leave this on for anything whose output is reported.
    warn_only
        Passed through to ``torch.use_deterministic_algorithms``. When ``True``,
        an op with no deterministic implementation warns instead of raising.
        Use only to diagnose which op is the problem, never for a reported run.

    Returns
    -------
    SeedState
        What was actually applied, for recording in the run manifest.
    """
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True

    return SeedState(
        seed=seed,
        deterministic_algorithms=deterministic,
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
    )


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader.

    PyTorch gives each worker a distinct ``torch`` seed but leaves ``numpy`` and
    ``random`` unseeded, so any numpy-based augmentation or sampling inside a
    worker is nondeterministic. Derive both from the torch seed.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = _DEFAULT_SEED):
    """A ``torch.Generator`` for DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_dataloader_kwargs(
    seed: int = _DEFAULT_SEED, num_workers: int = 0
) -> dict[str, Any]:
    """Keyword arguments that make a DataLoader deterministic.

    Spread into every ``DataLoader`` construction in the pipeline.
    """
    return {
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": make_generator(seed),
        "persistent_workers": num_workers > 0,
    }


__all__ = [
    "SeedState",
    "set_seed",
    "seed_worker",
    "make_generator",
    "make_dataloader_kwargs",
]
