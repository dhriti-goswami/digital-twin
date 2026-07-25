"""Determinism and configuration integrity.

``test_seed_reproducibility`` is the guard for the legacy pipeline's ad-hoc
seeding: ``torch.manual_seed`` only, sometimes called *after* the data had been
split, with numpy and the DataLoader workers left unseeded.

The config tests guard a subtler failure: a silently-ignored unknown key means a
recorded setting and the setting that actually ran can differ, which makes a
results table untraceable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from twin.config import Config, ConfigError
from twin.manifest import build_manifest, git_state, package_versions, sha256_file
from twin.seeding import make_dataloader_kwargs, set_seed


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def _draw() -> tuple[float, float, float]:
    """One sample from each source of randomness we depend on."""
    import random

    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1)),
    )


def test_seed_reproducibility():
    """The same seed must reproduce python, numpy, and torch draws alike."""
    set_seed(1234)
    first = _draw()
    set_seed(1234)
    second = _draw()
    assert first == second


def test_different_seeds_differ():
    """A sanity check that seeding is not accidentally constant."""
    set_seed(1)
    first = _draw()
    set_seed(2)
    second = _draw()
    assert first != second


def test_seed_state_is_recorded():
    """The returned state must describe what was applied, for the manifest."""
    state = set_seed(7, deterministic=True)
    assert state.seed == 7
    assert state.deterministic_algorithms is True
    assert state.cudnn_deterministic is True
    assert state.cudnn_benchmark is False
    assert state.torch_version


def test_model_init_is_reproducible():
    """Weight initialisation must be seed-determined, not merely seed-influenced."""
    from twin.physio.spline import SplineEvaluator, SplineGrid

    grid = SplineGrid.build()

    set_seed(99)
    layer_a = torch.nn.Linear(16, 12)
    module_a = SplineEvaluator(grid, n_basis=12)

    set_seed(99)
    layer_b = torch.nn.Linear(16, 12)
    module_b = SplineEvaluator(grid, n_basis=12)

    assert torch.equal(layer_a.weight, layer_b.weight)
    assert torch.equal(module_a.collocation_basis, module_b.collocation_basis)


def test_dataloader_shuffle_is_reproducible():
    """Two loaders built with the same seed must yield the same batch order."""
    data = torch.arange(200).unsqueeze(-1).float()
    dataset = torch.utils.data.TensorDataset(data)

    def batches() -> list[int]:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True, **make_dataloader_kwargs(42, num_workers=0)
        )
        return [int(batch[0][0, 0]) for batch in loader]

    assert batches() == batches()


def test_dataloader_kwargs_wire_up_worker_seeding():
    """The worker init function must be present, or numpy inside workers is unseeded."""
    from twin.seeding import seed_worker

    kwargs = make_dataloader_kwargs(0, num_workers=2)
    assert kwargs["worker_init_fn"] is seed_worker
    assert kwargs["generator"] is not None
    assert kwargs["persistent_workers"] is True

    assert make_dataloader_kwargs(0, num_workers=0)["persistent_workers"] is False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_unknown_config_key_is_rejected():
    """A typo must fail loudly rather than being silently dropped."""
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"train": {"learning_rate": 1e-3}})  # real name is `lr`


def test_unknown_top_level_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"trainer": {}})


def test_config_round_trips(tmp_path):
    """A config written and reread must be identical, so runs stay traceable."""
    config = Config.from_dict(
        {"run": {"name": "t", "seed": 7}, "split": {"protocol": "loso"}}
    )
    path = tmp_path / "resolved.yaml"
    config.to_yaml(path)
    assert Config.from_yaml(path).to_dict() == config.to_dict()


def test_horizons_must_divide_the_grid():
    """A horizon that is not a multiple of the sampling interval is unrepresentable."""
    with pytest.raises(ConfigError, match="multiples of"):
        Config.from_dict({"data": {"grid_minutes": 5, "horizons_min": [30, 45, 62]}})


def test_horizon_steps_derived_correctly():
    config = Config.from_dict({"data": {"grid_minutes": 5, "horizons_min": [30, 60, 90, 120]}})
    assert config.data.horizon_steps == (6, 12, 18, 24)
    assert config.data.max_horizon_steps == 24


def test_bad_split_protocol_rejected():
    with pytest.raises(ConfigError, match="official"):
        Config.from_dict({"split": {"protocol": "random"}})


def test_purge_steps_default_covers_window_and_horizon():
    """The default purge must be large enough that no window can straddle a split.

    A window spans ``seq_len`` input steps and reaches ``max_horizon_steps``
    forward, so anything smaller than their sum leaves overlap -- exactly the leak
    that invalidated the legacy fine-tuning validation set.
    """
    config = Config.from_dict({"data": {"seq_len": 24, "horizons_min": [30, 60, 90, 120]}})
    assert config.split.resolved_purge_steps(config.data) == 48


def test_purge_steps_override_respected():
    config = Config.from_dict({"split": {"purge_steps": 100}})
    assert config.split.resolved_purge_steps(config.data) == 100


def test_val_fraction_bounds_enforced():
    with pytest.raises(ConfigError, match="val_fraction"):
        Config.from_dict({"split": {"val_fraction": 0.9}})


def test_no_clinical_asymmetry_knob_exists():
    """There must be no way to configure an asymmetric hypo/hyper training penalty.

    The legacy loss weighted a missed hyperglycaemia at 6.0 and a missed
    hypoglycaemia at 2.0 -- backwards, since hypoglycaemia is the acute risk. Any
    asymmetric training loss also inflates error-grid zone A by construction.
    Clinical safety is reported through the error grids, not optimised into the
    objective, so the knob is deliberately absent.
    """
    from twin.config import TrainConfig

    names = {field.name for field in TrainConfig.__dataclass_fields__.values()}
    for forbidden in ("hypo_penalty", "hyper_penalty", "clinical_weight", "penalty_hypo"):
        assert forbidden not in names


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def test_manifest_records_provenance(tmp_path):
    """A manifest must pin code, data, and environment together."""
    data_file = tmp_path / "input.csv"
    data_file.write_text("a,b\n1,2\n")

    config = Config.from_dict({"run": {"name": "manifest-test"}})
    manifest = build_manifest(
        config=config.to_dict(),
        seed_state={"seed": 42},
        data_paths=[data_file],
        repo=tmp_path,
    )

    assert manifest.created_utc
    assert manifest.config["run"]["name"] == "manifest-test"
    assert len(manifest.data_files) == 1
    checksum = next(iter(manifest.data_files.values()))
    assert checksum == sha256_file(data_file)
    assert "python" in manifest.packages
    assert "torch" in manifest.packages

    written = manifest.write(tmp_path / "manifest.json")
    assert written.exists()


def test_checksum_detects_modification(tmp_path):
    """The checksum must change when the data changes, or provenance is worthless."""
    path = tmp_path / "d.csv"
    path.write_text("1")
    before = sha256_file(path)
    path.write_text("2")
    assert sha256_file(path) != before


def test_missing_data_file_is_skipped_not_fatal(tmp_path):
    """A path that does not exist must not abort manifest construction."""
    manifest = build_manifest(
        config={},
        seed_state={},
        data_paths=[tmp_path / "does-not-exist.csv"],
        repo=tmp_path,
    )
    assert manifest.data_files == {}


def test_git_state_reports_commit_and_dirtiness():
    """Provenance needs the commit and whether the tree was clean."""
    state = git_state(".")
    assert state.commit
    assert state.branch
    assert isinstance(state.dirty, bool)


def test_package_versions_include_core_stack():
    versions = package_versions()
    assert versions["torch"] != "not-installed"
    assert versions["numpy"] != "not-installed"
