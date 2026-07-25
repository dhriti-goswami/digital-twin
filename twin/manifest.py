"""Run provenance.

A results table is only trustworthy if you can say exactly which code, which
data, and which environment produced it. Every run writes a manifest next to its
artifacts recording:

* the git commit and whether the tree was dirty,
* a checksum of every input data file actually read,
* the resolved config,
* the seed state,
* package versions and hardware.

The legacy pipeline recorded none of this, which is why its numbers cannot be
reconciled with any particular commit.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = (
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "shap",
    "simglucose",
    "pyyaml",
)


@dataclass
class GitState:
    commit: str
    branch: str
    dirty: bool
    dirty_files: list[str] = field(default_factory=list)


@dataclass
class Manifest:
    created_utc: str
    git: GitState
    config: dict[str, Any]
    seed_state: dict[str, Any]
    data_files: dict[str, str]  # relative path -> sha256
    packages: dict[str, str]
    platform_info: dict[str, str]
    notes: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=False)
        return path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming checksum, so multi-hundred-MB inputs do not land in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_state(repo: str | Path = ".") -> GitState:
    repo = Path(repo)
    status = _git("status", "--porcelain", repo=repo)
    dirty_files = [line[3:] for line in status.splitlines() if line.strip()]
    return GitState(
        commit=_git("rev-parse", "HEAD", repo=repo) or "unknown",
        branch=_git("rev-parse", "--abbrev-ref", "HEAD", repo=repo) or "unknown",
        dirty=bool(dirty_files),
        dirty_files=dirty_files,
    )


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def platform_info() -> dict[str, str]:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda or "unknown"
            total = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_gb"] = f"{total / 1024**3:.1f}"
        else:
            info["gpu"] = "none"
    except Exception:  # pragma: no cover - torch always present in this project
        info["gpu"] = "unknown"
    return info


def build_manifest(
    *,
    config: dict[str, Any],
    seed_state: dict[str, Any],
    data_paths: list[str | Path],
    repo: str | Path = ".",
    notes: dict[str, Any] | None = None,
) -> Manifest:
    """Assemble a manifest. ``data_paths`` should be every input file read."""
    from datetime import datetime, timezone

    repo = Path(repo)
    checksums: dict[str, str] = {}
    for raw_path in sorted({str(p) for p in data_paths}):
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            key = str(path.resolve().relative_to(repo.resolve()))
        except ValueError:
            key = str(path.resolve())
        checksums[key] = sha256_file(path)

    return Manifest(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git=git_state(repo),
        config=config,
        seed_state=seed_state,
        data_files=checksums,
        packages=package_versions(),
        platform_info=platform_info(),
        notes=notes or {},
    )


__all__ = [
    "GitState",
    "Manifest",
    "build_manifest",
    "git_state",
    "package_versions",
    "platform_info",
    "sha256_file",
]
