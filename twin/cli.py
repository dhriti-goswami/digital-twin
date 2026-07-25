"""Command-line entry point.

One documented way to run each stage, so a result can always be traced to the
command that produced it::

    python -m twin data      --config configs/official.yaml
    python -m twin baselines --config configs/official.yaml
    python -m twin report    --config configs/official.yaml

Every subcommand writes a run manifest next to its artifacts recording the git
commit, the resolved config, data checksums, and package versions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from twin.config import Config
from twin.manifest import build_manifest
from twin.seeding import set_seed


def _load_config(path: str | None, overrides: list[str] | None = None) -> Config:
    config = Config.from_yaml(path) if path else Config()
    for override in overrides or []:
        if "=" not in override:
            raise SystemExit(f"--set expects key=value, got {override!r}")
        key, value = override.split("=", 1)
        target: object = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        current = getattr(target, parts[-1])
        cast = type(current) if current is not None else str
        setattr(target, parts[-1], cast(value) if cast is not bool else value.lower() == "true")
    return config


def _write_manifest(config: Config, out_dir: Path, notes: dict[str, object]) -> None:
    from twin.data.ohio import discover_files

    seed_state = set_seed(config.run.seed, deterministic=config.run.deterministic)
    try:
        data_paths = [str(path) for path in discover_files(config.data.root)]
    except Exception:
        data_paths = []
    manifest = build_manifest(
        config=config.to_dict(),
        seed_state=seed_state.__dict__,
        data_paths=data_paths,
        notes=notes,
    )
    manifest.write(out_dir / "manifest.json")
    config.to_yaml(out_dir / "resolved_config.yaml")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def command_data(config: Config) -> int:
    """Build the corpus and report window accounting per subject."""
    from twin.data.dataset import load_corpus
    from twin.data.sequencing import window_report_table
    from twin.data.splits import fold_table, loso_splits, official_split, verify_no_leakage

    set_seed(config.run.seed, deterministic=config.run.deterministic)
    corpus = load_corpus(config)
    train_sets = {key: value.windows for key, value in corpus["train"].items()}
    test_sets = {key: value.windows for key, value in corpus["test"].items()}

    windows = window_report_table(list(train_sets.values()) + list(test_sets.values()))
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out_dir / "window_report.csv", index=False)

    official = official_split(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=config.split.val_fraction,
        purge_steps=config.split.purge_steps,
    )
    verify_no_leakage(official, train_sets, test_sets)
    folds = loso_splits(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=config.split.val_fraction,
        purge_steps=config.split.purge_steps,
    )
    for fold in folds:
        verify_no_leakage(fold, train_sets, test_sets)
    fold_table([official, *folds]).to_csv(out_dir / "fold_report.csv", index=False)

    print(windows.to_string(index=False))
    print()
    print(f"windows kept: {windows['kept'].sum():,} of {windows['n_candidates'].sum():,} "
          f"candidates ({100 * windows['kept'].sum() / windows['n_candidates'].sum():.1f}%)")
    print(f"official split: {official.summary()}")
    print(f"loso folds: {len(folds)}, all pass leakage verification")
    print(f"wrote {out_dir / 'window_report.csv'} and {out_dir / 'fold_report.csv'}")

    _write_manifest(config, out_dir, {"stage": "data", "n_windows": int(windows["kept"].sum())})
    return 0


def command_baselines(config: Config, *, methods: list[str], part: str) -> int:
    """Run the non-learned baselines and write their tables."""
    from twin.data.dataset import load_corpus
    from twin.data.splits import official_split, verify_no_leakage
    from twin.eval.runner import evaluate_baseline, leaderboard, skill_score, write_result

    set_seed(config.run.seed, deterministic=config.run.deterministic)
    corpus = load_corpus(config)
    train_sets = {key: value.windows for key, value in corpus["train"].items()}
    test_sets = {key: value.windows for key, value in corpus["test"].items()}
    fold = official_split(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=config.split.val_fraction,
        purge_steps=config.split.purge_steps,
    )
    verify_no_leakage(fold, train_sets, test_sets)

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for method in methods:
        print(f"running {method} ...", flush=True)
        result = evaluate_baseline(
            method, fold=fold, part=part, corpus=corpus, config=config
        )
        write_result(result, out_dir)
        results.append(result)

    frames = []
    for metric in ("rmse", "mae"):
        frame = leaderboard(results, metric=metric)
        frame.insert(0, "metric", metric)
        frames.append(frame.rename(columns=lambda name: name.replace(f"{metric}_", "value_")))
    board = pd.concat(frames, ignore_index=True)
    board.to_csv(out_dir / "baseline_leaderboard.csv", index=False)

    reference = next((r for r in results if r.method == "persistence"), None)
    if reference is not None:
        skills = pd.concat(
            [
                skill_score(result, reference, metric=metric)
                for result in results
                if result.method != "persistence"
                for metric in ("rmse", "mae")
            ],
            ignore_index=True,
        ) if len(results) > 1 else pd.DataFrame()
        if not skills.empty:
            skills.to_csv(out_dir / "baseline_skill_vs_persistence.csv", index=False)

    print()
    print(board.to_string(index=False))
    print()
    print(f"wrote {out_dir / 'baseline_leaderboard.csv'}")
    _write_manifest(config, out_dir, {"stage": "baselines", "methods": methods, "part": part})
    return 0



def command_train(config: Config, *, fold_index: int | None = None) -> int:
    """Train the forecaster on one protocol and evaluate on the held-out test set."""
    from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
    from twin.data.features import N_FEATURES
    from twin.data.splits import loso_splits, official_split, verify_no_leakage
    from twin.eval.runner import evaluate_baseline, evaluate_predictions, leaderboard, skill_score, write_result
    from twin.models.forecaster import PhysicsGuidedForecaster
    from twin.train.loop import predict_loader, train_model

    set_seed(config.run.seed, deterministic=config.run.deterministic)
    corpus = load_corpus(config)
    train_sets = {key: value.windows for key, value in corpus["train"].items()}
    test_sets = {key: value.windows for key, value in corpus["test"].items()}

    if config.split.protocol == "official":
        folds = [official_split(list(train_sets.values()), list(test_sets.values()),
                                val_fraction=config.split.val_fraction,
                                purge_steps=config.split.purge_steps)]
    else:
        folds = loso_splits(list(train_sets.values()), list(test_sets.values()),
                            val_fraction=config.split.val_fraction,
                            purge_steps=config.split.purge_steps)
        if fold_index is not None:
            folds = [folds[fold_index]]

    out_root = config.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, object] = {}
    histories = []

    for fold in folds:
        verify_no_leakage(fold, train_sets, test_sets)
        scaler = fit_scaler(fold, corpus)
        loaders = {
            part: build_loader(build_dataset(fold, part, corpus, scaler, config),
                               config, shuffle=(part == "train"))
            for part in ("train", "val", "test")
        }
        print(f"=== {fold.name}: {fold.counts()} ===", flush=True)
        model = PhysicsGuidedForecaster(N_FEATURES, config)
        result = train_model(model, loaders["train"], loaders["val"], config,
                             scaler=scaler, fold_name=fold.name,
                             out_dir=out_root / fold.name)
        histories.append(result.history_frame().assign(fold=fold.name))

        evaluation = predict_loader(model, loaders["test"], config.resolve_device())
        dataset = loaders["test"].dataset
        subject_ids = dataset.subject_ids
        for index, subject_id in enumerate(subject_ids):
            mask = evaluation["subject_index"] == index
            predictions[subject_id] = evaluation["predictions"][mask]

        diagnostics = {
            "insulin_sensitivity": evaluation["insulin_sensitivity"],
            "subject_index": evaluation["subject_index"],
            "targets": evaluation["targets"],
        }
        for key in ("quantile_predictions", "quantile_levels"):
            if key in evaluation:
                diagnostics[key] = evaluation[key]
        np.savez_compressed(out_root / fold.name / "test_diagnostics.npz", **diagnostics)

    merged = folds[0] if len(folds) == 1 else None
    if merged is None:
        merged = official_split(list(train_sets.values()), list(test_sets.values()),
                                val_fraction=config.split.val_fraction,
                                purge_steps=config.split.purge_steps)
        merged.protocol = config.split.protocol
        merged.name = config.split.protocol

    model_result = evaluate_predictions(method=config.run.name, fold=merged, part="test",
                                        corpus=corpus, predictions=predictions, config=config)
    write_result(model_result, out_root)

    reference = evaluate_baseline("persistence", fold=merged, part="test",
                                 corpus=corpus, config=config)
    write_result(reference, out_root)

    board = leaderboard([reference, model_result], metric="rmse")
    board_mae = leaderboard([reference, model_result], metric="mae")
    board.to_csv(out_root / "leaderboard_rmse.csv", index=False)
    board_mae.to_csv(out_root / "leaderboard_mae.csv", index=False)
    skill = skill_score(model_result, reference, metric="rmse")
    skill.to_csv(out_root / "skill_vs_persistence.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(out_root / "training_history.csv", index=False)

    print()
    print(board.to_string(index=False))
    print()
    print(board_mae[["method", "horizon_min", "mae_mean", "mae_sd"]].to_string(index=False))
    print()
    print(skill.to_string(index=False))
    _write_manifest(config, out_root, {"stage": "train", "folds": [f.name for f in folds]})
    return 0



def command_ablate(config: Config, *, ids: list[str] | None) -> int:
    """Run the ablation matrix and write the comparison table.

    Every configuration shares the identical corpus, splits, scaler, seed and epoch
    budget, so the only difference between two rows is the thing being ablated.
    """
    from twin.train.ablations import matrix_table, resolve

    out_root = config.out_dir / "ablations"
    out_root.mkdir(parents=True, exist_ok=True)
    matrix_table().to_csv(out_root / "ablation_matrix.csv", index=False)

    rows = []
    for ablation in resolve(ids):
        variant = ablation.apply(config)
        print(f"\n=== {ablation.id}: {ablation.label} ===", flush=True)
        command_train(variant, fold_index=None)
        board = pd.read_csv(variant.out_dir / "leaderboard_mae.csv")
        board = board[board["method"] != "persistence"]
        for _, record in board.iterrows():
            rows.append(
                {
                    "ablation": ablation.id,
                    "label": ablation.label,
                    "isolates": ablation.isolates,
                    "horizon_min": record["horizon_min"],
                    "mae_mean": record["mae_mean"],
                    "mae_sd": record["mae_sd"],
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(out_root / "ablation_results.csv", index=False)
    print()
    if not table.empty:
        pivot = table.pivot_table(
            index="ablation", columns="horizon_min", values="mae_mean"
        ).round(2)
        print("MAE (mg/dL), mean across subjects:")
        print(pivot.to_string())
    _write_manifest(config, out_root, {"stage": "ablate", "ablations": [a.id for a in resolve(ids)]})
    return 0


def command_report(config: Config) -> int:
    """Regenerate every table and figure from saved predictions.

    Nothing is recomputed from a model and nothing is hand-typed: the figures and
    the tables are both derived from the stored prediction arrays, so they cannot
    disagree with each other or go stale.
    """
    import numpy as np

    from twin.eval.figures import (
        plot_error_grid,
        plot_horizon_metric,
        plot_learning_curves,
        plot_skill_score,
    )
    from twin.metrics.errorgrid import assert_verified
    from twin.physio.params import assert_bounds_sourced, second_hand_bounds

    # Reporting gates: refuse to emit a paper table from unverified boundaries or
    # placeholder parameter ranges.
    assert_verified("clarke")
    assert_verified("parkes")
    assert_bounds_sourced()

    root = config.out_dir
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, pd.DataFrame] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for directory in sorted((root / config.split.protocol / "test").glob("*")):
        if not (directory / "summary.csv").is_file():
            continue
        method = directory.name
        summaries[method] = pd.read_csv(directory / "summary.csv")
        archive = directory / "predictions.npz"
        if archive.is_file():
            with np.load(archive) as data:
                predictions[method] = {key: data[key] for key in data.files}

    if not summaries:
        print(f"no results found under {root}; run `train` or `baselines` first",
              file=sys.stderr)
        return 1

    written: list[Path] = []
    for metric in ("rmse", "mae"):
        if f"{metric}_mean" not in next(iter(summaries.values())).columns:
            continue
        path = figures / f"{metric}_by_horizon.png"
        plot_horizon_metric(summaries, path, metric=metric, protocol=config.split.protocol)
        written.append(path)

    for method, arrays in predictions.items():
        if method == "persistence":
            continue
        true = np.concatenate([v for k, v in sorted(arrays.items()) if k.startswith("true__")])
        pred = np.concatenate([v for k, v in sorted(arrays.items()) if k.startswith("pred__")])
        if true.size == 0:
            continue
        for index, horizon in enumerate(config.data.horizons_min):
            for grid in ("clarke", "parkes"):
                path = figures / f"{grid}_{method}_{horizon}min.png"
                plot_error_grid(
                    np.clip(true[:, index], 1.0, None),
                    np.clip(pred[:, index], 1.0, None),
                    path,
                    grid=grid,
                    horizon_min=horizon,
                    protocol=config.split.protocol,
                )
                written.append(path)

    skill_path = root / "skill_vs_persistence.csv"
    if skill_path.is_file():
        path = figures / "skill_vs_persistence.png"
        plot_skill_score(pd.read_csv(skill_path), path, protocol=config.split.protocol)
        written.append(path)

    history_path = root / "training_history.csv"
    if history_path.is_file():
        path = figures / "learning_curves.png"
        plot_learning_curves(pd.read_csv(history_path), path)
        written.append(path)

    disclosures = second_hand_bounds()
    if disclosures:
        pd.DataFrame(
            [{"parameter": k, "note": v} for k, v in sorted(disclosures.items())]
        ).to_csv(root / "second_hand_parameters.csv", index=False)

    print(f"wrote {len(written)} figures under {figures}")
    for path in written:
        print(f"  {path}")
    if disclosures:
        print()
        print("parameter ranges resting on a secondary source (disclose in the paper):")
        for name in sorted(disclosures):
            print(f"  {name}")
    _write_manifest(config, root, {"stage": "report", "n_figures": len(written)})
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="twin", description=__doc__)
    parser.add_argument("--config", help="path to a YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override a config field, e.g. --set run.seed=7",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data", help="build the corpus and report window accounting")

    baselines = subparsers.add_parser("baselines", help="run the non-learned baselines")
    baselines.add_argument(
        "--methods",
        nargs="+",
        default=["persistence", "roc_extrapolation"],
        help="baselines to run; 'arima' is considerably slower",
    )
    baselines.add_argument("--part", default="test", choices=["train", "val", "test"])

    train = subparsers.add_parser("train", help="train the forecaster")
    train.add_argument("--fold-index", type=int, default=None,
                       help="train only one LOSO fold, by index")

    ablate = subparsers.add_parser("ablate", help="run the ablation matrix")
    ablate.add_argument("--ids", nargs="+", default=None,
                        help="ablation ids to run, e.g. A0 A1 A3 (default: all runnable)")

    subparsers.add_parser("report", help="regenerate all tables and figures")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args.config, args.overrides)

    if args.command == "data":
        return command_data(config)
    if args.command == "baselines":
        return command_baselines(config, methods=args.methods, part=args.part)
    if args.command == "train":
        return command_train(config, fold_index=args.fold_index)
    if args.command == "ablate":
        return command_ablate(config, ids=args.ids)
    if args.command == "report":
        return command_report(config)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
