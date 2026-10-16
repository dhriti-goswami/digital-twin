#!/usr/bin/env python3
"""Aggregate per-run epoch counts and wall-clock time from training_history.csv files.

The paper (main.tex:465) claims "about 23 s per epoch (about 4.5 min per run)"
with early-stopping patience 12 (configs/official-small.yaml:55). Patience 12
means training only stops after 12 epochs with no improvement, i.e. the run
has at least best_epoch + 12 epochs, not a fixed epoch count. This script
reads real per-epoch wall-clock seconds (the "seconds" column already logged
by twin/train/loop.py) and reports mean epochs/run and mean seconds/epoch
from actual training runs, instead of an assumed constant.

Each training_history.csv row is one completed epoch, so len(df) IS the
number of epochs actually run for that arm (training stopped there).
"""

from pathlib import Path
import pandas as pd

# Single-seed official ablation arms (A0-A4, A7) + the final quantile model +
# the small-fast sanity run, all on the same official-small.yaml schedule
# (patience=12). LOSO is excluded: it is a different protocol (12 folds
# concatenated into one history file) and is not what "per run" means here.
RUNS = [
    "artifacts/official/abl-A0/training_history.csv",
    "artifacts/official/abl-A1/training_history.csv",
    "artifacts/official/abl-A2/training_history.csv",
    "artifacts/official/abl-A3/training_history.csv",
    "artifacts/official/abl-A4/training_history.csv",
    "artifacts/official/abl-A7/training_history.csv",
    "artifacts/official/quantile/training_history.csv",
    "artifacts/official/small-fast/training_history.csv",
]


def main():
    rows = []
    for path in RUNS:
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(p)
        n_epochs = len(df)
        mean_sec_per_epoch = df["seconds"].mean()
        total_sec = df["seconds"].sum()
        rows.append({
            "run": p.parent.name,
            "n_epochs": n_epochs,
            "mean_sec_per_epoch": mean_sec_per_epoch,
            "total_sec": total_sec,
            "total_min": total_sec / 60.0,
        })
        print(f"  {p.parent.name:12s}: {n_epochs:3d} epochs, "
              f"{mean_sec_per_epoch:5.2f} s/epoch (mean), "
              f"{total_sec:7.1f} s total ({total_sec / 60.0:.2f} min)")

    df_all = pd.DataFrame(rows)
    mean_epochs = df_all["n_epochs"].mean()
    mean_sec_per_epoch_overall = df_all["mean_sec_per_epoch"].mean()
    mean_total_min = df_all["total_min"].mean()

    print("\n=== Summary across runs (early-stopping patience=12, configs/official-small.yaml:55) ===")
    print(f"  Mean epochs per run:        {mean_epochs:.2f}")
    print(f"  Mean seconds per epoch:     {mean_sec_per_epoch_overall:.2f} s")
    print(f"  Mean wall-clock time / run: {mean_total_min:.2f} min")
    print("\n  NOTE: patience=12 means training runs at least best_epoch+12 epochs; it does")
    print("  NOT fix the run length. 'About 4.5 min per run' in main.tex:465 should be")
    print("  replaced with the measured mean above, and 'about 23 s per epoch' should")
    print("  cite the measured mean seconds/epoch, not a single assumed constant.")

    out_path = Path("results/tables/training_time_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved per-run summary to {out_path}")


if __name__ == "__main__":
    main()
