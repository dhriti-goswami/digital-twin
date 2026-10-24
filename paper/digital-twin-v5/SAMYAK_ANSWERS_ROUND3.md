# Samyak's answers — TODO_ROUND3.md S-1, S-4, S-5, S-6, S-7

## S-1. Early stopping — resolved, no retrain needed

Good news: the earlier (round-2) theory that the checkpoint was "never selected" was **wrong** —
investigated fresh against your new evidence (A0-A4 also stopping early, `physics_ramp=1.0` from
epoch 1) and found the real cause.

**Root cause — a provenance bug, not a training bug:** `twin/cli.py`'s `train` and `report`
subcommands both wrote to the same `out_dir/manifest.json`. `report` runs later (as a separate
invocation) and silently overwrote the training-time manifest with whatever config `report`
happened to be called with. For the quantile-head run, the manifest on disk now reflects a vanilla
`report --config configs/official-small.yaml` call (patience=12, ramp_end=25), NOT what `train`
actually used — confirmed because `training_history.csv`'s live-logged `physics_ramp` column reads
1.0 from epoch 1, which is only possible if the real training-time `ramp_end_epoch<=1`, contradicting
the (clobbered) manifest's `ramp_end_epoch=25`.

Cross-checked against all 6 ablation runs (A0-A4, A7): every one of THEIR manifests is internally
consistent with its own stop epoch (e.g. A7: ramp_end=0, warmup=0, patience=10 → best epoch 5,
stopped epoch 15 = exactly 10 non-improving epochs). Only the quantile run's manifest is broken —
consistent with it being a one-off manual run with custom overrides that a later `report` call
didn't repeat.

**Consequence**: selection for the quantile run began at epoch 1 (not epoch 26 as last round
claimed), and **the epoch-3 checkpoint WAS validly used as `best_state`** — this was a real,
working early-stopping run, just with an undocumented/inferred patience (arithmetic implies patience
was 8, since it stopped at epoch 11 with best at epoch 3; this exact value can't be certified since
the record was overwritten, but the selection mechanism itself worked correctly).

**No retrain needed.** Suggested Sec. II-I text:
> "Most runs used early-stopping patience 10 (ablations) or 12 (`configs/official-small.yaml`
> default). The deployed quantile-head run's training-time configuration was not preserved in its
> run manifest — a provenance bug in which a later `report` invocation overwrote the manifest
> written at training time — but its recorded per-epoch physics-ramp trace shows model selection
> began at epoch 1 (an instant full-physics curriculum), consistent with the observed stop at
> epoch 11 (best epoch 3, an inferred effective patience of 8)."

**Code fix applied** (prevents recurrence): `twin/cli.py` `_write_manifest` now writes
`manifest_{stage}.json` / `resolved_config_{stage}.yaml` per stage instead of a shared
`manifest.json` that `train` and `report` silently clobbered.

## S-4. Table V/VI — not actually broken in main.tex; my round-2 CSV edit was

Checked: `main.tex` was never changed by round 2's (bad) harmonization attempt — it already uses
the original single-seed ablation numbers consistently in both Table V (A0=13.51±1.63,
A7=13.25±1.69 at 30min) and Table VI (A0 sensitivity 0.637/+7.29 bias, A7 0.608/+8.61 bias), and
already has an explicit disambiguating sentence at `main.tex:1012`: *"Tables V and VI therefore
report different runs and architectures and should not be compared directly [to Table IX]."*

The actual bug was that round 2's data-file edits (`comprehensive_baselines.csv`,
`table_vi_hypo_quantile_head.csv`) drifted out of sync with the correct `.tex` — **reverted both**,
removed the stray `table_vi_hypo_quantile_head.csv` file entirely (its quantile-head sensitivity
numbers already exist in `results/tables/hypoglycemia_a0_vs_a7.csv`, which Table IX already
sources from). No Discussion-section number changes — the "1.3 mg/dL" claim (8.61-7.29) was never
actually altered, so it stays as-is.

If you still want extra insurance beyond the existing line-1012 sentence, a one-line Table V
caption addition: "A0/A7 rows are the single-seed (seed=42) ablation architecture, no quantile
head; see Table VII for the 5-seed replication." Table VI needs nothing further.

## S-5. Headline number — lead with the 5-seed mean

Recommend abstract/conclusion lead with **13.37 ± SD (5-seed mean, Table VII)** for the robustness
claim, not the single-run final-model number (13.08) or the single-seed ablation number (13.25).
Reserve 13.08 for describing the specific deployed/reported system elsewhere in the text, with
clear labeling (already discussed in round 2's answer doc, item 6) — this is now additionally
supported by S-1: since the quantile run's exact training provenance was reconstructed rather than
directly certified, leading with the 5-seed mean (whose provenance is clean across all 5 seeds) is
the more defensible headline choice.

## S-6. S_I vs Bergman reference — real limitation, not defensible as "different task"

Checked `scripts/fit_bergman_per_subject.py`'s fitting objective directly: it minimizes the SAME
multi-horizon forecast MSE used for evaluation, just on train data — **not** a local/short
calibration-window curve-fit. So "it's a different, better-conditioned estimation task" does NOT
hold as a defense.

Computed actual in-sample (train-set) fit quality:

| horizon | train MAE | train RMSE |
|---|---|---|
| 30 min | 25.50 ± 6.30 | 34.24 ± 7.94 |
| 60 min | 37.60 ± 6.87 | 49.71 ± 8.43 |
| 90 min | 44.17 ± 6.47 | 57.90 ± 7.96 |
| 120 min | 48.04 ± 6.30 | 62.74 ± 7.81 |

Nearly identical to test-set numbers (25.80/34.72/etc.) — this is a genuinely poorly-fitting ODE
in-sample too, not a train/test generalization gap. Honest limitation sentence for `main.tex:1181`
(where ρ=0.790 is reported):
> "Although the fitted per-subject Bergman model forecasts glucose poorly even in-sample
> (train-set MAE 25.5 mg/dL at 30 min, essentially matching its test-set error), its ratio of
> kinetic parameters defining S_I can still be loosely identified from the overall shape of the
> glucose response to insulin and meals; the agreement (ρ=0.790) should be read as weak convergent
> validity from a crude, independently-structured reference, not confirmation from a trustworthy
> forecaster." Consider weighting this check less heavily than the other three S_I checks (CV,
> ICC, TDD correlation), which don't share this weakness.

## S-7. Final hyperparameters — confirmed

From `configs/official-small.yaml`, verified to match `artifacts/official/quantile/`'s actual run
(architecture dims can't silently mismatch — checkpoint loading would fail on shape mismatch):
**d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.2, n_spline_basis=12,
pooling=attention, hybrid_residual=true, per_patient_params=true, physics weighting=Kendall.**
All within the stated search space in `main.tex:473` (at the smaller/simpler end).

Validation-only search confirmed: `configs/official-small.yaml`'s own header comment states
"Selected on validation only; the test split is untouched" — cite that comment as evidence.
