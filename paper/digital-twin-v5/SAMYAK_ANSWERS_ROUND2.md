# Samyak's answers — TODO_ROUND2.md S-1 to S-4

## S-2. CRPS — fixed

Old formula's docstring algebra was wrong and didn't reproduce its own stated derivation; the
code that originally produced the committed `crps` column is not in the repo (lost/undocumented).
Recomputed properly via the standard trapezoidal-rule CRPS-from-quantiles identity
(CRPS = 2·∫₀¹ pinball_τ(y) dτ, trapezoidal quadrature over τ∈{0.1,0.5,0.9} with explicit zero
boundaries at τ=0,1): **CRPS ≈ 0.5·pinball_q10 + 0.8·pinball_q50 + 0.5·pinball_q90**.

New values (30/60/90/120 min): **8.434 / 13.989 / 18.017 / 21.009 mg/dL** (replaces old
9.74/16.18/20.85/24.32 — update Table X and the Sec III-G4 CRPS sentence to these).
`scripts/compute_crps.py`, `results/tables/calibration_by_horizon.csv` +
`paper/digital-twin-v5/data/calibration_by_horizon.csv` updated.

## S-3. Legacy doc — done

`docs/TRAINING_METHODOLOGY.md` is referenced by README.md and docs/ARCHITECTURE.md, so not
deleted — added a legacy banner at the top flagging the stale patience=15 / MAE=5.53 numbers and
pointing to `paper/digital-twin-v5/main.tex`/`CHANGES.md` as the source of truth.

## S-4. Table V / VI harmonization — done

**Table V**: A0/A7 rows now use the 5-seed means from `multiseed_a0_vs_a7.csv` (same source as
Table VII), not the single seed-42 ablation run. New values:
- A0: 30min 13.88±0.25 / 19.72±0.25 → 120min 36.46±1.07 / 48.46±1.22
- A7: 30min **13.37±0.11 / 19.17±0.16** → 120min 34.08±0.64 / 45.47±0.86
(13.25 no longer appears anywhere in Table V — only in the response letter as the single-seed
ablation number, per S-5's earlier resolution.) Caption: "A0/A7 rows are 5-seed means, consistent
with Table VII."

**Table VI hypoglycemia columns**: could not reproduce the old 0.637/0.608 from the per-subject
ablation CSVs by any aggregation method tried (per-subject mean or event-weighted pooled mean both
missed) — same "lost original method" problem as CRPS. Replaced with the quantile-head models'
own sensitivity at 30 min, i.e. the SAME models as Table IX: **A0 sensitivity 0.413 (bias +13.36
mg/dL), A7 sensitivity 0.551 (bias +9.24 mg/dL)** — now matches Table IX exactly by construction
(bias values are newly computed, not previously reported). Table VI's MAE columns (30-120min) are
untouched — those are the pre-specified single-seed significance-test numbers, a legitimate
separate use case. Caption: "Hypoglycemia columns use the same quantile-head models as Table IX."
Source: `results/tables/table_vi_hypo_quantile_head.csv`.

## S-1. Early-stopping mismatch — NOT just a wording issue, needs a decision

The final quantile-head run (the one behind Table IV's headline 13.08 mg/dL) genuinely stopped at
epoch 11. But `twin/train/loop.py:120-129` computes
`selection_starts_at = 1 + max(param_warmup_epochs=10, ramp_end_epoch=25) = 26` — **model
selection and the early-stopping patience counter don't activate until epoch 26.** This run never
reached epoch 26, so:

- Patience=12 never had a chance to fire. The run was *interrupted* (cause unknown — no crash
  signature in file timestamps, no log found explaining it), not early-stopped.
- `best_state` is only ever set once selection starts (epoch ≥26); for this run it stayed `None`
  the whole time, so the checkpoint-loading branch at `loop.py:242` was skipped.
- The saved `best_model.pt` is just the raw weights after epoch 11 — **never a validation-selected
  checkpoint**. "Best validation MAE at epoch 3" is just the lowest raw per-epoch number among
  epochs 1-11; it was never the basis for any selection decision.

**This means Table IV's final-model numbers (13.08 MAE etc.) come from an unselected,
interrupted-mid-warmup checkpoint, not a properly validated model.** I have not touched this —
it needs your call:

1. **Rerun the final quantile-head training job to completion** (to epoch 26+ so selection
   actually engages), and replace Table IV's numbers with the properly-selected checkpoint's
   results. This is the methodologically correct fix but changes headline numbers two days before
   submission and takes real training time.
2. **Keep the current numbers but state the limitation honestly** in the paper, e.g.: "The
   deployed quantile-head model's training run was interrupted at epoch 11, before the
   curriculum-driven selection window opens (epoch 26 under this configuration); the reported
   weights reflect the model state at interruption rather than a validation-selected checkpoint."
   This is defensible but weakens the "final model" framing and a careful reviewer could flag it.

I'm not picking one of these without you — tell me which way to go and I'll execute it.
