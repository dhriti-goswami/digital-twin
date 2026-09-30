import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.features import FEATURE_NAMES, N_FEATURES
from twin.data.splits import official_split, verify_no_leakage
from twin.eval.runner import evaluate_predictions, write_result
from twin.models.forecaster import PhysicsGuidedForecaster, ForecastOutput
from twin.physio.params import population_params
from twin.seeding import set_seed
from twin.train.loop import predict_loader, train_model

MECHANISTIC_FEATURES = frozenset({
    "iob_u",
    "insulin_sc_1_u",
    "insulin_sc_2_u",
    "insulin_plasma_uU_mL",
    "insulin_action_per_min",
    "cob_g",
    "carbs_stomach_g",
    "carbs_gut_g",
    "glucose_appearance_mgdl_per_min",
})
NON_MECH_INDICES = [i for i, name in enumerate(FEATURE_NAMES) if name not in MECHANISTIC_FEATURES]
print(f"Non-mechanistic feature count: {len(NON_MECH_INDICES)} (out of {N_FEATURES})")


# --- 1. LSTM Model Definition ---
class LSTMBaseline(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )
        self.hybrid = False
        self.quantiles = ()

    def forward(self, batch, **kwargs):
        feats = batch["features"]
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        anchor = batch["anchor_glucose"].unsqueeze(-1)
        pred_deviations = self.head(last)
        horizons = anchor + pred_deviations
        pop = population_params(batch_size=len(feats), device=feats.device)
        return ForecastOutput(
            horizons=horizons,
            collocation=horizons,
            derivative=horizons,
            residual=torch.zeros_like(horizons),
            params=pop,
            coefficients=horizons,
        )


def main():
    config = Config.from_yaml("configs/official.yaml")
    config.train.batch_size = 256
    config.train.early_stopping_patience = 10
    config.train.num_workers = 3
    set_seed(42, deterministic=True)

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
    scaler = fit_scaler(fold, corpus)

    loaders = {
        part: build_loader(
            build_dataset(fold, part, corpus, scaler, config),
            config,
            shuffle=(part == "train"),
        )
        for part in ("train", "val", "test")
    }
    subject_ids = loaders["test"].dataset.subject_ids
    device = config.resolve_device()

    results_summary = []

    # === Train LSTM Baseline ===
    print("\n================== Training LSTM Baseline ==================")
    lstm_config = Config.from_yaml("configs/official.yaml")
    lstm_config.train.batch_size = 256
    lstm_config.train.early_stopping_patience = 10
    lstm_config.train.num_workers = 3
    lstm_config.physics.enabled = False
    lstm_config.model.hybrid_residual = False
    lstm_config.model.per_patient_params = False
    lstm_config.run.name = "LSTM"

    lstm_model = LSTMBaseline(n_features=N_FEATURES, hidden_size=64, num_layers=2, dropout=0.2)
    train_model(
        lstm_model,
        loaders["train"],
        loaders["val"],
        lstm_config,
        scaler=scaler,
        fold_name="official",
        out_dir=Path("artifacts/official/LSTM/official"),
        verbose=True,
    )

    evaluation_lstm = predict_loader(lstm_model, loaders["test"], device)
    preds_lstm = {}
    for index, s_id in enumerate(subject_ids):
        mask = evaluation_lstm["subject_index"] == index
        preds_lstm[s_id] = evaluation_lstm["predictions"][mask]

    res_lstm = evaluate_predictions(
        method="LSTM",
        fold=fold,
        part="test",
        corpus=corpus,
        predictions=preds_lstm,
        config=lstm_config,
    )
    write_result(res_lstm, Path("artifacts/official/LSTM"))
    for h in [30, 60, 90, 120]:
        row = res_lstm.summary[res_lstm.summary["horizon_min"] == h].iloc[0]
        results_summary.append({
            "model": "LSTM",
            "horizon_min": h,
            "mae_mean": row["mae_mean"],
            "mae_sd": row["mae_sd"],
            "rmse_mean": row["rmse_mean"],
            "rmse_sd": row["rmse_sd"],
        })

    # === Train A0_no_mech (Transformer with only 26 non-mechanistic features) ===
    print("\n================== Training A0_no_mech ==================")
    a0_nomech_config = Config.from_yaml("configs/official.yaml")
    a0_nomech_config.train.batch_size = 256
    a0_nomech_config.train.early_stopping_patience = 10
    a0_nomech_config.train.num_workers = 3
    a0_nomech_config.physics.enabled = False
    a0_nomech_config.model.hybrid_residual = False
    a0_nomech_config.model.per_patient_params = False
    a0_nomech_config.run.name = "A0_no_mech"

    class ForecasterNoMechWrapper(nn.Module):
        def __init__(self, inner_model, non_mech_indices):
            super().__init__()
            self.inner = inner_model
            self.non_mech_indices = non_mech_indices
            self.hybrid = False
            self.quantiles = ()

        def forward(self, batch, **kwargs):
            b_copy = dict(batch)
            b_copy["features"] = batch["features"][:, :, self.non_mech_indices]
            return self.inner(b_copy, **kwargs)

    inner_model = PhysicsGuidedForecaster(len(NON_MECH_INDICES), a0_nomech_config)
    wrapped_a0_nomech = ForecasterNoMechWrapper(inner_model, NON_MECH_INDICES)

    train_model(
        wrapped_a0_nomech,
        loaders["train"],
        loaders["val"],
        a0_nomech_config,
        scaler=scaler,
        fold_name="official",
        out_dir=Path("artifacts/official/A0_no_mech/official"),
        verbose=True,
    )

    evaluation_nomech = predict_loader(wrapped_a0_nomech, loaders["test"], device)
    preds_nomech = {}
    for index, s_id in enumerate(subject_ids):
        mask = evaluation_nomech["subject_index"] == index
        preds_nomech[s_id] = evaluation_nomech["predictions"][mask]

    res_nomech = evaluate_predictions(
        method="A0_no_mech",
        fold=fold,
        part="test",
        corpus=corpus,
        predictions=preds_nomech,
        config=a0_nomech_config,
    )
    write_result(res_nomech, Path("artifacts/official/A0_no_mech"))
    for h in [30, 60, 90, 120]:
        row = res_nomech.summary[res_nomech.summary["horizon_min"] == h].iloc[0]
        results_summary.append({
            "model": "A0_no_mech",
            "horizon_min": h,
            "mae_mean": row["mae_mean"],
            "mae_sd": row["mae_sd"],
            "rmse_mean": row["rmse_mean"],
            "rmse_sd": row["rmse_sd"],
        })

    print("\n=== Summary of New Baselines ===")
    df_new = pd.DataFrame(results_summary)
    print(df_new.to_string(index=False))
    df_new.to_csv("results/tables/baselines_lstm_a0nomech.csv", index=False)

if __name__ == "__main__":
    main()
