#!/usr/bin/env python3
"""Generate comprehensive baselines table across all 9 models on identical test windows.

Evaluates:
  1. Persistence
  2. Pure Bergman ODE (population)
  3. Fitted Bergman ODE (per-subject NLS/MAP)
  4. AR(6)
  5. ARX(6)
  6. LSTM (non-mechanistic)
  7. Transformer (A0_no_mech)
  8. Transformer (all features, A0)
  9. Hybrid physics-guided (A7)

Outputs:
  - results/tables/comprehensive_baselines.csv
  - paper/digital-twin-v4/data/comprehensive_baselines.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.splits import official_split
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.physio.bergman import integrate_glucose
from twin.physio.params import PatientParams
from scripts.run_ar_arx import extract_ar, extract_arx


def main():
    print("=== Generating Comprehensive Baselines (All 9 Models) ===")
    config = Config()
    config.train.batch_size = 256
    config.train.num_workers = 0
    corpus = load_corpus(config)
    train_sets = {key: value.windows for key, value in corpus["train"].items()}
    test_sets = {key: value.windows for key, value in corpus["test"].items()}
    fold = official_split(
        list(train_sets.values()),
        list(test_sets.values()),
        val_fraction=config.split.val_fraction,
        purge_steps=config.split.purge_steps,
    )
    subjects = sorted(list(corpus["test"].keys()))
    horizons = [30, 60, 90, 120]
    all_rows = []

    # 1. Persistence
    print("Loading Persistence...")
    df_pers = pd.read_csv("results/tables/per_subject_persistence.csv")
    for h in horizons:
        sub = df_pers[df_pers["horizon_min"] == h]
        all_rows.append({
            "model": "Persistence",
            "horizon_min": h,
            "mae_mean": round(float(sub["mae"].mean()), 2),
            "mae_sd": round(float(sub["mae"].std(ddof=1)), 2),
            "rmse_mean": round(float(sub["rmse"].mean()), 2),
            "rmse_sd": round(float(sub["rmse"].std(ddof=1)), 2),
        })

    # 2. Pure Bergman ODE (population parameters)
    print("Evaluating Pure Bergman ODE (population)...")
    scaler = fit_scaler(fold, corpus)
    test_ds = build_dataset(fold, "test", corpus, scaler, config)
    test_loader = build_loader(test_ds, config, shuffle=False)
    device = config.resolve_device()
    model = PhysicsGuidedForecaster(35, config).to(device)
    model.eval()

    colloc = model.spline.collocation_min.to(device)
    dt = float(colloc[1] - colloc[0])
    horizon_indices = [int(torch.argmin((colloc - m).abs())) for m in model.spline.horizon_min]

    all_preds_pop, all_targets, all_subjs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            params = model.resolve_params(
                batch_dev["features"],
                basal_glucose=batch_dev["basal_glucose"],
                body_weight_kg=batch_dev["body_weight_kg"],
                basal_insulin_rate=batch_dev["basal_insulin_rate"],
                use_population=True,
            )
            insulin_action, appearance = model.mechanistic_state(
                params, batch_dev["insulin_rate"], batch_dev["carb_rate"]
            )
            mechanistic = integrate_glucose(
                batch_dev["anchor_glucose"],
                insulin_action,
                appearance,
                params,
                dt=dt,
            )
            all_preds_pop.append(mechanistic[:, horizon_indices].cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
            all_subjs.append(batch["subject_index"].cpu().numpy())

    preds_pop = np.concatenate(all_preds_pop, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    subjs = np.concatenate(all_subjs, axis=0)

    for h_idx, h in enumerate(horizons):
        subj_maes = []
        subj_rmses = []
        for s_idx in range(len(test_ds.subject_ids)):
            mask = subjs == s_idx
            err = preds_pop[mask, h_idx] - targets[mask, h_idx]
            subj_maes.append(np.mean(np.abs(err)))
            subj_rmses.append(np.sqrt(np.mean(err**2)))
        all_rows.append({
            "model": "Bergman ODE (pop.)",
            "horizon_min": h,
            "mae_mean": round(float(np.mean(subj_maes)), 2),
            "mae_sd": round(float(np.std(subj_maes, ddof=1)), 2),
            "rmse_mean": round(float(np.mean(subj_rmses)), 2),
            "rmse_sd": round(float(np.std(subj_rmses, ddof=1)), 2),
        })

    # 3. Fitted Bergman ODE (per-subject NLS/MAP)
    print("Fitting Bergman ODE (per-subject NLS/MAP)...")
    train_ds = build_dataset(fold, "train", corpus, scaler, config)
    train_loader = build_loader(train_ds, config, shuffle=False)
    train_by_subj = {s: [] for s in range(len(train_ds.subject_ids))}
    for batch in train_loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for s in range(len(train_ds.subject_ids)):
            mask = batch_dev["subject_index"] == s
            if mask.any():
                train_by_subj[s].append({k: v[mask] for k, v in batch_dev.items() if isinstance(v, torch.Tensor)})

    fitted_p1, fitted_p2, fitted_p3 = {}, {}, {}
    for s, subject_id in enumerate(train_ds.subject_ids):
        u_p1 = torch.nn.Parameter(torch.zeros(1, device=device, requires_grad=True))
        u_p2 = torch.nn.Parameter(torch.zeros(1, device=device, requires_grad=True))
        u_p3 = torch.nn.Parameter(torch.zeros(1, device=device, requires_grad=True))
        optimizer = torch.optim.Adam([u_p1, u_p2, u_p3], lr=0.05)

        for _ in range(15):
            for b in train_by_subj[s]:
                optimizer.zero_grad()
                B = b["features"].shape[0]
                p1 = (torch.sigmoid(u_p1) * 0.030).expand(B)
                p2 = (0.005 + torch.sigmoid(u_p2) * (0.10 - 0.005)).expand(B)
                p3 = (1e-6 + torch.sigmoid(u_p3) * (3e-5 - 1e-6)).expand(B)
                pop = model.resolve_params(
                    b["features"],
                    basal_glucose=b["basal_glucose"],
                    body_weight_kg=b["body_weight_kg"],
                    basal_insulin_rate=b["basal_insulin_rate"],
                    use_population=True,
                )
                params = PatientParams(
                    p1=p1, p2=p2, p3=p3, n=pop.n, V_G=pop.V_G, V_I=pop.V_I,
                    tmax_I=pop.tmax_I, k_gri=pop.k_gri, k_abs=pop.k_abs, f=pop.f,
                    G_b=pop.G_b, I_b=pop.I_b,
                )
                insulin_action, appearance = model.mechanistic_state(params, b["insulin_rate"], b["carb_rate"])
                mechanistic = integrate_glucose(b["anchor_glucose"], insulin_action, appearance, params, dt=dt)
                pred_h = mechanistic[:, horizon_indices]
                loss = torch.nn.functional.mse_loss(pred_h, b["targets"]) + 0.01 * (u_p1**2 + u_p2**2 + u_p3**2)
                loss.backward()
                optimizer.step()

        fitted_p1[subject_id] = float(torch.sigmoid(u_p1).item() * 0.030)
        fitted_p2[subject_id] = float(0.005 + torch.sigmoid(u_p2).item() * (0.10 - 0.005))
        fitted_p3[subject_id] = float(1e-6 + torch.sigmoid(u_p3).item() * (3e-5 - 1e-6))

    # Evaluate fitted Bergman on test set
    all_preds_fit = []
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            subjs_in_batch = batch["subject_index"].numpy()
            p1_list = [fitted_p1[test_ds.subject_ids[s]] for s in subjs_in_batch]
            p2_list = [fitted_p2[test_ds.subject_ids[s]] for s in subjs_in_batch]
            p3_list = [fitted_p3[test_ds.subject_ids[s]] for s in subjs_in_batch]
            pop = model.resolve_params(
                batch_dev["features"],
                basal_glucose=batch_dev["basal_glucose"],
                body_weight_kg=batch_dev["body_weight_kg"],
                basal_insulin_rate=batch_dev["basal_insulin_rate"],
                use_population=True,
            )
            params = PatientParams(
                p1=torch.tensor(p1_list, device=device, dtype=pop.p1.dtype),
                p2=torch.tensor(p2_list, device=device, dtype=pop.p2.dtype),
                p3=torch.tensor(p3_list, device=device, dtype=pop.p3.dtype),
                n=pop.n, V_G=pop.V_G, V_I=pop.V_I, tmax_I=pop.tmax_I, k_gri=pop.k_gri,
                k_abs=pop.k_abs, f=pop.f, G_b=pop.G_b, I_b=pop.I_b,
            )
            insulin_action, appearance = model.mechanistic_state(params, batch_dev["insulin_rate"], batch_dev["carb_rate"])
            mechanistic = integrate_glucose(batch_dev["anchor_glucose"], insulin_action, appearance, params, dt=dt)
            all_preds_fit.append(mechanistic[:, horizon_indices].cpu().numpy())

    preds_fit = np.concatenate(all_preds_fit, axis=0)
    for h_idx, h in enumerate(horizons):
        subj_maes = []
        subj_rmses = []
        for s_idx in range(len(test_ds.subject_ids)):
            mask = subjs == s_idx
            err = preds_fit[mask, h_idx] - targets[mask, h_idx]
            subj_maes.append(np.mean(np.abs(err)))
            subj_rmses.append(np.sqrt(np.mean(err**2)))
        all_rows.append({
            "model": "Bergman ODE (fitted)",
            "horizon_min": h,
            "mae_mean": round(float(np.mean(subj_maes)), 2),
            "mae_sd": round(float(np.std(subj_maes, ddof=1)), 2),
            "rmse_mean": round(float(np.mean(subj_rmses)), 2),
            "rmse_sd": round(float(np.std(subj_rmses, ddof=1)), 2),
        })

    # 4 & 5. AR(6) and ARX(6)
    print("Evaluating AR(6) and ARX(6)...")
    for model_name, extract_fn in [("AR(6)", extract_ar), ("ARX(6)", extract_arx)]:
        subj_maes = {30: [], 60: [], 90: [], 120: []}
        subj_rmses = {30: [], 60: [], 90: [], 120: []}
        for s in subjects:
            train_data = corpus["train"][s]
            test_data = corpus["test"][s]
            train_sel = next(sel for sel in fold.train if sel.subject_id == s)
            test_sel = next(sel for sel in fold.test if sel.subject_id == s)
            X_train, Y_train = extract_fn(train_data, train_sel.indices)
            X_test, Y_test = extract_fn(test_data, test_sel.indices)
            W = np.linalg.solve(X_train.T @ X_train + 1.0 * np.eye(X_train.shape[1]), X_train.T @ Y_train)
            Y_pred = np.clip(X_test @ W, 40.0, 400.0)
            err = Y_pred - Y_test
            mae = np.mean(np.abs(err), axis=0)
            rmse = np.sqrt(np.mean(err**2, axis=0))
            for idx, h in enumerate(horizons):
                subj_maes[h].append(mae[idx])
                subj_rmses[h].append(rmse[idx])
        for h in horizons:
            all_rows.append({
                "model": model_name,
                "horizon_min": h,
                "mae_mean": round(float(np.mean(subj_maes[h])), 2),
                "mae_sd": round(float(np.std(subj_maes[h], ddof=1)), 2),
                "rmse_mean": round(float(np.mean(subj_rmses[h])), 2),
                "rmse_sd": round(float(np.std(subj_rmses[h], ddof=1)), 2),
            })

    # 6 & 7. LSTM and Transformer (non-mech)
    print("Loading LSTM and Transformer (non-mech)...")
    df_base = pd.read_csv("results/tables/baselines_lstm_a0nomech.csv")
    for _, r in df_base[df_base["model"] == "LSTM"].iterrows():
        all_rows.append({
            "model": "LSTM (non-mech)",
            "horizon_min": int(r["horizon_min"]),
            "mae_mean": round(float(r["mae_mean"]), 2),
            "mae_sd": round(float(r["mae_sd"]), 2),
            "rmse_mean": round(float(r["rmse_mean"]), 2),
            "rmse_sd": round(float(r["rmse_sd"]), 2),
        })
    for _, r in df_base[df_base["model"] == "A0_no_mech"].iterrows():
        all_rows.append({
            "model": "Transformer (non-mech)",
            "horizon_min": int(r["horizon_min"]),
            "mae_mean": round(float(r["mae_mean"]), 2),
            "mae_sd": round(float(r["mae_sd"]), 2),
            "rmse_mean": round(float(r["rmse_mean"]), 2),
            "rmse_sd": round(float(r["rmse_sd"]), 2),
        })

    # 8. Transformer (all features, A0)
    print("Loading Transformer (all features, A0)...")
    df_a0 = pd.read_csv("results/ablations/per_subject_A0.csv")
    for h in horizons:
        sub = df_a0[df_a0["horizon_min"] == h]
        all_rows.append({
            "model": "Transformer (all features, A0)",
            "horizon_min": h,
            "mae_mean": round(float(sub["mae"].mean()), 2),
            "mae_sd": round(float(sub["mae"].std(ddof=1)), 2),
            "rmse_mean": round(float(sub["rmse"].mean()), 2),
            "rmse_sd": round(float(sub["rmse"].std(ddof=1)), 2),
        })

    # 9. Hybrid physics-guided (A7)
    print("Loading Digital Twin (Hybrid, A7)...")
    df_a7 = pd.read_csv("results/ablations/per_subject_A7.csv")
    for h in horizons:
        sub = df_a7[df_a7["horizon_min"] == h]
        all_rows.append({
            "model": "Digital Twin (Hybrid, A7)",
            "horizon_min": h,
            "mae_mean": round(float(sub["mae"].mean()), 2),
            "mae_sd": round(float(sub["mae"].std(ddof=1)), 2),
            "rmse_mean": round(float(sub["rmse"].mean()), 2),
            "rmse_sd": round(float(sub["rmse"].std(ddof=1)), 2),
        })

    df_out = pd.DataFrame(all_rows)
    print("\n=== Comprehensive Baselines Table (Final) ===")
    print(df_out.to_string(index=False))

    out_csv = Path("paper/digital-twin-v4/data/comprehensive_baselines.csv")
    df_out.to_csv(out_csv, index=False)
    df_out.to_csv("results/tables/comprehensive_baselines.csv", index=False)
    print(f"\nSaved to {out_csv} and results/tables/comprehensive_baselines.csv")


if __name__ == "__main__":
    main()
