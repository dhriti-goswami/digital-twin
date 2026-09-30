import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.stats as stats
from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.splits import official_split
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.physio.bergman import integrate_glucose
from twin.physio.params import PatientParams, BOUNDS, POPULATION_MEANS, unconstrained_to_params

def main():
    config = Config()
    config.train.batch_size = 256
    config.train.num_workers = 0
    corpus = load_corpus(config)
    train_sets = {key: value.windows for key, value in corpus['train'].items()}
    test_sets = {key: value.windows for key, value in corpus['test'].items()}
    fold = official_split(list(train_sets.values()), list(test_sets.values()),
                          val_fraction=config.split.val_fraction,
                          purge_steps=config.split.purge_steps)
    scaler = fit_scaler(fold, corpus)
    train_ds = build_dataset(fold, 'train', corpus, scaler, config)
    test_ds = build_dataset(fold, 'test', corpus, scaler, config)
    train_loader = build_loader(train_ds, config, shuffle=False)
    test_loader = build_loader(test_ds, config, shuffle=False)

    device = config.resolve_device()
    model = PhysicsGuidedForecaster(35, config).to(device)
    model.eval()

    colloc = model.spline.collocation_min.to(device)
    dt = float(colloc[1] - colloc[0])
    horizon_indices = [int(torch.argmin((colloc - m).abs())) for m in model.spline.horizon_min]

    # Collect batches per subject on train
    train_by_subj = {s: [] for s in range(len(train_ds.subject_ids))}
    for batch in train_loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for s in range(len(train_ds.subject_ids)):
            mask = batch_dev['subject_index'] == s
            if mask.any():
                train_by_subj[s].append({k: v[mask] for k, v in batch_dev.items() if isinstance(v, torch.Tensor)})

    # Fit (p1, p2, p3) per subject
    fitted_si_bergman = {}
    fitted_params_per_subj = {}

    for s, subject_id in enumerate(train_ds.subject_ids):
        batches = train_by_subj[s]
        # Parameterize unconstrained logits for (p1, p2, p3)
        # Initialize at population mean
        # p1 in [0.0, 0.030], pop 0.013
        # p2 in [0.005, 0.10], pop 0.025
        # p3 in [1e-6, 3e-5], pop 6.25e-6
        # Let logit = 0 be population center
        u_p1 = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))
        u_p2 = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))
        u_p3 = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))

        opt = torch.optim.Adam([u_p1, u_p2, u_p3], lr=0.05)

        # Scale logit to bound
        b_p1 = BOUNDS['p1']
        b_p2 = BOUNDS['p2']
        b_p3 = BOUNDS['p3']

        for epoch in range(15):
            total_loss = 0.0
            n_samples = 0
            for b in batches:
                opt.zero_grad()
                p1_val = b_p1.low + (b_p1.high - b_p1.low) * torch.sigmoid(u_p1)
                p2_val = b_p2.low + (b_p2.high - b_p2.low) * torch.sigmoid(u_p2)
                p3_val = b_p3.low + (b_p3.high - b_p3.low) * torch.sigmoid(u_p3)

                B_size = len(b['anchor_glucose'])
                # Construct PatientParams for this batch
                pop = model.resolve_params(
                    b['features'],
                    basal_glucose=b['basal_glucose'],
                    body_weight_kg=b['body_weight_kg'],
                    basal_insulin_rate=b['basal_insulin_rate'],
                    use_population=True
                )
                params = PatientParams(
                    p1=p1_val.expand(B_size),
                    p2=p2_val.expand(B_size),
                    p3=p3_val.expand(B_size),
                    n=pop.n,
                    V_G=pop.V_G,
                    V_I=pop.V_I,
                    tmax_I=pop.tmax_I,
                    k_gri=pop.k_gri,
                    k_abs=pop.k_abs,
                    f=pop.f,
                    G_b=pop.G_b,
                    I_b=pop.I_b,
                )
                insulin_action, appearance = model.mechanistic_state(
                    params, b['insulin_rate'], b['carb_rate']
                )
                mechanistic = integrate_glucose(
                    b['anchor_glucose'],
                    insulin_action,
                    appearance,
                    params,
                    dt=dt
                )
                preds = mechanistic[:, horizon_indices]
                # MSE loss + small prior penalty (MAP)
                loss = nn.functional.mse_loss(preds, b['targets']) + 0.01 * (u_p1**2 + u_p2**2 + u_p3**2)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach()) * B_size
                n_samples += B_size

        final_p2 = float(b_p2.low + (b_p2.high - b_p2.low) * torch.sigmoid(u_p2).detach())
        final_p3 = float(b_p3.low + (b_p3.high - b_p3.low) * torch.sigmoid(u_p3).detach())
        si_fit = final_p3 / final_p2
        fitted_si_bergman[subject_id] = si_fit
        fitted_params_per_subj[subject_id] = (float(u_p1.detach()), float(u_p2.detach()), float(u_p3.detach()))
        print(f"Subject {subject_id}: p2={final_p2:.4f}, p3={final_p3:.2e}, S_I_Bergman={si_fit:.2e}")

    # Evaluate on test set
    all_preds = []
    all_targets = []
    all_subjs = []
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # build per-sample params using fitted values
            B_size = len(batch_dev['anchor_glucose'])
            subjs_in_batch = batch_dev['subject_index'].cpu().numpy()
            p1_list, p2_list, p3_list = [], [], []
            for s_idx in subjs_in_batch:
                s_id = test_ds.subject_ids[s_idx]
                u1, u2, u3 = fitted_params_per_subj[s_id]
                p1_list.append(b_p1.low + (b_p1.high - b_p1.low) * (1 / (1 + np.exp(-u1))))
                p2_list.append(b_p2.low + (b_p2.high - b_p2.low) * (1 / (1 + np.exp(-u2))))
                p3_list.append(b_p3.low + (b_p3.high - b_p3.low) * (1 / (1 + np.exp(-u3))))

            pop = model.resolve_params(
                batch_dev['features'],
                basal_glucose=batch_dev['basal_glucose'],
                body_weight_kg=batch_dev['body_weight_kg'],
                basal_insulin_rate=batch_dev['basal_insulin_rate'],
                use_population=True
            )
            params = PatientParams(
                p1=torch.tensor(p1_list, device=device, dtype=pop.p1.dtype),
                p2=torch.tensor(p2_list, device=device, dtype=pop.p2.dtype),
                p3=torch.tensor(p3_list, device=device, dtype=pop.p3.dtype),
                n=pop.n,
                V_G=pop.V_G,
                V_I=pop.V_I,
                tmax_I=pop.tmax_I,
                k_gri=pop.k_gri,
                k_abs=pop.k_abs,
                f=pop.f,
                G_b=pop.G_b,
                I_b=pop.I_b,
            )
            insulin_action, appearance = model.mechanistic_state(
                params, batch_dev['insulin_rate'], batch_dev['carb_rate']
            )
            mechanistic = integrate_glucose(
                batch_dev['anchor_glucose'],
                insulin_action,
                appearance,
                params,
                dt=dt
            )
            all_preds.append(mechanistic[:, horizon_indices].cpu().numpy())
            all_targets.append(batch['targets'].cpu().numpy())
            all_subjs.append(subjs_in_batch)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    subjs = np.concatenate(all_subjs, axis=0)

    rows = []
    for h_idx, h in enumerate([30, 60, 90, 120]):
        subj_maes = []
        subj_rmses = []
        for s_idx in range(len(test_ds.subject_ids)):
            mask = subjs == s_idx
            err = preds[mask, h_idx] - targets[mask, h_idx]
            subj_maes.append(np.mean(np.abs(err)))
            subj_rmses.append(np.sqrt(np.mean(err**2)))
        rows.append({
            'model': 'Fitted Bergman ODE (per-subject NLS/MAP)',
            'horizon_min': h,
            'mae_mean': np.mean(subj_maes),
            'mae_sd': np.std(subj_maes, ddof=1),
            'rmse_mean': np.mean(subj_rmses),
            'rmse_sd': np.std(subj_rmses, ddof=1),
        })
    print("\nFitted Bergman ODE performance on test set:")
    print(pd.DataFrame(rows).to_string(index=False))

    # Compare with learned S_I from A7
    diag = np.load('artifacts/official/abl-A7/official/test_diagnostics.npz')
    learned_si = {}
    for i, s_id in enumerate(test_ds.subject_ids):
        mask = diag['subject_index'] == i
        learned_si[s_id] = float(np.median(diag['insulin_sensitivity'][mask]))

    comp_df = pd.DataFrame([
        {'subject_id': s_id, 'S_I_Bergman': fitted_si_bergman[s_id], 'S_I_learned': learned_si[s_id]}
        for s_id in test_ds.subject_ids
    ])
    print("\nComparison of S_I Bergman (classic fit) vs S_I Learned (A7):")
    print(comp_df.to_string(index=False))

    r_spear, p_spear = stats.spearmanr(comp_df['S_I_Bergman'], comp_df['S_I_learned'])
    r_pear, p_pear = stats.pearsonr(comp_df['S_I_Bergman'], comp_df['S_I_learned'])
    print(f"\nAgreement S_I_Bergman vs S_I_Learned: Spearman rho = {r_spear:.4f} (p = {p_spear:.4f}), Pearson r = {r_pear:.4f} (p = {p_pear:.4f})")

if __name__ == '__main__':
    main()
