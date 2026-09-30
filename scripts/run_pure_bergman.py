import torch
import numpy as np
import pandas as pd
from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.splits import official_split
from twin.models.forecaster import PhysicsGuidedForecaster
from twin.physio.bergman import integrate_glucose

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
    test_ds = build_dataset(fold, 'test', corpus, scaler, config)
    loader = build_loader(test_ds, config, shuffle=False)

    device = config.resolve_device()
    model = PhysicsGuidedForecaster(35, config).to(device)
    model.eval()

    colloc = model.spline.collocation_min.to(device)
    dt = float(colloc[1] - colloc[0])
    horizon_indices = [int(torch.argmin((colloc - m).abs())) for m in model.spline.horizon_min]

    all_preds = []
    all_targets = []
    all_subjs = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            params = model.resolve_params(
                batch['features'],
                basal_glucose=batch['basal_glucose'],
                body_weight_kg=batch['body_weight_kg'],
                basal_insulin_rate=batch['basal_insulin_rate'],
                use_population=True
            )
            insulin_action, appearance = model.mechanistic_state(
                params, batch['insulin_rate'], batch['carb_rate']
            )
            mechanistic = integrate_glucose(
                batch['anchor_glucose'],
                insulin_action,
                appearance,
                params,
                dt=dt
            )
            all_preds.append(mechanistic[:, horizon_indices].cpu().numpy())
            all_targets.append(batch['targets'].cpu().numpy())
            all_subjs.append(batch['subject_index'].cpu().numpy())

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
            'model': 'Pure Bergman ODE (population)',
            'horizon_min': h,
            'mae_mean': np.mean(subj_maes),
            'mae_sd': np.std(subj_maes, ddof=1),
            'rmse_mean': np.mean(subj_rmses),
            'rmse_sd': np.std(subj_rmses, ddof=1),
        })

    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
