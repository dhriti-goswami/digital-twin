import torch
import numpy as np
from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.splits import official_split
from twin.physio.bergman import integrate_glucose
from twin.physio.params import PatientParams, POPULATION_MEANS, BOUNDS
from twin.models.forecaster import PhysicsGuidedForecaster

config = Config()
corpus = load_corpus(config)
train_sets = {key: value.windows for key, value in corpus['train'].items()}
test_sets = {key: value.windows for key, value in corpus['test'].items()}
fold = official_split(list(train_sets.values()), list(test_sets.values()),
                      val_fraction=config.split.val_fraction,
                      purge_steps=config.split.purge_steps)
scaler = fit_scaler(fold, corpus)
# Single process loader for quick test
config.train.num_workers = 0
train_ds = build_dataset(fold, 'train', corpus, scaler, config)
loader = build_loader(train_ds, config, shuffle=False)

# Collect batches for subject 540 (index 0)
s0_batches = []
for batch in loader:
    mask = batch['subject_index'] == 0
    if mask.any():
        s0_batches.append({k: v[mask] for k, v in batch.items() if isinstance(v, torch.Tensor)})

print(f"Collected {len(s0_batches)} batches for subject 540, total windows: {sum(len(b['targets']) for b in s0_batches)}")

# Test evaluation of Bergman ODE with population params on subject 540
model = PhysicsGuidedForecaster(35, config)
colloc = model.spline.collocation_min
dt = float(colloc[1] - colloc[0])
horizon_indices = [int(torch.argmin((colloc - m).abs())) for m in model.spline.horizon_min]

all_preds = []
all_targets = []
with torch.no_grad():
    for batch in s0_batches:
        params = model.resolve_params(
            None,
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
        all_preds.append(mechanistic[:, horizon_indices])
        all_targets.append(batch['targets'])

preds = torch.cat(all_preds, dim=0)
targets = torch.cat(all_targets, dim=0)
mae = (preds - targets).abs().mean(dim=0)
print("Population Bergman MAE on 540 training set:", mae.tolist())
