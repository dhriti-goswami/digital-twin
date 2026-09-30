import torch
import numpy as np
from scipy.optimize import minimize
from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus
from twin.data.splits import official_split
from twin.physio.bergman import integrate_glucose
from twin.physio.params import PatientParams, BOUNDS, POPULATION_MEANS

config = Config()
corpus = load_corpus(config)
train_sets = {key: value.windows for key, value in corpus['train'].items()}
test_sets = {key: value.windows for key, value in corpus['test'].items()}
fold = official_split(list(train_sets.values()), list(test_sets.values()),
                      val_fraction=config.split.val_fraction,
                      purge_steps=config.split.purge_steps)
scaler = fit_scaler(fold, corpus)
train_ds = build_dataset(fold, 'train', corpus, scaler, config)
loader = build_loader(train_ds, config, shuffle=False)

# Collect all windows for subject 0
sub_0_batches = []
for batch in loader:
    mask = batch['subject_index'] == 0
    if mask.any():
        sub_0_batches.append({
            'anchor_glucose': batch['anchor_glucose'][mask],
            'targets': batch['targets'][mask],
            'insulin_rate': batch['insulin_rate'][mask],
            'carb_rate': batch['carb_rate'][mask],
            'basal_glucose': batch['basal_glucose'][mask],
            'basal_insulin_rate': batch['basal_insulin_rate'][mask],
            'body_weight_kg': batch['body_weight_kg'][mask],
        })

print(f"Subject 0 has {sum(len(b['anchor_glucose']) for b in sub_0_batches)} train windows")
