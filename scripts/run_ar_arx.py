import numpy as np
import pandas as pd
from twin.config import Config
from twin.data.dataset import load_corpus
from twin.data.splits import official_split

config = Config()
corpus = load_corpus(config)
train_sets = {key: value.windows for key, value in corpus['train'].items()}
test_sets = {key: value.windows for key, value in corpus['test'].items()}
fold = official_split(list(train_sets.values()), list(test_sets.values()),
                      val_fraction=config.split.val_fraction,
                      purge_steps=config.split.purge_steps)

subjects = sorted(list(corpus['test'].keys()))

def extract_ar(data, indices, n_lags=6):
    g = data.frame['glucose_filled'].to_numpy()
    anchors = data.windows.anchors[indices]
    X = np.stack([g[anchors - lag] for lag in range(n_lags)], axis=1)
    X = np.hstack([np.ones((len(X), 1)), X])
    Y = data.windows.targets[indices]
    return X, Y

def extract_arx(data, indices, n_lags=6):
    g = data.frame['glucose_filled'].to_numpy()
    ins = data.frame['bolus_u_per_min'].to_numpy() + data.frame['basal_u_per_min'].to_numpy()
    carbs = data.frame['carbs_mg_per_min'].to_numpy()
    anchors = data.windows.anchors[indices]
    feats = [np.ones(len(indices))]
    for lag in range(n_lags):
        feats.append(g[anchors - lag])
        feats.append(ins[anchors - lag])
        feats.append(carbs[anchors - lag])
    X = np.column_stack(feats)
    Y = data.windows.targets[indices]
    return X, Y

rows = []
for model_name, extract_fn in [('AR(6)', extract_ar), ('ARX(6)', extract_arx)]:
    subj_maes = {30: [], 60: [], 90: [], 120: []}
    subj_rmses = {30: [], 60: [], 90: [], 120: []}
    
    for s in subjects:
        train_data = corpus['train'][s]
        test_data = corpus['test'][s]
        train_sel = next(sel for sel in fold.train if sel.subject_id == s)
        test_sel = next(sel for sel in fold.test if sel.subject_id == s)
        
        X_train, Y_train = extract_fn(train_data, train_sel.indices)
        X_test, Y_test = extract_fn(test_data, test_sel.indices)
        
        # Ridge regularized least squares
        W = np.linalg.solve(X_train.T @ X_train + 1.0 * np.eye(X_train.shape[1]), X_train.T @ Y_train)
        Y_pred = np.clip(X_test @ W, 40.0, 400.0)
        
        err = Y_pred - Y_test
        mae = np.mean(np.abs(err), axis=0)
        rmse = np.sqrt(np.mean(err**2, axis=0))
        for idx, h in enumerate([30, 60, 90, 120]):
            subj_maes[h].append(mae[idx])
            subj_rmses[h].append(rmse[idx])
            
    for h in [30, 60, 90, 120]:
        rows.append({
            'model': model_name,
            'horizon_min': h,
            'mae_mean': np.mean(subj_maes[h]),
            'mae_sd': np.std(subj_maes[h], ddof=1),
            'rmse_mean': np.mean(subj_rmses[h]),
            'rmse_sd': np.std(subj_rmses[h], ddof=1),
        })

df_res = pd.DataFrame(rows)
print(df_res.to_string(index=False))
