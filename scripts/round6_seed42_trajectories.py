# Round 6 helper. The seed-42 A0/A7 checkpoints (artifacts/official/abl-*) were trained before the quantile head was
# added (commit 1a275e8), so the current model class cannot load them. This script loads them with the original
# code (git worktree of 7eb9ad2) and stores trajectory statistics; it asserts nothing itself but prints the max abs
# difference to the saved predictions.npz (observed: 0.0 for both arms).
# Usage: git worktree add --detach /path/wt 7eb9ad2; ln -s $PWD/OhioT1DM /path/wt/OhioT1DM;
#        python scripts/round6_seed42_trajectories.py /path/wt
# Runs seed-42 (pre-quantile-head, commit 7eb9ad2) checkpoints with the OLD code to get
# trajectory stats; verifies predictions against the saved predictions.npz.
import sys, numpy as np, torch
WT = sys.argv[1]  # git worktree of commit 7eb9ad2 (pre-quantile-head code) with OhioT1DM symlinked in
sys.path.insert(0, WT)
import os; os.chdir(WT)
from twin.config import Config
from twin.data.dataset import build_dataset, build_loader, fit_scaler, load_corpus, FittedScaler
from twin.data.features import N_FEATURES
from twin.data.splits import official_split
from twin.models.forecaster import PhysicsGuidedForecaster
import twin
print(twin.__file__)
REPO=os.environ.get("REPO_ROOT", "/home/sammyyakk/projects/digital-twin")
base = Config.from_yaml("configs/official-small.yaml"); base.train.batch_size=256; base.train.num_workers=0
corpus = load_corpus(base, cache_dir=None)
tr={k:v.windows for k,v in corpus["train"].items()}; te={k:v.windows for k,v in corpus["test"].items()}
fold = official_split(list(tr.values()), list(te.values()), val_fraction=base.split.val_fraction, purge_steps=base.split.purge_steps)
scaler = fit_scaler(fold, corpus)
loader = build_loader(build_dataset(fold,"test",corpus,scaler,base), base, shuffle=False)
ids = loader.dataset.subject_ids
dev="cuda"
for arm in ("A0","A7"):
    ck = torch.load(f"{REPO}/artifacts/official/abl-{arm}/official/best_model.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    model = PhysicsGuidedForecaster(N_FEATURES, cfg).to(dev); model.load_state_dict(ck["model_state_dict"]); model.eval()
    hz,mx,mn,sl=[],[],[],[]
    with torch.no_grad():
        for b in loader:
            b={k:v.to(dev) for k,v in b.items()}
            o=model(b); c=o.collocation.float()
            dt=float(model.spline.collocation_min[1]-model.spline.collocation_min[0])
            hz.append(o.horizons.float().cpu().numpy()); mx.append(c.max(1).values.cpu().numpy()); mn.append(c.min(1).values.cpu().numpy())
            sl.append(((c[:,1:]-c[:,:-1]).abs()/dt).max(1).values.cpu().numpy())
    hz=np.concatenate(hz); saved=np.load(f"{REPO}/artifacts/official/abl-{arm}/official/test/abl-{arm}/predictions.npz")
    ref=np.concatenate([saved[f"pred__{s}"] for s in ids])
    print(arm, hz.shape, ref.shape, "max abs diff vs saved predictions:", np.abs(hz-ref).max(), "dt", dt)
    np.savez(f"{REPO}/artifacts/round6_seed42_traj_{arm}.npz", pred=hz, cmax=np.concatenate(mx), cmin=np.concatenate(mn), slope=np.concatenate(sl), ref=ref)
