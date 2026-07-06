"""
Fast vectorized feature engineering for ODE-simulated T1D data.

Replaces the slow loop-based GlucoseFeatureEngine for simulation CSVs,
computing all features with numpy/pandas vectorized operations.

Input columns (from generate_training_data.py):
  t_min, cgm_mg_dl, insulin_u_h, cho_g, basal_u_h

Output: (N, seq_len, n_features) sequences ready for the Transformer.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DT = 5   # minutes per CGM step


def _rolling(arr: np.ndarray, window: int, func: str) -> np.ndarray:
    """Rolling statistic using pandas (vectorized)."""
    s = pd.Series(arr)
    r = s.rolling(window, min_periods=1)
    if func == "mean":
        return r.mean().values
    elif func == "std":
        return r.std(ddof=0).values
    elif func == "min":
        return r.min().values
    elif func == "max":
        return r.max().values
    raise ValueError(func)


def _compute_iob(insulin_u_h: np.ndarray, basal_u_h: np.ndarray) -> np.ndarray:
    """
    Insulin On Board — vectorized via convolution.

    Models rapid-acting insulin with a Lispro/Aspart activity curve.
    Duration of action = 4 h = 48 steps at 5-min intervals.
    Activity curve: triangular peak at ~90 min, total 4 h.
    """
    n = len(insulin_u_h)
    doi = 48          # duration-of-action in 5-min steps (4 hours)
    peak = 18         # peak activity at step 18 = 90 min

    # Activity fraction remaining after k steps
    curve = np.zeros(doi)
    for k in range(doi):
        if k < peak:
            curve[k] = k / peak
        else:
            curve[k] = max(0, 1 - (k - peak) / (doi - peak))

    # Bolus = total dose above basal (U/h / 12 = U per 5-min step)
    bolus_per_step = np.maximum(0.0, (insulin_u_h - basal_u_h) / 12.0)

    # IOB = convolution of bolus with the remaining fraction
    iob = np.convolve(bolus_per_step, curve[::-1], mode="full")[:n]
    return iob.astype(np.float32)


def _compute_cob(cho_g: np.ndarray) -> np.ndarray:
    """
    Carbs On Board — vectorized via convolution.

    Assumes ~3-hour absorption (36 steps at 5-min).
    Simple exponential decay: cob(t) = Σ_{k≤t} cho_g[k] * exp(-(t-k)/τ)
    where τ = 36 steps = 3 h.
    """
    n = len(cho_g)
    tau = 36  # steps
    max_cob_horizon = 72  # 6 hours of lookback

    curve = np.exp(-np.arange(max_cob_horizon) / tau)
    cob = np.convolve(cho_g, curve, mode="full")[:n]
    return cob.astype(np.float32)


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build feature matrix from one patient's simulation DataFrame.

    Returns array of shape (N, n_features) where N = len(df).
    Feature names are returned alongside.
    """
    g = df["cgm_mg_dl"].values.astype(np.float32)
    ins = df["insulin_u_h"].values.astype(np.float32)
    cho = df["cho_g"].values.astype(np.float32)
    bas = df["basal_u_h"].values.astype(np.float32)
    t_min = df["t_min"].values.astype(np.float32)

    # ── CGM features ──────────────────────────────────────────────────────────
    roc_5 = np.diff(g, prepend=g[0])             # rate of change (5-min)
    roc_15 = np.diff(g, n=3, prepend=[g[0]] * 3) # roc 15-min
    roc_30 = np.diff(g, n=6, prepend=[g[0]] * 6) # roc 30-min

    g_mean_1h = _rolling(g, 12, "mean")
    g_std_1h = _rolling(g, 12, "std")
    g_min_1h = _rolling(g, 12, "min")
    g_max_1h = _rolling(g, 12, "max")
    g_mean_2h = _rolling(g, 24, "mean")
    g_std_2h = _rolling(g, 24, "std")
    g_cv_1h = np.where(g_mean_1h > 0, g_std_1h / g_mean_1h * 100, 0)
    g_cv_2h = np.where(g_mean_2h > 0, g_std_2h / g_mean_2h * 100, 0)
    g_range_1h = g_max_1h - g_min_1h
    is_hypo = (g < 70).astype(np.float32)
    is_hyper = (g > 180).astype(np.float32)
    is_inrange = ((g >= 70) & (g <= 180)).astype(np.float32)
    hypo_1h = _rolling(is_hypo, 12, "mean") * 12
    hyper_1h = _rolling(is_hyper, 12, "mean") * 12

    # ── Temporal features ────────────────────────────────────────────────────
    t_in_day = t_min % 1440   # minutes within day
    hour_cos = np.cos(2 * np.pi * t_in_day / 1440).astype(np.float32)
    hour_sin = np.sin(2 * np.pi * t_in_day / 1440).astype(np.float32)
    is_weekend = (((t_min // 1440) % 7) >= 5).astype(np.float32)
    day_frac = (t_in_day / 1440).astype(np.float32)
    is_night = ((t_in_day < 360) | (t_in_day > 1320)).astype(np.float32)

    # ── Insulin / IOB features ────────────────────────────────────────────────
    iob = _compute_iob(ins, bas)
    total_ins_1h = _rolling(ins, 12, "mean")  # mean U/h over last hour
    total_ins_2h = _rolling(ins, 24, "mean")

    # ── Meal / COB features ───────────────────────────────────────────────────
    cob = _compute_cob(cho)
    carbs_1h = _rolling(cho, 12, "mean") * 12  # total g in last hour
    carbs_2h = _rolling(cho, 24, "mean") * 24

    # Time since last meal (minutes since last cho_g > 0 event)
    last_meal_step = np.full(len(cho), 240.0, dtype=np.float32)
    last_meal_idx = -1
    for i in range(len(cho)):
        if cho[i] > 0:
            last_meal_idx = i
        if last_meal_idx >= 0:
            last_meal_step[i] = min(240.0, float((i - last_meal_idx) * DT))

    # ── Activity placeholder (zeros — no activity data in ODE sim) ────────────
    is_exercise = np.zeros(len(g), dtype=np.float32)
    exercise_intensity = np.zeros(len(g), dtype=np.float32)
    time_since_exercise = np.full(len(g), 240.0, dtype=np.float32)
    exercise_min_2h = np.zeros(len(g), dtype=np.float32)

    # ── Stack ─────────────────────────────────────────────────────────────────
    features = np.column_stack([
        # CGM (16 features)
        g, roc_5, roc_15, roc_30,
        g_mean_1h, g_std_1h, g_min_1h, g_max_1h,
        g_mean_2h, g_std_2h, g_cv_1h, g_cv_2h, g_range_1h,
        is_hypo, is_hyper, is_inrange, hypo_1h, hyper_1h,
        # Temporal (6 features)
        hour_cos, hour_sin, is_weekend, day_frac, is_night,
        t_in_day / 1440,
        # Insulin (3 features)
        iob, total_ins_1h, total_ins_2h,
        # Meal (4 features)
        cob, carbs_1h, carbs_2h, last_meal_step,
        # Activity (4 features)
        is_exercise, exercise_intensity, time_since_exercise, exercise_min_2h,
    ])

    return features.astype(np.float32)


FEATURE_NAMES = [
    # CGM (18)
    "glucose_mg_dl", "glucose_roc_5min", "glucose_roc_15min", "glucose_roc_30min",
    "glucose_mean_1h", "glucose_std_1h", "glucose_min_1h", "glucose_max_1h",
    "glucose_mean_2h", "glucose_std_2h", "glucose_cv_1h", "glucose_cv_2h", "glucose_range_1h",
    "is_hypoglycemic", "is_hyperglycemic", "is_in_range", "hypo_events_1h", "hyper_events_1h",
    # Temporal (6)
    "hour_cos", "hour_sin", "is_weekend", "day_frac", "is_night", "time_frac_day",
    # Insulin (3)
    "iob_rapid", "mean_insulin_1h", "mean_insulin_2h",
    # Meal (4)
    "cob", "carbs_1h", "carbs_2h", "time_since_last_meal",
    # Activity (4)
    "is_exercising", "exercise_intensity", "time_since_exercise", "exercise_minutes_2h",
]

N_FEATURES = len(FEATURE_NAMES)  # 35


def make_sequences(
    features: np.ndarray,
    glucose: np.ndarray,
    seq_len: int = 24,
    horizons: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice feature matrix into (input_seq, target) pairs.

    Returns:
        X: (n_samples, seq_len, n_features)
        y: (n_samples, n_horizons)
    """
    if horizons is None:
        horizons = [6, 12, 18, 24]    # 30, 60, 90, 120 min

    max_h = max(horizons)
    n = len(features) - seq_len - max_h
    if n <= 0:
        return np.empty((0, seq_len, features.shape[1])), np.empty((0, len(horizons)))

    idx = np.arange(seq_len, seq_len + n)
    X = np.stack([features[i - seq_len: i] for i in idx])
    y = np.array([[glucose[i + h] for h in horizons] for i in idx])

    return X.astype(np.float32), y.astype(np.float32)


def load_all_patients(
    sim_dir: Path,
    seq_len: int = 24,
    horizons: list = None,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Load all ODE patient CSVs, build features, split patients into train/val.

    Returns a dict with:
      train_X, train_y — (N, seq_len, n_features) and (N, n_horizons)
      val_X, val_y
      scaler — fitted StandardScaler
      feature_names — list of strings
    """
    sim_dir = Path(sim_dir)
    files = sorted(sim_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {sim_dir}")

    if horizons is None:
        horizons = [6, 12, 18, 24]

    # Patient-level split
    rng = np.random.RandomState(seed)
    idxs = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_split))
    val_files = {files[i] for i in idxs[:n_val]}
    train_files = {files[i] for i in idxs[n_val:]}

    logger.info("Patients — train: %d  val: %d", len(train_files), len(val_files))

    def _process(fset):
        Xs, ys = [], []
        for f in sorted(fset):
            df = pd.read_csv(f)
            feats = build_features(df)
            glucose = df["cgm_mg_dl"].values.astype(np.float32)
            X, y = make_sequences(feats, glucose, seq_len, horizons)
            if len(X) > 0:
                Xs.append(X)
                ys.append(y)
        if not Xs:
            return np.empty((0, seq_len, N_FEATURES)), np.empty((0, len(horizons)))
        return np.concatenate(Xs), np.concatenate(ys)

    logger.info("Building training sequences …")
    train_X, train_y = _process(train_files)
    logger.info("Building validation sequences …")
    val_X, val_y = _process(val_files)

    logger.info(
        "Sequences — train: %d  val: %d  features: %d",
        len(train_X), len(val_X), train_X.shape[2],
    )

    # Scale on training only
    scaler = StandardScaler()
    flat = train_X.reshape(-1, train_X.shape[2])
    scaler.fit(flat)

    train_X = scaler.transform(flat).reshape(train_X.shape).astype(np.float32)
    val_flat = val_X.reshape(-1, val_X.shape[2])
    val_X = scaler.transform(val_flat).reshape(val_X.shape).astype(np.float32)

    return dict(
        train_X=train_X, train_y=train_y,
        val_X=val_X, val_y=val_y,
        scaler=scaler,
        feature_names=FEATURE_NAMES,
        n_features=N_FEATURES,
    )
