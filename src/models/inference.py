"""
Model inference service for glucose prediction.

Loads the trained model and provides prediction interface.
The checkpoint stores feature_names and scaler so inference
always matches exactly what the model was trained on.
"""

import logging
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.models.glucose_predictor import GlucosePredictor
from src.data.preprocessing import GlucoseFeatureEngine

logger = logging.getLogger(__name__)


class GlucoseInferenceService:
    """Production inference service for glucose prediction."""

    def __init__(
        self,
        model_path: str = "checkpoints/best_model.pt",
        device: str = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.config = None
        self.feature_engine = None
        self.feature_names = None   # Loaded from checkpoint
        self.input_size = None      # Actual model input size
        self.model_loaded = False

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> bool:
        """Load trained model from checkpoint."""
        from dataclasses import dataclass

        path = Path(model_path)
        if not path.exists():
            logger.warning(f"Model not found at {model_path}")
            return False

        try:
            @dataclass
            class TrainingConfig:
                batch_size: int = 32
                learning_rate: float = 1e-3
                weight_decay: float = 0.01
                epochs: int = 100
                early_stopping_patience: int = 15
                val_split: float = 0.2
                gradient_clip: float = 1.0
                model_type: str = "transformer"
                hidden_size: int = 128
                num_layers: int = 4
                num_heads: int = 8
                dropout: float = 0.1
                use_pinn: bool = True
                pinn_lambda: float = 0.1
                checkpoint_dir: str = "./checkpoints"

            import __main__
            __main__.TrainingConfig = TrainingConfig

            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.config = checkpoint.get("config")
            metrics = checkpoint.get("metrics", {})
            state_dict = checkpoint["model_state_dict"]

            # Infer input size from first linear layer weight
            self.input_size = None
            for key, tensor in state_dict.items():
                if "input" in key and "weight" in key and len(tensor.shape) == 2:
                    self.input_size = tensor.shape[1]
                    break
            if self.input_size is None:
                self.input_size = 43  # fallback

            model_type = getattr(self.config, "model_type", "transformer")
            hidden_size = getattr(self.config, "hidden_size", 128)
            num_layers = getattr(self.config, "num_layers", 4)

            self.model = GlucosePredictor(
                input_size=self.input_size,
                model_type=model_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_horizons=4,
                use_pinn=True,
            ).to(self.device)

            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Load feature names saved during training (if present)
            self.feature_names = checkpoint.get("feature_names", None)

            # Load scaler saved during training (if present)
            self.feature_engine = GlucoseFeatureEngine(
                sequence_length=24,
                prediction_horizons=[6, 12, 18, 24],
                cgm_interval_minutes=5,
            )
            scaler_bytes = checkpoint.get("scaler", None)
            if scaler_bytes is not None:
                self.feature_engine.scaler = pickle.loads(scaler_bytes)
                self.feature_engine._is_fitted = True
                logger.info("Scaler loaded from checkpoint.")
            else:
                logger.warning(
                    "No scaler found in checkpoint. Predictions will be on unscaled data "
                    "until the model is retrained with scripts/train_model.py."
                )

            self.model_loaded = True
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model type: {model_type}, Input features: {self.input_size}")
            if self.feature_names:
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            logger.info(f"Validation MAE: {metrics.get('val_mae', 'N/A')}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def prepare_features(
        self,
        cgm_df: pd.DataFrame,
        insulin_df: pd.DataFrame = None,
        meals_df: pd.DataFrame = None,
        patient_profile: dict = None,
    ) -> np.ndarray:
        """
        Prepare features from raw data.

        Always produces the full 47-feature set then selects only the columns
        the model was trained on (via checkpoint feature_names) or trims to
        input_size if feature_names were not saved (backward compat).
        """
        if insulin_df is None:
            insulin_df = pd.DataFrame()
        if meals_df is None:
            meals_df = pd.DataFrame()

        if "timestamp" in cgm_df.columns:
            cgm_df = cgm_df.rename(columns={"timestamp": "time"})
        cgm_df["time"] = pd.to_datetime(cgm_df["time"], utc=True).dt.tz_localize(None)

        if "trend" not in cgm_df.columns:
            cgm_df["trend"] = "STABLE"

        # Build full feature set
        cgm_features = self.feature_engine.create_cgm_features(cgm_df)
        temporal_features = self.feature_engine.create_temporal_features(cgm_df["time"])
        insulin_features = self.feature_engine.create_insulin_features(cgm_df["time"], insulin_df)
        meal_features = self.feature_engine.create_meal_features(cgm_df["time"], meals_df)
        activity_features = self.feature_engine.create_activity_features(cgm_df["time"], pd.DataFrame())

        all_features = pd.concat([
            cgm_features.reset_index(drop=True),
            temporal_features.reset_index(drop=True),
            insulin_features.reset_index(drop=True),
            meal_features.reset_index(drop=True),
            activity_features.reset_index(drop=True),
        ], axis=1)
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Static clinical covariates
        if patient_profile:
            age = float(patient_profile.get("age", 35.0))
            weight = float(patient_profile.get("weight_kg", 75.0))
            height = float(patient_profile.get("height_cm", 170.0))
            bmi = weight / ((height / 100.0) ** 2) if height > 0 else 28.0
            hba1c = float(patient_profile.get("hba1c_baseline", 6.5))
            num_meds = float(patient_profile.get("num_medications", 16.0))
        else:
            age, bmi, hba1c, num_meds = 35.0, 28.0, 6.5, 16.0

        all_features["static_age"] = age
        all_features["static_bmi"] = bmi
        all_features["static_hba1c"] = hba1c
        all_features["static_num_meds"] = num_meds

        # Drop housekeeping columns
        cols_to_drop = ['time', 'trend', 'patient_id', 'trend_rate']
        all_features = all_features.drop(
            columns=[c for c in cols_to_drop if c in all_features.columns],
            errors='ignore',
        )

        for col in all_features.columns:
            all_features[col] = pd.to_numeric(all_features[col], errors='coerce')
        all_features = all_features.ffill().fillna(0)

        # --- Feature alignment ---
        # Case 1: checkpoint stored exact feature names → select them in order
        if self.feature_names is not None:
            available = set(all_features.columns)
            missing = [f for f in self.feature_names if f not in available]
            if missing:
                logger.warning(f"Missing features, filling with 0: {missing}")
                for m in missing:
                    all_features[m] = 0.0
            feature_cols = self.feature_names
        else:
            # Case 2: old checkpoint without feature_names — drop static features
            # if model expects 43, keep all if model expects 47
            feature_cols = [c for c in all_features.columns if c != 'glucose_mg_dl']
            if self.input_size == 43 and len(feature_cols) > 43:
                # Drop the 4 static features added after original training
                static_cols = ['static_age', 'static_bmi', 'static_hba1c', 'static_num_meds']
                feature_cols = [c for c in feature_cols if c not in static_cols]
                logger.info(
                    "Old checkpoint (43 features): dropping static clinical covariates "
                    "to match trained model. Retrain with train_model.py to use all 47 features."
                )

        # Pad / trim to exact input size as safety net
        if len(feature_cols) != self.input_size:
            logger.warning(
                f"Feature count mismatch after alignment: got {len(feature_cols)}, "
                f"model expects {self.input_size}. Truncating/padding."
            )
            if len(feature_cols) > self.input_size:
                feature_cols = feature_cols[:self.input_size]
            else:
                for i in range(self.input_size - len(feature_cols)):
                    pad_col = f"__pad_{i}"
                    all_features[pad_col] = 0.0
                    feature_cols.append(pad_col)

        # Extract last sequence_length rows
        seq_len = self.feature_engine.sequence_length
        X = all_features[feature_cols].values
        if len(X) < seq_len:
            padding = np.zeros((seq_len - len(X), len(feature_cols)))
            X = np.vstack([padding, X])
        X = X[-seq_len:]

        # Apply scaler
        if self.feature_engine._is_fitted:
            X = self.feature_engine.scaler.transform(X)

        return X.astype(np.float32)

    @torch.no_grad()
    def predict(
        self,
        cgm_df: pd.DataFrame,
        insulin_df: pd.DataFrame = None,
        meals_df: pd.DataFrame = None,
        return_uncertainty: bool = False,
    ) -> dict:
        """Make glucose predictions."""
        if not self.model_loaded:
            return self._fallback_prediction(cgm_df)

        try:
            X = self.prepare_features(cgm_df, insulin_df, meals_df)
            X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)

            if return_uncertainty:
                mean_pred, std_pred = self.model.predict(X_tensor)
                predictions = mean_pred.cpu().numpy()[0]
                uncertainties = std_pred.cpu().numpy()[0]
            else:
                predictions = self.model(X_tensor).cpu().numpy()[0]
                uncertainties = None

            horizons = [30, 60, 90, 120]
            result = {
                "predictions": {},
                "confidence_intervals": {},
                "model_used": True,
            }

            for i, horizon in enumerate(horizons):
                pred = float(predictions[i])
                result["predictions"][f"{horizon}min"] = round(pred, 1)

                if uncertainties is not None:
                    std = float(uncertainties[i])
                    result["confidence_intervals"][f"{horizon}min"] = (
                        round(pred - 1.96 * std, 1),
                        round(pred + 1.96 * std, 1),
                    )
                else:
                    margin = 10 + (horizon / 30) * 5
                    result["confidence_intervals"][f"{horizon}min"] = (
                        round(pred - margin, 1),
                        round(pred + margin, 1),
                    )

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(cgm_df)

    def _fallback_prediction(self, cgm_df: pd.DataFrame) -> dict:
        """Fallback to simple trend-based prediction."""
        cgm_df["glucose_mg_dl"] = pd.to_numeric(cgm_df["glucose_mg_dl"], errors="coerce")
        current = float(cgm_df["glucose_mg_dl"].iloc[-1])

        if len(cgm_df) >= 6:
            trend = float(cgm_df["glucose_mg_dl"].iloc[-1] - cgm_df["glucose_mg_dl"].iloc[-6]) / 5
        else:
            trend = 0

        result = {
            "predictions": {},
            "confidence_intervals": {},
            "model_used": False,
        }

        for horizon in [30, 60, 90, 120]:
            pred = current + (trend * horizon / 5)
            pred = pred * 0.9 + 110 * 0.1
            pred = max(40, min(400, pred))
            result["predictions"][f"{horizon}min"] = round(pred, 1)
            margin = 15 + horizon * 0.1
            result["confidence_intervals"][f"{horizon}min"] = (
                round(pred - margin, 1),
                round(pred + margin, 1),
            )

        return result

    def simulate_scenario(
        self,
        cgm_df: pd.DataFrame,
        carbs_grams: float = 0,
        insulin_units: float = 0,
        exercise_minutes: int = 0,
        exercise_intensity: str = "moderate",
    ) -> list[dict]:
        """Simulate what-if scenario. Returns trajectory over 3 hours."""
        current = float(cgm_df["glucose_mg_dl"].iloc[-1])
        trajectory = [{"time": 0, "glucose": current}]

        for t in range(15, 181, 15):
            glucose = current

            if carbs_grams > 0:
                meal_effect = carbs_grams * 3 * np.exp(-((t - 60) ** 2) / (2 * 30 ** 2))
                glucose += meal_effect

            if insulin_units > 0:
                isf = 50
                insulin_effect = insulin_units * isf * (1 - np.exp(-t / 30)) * np.exp(-(t - 90) / 120)
                glucose -= insulin_effect

            if exercise_minutes > 0 and t <= exercise_minutes + 60:
                intensity_map = {"light": 0.3, "moderate": 0.5, "vigorous": 0.8}
                factor = intensity_map.get(exercise_intensity, 0.5)
                glucose -= factor * min(t, exercise_minutes) * 0.5

            glucose = max(40, min(400, glucose))
            trajectory.append({"time": t, "glucose": round(glucose, 1)})

        return trajectory


_inference_service: Optional[GlucoseInferenceService] = None


def get_inference_service() -> GlucoseInferenceService:
    """Get or create the inference service singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = GlucoseInferenceService()
    return _inference_service
