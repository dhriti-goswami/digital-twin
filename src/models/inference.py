"""
Model inference service for glucose prediction.

Loads the trained model and provides prediction interface.
"""

import logging
from datetime import datetime, timedelta
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
        self.model_loaded = False

        # Try to load model on init
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> bool:
        """Load trained model from checkpoint."""
        from dataclasses import dataclass

        path = Path(model_path)
        if not path.exists():
            logger.warning(f"Model not found at {model_path}")
            return False

        try:
            # Define TrainingConfig for unpickling
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

            # Register for unpickling
            import __main__
            __main__.TrainingConfig = TrainingConfig

            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Get config from checkpoint
            self.config = checkpoint.get("config")
            metrics = checkpoint.get("metrics", {})

            # Infer model parameters
            state_dict = checkpoint["model_state_dict"]

            # Find input size from first layer
            for key, tensor in state_dict.items():
                if "input" in key and "weight" in key:
                    if len(tensor.shape) == 2:
                        input_size = tensor.shape[1]
                        break
            else:
                input_size = 42  # Default based on our feature engineering

            # Create model
            model_type = getattr(self.config, "model_type", "transformer")
            hidden_size = getattr(self.config, "hidden_size", 128)
            num_layers = getattr(self.config, "num_layers", 4)

            self.model = GlucosePredictor(
                input_size=input_size,
                model_type=model_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_horizons=4,
                use_pinn=True,
            ).to(self.device)

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Initialize feature engine
            self.feature_engine = GlucoseFeatureEngine(
                sequence_length=24,
                prediction_horizons=[6, 12, 18, 24],
                cgm_interval_minutes=5,
            )

            self.model_loaded = True
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model type: {model_type}, Input features: {input_size}")
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
    ) -> np.ndarray:
        """Prepare features from raw data."""
        if insulin_df is None:
            insulin_df = pd.DataFrame()
        if meals_df is None:
            meals_df = pd.DataFrame()

        # Ensure time column exists
        if "timestamp" in cgm_df.columns:
            cgm_df = cgm_df.rename(columns={"timestamp": "time"})
        cgm_df["time"] = pd.to_datetime(cgm_df["time"])

        # Add trend if missing
        if "trend" not in cgm_df.columns:
            cgm_df["trend"] = "STABLE"

        # Create features
        cgm_features = self.feature_engine.create_cgm_features(cgm_df)
        temporal_features = self.feature_engine.create_temporal_features(cgm_df["time"])
        insulin_features = self.feature_engine.create_insulin_features(cgm_df["time"], insulin_df)
        meal_features = self.feature_engine.create_meal_features(cgm_df["time"], meals_df)
        activity_features = self.feature_engine.create_activity_features(cgm_df["time"], pd.DataFrame())

        # Combine
        all_features = pd.concat([
            cgm_features.reset_index(drop=True),
            temporal_features.reset_index(drop=True),
            insulin_features.reset_index(drop=True),
            meal_features.reset_index(drop=True),
            activity_features.reset_index(drop=True),
        ], axis=1)

        # Remove duplicates and non-numeric columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        cols_to_drop = ['time', 'trend', 'patient_id']
        all_features = all_features.drop(
            columns=[c for c in cols_to_drop if c in all_features.columns],
            errors='ignore'
        )

        # Ensure numeric
        for col in all_features.columns:
            all_features[col] = pd.to_numeric(all_features[col], errors='coerce')
        all_features = all_features.ffill().fillna(0)

        # Get last sequence_length rows
        seq_len = self.feature_engine.sequence_length
        if len(all_features) < seq_len:
            # Pad with zeros if not enough data
            padding = pd.DataFrame(
                np.zeros((seq_len - len(all_features), len(all_features.columns))),
                columns=all_features.columns
            )
            all_features = pd.concat([padding, all_features], ignore_index=True)

        # Take last sequence
        feature_cols = [c for c in all_features.columns if c != 'glucose_mg_dl']
        X = all_features[feature_cols].values[-seq_len:]

        return X.astype(np.float32)

    @torch.no_grad()
    def predict(
        self,
        cgm_df: pd.DataFrame,
        insulin_df: pd.DataFrame = None,
        meals_df: pd.DataFrame = None,
        return_uncertainty: bool = False,
    ) -> dict:
        """
        Make glucose predictions.

        Args:
            cgm_df: DataFrame with 'time' and 'glucose_mg_dl' columns
            insulin_df: Optional insulin data
            meals_df: Optional meal data
            return_uncertainty: Whether to return confidence intervals

        Returns:
            Dictionary with predictions at 30, 60, 90, 120 minutes
        """
        if not self.model_loaded:
            return self._fallback_prediction(cgm_df)

        try:
            # Prepare features
            X = self.prepare_features(cgm_df, insulin_df, meals_df)

            # Convert to tensor
            X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)

            # Get prediction
            if return_uncertainty:
                mean_pred, std_pred = self.model.predict(X_tensor)
                predictions = mean_pred.cpu().numpy()[0]
                uncertainties = std_pred.cpu().numpy()[0]
            else:
                predictions = self.model(X_tensor).cpu().numpy()[0]
                uncertainties = None

            # Map to horizons
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
                    # Estimate uncertainty based on horizon
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

        # Calculate trend
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
            pred = pred * 0.9 + 110 * 0.1  # Regression to mean
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
        """
        Simulate what-if scenario.

        Returns trajectory of predicted glucose over 3 hours.
        """
        current = float(cgm_df["glucose_mg_dl"].iloc[-1])
        trajectory = [{"time": 0, "glucose": current}]

        for t in range(15, 181, 15):
            glucose = current

            # Meal effect
            if carbs_grams > 0:
                meal_peak = 60
                meal_effect = carbs_grams * 3 * np.exp(-((t - meal_peak) ** 2) / (2 * 30 ** 2))
                glucose += meal_effect

            # Insulin effect
            if insulin_units > 0:
                insulin_peak = 90
                isf = 50  # Insulin sensitivity factor
                insulin_effect = insulin_units * isf * (1 - np.exp(-t / 30)) * np.exp(-(t - insulin_peak) / 120)
                glucose -= insulin_effect

            # Exercise effect
            if exercise_minutes > 0 and t <= exercise_minutes + 60:
                intensity_map = {"light": 0.3, "moderate": 0.5, "vigorous": 0.8}
                factor = intensity_map.get(exercise_intensity, 0.5)
                exercise_effect = factor * min(t, exercise_minutes) * 0.5
                glucose -= exercise_effect

            glucose = max(40, min(400, glucose))
            trajectory.append({"time": t, "glucose": round(glucose, 1)})

        return trajectory


# Singleton instance
_inference_service: Optional[GlucoseInferenceService] = None


def get_inference_service() -> GlucoseInferenceService:
    """Get or create the inference service singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = GlucoseInferenceService()
    return _inference_service
