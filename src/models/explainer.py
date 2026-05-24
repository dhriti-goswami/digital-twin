"""
Explainability module using SHAP for glucose predictions.

Provides human-readable explanations for model predictions,
translating feature importance into clinical insights.
"""

import logging
from typing import Optional

import numpy as np
import torch

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


# Feature name to clinical description mapping
FEATURE_DESCRIPTIONS = {
    # CGM features
    "glucose_mg_dl": "Current blood glucose",
    "glucose_roc_5min": "Rate of glucose change (last 5 min)",
    "glucose_roc_15min": "Rate of glucose change (last 15 min)",
    "glucose_roc_30min": "Rate of glucose change (last 30 min)",
    "glucose_mean_1h": "Average glucose (last hour)",
    "glucose_std_1h": "Glucose variability (last hour)",
    "glucose_min_1h": "Minimum glucose (last hour)",
    "glucose_max_1h": "Maximum glucose (last hour)",
    "glucose_mean_2h": "Average glucose (last 2 hours)",
    "glucose_cv_1h": "Glucose coefficient of variation",
    "glucose_range_1h": "Glucose range (last hour)",
    "trend_encoded": "Current glucose trend",
    "is_hypoglycemic": "Currently low glucose",
    "is_hyperglycemic": "Currently high glucose",
    "hypo_events_1h": "Hypoglycemic events (last hour)",
    "hyper_events_1h": "Hyperglycemic events (last hour)",

    # Insulin features
    "iob_rapid": "Insulin on board (rapid-acting)",
    "iob_total": "Total insulin on board",
    "recent_bolus_1h": "Recent bolus insulin (last hour)",
    "recent_bolus_2h": "Recent bolus insulin (last 2 hours)",

    # Meal features
    "cob": "Carbs on board",
    "recent_carbs_1h": "Recent carbs (last hour)",
    "recent_carbs_2h": "Recent carbs (last 2 hours)",
    "time_since_last_meal": "Time since last meal",

    # Activity features
    "is_exercising": "Currently exercising",
    "exercise_intensity": "Exercise intensity",
    "time_since_exercise": "Time since exercise",
    "exercise_minutes_2h": "Exercise duration (last 2 hours)",

    # Temporal features
    "hour": "Hour of day",
    "hour_sin": "Time of day (cyclical)",
    "hour_cos": "Time of day (cyclical)",
    "is_weekend": "Weekend",
    "is_breakfast_time": "Breakfast time window",
    "is_lunch_time": "Lunch time window",
    "is_dinner_time": "Dinner time window",
    "is_dawn_window": "Dawn phenomenon window",
    "is_night": "Night time",
}


class GlucoseExplainer:
    """
    SHAP-based explainability for glucose predictions.

    Provides:
    - Feature importance rankings
    - Natural language explanations
    - Visualization-ready data
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: list[str],
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100,
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability. Install with: pip install shap")

        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device

        # Create model wrapper for SHAP
        self.model_fn = self._create_model_function()

        # Initialize SHAP explainer background data on the last step (for speed & Transformer compatibility)
        if background_data is not None:
            if background_data.ndim == 3:
                # Take last step features for model-agnostic KernelExplainer
                self.background_last = background_data[:, -1, :]
            else:
                self.background_last = background_data

            if len(self.background_last) > n_background_samples:
                indices = np.random.choice(len(self.background_last), n_background_samples, replace=False)
                self.background_last = self.background_last[indices]
        else:
            self.background_last = None

        logger.info(f"GlucoseExplainer initialized with {len(feature_names)} features")

    def _create_model_function(self):
        """Create a function wrapper for SHAP."""
        def predict(x):
            self.model.eval()
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(self.device)
                return self.model(x).cpu().numpy()
        return predict

    def explain_prediction(
        self,
        input_sequence: np.ndarray,
        horizon_index: int = 0,
        top_k: int = 5,
    ) -> dict:
        """
        Generate explanation for a single prediction using robust model-agnostic KernelExplainer.

        Args:
            input_sequence: Input features (seq_len, n_features) or (1, seq_len, n_features)
            horizon_index: Which prediction horizon to explain (0-3 for 30-120 min)
            top_k: Number of top features to include

        Returns:
            Dictionary with SHAP values, feature importance, and natural language explanation
        """
        # Ensure correct shape
        if input_sequence.ndim == 2:
            input_sequence = input_sequence[np.newaxis, ...]

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(input_sequence).to(self.device)
            prediction = self.model(x).cpu().numpy()[0, horizon_index]

        # Calculate SHAP values
        if self.background_last is not None:
            # We explain the last step's features since they govern the predictions most directly
            X_last = input_sequence[:, -1, :]  # Shape (1, n_features)

            # Model prediction function wrapper targeting only this horizon and sequence length
            def predict_wrapper(X_np):
                with torch.no_grad():
                    # Replicate perturbations over sequence length to match Transformer expected inputs
                    X_seq = np.tile(X_np[:, np.newaxis, :], (1, input_sequence.shape[1], 1))
                    X_tensor = torch.FloatTensor(X_seq).to(self.device)
                    preds = self.model(X_tensor).cpu().numpy()
                    return preds[:, horizon_index]  # Shape (n_evaluations,)

            # Use KernelExplainer which is completely compatible with Transformers and LSTMs!
            explainer = shap.KernelExplainer(predict_wrapper, self.background_last[:30]) # Use subset of background for speed
            shap_values = explainer.shap_values(X_last, nsamples=50)

            # Handle multi-output / single-output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            feature_shap = np.abs(shap_values[0])
        else:
            # Fallback: use high-fidelity Integrated Gradients
            feature_shap = self._integrated_gradients(input_sequence, horizon_index)

        # Get top features
        top_indices = np.argsort(feature_shap)[-top_k:][::-1]
        top_features = [(self.feature_names[i], float(feature_shap[i])) for i in top_indices]

        # Generate natural language explanation
        explanation = self._generate_explanation(
            prediction=prediction,
            top_features=top_features,
            input_sequence=input_sequence,
            horizon_index=horizon_index,
        )

        # Prepare visualization data
        viz_data = {
            "feature_names": [f[0] for f in top_features],
            "shap_values": [f[1] for f in top_features],
            "feature_descriptions": [
                FEATURE_DESCRIPTIONS.get(f[0], f[0]) for f in top_features
            ],
        }

        return {
            "prediction": float(prediction),
            "horizon_minutes": (horizon_index + 1) * 30,
            "top_features": top_features,
            "explanation": explanation,
            "visualization_data": viz_data,
            "all_shap_values": feature_shap.tolist(),
        }

    def _integrated_gradients(self, input_sequence: np.ndarray, horizon_index: int, steps: int = 20) -> np.ndarray:
        """Calculate Integrated Gradients as a fast, high-fidelity fallback for sequence models."""
        self.model.eval()
        x = torch.FloatTensor(input_sequence).to(self.device)
        baseline = torch.zeros_like(x)  # Zero baseline

        # Accumulate gradients along the path from baseline to input
        grads_list = []
        for i in range(steps + 1):
            alpha = i / steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad = True
            
            output = self.model(interpolated)[0, horizon_index]
            self.model.zero_grad()
            output.backward()
            
            grads_list.append(interpolated.grad.cpu().numpy()[0])

        # Average gradients
        avg_grads = np.mean(grads_list, axis=0)
        # Integrated Gradients = (input - baseline) * avg_grads
        ig = (input_sequence[0] - baseline.cpu().numpy()[0]) * avg_grads
        # Average over sequence dimension
        importance = np.abs(ig).mean(axis=0)
        
        return importance

    def _gradient_importance(self, input_sequence: np.ndarray, horizon_index: int) -> np.ndarray:
        """Calculate gradient-based feature importance as fallback."""
        x = torch.FloatTensor(input_sequence).to(self.device)
        x.requires_grad = True

        self.model.eval()
        output = self.model(x)[0, horizon_index]
        output.backward()

        # Get gradients and average over sequence
        gradients = x.grad.cpu().numpy()[0]
        importance = np.abs(gradients).mean(axis=0)

        return importance

    def _generate_explanation(
        self,
        prediction: float,
        top_features: list[tuple[str, float]],
        input_sequence: np.ndarray,
        horizon_index: int,
    ) -> str:
        """Generate a natural language explanation."""
        horizon_minutes = (horizon_index + 1) * 30

        # Classify the prediction
        if prediction < 70:
            risk_level = "low (hypoglycemia risk)"
            risk_emoji = "🔴"
        elif prediction > 180:
            risk_level = "high (hyperglycemia)"
            risk_emoji = "🟡"
        elif prediction > 250:
            risk_level = "very high"
            risk_emoji = "🔴"
        else:
            risk_level = "in target range"
            risk_emoji = "🟢"

        # Build explanation
        explanation_parts = [
            f"**Predicted glucose in {horizon_minutes} minutes: {prediction:.0f} mg/dL** {risk_emoji}",
            f"This prediction is {risk_level}.",
            "",
            "**Key factors influencing this prediction:**",
        ]

        for i, (feature, importance) in enumerate(top_features[:5], 1):
            description = FEATURE_DESCRIPTIONS.get(feature, feature)

            # Get the actual value from the last timestep
            feature_idx = self.feature_names.index(feature) if feature in self.feature_names else -1
            if feature_idx >= 0:
                value = input_sequence[0, -1, feature_idx]
                value_str = f" (current value: {value:.1f})"
            else:
                value_str = ""

            explanation_parts.append(f"{i}. **{description}**{value_str}")

        # Add clinical context
        explanation_parts.extend([
            "",
            "**Clinical context:**",
        ])

        # Add specific insights based on top features
        for feature, importance in top_features[:3]:
            if "carb" in feature.lower() or "cob" in feature.lower():
                explanation_parts.append(
                    "- Recent carbohydrate intake is significantly affecting the prediction."
                )
            elif "insulin" in feature.lower() or "iob" in feature.lower():
                explanation_parts.append(
                    "- Insulin activity is a major factor in this prediction."
                )
            elif "exercise" in feature.lower():
                explanation_parts.append(
                    "- Physical activity is influencing glucose levels."
                )
            elif "dawn" in feature.lower() or "hour" in feature.lower():
                explanation_parts.append(
                    "- Time of day (circadian rhythm) is affecting the prediction."
                )
            elif "trend" in feature.lower() or "roc" in feature.lower():
                explanation_parts.append(
                    "- The recent glucose trend is a significant predictor."
                )

        return "\n".join(explanation_parts)

    def get_feature_importance_summary(
        self,
        X: np.ndarray,
        n_samples: int = 100,
    ) -> dict:
        """
        Get overall feature importance across multiple samples.

        Args:
            X: Input data (n_samples, seq_len, n_features)
            n_samples: Number of samples to analyze

        Returns:
            Dictionary with feature importance rankings
        """
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]

        all_importance = []

        for i in range(len(X)):
            result = self.explain_prediction(X[i])
            all_importance.append(result["all_shap_values"])

        # Average importance across samples
        mean_importance = np.mean(all_importance, axis=0)

        # Sort features by importance
        sorted_indices = np.argsort(mean_importance)[::-1]
        sorted_features = [
            {
                "feature": self.feature_names[i],
                "description": FEATURE_DESCRIPTIONS.get(self.feature_names[i], self.feature_names[i]),
                "importance": float(mean_importance[i]),
            }
            for i in sorted_indices
        ]

        return {
            "feature_ranking": sorted_features,
            "top_10_features": sorted_features[:10],
        }


def explain_glucose_prediction(
    model: torch.nn.Module,
    input_data: np.ndarray,
    feature_names: list[str],
    background_data: Optional[np.ndarray] = None,
    horizon_index: int = 0,
) -> dict:
    """
    Convenience function to explain a glucose prediction.

    Args:
        model: Trained PyTorch model
        input_data: Input features to explain
        feature_names: List of feature names
        background_data: Background data for SHAP
        horizon_index: Prediction horizon to explain

    Returns:
        Explanation dictionary
    """
    explainer = GlucoseExplainer(model, feature_names, background_data)
    return explainer.explain_prediction(input_data, horizon_index)
