"""
Drift Detection and Adaptive Learning Module.

Monitors model performance and triggers retraining when:
- Data distribution shifts (concept drift)
- Model accuracy degrades
- Patient physiology changes

Uses statistical tests and continuous monitoring to ensure
the digital twin remains accurate for each patient.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Container for drift detection metrics."""
    psi: float  # Population Stability Index
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float  # KS test p-value
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    drift_detected: bool
    drift_type: Optional[str]
    timestamp: datetime


class DriftDetector:
    """
    Detects data and concept drift in glucose predictions.

    Implements multiple statistical tests to identify when:
    - The distribution of input features changes (data drift)
    - The relationship between features and target changes (concept drift)
    - Model performance degrades below acceptable thresholds
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_pvalue_threshold: float = 0.05,
        mape_threshold: float = 15.0,
        rmse_threshold: float = 25.0,
        window_size: int = 288,  # 24 hours of 5-min readings
    ):
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.mape_threshold = mape_threshold
        self.rmse_threshold = rmse_threshold
        self.window_size = window_size

        # Historical data for comparison
        self.reference_distribution = None
        self.reference_predictions = None
        self.recent_predictions = []
        self.recent_actuals = []

    def set_reference(self, glucose_values: np.ndarray, predictions: Optional[np.ndarray] = None):
        """
        Set reference distribution from training data.

        Args:
            glucose_values: Historical glucose values
            predictions: Corresponding model predictions (if available)
        """
        self.reference_distribution = glucose_values
        self.reference_predictions = predictions
        logger.info(f"Set reference distribution with {len(glucose_values)} samples")

    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures how much a distribution has shifted:
        - PSI < 0.1: No significant shift
        - 0.1 <= PSI < 0.2: Moderate shift
        - PSI >= 0.2: Significant shift (drift detected)
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Calculate percentages in each bin
        ref_percentages = np.histogram(reference, bins=bin_edges)[0] / len(reference)
        cur_percentages = np.histogram(current, bins=bin_edges)[0] / len(current)

        # Avoid division by zero and log(0)
        ref_percentages = np.clip(ref_percentages, 0.0001, None)
        cur_percentages = np.clip(cur_percentages, 0.0001, None)

        # Calculate PSI
        psi = np.sum((cur_percentages - ref_percentages) * np.log(cur_percentages / ref_percentages))

        return psi

    def check_data_drift(self, current_values: np.ndarray) -> tuple[bool, dict]:
        """
        Check for data distribution drift using multiple tests.

        Args:
            current_values: Recent glucose values to compare

        Returns:
            (drift_detected, metrics_dict)
        """
        if self.reference_distribution is None:
            return False, {"error": "No reference distribution set"}

        # PSI test
        psi = self.calculate_psi(self.reference_distribution, current_values)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(self.reference_distribution, current_values)

        # Mean comparison
        ref_mean = np.mean(self.reference_distribution)
        cur_mean = np.mean(current_values)
        mean_shift = abs(cur_mean - ref_mean)

        # Variance comparison (Levene's test)
        _, levene_pvalue = stats.levene(self.reference_distribution, current_values)

        # Determine if drift is detected
        drift_detected = (
            psi >= self.psi_threshold or
            ks_pvalue < self.ks_pvalue_threshold or
            mean_shift > 20  # mg/dL shift
        )

        return drift_detected, {
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "mean_shift": mean_shift,
            "reference_mean": ref_mean,
            "current_mean": cur_mean,
            "levene_pvalue": levene_pvalue,
        }

    def check_performance_drift(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> tuple[bool, dict]:
        """
        Check for model performance degradation.

        Args:
            predictions: Model predictions
            actuals: Actual glucose values

        Returns:
            (drift_detected, metrics_dict)
        """
        # Calculate error metrics
        errors = actuals - predictions
        abs_errors = np.abs(errors)
        pct_errors = abs_errors / np.clip(actuals, 50, None) * 100

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(pct_errors)

        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean(errors ** 2))

        # MAE (Mean Absolute Error)
        mae = np.mean(abs_errors)

        # Bias (systematic over/under prediction)
        bias = np.mean(errors)

        # Proportion of predictions within clinically acceptable range
        # (within 20% or 20 mg/dL for glucose)
        within_20_pct = np.mean((abs_errors <= 20) | (pct_errors <= 20)) * 100

        # Determine if performance has degraded
        drift_detected = (
            mape >= self.mape_threshold or
            rmse >= self.rmse_threshold
        )

        return drift_detected, {
            "mape": mape,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "within_20_pct": within_20_pct,
        }

    def update_predictions(self, prediction: float, actual: float):
        """
        Update rolling window of predictions and actuals.

        Args:
            prediction: Model prediction
            actual: Actual glucose value
        """
        self.recent_predictions.append(prediction)
        self.recent_actuals.append(actual)

        # Maintain window size
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions.pop(0)
            self.recent_actuals.pop(0)

    def detect_drift(
        self,
        current_glucose: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
    ) -> DriftMetrics:
        """
        Comprehensive drift detection combining data and performance drift.

        Args:
            current_glucose: Recent glucose values for data drift detection
            predictions: Recent predictions for performance drift
            actuals: Recent actual values for performance drift

        Returns:
            DriftMetrics object with all drift information
        """
        drift_detected = False
        drift_type = None
        data_metrics = {}
        perf_metrics = {}

        # Check data drift
        if current_glucose is not None and self.reference_distribution is not None:
            data_drift, data_metrics = self.check_data_drift(current_glucose)
            if data_drift:
                drift_detected = True
                drift_type = "data"

        # Check performance drift
        if predictions is not None and actuals is not None:
            perf_drift, perf_metrics = self.check_performance_drift(predictions, actuals)
            if perf_drift:
                drift_detected = True
                drift_type = "performance" if drift_type is None else "both"

        # Use rolling window if individual arrays not provided
        elif len(self.recent_predictions) >= 50:
            preds = np.array(self.recent_predictions)
            acts = np.array(self.recent_actuals)
            perf_drift, perf_metrics = self.check_performance_drift(preds, acts)
            if perf_drift:
                drift_detected = True
                drift_type = "performance" if drift_type is None else "both"

        return DriftMetrics(
            psi=data_metrics.get("psi", 0.0),
            ks_statistic=data_metrics.get("ks_statistic", 0.0),
            ks_pvalue=data_metrics.get("ks_pvalue", 1.0),
            mape=perf_metrics.get("mape", 0.0),
            rmse=perf_metrics.get("rmse", 0.0),
            drift_detected=drift_detected,
            drift_type=drift_type,
            timestamp=datetime.now(),
        )


class AdaptiveLearner:
    """
    Manages adaptive learning and model retraining.

    Responsibilities:
    - Monitor drift metrics continuously
    - Decide when to trigger retraining
    - Orchestrate the retraining process
    - Handle model versioning and rollback
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        min_retrain_interval_hours: int = 24,
        max_drift_tolerance_count: int = 3,
    ):
        self.drift_detector = drift_detector
        self.min_retrain_interval = timedelta(hours=min_retrain_interval_hours)
        self.max_drift_tolerance = max_drift_tolerance_count

        self.last_retrain_time = None
        self.consecutive_drift_count = 0
        self.drift_history = []
        self.model_versions = []

    def should_retrain(self, drift_metrics: DriftMetrics) -> tuple[bool, str]:
        """
        Decide if model should be retrained.

        Args:
            drift_metrics: Latest drift detection results

        Returns:
            (should_retrain, reason)
        """
        # Store drift history
        self.drift_history.append(drift_metrics)
        if len(self.drift_history) > 100:
            self.drift_history.pop(0)

        # Check if enough time has passed since last retrain
        can_retrain = True
        if self.last_retrain_time is not None:
            time_since_retrain = datetime.now() - self.last_retrain_time
            can_retrain = time_since_retrain >= self.min_retrain_interval

        if not can_retrain:
            return False, "Minimum retrain interval not reached"

        # Count consecutive drift detections
        if drift_metrics.drift_detected:
            self.consecutive_drift_count += 1
        else:
            self.consecutive_drift_count = 0

        # Decision logic
        if drift_metrics.drift_detected and drift_metrics.drift_type == "both":
            return True, "Both data and performance drift detected"

        if self.consecutive_drift_count >= self.max_drift_tolerance:
            return True, f"Consecutive drift detected {self.consecutive_drift_count} times"

        if drift_metrics.mape > 20:  # Severe performance degradation
            return True, f"Severe performance degradation (MAPE: {drift_metrics.mape:.1f}%)"

        return False, "No retraining needed"

    def trigger_retrain(self, patient_id: Optional[int] = None) -> dict:
        """
        Trigger model retraining.

        Args:
            patient_id: Optional patient-specific retraining

        Returns:
            Status dictionary
        """
        logger.info(f"Triggering retrain for patient_id={patient_id}")

        # In production, this would:
        # 1. Collect recent training data
        # 2. Run incremental training or full retraining
        # 3. Validate new model on holdout set
        # 4. Replace old model if new one is better
        # 5. Update version tracking

        self.last_retrain_time = datetime.now()
        self.consecutive_drift_count = 0

        return {
            "status": "triggered",
            "timestamp": self.last_retrain_time.isoformat(),
            "patient_id": patient_id,
        }

    def get_status(self) -> dict:
        """Get current adaptive learning status."""
        recent_drift_rate = 0
        if len(self.drift_history) >= 10:
            recent = self.drift_history[-10:]
            recent_drift_rate = sum(1 for d in recent if d.drift_detected) / 10 * 100

        return {
            "last_retrain": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "consecutive_drift_count": self.consecutive_drift_count,
            "recent_drift_rate_pct": recent_drift_rate,
            "total_drift_events": sum(1 for d in self.drift_history if d.drift_detected),
            "model_versions": len(self.model_versions),
        }


class PatientPhysiologyTracker:
    """
    Tracks patient-specific physiological changes over time.

    Monitors for:
    - Changes in insulin sensitivity
    - Changes in carb ratios
    - Circadian pattern shifts
    - Seasonal variations
    """

    def __init__(self, patient_id: int):
        self.patient_id = patient_id
        self.insulin_sensitivity_history = []
        self.carb_ratio_history = []
        self.avg_glucose_history = []

    def update_metrics(
        self,
        avg_glucose: float,
        estimated_isf: Optional[float] = None,
        estimated_cr: Optional[float] = None,
    ):
        """Update tracked metrics."""
        self.avg_glucose_history.append({
            "value": avg_glucose,
            "timestamp": datetime.now(),
        })

        if estimated_isf:
            self.insulin_sensitivity_history.append({
                "value": estimated_isf,
                "timestamp": datetime.now(),
            })

        if estimated_cr:
            self.carb_ratio_history.append({
                "value": estimated_cr,
                "timestamp": datetime.now(),
            })

        # Keep last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        self.avg_glucose_history = [
            x for x in self.avg_glucose_history if x["timestamp"] > cutoff
        ]

    def detect_physiology_change(self) -> tuple[bool, Optional[str]]:
        """
        Detect significant changes in patient physiology.

        Returns:
            (change_detected, description)
        """
        if len(self.avg_glucose_history) < 14:  # Need at least 2 weeks
            return False, None

        # Compare first week to last week
        first_week = [x["value"] for x in self.avg_glucose_history[:7]]
        last_week = [x["value"] for x in self.avg_glucose_history[-7:]]

        first_mean = np.mean(first_week)
        last_mean = np.mean(last_week)

        # Check for significant change (>15 mg/dL shift)
        if abs(last_mean - first_mean) > 15:
            direction = "increased" if last_mean > first_mean else "decreased"
            return True, f"Average glucose has {direction} by {abs(last_mean - first_mean):.0f} mg/dL over the past 2 weeks"

        return False, None

    def get_trend_report(self) -> dict:
        """Generate a report on physiological trends."""
        if len(self.avg_glucose_history) < 7:
            return {"status": "insufficient_data"}

        values = [x["value"] for x in self.avg_glucose_history[-7:]]

        return {
            "patient_id": self.patient_id,
            "period_days": 7,
            "avg_glucose": np.mean(values),
            "glucose_std": np.std(values),
            "trend": "stable" if np.std(values) < 20 else "variable",
            "physiology_change": self.detect_physiology_change()[0],
        }
