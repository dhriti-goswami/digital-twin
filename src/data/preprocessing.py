"""Feature engineering and preprocessing for glucose prediction models."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GlucoseFeatureEngine:
    """
    Feature engineering pipeline for glucose prediction.

    Creates multi-modal features from CGM, insulin, meal, and activity data
    suitable for LSTM/Transformer models.
    """

    def __init__(
        self,
        sequence_length: int = 24,  # 2 hours at 5-min intervals
        prediction_horizons: list[int] = None,  # Steps ahead to predict
        cgm_interval_minutes: int = 5,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or [6, 12, 18, 24]  # 30, 60, 90, 120 min
        self.cgm_interval_minutes = cgm_interval_minutes
        self.scaler = StandardScaler()
        self._is_fitted = False

    def create_cgm_features(self, cgm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from CGM time series.

        Features include:
        - Current glucose and recent history
        - Rolling statistics (mean, std, min, max)
        - Rate of change features
        - Variability metrics
        """
        df = cgm_df.copy()
        df = df.sort_values("time")

        # Ensure numeric
        df["glucose_mg_dl"] = pd.to_numeric(df["glucose_mg_dl"], errors="coerce")

        # Rate of change
        df["glucose_roc_5min"] = df["glucose_mg_dl"].diff()
        df["glucose_roc_15min"] = df["glucose_mg_dl"].diff(3)
        df["glucose_roc_30min"] = df["glucose_mg_dl"].diff(6)

        # Rolling statistics (1 hour window = 12 readings)
        df["glucose_mean_1h"] = df["glucose_mg_dl"].rolling(12, min_periods=1).mean()
        df["glucose_std_1h"] = df["glucose_mg_dl"].rolling(12, min_periods=1).std()
        df["glucose_min_1h"] = df["glucose_mg_dl"].rolling(12, min_periods=1).min()
        df["glucose_max_1h"] = df["glucose_mg_dl"].rolling(12, min_periods=1).max()

        # 2 hour window
        df["glucose_mean_2h"] = df["glucose_mg_dl"].rolling(24, min_periods=1).mean()
        df["glucose_std_2h"] = df["glucose_mg_dl"].rolling(24, min_periods=1).std()

        # Coefficient of variation
        df["glucose_cv_1h"] = df["glucose_std_1h"] / df["glucose_mean_1h"] * 100
        df["glucose_cv_2h"] = df["glucose_std_2h"] / df["glucose_mean_2h"] * 100

        # Range
        df["glucose_range_1h"] = df["glucose_max_1h"] - df["glucose_min_1h"]

        # Trend encoding
        trend_map = {
            "RISING_RAPIDLY": 2,
            "RISING": 1,
            "STABLE": 0,
            "FALLING": -1,
            "FALLING_RAPIDLY": -2,
        }
        df["trend_encoded"] = df["trend"].map(trend_map).fillna(0)

        # Time in range indicators (for recent history)
        df["is_hypoglycemic"] = (df["glucose_mg_dl"] < 70).astype(int)
        df["is_hyperglycemic"] = (df["glucose_mg_dl"] > 180).astype(int)
        df["is_in_range"] = ((df["glucose_mg_dl"] >= 70) & (df["glucose_mg_dl"] <= 180)).astype(int)

        # Recent hypo/hyper events
        df["hypo_events_1h"] = df["is_hypoglycemic"].rolling(12, min_periods=1).sum()
        df["hyper_events_1h"] = df["is_hyperglycemic"].rolling(12, min_periods=1).sum()

        return df

    def create_temporal_features(self, timestamps: pd.Series) -> pd.DataFrame:
        """
        Create time-based features.

        Features include:
        - Hour, day of week
        - Cyclical encoding (sin/cos)
        - Weekend indicator
        - Meal time indicators
        """
        df = pd.DataFrame()

        # Convert to datetime if needed
        timestamps = pd.to_datetime(timestamps)

        # Basic temporal
        df["hour"] = timestamps.dt.hour
        df["minute"] = timestamps.dt.minute
        df["day_of_week"] = timestamps.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical encoding for hour (captures daily patterns)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Meal time indicators (typical meal windows)
        df["is_breakfast_time"] = ((df["hour"] >= 6) & (df["hour"] < 10)).astype(int)
        df["is_lunch_time"] = ((df["hour"] >= 11) & (df["hour"] < 14)).astype(int)
        df["is_dinner_time"] = ((df["hour"] >= 17) & (df["hour"] < 21)).astype(int)

        # Dawn phenomenon window (5-8 AM)
        df["is_dawn_window"] = ((df["hour"] >= 5) & (df["hour"] < 8)).astype(int)

        # Night time (higher risk for undetected hypos)
        df["is_night"] = ((df["hour"] >= 23) | (df["hour"] < 6)).astype(int)

        return df

    def create_insulin_features(
        self,
        cgm_times: pd.Series,
        insulin_df: pd.DataFrame,
        lookback_hours: int = 4,
    ) -> pd.DataFrame:
        """
        Create insulin-related features aligned with CGM timestamps.

        Models insulin on board (IOB) with typical pharmacokinetic curves.
        """
        if insulin_df.empty:
            return pd.DataFrame({
                "iob_rapid": np.zeros(len(cgm_times)),
                "iob_total": np.zeros(len(cgm_times)),
                "recent_bolus_1h": np.zeros(len(cgm_times)),
                "recent_bolus_2h": np.zeros(len(cgm_times)),
            })

        df = pd.DataFrame(index=range(len(cgm_times)))
        cgm_times = pd.to_datetime(cgm_times)
        insulin_df = insulin_df.copy()
        insulin_df["time"] = pd.to_datetime(insulin_df["time"])

        iob_rapid = []
        iob_total = []
        recent_1h = []
        recent_2h = []

        for cgm_time in cgm_times:
            # Filter insulin doses in lookback window
            lookback_start = cgm_time - timedelta(hours=lookback_hours)
            recent_insulin = insulin_df[
                (insulin_df["time"] >= lookback_start) & (insulin_df["time"] <= cgm_time)
            ]

            # Calculate IOB using exponential decay model
            # Rapid insulin: ~4 hour duration, peak at ~1 hour
            iob_r = 0
            iob_t = 0
            bolus_1h = 0
            bolus_2h = 0

            for _, dose in recent_insulin.iterrows():
                minutes_ago = (cgm_time - dose["time"]).total_seconds() / 60
                units = dose["dose_units"]

                if dose.get("insulin_type") == "rapid" or dose.get("insulin_type") != "long":
                    # Rapid insulin IOB curve (simplified exponential)
                    if minutes_ago <= 240:  # 4 hour duration
                        remaining_fraction = max(0, 1 - (minutes_ago / 240) ** 1.5)
                        iob_r += units * remaining_fraction

                    if minutes_ago <= 60:
                        bolus_1h += units
                    if minutes_ago <= 120:
                        bolus_2h += units
                else:
                    # Long-acting insulin (much slower curve)
                    if minutes_ago <= 24 * 60:  # 24 hour duration
                        remaining_fraction = max(0, 1 - minutes_ago / (24 * 60))
                        iob_t += units * remaining_fraction

                iob_t += iob_r

            iob_rapid.append(round(iob_r, 2))
            iob_total.append(round(iob_t, 2))
            recent_1h.append(round(bolus_1h, 2))
            recent_2h.append(round(bolus_2h, 2))

        df["iob_rapid"] = iob_rapid
        df["iob_total"] = iob_total
        df["recent_bolus_1h"] = recent_1h
        df["recent_bolus_2h"] = recent_2h

        return df

    def create_meal_features(
        self,
        cgm_times: pd.Series,
        meals_df: pd.DataFrame,
        lookback_hours: int = 4,
    ) -> pd.DataFrame:
        """
        Create meal-related features aligned with CGM timestamps.

        Models carbs on board (COB) with glucose appearance curves.
        """
        if meals_df.empty:
            return pd.DataFrame({
                "cob": np.zeros(len(cgm_times)),
                "recent_carbs_1h": np.zeros(len(cgm_times)),
                "recent_carbs_2h": np.zeros(len(cgm_times)),
                "time_since_last_meal": np.full(len(cgm_times), lookback_hours * 60),
            })

        df = pd.DataFrame(index=range(len(cgm_times)))
        cgm_times = pd.to_datetime(cgm_times)
        meals_df = meals_df.copy()
        meals_df["time"] = pd.to_datetime(meals_df["time"])

        cob = []
        recent_1h = []
        recent_2h = []
        time_since_meal = []

        for cgm_time in cgm_times:
            lookback_start = cgm_time - timedelta(hours=lookback_hours)
            recent_meals = meals_df[
                (meals_df["time"] >= lookback_start) & (meals_df["time"] <= cgm_time)
            ]

            current_cob = 0
            carbs_1h = 0
            carbs_2h = 0
            last_meal_time = lookback_hours * 60

            for _, meal in recent_meals.iterrows():
                minutes_ago = (cgm_time - meal["time"]).total_seconds() / 60
                carbs = meal["carbs_grams"]

                # COB curve: carbs absorbed over ~2-3 hours
                if minutes_ago <= 180:  # 3 hour absorption
                    remaining_fraction = max(0, 1 - (minutes_ago / 180) ** 1.2)
                    current_cob += carbs * remaining_fraction

                if minutes_ago <= 60:
                    carbs_1h += carbs
                if minutes_ago <= 120:
                    carbs_2h += carbs

                last_meal_time = min(last_meal_time, minutes_ago)

            cob.append(round(current_cob, 1))
            recent_1h.append(round(carbs_1h, 1))
            recent_2h.append(round(carbs_2h, 1))
            time_since_meal.append(round(last_meal_time, 1))

        df["cob"] = cob
        df["recent_carbs_1h"] = recent_1h
        df["recent_carbs_2h"] = recent_2h
        df["time_since_last_meal"] = time_since_meal

        return df

    def create_activity_features(
        self,
        cgm_times: pd.Series,
        activities_df: pd.DataFrame,
        lookback_hours: int = 4,
    ) -> pd.DataFrame:
        """Create activity/exercise related features."""
        if activities_df.empty:
            return pd.DataFrame({
                "is_exercising": np.zeros(len(cgm_times)),
                "exercise_intensity": np.zeros(len(cgm_times)),
                "time_since_exercise": np.full(len(cgm_times), lookback_hours * 60),
                "exercise_minutes_2h": np.zeros(len(cgm_times)),
            })

        df = pd.DataFrame(index=range(len(cgm_times)))
        cgm_times = pd.to_datetime(cgm_times)
        activities_df = activities_df.copy()
        activities_df["start_time"] = pd.to_datetime(activities_df["start_time"])

        intensity_map = {"light": 1, "moderate": 2, "vigorous": 3}

        is_exercising = []
        exercise_intensity = []
        time_since_exercise = []
        exercise_minutes_2h = []

        for cgm_time in cgm_times:
            lookback_start = cgm_time - timedelta(hours=lookback_hours)
            recent_activities = activities_df[
                (activities_df["start_time"] >= lookback_start)
                & (activities_df["start_time"] <= cgm_time)
            ]

            current_exercise = 0
            current_intensity = 0
            last_exercise_time = lookback_hours * 60
            total_exercise_2h = 0

            for _, activity in recent_activities.iterrows():
                start_minutes_ago = (cgm_time - activity["start_time"]).total_seconds() / 60
                duration = activity.get("duration_minutes", 30)
                end_time = activity["start_time"] + timedelta(minutes=duration)
                end_minutes_ago = (cgm_time - end_time).total_seconds() / 60

                # Check if currently exercising
                if end_minutes_ago < 0:  # Exercise still ongoing
                    current_exercise = 1
                    current_intensity = intensity_map.get(activity.get("intensity", "moderate"), 2)

                last_exercise_time = min(last_exercise_time, max(0, end_minutes_ago))

                if start_minutes_ago <= 120:
                    total_exercise_2h += min(duration, 120 - start_minutes_ago)

            is_exercising.append(current_exercise)
            exercise_intensity.append(current_intensity)
            time_since_exercise.append(round(last_exercise_time, 1))
            exercise_minutes_2h.append(round(total_exercise_2h, 1))

        df["is_exercising"] = is_exercising
        df["exercise_intensity"] = exercise_intensity
        df["time_since_exercise"] = time_since_exercise
        df["exercise_minutes_2h"] = exercise_minutes_2h

        return df

    def create_sequences(
        self,
        features_df: pd.DataFrame,
        target_col: str = "glucose_mg_dl",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer training.

        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples, n_horizons)
        """
        # Select feature columns (exclude time and target)
        feature_cols = [c for c in features_df.columns if c not in ["time", target_col]]

        data = features_df[feature_cols].values
        target = features_df[target_col].values

        X, y = [], []
        max_horizon = max(self.prediction_horizons)

        for i in range(self.sequence_length, len(data) - max_horizon):
            X.append(data[i - self.sequence_length : i])
            y.append([target[i + h] for h in self.prediction_horizons])

        return np.array(X), np.array(y)

    def fit_scaler(self, features_df: pd.DataFrame) -> "GlucoseFeatureEngine":
        """Fit the scaler on training data."""
        feature_cols = [c for c in features_df.columns if c not in ["time", "glucose_mg_dl"]]
        self.scaler.fit(features_df[feature_cols].fillna(0))
        self._is_fitted = True
        return self

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to features."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")

        df = features_df.copy()
        feature_cols = [c for c in df.columns if c not in ["time", "glucose_mg_dl"]]
        df[feature_cols] = self.scaler.transform(df[feature_cols].fillna(0))
        return df

    def prepare_training_data(
        self,
        cgm_df: pd.DataFrame,
        insulin_df: pd.DataFrame,
        meals_df: pd.DataFrame,
        activities_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Complete pipeline to prepare training data.

        Returns:
            X: Feature sequences
            y: Target values
            feature_names: List of feature names
        """
        # Create all features
        cgm_features = self.create_cgm_features(cgm_df)
        temporal_features = self.create_temporal_features(cgm_df["time"])
        insulin_features = self.create_insulin_features(cgm_df["time"], insulin_df)
        meal_features = self.create_meal_features(cgm_df["time"], meals_df)
        activity_features = self.create_activity_features(cgm_df["time"], activities_df)

        # Combine all features
        all_features = pd.concat(
            [
                cgm_features.reset_index(drop=True),
                temporal_features.reset_index(drop=True),
                insulin_features.reset_index(drop=True),
                meal_features.reset_index(drop=True),
                activity_features.reset_index(drop=True),
            ],
            axis=1,
        )

        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Fill NaN values
        all_features = all_features.fillna(method="ffill").fillna(0)

        # Get feature names before scaling
        feature_cols = [c for c in all_features.columns if c not in ["time", "glucose_mg_dl"]]

        # Fit and transform
        self.fit_scaler(all_features)
        scaled_features = self.transform(all_features)

        # Create sequences
        X, y = self.create_sequences(scaled_features)

        return X, y, feature_cols
