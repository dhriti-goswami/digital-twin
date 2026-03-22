"""
Physiologically Realistic Diabetes Data Generator

Uses the Bergman Minimal Model and extensions to generate realistic CGM data
that reflects actual glucose-insulin dynamics in Type 1 Diabetes.

References:
- Bergman RN, et al. "Physiologic evaluation of factors controlling glucose tolerance in man"
- Hovorka R, et al. "Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes"
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


@dataclass
class PatientParameters:
    """Patient-specific physiological parameters."""

    # Bergman Minimal Model parameters
    p1: float = 0.028  # Glucose effectiveness (min^-1)
    p2: float = 0.025  # Rate of insulin action decay (min^-1)
    p3: float = 5.0e-5  # Insulin sensitivity (min^-2 per µU/ml)

    # Steady state values
    gb: float = 120.0  # Basal glucose (mg/dL)
    ib: float = 10.0  # Basal insulin (µU/ml)

    # Meal absorption (Dalla Man model)
    kabs: float = 0.05  # Rate of glucose absorption (min^-1)
    kmax: float = 0.02  # Maximum absorption rate (min^-1)
    kmin: float = 0.008  # Minimum absorption rate (min^-1)
    b: float = 0.82  # Bioavailability fraction
    d: float = 0.01  # Rate of gastric emptying (min^-1)

    # Insulin kinetics
    ka: float = 0.02  # Subcutaneous insulin absorption rate (min^-1)
    ke: float = 0.138  # Insulin elimination rate (min^-1)
    vi: float = 0.12  # Insulin distribution volume (L/kg)

    # Patient characteristics
    weight_kg: float = 70.0
    insulin_sensitivity_factor: float = 50.0  # mg/dL drop per unit insulin
    carb_ratio: float = 10.0  # grams of carbs per unit insulin

    # Variability parameters
    cgm_noise_std: float = 5.0  # CGM measurement noise (mg/dL)
    dawn_phenomenon_peak: float = 1.3  # Peak effect at 5-7 AM
    exercise_sensitivity_multiplier: float = 1.5  # Increased insulin sensitivity during exercise


class DiabetesSimulator:
    """
    Simulates realistic glucose dynamics using physiological models.

    This generates medically realistic CGM data that exhibits:
    - Proper meal responses with realistic peak timing
    - Insulin action with appropriate time delays
    - Dawn phenomenon (early morning glucose rise)
    - Exercise effects (increased insulin sensitivity)
    - Realistic CGM noise and sensor artifacts
    """

    def __init__(self, params: Optional[PatientParameters] = None, seed: Optional[int] = None):
        self.params = params or PatientParameters()
        self.rng = np.random.default_rng(seed)

    def _bergman_ode(self, t: float, y: np.ndarray, meal_signal: callable, insulin_signal: callable) -> np.ndarray:
        """
        Bergman Minimal Model ODEs with meal and insulin inputs.

        State variables:
        y[0] = G: Plasma glucose (mg/dL)
        y[1] = X: Remote insulin effect (min^-1)
        y[2] = I: Plasma insulin (µU/ml)
        y[3] = Q1: Gut glucose compartment 1
        y[4] = Q2: Gut glucose compartment 2
        y[5] = S1: Subcutaneous insulin compartment 1
        y[6] = S2: Subcutaneous insulin compartment 2
        """
        G, X, I, Q1, Q2, S1, S2 = y
        p = self.params

        # Meal appearance rate (mg/min/kg)
        Ra = p.kabs * Q2 / p.weight_kg

        # Insulin appearance rate (µU/min)
        Ra_insulin = p.ka * S2

        # Get current inputs
        D = meal_signal(t)  # Meal input (mg CHO)
        U = insulin_signal(t)  # Insulin input (Units)

        # ODEs
        dG = -p.p1 * (G - p.gb) - X * G + Ra
        dX = -p.p2 * X + p.p3 * (I - p.ib)
        dI = -p.ke * I + Ra_insulin / (p.vi * p.weight_kg)
        dQ1 = -p.kmax * Q1 + D * p.b
        dQ2 = p.kmax * Q1 - p.kabs * Q2
        dS1 = -p.ka * S1 + U * 1000 / p.weight_kg  # Convert units to µU/kg
        dS2 = p.ka * S1 - p.ka * S2

        return np.array([dG, dX, dI, dQ1, dQ2, dS1, dS2])

    def _circadian_effect(self, hour: float) -> float:
        """Model dawn phenomenon and circadian glucose variation."""
        # Dawn phenomenon peaks around 4-7 AM
        dawn_peak_hour = 5.5
        dawn_width = 2.0

        dawn_effect = self.params.dawn_phenomenon_peak * np.exp(
            -0.5 * ((hour - dawn_peak_hour) / dawn_width) ** 2
        )

        # Slight increase in evening as well
        evening_effect = 1.05 * np.exp(-0.5 * ((hour - 20) / 3) ** 2)

        return 1.0 + (dawn_effect - 1.0) + (evening_effect - 1.0)

    def _add_cgm_noise(self, glucose_values: np.ndarray) -> np.ndarray:
        """Add realistic CGM noise and artifacts."""
        noise = self.rng.normal(0, self.params.cgm_noise_std, len(glucose_values))

        # Occasional larger artifacts (sensor compression, etc.)
        artifact_mask = self.rng.random(len(glucose_values)) < 0.01
        artifacts = self.rng.normal(0, 20, len(glucose_values)) * artifact_mask

        result = glucose_values + noise + artifacts

        # Ensure physiological bounds
        return np.clip(result, 40, 400)

    def simulate_day(
        self,
        start_time: datetime,
        meals: list[tuple[datetime, float]],  # (time, carbs_grams)
        insulin_doses: list[tuple[datetime, float]],  # (time, units)
        activities: Optional[list[tuple[datetime, int, str]]] = None,  # (time, duration_min, intensity)
        initial_glucose: float = 120.0,
        sampling_interval_minutes: int = 5,
    ) -> pd.DataFrame:
        """
        Simulate a full day of CGM readings.

        Args:
            start_time: Start datetime for simulation
            meals: List of (datetime, carbs_grams) tuples
            insulin_doses: List of (datetime, units) tuples
            activities: Optional list of (datetime, duration_minutes, intensity) tuples
            initial_glucose: Starting glucose level (mg/dL)
            sampling_interval_minutes: CGM sampling frequency

        Returns:
            DataFrame with columns: time, glucose_mg_dl, trend, trend_rate
        """
        duration_minutes = 24 * 60
        t_span = (0, duration_minutes)
        t_eval = np.arange(0, duration_minutes, sampling_interval_minutes)

        # Create meal signal function
        def meal_signal(t):
            total = 0
            for meal_time, carbs in meals:
                meal_t = (meal_time - start_time).total_seconds() / 60
                if 0 <= t - meal_t < 5:  # Meal consumed over 5 minutes
                    total += carbs * 1000 / 5  # mg CHO / min
            return total

        # Create insulin signal function
        def insulin_signal(t):
            total = 0
            for dose_time, units in insulin_doses:
                dose_t = (dose_time - start_time).total_seconds() / 60
                if 0 <= t - dose_t < 1:  # Injection takes 1 minute
                    total += units
            return total

        # Initial conditions
        y0 = np.array([
            initial_glucose,  # G
            0.0,  # X
            self.params.ib,  # I
            0.0,  # Q1
            0.0,  # Q2
            0.0,  # S1
            0.0,  # S2
        ])

        # Solve ODEs
        solution = solve_ivp(
            lambda t, y: self._bergman_ode(t, y, meal_signal, insulin_signal),
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            max_step=1.0,
        )

        # Extract glucose values
        glucose = solution.y[0]

        # Apply circadian effects
        for i, t in enumerate(t_eval):
            current_time = start_time + timedelta(minutes=float(t))
            hour = current_time.hour + current_time.minute / 60
            glucose[i] *= self._circadian_effect(hour)

        # Apply activity effects (reduced glucose during/after exercise)
        if activities:
            for activity_time, duration, intensity in activities:
                activity_start = (activity_time - start_time).total_seconds() / 60
                activity_end = activity_start + duration

                # Effect lasts 2x the activity duration
                effect_end = activity_end + duration

                intensity_factor = {"light": 0.02, "moderate": 0.04, "vigorous": 0.06}.get(intensity, 0.03)

                for i, t in enumerate(t_eval):
                    if activity_start <= t <= effect_end:
                        # Gradual glucose reduction during and after exercise
                        factor = 1.0 - intensity_factor * min(1.0, (t - activity_start) / 30)
                        glucose[i] *= max(0.7, factor)

        # Add sensor noise
        glucose = self._add_cgm_noise(glucose)

        # Calculate trends
        trends = []
        trend_rates = []
        for i in range(len(glucose)):
            if i < 3:
                rate = 0.0
            else:
                # Rate over last 15 minutes (3 readings)
                rate = (glucose[i] - glucose[i - 3]) / 15.0

            trend_rates.append(rate)

            if rate > 2:
                trends.append("RISING_RAPIDLY")
            elif rate > 1:
                trends.append("RISING")
            elif rate < -2:
                trends.append("FALLING_RAPIDLY")
            elif rate < -1:
                trends.append("FALLING")
            else:
                trends.append("STABLE")

        # Build DataFrame
        times = [start_time + timedelta(minutes=float(t)) for t in t_eval]

        return pd.DataFrame({
            "time": times,
            "glucose_mg_dl": glucose.round(1),
            "trend": trends,
            "trend_rate": np.round(trend_rates, 2),
        })

    def generate_realistic_day_events(
        self,
        date: datetime,
    ) -> tuple[list, list, list]:
        """
        Generate realistic meal, insulin, and activity patterns for a day.

        Returns:
            Tuple of (meals, insulin_doses, activities)
        """
        meals = []
        insulin_doses = []
        activities = []

        # Breakfast (7-9 AM)
        breakfast_time = date.replace(hour=7) + timedelta(minutes=self.rng.integers(0, 120))
        breakfast_carbs = self.rng.integers(30, 70)
        meals.append((breakfast_time, float(breakfast_carbs)))

        # Breakfast bolus (a few minutes before)
        bolus_units = breakfast_carbs / self.params.carb_ratio
        bolus_units += self.rng.normal(0, 0.5)  # Some variability
        insulin_doses.append((breakfast_time - timedelta(minutes=self.rng.integers(0, 15)), max(0, bolus_units)))

        # Morning snack (10-11 AM, 50% chance)
        if self.rng.random() < 0.5:
            snack_time = date.replace(hour=10) + timedelta(minutes=self.rng.integers(0, 60))
            snack_carbs = self.rng.integers(10, 25)
            meals.append((snack_time, float(snack_carbs)))
            if snack_carbs > 15:
                insulin_doses.append((snack_time, snack_carbs / self.params.carb_ratio))

        # Lunch (12-2 PM)
        lunch_time = date.replace(hour=12) + timedelta(minutes=self.rng.integers(0, 120))
        lunch_carbs = self.rng.integers(40, 80)
        meals.append((lunch_time, float(lunch_carbs)))
        lunch_bolus = lunch_carbs / self.params.carb_ratio + self.rng.normal(0, 0.5)
        insulin_doses.append((lunch_time - timedelta(minutes=self.rng.integers(0, 10)), max(0, lunch_bolus)))

        # Afternoon snack (3-4 PM, 40% chance)
        if self.rng.random() < 0.4:
            snack_time = date.replace(hour=15) + timedelta(minutes=self.rng.integers(0, 60))
            snack_carbs = self.rng.integers(10, 30)
            meals.append((snack_time, float(snack_carbs)))

        # Exercise (30% chance, afternoon/evening)
        if self.rng.random() < 0.3:
            exercise_time = date.replace(hour=self.rng.integers(16, 19))
            duration = self.rng.integers(20, 60)
            intensity = self.rng.choice(["light", "moderate", "vigorous"])
            activities.append((exercise_time, duration, intensity))

        # Dinner (6-8 PM)
        dinner_time = date.replace(hour=18) + timedelta(minutes=self.rng.integers(0, 120))
        dinner_carbs = self.rng.integers(50, 100)
        meals.append((dinner_time, float(dinner_carbs)))
        dinner_bolus = dinner_carbs / self.params.carb_ratio + self.rng.normal(0, 0.5)
        insulin_doses.append((dinner_time - timedelta(minutes=self.rng.integers(0, 15)), max(0, dinner_bolus)))

        # Evening snack (9-10 PM, 60% chance)
        if self.rng.random() < 0.6:
            snack_time = date.replace(hour=21) + timedelta(minutes=self.rng.integers(0, 60))
            snack_carbs = self.rng.integers(10, 30)
            meals.append((snack_time, float(snack_carbs)))

        # Basal insulin (long-acting, once daily in evening)
        basal_time = date.replace(hour=22)
        basal_units = self.params.weight_kg * 0.4 / 2  # Roughly half TDD as basal
        insulin_doses.append((basal_time, basal_units))

        # Correction doses (if glucose would be high, add corrections)
        # This is simplified - in real simulation we'd check predicted glucose
        if self.rng.random() < 0.3:
            correction_time = date.replace(hour=self.rng.integers(10, 20))
            correction_units = self.rng.uniform(1, 3)
            insulin_doses.append((correction_time, correction_units))

        return meals, insulin_doses, activities

    def generate_patient_data(
        self,
        patient_id: str,
        start_date: datetime,
        num_days: int = 30,
        seed: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate complete patient data for multiple days.

        Returns:
            Dictionary with 'cgm', 'meals', 'insulin', 'activities' DataFrames
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        all_cgm = []
        all_meals = []
        all_insulin = []
        all_activities = []

        current_glucose = self.rng.uniform(100, 140)  # Starting glucose

        for day in range(num_days):
            day_start = start_date + timedelta(days=day)
            day_start = day_start.replace(hour=0, minute=0, second=0, microsecond=0)

            # Generate daily events
            meals, insulin_doses, activities = self.generate_realistic_day_events(day_start)

            # Simulate the day
            cgm_df = self.simulate_day(
                start_time=day_start,
                meals=meals,
                insulin_doses=insulin_doses,
                activities=activities,
                initial_glucose=current_glucose,
            )

            # Use end glucose as start for next day
            current_glucose = cgm_df["glucose_mg_dl"].iloc[-1]

            all_cgm.append(cgm_df)

            # Store meals
            for meal_time, carbs in meals:
                all_meals.append({
                    "time": meal_time,
                    "carbs_grams": carbs,
                    "meal_type": self._get_meal_type(meal_time.hour),
                })

            # Store insulin
            for dose_time, units in insulin_doses:
                all_insulin.append({
                    "time": dose_time,
                    "dose_units": round(units, 1),
                    "insulin_type": "long" if dose_time.hour == 22 else "rapid",
                })

            # Store activities
            for act_time, duration, intensity in activities:
                all_activities.append({
                    "start_time": act_time,
                    "duration_minutes": duration,
                    "intensity": intensity,
                    "activity_type": "exercise",
                })

        return {
            "cgm": pd.concat(all_cgm, ignore_index=True),
            "meals": pd.DataFrame(all_meals),
            "insulin": pd.DataFrame(all_insulin),
            "activities": pd.DataFrame(all_activities),
        }

    def _get_meal_type(self, hour: int) -> str:
        """Determine meal type based on hour."""
        if 5 <= hour < 10:
            return "breakfast"
        elif 10 <= hour < 12:
            return "snack"
        elif 12 <= hour < 15:
            return "lunch"
        elif 15 <= hour < 18:
            return "snack"
        elif 18 <= hour < 21:
            return "dinner"
        else:
            return "snack"
