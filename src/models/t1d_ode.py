"""
T1D UVA/Padova ODE Patient Simulator.

Implements the FDA-accepted UVA/Padova T1D glucose dynamics model
using equations extracted from simglucose (without pkg_resources import).

State vector x (13 elements):
  x[0]  - stomach solid content (mg)
  x[1]  - stomach liquid content (mg)
  x[2]  - intestine glucose content (mg)
  x[3]  - plasma glucose (mg/kg)
  x[4]  - tissue glucose (mg/kg)
  x[5]  - plasma insulin (pmol/kg)
  x[6]  - insulin action on glucose utilization (min^-1)
  x[7]  - insulin action on production (pmol/L)
  x[8]  - delayed insulin action on production (pmol/L)
  x[9]  - liver insulin (pmol/kg)
  x[10] - subcutaneous insulin layer 1 (pmol/kg)
  x[11] - subcutaneous insulin layer 2 (pmol/kg)
  x[12] - subcutaneous glucose (mg/kg)

CGM reading: Gsub = x[12] / Vg   [mg/dL]
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import ode

logger = logging.getLogger(__name__)

PARAMS_FILE = (
    Path(__file__).parent.parent.parent
    / ".venv/lib/python3.14/site-packages/simglucose/params/vpatient_params.csv"
)


def _load_params() -> pd.DataFrame:
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Patient params not found: {PARAMS_FILE}")
    return pd.read_csv(PARAMS_FILE)


def list_patient_names() -> list[str]:
    return _load_params()["Name"].tolist()


class T1DPatient:
    """
    UVA/Padova T1D ODE simulator.
    Equations mirror simglucose's T1DPatient without the pkg_resources import.
    Uses 5-minute integration steps to match CGM sampling and run 5× faster.
    """

    SAMPLE_TIME = 5   # minutes (matches CGM interval)
    EAT_RATE = 25     # g per SAMPLE_TIME (= 5 g/min equivalent)

    def __init__(self, params: pd.Series, init_state: Optional[np.ndarray] = None, t0: float = 0):
        self._params = params
        self._init_state = init_state
        self.t0 = t0
        self.reset()

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "T1DPatient":
        df = _load_params()
        row = df.loc[df["Name"] == name].squeeze()
        if row.empty:
            raise ValueError(f"Patient '{name}' not found. Available: {df['Name'].tolist()}")
        return cls(row, **kwargs)

    @classmethod
    def from_index(cls, idx: int, **kwargs) -> "T1DPatient":
        df = _load_params()
        row = df.iloc[idx]
        return cls(row, **kwargs)

    @staticmethod
    def _ode(t, x, action_cho, action_insulin, params, last_Qsto, last_foodtaken):
        """UVA/Padova ODE — pure numpy, no simglucose import required."""
        dxdt = np.zeros(13)
        d = action_cho * 1000          # g → mg
        insulin = action_insulin * 6000 / params.BW   # U/min → pmol/kg/min
        basal = params.u2ss * params.BW / 6000         # U/min

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        qsto = x[0] + x[1]
        Dbar = last_Qsto + last_foodtaken * 1000
        if Dbar > 0:
            aa = 5 / (2 * Dbar * (1 - params.b))
            cc = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar))
                - np.tanh(cc * (qsto - params.d * Dbar))
                + 2
            )
        else:
            kgut = params.kmax

        # Stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # Intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        Rat = params.f * params.kabs * x[2] / params.BW
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        Uiit = params.Fsnc

        Et = max(0, params.ke1 * (x[3] - params.ke2)) if x[3] > params.ke2 else 0

        # Glucose kinetics
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # Insulin kinetics
        dxdt[5] = (
            -(params.m2 + params.m4) * x[5]
            + params.m1 * x[9]
            + params.ka1 * x[10]
            + params.ka2 * x[11]
        )
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)
        dxdt[7] = -params.ki * (x[7] - It)
        dxdt[8] = -params.ki * (x[8] - x[7])

        # Liver insulin
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # Subcutaneous insulin
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # Subcutaneous glucose (CGM sensor site)
        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        return dxdt

    @property
    def cgm(self) -> float:
        """Current CGM reading in mg/dL."""
        return float(self._odesolver.y[12] / self._params.Vg)

    @property
    def t(self) -> float:
        return float(self._odesolver.t)

    @property
    def state(self) -> np.ndarray:
        return self._odesolver.y.copy()

    @property
    def basal_rate(self) -> float:
        """Steady-state basal insulin rate in U/min."""
        return float(self._params.u2ss * self._params.BW / 6000)

    def step(self, cho_g: float, insulin_u_min: float):
        """Advance simulation by SAMPLE_TIME minutes."""
        to_eat = self._announce_meal(cho_g)

        if to_eat > 0 or self._last_action_cho > 0:
            self._last_Qsto = self._odesolver.y[0] + self._odesolver.y[1]
            if to_eat > 0:
                self._last_foodtaken += to_eat

        self._odesolver.set_f_params(
            to_eat, insulin_u_min, self._params, self._last_Qsto, self._last_foodtaken
        )
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.SAMPLE_TIME)
        else:
            logger.error("ODE solver failed at t=%s", self.t)
            raise RuntimeError("ODE solver failed")

        self._last_action_cho = to_eat

    def _announce_meal(self, meal: float) -> float:
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    def reset(self):
        if self._init_state is None:
            self.init_state = self._params.iloc[2:15].values.astype(float)
        else:
            self.init_state = self._init_state.copy()

        self._last_Qsto = float(self.init_state[0] + self.init_state[1])
        self._last_foodtaken = 0.0
        self._last_action_cho = 0.0
        self.planned_meal = 0.0

        solver = ode(self._ode).set_integrator("dopri5", nsteps=500)
        solver.set_initial_value(self.init_state, self.t0)
        self._odesolver = solver


def simulate_patient(
    patient: T1DPatient,
    n_days: int = 30,
    rng: Optional[np.random.RandomState] = None,
) -> pd.DataFrame:
    """
    Simulate n_days of T1D data for a single patient.

    Returns a DataFrame with columns:
      t, cgm_mg_dl, insulin_u_h, cho_g, basal_u_h
    at 5-minute intervals.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    STEP = T1DPatient.SAMPLE_TIME   # 5 minutes
    STEPS_PER_DAY = 1440 // STEP    # 288 samples per day
    total_steps = n_days * STEPS_PER_DAY

    # Meal schedule: breakfast ~7am, lunch ~12pm, dinner ~6pm
    # + possible snack at ~3pm; all with ±60-min jitter
    MEAL_WINDOWS = [
        (7 * 60, 45, 30),    # (mean_min, mean_g, std_g)
        (12 * 60, 55, 25),
        (18 * 60, 60, 30),
    ]
    SNACK_PROB = 0.3         # probability of afternoon snack

    basal = patient.basal_rate   # U/min

    # Patient-appropriate ICR via 500-rule: ICR = 500 / TDD
    # Basal typically ~50% of TDD in T1D, so TDD ≈ 2 × daily basal
    tdd_est = max(10.0, basal * 60 * 24 * 2)
    icr = max(5.0, min(50.0, 500.0 / tdd_est))

    rows = []
    patient.reset()

    for step_idx in range(total_steps):
        t_min = step_idx * STEP
        day = step_idx // STEPS_PER_DAY
        t_in_day = t_min % 1440

        # ── compute 5-min interval inputs ──────────────────────────────────
        bolus_u_min = 0.0
        cho_this_step = 0.0

        for window_center, cho_mean, cho_std in MEAL_WINDOWS:
            # Each meal window starts once per day; jitter by ±30 min
            meal_time = int(window_center + rng.uniform(-30, 30)) % 1440
            if t_in_day == meal_time:
                cho_g = max(5.0, rng.normal(cho_mean, cho_std))
                cho_this_step = cho_g
                bolus_u = cho_g / icr
                # Deliver over 5 minutes (to ODE's U/min)
                bolus_u_min = bolus_u / STEP

        # Snack at ~3pm
        if t_in_day == (15 * 60) and rng.random() < SNACK_PROB:
            snack_g = rng.uniform(10, 30)
            cho_this_step += snack_g
            bolus_u_min += snack_g / icr / STEP

        insulin_u_min = basal + bolus_u_min

        # ── advance ODE by STEP minutes ────────────────────────────────────
        # T1DPatient.SAMPLE_TIME is now 5 min, so one step = one CGM interval
        patient.step(cho_this_step, insulin_u_min)

        cgm = max(40.0, min(400.0, patient.cgm))
        insulin_u_h = insulin_u_min * 60

        rows.append({
            "t_min": t_min,
            "cgm_mg_dl": round(cgm, 2),
            "insulin_u_h": round(insulin_u_h, 4),
            "cho_g": round(cho_this_step, 2),
            "basal_u_h": round(basal * 60, 4),
        })

    return pd.DataFrame(rows)
