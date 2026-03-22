
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import text
from src.data.database import sync_engine

def inject_direct():
    patient_id = 1
    now = datetime.now()
    times = [now - timedelta(minutes=5 * i) for i in range(576, -1, -1)]
    np.random.seed(42)
    t_hours = np.linspace(0, 48, len(times))
    glucose = 120 + 40 * np.sin(2 * np.pi * t_hours / 6) + 15 * np.cos(2 * np.pi * t_hours / 24) + np.random.normal(0, 5, len(times))
    glucose = np.clip(glucose, 60, 300)
    
    with sync_engine.begin() as conn:
        conn.execute(text("INSERT INTO patients (id, external_id, age, weight_kg, height_cm, diabetes_type) VALUES (1, 'DEMO001', 35, 75.0, 175.0, 'Type 1') ON CONFLICT DO NOTHING"))
        conn.execute(text("DELETE FROM cgm_readings WHERE patient_id = 1"))
        values = [{"time": t, "patient_id": patient_id, "glucose": float(g), "trend": "FLAT", "trend_rate": 0.0, "device": "DEMO"} for t, g in zip(times, glucose)]
        conn.execute(text("INSERT INTO cgm_readings (time, patient_id, glucose_mg_dl, trend, trend_rate, device_id) VALUES (:time, :patient_id, :glucose, :trend, :trend_rate, :device) ON CONFLICT DO NOTHING"), values)
    print(f"Force injected {len(values)} records successfully!")

if __name__ == "__main__":
    inject_direct()
