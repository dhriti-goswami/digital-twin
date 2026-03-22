# CGM Trace Data

These CGM (Continuous Glucose Monitor) traces were generated from real PIMA glucose values.

## Data Format

- **Frequency**: 5-minute intervals (288 readings per day)
- **Duration**: 7 days per patient
- **Columns**:
  - timestamp: ISO8601 datetime
  - glucose_mg_dl: Blood glucose in mg/dL

## Generation Method

Each trace is centered around real glucose values from the PIMA dataset and includes:
1. Circadian rhythm effects (dawn phenomenon)
2. Meal response patterns (breakfast, lunch, dinner)
3. Physiological noise (sensor + biological variation)

## Physiological Realism

- Dawn phenomenon: ~10 mg/dL peak around 6 AM
- Post-meal spikes: 25-35 mg/dL above baseline
- Sensor noise: σ = 5 mg/dL (typical CGM accuracy)
- Bounds: 40-400 mg/dL (physiological limits)

## Files

- `patient_001_cgm_trace.csv` through `patient_010_cgm_trace.csv`

Total: ~20,160 glucose readings per patient (10 patients = 201,600 total readings)
