-- Digital Twin Database Initialization Script
-- Creates all necessary tables with TimescaleDB hypertables for time-series data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Patients table
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(50) UNIQUE NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    weight_kg DECIMAL(5,2),
    height_cm DECIMAL(5,2),
    diabetes_type VARCHAR(20) NOT NULL DEFAULT 'type1',
    diagnosis_date DATE,
    hba1c_baseline DECIMAL(4,2),
    total_daily_insulin DECIMAL(5,2),
    carb_ratio DECIMAL(4,2),          -- insulin units per gram of carbs
    correction_factor DECIMAL(5,2),    -- mg/dL drop per unit of insulin
    target_glucose_low INTEGER DEFAULT 70,
    target_glucose_high INTEGER DEFAULT 180,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- CGM Readings - TimescaleDB hypertable for efficient time-series queries
CREATE TABLE IF NOT EXISTS cgm_readings (
    time TIMESTAMPTZ NOT NULL,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    glucose_mg_dl DECIMAL(6,2) NOT NULL,
    trend VARCHAR(20),  -- RISING, FALLING, STABLE, RISING_RAPIDLY, FALLING_RAPIDLY
    trend_rate DECIMAL(5,2),  -- mg/dL per minute
    device_id VARCHAR(50),
    is_calibration BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (time, patient_id)
);

-- Convert to hypertable for efficient time-series operations
SELECT create_hypertable('cgm_readings', 'time', if_not_exists => TRUE);

-- Insulin Doses
CREATE TABLE IF NOT EXISTS insulin_doses (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    dose_units DECIMAL(5,2) NOT NULL,
    insulin_type VARCHAR(50) NOT NULL,  -- rapid, short, intermediate, long, mixed
    brand_name VARCHAR(100),
    injection_site VARCHAR(50),
    is_correction BOOLEAN DEFAULT FALSE,
    is_meal_bolus BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (time, id)
);

SELECT create_hypertable('insulin_doses', 'time', if_not_exists => TRUE);

-- Meals and Carbohydrate Intake
CREATE TABLE IF NOT EXISTS meals (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    carbs_grams DECIMAL(6,2) NOT NULL,
    protein_grams DECIMAL(6,2),
    fat_grams DECIMAL(6,2),
    fiber_grams DECIMAL(6,2),
    glycemic_index INTEGER,  -- 0-100
    meal_type VARCHAR(50),   -- breakfast, lunch, dinner, snack
    description TEXT,
    PRIMARY KEY (time, id)
);

SELECT create_hypertable('meals', 'time', if_not_exists => TRUE);

-- Physical Activity
CREATE TABLE IF NOT EXISTS activities (
    id SERIAL,
    start_time TIMESTAMPTZ NOT NULL,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    end_time TIMESTAMPTZ,
    duration_minutes INTEGER,
    activity_type VARCHAR(100),
    intensity VARCHAR(20),  -- light, moderate, vigorous
    heart_rate_avg INTEGER,
    heart_rate_max INTEGER,
    calories_burned INTEGER,
    steps INTEGER,
    PRIMARY KEY (start_time, id)
);

SELECT create_hypertable('activities', 'start_time', if_not_exists => TRUE);

-- Sleep Records
CREATE TABLE IF NOT EXISTS sleep_records (
    id SERIAL,
    start_time TIMESTAMPTZ NOT NULL,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    end_time TIMESTAMPTZ,
    duration_minutes INTEGER,
    quality_score INTEGER,  -- 0-100
    deep_sleep_minutes INTEGER,
    rem_sleep_minutes INTEGER,
    awakenings INTEGER,
    PRIMARY KEY (start_time, id)
);

SELECT create_hypertable('sleep_records', 'start_time', if_not_exists => TRUE);

-- Clinical Lab Results
CREATE TABLE IF NOT EXISTS lab_results (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    test_date DATE NOT NULL,
    test_type VARCHAR(100) NOT NULL,
    value DECIMAL(10,4),
    unit VARCHAR(50),
    reference_low DECIMAL(10,4),
    reference_high DECIMAL(10,4),
    is_abnormal BOOLEAN DEFAULT FALSE,
    notes TEXT
);

-- Model Predictions (for tracking model performance)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL,
    prediction_time TIMESTAMPTZ NOT NULL,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    target_time TIMESTAMPTZ NOT NULL,
    predicted_glucose DECIMAL(6,2) NOT NULL,
    actual_glucose DECIMAL(6,2),
    model_version VARCHAR(100),
    confidence_low DECIMAL(6,2),
    confidence_high DECIMAL(6,2),
    PRIMARY KEY (prediction_time, id)
);

SELECT create_hypertable('predictions', 'prediction_time', if_not_exists => TRUE);

-- Drift Detection Events
CREATE TABLE IF NOT EXISTS drift_events (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    drift_type VARCHAR(50),  -- concept, data, performance
    metric_name VARCHAR(100),
    metric_value DECIMAL(10,6),
    threshold DECIMAL(10,6),
    action_taken VARCHAR(100),
    retrain_triggered BOOLEAN DEFAULT FALSE
);

-- Chat History
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    role VARCHAR(20) NOT NULL,  -- user, assistant, system
    content TEXT NOT NULL,
    metadata JSONB
);

-- Alerts and Notifications
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    alert_type VARCHAR(50) NOT NULL,  -- hypoglycemia, hyperglycemia, trend, prediction
    severity VARCHAR(20) NOT NULL,     -- info, warning, critical
    message TEXT NOT NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_cgm_patient_time ON cgm_readings (patient_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_insulin_patient_time ON insulin_doses (patient_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_meals_patient_time ON meals (patient_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_activities_patient_time ON activities (patient_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions (patient_id, prediction_time DESC);

-- Create continuous aggregates for efficient dashboard queries
CREATE MATERIALIZED VIEW IF NOT EXISTS cgm_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    patient_id,
    AVG(glucose_mg_dl) AS avg_glucose,
    MIN(glucose_mg_dl) AS min_glucose,
    MAX(glucose_mg_dl) AS max_glucose,
    STDDEV(glucose_mg_dl) AS stddev_glucose,
    COUNT(*) AS reading_count
FROM cgm_readings
GROUP BY time_bucket('1 hour', time), patient_id
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('cgm_hourly',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Daily glucose statistics view
CREATE MATERIALIZED VIEW IF NOT EXISTS cgm_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    patient_id,
    AVG(glucose_mg_dl) AS avg_glucose,
    MIN(glucose_mg_dl) AS min_glucose,
    MAX(glucose_mg_dl) AS max_glucose,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY glucose_mg_dl) AS median_glucose,
    COUNT(*) AS reading_count,
    SUM(CASE WHEN glucose_mg_dl < 70 THEN 1 ELSE 0 END) AS hypo_count,
    SUM(CASE WHEN glucose_mg_dl > 180 THEN 1 ELSE 0 END) AS hyper_count,
    SUM(CASE WHEN glucose_mg_dl BETWEEN 70 AND 180 THEN 1 ELSE 0 END) AS in_range_count
FROM cgm_readings
GROUP BY time_bucket('1 day', time), patient_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('cgm_daily',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Function to calculate Time in Range (TIR)
CREATE OR REPLACE FUNCTION calculate_tir(
    p_patient_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ,
    p_low_threshold DECIMAL DEFAULT 70,
    p_high_threshold DECIMAL DEFAULT 180
)
RETURNS TABLE(
    total_readings BIGINT,
    in_range_readings BIGINT,
    below_range_readings BIGINT,
    above_range_readings BIGINT,
    tir_percentage DECIMAL,
    tbr_percentage DECIMAL,
    tar_percentage DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT AS total_readings,
        SUM(CASE WHEN glucose_mg_dl BETWEEN p_low_threshold AND p_high_threshold THEN 1 ELSE 0 END)::BIGINT AS in_range_readings,
        SUM(CASE WHEN glucose_mg_dl < p_low_threshold THEN 1 ELSE 0 END)::BIGINT AS below_range_readings,
        SUM(CASE WHEN glucose_mg_dl > p_high_threshold THEN 1 ELSE 0 END)::BIGINT AS above_range_readings,
        ROUND(100.0 * SUM(CASE WHEN glucose_mg_dl BETWEEN p_low_threshold AND p_high_threshold THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS tir_percentage,
        ROUND(100.0 * SUM(CASE WHEN glucose_mg_dl < p_low_threshold THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS tbr_percentage,
        ROUND(100.0 * SUM(CASE WHEN glucose_mg_dl > p_high_threshold THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS tar_percentage
    FROM cgm_readings
    WHERE patient_id = p_patient_id
      AND time BETWEEN p_start_time AND p_end_time;
END;
$$ LANGUAGE plpgsql;

-- Function to get glucose coefficient of variation
CREATE OR REPLACE FUNCTION calculate_glucose_cv(
    p_patient_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS DECIMAL AS $$
DECLARE
    v_cv DECIMAL;
BEGIN
    SELECT
        ROUND(100.0 * STDDEV(glucose_mg_dl) / NULLIF(AVG(glucose_mg_dl), 0), 2)
    INTO v_cv
    FROM cgm_readings
    WHERE patient_id = p_patient_id
      AND time BETWEEN p_start_time AND p_end_time;

    RETURN v_cv;
END;
$$ LANGUAGE plpgsql;

COMMIT;
