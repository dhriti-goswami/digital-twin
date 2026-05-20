// Patient types - matching backend schemas
export interface Patient {
  id: number;
  external_id: string;
  name?: string; // Added in frontend for display
  age: number;
  gender?: string;
  weight_kg: number;
  height_cm: number;
  diabetes_type: string;
  hba1c_baseline?: number;
  carb_ratio?: number;
  correction_factor?: number;
  created_at: string;
}

export interface PatientCreate {
  external_id?: string;
  name?: string; // Frontend-only, used to generate external_id
  age: number;
  gender?: 'M' | 'F' | 'Other' | string;
  weight_kg?: number;
  height_cm?: number;
  diabetes_type: string;
  hba1c_baseline?: number;
  carb_ratio?: number;
  correction_factor?: number;
  diagnosis_year?: number;
}

// Glucose types
export interface CGMReading {
  timestamp: string;
  glucose_mg_dl: number;
  trend?: GlucoseTrend;
  trend_rate?: number;
  device_id?: string;
}

export type GlucoseTrend =
  | 'RISING_FAST'
  | 'RISING'
  | 'STABLE'
  | 'FALLING'
  | 'FALLING_FAST';

export interface GlucoseStats {
  patient_id: number;
  period_hours: number;
  average_glucose: number;
  std_glucose: number;
  min_glucose: number;
  max_glucose: number;
  time_in_range: number;
  time_below_range: number;
  time_above_range: number;
  coefficient_of_variation: number;
  hypo_events: number;
  hyper_events: number;
}

export type GlucoseStatus = 'critical-low' | 'low' | 'normal' | 'high' | 'critical-high';

// Prediction types - used by the trained ML model
export interface PredictionRequest {
  patient_id: number;
  horizon_minutes?: number;
}

export interface PredictionResponse {
  patient_id: number;
  current_glucose: number;
  predictions: Record<string, number>; // {"30min": 120, "60min": 115, ...}
  confidence_intervals: Record<string, [number, number]>;
  timestamp: string;
  risk_level: 'low' | 'normal' | 'elevated' | 'high';
  model_used: boolean; // false = trend fallback was used (ML model unavailable)
}

// Simulation types
export interface SimulationRequest {
  patient_id: number;
  current_glucose?: number;
  carbs_grams?: number;
  insulin_units?: number;
  exercise_minutes?: number;
  exercise_intensity?: 'light' | 'moderate' | 'vigorous';
}

export interface SimulationPoint {
  time: number;
  glucose: number;
}

export interface SimulationResponse {
  patient_id: number;
  current_glucose: number;
  simulated_trajectory: SimulationPoint[];
  peak_glucose: number;
  peak_time_minutes: number;
  time_to_baseline_minutes: number;
  recommendations: string[];
}

// Chat types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface ChatRequest {
  patient_id: number;
  message: string;
  include_context?: boolean;
}

export interface ChatResponse {
  patient_id: number;
  response: string;
  suggested_actions?: string[];
  alerts?: Array<{ type: string; severity: string }>;
  timestamp: string;
}

// Insulin types
export interface InsulinDose {
  timestamp: string;
  dose_units: number;
  insulin_type: 'rapid' | 'long' | string;
  is_meal_bolus?: boolean;
  is_correction?: boolean;
}

// Meal types
export interface Meal {
  timestamp: string;
  carbs_grams: number;
  meal_type?: string;
  description?: string;
  protein_grams?: number;
  fat_grams?: number;
  glycemic_index?: number;
}

// Explanation types - SHAP analysis
export interface ExplanationResponse {
  patient_id: number;
  horizon_minutes: number;
  predicted_glucose: number;
  top_factors: Array<{
    feature: string;
    importance: number;
    description: string;
  }>;
  explanation_text: string;
  risk_level: string;
}

// Drift detection types
export interface DriftStatus {
  patient_id: number;
  drift_detected: boolean;
  drift_type: string | null;
  metric_values: Record<string, number>;
  last_checked: string;
  retrain_recommended: boolean;
}

// Onboarding types
export interface OnboardingData {
  step: number;
  name: string;
  age: number;
  gender: string;
  weight_kg: number;
  height_cm: number;
  diabetes_type: string;
  diabetesType?: string; // UI convenience field (mapped to diabetes_type)
  yearsWithDiabetes?: number;
  usesInsulin?: boolean;
  hba1c_baseline?: number;
  carb_ratio: number;
  carbRatio?: number; // UI convenience field (mapped to carb_ratio)
  correction_factor: number;
  correctionFactor?: number; // UI convenience field (mapped to correction_factor)
  current_glucose?: number;
  currentGlucose?: number; // UI convenience field (mapped to current_glucose)
  targetMin?: number;
  targetMax?: number;
}

// Health check response
export interface HealthStatus {
  status: string;
  timestamp?: string;
  // Backend returns nested components object
  components?: {
    model_loaded: boolean;
    agent: boolean;
    rag: boolean;
  };
  // Legacy flat fields for backwards compatibility
  model_loaded?: boolean;
  agent_ready?: boolean;
  rag_ready?: boolean;
}
