import {
  Patient,
  PatientCreate,
  CGMReading,
  GlucoseStats,
  PredictionResponse,
  SimulationRequest,
  SimulationResponse,
  InsulinDose,
  Meal,
  ChatResponse,
} from './types';

// Production API base URL - Python FastAPI backend
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

class APIError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    // FastAPI validation errors return detail as an array of objects
    const detail = Array.isArray(error.detail)
      ? error.detail.map((e: { msg?: string; loc?: string[] }) => `${e.loc?.slice(-1)[0] ?? 'field'}: ${e.msg ?? 'invalid'}`).join(', ')
      : (error.detail || `HTTP ${response.status}`);
    throw new APIError(detail, response.status);
  }

  return response.json();
}

// Health check - verify backend is running
export async function healthCheck(): Promise<{
  status: string;
  model_loaded: boolean;
  agent_ready: boolean;
  rag_ready: boolean;
}> {
  return fetchAPI('/health');
}

// Patient API
export async function createPatient(data: PatientCreate): Promise<Patient> {
  return fetchAPI<Patient>('/api/v1/patients', {
    method: 'POST',
    body: JSON.stringify({
      external_id: data.external_id || `patient-${Date.now()}`,
      age: data.age,
      gender: data.gender,
      weight_kg: data.weight_kg || 70,
      height_cm: data.height_cm || 170,
      diabetes_type: data.diabetes_type,
      hba1c_baseline: data.hba1c_baseline,
      carb_ratio: data.carb_ratio || 10,
      correction_factor: data.correction_factor || 50,
    }),
  });
}

export async function getPatient(patientId: number): Promise<Patient> {
  return fetchAPI<Patient>(`/api/v1/patients/${patientId}`);
}

// CGM API
export async function ingestCGM(
  patientId: string | number,
  glucoseValue: number,
  trend?: string
): Promise<{ status: string; alerts?: string[] }> {
  const reading: CGMReading = {
    timestamp: new Date().toISOString(),
    glucose_mg_dl: glucoseValue,
    trend: trend as CGMReading['trend'],
    device_id: 'web_manual',
  };

  return fetchAPI(`/api/v1/ingest/cgm?patient_id=${patientId}`, {
    method: 'POST',
    body: JSON.stringify(reading),
  });
}

export async function batchIngestCGM(
  patientId: number,
  readings: CGMReading[]
): Promise<{ status: string; ingested_count: number }> {
  return fetchAPI('/api/v1/ingest/cgm/batch', {
    method: 'POST',
    body: JSON.stringify({
      patient_id: patientId,
      readings,
    }),
  });
}

export async function getCGMHistory(
  patientId: number,
  periodHours: number = 24
): Promise<{ time: string[]; glucose: number[] }> {
  return fetchAPI(`/api/v1/patients/${patientId}/cgm?period_hours=${periodHours}`);
}

export async function getGlucoseStats(
  patientId: number,
  periodHours: number = 24
): Promise<GlucoseStats> {
  return fetchAPI(`/api/v1/patients/${patientId}/stats?period_hours=${periodHours}`);
}

// Prediction API - uses the trained ML model
export async function getPredictions(
  patientId: string | number,
  horizonMinutes: number = 120
): Promise<PredictionResponse> {
  return fetchAPI<PredictionResponse>('/api/v1/predict', {
    method: 'POST',
    body: JSON.stringify({
      patient_id: Number(patientId),
      horizon_minutes: horizonMinutes,
    }),
  });
}

// Simulation API - what-if scenarios
export async function runSimulation(
  patientId: string | number,
  data: Partial<SimulationRequest>
): Promise<SimulationResponse> {
  return fetchAPI<SimulationResponse>('/api/v1/simulate', {
    method: 'POST',
    body: JSON.stringify({
      patient_id: Number(patientId),
      ...data,
    }),
  });
}

// Insulin API
export async function logInsulin(
  patientId: number,
  dose: InsulinDose
): Promise<{ status: string }> {
  return fetchAPI(`/api/v1/ingest/insulin?patient_id=${patientId}`, {
    method: 'POST',
    body: JSON.stringify(dose),
  });
}

// Meal API
export async function logMeal(
  patientId: number,
  meal: Meal
): Promise<{ status: string }> {
  return fetchAPI(`/api/v1/ingest/meal?patient_id=${patientId}`, {
    method: 'POST',
    body: JSON.stringify(meal),
  });
}

// Chat API - uses backend AI agent with full patient context
export async function sendChatMessage(
  patientId: number,
  message: string,
  includeContext: boolean = true
): Promise<ChatResponse> {
  return fetchAPI<ChatResponse>('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({
      patient_id: patientId,
      message,
      include_context: includeContext,
    }),
  });
}

// Explanation API - SHAP-based predictions
export async function getExplanation(
  patientId: number,
  horizonMinutes: number = 60
): Promise<{
  patient_id: number;
  horizon_minutes: number;
  predicted_glucose: number;
  top_factors: Array<{ feature: string; importance: number; description: string }>;
  explanation_text: string;
  risk_level: string;
}> {
  return fetchAPI('/api/v1/explain', {
    method: 'POST',
    body: JSON.stringify({
      patient_id: patientId,
      horizon_minutes: horizonMinutes,
    }),
  });
}

// Drift Detection API
export async function checkDrift(patientId: number): Promise<{
  patient_id: number;
  drift_detected: boolean;
  drift_type: string | null;
  metric_values: Record<string, number>;
  last_checked: string;
  retrain_recommended: boolean;
}> {
  return fetchAPI(`/api/v1/drift?patient_id=${patientId}`);
}

// Retraining API
export async function triggerRetrain(
  patientId?: number,
  force: boolean = false
): Promise<{
  status: string;
  job_id: string;
  estimated_time_minutes: number;
  message: string;
}> {
  return fetchAPI('/api/v1/retrain', {
    method: 'POST',
    body: JSON.stringify({
      patient_id: patientId,
      force,
    }),
  });
}

export { APIError };
