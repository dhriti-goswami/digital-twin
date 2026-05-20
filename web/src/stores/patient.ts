import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Patient, CGMReading, PredictionResponse, ChatMessage, OnboardingData } from '@/lib/types';

interface PatientState {
  patient: Patient | null;
  currentGlucose: number | null;
  glucoseHistory: CGMReading[];
  predictions: PredictionResponse | null;
  chatMessages: ChatMessage[];
  onboardingData: OnboardingData;
  isOnboarded: boolean;

  // Actions
  setPatient: (patient: Patient) => void;
  setCurrentGlucose: (glucose: number | null) => void;
  addGlucoseReading: (reading: CGMReading) => void;
  setPredictions: (predictions: PredictionResponse) => void;
  addChatMessage: (message: ChatMessage) => void;
  updateOnboardingData: (data: Partial<OnboardingData>) => void;
  updateOnboarding: (data: Partial<OnboardingData>) => void;
  setOnboarded: (value: boolean) => void;
  completeOnboarding: () => void;
  reset: () => void;
}

const initialOnboardingData: OnboardingData = {
  step: 1,
  name: '',
  age: 30,
  gender: '',
  weight_kg: 70,
  height_cm: 170,
  diabetes_type: 'type1',
  carb_ratio: 10,
  correction_factor: 50,
};

export const usePatientStore = create<PatientState>()(
  persist(
    (set) => ({
      patient: null,
      currentGlucose: null,
      glucoseHistory: [],
      predictions: null,
      chatMessages: [],
      onboardingData: initialOnboardingData,
      isOnboarded: false,

      setPatient: (patient) => set({ patient }),

      setCurrentGlucose: (glucose) => set({ currentGlucose: glucose }),

      addGlucoseReading: (reading) =>
        set((state) => ({
          glucoseHistory: [...state.glucoseHistory.slice(-287), reading],
          currentGlucose: reading.glucose_mg_dl,
        })),

      setPredictions: (predictions) => set({ predictions }),

      addChatMessage: (message) =>
        set((state) => ({
          chatMessages: [...state.chatMessages, message],
        })),

      updateOnboardingData: (data) =>
        set((state) => ({
          onboardingData: { ...state.onboardingData, ...data },
        })),

      updateOnboarding: (data) =>
        set((state) => ({
          onboardingData: { ...state.onboardingData, ...data },
        })),

      setOnboarded: (value) => set({ isOnboarded: value }),

      completeOnboarding: () => set({ isOnboarded: true }),

      reset: () =>
        set({
          patient: null,
          currentGlucose: null,
          glucoseHistory: [],
          predictions: null,
          chatMessages: [],
          onboardingData: initialOnboardingData,
          isOnboarded: false,
        }),
    }),
    {
      name: 'patient-storage',
      partialize: (state) => ({
        patient: state.patient,
        isOnboarded: state.isOnboarded,
        onboardingData: state.onboardingData,
      }),
    }
  )
);
