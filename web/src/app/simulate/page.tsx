'use client';

import * as React from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Navbar } from '@/components/layout/Navbar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { GlucoseDisplay } from '@/components/ui/GlucoseDisplay';
import { PredictionTimeline } from '@/components/predictions/PredictionTimeline';
import { FadeIn } from '@/components/animations/PageTransition';
import { usePatientStore } from '@/stores/patient';
import { runSimulation } from '@/lib/api';
import {
  Utensils,
  Syringe,
  Dumbbell,
  Play,
  RotateCcw,
} from 'lucide-react';
import type { PredictionResponse } from '@/lib/types';

export default function SimulatePage() {
  const router = useRouter();
  const { isOnboarded, patient, currentGlucose } = usePatientStore();
  const [simulationResult, setSimulationResult] = React.useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);

  const [carbs, setCarbs] = React.useState(0);
  const [insulin, setInsulin] = React.useState(0);
  const [exercise, setExercise] = React.useState(0);

  React.useEffect(() => {
    if (!isOnboarded) {
      router.push('/onboarding');
    }
  }, [isOnboarded, router]);

  const handleSimulate = async () => {
    if (!patient?.id || !currentGlucose) return;

    setIsLoading(true);
    try {
      const result = await runSimulation(patient.id, {
        current_glucose: currentGlucose ?? undefined,
        carbs_grams: carbs || undefined,
        insulin_units: insulin || undefined,
        exercise_minutes: exercise || undefined,
      });

      // Convert SimulationResponse to PredictionResponse format
      const trajectory = result.simulated_trajectory || [];
      const getValueAtTime = (minutes: number) => {
        const point = trajectory.find(p => p.time === minutes);
        return point?.glucose || currentGlucose;
      };

      setSimulationResult({
        patient_id: result.patient_id,
        current_glucose: result.current_glucose,
        predictions: {
          '30min': getValueAtTime(30),
          '60min': getValueAtTime(60),
          '90min': getValueAtTime(90),
          '120min': getValueAtTime(120),
        },
        confidence_intervals: {
          '30min': [getValueAtTime(30) - 10, getValueAtTime(30) + 10],
          '60min': [getValueAtTime(60) - 15, getValueAtTime(60) + 15],
          '90min': [getValueAtTime(90) - 20, getValueAtTime(90) + 20],
          '120min': [getValueAtTime(120) - 25, getValueAtTime(120) + 25],
        },
        timestamp: new Date().toISOString(),
        risk_level: 'normal',
      });
    } catch (error) {
      console.error('Simulation failed:', error);
      alert('Simulation failed. Please check backend connection.');
    }
    setIsLoading(false);
  };

  const handleReset = () => {
    setCarbs(0);
    setInsulin(0);
    setExercise(0);
    setSimulationResult(null);
  };

  if (!isOnboarded) return null;

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="pt-20 pb-8 px-4 max-w-5xl mx-auto">
        <FadeIn>
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">What-If Simulator</h1>
            <p className="text-muted-foreground">
              Explore how different factors affect your glucose
            </p>
          </div>
        </FadeIn>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Input Section */}
          <div className="space-y-6">
            {/* Current Glucose */}
            <Card>
              <CardHeader>
                <CardTitle>Current Status</CardTitle>
              </CardHeader>
              <CardContent>
                {currentGlucose ? (
                  <GlucoseDisplay value={currentGlucose} size="lg" showStatus />
                ) : (
                  <p className="text-muted-foreground text-center py-4">
                    Enter your glucose on the dashboard first
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Simulation Inputs */}
            <Card>
              <CardHeader>
                <CardTitle>Simulation Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <SimulationSlider
                  icon={Utensils}
                  label="Carbohydrates"
                  value={carbs}
                  onChange={setCarbs}
                  max={150}
                  unit="g"
                  color="text-warning"
                />

                <SimulationSlider
                  icon={Syringe}
                  label="Insulin"
                  value={insulin}
                  onChange={setInsulin}
                  max={20}
                  unit="units"
                  step={0.5}
                  color="text-primary"
                />

                <SimulationSlider
                  icon={Dumbbell}
                  label="Exercise"
                  value={exercise}
                  onChange={setExercise}
                  max={120}
                  unit="min"
                  color="text-success"
                />

                <div className="flex gap-3 pt-4">
                  <Button
                    onClick={handleSimulate}
                    isLoading={isLoading}
                    disabled={!currentGlucose}
                    className="flex-1"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    Run Simulation
                  </Button>
                  <Button variant="outline" onClick={handleReset}>
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            <PredictionTimeline predictions={simulationResult} />

            {simulationResult && (
              <Card>
                <CardHeader>
                  <CardTitle>Simulation Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-4 rounded-lg bg-accent">
                        <p className="text-sm text-muted-foreground">Starting</p>
                        <p className="text-2xl font-bold">{currentGlucose}</p>
                        <p className="text-xs text-muted-foreground">mg/dL</p>
                      </div>
                      <div className="text-center p-4 rounded-lg bg-accent">
                        <p className="text-sm text-muted-foreground">After 2 hours</p>
                        <p className="text-2xl font-bold">
                          {Math.round(simulationResult.predictions['120min'])}
                        </p>
                        <p className="text-xs text-muted-foreground">mg/dL</p>
                      </div>
                    </div>

                    <div className="text-sm text-muted-foreground">
                      <p className="mb-2">Factors applied:</p>
                      <ul className="list-disc list-inside space-y-1">
                        {carbs > 0 && <li>{carbs}g carbohydrates (raises glucose)</li>}
                        {insulin > 0 && <li>{insulin} units insulin (lowers glucose)</li>}
                        {exercise > 0 && <li>{exercise} min exercise (lowers glucose)</li>}
                        {carbs === 0 && insulin === 0 && exercise === 0 && (
                          <li>No changes applied</li>
                        )}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {!simulationResult && (
              <Card className="flex items-center justify-center min-h-[200px]">
                <div className="text-center text-muted-foreground">
                  <Dumbbell className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Adjust the parameters and run a simulation</p>
                  <p className="text-sm">to see predicted glucose outcomes</p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

function SimulationSlider({
  icon: Icon,
  label,
  value,
  onChange,
  max,
  unit,
  step = 1,
  color = 'text-foreground',
}: {
  icon: React.ElementType;
  label: string;
  value: number;
  onChange: (value: number) => void;
  max: number;
  unit: string;
  step?: number;
  color?: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className={cn('h-5 w-5', color)} />
          <span className="font-medium">{label}</span>
        </div>
        <span className={cn('font-bold', color)}>
          {value} {unit}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-accent rounded-lg appearance-none cursor-pointer"
      />
      <div className="flex justify-between text-xs text-muted-foreground mt-1">
        <span>0</span>
        <span>{max} {unit}</span>
      </div>
    </div>
  );
}
