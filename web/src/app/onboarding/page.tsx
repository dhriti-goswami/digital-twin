'use client';

import * as React from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card } from '@/components/ui/Card';
import { usePatientStore } from '@/stores/patient';
import { createPatient, ingestCGM } from '@/lib/api';
import {
  User,
  Activity,
  Pill,
  CheckCircle,
  ArrowRight,
  ArrowLeft,
  Droplet,
} from 'lucide-react';
import type { OnboardingData } from '@/lib/types';

const STEPS = [
  { id: 1, title: 'Welcome', icon: User },
  { id: 2, title: 'Diabetes Profile', icon: Activity },
  { id: 3, title: 'Treatment', icon: Pill },
  { id: 4, title: 'Current Status', icon: Droplet },
  { id: 5, title: 'Complete', icon: CheckCircle },
];

export default function OnboardingPage() {
  const router = useRouter();
  const { onboardingData, updateOnboarding, setPatient, setOnboarded, setCurrentGlucose } = usePatientStore();
  const [currentStep, setCurrentStep] = React.useState(1);
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const handleNext = async () => {
    setError(null);

    if (currentStep === 4) {
      // Final step - create patient
      setIsLoading(true);
      try {
        const patient = await createPatient({
          external_id: `${onboardingData.name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
          name: onboardingData.name,
          age: onboardingData.age,
          gender: onboardingData.gender === 'Male' ? 'M' : onboardingData.gender === 'Female' ? 'F' : 'Other',
          weight_kg: onboardingData.weight_kg || 70,
          height_cm: onboardingData.height_cm || 170,
          diabetes_type: onboardingData.diabetesType || onboardingData.diabetes_type || 'type1',
          hba1c_baseline: onboardingData.hba1c_baseline,
          carb_ratio: onboardingData.carb_ratio || onboardingData.carbRatio || 10,
          correction_factor: onboardingData.correction_factor || onboardingData.correctionFactor || 50,
        });

        setPatient({
          id: patient.id,
          external_id: patient.external_id,
          name: onboardingData.name,
          age: patient.age,
          gender: patient.gender,
          weight_kg: patient.weight_kg,
          height_cm: patient.height_cm,
          diabetes_type: patient.diabetes_type,
          created_at: patient.created_at,
        });

        // Ingest initial glucose
        if (onboardingData.currentGlucose) {
          await ingestCGM(patient.id, onboardingData.currentGlucose);
          setCurrentGlucose(onboardingData.currentGlucose);
        }

        setCurrentStep(5);
      } catch (err) {
        console.error('Failed to create patient:', err);
        setError('Failed to create profile. Please try again.');
      }
      setIsLoading(false);
    } else {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    setCurrentStep((prev) => prev - 1);
  };

  const handleComplete = () => {
    setOnboarded(true);
    router.push('/dashboard');
  };

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return onboardingData.name && onboardingData.age && onboardingData.gender;
      case 2:
        return onboardingData.diabetesType || onboardingData.diabetes_type;
      case 3:
        return true; // Treatment info is optional
      case 4:
        return onboardingData.currentGlucose && onboardingData.currentGlucose > 0;
      default:
        return true;
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Card className="w-full max-w-lg">
        {/* Progress */}
        <div className="flex items-center justify-between mb-8">
          {STEPS.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep === step.id;
            const isCompleted = currentStep > step.id;

            return (
              <React.Fragment key={step.id}>
                <motion.div
                  className={cn(
                    'w-10 h-10 rounded-full flex items-center justify-center',
                    'border-2 transition-colors',
                    isActive && 'border-foreground bg-foreground text-background',
                    isCompleted && 'border-success bg-success text-success-foreground',
                    !isActive && !isCompleted && 'border-border text-muted-foreground'
                  )}
                  animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                  transition={{ duration: 0.3 }}
                >
                  <Icon className="h-5 w-5" />
                </motion.div>
                {index < STEPS.length - 1 && (
                  <div
                    className={cn(
                      'flex-1 h-0.5 mx-2',
                      isCompleted ? 'bg-success' : 'bg-border'
                    )}
                  />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {currentStep === 1 && (
              <Step1 data={onboardingData} onChange={updateOnboarding} />
            )}
            {currentStep === 2 && (
              <Step2 data={onboardingData} onChange={updateOnboarding} />
            )}
            {currentStep === 3 && (
              <Step3 data={onboardingData} onChange={updateOnboarding} />
            )}
            {currentStep === 4 && (
              <Step4 data={onboardingData} onChange={updateOnboarding} />
            )}
            {currentStep === 5 && <Step5 name={onboardingData.name} />}
          </motion.div>
        </AnimatePresence>

        {error && (
          <p className="text-destructive text-sm mt-4">{error}</p>
        )}

        {/* Navigation */}
        <div className="flex justify-between mt-8">
          {currentStep > 1 && currentStep < 5 ? (
            <Button type="button" variant="ghost" onClick={handleBack}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
          ) : (
            <div />
          )}

          {currentStep < 5 ? (
            <Button
              type="button"
              onClick={handleNext}
              disabled={!canProceed() || isLoading}
              isLoading={isLoading}
            >
              {currentStep === 4 ? 'Create Profile' : 'Next'}
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          ) : (
            <Button type="button" onClick={handleComplete} className="w-full">
              Go to Dashboard
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          )}
        </div>
      </Card>
    </div>
  );
}

function Step1({
  data,
  onChange,
}: {
  data: OnboardingData;
  onChange: (data: Partial<OnboardingData>) => void;
}) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Welcome</h2>
        <p className="text-muted-foreground">
          Let&apos;s set up your digital twin profile
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Name</label>
          <Input
            placeholder="Your name"
            value={data.name}
            onChange={(e) => onChange({ name: e.target.value })}
            icon={User}
          />
        </div>

        <div>
          <label className="text-sm font-medium mb-2 block">Age</label>
          <Input
            type="number"
            placeholder="Your age"
            value={data.age || ''}
            onChange={(e) => onChange({ age: parseInt(e.target.value) || undefined })}
          />
        </div>

        <div>
          <label className="text-sm font-medium mb-2 block">Gender</label>
          <div className="grid grid-cols-3 gap-2">
            {['Male', 'Female', 'Other'].map((gender) => (
              <Button
                key={gender}
                type="button"
                variant={data.gender === gender ? 'default' : 'outline'}
                onClick={() => onChange({ gender })}
                className="w-full"
              >
                {gender}
              </Button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Step2({
  data,
  onChange,
}: {
  data: OnboardingData;
  onChange: (data: Partial<OnboardingData>) => void;
}) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Diabetes Profile</h2>
        <p className="text-muted-foreground">
          Tell us about your diabetes type
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Diabetes Type</label>
          <div className="grid grid-cols-1 gap-2">
            {[
              { value: 'Type 1', desc: 'Insulin-dependent' },
              { value: 'Type 2', desc: 'Often managed with medication' },
              { value: 'Gestational', desc: 'Pregnancy-related' },
              { value: 'Prediabetes', desc: 'At risk' },
            ].map(({ value, desc }) => (
              <Button
                key={value}
                type="button"
                variant={data.diabetesType === value ? 'default' : 'outline'}
                onClick={() => onChange({ diabetesType: value })}
                className="w-full justify-start h-auto py-3"
              >
                <div className="text-left">
                  <div className="font-medium">{value}</div>
                  <div className="text-xs opacity-70">{desc}</div>
                </div>
              </Button>
            ))}
          </div>
        </div>

        <div>
          <label className="text-sm font-medium mb-2 block">
            Years with Diabetes
          </label>
          <Input
            type="number"
            placeholder="Years since diagnosis"
            value={data.yearsWithDiabetes || ''}
            onChange={(e) =>
              onChange({ yearsWithDiabetes: parseInt(e.target.value) || undefined })
            }
          />
        </div>
      </div>
    </div>
  );
}

function Step3({
  data,
  onChange,
}: {
  data: OnboardingData;
  onChange: (data: Partial<OnboardingData>) => void;
}) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Treatment</h2>
        <p className="text-muted-foreground">
          Your current treatment (optional)
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Uses Insulin</label>
          <div className="grid grid-cols-2 gap-2">
            <Button
              type="button"
              variant={data.usesInsulin === true ? 'default' : 'outline'}
              onClick={() => onChange({ usesInsulin: true })}
            >
              Yes
            </Button>
            <Button
              type="button"
              variant={data.usesInsulin === false ? 'default' : 'outline'}
              onClick={() => onChange({ usesInsulin: false })}
            >
              No
            </Button>
          </div>
        </div>

        {data.usesInsulin && (
          <>
            <div>
              <label className="text-sm font-medium mb-2 block">
                Carb Ratio (1 unit per X grams)
              </label>
              <Input
                type="number"
                placeholder="e.g., 10"
                value={data.carbRatio || ''}
                onChange={(e) =>
                  onChange({ carbRatio: parseInt(e.target.value) || undefined })
                }
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">
                Correction Factor (1 unit lowers X mg/dL)
              </label>
              <Input
                type="number"
                placeholder="e.g., 50"
                value={data.correctionFactor || ''}
                onChange={(e) =>
                  onChange({
                    correctionFactor: parseInt(e.target.value) || undefined,
                  })
                }
              />
            </div>
          </>
        )}

        <div>
          <label className="text-sm font-medium mb-2 block">Target Glucose Range</label>
          <div className="grid grid-cols-2 gap-2">
            <Input
              type="number"
              placeholder="Min (e.g., 70)"
              value={data.targetMin || ''}
              onChange={(e) =>
                onChange({ targetMin: parseInt(e.target.value) || undefined })
              }
            />
            <Input
              type="number"
              placeholder="Max (e.g., 180)"
              value={data.targetMax || ''}
              onChange={(e) =>
                onChange({ targetMax: parseInt(e.target.value) || undefined })
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function Step4({
  data,
  onChange,
}: {
  data: OnboardingData;
  onChange: (data: Partial<OnboardingData>) => void;
}) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Current Status</h2>
        <p className="text-muted-foreground">
          Enter your current glucose reading
        </p>
      </div>

      <div className="flex flex-col items-center">
        <div className="relative">
          <input
            type="number"
            value={data.currentGlucose || ''}
            onChange={(e) =>
              onChange({ currentGlucose: parseInt(e.target.value) || undefined })
            }
            placeholder="120"
            className={cn(
              'w-40 h-24 text-center text-5xl font-bold bg-transparent',
              'border-b-4 border-border focus:border-foreground',
              'outline-none transition-colors tabular-nums',
              '[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none'
            )}
          />
          <span className="absolute -right-16 bottom-4 text-lg text-muted-foreground">
            mg/dL
          </span>
        </div>

        <p className="text-sm text-muted-foreground mt-4 text-center">
          This will be used to generate your first predictions
        </p>
      </div>
    </div>
  );
}

function Step5({ name }: { name: string }) {
  return (
    <div className="text-center py-8">
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: 'spring', stiffness: 200, damping: 15 }}
      >
        <div className="w-20 h-20 rounded-full bg-success mx-auto flex items-center justify-center mb-6">
          <CheckCircle className="h-10 w-10 text-success-foreground" />
        </div>
      </motion.div>

      <h2 className="text-2xl font-bold mb-2">All Set, {name}!</h2>
      <p className="text-muted-foreground">
        Your digital twin is ready. Let&apos;s head to your dashboard.
      </p>
    </div>
  );
}
