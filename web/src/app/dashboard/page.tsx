'use client';

import * as React from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Navbar } from '@/components/layout/Navbar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { GlucoseInput } from '@/components/glucose/GlucoseInput';
import { GlucoseDisplay } from '@/components/ui/GlucoseDisplay';
import { PredictionTimeline } from '@/components/predictions/PredictionTimeline';
import { LogInsulinModal } from '@/components/modals/LogInsulinModal';
import { LogMealModal } from '@/components/modals/LogMealModal';
import { FadeIn, Stagger } from '@/components/animations/PageTransition';
import { PredictionChart, TimeInRangeChart } from '@/components/charts';
import { usePatientStore } from '@/stores/patient';
import { ingestCGM, getPredictions, logInsulin, logMeal, healthCheck } from '@/lib/api';
import {
  MessageSquare,
  Utensils,
  Syringe,
  Activity,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  BarChart3,
} from 'lucide-react';

interface ActivityLog {
  id: string;
  time: string;
  text: string;
  icon: React.ElementType;
}

export default function DashboardPage() {
  const router = useRouter();
  const {
    patient,
    isOnboarded,
    currentGlucose,
    setCurrentGlucose,
    predictions,
    setPredictions,
  } = usePatientStore();

  const [inputGlucose, setInputGlucose] = React.useState(currentGlucose || 120);
  const [isLoading, setIsLoading] = React.useState(false);
  const [showInsulinModal, setShowInsulinModal] = React.useState(false);
  const [showMealModal, setShowMealModal] = React.useState(false);
  const [activityLog, setActivityLog] = React.useState<ActivityLog[]>([]);
  const [backendStatus, setBackendStatus] = React.useState<'checking' | 'connected' | 'error'>('checking');
  const [error, setError] = React.useState<string | null>(null);
  const [glucoseHistory, setGlucoseHistory] = React.useState<number[]>([]);

  // Check backend connection on mount
  React.useEffect(() => {
    checkBackend();
  }, []);

  React.useEffect(() => {
    if (!isOnboarded) {
      router.push('/onboarding');
    }
  }, [isOnboarded, router]);

  const checkBackend = async () => {
    setBackendStatus('checking');
    try {
      const health = await healthCheck();
      // Backend returns { status, components: { model_loaded, agent, rag } }
      const modelLoaded = (health as any).components?.model_loaded ?? health.model_loaded ?? false;

      if (health.status === 'healthy' && modelLoaded) {
        setBackendStatus('connected');
        setError(null);
      } else {
        setBackendStatus('error');
        setError(`Model not loaded. Status: ${health.status}`);
      }
    } catch {
      setBackendStatus('error');
      setError('Cannot connect to backend. Run: docker compose up');
    }
  };

  const addActivity = (text: string, icon: React.ElementType) => {
    const now = new Date();
    setActivityLog((prev) => [
      {
        id: Date.now().toString(),
        time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text,
        icon,
      },
      ...prev.slice(0, 9),
    ]);
  };

  const refreshPredictions = async () => {
    if (!patient?.id) return;

    try {
      const preds = await getPredictions(patient.id, 120);
      setPredictions(preds);
      setError(null);
    } catch (err) {
      console.error('Failed to refresh predictions:', err);
      setError('Failed to get predictions from ML model');
    }
  };

  const handleSubmitGlucose = async () => {
    if (!patient?.id || !inputGlucose) return;

    setIsLoading(true);
    setError(null);

    try {
      await ingestCGM(patient.id, inputGlucose);
      setCurrentGlucose(inputGlucose);

      // Add to glucose history for charts
      setGlucoseHistory(prev => [...prev, inputGlucose].slice(-50));

      // Get predictions from the real ML model
      const preds = await getPredictions(patient.id, 120);
      setPredictions(preds);

      addActivity(`Glucose: ${inputGlucose} mg/dL`, Activity);
    } catch (err) {
      console.error('Failed to submit glucose:', err);
      setError('Failed to submit glucose. Check backend connection.');
    }
    setIsLoading(false);
  };

  const handleLogInsulin = async (data: { units: number; type: string; isMealBolus: boolean }) => {
    if (!patient?.id) return;

    try {
      await logInsulin(Number(patient.id), {
        timestamp: new Date().toISOString(),
        dose_units: data.units,
        insulin_type: data.type === 'rapid' ? 'rapid' : 'long',
        is_meal_bolus: data.isMealBolus,
      });

      addActivity(`Insulin: ${data.units}u ${data.type}${data.isMealBolus ? ' (meal)' : ''}`, Syringe);

      // Refresh predictions - insulin affects future glucose via IOB
      await refreshPredictions();
    } catch (err) {
      console.error('Failed to log insulin:', err);
      setError('Failed to log insulin');
      throw err;
    }
  };

  const handleLogMeal = async (data: { carbs: number; mealType: string; description?: string }) => {
    if (!patient?.id) return;

    try {
      await logMeal(Number(patient.id), {
        timestamp: new Date().toISOString(),
        carbs_grams: data.carbs,
        meal_type: data.mealType,
        description: data.description,
      });

      addActivity(`Meal: ${data.carbs}g carbs (${data.mealType})`, Utensils);

      // Refresh predictions - carbs affect future glucose via COB
      await refreshPredictions();
    } catch (err) {
      console.error('Failed to log meal:', err);
      setError('Failed to log meal');
      throw err;
    }
  };

  if (!isOnboarded || !patient) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="pt-20 pb-8 px-4 max-w-7xl mx-auto">
        {/* Backend Status Banner */}
        {backendStatus === 'error' && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-4 rounded-lg border border-destructive bg-destructive/10 flex items-center gap-3"
          >
            <XCircle className="h-5 w-5 text-destructive flex-shrink-0" />
            <div className="flex-1">
              <p className="font-medium text-destructive">Backend Not Connected</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </div>
            <Button variant="outline" size="sm" onClick={checkBackend}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </motion.div>
        )}

        {backendStatus === 'checking' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-4 p-3 rounded-lg border border-border bg-muted/50 flex items-center gap-2"
          >
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span className="text-sm">Connecting to ML model...</span>
          </motion.div>
        )}

        {backendStatus === 'connected' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-4 p-3 rounded-lg border border-success/30 bg-success/10 flex items-center gap-2"
          >
            <CheckCircle className="h-4 w-4 text-success" />
            <span className="text-sm text-success">ML Model Active - Production Mode</span>
          </motion.div>
        )}

        {error && backendStatus === 'connected' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-4 p-3 rounded-lg border border-warning/30 bg-warning/10 flex items-center gap-2"
          >
            <AlertTriangle className="h-4 w-4 text-warning" />
            <span className="text-sm">{error}</span>
          </motion.div>
        )}

        <FadeIn>
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">
              Hello, {patient.name || patient.external_id}
            </h1>
            <p className="text-muted-foreground">
              Your diabetes management dashboard
            </p>
          </div>
        </FadeIn>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main glucose section */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardContent className="py-8">
                {currentGlucose ? (
                  <div className="flex flex-col items-center gap-6">
                    <GlucoseDisplay
                      value={currentGlucose}
                      size="xl"
                      showStatus
                    />
                    {predictions?.risk_level && (
                      <div className={cn(
                        'px-3 py-1 rounded-full text-sm font-medium',
                        predictions.risk_level === 'low' && 'bg-success/20 text-success',
                        predictions.risk_level === 'normal' && 'bg-muted text-foreground',
                        predictions.risk_level === 'elevated' && 'bg-warning/20 text-warning',
                        predictions.risk_level === 'high' && 'bg-destructive/20 text-destructive'
                      )}>
                        Risk: {predictions.risk_level.charAt(0).toUpperCase() + predictions.risk_level.slice(1)}
                      </div>
                    )}
                    <Button
                      variant="outline"
                      onClick={() => setCurrentGlucose(null)}
                    >
                      Update Reading
                    </Button>
                  </div>
                ) : (
                  <GlucoseInput
                    value={inputGlucose}
                    onChange={setInputGlucose}
                    onSubmit={handleSubmitGlucose}
                    isLoading={isLoading}
                  />
                )}
              </CardContent>
            </Card>

            {/* Predictions from ML Model */}
            <PredictionTimeline predictions={predictions} />

            {/* Charts Section */}
            {currentGlucose && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-6"
              >
                <PredictionChart
                  currentGlucose={currentGlucose}
                  predictions={predictions}
                />
                <TimeInRangeChart
                  inRange={glucoseHistory.filter(g => g >= 70 && g <= 180).length || (currentGlucose >= 70 && currentGlucose <= 180 ? 1 : 0)}
                  belowRange={glucoseHistory.filter(g => g < 70).length || (currentGlucose < 70 ? 1 : 0)}
                  aboveRange={glucoseHistory.filter(g => g > 180).length || (currentGlucose > 180 ? 1 : 0)}
                  size="sm"
                />
              </motion.div>
            )}

            {/* Quick Actions */}
            <Stagger className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <QuickAction
                icon={MessageSquare}
                label="Chat with AI"
                onClick={() => router.push('/chat')}
              />
              <QuickAction
                icon={Utensils}
                label="Log Meal"
                onClick={() => setShowMealModal(true)}
                disabled={backendStatus === 'error'}
              />
              <QuickAction
                icon={Syringe}
                label="Log Insulin"
                onClick={() => setShowInsulinModal(true)}
                disabled={backendStatus === 'error'}
              />
              <QuickAction
                icon={Activity}
                label="Simulate"
                onClick={() => router.push('/simulate')}
              />
            </Stagger>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Stats Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  ML Predictions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <StatItem
                    label="Current Glucose"
                    value={predictions?.current_glucose ? `${Math.round(predictions.current_glucose)} mg/dL` : currentGlucose ? `${currentGlucose} mg/dL` : '--'}
                  />
                  <StatItem
                    label="30min Prediction"
                    value={predictions?.predictions?.['30min'] ? `${Math.round(predictions.predictions['30min'])} mg/dL` : '--'}
                  />
                  <StatItem
                    label="60min Prediction"
                    value={predictions?.predictions?.['60min'] ? `${Math.round(predictions.predictions['60min'])} mg/dL` : '--'}
                  />
                  <StatItem
                    label="Risk Level"
                    value={predictions?.risk_level ? predictions.risk_level.charAt(0).toUpperCase() + predictions.risk_level.slice(1) : '--'}
                    color={
                      predictions?.risk_level === 'low' ? 'text-success' :
                      predictions?.risk_level === 'elevated' ? 'text-warning' :
                      predictions?.risk_level === 'high' ? 'text-destructive' :
                      undefined
                    }
                  />
                </div>
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5" />
                  Recent Activity
                </CardTitle>
              </CardHeader>
              <CardContent>
                {activityLog.length > 0 ? (
                  <div className="space-y-3">
                    {activityLog.map((item) => (
                      <ActivityItem
                        key={item.id}
                        time={item.time}
                        text={item.text}
                        icon={item.icon}
                      />
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground text-sm">
                    No activity yet. Enter your glucose to get started.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Profile Card */}
            <Card>
              <CardContent className="pt-6">
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Type</span>
                    <span className="font-medium">{patient.diabetes_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Age</span>
                    <span className="font-medium">{patient.age}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Backend</span>
                    <span className={cn(
                      'font-medium',
                      backendStatus === 'connected' ? 'text-success' : 'text-destructive'
                    )}>
                      {backendStatus === 'connected' ? 'Production' : backendStatus === 'checking' ? 'Connecting...' : 'Disconnected'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Patient ID</span>
                    <span className="font-mono text-xs">{patient.id}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* Modals */}
      <LogInsulinModal
        isOpen={showInsulinModal}
        onClose={() => setShowInsulinModal(false)}
        onSubmit={handleLogInsulin}
      />
      <LogMealModal
        isOpen={showMealModal}
        onClose={() => setShowMealModal(false)}
        onSubmit={handleLogMeal}
      />
    </div>
  );
}

function QuickAction({
  icon: Icon,
  label,
  onClick,
  disabled,
}: {
  icon: React.ElementType;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <motion.button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'flex flex-col items-center gap-2 p-4 rounded-xl',
        'border border-border bg-card',
        'hover:bg-accent transition-colors',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
      whileHover={!disabled ? { scale: 1.02 } : undefined}
      whileTap={!disabled ? { scale: 0.98 } : undefined}
    >
      <Icon className="h-6 w-6" />
      <span className="text-sm font-medium">{label}</span>
    </motion.button>
  );
}

function StatItem({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-muted-foreground text-sm">{label}</span>
      <span className={cn('font-semibold', color)}>{value}</span>
    </div>
  );
}

function ActivityItem({
  time,
  text,
  icon: Icon,
}: {
  time: string;
  text: string;
  icon: React.ElementType;
}) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
        <Icon className="h-4 w-4" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm truncate">{text}</p>
        <p className="text-xs text-muted-foreground">{time}</p>
      </div>
    </div>
  );
}
