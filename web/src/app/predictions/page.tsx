'use client';

import * as React from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { cn, getGlucoseStatus, getGlucoseStatusColor } from '@/lib/utils';
import { Navbar } from '@/components/layout/Navbar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { GlucoseDisplay } from '@/components/ui/GlucoseDisplay';
import { FadeIn, Stagger } from '@/components/animations/PageTransition';
import { usePatientStore } from '@/stores/patient';
import {
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertCircle,
  CheckCircle,
  Info,
} from 'lucide-react';

export default function PredictionsPage() {
  const router = useRouter();
  const { isOnboarded, currentGlucose, predictions } = usePatientStore();

  React.useEffect(() => {
    if (!isOnboarded) {
      router.push('/onboarding');
    }
  }, [isOnboarded, router]);

  if (!isOnboarded) return null;

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="pt-20 pb-8 px-4 max-w-5xl mx-auto">
        <FadeIn>
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Glucose Predictions</h1>
            <p className="text-muted-foreground">
              AI-powered multi-horizon glucose forecasting
            </p>
          </div>
        </FadeIn>

        {!currentGlucose ? (
          <Card className="text-center py-12">
            <Info className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h2 className="text-xl font-semibold mb-2">No glucose data</h2>
            <p className="text-muted-foreground mb-4">
              Enter your current glucose reading on the dashboard to see predictions
            </p>
            <motion.button
              onClick={() => router.push('/dashboard')}
              className="px-4 py-2 bg-foreground text-background rounded-lg font-medium"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Go to Dashboard
            </motion.button>
          </Card>
        ) : (
          <div className="space-y-6">
            {/* Current Reading */}
            <Card>
              <CardContent className="py-8">
                <div className="flex items-center justify-center gap-8">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground mb-2">Current</p>
                    <GlucoseDisplay value={currentGlucose} size="xl" showStatus />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Predictions Grid */}
            {predictions ? (
              <Stagger className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {(['30min', '60min', '90min', '120min'] as const).map((horizon) => {
                  const value = Math.round(predictions.predictions[horizon]);
                  const confidence = predictions.confidence_intervals[horizon];
                  const status = getGlucoseStatus(value);
                  const delta = value - currentGlucose;
                  const TrendIcon = delta > 5 ? TrendingUp : delta < -5 ? TrendingDown : Minus;

                  return (
                    <PredictionCard
                      key={horizon}
                      horizon={horizon}
                      value={value}
                      delta={delta}
                      confidence={confidence}
                      status={status}
                      TrendIcon={TrendIcon}
                    />
                  );
                })}
              </Stagger>
            ) : (
              <Card className="text-center py-8">
                <Clock className="h-8 w-8 mx-auto mb-2 text-muted-foreground animate-pulse" />
                <p className="text-muted-foreground">Loading predictions...</p>
              </Card>
            )}

            {/* Analysis */}
            {predictions && (
              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Info className="h-5 w-5" />
                      Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {(() => {
                        const pred120 = predictions.predictions['120min'];
                        const trend = pred120 - currentGlucose;
                        const isStable = Math.abs(trend) < 20;
                        const isRising = trend >= 20;
                        const isFalling = trend <= -20;

                        return (
                          <>
                            <div className="flex items-start gap-3">
                              {isStable && <CheckCircle className="h-5 w-5 text-success mt-0.5" />}
                              {isRising && <AlertCircle className="h-5 w-5 text-warning mt-0.5" />}
                              {isFalling && <AlertCircle className="h-5 w-5 text-warning mt-0.5" />}
                              <div>
                                <p className="font-medium">
                                  {isStable && 'Stable Trend'}
                                  {isRising && 'Rising Trend'}
                                  {isFalling && 'Falling Trend'}
                                </p>
                                <p className="text-sm text-muted-foreground">
                                  {isStable && 'Your glucose is predicted to stay relatively stable over the next 2 hours.'}
                                  {isRising && `Your glucose is predicted to rise by approximately ${Math.round(trend)} mg/dL.`}
                                  {isFalling && `Your glucose is predicted to fall by approximately ${Math.abs(Math.round(trend))} mg/dL.`}
                                </p>
                              </div>
                            </div>

                            {pred120 > 180 && (
                              <div className="flex items-start gap-3">
                                <AlertCircle className="h-5 w-5 text-warning mt-0.5" />
                                <div>
                                  <p className="font-medium">Hyperglycemia Risk</p>
                                  <p className="text-sm text-muted-foreground">
                                    Predicted glucose may exceed target range. Consider a correction dose if appropriate.
                                  </p>
                                </div>
                              </div>
                            )}

                            {pred120 < 70 && (
                              <div className="flex items-start gap-3">
                                <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                                <div>
                                  <p className="font-medium">Hypoglycemia Risk</p>
                                  <p className="text-sm text-muted-foreground">
                                    Predicted glucose may fall below target. Consider having fast-acting carbs ready.
                                  </p>
                                </div>
                              </div>
                            )}
                          </>
                        );
                      })()}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Clock className="h-5 w-5" />
                      Model Info
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Architecture</span>
                        <span className="font-medium text-success">
                          Transformer + PINN
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">MAE</span>
                        <span className="font-medium">5.55 mg/dL</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Validation</span>
                        <span className="font-medium text-success">Excellent</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Horizons</span>
                        <span className="font-medium">30, 60, 90, 120 min</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Confidence</span>
                        <span className="font-medium">95% interval</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

function PredictionCard({
  horizon,
  value,
  delta,
  confidence,
  status,
  TrendIcon,
}: {
  horizon: string;
  value: number;
  delta: number;
  confidence: [number, number];
  status: ReturnType<typeof getGlucoseStatus>;
  TrendIcon: React.ElementType;
}) {
  const color = getGlucoseStatusColor(status);
  const horizonLabel = horizon.replace('min', ' minutes');

  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
    >
      <Card className="h-full">
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 mb-4">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">+{horizonLabel}</span>
          </div>

          <div className="flex items-baseline gap-2 mb-2">
            <span className={cn('text-4xl font-bold', color)}>{value}</span>
            <span className="text-muted-foreground">mg/dL</span>
          </div>

          <div className="flex items-center gap-2 mb-4">
            <TrendIcon className={cn('h-4 w-4', delta > 0 ? 'text-warning' : delta < 0 ? 'text-success' : 'text-muted-foreground')} />
            <span className={cn(
              'text-sm font-medium',
              delta > 0 ? 'text-warning' : delta < 0 ? 'text-success' : 'text-muted-foreground'
            )}>
              {delta > 0 ? '+' : ''}{delta} mg/dL
            </span>
          </div>

          <div className="text-xs text-muted-foreground">
            95% CI: {Math.round(confidence[0])} - {Math.round(confidence[1])} mg/dL
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
