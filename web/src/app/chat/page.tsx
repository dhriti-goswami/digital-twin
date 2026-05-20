'use client';

import * as React from 'react';
import { useRouter } from 'next/navigation';
import { Navbar } from '@/components/layout/Navbar';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { Card } from '@/components/ui/Card';
import { GlucoseDisplay } from '@/components/ui/GlucoseDisplay';
import { PredictionTimeline } from '@/components/predictions/PredictionTimeline';
import { FadeIn } from '@/components/animations/PageTransition';
import { usePatientStore } from '@/stores/patient';

export default function ChatPage() {
  const router = useRouter();
  const { isOnboarded, currentGlucose, predictions } = usePatientStore();

  React.useEffect(() => {
    if (!isOnboarded) {
      router.push('/onboarding');
    }
  }, [isOnboarded, router]);

  if (!isOnboarded) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navbar />

      <main className="flex-1 pt-16 flex">
        {/* Chat Section */}
        <div className="flex-1 flex flex-col">
          <ChatInterface className="flex-1" />
        </div>

        {/* Sidebar - Context Panel */}
        <aside className="hidden lg:block w-80 border-l border-border p-4 pt-20">
          <FadeIn>
            <div className="space-y-6">
              {/* Current Glucose */}
              {currentGlucose && (
                <Card className="p-4">
                  <h3 className="text-sm font-medium text-muted-foreground mb-3">
                    Current Reading
                  </h3>
                  <GlucoseDisplay
                    value={currentGlucose}
                    size="md"
                    showStatus
                  />
                </Card>
              )}

              {/* Predictions Summary */}
              {predictions && (
                <Card className="p-4">
                  <h3 className="text-sm font-medium text-muted-foreground mb-3">
                    Predictions
                  </h3>
                  <div className="space-y-2">
                    {Object.entries(predictions.predictions).map(([horizon, value]) => (
                      <div key={horizon} className="flex justify-between text-sm">
                        <span className="text-muted-foreground">+{horizon.replace('min', ' min')}</span>
                        <span className="font-medium">{Math.round(value)} mg/dL</span>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* Suggestions */}
              <Card className="p-4">
                <h3 className="text-sm font-medium text-muted-foreground mb-3">
                  Try asking
                </h3>
                <div className="space-y-2">
                  {[
                    'What should I eat before exercise?',
                    'Why is my glucose high in the morning?',
                    'How much insulin for 40g carbs?',
                    'Tips for managing stress and glucose',
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      className="w-full text-left text-sm p-2 rounded-lg hover:bg-accent transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </Card>
            </div>
          </FadeIn>
        </aside>
      </main>
    </div>
  );
}
