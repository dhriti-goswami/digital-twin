'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import { cn, getGlucoseStatus, getGlucoseStatusColor } from '@/lib/utils';
import { Card } from '@/components/ui/Card';
import { Clock, ArrowRight } from 'lucide-react';
import type { PredictionResponse } from '@/lib/types';

interface PredictionTimelineProps {
  predictions: PredictionResponse | null;
  className?: string;
}

export function PredictionTimeline({ predictions, className }: PredictionTimelineProps) {
  if (!predictions) {
    return (
      <Card className={cn('p-6', className)}>
        <div className="text-center text-muted-foreground">
          <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p>Enter glucose to see predictions</p>
        </div>
      </Card>
    );
  }

  const horizons = [
    { key: '30min', label: '+30 min' },
    { key: '60min', label: '+60 min' },
    { key: '90min', label: '+90 min' },
    { key: '120min', label: '+120 min' },
  ] as const;

  return (
    <Card className={cn('p-4 sm:p-6', className)}>
      <div className="flex items-center gap-2 mb-4 sm:mb-6">
        <Clock className="h-5 w-5" />
        <h3 className="font-semibold">Glucose Predictions</h3>
      </div>

      <div className="flex items-center justify-between gap-1 sm:gap-2 overflow-x-auto pb-2">
        {horizons.map(({ key, label }, index) => {
          const value = Math.round(predictions.predictions[key]);
          const status = getGlucoseStatus(value);
          const color = getGlucoseStatusColor(status);
          const confidence = predictions.confidence_intervals[key];

          return (
            <React.Fragment key={key}>
              <motion.div
                className="flex flex-col items-center gap-1 sm:gap-2 min-w-fit"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <span className="text-[10px] sm:text-xs text-muted-foreground whitespace-nowrap">{label}</span>
                <div
                  className={cn(
                    'w-12 h-12 sm:w-16 sm:h-16 rounded-full flex items-center justify-center',
                    'border-2 font-bold text-sm sm:text-lg',
                    status === 'normal' && 'border-success bg-success/10',
                    status === 'low' && 'border-warning bg-warning/10',
                    status === 'high' && 'border-warning bg-warning/10',
                    (status === 'critical-low' || status === 'critical-high') &&
                      'border-destructive bg-destructive/10'
                  )}
                >
                  <span className={color}>{value}</span>
                </div>
                {confidence && (
                  <span className="text-[9px] sm:text-xs text-muted-foreground whitespace-nowrap">
                    {Math.round(confidence[0])}-{Math.round(confidence[1])}
                  </span>
                )}
              </motion.div>

              {index < horizons.length - 1 && (
                <ArrowRight className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground flex-shrink-0 hidden xs:block sm:block" />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </Card>
  );
}
