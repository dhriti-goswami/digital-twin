'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
} from 'recharts';
import { cn } from '@/lib/utils';
import type { PredictionResponse } from '@/lib/types';

interface PredictionChartProps {
  currentGlucose: number;
  predictions: PredictionResponse | null;
  className?: string;
  animated?: boolean;
}

const TARGET_LOW = 70;
const TARGET_HIGH = 180;

export function PredictionChart({
  currentGlucose,
  predictions,
  className,
  animated = true,
}: PredictionChartProps) {
  const chartData = React.useMemo(() => {
    const data = [
      {
        time: 'Now',
        minutes: 0,
        glucose: currentGlucose,
        predicted: currentGlucose,
        lower: currentGlucose,
        upper: currentGlucose,
        isCurrent: true,
      },
    ];

    if (predictions) {
      const horizons = ['30min', '60min', '90min', '120min'] as const;
      const times = ['30m', '60m', '90m', '120m'];

      horizons.forEach((horizon, index) => {
        const value = predictions.predictions[horizon];
        const ci = predictions.confidence_intervals[horizon];
        data.push({
          time: times[index],
          minutes: (index + 1) * 30,
          glucose: value,
          predicted: value,
          lower: ci[0],
          upper: ci[1],
          isCurrent: false,
        });
      });
    }

    return data;
  }, [currentGlucose, predictions]);

  if (!predictions) {
    return (
      <div className={cn('flex items-center justify-center h-64 bg-card rounded-lg border border-border', className)}>
        <p className="text-muted-foreground">Enter glucose to see predictions</p>
      </div>
    );
  }

  const allValues = chartData.flatMap(d => [d.glucose, d.lower, d.upper]);
  const minValue = Math.min(...allValues, TARGET_LOW - 10);
  const maxValue = Math.max(...allValues, TARGET_HIGH + 10);

  const CustomTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ value: number; name: string }>;
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      const point = chartData.find(d => d.time === label);
      if (!point) return null;

      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm text-muted-foreground mb-1">
            {point.isCurrent ? 'Current Reading' : `+${label}`}
          </p>
          <p className="text-lg font-bold">{Math.round(point.glucose)} mg/dL</p>
          {!point.isCurrent && (
            <p className="text-xs text-muted-foreground">
              95% CI: {Math.round(point.lower)} - {Math.round(point.upper)}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={animated ? { opacity: 0, y: 20 } : undefined}
      animate={animated ? { opacity: 1, y: 0 } : undefined}
      transition={{ duration: 0.5, ease: 'easeOut', delay: 0.1 }}
      className={cn('bg-card rounded-lg border border-border p-4', className)}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-muted-foreground">Glucose Prediction</h3>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-foreground" />
            Predicted
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-foreground/20" />
            95% CI
          </span>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--foreground))" stopOpacity={0.2} />
                <stop offset="95%" stopColor="hsl(var(--foreground))" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <ReferenceLine
              y={TARGET_LOW}
              stroke="hsl(var(--destructive))"
              strokeDasharray="3 3"
              strokeOpacity={0.5}
            />
            <ReferenceLine
              y={TARGET_HIGH}
              stroke="hsl(var(--warning))"
              strokeDasharray="3 3"
              strokeOpacity={0.5}
            />

            <XAxis
              dataKey="time"
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[minValue, maxValue]}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Confidence interval area */}
            <Area
              type="monotone"
              dataKey="upper"
              stroke="none"
              fill="url(#confidenceGradient)"
              isAnimationActive={animated}
              animationDuration={1200}
            />
            <Area
              type="monotone"
              dataKey="lower"
              stroke="none"
              fill="hsl(var(--background))"
              isAnimationActive={animated}
              animationDuration={1200}
            />

            {/* Main prediction line */}
            <Line
              type="monotone"
              dataKey="glucose"
              stroke="hsl(var(--foreground))"
              strokeWidth={2}
              strokeDasharray="0"
              dot={(props) => {
                const { cx, cy, index } = props;
                return (
                  <circle
                    key={index}
                    cx={cx}
                    cy={cy}
                    r={index === 0 ? 6 : 4}
                    fill={index === 0 ? 'hsl(var(--foreground))' : 'hsl(var(--background))'}
                    stroke="hsl(var(--foreground))"
                    strokeWidth={2}
                  />
                );
              }}
              activeDot={{
                r: 8,
                fill: 'hsl(var(--foreground))',
                stroke: 'hsl(var(--background))',
                strokeWidth: 2,
              }}
              isAnimationActive={animated}
              animationDuration={1500}
              animationEasing="ease-out"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
