'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
} from 'recharts';
import { cn } from '@/lib/utils';

interface DataPoint {
  time: string;
  glucose: number;
  timestamp?: Date;
}

interface GlucoseTrendChartProps {
  data: DataPoint[];
  className?: string;
  showTargetRange?: boolean;
  animated?: boolean;
}

const TARGET_LOW = 70;
const TARGET_HIGH = 180;

export function GlucoseTrendChart({
  data,
  className,
  showTargetRange = true,
  animated = true,
}: GlucoseTrendChartProps) {
  const [isVisible, setIsVisible] = React.useState(!animated);

  React.useEffect(() => {
    if (animated) {
      const timer = setTimeout(() => setIsVisible(true), 100);
      return () => clearTimeout(timer);
    }
  }, [animated]);

  if (data.length === 0) {
    return (
      <div className={cn('flex items-center justify-center h-64 bg-card rounded-lg border border-border', className)}>
        <p className="text-muted-foreground">No glucose data available</p>
      </div>
    );
  }

  const minGlucose = Math.min(...data.map(d => d.glucose), TARGET_LOW - 10);
  const maxGlucose = Math.max(...data.map(d => d.glucose), TARGET_HIGH + 10);

  const CustomTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ value: number }>;
    label?: string;
  }) => {
    if (active && payload && payload.length) {
      const value = payload[0].value;
      const status = value < TARGET_LOW ? 'Low' : value > TARGET_HIGH ? 'High' : 'In Range';
      const statusColor = value < TARGET_LOW ? 'text-destructive' : value > TARGET_HIGH ? 'text-warning' : 'text-success';

      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-lg font-bold">{Math.round(value)} mg/dL</p>
          <p className={cn('text-xs font-medium', statusColor)}>{status}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={animated ? { opacity: 0, y: 20 } : undefined}
      animate={animated ? { opacity: 1, y: 0 } : undefined}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className={cn('bg-card rounded-lg border border-border p-4', className)}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-muted-foreground">Glucose Trend</h3>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-success/30" />
            Target Range
          </span>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            {showTargetRange && (
              <ReferenceArea
                y1={TARGET_LOW}
                y2={TARGET_HIGH}
                fill="hsl(var(--success))"
                fillOpacity={0.1}
              />
            )}
            <ReferenceLine
              y={TARGET_LOW}
              stroke="hsl(var(--success))"
              strokeDasharray="3 3"
              strokeOpacity={0.5}
            />
            <ReferenceLine
              y={TARGET_HIGH}
              stroke="hsl(var(--success))"
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
              domain={[minGlucose, maxGlucose]}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="glucose"
              stroke="hsl(var(--foreground))"
              strokeWidth={2}
              dot={false}
              activeDot={{
                r: 6,
                fill: 'hsl(var(--foreground))',
                stroke: 'hsl(var(--background))',
                strokeWidth: 2,
              }}
              isAnimationActive={animated}
              animationDuration={1500}
              animationEasing="ease-out"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
