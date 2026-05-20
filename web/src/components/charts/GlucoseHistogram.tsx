'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { cn } from '@/lib/utils';

interface GlucoseHistogramProps {
  readings: number[];
  className?: string;
  animated?: boolean;
}

const BUCKETS = [
  { label: '<54', min: 0, max: 54, color: 'hsl(var(--destructive))' },
  { label: '54-70', min: 54, max: 70, color: 'hsl(var(--destructive))', opacity: 0.7 },
  { label: '70-180', min: 70, max: 180, color: 'hsl(var(--success))' },
  { label: '180-250', min: 180, max: 250, color: 'hsl(var(--warning))', opacity: 0.7 },
  { label: '>250', min: 250, max: Infinity, color: 'hsl(var(--warning))' },
];

export function GlucoseHistogram({
  readings,
  className,
  animated = true,
}: GlucoseHistogramProps) {
  const data = React.useMemo(() => {
    return BUCKETS.map(bucket => {
      const count = readings.filter(r => r >= bucket.min && r < bucket.max).length;
      const percentage = readings.length > 0 ? (count / readings.length) * 100 : 0;
      return {
        ...bucket,
        count,
        percentage,
      };
    });
  }, [readings]);

  if (readings.length === 0) {
    return (
      <div className={cn('flex items-center justify-center h-48 bg-card rounded-lg border border-border', className)}>
        <p className="text-muted-foreground">No readings to display</p>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload }: {
    active?: boolean;
    payload?: Array<{ payload: { label: string; count: number; percentage: number } }>;
  }) => {
    if (active && payload && payload.length) {
      const { label, count, percentage } = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-2 shadow-lg text-sm">
          <p className="font-medium">{label} mg/dL</p>
          <p className="text-muted-foreground">
            {count} readings ({percentage.toFixed(1)}%)
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={animated ? { opacity: 0, y: 20 } : undefined}
      animate={animated ? { opacity: 1, y: 0 } : undefined}
      transition={{ duration: 0.5, ease: 'easeOut', delay: 0.3 }}
      className={cn('bg-card rounded-lg border border-border p-4', className)}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-muted-foreground">Glucose Distribution</h3>
        <span className="text-xs text-muted-foreground">{readings.length} readings</span>
      </div>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <XAxis
              dataKey="label"
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              dataKey="percentage"
              radius={[4, 4, 0, 0]}
              isAnimationActive={animated}
              animationDuration={1000}
              animationEasing="ease-out"
            >
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.color}
                  fillOpacity={entry.opacity || 1}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Stats row */}
      <div className="flex justify-between mt-4 pt-4 border-t border-border text-xs">
        <div>
          <span className="text-muted-foreground">Avg: </span>
          <span className="font-medium">
            {Math.round(readings.reduce((a, b) => a + b, 0) / readings.length)} mg/dL
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">Min: </span>
          <span className="font-medium">{Math.min(...readings)} mg/dL</span>
        </div>
        <div>
          <span className="text-muted-foreground">Max: </span>
          <span className="font-medium">{Math.max(...readings)} mg/dL</span>
        </div>
      </div>
    </motion.div>
  );
}
