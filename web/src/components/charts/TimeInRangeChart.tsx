'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { cn } from '@/lib/utils';

interface TimeInRangeChartProps {
  inRange: number;
  belowRange: number;
  aboveRange: number;
  className?: string;
  animated?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const COLORS = {
  inRange: 'hsl(var(--success))',
  belowRange: 'hsl(var(--destructive))',
  aboveRange: 'hsl(var(--warning))',
};

export function TimeInRangeChart({
  inRange,
  belowRange,
  aboveRange,
  className,
  animated = true,
  size = 'md',
}: TimeInRangeChartProps) {
  const total = inRange + belowRange + aboveRange;

  const data = [
    { name: 'In Range (70-180)', value: inRange, color: COLORS.inRange, percentage: total > 0 ? (inRange / total) * 100 : 0 },
    { name: 'Below Range (<70)', value: belowRange, color: COLORS.belowRange, percentage: total > 0 ? (belowRange / total) * 100 : 0 },
    { name: 'Above Range (>180)', value: aboveRange, color: COLORS.aboveRange, percentage: total > 0 ? (aboveRange / total) * 100 : 0 },
  ].filter(d => d.value > 0);

  const sizeConfig = {
    sm: { height: 150, innerRadius: 35, outerRadius: 55 },
    md: { height: 200, innerRadius: 50, outerRadius: 75 },
    lg: { height: 250, innerRadius: 65, outerRadius: 95 },
  };

  const { height, innerRadius, outerRadius } = sizeConfig[size];

  const CustomTooltip = ({ active, payload }: {
    active?: boolean;
    payload?: Array<{ payload: { name: string; value: number; percentage: number } }>;
  }) => {
    if (active && payload && payload.length) {
      const { name, value, percentage } = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-2 shadow-lg text-sm">
          <p className="font-medium">{name}</p>
          <p className="text-muted-foreground">{value} readings ({percentage.toFixed(1)}%)</p>
        </div>
      );
    }
    return null;
  };

  const inRangePercentage = total > 0 ? (inRange / total) * 100 : 0;

  return (
    <motion.div
      initial={animated ? { opacity: 0, scale: 0.9 } : undefined}
      animate={animated ? { opacity: 1, scale: 1 } : undefined}
      transition={{ duration: 0.5, ease: 'easeOut', delay: 0.2 }}
      className={cn('bg-card rounded-lg border border-border p-4', className)}
    >
      <h3 className="text-sm font-medium text-muted-foreground mb-2">Time in Range</h3>

      <div className="relative" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={innerRadius}
              outerRadius={outerRadius}
              paddingAngle={2}
              dataKey="value"
              isAnimationActive={animated}
              animationDuration={1000}
              animationEasing="ease-out"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>

        {/* Center label */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <motion.span
              initial={animated ? { opacity: 0 } : undefined}
              animate={animated ? { opacity: 1 } : undefined}
              transition={{ delay: 0.8, duration: 0.3 }}
              className="text-3xl font-bold"
            >
              {inRangePercentage.toFixed(0)}%
            </motion.span>
            <p className="text-xs text-muted-foreground">in range</p>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-4 mt-4 text-xs">
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-success" />
          <span>In Range</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-destructive" />
          <span>Low</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-warning" />
          <span>High</span>
        </div>
      </div>
    </motion.div>
  );
}
