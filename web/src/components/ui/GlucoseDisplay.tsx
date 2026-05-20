'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import { cn, getGlucoseStatus, getGlucoseStatusColor } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, Activity } from 'lucide-react';

interface GlucoseDisplayProps {
  value: number;
  trend?: 'rising' | 'falling' | 'stable';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showUnit?: boolean;
  showStatus?: boolean;
  className?: string;
}

export function GlucoseDisplay({
  value,
  trend,
  size = 'md',
  showUnit = true,
  showStatus = true,
  className,
}: GlucoseDisplayProps) {
  const status = getGlucoseStatus(value);
  const statusColor = getGlucoseStatusColor(status);

  const sizes = {
    sm: 'text-2xl',
    md: 'text-4xl',
    lg: 'text-6xl',
    xl: 'text-8xl',
  };

  const TrendIcon = trend === 'rising'
    ? TrendingUp
    : trend === 'falling'
      ? TrendingDown
      : Minus;

  const isCritical = status === 'critical-low' || status === 'critical-high';

  return (
    <motion.div
      className={cn('flex flex-col items-center gap-2', className)}
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 200, damping: 20 }}
    >
      <div className="flex items-baseline gap-2">
        {isCritical && (
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            <AlertTriangle className={cn('h-8 w-8', statusColor)} />
          </motion.div>
        )}
        <motion.span
          className={cn(sizes[size], 'font-bold tabular-nums', statusColor)}
          key={value}
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 25 }}
        >
          {value}
        </motion.span>
        {showUnit && (
          <span className="text-muted-foreground text-lg">mg/dL</span>
        )}
        {trend && (
          <TrendIcon className={cn('h-6 w-6', statusColor)} />
        )}
      </div>

      {showStatus && (
        <motion.div
          className={cn(
            'flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium',
            status === 'normal' && 'bg-success/10 text-success',
            status === 'low' && 'bg-warning/10 text-warning',
            status === 'high' && 'bg-warning/10 text-warning',
            (status === 'critical-low' || status === 'critical-high') && 'bg-destructive/10 text-destructive'
          )}
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          <Activity className="h-4 w-4" />
          {status === 'normal' && 'In Range'}
          {status === 'low' && 'Low'}
          {status === 'high' && 'High'}
          {status === 'critical-low' && 'Critically Low'}
          {status === 'critical-high' && 'Critically High'}
        </motion.div>
      )}
    </motion.div>
  );
}
