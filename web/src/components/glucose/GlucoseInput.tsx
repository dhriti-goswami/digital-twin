'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import { Activity, Plus, Minus } from 'lucide-react';

interface GlucoseInputProps {
  value: number;
  onChange: (value: number) => void;
  onSubmit: () => void;
  isLoading?: boolean;
  className?: string;
}

export function GlucoseInput({
  value,
  onChange,
  onSubmit,
  isLoading,
  className,
}: GlucoseInputProps) {
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleIncrement = () => {
    onChange(Math.min(400, value + 5));
  };

  const handleDecrement = () => {
    onChange(Math.max(20, value - 5));
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value) || 0;
    onChange(Math.max(0, Math.min(600, newValue)));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      onSubmit();
    }
  };

  return (
    <motion.div
      className={cn('flex flex-col items-center gap-6', className)}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="text-center">
        <h2 className="text-lg font-medium text-muted-foreground mb-2">
          Current Glucose
        </h2>
        <div className="flex items-center gap-4">
          <motion.button
            onClick={handleDecrement}
            className="w-12 h-12 rounded-full border border-border flex items-center justify-center hover:bg-accent transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Minus className="h-5 w-5" />
          </motion.button>

          <div className="relative">
            <input
              ref={inputRef}
              type="number"
              value={value || ''}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder="120"
              className={cn(
                'w-40 h-24 text-center text-6xl font-bold bg-transparent',
                'border-b-4 border-border focus:border-foreground',
                'outline-none transition-colors tabular-nums',
                '[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none'
              )}
            />
            <span className="absolute -right-12 bottom-4 text-lg text-muted-foreground">
              mg/dL
            </span>
          </div>

          <motion.button
            onClick={handleIncrement}
            className="w-12 h-12 rounded-full border border-border flex items-center justify-center hover:bg-accent transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Plus className="h-5 w-5" />
          </motion.button>
        </div>
      </div>

      <Button
        onClick={onSubmit}
        isLoading={isLoading}
        size="lg"
        className="min-w-[200px]"
      >
        <Activity className="mr-2 h-5 w-5" />
        Get Predictions
      </Button>

      <div className="flex gap-2 text-sm text-muted-foreground">
        <span className="px-2 py-1 rounded bg-success/10 text-success">70-180 Normal</span>
        <span className="px-2 py-1 rounded bg-warning/10 text-warning">&lt;70 Low</span>
        <span className="px-2 py-1 rounded bg-warning/10 text-warning">&gt;180 High</span>
      </div>
    </motion.div>
  );
}
