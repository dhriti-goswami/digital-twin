'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import { X, Utensils, Check } from 'lucide-react';

interface LogMealModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: { carbs: number; mealType: string; description?: string }) => void;
}

const MEAL_TYPES = [
  { value: 'breakfast', label: 'Breakfast' },
  { value: 'lunch', label: 'Lunch' },
  { value: 'dinner', label: 'Dinner' },
  { value: 'snack', label: 'Snack' },
];

const QUICK_CARBS = [15, 30, 45, 60, 75, 90];

export function LogMealModal({ isOpen, onClose, onSubmit }: LogMealModalProps) {
  const [carbs, setCarbs] = React.useState<number>(30);
  const [mealType, setMealType] = React.useState<string>('snack');
  const [description, setDescription] = React.useState<string>('');
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [success, setSuccess] = React.useState(false);

  const handleSubmit = async () => {
    if (carbs <= 0) return;

    setIsSubmitting(true);
    await onSubmit({ carbs, mealType, description: description || undefined });
    setIsSubmitting(false);
    setSuccess(true);

    setTimeout(() => {
      setSuccess(false);
      setCarbs(30);
      setDescription('');
      onClose();
    }, 1500);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-40 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="w-full max-w-md bg-card border border-border rounded-2xl p-6 shadow-xl z-50"
          onClick={(e) => e.stopPropagation()}
        >
          {success ? (
            <div className="flex flex-col items-center py-8">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-16 h-16 rounded-full bg-success flex items-center justify-center mb-4"
              >
                <Check className="h-8 w-8 text-success-foreground" />
              </motion.div>
              <p className="text-lg font-medium">Meal Logged</p>
              <p className="text-muted-foreground">{carbs}g carbs recorded</p>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-accent flex items-center justify-center">
                    <Utensils className="h-5 w-5" />
                  </div>
                  <h2 className="text-xl font-semibold">Log Meal</h2>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-accent rounded-lg transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="space-y-6">
                {/* Carbs Input */}
                <div>
                  <label className="text-sm font-medium mb-2 block">Carbohydrates (g)</label>
                  <div className="flex items-center justify-center gap-4">
                    <button
                      onClick={() => setCarbs(Math.max(0, carbs - 5))}
                      className="w-14 h-14 rounded-xl bg-accent hover:bg-accent/80 flex items-center justify-center text-2xl font-bold flex-shrink-0"
                    >
                      -
                    </button>
                    <input
                      type="number"
                      value={carbs}
                      onChange={(e) => setCarbs(Math.max(0, parseInt(e.target.value) || 0))}
                      className={cn(
                        'w-24 h-16 text-center text-3xl font-bold bg-transparent',
                        'border-b-2 border-border focus:border-foreground',
                        'outline-none transition-colors',
                        '[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none'
                      )}
                    />
                    <button
                      onClick={() => setCarbs(carbs + 5)}
                      className="w-14 h-14 rounded-xl bg-accent hover:bg-accent/80 flex items-center justify-center text-2xl font-bold flex-shrink-0"
                    >
                      +
                    </button>
                  </div>
                </div>

                {/* Quick Carb Buttons */}
                <div>
                  <label className="text-sm font-medium mb-2 block">Quick Select</label>
                  <div className="grid grid-cols-6 gap-2">
                    {QUICK_CARBS.map((value) => (
                      <button
                        key={value}
                        onClick={() => setCarbs(value)}
                        className={cn(
                          'h-10 rounded-lg text-sm font-medium transition-colors',
                          carbs === value
                            ? 'bg-foreground text-background'
                            : 'bg-accent hover:bg-accent/80'
                        )}
                      >
                        {value}g
                      </button>
                    ))}
                  </div>
                </div>

                {/* Meal Type */}
                <div>
                  <label className="text-sm font-medium mb-2 block">Meal Type</label>
                  <div className="grid grid-cols-2 gap-2">
                    {MEAL_TYPES.map(({ value, label }) => (
                      <Button
                        key={value}
                        variant={mealType === value ? 'default' : 'outline'}
                        onClick={() => setMealType(value)}
                        className="h-11"
                      >
                        {label}
                      </Button>
                    ))}
                  </div>
                </div>

                {/* Description */}
                <div>
                  <label className="text-sm font-medium mb-2 block">Description (optional)</label>
                  <input
                    type="text"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="e.g., Pasta with vegetables"
                    className={cn(
                      'w-full h-12 px-4 rounded-xl bg-accent border border-border',
                      'focus:outline-none focus:ring-2 focus:ring-ring',
                      'placeholder:text-muted-foreground'
                    )}
                  />
                </div>

                {/* Submit */}
                <Button
                  onClick={handleSubmit}
                  disabled={carbs <= 0 || isSubmitting}
                  isLoading={isSubmitting}
                  className="w-full h-12"
                >
                  Log {carbs}g Carbs
                </Button>
              </div>
            </>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
