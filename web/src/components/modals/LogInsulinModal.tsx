'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { X, Syringe, Check } from 'lucide-react';

interface LogInsulinModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: { units: number; type: string; isMealBolus: boolean }) => void;
}

export function LogInsulinModal({ isOpen, onClose, onSubmit }: LogInsulinModalProps) {
  const [units, setUnits] = React.useState<number>(0);
  const [insulinType, setInsulinType] = React.useState<'rapid' | 'long'>('rapid');
  const [isMealBolus, setIsMealBolus] = React.useState(false);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [success, setSuccess] = React.useState(false);

  const handleSubmit = async () => {
    if (units <= 0) return;

    setIsSubmitting(true);
    await onSubmit({ units, type: insulinType, isMealBolus });
    setIsSubmitting(false);
    setSuccess(true);

    setTimeout(() => {
      setSuccess(false);
      setUnits(0);
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
              <p className="text-lg font-medium">Insulin Logged</p>
              <p className="text-muted-foreground">{units} units recorded</p>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-accent flex items-center justify-center">
                    <Syringe className="h-5 w-5" />
                  </div>
                  <h2 className="text-xl font-semibold">Log Insulin</h2>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-accent rounded-lg transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="space-y-6">
                {/* Units Input */}
                <div>
                  <label className="text-sm font-medium mb-2 block">Units</label>
                  <div className="flex items-center justify-center gap-4">
                    <button
                      onClick={() => setUnits(Math.max(0, units - 0.5))}
                      className="w-14 h-14 rounded-xl bg-accent hover:bg-accent/80 flex items-center justify-center text-2xl font-bold flex-shrink-0"
                    >
                      -
                    </button>
                    <input
                      type="number"
                      value={units}
                      onChange={(e) => setUnits(Math.max(0, parseFloat(e.target.value) || 0))}
                      step={0.5}
                      className={cn(
                        'w-24 h-16 text-center text-3xl font-bold bg-transparent',
                        'border-b-2 border-border focus:border-foreground',
                        'outline-none transition-colors',
                        '[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none'
                      )}
                    />
                    <button
                      onClick={() => setUnits(units + 0.5)}
                      className="w-14 h-14 rounded-xl bg-accent hover:bg-accent/80 flex items-center justify-center text-2xl font-bold flex-shrink-0"
                    >
                      +
                    </button>
                  </div>
                </div>

                {/* Insulin Type */}
                <div>
                  <label className="text-sm font-medium mb-2 block">Insulin Type</label>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant={insulinType === 'rapid' ? 'default' : 'outline'}
                      onClick={() => setInsulinType('rapid')}
                      className="h-12"
                    >
                      Rapid Acting
                    </Button>
                    <Button
                      variant={insulinType === 'long' ? 'default' : 'outline'}
                      onClick={() => setInsulinType('long')}
                      className="h-12"
                    >
                      Long Acting
                    </Button>
                  </div>
                </div>

                {/* Meal Bolus Toggle */}
                {insulinType === 'rapid' && (
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={isMealBolus}
                      onChange={(e) => setIsMealBolus(e.target.checked)}
                      className="w-5 h-5 rounded border-border"
                    />
                    <span className="text-sm">This is a meal bolus</span>
                  </label>
                )}

                {/* Submit */}
                <Button
                  onClick={handleSubmit}
                  disabled={units <= 0 || isSubmitting}
                  isLoading={isSubmitting}
                  className="w-full h-12"
                >
                  Log {units} Units
                </Button>
              </div>
            </>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
