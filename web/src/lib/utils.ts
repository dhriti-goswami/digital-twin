import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { GlucoseStatus } from './types';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getGlucoseStatus(glucose: number): GlucoseStatus {
  if (glucose < 54) return 'critical-low';
  if (glucose < 70) return 'low';
  if (glucose <= 180) return 'normal';
  if (glucose <= 250) return 'high';
  return 'critical-high';
}

export function getGlucoseColor(glucose: number): string {
  const status = getGlucoseStatus(glucose);
  switch (status) {
    case 'critical-low':
      return '#dc2626';
    case 'low':
      return '#ef4444';
    case 'normal':
      return '#22c55e';
    case 'high':
      return '#f59e0b';
    case 'critical-high':
      return '#dc2626';
  }
}

export function getGlucoseStatusColor(status: GlucoseStatus): string {
  switch (status) {
    case 'critical-low':
      return 'text-red-600';
    case 'low':
      return 'text-red-500';
    case 'normal':
      return 'text-green-500';
    case 'high':
      return 'text-amber-500';
    case 'critical-high':
      return 'text-red-600';
  }
}

export function getGlucoseLabel(glucose: number): string {
  const status = getGlucoseStatus(glucose);
  switch (status) {
    case 'critical-low':
      return 'Critical Low';
    case 'low':
      return 'Low';
    case 'normal':
      return 'In Range';
    case 'high':
      return 'High';
    case 'critical-high':
      return 'Critical High';
  }
}

export function formatGlucose(glucose: number): string {
  return `${Math.round(glucose)} mg/dL`;
}

export function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

export function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

export function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
