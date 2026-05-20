'use client';

import * as React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, useInView } from 'framer-motion';
import { Button } from '@/components/ui/Button';
import { usePatientStore } from '@/stores/patient';
import {
  Activity,
  ArrowRight,
  Brain,
  LineChart,
  MessageSquare,
  Shield,
  Sparkles,
} from 'lucide-react';

// Animated counter component
function AnimatedNumber({ value, suffix = '' }: { value: number; suffix?: string }) {
  const ref = React.useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });
  const [displayValue, setDisplayValue] = React.useState(0);

  React.useEffect(() => {
    if (isInView) {
      const duration = 2000;
      const startTime = Date.now();
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        setDisplayValue(Math.round(value * eased * 100) / 100);
        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };
      requestAnimationFrame(animate);
    }
  }, [isInView, value]);

  return (
    <span ref={ref}>
      {typeof value === 'number' && value % 1 !== 0
        ? displayValue.toFixed(2)
        : Math.round(displayValue)}
      {suffix}
    </span>
  );
}

export default function HomePage() {
  const router = useRouter();
  const { isOnboarded } = usePatientStore();

  const handleGetStarted = () => {
    if (isOnboarded) {
      router.push('/dashboard');
    } else {
      router.push('/onboarding');
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 border-b border-border bg-background/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-foreground flex items-center justify-center">
                <Activity className="h-5 w-5 text-background" />
              </div>
              <span className="font-semibold text-lg">Digital Twin</span>
            </div>

            <Button onClick={handleGetStarted}>
              {isOnboarded ? 'Go to Dashboard' : 'Get Started'}
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="pt-32 pb-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-6">
              Your Personal
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-foreground to-muted-foreground">
                Diabetes Digital Twin
              </span>
            </h1>

            <p className="text-lg sm:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              AI-powered glucose prediction and personalized diabetes management.
              Get real-time insights, predictions, and guidance from your virtual health assistant.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" onClick={handleGetStarted}>
                {isOnboarded ? 'Open Dashboard' : 'Create Your Twin'}
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Link
                href="#features"
                className="inline-flex items-center justify-center rounded-lg font-medium h-12 px-6 text-lg border border-border bg-transparent hover:bg-accent hover:text-accent-foreground transition-colors"
              >
                Learn More
              </Link>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            className="mt-16 grid grid-cols-3 gap-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <div className="text-3xl sm:text-4xl font-bold">
                <AnimatedNumber value={5.55} />
              </div>
              <div className="text-sm text-muted-foreground">mg/dL MAE</div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <div className="text-3xl sm:text-4xl font-bold">
                <AnimatedNumber value={120} />
              </div>
              <div className="text-sm text-muted-foreground">min forecast</div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <div className="text-3xl sm:text-4xl font-bold">24/7</div>
              <div className="text-sm text-muted-foreground">AI Assistant</div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-20 px-4 border-t border-border">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Intelligent Features</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Powered by advanced machine learning and physics-informed neural networks
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard
              icon={LineChart}
              title="Glucose Predictions"
              description="Multi-horizon forecasting at 30, 60, 90, and 120 minutes with confidence intervals"
              index={0}
            />
            <FeatureCard
              icon={MessageSquare}
              title="AI Health Assistant"
              description="Context-aware conversations powered by advanced LLMs with medical knowledge"
              index={1}
            />
            <FeatureCard
              icon={Brain}
              title="Physics-Informed AI"
              description="Predictions grounded in physiological models for clinical accuracy"
              index={2}
            />
            <FeatureCard
              icon={Sparkles}
              title="What-If Simulation"
              description="Explore how meals, insulin, and exercise affect your glucose"
              index={3}
            />
            <FeatureCard
              icon={Shield}
              title="Privacy First"
              description="Your data stays on your device. No cloud storage required"
              index={4}
            />
            <FeatureCard
              icon={Activity}
              title="Real-time Analysis"
              description="Instant insights and actionable recommendations"
              index={5}
            />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 border-t border-border">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to get started?</h2>
          <p className="text-muted-foreground mb-8">
            Create your digital twin in under a minute. No account required.
          </p>
          <Button size="lg" onClick={handleGetStarted}>
            {isOnboarded ? 'Open Dashboard' : 'Get Started Free'}
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-border">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            <span className="font-medium">Digital Twin</span>
          </div>
          <p className="text-sm text-muted-foreground">
            Research prototype. Not for medical decisions.
          </p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  description,
  index = 0,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  index?: number;
}) {
  return (
    <motion.div
      className="p-6 rounded-xl border border-border bg-card group cursor-default"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
    >
      <motion.div
        className="w-12 h-12 rounded-lg bg-accent flex items-center justify-center mb-4"
        whileHover={{ scale: 1.1, rotate: 5 }}
        transition={{ type: 'spring', stiffness: 300 }}
      >
        <Icon className="h-6 w-6" />
      </motion.div>
      <h3 className="font-semibold mb-2 group-hover:text-foreground transition-colors">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </motion.div>
  );
}
