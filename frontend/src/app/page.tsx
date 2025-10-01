// /home/ghost/fake-news-game-theory/frontend/src/app/page.tsx
'use client';

import { useRef, useState, useEffect } from 'react';
import { motion, useInView, useScroll, useTransform } from 'framer-motion';
import Link from 'next/link';
import {
  Target, Brain, Network, TrendingUp, Shield, BarChart3,
  ArrowRight, Sparkles, Zap, Users, Database,
  GitBranch, Layers, ChevronRight, Play
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import CountUp from '@/components/countup-wrapper';

export default function HomePage() {
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.2], [1, 0.95]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-gray-50 to-white overflow-hidden">
      {/* Hero Section */}
      <motion.section
        style={{ opacity, scale }}
        className="relative min-h-screen flex items-center justify-center overflow-hidden"
      >
        {/* Video Background */}
        <div className="absolute inset-0">
          {/* Video Element */}
          <video
            autoPlay
            loop
            muted
            playsInline
            className="absolute inset-0 w-full h-full object-cover"
          >
            <source src="/videos/hero.mp4" type="video/mp4" />
          </video>

          {/* Blur and Overlay */}
          <div className="absolute inset-0 backdrop-blur-md bg-white/30" />

          {/* Gradient Overlay for better text visibility */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-900/40 via-indigo-900/30 to-purple-900/40" />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-6 py-24 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-6 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/20 backdrop-blur-md text-white text-sm font-medium border border-white/30"
          >
            <Sparkles className="h-4 w-4" />
            <span>Advanced Research Platform</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="text-6xl md:text-8xl font-bold mb-8 text-white drop-shadow-2xl"
          >
            Combating
            <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400">
              Misinformation
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-xl md:text-2xl text-white mb-12 max-w-3xl mx-auto leading-relaxed drop-shadow-lg"
          >
            An innovative platform integrating <span className="font-semibold text-blue-300">Game Theory</span>,
            <span className="font-semibold text-indigo-300"> Machine Learning</span>, and
            <span className="font-semibold text-purple-300"> Network Analysis</span> to predict and combat fake news propagation.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16"
          >
            <Link href="/simulation">
              <Button
                size="lg"
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-10 py-7 text-lg group shadow-2xl hover:shadow-blue-500/50 transition-all duration-300"
              >
                <Play className="mr-2 h-5 w-5" />
                Start Exploring
                <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Link href="/research">
              <Button
                size="lg"
                variant="outline"
                className="border-2 border-white text-white hover:bg-white hover:text-gray-900 px-10 py-7 text-lg shadow-xl transition-all duration-300"
              >
                View Research
                <ChevronRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
          </motion.div>

          {/* Stats Section */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto"
          >
            <StatCard number={95} suffix="%" label="Accuracy" delay={0.6} />
            <StatCard number={1000000} suffix="+" label="Data Points" delay={0.7} />
            <StatCard number={50} suffix="+" label="Simulations" delay={0.8} />
            <StatCard number={99} suffix="%" label="Precision" delay={0.9} />
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1, y: [0, 10, 0] }}
          transition={{
            opacity: { delay: 1, duration: 0.5 },
            y: { repeat: Infinity, duration: 2 }
          }}
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
        >
          <div className="w-6 h-10 border-2 border-white/60 rounded-full flex justify-center">
            <motion.div className="w-1 h-2 bg-white rounded-full mt-2" />
          </div>
        </motion.div>
      </motion.section>

      {/* Core Pillars Section */}
      <section className="py-32 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <SectionHeader
            badge="Core Pillars"
            title="Three-Dimensional Approach"
            description="Our research integrates multiple disciplines to provide comprehensive insights into fake news dynamics."
          />

          <div className="grid md:grid-cols-3 gap-8 mt-16">
            {disciplines.map((discipline, index) => (
              <DisciplineCard key={index} {...discipline} delay={index * 0.1} />
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-32 px-6 bg-gradient-to-br from-gray-50 to-blue-50 relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-gray-900/[0.02] bg-[size:32px_32px]" />
        <div className="max-w-7xl mx-auto relative z-10">
          <SectionHeader
            badge="Methodology"
            title="How It Works"
            description="A seamless pipeline from data ingestion to actionable insights."
          />

          <div className="mt-16 space-y-6">
            {workflow.map((step, index) => (
              <WorkflowStep key={index} step={step} index={index} />
            ))}
          </div>
        </div>
      </section>

      {/* Research Objectives Section */}
      <section className="py-32 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <SectionHeader
            badge="Research Goals"
            title="Our Objectives"
            description="Driving innovation in understanding and combating misinformation."
          />

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mt-16">
            {objectives.map((objective, index) => (
              <ObjectiveCard key={index} {...objective} delay={index * 0.1} />
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 px-6 bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 relative overflow-hidden">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
          }}
          className="absolute top-0 right-0 w-96 h-96 bg-white rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
          }}
          className="absolute bottom-0 left-0 w-96 h-96 bg-pink-400 rounded-full blur-3xl"
        />

        <div className="max-w-4xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">
              Ready to Explore?
            </h2>
            <p className="text-xl text-blue-100 mb-12 max-w-2xl mx-auto">
              Dive into our interactive simulations and discover how game theory shapes the landscape of information warfare.
            </p>
            <Link href="/simulation">
              <Button
                size="lg"
                className="bg-white text-blue-600 hover:bg-gray-100 px-12 py-8 text-xl font-semibold shadow-2xl hover:scale-105 transition-all duration-300"
              >
                Launch Dashboard
                <Zap className="ml-3 h-6 w-6" />
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
}

// Components

interface StatCardProps {
  number: number;
  suffix: string;
  label: string;
  delay: number;
}

function StatCard({ number, suffix, label, delay }: StatCardProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay }}
      className="text-center"
    >
      <div className="text-4xl md:text-5xl font-bold text-white mb-2 drop-shadow-lg">
        {isInView && <CountUp end={number} duration={2} />}
        {suffix}
      </div>
      <div className="text-white font-medium drop-shadow-md">{label}</div>
    </motion.div>
  );
}

interface SectionHeaderProps {
  badge: string;
  title: string;
  description: string;
}

function SectionHeader({ badge, title, description }: SectionHeaderProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.8 }}
      className="text-center max-w-3xl mx-auto"
    >
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100 text-blue-700 text-sm font-medium mb-6">
        <Sparkles className="h-4 w-4" />
        <span>{badge}</span>
      </div>
      <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">{title}</h2>
      <p className="text-xl text-gray-600 leading-relaxed">{description}</p>
    </motion.div>
  );
}

interface DisciplineProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  delay: number;
}

function DisciplineCard({ icon, title, description, delay }: DisciplineProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 50, scale: 0.95 }}
      animate={isInView ? { opacity: 1, y: 0, scale: 1 } : {}}
      transition={{ duration: 0.6, delay }}
      whileHover={{ y: -10 }}
      className="group"
    >
      <Card className="p-8 h-full hover:shadow-2xl transition-all duration-300 border-2 border-gray-100 hover:border-blue-200 bg-white/80 backdrop-blur">
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          transition={{ type: "spring", stiffness: 300 }}
          className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mb-6 shadow-lg group-hover:shadow-blue-500/50"
        >
          <div className="text-white">{icon}</div>
        </motion.div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4 group-hover:text-blue-600 transition-colors">
          {title}
        </h3>
        <p className="text-gray-600 leading-relaxed">{description}</p>
      </Card>
    </motion.div>
  );
}

interface WorkflowStepProps {
  step: string;
  index: number;
}

function WorkflowStep({ step, index }: WorkflowStepProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, x: -50 }}
      animate={isInView ? { opacity: 1, x: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1 }}
      className="flex items-center gap-6 bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 group cursor-pointer hover:scale-[1.02]"
    >
      <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center text-white font-bold text-lg shadow-lg group-hover:shadow-blue-500/50">
        {index + 1}
      </div>
      <p className="text-lg text-gray-700 font-medium group-hover:text-blue-600 transition-colors">
        {step}
      </p>
      <ChevronRight className="ml-auto h-6 w-6 text-gray-400 group-hover:text-blue-600 group-hover:translate-x-2 transition-all" />
    </motion.div>
  );
}

interface ObjectiveProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  delay: number;
}

function ObjectiveCard({ icon, title, description, delay }: ObjectiveProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay }}
      className="text-center group"
    >
      <motion.div
        whileHover={{ scale: 1.1, rotate: 360 }}
        transition={{ duration: 0.6 }}
        className="w-24 h-24 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-lg group-hover:shadow-xl group-hover:from-blue-500 group-hover:to-indigo-600 transition-all duration-300"
      >
        <div className="text-blue-600 group-hover:text-white transition-colors">
          {icon}
        </div>
      </motion.div>
      <h3 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-blue-600 transition-colors">
        {title}
      </h3>
      <p className="text-gray-600 leading-relaxed">{description}</p>
    </motion.div>
  );
}

// Data
const disciplines = [
  {
    icon: <Target className="h-10 w-10" />,
    title: "Game Theory Framework",
    description: "Model the strategic interactions between news spreaders, fact-checkers, and platforms to identify equilibrium outcomes and vulnerabilities."
  },
  {
    icon: <Brain className="h-10 w-10" />,
    title: "Machine Learning Classification",
    description: "Utilize advanced NLP models like BERT to classify news content with high accuracy, identifying linguistic and stylistic patterns indicative of misinformation."
  },
  {
    icon: <Network className="h-10 w-10" />,
    title: "Social Network Analysis",
    description: "Simulate information cascades on scale-free networks to understand how topology influences the speed and reach of fake news."
  }
];

const workflow = [
  "Ingest real-world datasets from FakeNewsNet and LIAR",
  "Extract linguistic and network-based features",
  "Classify content using ensemble ML models",
  "Parameterize game-theoretic simulations",
  "Analyze Nash equilibria and propagation patterns"
];

const objectives = [
  {
    icon: <Target className="h-10 w-10" />,
    title: "Model Strategic Interactions",
    description: "Understand the decision-making of different actors in the information ecosystem."
  },
  {
    icon: <TrendingUp className="h-10 w-10" />,
    title: "Predict Propagation Patterns",
    description: "Forecast the spread and reach of fake news using ML and network metrics."
  },
  {
    icon: <Shield className="h-10 w-10" />,
    title: "Optimize Platform Policies",
    description: "Provide data-driven recommendations for effective content moderation strategies."
  },
  {
    icon: <BarChart3 className="h-10 w-10" />,
    title: "Visualize Complex Dynamics",
    description: "Create intuitive interfaces for exploring the interplay between strategy and network structure."
  }
];