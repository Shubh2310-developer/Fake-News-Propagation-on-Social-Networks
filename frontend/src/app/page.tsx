// /home/ghost/fake-news-game-theory/frontend/src/app/page.tsx
'use client';

import { useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, useInView } from 'framer-motion';
import Link from 'next/link';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { 
  Brain, 
  Network, 
  Target, 
  TrendingUp, 
  Shield, 
  BarChart3,
  ArrowRight,
  CheckCircle2
} from 'lucide-react';

export default function HomePage() {
  const { scrollYProgress } = useScroll();
  const heroRef = useRef<HTMLDivElement>(null);
  
  // Parallax effect for hero background
  const heroY = useTransform(scrollYProgress, [0, 0.3], [0, 100]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.2], [1, 0.3]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section 
        ref={heroRef}
        className="relative h-screen flex items-center justify-center overflow-hidden"
      >
        {/* Parallax Background */}
        <motion.div 
          className="absolute inset-0 z-0"
          style={{ y: heroY, opacity: heroOpacity }}
        >
          <Image
            src="/images/hero-bg.webp"
            alt="Network visualization background"
            fill
            className="object-cover"
            priority
          />
          <div className="absolute inset-0 bg-gradient-to-b from-gray-900/80 via-gray-900/70 to-gray-900/90" />
        </motion.div>

        {/* Hero Content */}
        <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
          <motion.h1 
            className="text-5xl md:text-7xl font-bold text-white mb-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, staggerChildren: 0.03 }}
          >
            {Array.from("Modeling Misinformation:").map((char, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.03 }}
              >
                {char}
              </motion.span>
            ))}
            <br />
            {Array.from("A Game Theory Approach").map((char, i) => (
              <motion.span
                key={i + 100}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: (i * 0.03) + 0.6 }}
              >
                {char}
              </motion.span>
            ))}
          </motion.h1>

          <motion.p 
            className="text-xl md:text-2xl text-gray-200 mb-12 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.5 }}
          >
            An innovative research platform integrating machine learning and network 
            analysis to predict fake news propagation and inform policy.
          </motion.p>

          <motion.div 
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.8 }}
          >
            <Link href="/simulation">
              <Button 
                size="lg" 
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-6 text-lg group"
              >
                Explore the Dashboard
                <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Link href="/research">
              <Button 
                size="lg" 
                variant="outline" 
                className="border-2 border-white text-white hover:bg-white hover:text-gray-900 px-8 py-6 text-lg"
              >
                Read Our Research
              </Button>
            </Link>
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div 
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="w-6 h-10 border-2 border-white/50 rounded-full flex justify-center pt-2">
            <div className="w-1.5 h-3 bg-white/50 rounded-full" />
          </div>
        </motion.div>
      </section>

      {/* Core Disciplines Section */}
      <section className="py-24 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Three Foundational Pillars
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our multidisciplinary approach combines cutting-edge methodologies 
              to understand and combat misinformation.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {disciplines.map((discipline, index) => (
              <DisciplineCard 
                key={discipline.title}
                {...discipline}
                delay={index * 0.2}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Visualized Workflow Section */}
      <section className="py-24 px-6 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                From Data to Insight
              </h2>
              <p className="text-lg text-gray-700 mb-6 leading-relaxed">
                Our platform processes real-world datasets through a sophisticated pipeline. 
                Machine learning models classify content, which then informs the parameters 
                of our game-theoretic simulation. The resulting strategic dynamics are 
                visualized on complex network graphs, revealing patterns of misinformation spread.
              </p>
              <ul className="space-y-4 mb-8">
                {workflow.map((step, index) => (
                  <motion.li
                    key={step}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="flex items-start"
                  >
                    <CheckCircle2 className="h-6 w-6 text-green-600 mr-3 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">{step}</span>
                  </motion.li>
                ))}
              </ul>
              <Link href="/methodology">
                <Button variant="link" className="text-blue-600 p-0 h-auto group">
                  Learn more about the methodology
                  <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.8 }}
              className="relative"
            >
              <div className="relative aspect-square">
                <Image
                  src="/images/network-visualization.png"
                  alt="Network visualization"
                  fill
                  className="object-contain rounded-lg shadow-2xl"
                />
              </div>
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.3 }}
                className="absolute -bottom-6 -left-6 bg-white p-6 rounded-lg shadow-xl border border-gray-200"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                    <BarChart3 className="h-6 w-6 text-blue-600" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-gray-900">88.4%</div>
                    <div className="text-sm text-gray-600">Classification Accuracy</div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Project Objectives Section */}
      <section className="py-24 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Our Mission
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Advancing the science of misinformation through rigorous analysis 
              and actionable insights.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {objectives.map((objective, index) => (
              <ObjectiveCard 
                key={objective.title}
                {...objective}
                delay={index * 0.15}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="py-32 px-6 bg-gradient-to-br from-blue-600 to-blue-800 relative overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 left-0 w-96 h-96 bg-white rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2" />
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-white rounded-full blur-3xl translate-x-1/2 translate-y-1/2" />
        </div>

        <div className="max-w-4xl mx-auto text-center relative z-10">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-4xl md:text-5xl font-bold text-white mb-6"
          >
            Ready to Analyze the Dynamics of Misinformation?
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-xl text-blue-100 mb-12"
          >
            Explore interactive simulations, analyze network dynamics, and discover 
            insights that inform real-world policy decisions.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Link href="/simulation">
              <Button 
                size="lg" 
                className="bg-white text-blue-600 hover:bg-gray-100 px-12 py-7 text-xl font-semibold group shadow-2xl"
              >
                Launch Simulation Dashboard
                <ArrowRight className="ml-3 h-6 w-6 group-hover:translate-x-2 transition-transform" />
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
}

// Discipline Card Component
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
    >
      <Card className="p-8 h-full hover:shadow-xl transition-all duration-300 hover:-translate-y-2 border-2 border-gray-100">
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          transition={{ type: "spring", stiffness: 300 }}
          className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mb-6"
        >
          {icon}
        </motion.div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">{title}</h3>
        <p className="text-gray-600 leading-relaxed">{description}</p>
      </Card>
    </motion.div>
  );
}

// Objective Card Component
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
      className="text-center"
    >
      <motion.div
        whileHover={{ scale: 1.1, rotate: 360 }}
        transition={{ duration: 0.6 }}
        className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6"
      >
        {icon}
      </motion.div>
      <h3 className="text-xl font-bold text-gray-900 mb-3">{title}</h3>
      <p className="text-gray-600 leading-relaxed">{description}</p>
    </motion.div>
  );
}

// Data
const disciplines = [
  {
    icon: <Target className="h-8 w-8 text-blue-600" />,
    title: "Game Theory Framework",
    description: "Model the strategic interactions between news spreaders, fact-checkers, and platforms to identify equilibrium outcomes and vulnerabilities."
  },
  {
    icon: <Brain className="h-8 w-8 text-blue-600" />,
    title: "Machine Learning Classification",
    description: "Utilize advanced NLP models like BERT to classify news content with high accuracy, identifying linguistic and stylistic patterns indicative of misinformation."
  },
  {
    icon: <Network className="h-8 w-8 text-blue-600" />,
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
    icon: <Target className="h-10 w-10 text-blue-600" />,
    title: "Model Strategic Interactions",
    description: "Understand the decision-making of different actors in the information ecosystem."
  },
  {
    icon: <TrendingUp className="h-10 w-10 text-blue-600" />,
    title: "Predict Propagation Patterns",
    description: "Forecast the spread and reach of fake news using ML and network metrics."
  },
  {
    icon: <Shield className="h-10 w-10 text-blue-600" />,
    title: "Optimize Platform Policies",
    description: "Provide data-driven recommendations for effective content moderation strategies."
  },
  {
    icon: <BarChart3 className="h-10 w-10 text-blue-600" />,
    title: "Visualize Complex Dynamics",
    description: "Create intuitive interfaces for exploring the interplay between strategy and network structure."
  }
];