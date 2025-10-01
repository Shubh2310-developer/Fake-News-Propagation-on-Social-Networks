// frontend/src/app/about/page.tsx

"use client";

import React, { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import {
  Network,
  Brain,
  BarChart3,
  Target,
  Users,
  GitBranch,
  Github,
  Mail,
  ArrowRight,
  Cpu,
  Database,
  Zap,
  Shield
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import Image from 'next/image';

/**
 * AnimatedSection: Wrapper component for scroll-triggered animations
 */
const AnimatedSection: React.FC<{
  children: React.ReactNode;
  delay?: number;
  className?: string;
}> = ({ children, delay = 0, className = "" }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 50 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
      transition={{ duration: 0.6, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
};

/**
 * MethodologyCard: Component for methodology discipline cards
 */
const MethodologyCard: React.FC<{
  icon: React.ReactNode;
  title: string;
  description: string;
  delay: number;
}> = ({ icon, title, description, delay }) => {
  return (
    <AnimatedSection delay={delay}>
      <Card className="h-full transition-all duration-300 hover:shadow-lg hover:scale-105">
        <CardHeader>
          <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 text-white">
            {icon}
          </div>
          <CardTitle className="text-center text-xl">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-slate-600 dark:text-slate-400 leading-relaxed">
            {description}
          </p>
        </CardContent>
      </Card>
    </AnimatedSection>
  );
};

/**
 * ObjectiveItem: Component for key objectives
 */
const ObjectiveItem: React.FC<{
  icon: React.ReactNode;
  title: string;
  description: string;
}> = ({ icon, title, description }) => {
  return (
    <div className="flex gap-4 p-4 rounded-lg bg-slate-50 dark:bg-slate-800/50 transition-all duration-300 hover:bg-slate-100 dark:hover:bg-slate-800">
      <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 text-white flex items-center justify-center">
        {icon}
      </div>
      <div>
        <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-1">{title}</h4>
        <p className="text-sm text-slate-600 dark:text-slate-400">{description}</p>
      </div>
    </div>
  );
};

/**
 * TechStackItem: Component for technology stack items
 */
const TechStackItem: React.FC<{
  name: string;
  category: string;
  color: string;
}> = ({ name, category, color }) => {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-slate-800 shadow-sm border border-slate-200 dark:border-slate-700"
    >
      <div className={`w-12 h-12 rounded-full ${color} flex items-center justify-center mb-2`}>
        <Cpu className="w-6 h-6 text-white" />
      </div>
      <span className="font-semibold text-sm text-slate-900 dark:text-slate-100">{name}</span>
      <span className="text-xs text-slate-500 dark:text-slate-400">{category}</span>
    </motion.div>
  );
};

/**
 * About Page: Main component
 */
export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-950 dark:to-slate-900">
      {/* Hero Section */}
      <section className="relative py-20 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-green-500/10" />
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-4xl mx-auto text-center"
          >
            <Badge className="mb-4 px-4 py-1 text-sm font-medium">Research Platform</Badge>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 bg-clip-text text-transparent">
              About The Project
            </h1>
            <p className="text-xl md:text-2xl text-slate-700 dark:text-slate-300 leading-relaxed">
              A comprehensive research platform combining game theory, machine learning, and network analysis
              to understand and combat fake news propagation in social networks.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Vision Section */}
      <section className="py-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="max-w-4xl mx-auto">
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-3xl flex items-center gap-3">
                  <Target className="w-8 h-8 text-blue-600" />
                  Our Vision
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                  In the digital age, misinformation spreads faster than ever before, threatening democratic processes,
                  public health, and social cohesion. Traditional detection methods struggle to keep pace with the
                  strategic behavior of information spreaders who continuously adapt their tactics.
                </p>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                  Our project takes a revolutionary approach by modeling fake news propagation as a <span className="font-semibold text-blue-600">strategic game</span> between
                  information spreaders and fact-checkers. By combining <span className="font-semibold text-purple-600">game theory</span> with
                  advanced <span className="font-semibold text-green-600">machine learning</span> and <span className="font-semibold text-orange-600">network analysis</span>,
                  we provide unprecedented insights into how misinformation spreads and develop optimal counter-strategies.
                </p>
                <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
                  <p className="text-base text-slate-600 dark:text-slate-400 italic">
                    This multi-disciplinary approach allows us to not just detect fake news, but to understand the
                    strategic incentives driving its spread and design platform policies that fundamentally change
                    the economics of misinformation.
                  </p>
                </div>
              </CardContent>
            </Card>
          </AnimatedSection>
        </div>
      </section>

      {/* Methodology Section */}
      <section className="py-16 bg-slate-100 dark:bg-slate-900/50">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4 text-slate-900 dark:text-slate-100">Our Methodology</h2>
            <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              A comprehensive, interdisciplinary approach combining three core disciplines
            </p>
          </AnimatedSection>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <MethodologyCard
              icon={<BarChart3 className="w-8 h-8" />}
              title="Game Theory"
              description="Models strategic interactions between information spreaders, fact-checkers, and platforms.
                We compute Nash equilibria to identify stable strategy profiles and optimal counter-strategies,
                considering costs, benefits, and reputation effects."
              delay={0.1}
            />
            <MethodologyCard
              icon={<Brain className="w-8 h-8" />}
              title="Machine Learning"
              description="Advanced classifiers including Random Forest, Ensemble models, and transformer-based architectures
                achieving 87.8% accuracy. Our models process 2,031 engineered features for real-time fake news detection
                with sub-100ms inference."
              delay={0.2}
            />
            <MethodologyCard
              icon={<Network className="w-8 h-8" />}
              title="Network Analysis"
              description="Analyzes social network topologies and information diffusion patterns using graph theory and
                complex systems modeling. We identify influential nodes, community structures, and optimal intervention
                points to prevent misinformation cascades."
              delay={0.3}
            />
          </div>
        </div>
      </section>

      {/* Key Objectives Section */}
      <section className="py-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4 text-slate-900 dark:text-slate-100">Key Objectives</h2>
            <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              Our primary aims in combating misinformation and informing policy
            </p>
          </AnimatedSection>

          <AnimatedSection delay={0.2} className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4">
            <ObjectiveItem
              icon={<Users className="w-6 h-6" />}
              title="Model Strategic Interactions"
              description="Mathematically model the game between spreaders, fact-checkers, and platforms to understand
                incentive structures and predict behavior."
            />
            <ObjectiveItem
              icon={<GitBranch className="w-6 h-6" />}
              title="Predict Propagation Patterns"
              description="Simulate information diffusion across diverse network topologies to forecast misinformation
                reach and identify vulnerable communities."
            />
            <ObjectiveItem
              icon={<Shield className="w-6 h-6" />}
              title="Optimize Platform Policies"
              description="Design and evaluate content moderation policies, algorithmic interventions, and platform
                rules that minimize misinformation spread."
            />
            <ObjectiveItem
              icon={<Zap className="w-6 h-6" />}
              title="Real-Time Detection"
              description="Deploy production-ready machine learning models for immediate fake news classification with
                high accuracy and interpretability."
            />
            <ObjectiveItem
              icon={<Target className="w-6 h-6" />}
              title="Inform Policy Makers"
              description="Provide evidence-based recommendations for regulatory frameworks and platform governance
                structures to combat systemic misinformation."
            />
            <ObjectiveItem
              icon={<Database className="w-6 h-6" />}
              title="Build Open Research Tools"
              description="Create reusable, extensible software for researchers to conduct game-theoretic analysis of
                information ecosystems."
            />
          </AnimatedSection>
        </div>
      </section>

      {/* Technology Stack Section */}
      <section className="py-16 bg-slate-100 dark:bg-slate-900/50">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4 text-slate-900 dark:text-slate-100">Technology Stack</h2>
            <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              Built with cutting-edge technologies for research and production deployment
            </p>
          </AnimatedSection>

          <AnimatedSection delay={0.2} className="max-w-5xl mx-auto">
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              <TechStackItem name="Next.js" category="Frontend" color="bg-slate-900" />
              <TechStackItem name="FastAPI" category="Backend" color="bg-green-600" />
              <TechStackItem name="PyTorch" category="ML Framework" color="bg-orange-600" />
              <TechStackItem name="D3.js" category="Visualization" color="bg-yellow-600" />
              <TechStackItem name="PostgreSQL" category="Database" color="bg-blue-700" />
              <TechStackItem name="NetworkX" category="Graph Analysis" color="bg-purple-600" />
              <TechStackItem name="Transformers" category="NLP" color="bg-pink-600" />
              <TechStackItem name="Tailwind CSS" category="Styling" color="bg-cyan-600" />
              <TechStackItem name="Docker" category="Infrastructure" color="bg-blue-500" />
              <TechStackItem name="TypeScript" category="Language" color="bg-blue-600" />
            </div>
          </AnimatedSection>

          <AnimatedSection delay={0.4} className="mt-12 max-w-4xl mx-auto">
            <Card className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-blue-200 dark:border-blue-800">
              <CardContent className="pt-6">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-full bg-blue-600 text-white flex items-center justify-center">
                    <Cpu className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-lg mb-2 text-slate-900 dark:text-slate-100">
                      Performance Highlights
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="font-bold text-2xl text-blue-600">87.8%</span>
                        <p className="text-slate-600 dark:text-slate-400">Classification Accuracy</p>
                      </div>
                      <div>
                        <span className="font-bold text-2xl text-purple-600">&lt;100ms</span>
                        <p className="text-slate-600 dark:text-slate-400">Inference Time</p>
                      </div>
                      <div>
                        <span className="font-bold text-2xl text-green-600">2,031</span>
                        <p className="text-slate-600 dark:text-slate-400">Engineered Features</p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </AnimatedSection>
        </div>
      </section>

      {/* Call for Collaboration Section */}
      <section className="py-20 bg-gradient-to-br from-blue-600 via-purple-600 to-green-600 text-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="max-w-4xl mx-auto text-center">
            <motion.div
              initial={{ scale: 0.9 }}
              whileInView={{ scale: 1 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl md:text-5xl font-bold mb-6">Join Our Research Community</h2>
              <p className="text-xl mb-8 text-blue-100 leading-relaxed">
                We envision this as an <span className="font-semibold">open research platform</span> where academics,
                platform engineers, policymakers, and fact-checkers can collaborate to build more resilient information
                ecosystems. Our tools and methodologies are freely available for researchers worldwide.
              </p>
              <p className="text-lg mb-10 text-blue-100">
                Whether you're interested in extending our game-theoretic models, contributing new datasets,
                or applying our methods to new domains, we welcome your collaboration.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button
                    size="lg"
                    className="bg-white text-blue-600 hover:bg-blue-50 font-semibold px-8 py-6 text-lg group"
                    onClick={() => window.open('https://github.com/your-username/fake-news-game-theory', '_blank')}
                  >
                    <Github className="mr-2 w-5 h-5 group-hover:rotate-12 transition-transform" />
                    View on GitHub
                    <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </Button>
                </motion.div>

                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button
                    size="lg"
                    variant="outline"
                    className="bg-transparent border-2 border-white text-white hover:bg-white/10 font-semibold px-8 py-6 text-lg group"
                    onClick={() => window.location.href = 'mailto:research@example.com'}
                  >
                    <Mail className="mr-2 w-5 h-5 group-hover:scale-110 transition-transform" />
                    Contact Research Team
                  </Button>
                </motion.div>
              </div>

              <div className="mt-12 pt-8 border-t border-white/20">
                <p className="text-sm text-blue-100">
                  Licensed under MIT " Built with d for the research community
                </p>
              </div>
            </motion.div>
          </AnimatedSection>
        </div>
      </section>
    </div>
  );
}
