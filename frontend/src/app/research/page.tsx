"use client";

import { motion } from "framer-motion";
import dynamic from 'next/dynamic';

const CountUp = dynamic(() => import('@/components/countup-wrapper'), { ssr: false });
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Image from 'next/image';

const ResearchPage = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring",
        stiffness: 100,
      },
    },
  };

  const hoverEffect = {
    scale: 1.05,
    boxShadow: "0px 10px 30px rgba(0, 0, 0, 0.1)",
  };

  return (
    <div className="container mx-auto py-12">
      <PageHeader
        title="Research & Findings"
        subtitle="An in-depth look at our multi-disciplinary approach to understanding and combating misinformation."
      />

      <motion.div
        className="max-w-4xl mx-auto"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.section className="mb-16" variants={itemVariants}>
          <h2 className="text-3xl font-bold mb-4">Abstract</h2>
          <p className="text-lg text-gray-700 dark:text-gray-300">
            This project investigates the complex dynamics of misinformation through a unique combination of game theory, network analysis, and machine learning. We model the strategic interactions between fake news spreaders, fact-checkers, and platforms to identify equilibrium behaviors and design effective intervention strategies. Our findings reveal that a multi-faceted approach, combining automated detection with well-designed incentives, is crucial for mitigating the spread of false information.
          </p>
        </motion.section>

        <motion.section className="mb-16" variants={itemVariants}>
          <h2 className="text-3xl font-bold mb-8 text-center">Key Highlights 📊</h2>
          <motion.div
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
            variants={containerVariants}
          >
            <motion.div variants={itemVariants} whileHover={hoverEffect}>
              <Card className="text-center h-full">
                <CardHeader>
                  <CardTitle className="text-5xl font-extrabold text-blue-600">
                    <CountUp end={79} duration={2.5} />%
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <h3 className="text-xl font-semibold mb-2">Effectiveness</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Combined intervention strategies (labeling + penalties) show significant synergistic effects in reducing misinformation.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
            <motion.div variants={itemVariants} whileHover={hoverEffect}>
              <Card className="text-center h-full">
                <CardHeader>
                  <CardTitle className="text-5xl font-extrabold text-green-600">
                    <CountUp end={2.4} decimals={1} duration={2.5} />x
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <h3 className="text-xl font-semibold mb-2">Cascade Amplification</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Scale-free networks, mimicking real social media, amplify the spread of fake news dramatically compared to random networks.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
            <motion.div variants={itemVariants} whileHover={hoverEffect}>
              <Card className="text-center h-full">
                <CardHeader>
                  <CardTitle className="text-5xl font-extrabold text-red-600">
                    Stable
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <h3 className="text-xl font-semibold mb-2">Suboptimal Equilibrium</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Platforms often settle into a state where some misinformation is tolerated to maintain user engagement.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </motion.section>

        <motion.section className="mb-16" variants={itemVariants}>
          <h2 className="text-3xl font-bold mb-8 text-center">Visualized Results 📈</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <motion.div variants={itemVariants}>
              <Image
                src="/enhanced_propagation_analysis.png"
                alt="Propagation Analysis"
                width={600}
                height={400}
                className="rounded-lg shadow-lg"
              />
            </motion.div>
            <motion.div variants={itemVariants}>
              <p className="text-lg text-gray-700 dark:text-gray-300">
                This chart demonstrates the clear divergence in velocity and reach between fake and real news over time. Fake news, represented by the red line, shows a much faster initial spread and a wider overall reach compared to real news (blue line), highlighting the urgent need for early detection and intervention.
              </p>
            </motion.div>
          </div>
        </motion.section>

        <motion.section className="mb-16" variants={itemVariants}>
          <h2 className="text-3xl font-bold mb-8 text-center">Explore Our Research Hub 探索</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <motion.div variants={itemVariants} whileHover={hoverEffect}>
              <Card className="h-full">
                <CardContent className="p-6">
                  <h3 className="text-xl font-semibold mb-2">The Methodology</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    A deep dive into the game theory models, machine learning architectures, and network analysis techniques that power our findings.
                  </p>
                  <Button asChild variant="outline">
                    <a href="/research/methodology">Learn More</a>
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
            <motion.div variants={itemVariants} whileHover={hoverEffect}>
              <Card className="h-full">
                <CardContent className="p-6">
                  <h3 className="text-xl font-semibold mb-2">Interactive Dashboard</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Run your own simulations. Adjust parameters and explore the strategic dynamics of misinformation spread in real-time.
                  </p>
                  <Button asChild>
                    <a href="/simulation">Explore</a>
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </motion.section>

        <motion.section className="mb-16" variants={itemVariants}>
          <h2 className="text-3xl font-bold mb-4">Policy Recommendations 🏛️</h2>
          <ul className="list-disc list-inside space-y-2 text-lg text-gray-700 dark:text-gray-300">
            <li><b>Implement Graduated Response Systems:</b> Platforms should apply a range of interventions, from gentle warnings to account suspensions, based on the severity and frequency of misinformation spread.</li>
            <li><b>Incentivize Fact-Checkers:</b> Create reward mechanisms for professional and citizen fact-checkers to increase the supply of accurate information.</li>
            <li><b>Promote Algorithmic Transparency:</b> Regulators should require platforms to provide greater transparency into the algorithms that rank and recommend content.</li>
          </ul>
        </motion.section>

        <motion.div className="text-center" variants={itemVariants}>
          <Button asChild size="lg">
            <a href="/assets/papers/main_research_paper.pdf" download>
              Download Full Research Paper
            </a>
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default ResearchPage;
