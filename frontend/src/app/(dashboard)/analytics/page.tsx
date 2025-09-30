// frontend/src/app/(dashboard)/analytics/page.tsx

"use client";

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Target,
  TrendingUp,
  Network,
  Shield,
  BarChart3,
  Zap,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart } from '@/components/charts/LineChart';
import { BarChart } from '@/components/charts/BarChart';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

// ================================================================
// Data Models - Based on Project Research Results
// ================================================================

// Model Performance Data (from notebooks)
const modelPerformanceData = [
  {
    model: 'BERT',
    accuracy: 87.2,
    f1Score: 86.8,
    auc: 92.5
  },
  {
    model: 'LSTM',
    accuracy: 84.3,
    f1Score: 83.7,
    auc: 89.2
  },
  {
    model: 'Ensemble',
    accuracy: 88.4,
    f1Score: 88.1,
    auc: 93.8
  },
  {
    model: 'Traditional ML',
    accuracy: 79.6,
    f1Score: 78.4,
    auc: 84.7
  },
];

// Feature Importance Data (Top 15 features from analysis)
const featureImportanceData = [
  { feature: 'Source Credibility', importance: 0.284, category: 'Network' },
  { feature: 'Sentiment Polarity', importance: 0.267, category: 'Linguistic' },
  { feature: 'Named Entity Density', importance: 0.241, category: 'Linguistic' },
  { feature: 'Network Centrality', importance: 0.228, category: 'Network' },
  { feature: 'Readability Score', importance: 0.215, category: 'Stylistic' },
  { feature: 'Emotional Language', importance: 0.203, category: 'Linguistic' },
  { feature: 'Propagation Velocity', importance: 0.192, category: 'Network' },
  { feature: 'Clickbait Indicators', importance: 0.187, category: 'Stylistic' },
  { feature: 'Citation Quality', importance: 0.176, category: 'Network' },
  { feature: 'Writing Complexity', importance: 0.168, category: 'Stylistic' },
  { feature: 'Engagement Rate', importance: 0.157, category: 'Network' },
  { feature: 'Sensationalism Score', importance: 0.149, category: 'Stylistic' },
  { feature: 'Temporal Patterns', importance: 0.138, category: 'Network' },
  { feature: 'Lexical Diversity', importance: 0.126, category: 'Linguistic' },
  { feature: 'Factual Claims Ratio', importance: 0.114, category: 'Linguistic' },
];

// Information Propagation Data (24-hour reach comparison)
const propagationData = [
  { hour: 0, fakeNews: 0, realNews: 0 },
  { hour: 2, fakeNews: 1250, realNews: 520 },
  { hour: 4, fakeNews: 3840, realNews: 1430 },
  { hour: 6, fakeNews: 7920, realNews: 3210 },
  { hour: 8, fakeNews: 14500, realNews: 5890 },
  { hour: 10, fakeNews: 23400, realNews: 9450 },
  { hour: 12, fakeNews: 35800, realNews: 14200 },
  { hour: 14, fakeNews: 48900, realNews: 19800 },
  { hour: 16, fakeNews: 61200, realNews: 26100 },
  { hour: 18, fakeNews: 72500, realNews: 32900 },
  { hour: 20, fakeNews: 81300, realNews: 39200 },
  { hour: 22, fakeNews: 87600, realNews: 44800 },
  { hour: 24, fakeNews: 91200, realNews: 49300 },
];

// Intervention Strategy Data
const interventionStrategies = [
  {
    id: 1,
    strategy: 'Content Labeling',
    reduction: 42,
    costEffectiveness: 'High',
    complexity: 'Low',
    implementationTime: '2-4 weeks',
  },
  {
    id: 2,
    strategy: 'Fact-Checking Alerts',
    reduction: 58,
    costEffectiveness: 'Medium',
    complexity: 'Medium',
    implementationTime: '4-8 weeks',
  },
  {
    id: 3,
    strategy: 'Source Verification',
    reduction: 67,
    costEffectiveness: 'Medium',
    complexity: 'High',
    implementationTime: '8-12 weeks',
  },
  {
    id: 4,
    strategy: 'Network Penalties',
    reduction: 54,
    costEffectiveness: 'High',
    complexity: 'Medium',
    implementationTime: '6-10 weeks',
  },
  {
    id: 5,
    strategy: 'Labeling + Penalties',
    reduction: 79,
    costEffectiveness: 'Very High',
    complexity: 'Medium',
    implementationTime: '8-12 weeks',
  },
  {
    id: 6,
    strategy: 'Full Multi-Layer Strategy',
    reduction: 86,
    costEffectiveness: 'Medium',
    complexity: 'Very High',
    implementationTime: '16-24 weeks',
  },
];

// ================================================================
// Components
// ================================================================

// Animated Counter Hook
function useCountUp(end: number, duration: number = 2000) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTime: number | null = null;
    const startValue = 0;

    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);

      // Easing function for smooth animation
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const currentCount = startValue + (end - startValue) * easeOutQuart;

      setCount(currentCount);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        setCount(end);
      }
    };

    requestAnimationFrame(animate);
  }, [end, duration]);

  return count;
}

// KPI Metrics Card with Animation
interface MetricsCardProps {
  title: string;
  value: number;
  suffix?: string;
  prefix?: string;
  icon: React.ReactNode;
  description: string;
  trend?: number;
  delay?: number;
}

const MetricsCard: React.FC<MetricsCardProps> = ({
  title,
  value,
  suffix = '',
  prefix = '',
  icon,
  description,
  trend,
  delay = 0
}) => {
  const animatedValue = useCountUp(value, 2000);
  const displayValue = prefix + animatedValue.toFixed(value % 1 === 0 ? 0 : 1) + suffix;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
    >
      <Card className="hover:shadow-lg transition-shadow duration-200">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {title}
          </CardTitle>
          <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            {icon}
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold text-gray-900 dark:text-gray-50">
            {displayValue}
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
            {description}
          </p>
          {trend !== undefined && (
            <div className="flex items-center gap-1 mt-2">
              <TrendingUp className={cn(
                "h-3 w-3",
                trend >= 0 ? "text-green-600" : "text-red-600"
              )} />
              <span className={cn(
                "text-xs font-medium",
                trend >= 0 ? "text-green-600" : "text-red-600"
              )}>
                {trend > 0 ? '+' : ''}{trend}% vs baseline
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

// Filter Bar Component
const FilterBar: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="flex flex-wrap items-center gap-3 p-4 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800"
    >
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Dataset:
      </span>
      <Badge variant="default" className="cursor-pointer">
        FakeNewsNet
      </Badge>
      <Badge variant="outline" className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800">
        LIAR
      </Badge>
      <Badge variant="outline" className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800">
        Combined
      </Badge>
      <div className="flex-1" />
      <span className="text-xs text-gray-500 dark:text-gray-400">
        Last updated: 2 hours ago
      </span>
    </motion.div>
  );
};

// Main Analytics Page
export default function AnalyticsPage() {
  // Chart series configurations
  const modelPerformanceSeries = [
    { dataKey: 'accuracy', name: 'Accuracy (%)', color: '#3b82f6' },
    { dataKey: 'f1Score', name: 'F1-Score (%)', color: '#8b5cf6' },
    { dataKey: 'auc', name: 'AUC-ROC (%)', color: '#10b981' },
  ];

  const propagationSeries = [
    { dataKey: 'fakeNews', name: 'Fake News Reach', color: '#ef4444' },
    { dataKey: 'realNews', name: 'Real News Reach', color: '#10b981' },
  ];

  // Get category color for feature importance
  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Linguistic': return '#3b82f6'; // Blue
      case 'Network': return '#8b5cf6'; // Purple
      case 'Stylistic': return '#f59e0b'; // Amber
      default: return '#6b7280'; // Gray
    }
  };

  // Add color to feature importance data
  const featureChartData = featureImportanceData.map(item => ({
    ...item,
    importancePercent: (item.importance * 100).toFixed(1),
    color: getCategoryColor(item.category),
  }));

  // Get complexity badge color
  const getComplexityBadge = (complexity: string) => {
    switch (complexity.toLowerCase()) {
      case 'low':
        return <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">Low</Badge>;
      case 'medium':
        return <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400">Medium</Badge>;
      case 'high':
        return <Badge className="bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400">High</Badge>;
      case 'very high':
        return <Badge className="bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400">Very High</Badge>;
      default:
        return <Badge variant="secondary">{complexity}</Badge>;
    }
  };

  // Get cost effectiveness badge
  const getCostBadge = (cost: string) => {
    switch (cost.toLowerCase()) {
      case 'very high':
        return <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">Very High</Badge>;
      case 'high':
        return <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400">High</Badge>;
      case 'medium':
        return <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400">Medium</Badge>;
      default:
        return <Badge variant="secondary">{cost}</Badge>;
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
          Project Analytics & Insights
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-3 text-lg">
          Visualizing model performance, information propagation, and strategic outcomes
        </p>
      </motion.div>

      {/* Filter Bar */}
      <FilterBar />

      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricsCard
          title="Best Model Accuracy"
          value={88.4}
          suffix="%"
          icon={<Target className="h-5 w-5 text-blue-600" />}
          description="Ensemble model performance"
          trend={4.2}
          delay={0.1}
        />
        <MetricsCard
          title="Top Feature Importance"
          value={28.4}
          suffix="%"
          icon={<BarChart3 className="h-5 w-5 text-purple-600" />}
          description="Source Credibility impact"
          delay={0.2}
        />
        <MetricsCard
          title="Propagation Speed Ratio"
          value={1.7}
          suffix="x"
          icon={<Zap className="h-5 w-5 text-yellow-600" />}
          description="Fake news spreads faster"
          delay={0.3}
        />
        <MetricsCard
          title="Best Intervention"
          value={79}
          suffix="%"
          icon={<Shield className="h-5 w-5 text-green-600" />}
          description="Labeling + Penalties strategy"
          trend={25}
          delay={0.4}
        />
      </div>

      {/* Model Performance Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-blue-600" />
              Model Performance Comparison
            </CardTitle>
            <CardDescription>
              Accuracy, F1-Score, and AUC-ROC metrics across classification models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <BarChart
              data={modelPerformanceData}
              series={modelPerformanceSeries}
              xAxisKey="model"
            />
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Key Finding
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    The Ensemble model achieves the highest accuracy (88.4%) and AUC-ROC (93.8%),
                    demonstrating superior fake news detection capabilities by combining multiple
                    machine learning approaches.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Feature Importance Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-purple-600" />
              Feature Importance Analysis
            </CardTitle>
            <CardDescription>
              Top 15 features ranked by their impact on fake news detection
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Custom horizontal bar chart */}
            <div className="space-y-3">
              {featureChartData.slice(0, 10).map((item, index) => (
                <motion.div
                  key={item.feature}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.4, delay: 0.4 + index * 0.05 }}
                  className="space-y-1"
                >
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-gray-700 dark:text-gray-300">
                      {item.feature}
                    </span>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className="text-xs"
                        style={{
                          borderColor: item.color,
                          color: item.color,
                        }}
                      >
                        {item.category}
                      </Badge>
                      <span className="text-gray-900 dark:text-gray-100 font-semibold min-w-[3rem] text-right">
                        {item.importancePercent}%
                      </span>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2.5">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.importance * 100}%` }}
                      transition={{ duration: 1, delay: 0.5 + index * 0.05, ease: "easeOut" }}
                      className="h-2.5 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
            <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-5 w-5 text-purple-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Insight
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Source Credibility (28.4%) and Sentiment Polarity (26.7%) are the most powerful
                    indicators of misinformation, highlighting the importance of network-based and
                    linguistic features in detection.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Information Propagation Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5 text-red-600" />
              Information Propagation Dynamics
            </CardTitle>
            <CardDescription>
              24-hour reach comparison: Fake news vs. Real news spread patterns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <LineChart
              data={propagationData}
              series={propagationSeries}
              xAxisKey="hour"
            />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
              <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                <div className="text-2xl font-bold text-red-600">91.2K</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Peak fake news reach
                </div>
              </div>
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                <div className="text-2xl font-bold text-green-600">49.3K</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Peak real news reach
                </div>
              </div>
              <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                <div className="text-2xl font-bold text-yellow-600">1.85x</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Final spread ratio
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Intervention Strategies Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-green-600" />
              Intervention Strategy Effectiveness
            </CardTitle>
            <CardDescription>
              Comparative analysis of misinformation reduction strategies
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg border border-gray-200 dark:border-gray-800">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="font-semibold">Strategy</TableHead>
                    <TableHead className="font-semibold">Reduction</TableHead>
                    <TableHead className="font-semibold">Cost-Effectiveness</TableHead>
                    <TableHead className="font-semibold">Complexity</TableHead>
                    <TableHead className="font-semibold">Implementation Time</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {interventionStrategies.map((strategy, index) => (
                    <motion.tr
                      key={strategy.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: 0.6 + index * 0.05 }}
                      className="border-b transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/50"
                    >
                      <TableCell className="font-medium">
                        {strategy.strategy}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                            {strategy.reduction}%
                          </div>
                          <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${strategy.reduction}%` }}
                              transition={{ duration: 1, delay: 0.7 + index * 0.05 }}
                              className={cn(
                                "h-2 rounded-full",
                                strategy.reduction >= 75 ? "bg-green-500" :
                                strategy.reduction >= 50 ? "bg-blue-500" :
                                "bg-yellow-500"
                              )}
                            />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        {getCostBadge(strategy.costEffectiveness)}
                      </TableCell>
                      <TableCell>
                        {getComplexityBadge(strategy.complexity)}
                      </TableCell>
                      <TableCell className="text-sm text-gray-600 dark:text-gray-400">
                        {strategy.implementationTime}
                      </TableCell>
                    </motion.tr>
                  ))}
                </TableBody>
              </Table>
            </div>
            <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Recommended Strategy
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    The combined "Labeling + Penalties" approach offers the best balance with 79%
                    misinformation reduction, high cost-effectiveness, and moderate complexity,
                    making it ideal for most implementation scenarios.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}