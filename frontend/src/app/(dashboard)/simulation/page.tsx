// frontend/src/app/(dashboard)/simulation/page.tsx

"use client";

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Square,
  RotateCcw,
  Loader2,
  Network,
  TrendingUp,
  BarChart3,
  Target,
  AlertCircle,
  CheckCircle2,
  Activity,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { LineChart } from '@/components/charts/LineChart';
import { PayoffMatrix } from '@/components/game-theory/PayoffMatrix';
import { GameParameters } from './components/GameParameters';
import { NetworkGraph } from './components/NetworkGraph';
import { useSimulationStore } from '@/store/simulationStore';
import { cn } from '@/lib/utils';

// ================================================================
// Types and Mock Data
// ================================================================

// Mock network data generator
function generateMockNetworkData(nodeCount: number = 100) {
  const nodes = Array.from({ length: nodeCount }, (_, i) => ({
    id: `node-${i}`,
    type: i < 5 ? 'spreader' : i < 15 ? 'moderator' : i < 20 ? 'bot' : 'user',
    status: i < 5 ? 'infected' : 'susceptible',
    influence: Math.random() * 5 + 1,
    connections: 0,
    label: `N${i}`,
  })) as any[];

  const links = [];
  for (let i = 0; i < nodeCount * 2; i++) {
    const source = Math.floor(Math.random() * nodeCount);
    const target = Math.floor(Math.random() * nodeCount);
    if (source !== target) {
      links.push({
        source: `node-${source}`,
        target: `node-${target}`,
        strength: Math.random() * 0.5 + 0.5,
        type: 'follow',
      });
    }
  }

  nodes.forEach(node => {
    node.connections = links.filter(l =>
      l.source === node.id || l.target === node.id
    ).length;
  });

  return { nodes, links };
}

// Animated counter hook
function useCountUp(end: number, duration: number = 1500) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTime: number | null = null;
    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      setCount(end * easeOutQuart);

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

// Metrics Card Component
interface MetricsCardProps {
  title: string;
  value: number;
  unit?: string;
  icon: React.ReactNode;
  description: string;
  animated?: boolean;
}

const MetricsCard: React.FC<MetricsCardProps> = ({
  title,
  value,
  unit = '',
  icon,
  description,
  animated = false,
}) => {
  const animatedValue = useCountUp(animated ? value : 0, 1500);
  const displayValue = animated ? animatedValue : value;

  return (
    <Card>
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
          {displayValue.toFixed(value % 1 === 0 ? 0 : 1)}{unit}
        </div>
        <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
          {description}
        </p>
      </CardContent>
    </Card>
  );
};

// Status Indicator Component
const StatusIndicator: React.FC<{ state: string; isRunning: boolean }> = ({ state, isRunning }) => {
  const getStatusConfig = () => {
    if (isRunning) {
      return {
        label: 'Running',
        color: 'bg-blue-500',
        icon: <Loader2 className="h-3 w-3 animate-spin" />,
      };
    }
    switch (state) {
      case 'completed':
        return {
          label: 'Completed',
          color: 'bg-green-500',
          icon: <CheckCircle2 className="h-3 w-3" />,
        };
      case 'error':
        return {
          label: 'Error',
          color: 'bg-red-500',
          icon: <AlertCircle className="h-3 w-3" />,
        };
      default:
        return {
          label: 'Idle',
          color: 'bg-gray-400',
          icon: <Activity className="h-3 w-3" />,
        };
    }
  };

  const config = getStatusConfig();

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-900 rounded-full shadow-lg border border-gray-200 dark:border-gray-800"
    >
      <div className={cn("h-2 w-2 rounded-full", config.color, isRunning && "animate-pulse")} />
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-1">
        {config.icon}
        {config.label}
      </span>
    </motion.div>
  );
};

// ================================================================
// Main Simulation Page Component
// ================================================================

export default function SimulationPage() {
  const {
    gameParameters,
    simulationState,
    isRunning,
    results,
    error,
    setGameParameters,
    startSimulation,
    stopSimulation,
    resetSimulation,
  } = useSimulationStore();

  const [networkData, setNetworkData] = useState(generateMockNetworkData(gameParameters.network?.size || 1000));
  const [showResults, setShowResults] = useState(false);

  // Update network when simulation completes
  useEffect(() => {
    if (simulationState === 'completed' && results) {
      setShowResults(true);
      // Update network with simulation results
      setNetworkData(generateMockNetworkData(gameParameters.network?.size || 1000));
    }
  }, [simulationState, results, gameParameters.network?.size]);

  // Handle parameter submission
  const handleParametersSubmit = async (params: any) => {
    setGameParameters(params);
    await startSimulation();
  };

  // Mock propagation timeline data
  const propagationData = Array.from({ length: 20 }, (_, i) => ({
    step: i * 5,
    infected: Math.min(100, i * 5 + Math.random() * 10),
    susceptible: Math.max(0, 100 - i * 5 - Math.random() * 10),
    recovered: i * 2 + Math.random() * 5,
  }));

  const propagationSeries = [
    { dataKey: 'infected', name: 'Infected', color: '#ef4444' },
    { dataKey: 'susceptible', name: 'Susceptible', color: '#94a3b8' },
    { dataKey: 'recovered', name: 'Recovered', color: '#10b981' },
  ];

  // Mock payoff matrix
  const mockPayoffMatrix = {
    players: ['Spreader', 'Moderator'] as [string, string],
    strategies: {
      Spreader: ['Aggressive', 'Conservative'],
      Moderator: ['Strict', 'Lenient'],
    },
    payoffs: [
      [
        { Spreader: 1.2, Moderator: -0.8 },
        { Spreader: 1.8, Moderator: 0.4 },
      ],
      [
        { Spreader: 0.6, Moderator: -0.3 },
        { Spreader: 0.9, Moderator: 0.7 },
      ],
    ],
    equilibrium: {
      strategies: [1, 1] as [number, number],
      payoffs: { Spreader: 0.9, Moderator: 0.7 },
      type: 'pure' as const,
      stability: 0.85,
      classification: 'strict' as const,
    },
  };

  // Mock summary metrics
  const summaryMetrics = {
    totalReach: results?.propagationStats?.totalViews || 8542,
    finalReputation: 72.5,
    detectionRate: results?.propagationStats?.detectionRate || 68.3,
    equilibriumStability: results?.equilibrium?.stability || 0.85,
  };

  return (
    <div className="space-y-8 relative">
      {/* Status Indicator */}
      <StatusIndicator state={simulationState} isRunning={isRunning} />

      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
            <Network className="h-6 w-6 text-green-600" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
            Game Theory Simulation
          </h1>
        </div>
        <p className="text-gray-600 dark:text-gray-400 text-lg">
          Model and visualize fake news propagation through strategic interactions on a social network
        </p>
      </motion.div>

      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Simulation Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Global Action Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="flex items-center gap-3"
      >
        <Button
          onClick={startSimulation}
          disabled={isRunning}
          size="lg"
          className="flex items-center gap-2"
        >
          {isRunning ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              Start Simulation
            </>
          )}
        </Button>

        {isRunning && (
          <Button
            onClick={stopSimulation}
            variant="outline"
            size="lg"
            className="flex items-center gap-2"
          >
            <Square className="h-4 w-4" />
            Stop Simulation
          </Button>
        )}

        <Button
          onClick={resetSimulation}
          variant="outline"
          size="lg"
          className="flex items-center gap-2"
        >
          <RotateCcw className="h-4 w-4" />
          Reset
        </Button>

        <div className="flex-1" />

        {showResults && (
          <Badge variant="outline" className="text-base px-4 py-2">
            <CheckCircle2 className="h-4 w-4 mr-2 text-green-600" />
            Results Available
          </Badge>
        )}
      </motion.div>

      {/* Two-Column Layout: Controls and Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Control Panel (1/3 width) */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="lg:col-span-1"
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-purple-600" />
                Simulation Parameters
              </CardTitle>
              <CardDescription>
                Configure the game theory simulation settings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <GameParameters
                onSubmit={handleParametersSubmit}
                isLoading={isRunning}
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* Right Column: Network Visualization (2/3 width) */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="lg:col-span-2"
        >
          <NetworkGraph
            data={networkData}
            width={800}
            height={600}
            showLabels={false}
            highlightPaths={isRunning}
          />
        </motion.div>
      </div>

      {/* Detailed Results Section (Conditional) */}
      <AnimatePresence>
        {showResults && results && (
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -40 }}
            transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
            className="space-y-8"
          >
            {/* Summary Metrics */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-50 mb-6">
                Simulation Results
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricsCard
                  title="Total Reach"
                  value={summaryMetrics.totalReach}
                  icon={<TrendingUp className="h-5 w-5 text-blue-600" />}
                  description="Total nodes reached"
                  animated
                />
                <MetricsCard
                  title="Platform Reputation"
                  value={summaryMetrics.finalReputation}
                  unit="%"
                  icon={<Target className="h-5 w-5 text-green-600" />}
                  description="Final reputation score"
                  animated
                />
                <MetricsCard
                  title="Detection Rate"
                  value={summaryMetrics.detectionRate}
                  unit="%"
                  icon={<BarChart3 className="h-5 w-5 text-purple-600" />}
                  description="Misinformation detection"
                  animated
                />
                <MetricsCard
                  title="Equilibrium Stability"
                  value={summaryMetrics.equilibriumStability * 100}
                  unit="%"
                  icon={<Activity className="h-5 w-5 text-yellow-600" />}
                  description="Nash equilibrium stability"
                  animated
                />
              </div>
            </motion.div>

            {/* Tabbed Results */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <Tabs defaultValue="summary" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="summary">Summary</TabsTrigger>
                  <TabsTrigger value="propagation">Propagation Timeline</TabsTrigger>
                  <TabsTrigger value="equilibrium">Game Outcome</TabsTrigger>
                </TabsList>

                {/* Summary Tab */}
                <TabsContent value="summary" className="space-y-6 mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Simulation Overview</CardTitle>
                      <CardDescription>High-level summary of simulation outcomes</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                          <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
                            Network Characteristics
                          </h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span>Network Size:</span>
                              <span className="font-mono">{gameParameters.network?.size || 1000}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Network Type:</span>
                              <span className="font-mono">{gameParameters.network?.type || 'small_world'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Avg Degree:</span>
                              <span className="font-mono">{gameParameters.network?.parameters?.averageDegree || 6}</span>
                            </div>
                          </div>
                        </div>

                        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                          <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
                            Agent Distribution
                          </h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span>Spreaders:</span>
                              <span className="font-mono">
                                {gameParameters.numPlayers?.spreaders || 10}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Fact-Checkers:</span>
                              <span className="font-mono">
                                {gameParameters.numPlayers?.factCheckers || 5}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Users:</span>
                              <span className="font-mono">
                                {gameParameters.numPlayers?.users || 100}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <Alert>
                        <CheckCircle2 className="h-4 w-4" />
                        <AlertTitle>Simulation Completed Successfully</AlertTitle>
                        <AlertDescription>
                          The simulation reached convergence after {gameParameters.dynamics?.timeHorizon || 50} time steps.
                          The network achieved a stable equilibrium with {(summaryMetrics.equilibriumStability * 100).toFixed(1)}% stability.
                        </AlertDescription>
                      </Alert>
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Propagation Timeline Tab */}
                <TabsContent value="propagation" className="space-y-6 mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Information Propagation Over Time</CardTitle>
                      <CardDescription>
                        Tracking of infected, susceptible, and recovered nodes throughout the simulation
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <LineChart
                        data={propagationData}
                        series={propagationSeries}
                        xAxisKey="step"
                      />
                      <div className="grid grid-cols-3 gap-4 mt-6">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800 text-center">
                          <div className="text-lg font-bold text-red-700 dark:text-red-400">
                            {propagationData[propagationData.length - 1]?.infected.toFixed(0) || 0}
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                            Final Infected
                          </div>
                        </div>
                        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-center">
                          <div className="text-lg font-bold text-gray-700 dark:text-gray-300">
                            {propagationData[propagationData.length - 1]?.susceptible.toFixed(0) || 0}
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                            Final Susceptible
                          </div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800 text-center">
                          <div className="text-lg font-bold text-green-700 dark:text-green-400">
                            {propagationData[propagationData.length - 1]?.recovered.toFixed(0) || 0}
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                            Final Recovered
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Game Outcome Tab */}
                <TabsContent value="equilibrium" className="space-y-6 mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Strategic Equilibrium Analysis</CardTitle>
                      <CardDescription>
                        Final Nash equilibrium and player payoffs from the game theory model
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <PayoffMatrix
                        data={mockPayoffMatrix}
                        title="Spreader vs. Moderator Game"
                        highlightEquilibrium
                      />
                      <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                          Equilibrium Interpretation
                        </h4>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          The simulation converged to a {mockPayoffMatrix.equilibrium.type} strategy Nash equilibrium
                          where spreaders adopt a <span className="font-semibold">Conservative</span> strategy and
                          moderators adopt a <span className="font-semibold">Lenient</span> policy. This equilibrium
                          exhibits {(mockPayoffMatrix.equilibrium.stability * 100).toFixed(0)}% stability, indicating
                          that neither player has an incentive to unilaterally deviate from their chosen strategy.
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}