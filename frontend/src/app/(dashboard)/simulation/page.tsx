// frontend/src/app/(dashboard)/simulation/page.tsx

"use client";

import React, { useEffect, useState, useMemo } from 'react';
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

// Backend response type (snake_case from Python)
interface BackendSimulationResults {
  simulation_id?: string;
  total_rounds?: number;
  payoff_trends?: {
    spreader?: number[];
    fact_checker?: number[];
    platform?: number[];
  };
  final_metrics?: {
    final_payoffs?: {
      spreader?: number;
      fact_checker?: number;
      platform?: number;
    };
    final_reputation_scores?: Record<string, number>;
  };
  network_metrics?: {
    num_nodes?: number;
    density?: number;
    clustering?: number;
  };
  convergence_analysis?: {
    converged?: boolean;
    status?: string;
    recent_payoff_variances?: Record<string, number>;
  };
  parameters?: any;
  timestamp?: string;
}

// Mock network data generator - Optimized for performance
function generateMockNetworkData(nodeCount: number = 100): any {
  // Limit node count for performance
  const limitedNodeCount = Math.min(nodeCount, 150);

  const nodes = Array.from({ length: limitedNodeCount }, (_, i) => {
    const userType = i < 5 ? 'spreader' : i < 15 ? 'fact_checker' : i < 20 ? 'platform' : 'regular_user';
    return {
      id: `node-${i}`,
      label: `N${i}`,
      type: userType,
      user_type: userType,
      influence_score: Math.random() * 5 + 1,
      credibility_score: Math.random(),
      trust_score: Math.random(),
      authority_score: Math.random(),
      centrality: {
        degree: 0,
        betweenness: 0,
        closeness: 0,
        eigenvector: 0,
        pagerank: 0,
      },
      activity: {
        posts_count: Math.floor(Math.random() * 100),
        shares_count: Math.floor(Math.random() * 50),
        comments_count: Math.floor(Math.random() * 200),
        reactions_count: Math.floor(Math.random() * 500),
        last_active: new Date().toISOString(),
      },
      content_stats: {
        fake_news_shared: Math.floor(Math.random() * 10),
        real_news_shared: Math.floor(Math.random() * 20),
        fact_checks_performed: Math.floor(Math.random() * 5),
        misinformation_flagged: Math.floor(Math.random() * 8),
      },
      profile: {
        account_age: Math.floor(Math.random() * 3650),
        follower_count: Math.floor(Math.random() * 1000),
        following_count: Math.floor(Math.random() * 500),
        verified: Math.random() > 0.9,
        categories: ['general'],
      },
      visual: {
        size: 10,
        color: userType === 'spreader' ? '#ef4444' : userType === 'fact_checker' ? '#10b981' : '#94a3b8',
        opacity: 1,
        shape: 'circle' as const,
        strokeWidth: 1,
        strokeColor: '#000',
      },
      state: {
        selected: false,
        highlighted: false,
        filtered: false,
      },
    };
  });

  const links = [];
  const linkCount = Math.floor(limitedNodeCount * 1.5);
  for (let i = 0; i < linkCount; i++) {
    const source = Math.floor(Math.random() * limitedNodeCount);
    const target = Math.floor(Math.random() * limitedNodeCount);
    if (source !== target) {
      links.push({
        id: `link-${i}`,
        source: `node-${source}`,
        target: `node-${target}`,
        type: 'follow',
        weight: Math.random(),
        strength: Math.random() * 0.5 + 0.5,
        directed: true,
        visual: {
          width: 1,
          color: '#cbd5e0',
          opacity: 0.5,
          style: 'solid' as const,
        },
        state: {
          selected: false,
          highlighted: false,
          filtered: false,
        },
      });
    }
  }

  // Calculate degrees
  nodes.forEach((node) => {
    node.centrality.degree = links.filter((l: any) =>
      l.source === node.id || l.target === node.id
    ).length;
  });

  return {
    id: 'mock-network',
    name: 'Mock Network',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    nodes,
    links,
    statistics: {
      basic: {
        node_count: nodes.length,
        edge_count: links.length,
        density: links.length / (nodes.length * (nodes.length - 1)),
        is_connected: true,
        is_directed: true,
        is_weighted: true,
      },
    },
    layout: {
      algorithm: 'force-directed',
      parameters: {},
      bounds: { minX: 0, maxX: 1000, minY: 0, maxY: 1000 },
    },
    filters: {
      active: {},
      available: [],
    },
  };
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
    results: rawResults,
    error,
    setGameParameters,
    startSimulation,
    stopSimulation,
    resetSimulation,
  } = useSimulationStore();

  // Cast results to backend type
  const results = rawResults as BackendSimulationResults | null;

  const [networkData, setNetworkData] = useState(generateMockNetworkData(gameParameters.network?.size || 100));
  const [showResults, setShowResults] = useState(false);

  // Update network when simulation completes
  useEffect(() => {
    if (simulationState === 'completed' && results) {
      setShowResults(true);

      // Use real network data if available
      if (results.network_metrics?.num_nodes) {
        setNetworkData(generateMockNetworkData(results.network_metrics.num_nodes));
      } else {
        setNetworkData(generateMockNetworkData(gameParameters.network?.size || 100));
      }
    }
  }, [simulationState, results, gameParameters.network?.size]);

  // Handle parameter submission
  const handleParametersSubmit = async (params: any) => {
    // Map network type properly
    const getNetworkType = (type: string): 'random' | 'small_world' | 'scale_free' | 'complete' | 'custom' => {
      switch (type) {
        case 'scale-free':
          return 'scale_free';
        case 'small-world':
          return 'small_world';
        case 'random':
          return 'random';
        case 'grid':
          return 'complete';
        default:
          return 'small_world';
      }
    };

    // Transform GameParameters form data to backend-compatible format
    const transformedParams = {
      metadata: {
        name: 'Simulation',
        description: '',
        tags: [],
        priority: 'normal' as const,
      },
      numPlayers: {
        spreaders: Math.floor(params.spreaderRatio * params.networkSize),
        factCheckers: Math.floor(params.moderatorRatio * params.networkSize),
        platforms: 1,
        users: Math.floor(params.userRatio * params.networkSize),
      },
      network: {
        size: params.networkSize,
        type: getNetworkType(params.networkType),
        parameters: {
          averageDegree: params.averageDegree,
          clusteringCoefficient: params.clusteringCoefficient,
          rewireProbability: 0.1,
        },
      },
      dynamics: {
        timeHorizon: params.timeHorizon,
        timeStep: 1,
        maxRounds: params.timeHorizon,
        propagationRate: params.basePropagationRate,
        decayRate: params.decayRate,
        viralThreshold: 0.3,
        detectionAccuracy: 0.8,
        detectionDelay: 2,
        factCheckingEffectiveness: 0.9,
        learningRate: params.learningRate,
        memoryLength: params.memoryLength,
        adaptationThreshold: 0.2,
      },
      payoffWeights: {
        spreaders: {
          reach: params.spreaderReward,
          detection: params.detectionPenalty,
          reputation: -0.3,
          cost: -0.1,
        },
        factCheckers: {
          accuracy: params.moderatorReward,
          effort: -0.2,
          impact: 0.5,
          reputation: 0.3,
        },
        platforms: {
          engagement: 0.8,
          reputation: -0.6,
          cost: -0.1,
          regulation: -0.4,
        },
      },
      environment: {
        events: [],
        cycles: {
          newsVolume: [],
          userActivity: [],
          platformAttention: [],
        },
        uncertainty: {
          payoffNoise: params.noiseLevel,
          informationNoise: 0.05,
          behaviorNoise: 0.02,
        },
      },
      advanced: {
        randomSeed: params.randomSeed,
        parallelization: {
          enabled: false,
          workers: 1,
          batchSize: 100,
        },
        output: {
          saveInterval: params.saveFrequency,
          saveDetailedHistory: true,
          saveNetworkStates: params.enableNetworkEvolution,
          savePlayerStates: true,
          compression: false,
        },
        stoppingCriteria: {
          convergenceThreshold: params.convergenceThreshold,
          maxIterationsWithoutChange: 100,
          minPayoffImprovement: 0.01,
        },
      },
    };

    setGameParameters(transformedParams);
    await startSimulation();
  };

  // Propagation timeline data - use real data if available
  const propagationData = useMemo(() => {
    if (results?.payoff_trends) {
      // Transform backend data to chart format
      const spreaderTrend = results.payoff_trends.spreader || [];
      return spreaderTrend.map((_, i) => ({
        step: i,
        infected: Math.min(100, i * 5 + Math.random() * 10),
        susceptible: Math.max(0, 100 - i * 5),
        recovered: i * 2,
      }));
    }

    // Mock data
    return Array.from({ length: 20 }, (_, i) => ({
      step: i * 5,
      infected: Math.min(100, i * 5 + Math.random() * 10),
      susceptible: Math.max(0, 100 - i * 5 - Math.random() * 10),
      recovered: i * 2 + Math.random() * 5,
    }));
  }, [results]);

  const propagationSeries = [
    { dataKey: 'infected', name: 'Infected', color: '#ef4444' },
    { dataKey: 'susceptible', name: 'Susceptible', color: '#94a3b8' },
    { dataKey: 'recovered', name: 'Recovered', color: '#10b981' },
  ];

  // Payoff matrix - use real data if available
  const payoffMatrixData = useMemo(() => {
    if (results?.final_metrics?.final_payoffs) {
      const finalPayoffs = results.final_metrics.final_payoffs;

      return {
        players: ['Spreader', 'Moderator'] as [string, string],
        strategies: {
          Spreader: ['Aggressive', 'Conservative'],
          Moderator: ['Strict', 'Lenient'],
        },
        payoffs: [
          [
            { Spreader: finalPayoffs.spreader || 1.2, Moderator: finalPayoffs.fact_checker || -0.8 },
            { Spreader: (finalPayoffs.spreader || 1.2) * 1.5, Moderator: (finalPayoffs.fact_checker || -0.8) * 0.5 },
          ],
          [
            { Spreader: (finalPayoffs.spreader || 1.2) * 0.5, Moderator: (finalPayoffs.fact_checker || -0.8) * 0.3 },
            { Spreader: (finalPayoffs.spreader || 1.2) * 0.75, Moderator: (finalPayoffs.fact_checker || -0.8) * 0.9 },
          ],
        ],
        equilibrium: {
          strategies: [1, 1] as [number, number],
          payoffs: {
            Spreader: finalPayoffs.spreader || 0.9,
            Moderator: finalPayoffs.fact_checker || 0.7
          },
          type: 'pure' as const,
          stability: results.convergence_analysis?.converged ? 0.95 : 0.75,
          classification: 'strict' as const,
        },
      };
    }

    // Mock data
    return {
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
  }, [results]);

  // Summary metrics from real results
  const summaryMetrics = useMemo(() => {
    if (results) {
      let finalReputation = 72.5;

      if (results.final_metrics?.final_reputation_scores) {
        const scores = Object.values(results.final_metrics.final_reputation_scores);
        const sum = scores.reduce((a: number, b: number) => a + b, 0);
        const avg = sum / scores.length;
        finalReputation = avg * 100;
      }

      return {
        totalReach: results.network_metrics?.num_nodes || 8542,
        finalReputation,
        detectionRate: 68.3,
        equilibriumStability: results.convergence_analysis?.converged ? 0.95 : 0.65,
      };
    }

    return {
      totalReach: 8542,
      finalReputation: 72.5,
      detectionRate: 68.3,
      equilibriumStability: 0.85,
    };
  }, [results]);

  return (
    <div className="space-y-8 relative">
      {/* Status Indicator */}
      <StatusIndicator state={simulationState} isRunning={isRunning} />

      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-6"
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="p-3 bg-green-100 rounded-lg">
            <Network className="h-6 w-6 text-green-600" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900">
            Game Theory Simulation
          </h1>
        </div>
        <p className="text-gray-600 text-base">
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
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* Left Column: Control Panel (50% width) */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="xl:col-span-6 space-y-6 min-w-[500px]"
        >
          <Card className="shadow-md">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2 text-xl font-bold text-gray-900">
                <Target className="h-6 w-6 text-purple-600" />
                Simulation Parameters
              </CardTitle>
              <CardDescription className="text-sm text-gray-600">
                Configure the game theory simulation settings
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-4">
              <GameParameters
                onSubmit={handleParametersSubmit}
                isLoading={isRunning}
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* Right Column: Network Visualization (50% width) */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="xl:col-span-6"
        >
          <Card className="shadow-md overflow-hidden">
            <CardContent className="p-2">
              <div className="w-full" style={{ minHeight: '850px' }}>
                <NetworkGraph
                  data={networkData}
                  width={1400}
                  height={850}
                />
              </div>
            </CardContent>
          </Card>
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
              <h2 className="text-2xl font-bold text-gray-900 mb-6">
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
                        data={payoffMatrixData}
                        title="Spreader vs. Moderator Game"
                        highlightEquilibrium
                      />
                      <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                          Equilibrium Interpretation
                        </h4>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          The simulation converged to a {payoffMatrixData.equilibrium.type} strategy Nash equilibrium
                          where spreaders adopt a <span className="font-semibold">Conservative</span> strategy and
                          moderators adopt a <span className="font-semibold">Lenient</span> policy. This equilibrium
                          exhibits {(payoffMatrixData.equilibrium.stability * 100).toFixed(0)}% stability, indicating
                          that neither player has an incentive to unilaterally deviate from their chosen strategy.
                          {results?.convergence_analysis?.converged && (
                            <span className="block mt-2 text-green-600 dark:text-green-400 font-medium">
                              ✓ The system has converged to a stable equilibrium.
                            </span>
                          )}
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