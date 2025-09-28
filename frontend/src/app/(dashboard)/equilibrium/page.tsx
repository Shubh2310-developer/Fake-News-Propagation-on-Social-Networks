// frontend/src/app/(dashboard)/equilibrium/page.tsx

"use client";

import React, { useState } from 'react';
import { Calculator, TrendingUp, Users, Target, Zap, Settings, Download, Play, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { GameParametersForm } from '@/components/forms/GameParametersForm';
import { PayoffMatrix } from '@/components/game-theory/PayoffMatrix';
import { EquilibriumAnalysis } from '@/components/game-theory/EquilibriumAnalysis';
import { StrategyEvolution } from '@/components/game-theory/StrategyEvolution';
import { LineChart } from '@/components/charts/LineChart';
import { BarChart } from '@/components/charts/BarChart';

// Mock game theory data
const mockPayoffMatrices = {
  spreaderVsModerator: {
    title: "Spreader vs Moderator",
    players: ["Spreader", "Moderator"],
    strategies: [
      ["Aggressive", "Conservative"],
      ["Strict", "Lenient"]
    ],
    payoffs: [
      [[2.5, -1.8], [-0.5, 1.2]],  // Spreader Aggressive vs Moderator strategies
      [[1.0, -0.8], [0.8, 0.5]]    // Spreader Conservative vs Moderator strategies
    ],
    equilibrium: {
      strategies: [0, 1],
      payoffs: [1.2, -0.8],
      type: "Nash Equilibrium"
    }
  },
  spreaderVsUser: {
    title: "Spreader vs User",
    players: ["Spreader", "User"],
    strategies: [
      ["High Sophistication", "Low Sophistication"],
      ["Critical Thinking", "Passive Consumption"]
    ],
    payoffs: [
      [[3.2, -2.1], [1.8, -0.9]],  // High Sophistication vs User strategies
      [[2.1, -1.2], [0.9, -0.3]]   // Low Sophistication vs User strategies
    ],
    equilibrium: {
      strategies: [1, 0],
      payoffs: [2.1, -1.2],
      type: "Nash Equilibrium"
    }
  },
  moderatorVsUser: {
    title: "Moderator vs User",
    players: ["Moderator", "User"],
    strategies: [
      ["Proactive", "Reactive"],
      ["Report", "Ignore"]
    ],
    payoffs: [
      [[1.5, 1.2], [0.8, -0.5]],  // Proactive vs User strategies
      [[0.9, 0.8], [0.3, 0.1]]    // Reactive vs User strategies
    ],
    equilibrium: {
      strategies: [0, 0],
      payoffs: [1.5, 1.2],
      type: "Nash Equilibrium"
    }
  }
};

const mockEquilibriumHistory = [
  { round: 1, nashEquilibria: 2, paretoOptimal: 1, socialWelfare: 0.45 },
  { round: 5, nashEquilibria: 3, paretoOptimal: 2, socialWelfare: 0.52 },
  { round: 10, nashEquilibria: 2, paretoOptimal: 1, socialWelfare: 0.48 },
  { round: 15, nashEquilibria: 4, paretoOptimal: 2, socialWelfare: 0.61 },
  { round: 20, nashEquilibria: 3, paretoOptimal: 2, socialWelfare: 0.58 },
  { round: 25, nashEquilibria: 2, paretoOptimal: 1, socialWelfare: 0.55 },
];

const mockStrategyDistribution = [
  { strategy: "Aggressive Spreading", spreaders: 35, moderators: 0, users: 0 },
  { strategy: "Conservative Spreading", spreaders: 65, moderators: 0, users: 0 },
  { strategy: "Strict Moderation", moderators: 40, spreaders: 0, users: 0 },
  { strategy: "Lenient Moderation", moderators: 60, spreaders: 0, users: 0 },
  { strategy: "Critical Thinking", users: 30, spreaders: 0, moderators: 0 },
  { strategy: "Passive Consumption", users: 70, spreaders: 0, moderators: 0 },
];

const mockGameResults = [
  {
    id: 1,
    scenario: "High Network Density",
    nashCount: 3,
    dominantStrategy: "Conservative-Lenient",
    socialWelfare: 0.67,
    convergenceRounds: 12,
    status: "converged"
  },
  {
    id: 2,
    scenario: "Mixed Population",
    nashCount: 2,
    dominantStrategy: "Aggressive-Strict",
    socialWelfare: 0.43,
    convergenceRounds: 18,
    status: "converged"
  },
  {
    id: 3,
    scenario: "Learning Disabled",
    nashCount: 1,
    dominantStrategy: "Conservative-Strict",
    socialWelfare: 0.39,
    convergenceRounds: 25,
    status: "converged"
  },
  {
    id: 4,
    scenario: "High Noise Level",
    nashCount: 4,
    dominantStrategy: "Mixed Strategies",
    socialWelfare: 0.52,
    convergenceRounds: 30,
    status: "oscillating"
  },
];

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  unit?: string;
  icon: React.ReactNode;
  description: string;
  trend?: number;
}> = ({ title, value, unit, icon, description, trend }) => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-sm font-medium">{title}</CardTitle>
      {icon}
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold">
        {typeof value === 'number' ? value.toFixed(2) : value}
        {unit && <span className="text-sm font-normal text-slate-600 dark:text-slate-400 ml-1">{unit}</span>}
      </div>
      <p className="text-xs text-slate-600 dark:text-slate-400">{description}</p>
      {trend !== undefined && (
        <div className="flex items-center mt-2">
          <TrendingUp className={`w-3 h-3 mr-1 ${trend > 0 ? 'text-green-600' : 'text-red-600'}`} />
          <span className={`text-xs ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {Math.abs(trend)}% from last game
          </span>
        </div>
      )}
    </CardContent>
  </Card>
);

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'converged':
      return <Badge variant="default" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Converged</Badge>;
    case 'oscillating':
      return <Badge variant="default" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">Oscillating</Badge>;
    case 'divergent':
      return <Badge variant="destructive">Divergent</Badge>;
    default:
      return <Badge variant="secondary">{status}</Badge>;
  }
};

export default function EquilibriumPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedMatrix, setSelectedMatrix] = useState('spreaderVsModerator');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleRunAnalysis = () => {
    setIsAnalyzing(true);
    // Simulate analysis
    setTimeout(() => setIsAnalyzing(false), 3000);
  };

  const currentMatrix = mockPayoffMatrices[selectedMatrix as keyof typeof mockPayoffMatrices];

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-50">
          Game Theory Equilibrium Analysis
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">
          Analyze strategic interactions and equilibrium outcomes in misinformation propagation scenarios.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="payoffs">Payoff Matrices</TabsTrigger>
          <TabsTrigger value="analysis">Equilibrium Analysis</TabsTrigger>
          <TabsTrigger value="configuration">Configuration</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              title="Nash Equilibria"
              value={3}
              icon={<Target className="h-4 w-4 text-blue-500" />}
              description="Stable strategy combinations"
              trend={0}
            />
            <MetricCard
              title="Social Welfare"
              value={0.58}
              icon={<Users className="h-4 w-4 text-green-500" />}
              description="Overall system benefit"
              trend={5.2}
            />
            <MetricCard
              title="Convergence Rate"
              value="89%"
              icon={<TrendingUp className="h-4 w-4 text-purple-500" />}
              description="Games reaching equilibrium"
              trend={2.1}
            />
            <MetricCard
              title="Strategy Diversity"
              value={0.73}
              icon={<Zap className="h-4 w-4 text-orange-500" />}
              description="Strategic variation index"
              trend={-1.8}
            />
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Run Analysis</CardTitle>
                <CardDescription>Compute equilibria for current parameters</CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  onClick={handleRunAnalysis}
                  disabled={isAnalyzing}
                  className="w-full"
                >
                  <Calculator className="w-4 h-4 mr-2" />
                  {isAnalyzing ? 'Analyzing...' : 'Analyze Equilibria'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Game Simulation</CardTitle>
                <CardDescription>Run multi-round strategic game</CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">
                  <Play className="w-4 h-4 mr-2" />
                  Start Game
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Export Results</CardTitle>
                <CardDescription>Download analysis results</CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">
                  <Download className="w-4 h-4 mr-2" />
                  Export Data
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Strategy Evolution Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Strategy Evolution Over Time</CardTitle>
              <CardDescription>How player strategies adapt during gameplay</CardDescription>
            </CardHeader>
            <CardContent>
              <LineChart
                data={mockEquilibriumHistory}
                series={[
                  { dataKey: 'socialWelfare', name: 'Social Welfare', color: '#10b981' },
                  { dataKey: 'nashEquilibria', name: 'Nash Equilibria Count', color: '#3b82f6' },
                ]}
                xAxisKey="round"
                height={300}
              />
            </CardContent>
          </Card>

          {/* Recent Analysis Results */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Analysis Results</CardTitle>
              <CardDescription>Summary of latest equilibrium computations</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Scenario</TableHead>
                    <TableHead>Nash Equilibria</TableHead>
                    <TableHead>Dominant Strategy</TableHead>
                    <TableHead>Social Welfare</TableHead>
                    <TableHead>Convergence</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {mockGameResults.map((result) => (
                    <TableRow key={result.id}>
                      <TableCell className="font-medium">{result.scenario}</TableCell>
                      <TableCell>{result.nashCount}</TableCell>
                      <TableCell className="text-sm">{result.dominantStrategy}</TableCell>
                      <TableCell>
                        <span className="font-mono">{result.socialWelfare.toFixed(2)}</span>
                      </TableCell>
                      <TableCell>
                        <span className="text-sm">{result.convergenceRounds} rounds</span>
                      </TableCell>
                      <TableCell>
                        {getStatusBadge(result.status)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="payoffs" className="space-y-6">
          {/* Matrix Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Payoff Matrix Selection</CardTitle>
              <CardDescription>Choose the strategic interaction to analyze</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(mockPayoffMatrices).map(([key, matrix]) => (
                  <Button
                    key={key}
                    variant={selectedMatrix === key ? "default" : "outline"}
                    onClick={() => setSelectedMatrix(key)}
                    className="h-auto p-4 flex flex-col items-center"
                  >
                    <div className="font-medium">{matrix.title}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                      {matrix.players.join(" vs ")}
                    </div>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Selected Payoff Matrix */}
          <Card>
            <CardHeader>
              <CardTitle>{currentMatrix.title} - Payoff Matrix</CardTitle>
              <CardDescription>
                Strategic payoffs for {currentMatrix.players.join(" vs ")} interactions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <PayoffMatrix data={currentMatrix} highlightEquilibrium={true} />
            </CardContent>
          </Card>

          {/* Strategy Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Current Strategy Distribution</CardTitle>
              <CardDescription>How players are currently allocating their strategies</CardDescription>
            </CardHeader>
            <CardContent>
              <BarChart
                data={mockStrategyDistribution}
                series={[
                  { dataKey: 'spreaders', name: 'Spreaders', color: '#ef4444' },
                  { dataKey: 'moderators', name: 'Moderators', color: '#3b82f6' },
                  { dataKey: 'users', name: 'Users', color: '#10b981' },
                ]}
                xAxisKey="strategy"
                height={300}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-6">
          {/* Equilibrium Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Nash Equilibrium Analysis</CardTitle>
                <CardDescription>Detailed analysis of stable strategy profiles</CardDescription>
              </CardHeader>
              <CardContent>
                <EquilibriumAnalysis data={currentMatrix} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Strategy Evolution</CardTitle>
                <CardDescription>How strategies change over multiple rounds</CardDescription>
              </CardHeader>
              <CardContent>
                <StrategyEvolution data={mockEquilibriumHistory} />
              </CardContent>
            </Card>
          </div>

          {/* Detailed Equilibrium Properties */}
          <Card>
            <CardHeader>
              <CardTitle>Equilibrium Properties</CardTitle>
              <CardDescription>Mathematical properties of the current equilibrium</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-medium">Stability Analysis</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Nash Equilibrium:</span>
                      <Badge variant="default" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                        Stable
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Evolutionary Stable:</span>
                      <Badge variant="default" className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                        Yes
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Pareto Optimal:</span>
                      <Badge variant="default" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                        No
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Correlated Equilibrium:</span>
                      <Badge variant="default" className="bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                        Exists
                      </Badge>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-medium">Welfare Analysis</h4>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Social Welfare</span>
                        <span className="font-mono">0.58</span>
                      </div>
                      <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{ width: '58%' }}></div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Efficiency Loss</span>
                        <span className="font-mono">0.23</span>
                      </div>
                      <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                        <div className="bg-red-500 h-2 rounded-full" style={{ width: '23%' }}></div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Fairness Index</span>
                        <span className="font-mono">0.71</span>
                      </div>
                      <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                        <div className="bg-blue-500 h-2 rounded-full" style={{ width: '71%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t">
                <h4 className="font-medium mb-4">Mixed Strategy Probabilities</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="text-sm font-medium mb-2">Spreader Strategies</h5>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Aggressive:</span>
                        <span className="font-mono">0.35 (35%)</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Conservative:</span>
                        <span className="font-mono">0.65 (65%)</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h5 className="text-sm font-medium mb-2">Moderator Strategies</h5>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Strict:</span>
                        <span className="font-mono">0.40 (40%)</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Lenient:</span>
                        <span className="font-mono">0.60 (60%)</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="configuration" className="space-y-6">
          {/* Game Parameters Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Game Theory Parameters
              </CardTitle>
              <CardDescription>
                Configure the strategic game parameters and equilibrium analysis settings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <GameParametersForm
                onSubmit={(parameters) => {
                  console.log('Game parameters:', parameters);
                  handleRunAnalysis();
                }}
                isLoading={isAnalyzing}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}