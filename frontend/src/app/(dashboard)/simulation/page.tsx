// frontend/src/app/(dashboard)/simulation/page.tsx

"use client";

import React, { useState } from 'react';
import { Play, Pause, Square, RotateCcw, Download, Settings, Network, TrendingUp, Users, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { SimulationConfigForm } from '@/components/forms/SimulationConfigForm';
import { NetworkVisualization } from '@/components/charts/NetworkVisualization';
import { LineChart } from '@/components/charts/LineChart';
import { BarChart } from '@/components/charts/BarChart';
import { PropagationMetrics } from '@/components/simulation/PropagationMetrics';
import { SimulationControls } from '@/components/simulation/SimulationControls';

// Mock simulation data
const mockSimulationState = {
  isRunning: false,
  isPaused: false,
  currentStep: 0,
  totalSteps: 100,
  progress: 0,
  elapsedTime: '00:00:00',
  estimatedTimeRemaining: '00:05:30',
};

const mockNetworkData = {
  nodes: [
    { id: '1', type: 'user', status: 'susceptible', x: 100, y: 100 },
    { id: '2', type: 'spreader', status: 'infected', x: 200, y: 150 },
    { id: '3', type: 'moderator', status: 'immune', x: 150, y: 200 },
    { id: '4', type: 'user', status: 'infected', x: 250, y: 100 },
    { id: '5', type: 'user', status: 'recovered', x: 300, y: 180 },
  ],
  links: [
    { source: '1', target: '2', strength: 0.8 },
    { source: '2', target: '3', strength: 0.6 },
    { source: '2', target: '4', strength: 0.9 },
    { source: '3', target: '5', strength: 0.7 },
    { source: '4', target: '5', strength: 0.5 },
  ]
};

const mockPropagationData = [
  { step: 0, susceptible: 95, infected: 5, recovered: 0, immune: 0 },
  { step: 10, susceptible: 78, infected: 15, recovered: 5, immune: 2 },
  { step: 20, susceptible: 62, infected: 25, recovered: 10, immune: 3 },
  { step: 30, susceptible: 45, infected: 35, recovered: 15, immune: 5 },
  { step: 40, susceptible: 30, infected: 30, recovered: 32, immune: 8 },
  { step: 50, susceptible: 20, infected: 20, recovered: 50, immune: 10 },
];

const mockMetrics = {
  peakInfection: 38.5,
  finalRecovered: 72.3,
  propagationRate: 0.34,
  containmentEffectiveness: 0.67,
  networkDensity: 0.25,
  clusteringCoefficient: 0.42,
};

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
        {typeof value === 'number' ? value.toFixed(1) : value}
        {unit && <span className="text-sm font-normal text-slate-600 dark:text-slate-400 ml-1">{unit}</span>}
      </div>
      <p className="text-xs text-slate-600 dark:text-slate-400">{description}</p>
      {trend !== undefined && (
        <div className="flex items-center mt-2">
          <TrendingUp className={`w-3 h-3 mr-1 ${trend > 0 ? 'text-green-600' : 'text-red-600'}`} />
          <span className={`text-xs ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {Math.abs(trend)}% from last run
          </span>
        </div>
      )}
    </CardContent>
  </Card>
);

export default function SimulationPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [simulationState, setSimulationState] = useState(mockSimulationState);
  const [playbackSpeed, setPlaybackSpeed] = useState([1]);
  const [showLabels, setShowLabels] = useState(true);
  const [highlightPaths, setHighlightPaths] = useState(false);

  const handleStartSimulation = () => {
    setSimulationState(prev => ({ ...prev, isRunning: true, isPaused: false }));
    // Simulate progress
    const interval = setInterval(() => {
      setSimulationState(prev => {
        const newStep = prev.currentStep + 1;
        const newProgress = (newStep / prev.totalSteps) * 100;

        if (newStep >= prev.totalSteps) {
          clearInterval(interval);
          return {
            ...prev,
            isRunning: false,
            currentStep: prev.totalSteps,
            progress: 100
          };
        }

        return {
          ...prev,
          currentStep: newStep,
          progress: newProgress
        };
      });
    }, 100 / playbackSpeed[0]);
  };

  const handlePauseSimulation = () => {
    setSimulationState(prev => ({ ...prev, isPaused: !prev.isPaused }));
  };

  const handleStopSimulation = () => {
    setSimulationState(prev => ({
      ...prev,
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      progress: 0
    }));
  };

  const handleResetSimulation = () => {
    setSimulationState(prev => ({
      ...prev,
      isRunning: false,
      isPaused: false,
      currentStep: 0,
      progress: 0
    }));
  };

  const propagationSeries = [
    { dataKey: 'susceptible', name: 'Susceptible', color: '#94a3b8' },
    { dataKey: 'infected', name: 'Infected', color: '#ef4444' },
    { dataKey: 'recovered', name: 'Recovered', color: '#10b981' },
    { dataKey: 'immune', name: 'Immune', color: '#3b82f6' },
  ];

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-50">
          Misinformation Propagation Simulation
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">
          Model and analyze how misinformation spreads through social networks.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="network">Network View</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="configuration">Configuration</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Simulation Controls */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Simulation Control Panel
              </CardTitle>
              <CardDescription>
                Control the simulation execution and monitor progress
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span>{simulationState.currentStep}/{simulationState.totalSteps} steps</span>
                  </div>
                  <Progress value={simulationState.progress} className="h-2" />
                  <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400">
                    <span>Elapsed: {simulationState.elapsedTime}</span>
                    <span>Remaining: {simulationState.estimatedTimeRemaining}</span>
                  </div>
                </div>

                {/* Control Buttons */}
                <div className="flex gap-3">
                  <Button
                    onClick={handleStartSimulation}
                    disabled={simulationState.isRunning}
                    className="flex items-center gap-2"
                  >
                    <Play className="w-4 h-4" />
                    Start
                  </Button>

                  <Button
                    onClick={handlePauseSimulation}
                    disabled={!simulationState.isRunning}
                    variant="outline"
                    className="flex items-center gap-2"
                  >
                    <Pause className="w-4 h-4" />
                    {simulationState.isPaused ? 'Resume' : 'Pause'}
                  </Button>

                  <Button
                    onClick={handleStopSimulation}
                    disabled={!simulationState.isRunning && !simulationState.isPaused}
                    variant="outline"
                    className="flex items-center gap-2"
                  >
                    <Square className="w-4 h-4" />
                    Stop
                  </Button>

                  <Button
                    onClick={handleResetSimulation}
                    variant="outline"
                    className="flex items-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Reset
                  </Button>

                  <Button variant="outline" className="flex items-center gap-2 ml-auto">
                    <Download className="w-4 h-4" />
                    Export Results
                  </Button>
                </div>

                {/* Playback Controls */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t">
                  <div className="space-y-2">
                    <Label>Playback Speed</Label>
                    <div className="space-y-2">
                      <Slider
                        value={playbackSpeed}
                        onValueChange={setPlaybackSpeed}
                        min={0.1}
                        max={5}
                        step={0.1}
                      />
                      <div className="text-sm text-slate-600 dark:text-slate-400 text-center">
                        {playbackSpeed[0]}x speed
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="show-labels"
                        checked={showLabels}
                        onCheckedChange={setShowLabels}
                      />
                      <Label htmlFor="show-labels">Show Node Labels</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="highlight-paths"
                        checked={highlightPaths}
                        onCheckedChange={setHighlightPaths}
                      />
                      <Label htmlFor="highlight-paths">Highlight Propagation Paths</Label>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Current Status</Label>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span>State:</span>
                        <Badge variant={simulationState.isRunning ? "default" : "secondary"}>
                          {simulationState.isRunning
                            ? (simulationState.isPaused ? 'Paused' : 'Running')
                            : 'Stopped'
                          }
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Step:</span>
                        <span className="font-mono">{simulationState.currentStep}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              title="Peak Infection"
              value={mockMetrics.peakInfection}
              unit="%"
              icon={<TrendingUp className="h-4 w-4 text-red-500" />}
              description="Maximum infected population"
              trend={-2.3}
            />
            <MetricCard
              title="Final Recovery"
              value={mockMetrics.finalRecovered}
              unit="%"
              icon={<Users className="h-4 w-4 text-green-500" />}
              description="Total recovered population"
              trend={5.7}
            />
            <MetricCard
              title="Propagation Rate"
              value={mockMetrics.propagationRate}
              icon={<Network className="h-4 w-4 text-blue-500" />}
              description="Average spread velocity"
              trend={-1.2}
            />
            <MetricCard
              title="Containment Effectiveness"
              value={`${(mockMetrics.containmentEffectiveness * 100).toFixed(0)}%`}
              icon={<Activity className="h-4 w-4 text-purple-500" />}
              description="Moderation success rate"
              trend={8.1}
            />
          </div>

          {/* Propagation Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Population Dynamics</CardTitle>
              <CardDescription>
                Evolution of different population states over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <LineChart
                data={mockPropagationData}
                series={propagationSeries}
                xAxisKey="step"
                height={300}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="space-y-6">
          {/* Network Visualization */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Network className="w-5 h-5" />
                Network Visualization
              </CardTitle>
              <CardDescription>
                Interactive view of the social network and information propagation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[500px] border rounded-lg bg-slate-50 dark:bg-slate-900">
                <NetworkVisualization
                  data={mockNetworkData}
                  showLabels={showLabels}
                  highlightPaths={highlightPaths}
                />
              </div>
            </CardContent>
          </Card>

          {/* Network Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Network Properties</CardTitle>
                <CardDescription>Structural characteristics of the network</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-slate-600 dark:text-slate-400">Nodes:</span>
                    <span className="ml-2 font-mono font-bold">{mockNetworkData.nodes.length}</span>
                  </div>
                  <div>
                    <span className="text-slate-600 dark:text-slate-400">Edges:</span>
                    <span className="ml-2 font-mono font-bold">{mockNetworkData.links.length}</span>
                  </div>
                  <div>
                    <span className="text-slate-600 dark:text-slate-400">Density:</span>
                    <span className="ml-2 font-mono font-bold">{mockMetrics.networkDensity}</span>
                  </div>
                  <div>
                    <span className="text-slate-600 dark:text-slate-400">Clustering:</span>
                    <span className="ml-2 font-mono font-bold">{mockMetrics.clusteringCoefficient}</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium">Node Types Distribution</Label>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span>Regular Users</span>
                      <span>60%</span>
                    </div>
                    <Progress value={60} className="h-1" />

                    <div className="flex justify-between text-xs">
                      <span>Spreaders</span>
                      <span>20%</span>
                    </div>
                    <Progress value={20} className="h-1" />

                    <div className="flex justify-between text-xs">
                      <span>Moderators</span>
                      <span>20%</span>
                    </div>
                    <Progress value={20} className="h-1" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Current State Distribution</CardTitle>
                <CardDescription>Population breakdown by infection status</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 border rounded-lg">
                      <div className="text-lg font-bold text-slate-600">45</div>
                      <div className="text-xs text-slate-500">Susceptible</div>
                    </div>
                    <div className="text-center p-3 border rounded-lg">
                      <div className="text-lg font-bold text-red-600">12</div>
                      <div className="text-xs text-slate-500">Infected</div>
                    </div>
                    <div className="text-center p-3 border rounded-lg">
                      <div className="text-lg font-bold text-green-600">30</div>
                      <div className="text-xs text-slate-500">Recovered</div>
                    </div>
                    <div className="text-center p-3 border rounded-lg">
                      <div className="text-lg font-bold text-blue-600">13</div>
                      <div className="text-xs text-slate-500">Immune</div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span>Infection Rate</span>
                      <span>12%</span>
                    </div>
                    <Progress value={12} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Detailed Propagation Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Propagation Analysis</CardTitle>
                <CardDescription>Detailed metrics on information spread patterns</CardDescription>
              </CardHeader>
              <CardContent>
                <PropagationMetrics data={mockPropagationData} />
              </CardContent>
            </Card>

            {/* Agent Performance */}
            <Card>
              <CardHeader>
                <CardTitle>Agent Performance</CardTitle>
                <CardDescription>Effectiveness of different agent types</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Spreader Success Rate</span>
                      <span className="text-sm font-mono">73%</span>
                    </div>
                    <Progress value={73} />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Moderator Detection Rate</span>
                      <span className="text-sm font-mono">67%</span>
                    </div>
                    <Progress value={67} />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">User Resistance Rate</span>
                      <span className="text-sm font-mono">45%</span>
                    </div>
                    <Progress value={45} />
                  </div>

                  <div className="pt-4 border-t">
                    <h4 className="font-medium mb-2">Agent Strategy Distribution</h4>
                    <BarChart
                      data={[
                        { strategy: 'Aggressive', count: 25 },
                        { strategy: 'Moderate', count: 45 },
                        { strategy: 'Conservative', count: 30 },
                      ]}
                      series={[{ dataKey: 'count', name: 'Agents', color: '#3b82f6' }]}
                      xAxisKey="strategy"
                      height={200}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Time Series Analysis */}
          <Card>
            <CardHeader>
              <CardTitle>Transmission Dynamics</CardTitle>
              <CardDescription>Rate of information transmission over time</CardDescription>
            </CardHeader>
            <CardContent>
              <LineChart
                data={mockPropagationData.map((d, i) => ({
                  ...d,
                  transmissionRate: Math.max(0, (d.infected - (i > 0 ? mockPropagationData[i-1].infected : 0)) / 10),
                  detectionRate: Math.max(0, (d.recovered - (i > 0 ? mockPropagationData[i-1].recovered : 0)) / 10),
                }))}
                series={[
                  { dataKey: 'transmissionRate', name: 'Transmission Rate', color: '#ef4444' },
                  { dataKey: 'detectionRate', name: 'Detection Rate', color: '#10b981' },
                ]}
                xAxisKey="step"
                height={300}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="configuration" className="space-y-6">
          {/* Simulation Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Simulation Parameters
              </CardTitle>
              <CardDescription>
                Configure the simulation settings and parameters
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SimulationConfigForm
                onSubmit={(config) => {
                  console.log('Simulation config:', config);
                  handleResetSimulation();
                }}
                isLoading={simulationState.isRunning}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}