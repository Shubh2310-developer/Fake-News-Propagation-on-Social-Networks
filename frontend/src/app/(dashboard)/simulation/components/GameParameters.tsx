// frontend/src/app/(dashboard)/simulation/components/GameParameters.tsx

"use client";

import React, { useState } from 'react';
import { Info, RotateCcw, Play, Save, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

// Game parameters interface
interface GameParameters {
  // Network Configuration
  networkSize: number;
  networkType: 'scale-free' | 'small-world' | 'random' | 'grid';
  averageDegree: number;
  clusteringCoefficient: number;

  // Agent Configuration
  spreaderRatio: number;
  moderatorRatio: number;
  userRatio: number;
  botRatio: number;

  // Propagation Parameters
  basePropagationRate: number;
  decayRate: number;
  recoveryRate: number;
  immunityRate: number;

  // Game Theory Parameters
  spreaderReward: number;
  moderatorReward: number;
  detectionPenalty: number;
  falsePositivePenalty: number;
  learningRate: number;
  adaptationFrequency: number;

  // Simulation Settings
  timeHorizon: number;
  randomSeed?: number;
  enableLearning: boolean;
  enableNetworkEvolution: boolean;
  saveFrequency: number;

  // Advanced Settings
  noiseLevel: number;
  memoryLength: number;
  explorationRate: number;
  convergenceThreshold: number;
}

interface GameParametersProps {
  initialValues?: Partial<GameParameters>;
  onSubmit: (parameters: GameParameters) => void;
  onSave?: (parameters: GameParameters) => void;
  onLoad?: () => void;
  onReset?: () => void;
  isLoading?: boolean;
  className?: string;
}

// Default parameter values
const DEFAULT_PARAMETERS: GameParameters = {
  // Network Configuration
  networkSize: 1000,
  networkType: 'scale-free',
  averageDegree: 6,
  clusteringCoefficient: 0.3,

  // Agent Configuration
  spreaderRatio: 0.05,
  moderatorRatio: 0.1,
  userRatio: 0.8,
  botRatio: 0.05,

  // Propagation Parameters
  basePropagationRate: 0.1,
  decayRate: 0.05,
  recoveryRate: 0.08,
  immunityRate: 0.02,

  // Game Theory Parameters
  spreaderReward: 2.0,
  moderatorReward: 1.5,
  detectionPenalty: -1.0,
  falsePositivePenalty: -0.5,
  learningRate: 0.1,
  adaptationFrequency: 5,

  // Simulation Settings
  timeHorizon: 100,
  enableLearning: true,
  enableNetworkEvolution: false,
  saveFrequency: 10,

  // Advanced Settings
  noiseLevel: 0.1,
  memoryLength: 10,
  explorationRate: 0.1,
  convergenceThreshold: 0.001,
};

// Form section component with consistent styling
const FormSection: React.FC<{
  title: string;
  description?: string;
  children: React.ReactNode;
  icon?: React.ReactNode;
}> = ({ title, description, children, icon }) => (
  <Card className="mb-6">
    <CardHeader>
      <CardTitle className="text-lg flex items-center gap-2">
        {icon}
        {title}
      </CardTitle>
      {description && <CardDescription>{description}</CardDescription>}
    </CardHeader>
    <CardContent className="space-y-6">
      {children}
    </CardContent>
  </Card>
);

// Form field component with tooltip support
const FormField: React.FC<{
  label: string;
  tooltip?: string;
  children: React.ReactNode;
  className?: string;
  required?: boolean;
}> = ({
  label,
  tooltip,
  children,
  className = "grid grid-cols-1 md:grid-cols-2 items-center gap-4",
  required = false
}) => (
  <div className={className}>
    <Label className="flex items-center text-sm font-medium">
      {label}
      {required && <span className="text-red-500 ml-1">*</span>}
      {tooltip && (
        <div className="relative group ml-2">
          <Info className="w-4 h-4 text-slate-400 hover:text-slate-600 cursor-help" />
          <div className="absolute left-0 top-6 z-10 w-64 p-3 bg-slate-900 text-white text-xs rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            {tooltip}
          </div>
        </div>
      )}
    </Label>
    <div>{children}</div>
  </div>
);

// Slider field component with value display
const SliderField: React.FC<{
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  tooltip?: string;
  unit?: string;
  format?: 'number' | 'percentage';
}> = ({ label, value, onChange, min, max, step, tooltip, unit, format = 'number' }) => {
  const formatValue = (val: number) => {
    if (format === 'percentage') return `${(val * 100).toFixed(1)}%`;
    return `${val.toFixed(step < 1 ? 2 : 0)}${unit || ''}`;
  };

  return (
    <FormField label={label} tooltip={tooltip}>
      <div className="space-y-2">
        <Slider
          value={[value]}
          onValueChange={([newValue]) => onChange(newValue)}
          min={min}
          max={max}
          step={step}
          className="w-full"
        />
        <div className="text-sm text-slate-500 text-right">
          {formatValue(value)}
        </div>
      </div>
    </FormField>
  );
};

export const GameParameters: React.FC<GameParametersProps> = ({
  initialValues = {},
  onSubmit,
  onSave,
  onLoad,
  onReset,
  isLoading = false,
  className = ""
}) => {
  const [parameters, setParameters] = useState<GameParameters>({
    ...DEFAULT_PARAMETERS,
    ...initialValues,
  });

  const [activeTab, setActiveTab] = useState('network');
  const [presetName, setPresetName] = useState('');

  const updateParameter = <K extends keyof GameParameters>(
    key: K,
    value: GameParameters[K]
  ) => {
    setParameters(prev => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(parameters);
  };

  const handleReset = () => {
    setParameters({ ...DEFAULT_PARAMETERS, ...initialValues });
    onReset?.();
  };

  const handleSave = () => {
    onSave?.(parameters);
  };

  const validateParameters = () => {
    const { spreaderRatio, moderatorRatio, userRatio, botRatio } = parameters;
    const total = spreaderRatio + moderatorRatio + userRatio + botRatio;
    return Math.abs(total - 1.0) < 0.001; // Allow small floating point errors
  };

  const isValid = validateParameters();

  return (
    <div className={className}>
      <form onSubmit={handleSubmit} className="w-full max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-50 mb-2">
            Game Theory Simulation Parameters
          </h2>
          <p className="text-slate-600 dark:text-slate-400">
            Configure network structure, agent behaviors, and game dynamics for your simulation.
          </p>
        </div>

        {/* Parameter Validation Alert */}
        {!isValid && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <div className="flex items-center gap-2 text-red-800 dark:text-red-200">
              <Info className="w-4 h-4" />
              <span className="font-medium">Parameter Validation Error</span>
            </div>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">
              Agent ratios must sum to 1.0. Current total: {(parameters.spreaderRatio + parameters.moderatorRatio + parameters.userRatio + parameters.botRatio).toFixed(3)}
            </p>
          </div>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="network">Network</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="propagation">Propagation</TabsTrigger>
            <TabsTrigger value="game-theory">Game Theory</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="network" className="space-y-6">
            <FormSection
              title="Network Structure"
              description="Define the topology and connectivity of the social network."
            >
              <FormField
                label="Network Size"
                tooltip="Total number of nodes (agents) in the network. Larger networks provide more realistic results but require more computational resources."
                required
              >
                <Input
                  type="number"
                  min={10}
                  max={10000}
                  value={parameters.networkSize}
                  onChange={(e) => updateParameter('networkSize', parseInt(e.target.value))}
                />
              </FormField>

              <FormField
                label="Network Type"
                tooltip="The topology structure determining how nodes are connected. Scale-free networks have hubs, small-world networks have high clustering, random networks have uniform connectivity."
              >
                <Select
                  value={parameters.networkType}
                  onValueChange={(value) => updateParameter('networkType', value as any)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="scale-free">Scale-Free (Preferential Attachment)</SelectItem>
                    <SelectItem value="small-world">Small World (Watts-Strogatz)</SelectItem>
                    <SelectItem value="random">Random (Erdős-Rényi)</SelectItem>
                    <SelectItem value="grid">Grid (Regular Lattice)</SelectItem>
                  </SelectContent>
                </Select>
              </FormField>

              <SliderField
                label="Average Degree"
                value={parameters.averageDegree}
                onChange={(value) => updateParameter('averageDegree', value)}
                min={2}
                max={20}
                step={1}
                tooltip="Average number of connections per node. Higher values create denser networks with faster information spread."
              />

              <SliderField
                label="Clustering Coefficient"
                value={parameters.clusteringCoefficient}
                onChange={(value) => updateParameter('clusteringCoefficient', value)}
                min={0}
                max={1}
                step={0.01}
                tooltip="Measure of how much nodes tend to cluster together. Higher values create more tightly-knit communities."
                format="percentage"
              />
            </FormSection>
          </TabsContent>

          <TabsContent value="agents" className="space-y-6">
            <FormSection
              title="Agent Population"
              description="Define the distribution of different agent types in the network."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <SliderField
                    label="Spreader Ratio"
                    value={parameters.spreaderRatio}
                    onChange={(value) => updateParameter('spreaderRatio', value)}
                    min={0}
                    max={0.3}
                    step={0.01}
                    tooltip="Percentage of agents that actively spread misinformation."
                    format="percentage"
                  />

                  <SliderField
                    label="Moderator Ratio"
                    value={parameters.moderatorRatio}
                    onChange={(value) => updateParameter('moderatorRatio', value)}
                    min={0}
                    max={0.3}
                    step={0.01}
                    tooltip="Percentage of agents that act as content moderators."
                    format="percentage"
                  />
                </div>

                <div className="space-y-4">
                  <SliderField
                    label="Regular User Ratio"
                    value={parameters.userRatio}
                    onChange={(value) => updateParameter('userRatio', value)}
                    min={0.3}
                    max={0.95}
                    step={0.01}
                    tooltip="Percentage of passive users who consume and share content."
                    format="percentage"
                  />

                  <SliderField
                    label="Bot Ratio"
                    value={parameters.botRatio}
                    onChange={(value) => updateParameter('botRatio', value)}
                    min={0}
                    max={0.2}
                    step={0.01}
                    tooltip="Percentage of automated agents (bots) in the network."
                    format="percentage"
                  />
                </div>
              </div>

              <Separator />

              <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <h4 className="font-medium mb-3">Population Summary</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="text-center">
                    <Badge variant="destructive" className="mb-1">Spreaders</Badge>
                    <div className="font-mono">{(parameters.spreaderRatio * parameters.networkSize).toFixed(0)}</div>
                  </div>
                  <div className="text-center">
                    <Badge variant="default" className="mb-1">Moderators</Badge>
                    <div className="font-mono">{(parameters.moderatorRatio * parameters.networkSize).toFixed(0)}</div>
                  </div>
                  <div className="text-center">
                    <Badge variant="secondary" className="mb-1">Users</Badge>
                    <div className="font-mono">{(parameters.userRatio * parameters.networkSize).toFixed(0)}</div>
                  </div>
                  <div className="text-center">
                    <Badge variant="outline" className="mb-1">Bots</Badge>
                    <div className="font-mono">{(parameters.botRatio * parameters.networkSize).toFixed(0)}</div>
                  </div>
                </div>
              </div>
            </FormSection>
          </TabsContent>

          <TabsContent value="propagation" className="space-y-6">
            <FormSection
              title="Information Propagation"
              description="Configure how information spreads through the network over time."
            >
              <SliderField
                label="Base Propagation Rate"
                value={parameters.basePropagationRate}
                onChange={(value) => updateParameter('basePropagationRate', value)}
                min={0}
                max={1}
                step={0.01}
                tooltip="Base probability that information spreads from one node to another."
                format="percentage"
              />

              <SliderField
                label="Decay Rate"
                value={parameters.decayRate}
                onChange={(value) => updateParameter('decayRate', value)}
                min={0}
                max={0.2}
                step={0.01}
                tooltip="Rate at which information loses potency over time."
                format="percentage"
              />

              <SliderField
                label="Recovery Rate"
                value={parameters.recoveryRate}
                onChange={(value) => updateParameter('recoveryRate', value)}
                min={0}
                max={0.5}
                step={0.01}
                tooltip="Rate at which infected nodes recover and become immune to reinfection."
                format="percentage"
              />

              <SliderField
                label="Immunity Rate"
                value={parameters.immunityRate}
                onChange={(value) => updateParameter('immunityRate', value)}
                min={0}
                max={0.1}
                step={0.01}
                tooltip="Rate at which susceptible nodes develop natural immunity."
                format="percentage"
              />
            </FormSection>
          </TabsContent>

          <TabsContent value="game-theory" className="space-y-6">
            <FormSection
              title="Payoff Structure"
              description="Define rewards and penalties for different strategic behaviors."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-medium">Rewards</h4>

                  <FormField
                    label="Spreader Reward"
                    tooltip="Payoff for successful misinformation propagation."
                  >
                    <Input
                      type="number"
                      step="0.1"
                      value={parameters.spreaderReward}
                      onChange={(e) => updateParameter('spreaderReward', parseFloat(e.target.value))}
                    />
                  </FormField>

                  <FormField
                    label="Moderator Reward"
                    tooltip="Payoff for successful content moderation."
                  >
                    <Input
                      type="number"
                      step="0.1"
                      value={parameters.moderatorReward}
                      onChange={(e) => updateParameter('moderatorReward', parseFloat(e.target.value))}
                    />
                  </FormField>
                </div>

                <div className="space-y-4">
                  <h4 className="font-medium">Penalties</h4>

                  <FormField
                    label="Detection Penalty"
                    tooltip="Negative payoff when spreaders are caught by moderators."
                  >
                    <Input
                      type="number"
                      step="0.1"
                      value={parameters.detectionPenalty}
                      onChange={(e) => updateParameter('detectionPenalty', parseFloat(e.target.value))}
                    />
                  </FormField>

                  <FormField
                    label="False Positive Penalty"
                    tooltip="Negative payoff for moderators incorrectly flagging legitimate content."
                  >
                    <Input
                      type="number"
                      step="0.1"
                      value={parameters.falsePositivePenalty}
                      onChange={(e) => updateParameter('falsePositivePenalty', parseFloat(e.target.value))}
                    />
                  </FormField>
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <h4 className="font-medium">Learning Dynamics</h4>

                <SliderField
                  label="Learning Rate"
                  value={parameters.learningRate}
                  onChange={(value) => updateParameter('learningRate', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  tooltip="How quickly agents adapt their strategies based on outcomes."
                  format="percentage"
                />

                <FormField
                  label="Adaptation Frequency"
                  tooltip="How often (in time steps) agents can change their strategies."
                >
                  <Input
                    type="number"
                    min={1}
                    max={50}
                    value={parameters.adaptationFrequency}
                    onChange={(e) => updateParameter('adaptationFrequency', parseInt(e.target.value))}
                  />
                </FormField>
              </div>
            </FormSection>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-6">
            <FormSection
              title="Simulation Settings"
              description="Advanced configuration options for the simulation."
            >
              <FormField
                label="Time Horizon"
                tooltip="Total number of time steps the simulation will run."
                required
              >
                <Input
                  type="number"
                  min={10}
                  max={1000}
                  value={parameters.timeHorizon}
                  onChange={(e) => updateParameter('timeHorizon', parseInt(e.target.value))}
                />
              </FormField>

              <FormField
                label="Random Seed"
                tooltip="Seed for random number generation. Leave empty for truly random behavior."
              >
                <Input
                  type="number"
                  value={parameters.randomSeed || ''}
                  onChange={(e) => updateParameter('randomSeed', e.target.value ? parseInt(e.target.value) : undefined)}
                  placeholder="Leave empty for random"
                />
              </FormField>

              <FormField
                label="Save Frequency"
                tooltip="How often (in time steps) to save simulation state."
              >
                <Input
                  type="number"
                  min={1}
                  max={100}
                  value={parameters.saveFrequency}
                  onChange={(e) => updateParameter('saveFrequency', parseInt(e.target.value))}
                />
              </FormField>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  label="Enable Learning"
                  tooltip="Allow agents to adapt their strategies over time."
                  className="flex items-center justify-between"
                >
                  <Switch
                    checked={parameters.enableLearning}
                    onCheckedChange={(checked) => updateParameter('enableLearning', checked)}
                  />
                </FormField>

                <FormField
                  label="Network Evolution"
                  tooltip="Allow the network structure to change during simulation."
                  className="flex items-center justify-between"
                >
                  <Switch
                    checked={parameters.enableNetworkEvolution}
                    onCheckedChange={(checked) => updateParameter('enableNetworkEvolution', checked)}
                  />
                </FormField>
              </div>

              <Separator />

              <div className="space-y-4">
                <h4 className="font-medium">Fine-tuning Parameters</h4>

                <SliderField
                  label="Noise Level"
                  value={parameters.noiseLevel}
                  onChange={(value) => updateParameter('noiseLevel', value)}
                  min={0}
                  max={0.5}
                  step={0.01}
                  tooltip="Amount of random noise in agent decision-making."
                  format="percentage"
                />

                <FormField
                  label="Memory Length"
                  tooltip="How many past interactions agents remember when making decisions."
                >
                  <Input
                    type="number"
                    min={1}
                    max={50}
                    value={parameters.memoryLength}
                    onChange={(e) => updateParameter('memoryLength', parseInt(e.target.value))}
                  />
                </FormField>

                <SliderField
                  label="Exploration Rate"
                  value={parameters.explorationRate}
                  onChange={(value) => updateParameter('explorationRate', value)}
                  min={0}
                  max={0.5}
                  step={0.01}
                  tooltip="Probability that agents explore new strategies instead of exploiting known ones."
                  format="percentage"
                />

                <SliderField
                  label="Convergence Threshold"
                  value={parameters.convergenceThreshold}
                  onChange={(value) => updateParameter('convergenceThreshold', value)}
                  min={0.0001}
                  max={0.1}
                  step={0.0001}
                  tooltip="Threshold for determining when the simulation has converged to equilibrium."
                />
              </div>
            </FormSection>
          </TabsContent>
        </Tabs>

        {/* Action Buttons */}
        <div className="flex justify-between items-center pt-6 border-t">
          <div className="flex gap-3">
            <Button
              type="button"
              variant="outline"
              onClick={handleReset}
              disabled={isLoading}
              className="flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Reset to Defaults
            </Button>

            <Button
              type="button"
              variant="outline"
              onClick={handleSave}
              disabled={isLoading || !isValid}
              className="flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Save Preset
            </Button>

            <Button
              type="button"
              variant="outline"
              onClick={onLoad}
              disabled={isLoading}
              className="flex items-center gap-2"
            >
              <Upload className="w-4 h-4" />
              Load Preset
            </Button>
          </div>

          <Button
            type="submit"
            size="lg"
            disabled={isLoading || !isValid}
            className="min-w-[160px] flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            {isLoading ? 'Starting...' : 'Run Simulation'}
          </Button>
        </div>
      </form>
    </div>
  );
};