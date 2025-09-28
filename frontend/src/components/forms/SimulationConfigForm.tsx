// frontend/src/components/forms/SimulationConfigForm.tsx

"use client";

import React, { useState } from 'react';
import { Info, RotateCcw, Play, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { SimulationConfig } from '@/types/simulation';
import { DEFAULT_SIMULATION_PARAMS } from '@/lib/constants';

interface SimulationConfigFormProps {
  initialValues?: Partial<SimulationConfig>;
  onSubmit: (config: SimulationConfig) => void;
  onReset?: () => void;
  isLoading?: boolean;
}

// Reusable form section component
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

// Reusable form field with tooltip
const FormField: React.FC<{
  label: string;
  tooltip?: string;
  children: React.ReactNode;
  className?: string;
}> = ({ label, tooltip, children, className = "grid grid-cols-1 md:grid-cols-2 items-center gap-4" }) => (
  <div className={className}>
    <Label className="flex items-center text-sm font-medium">
      {label}
      {tooltip && (
        <div className="relative group ml-2">
          <Info className="w-4 h-4 text-slate-400 hover:text-slate-600 cursor-help" />
          <div className="absolute left-0 top-6 z-10 w-64 p-2 bg-slate-900 text-white text-xs rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            {tooltip}
          </div>
        </div>
      )}
    </Label>
    <div>{children}</div>
  </div>
);

export const SimulationConfigForm: React.FC<SimulationConfigFormProps> = ({
  initialValues = {},
  onSubmit,
  onReset,
  isLoading = false,
}) => {
  const [config, setConfig] = useState<SimulationConfig>({
    ...DEFAULT_SIMULATION_PARAMS,
    ...initialValues,
  });

  const updateConfig = <K extends keyof SimulationConfig>(
    key: K,
    value: SimulationConfig[K]
  ) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(config);
  };

  const handleReset = () => {
    setConfig({ ...DEFAULT_SIMULATION_PARAMS, ...initialValues });
    if (onReset) onReset();
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-4xl mx-auto space-y-6">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-50 mb-2">
          Simulation Configuration
        </h2>
        <p className="text-slate-600 dark:text-slate-400">
          Configure the parameters and settings for your misinformation propagation simulation.
        </p>
      </div>

      <Tabs defaultValue="basic" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="basic">Basic</TabsTrigger>
          <TabsTrigger value="propagation">Propagation</TabsTrigger>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="output">Output</TabsTrigger>
        </TabsList>

        <TabsContent value="basic" className="space-y-6">
          <FormSection
            title="Basic Configuration"
            description="Essential simulation parameters and runtime settings."
            icon={<Settings className="w-5 h-5" />}
          >
            <FormField
              label="Simulation Name"
              tooltip="A descriptive name for this simulation run."
            >
              <Input
                value={config.name}
                onChange={(e) => updateConfig('name', e.target.value)}
                placeholder="e.g., Baseline Propagation Study"
              />
            </FormField>

            <FormField
              label="Description"
              tooltip="Detailed description of the simulation purpose and methodology."
              className="grid grid-cols-1 gap-4"
            >
              <Textarea
                value={config.description}
                onChange={(e) => updateConfig('description', e.target.value)}
                placeholder="Describe the objectives and methodology of this simulation..."
                rows={3}
              />
            </FormField>

            <FormField
              label="Duration (Time Steps)"
              tooltip="Total number of time steps the simulation will run."
            >
              <Input
                type="number"
                min={1}
                max={10000}
                value={config.duration}
                onChange={(e) => updateConfig('duration', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Random Seed"
              tooltip="Seed for reproducible random number generation."
            >
              <Input
                type="number"
                value={config.randomSeed || ''}
                onChange={(e) => updateConfig('randomSeed', e.target.value ? parseInt(e.target.value) : undefined)}
                placeholder="Leave empty for random seed"
              />
            </FormField>

            <FormField
              label="Parallel Processing"
              tooltip="Enable multi-threaded processing for faster execution."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.enableParallelProcessing}
                  onCheckedChange={(checked) => updateConfig('enableParallelProcessing', checked)}
                />
              </div>
            </FormField>
          </FormSection>
        </TabsContent>

        <TabsContent value="propagation" className="space-y-6">
          <FormSection
            title="Information Propagation"
            description="Configure how information spreads through the network."
          >
            <FormField
              label="Initial Infection Rate"
              tooltip="Percentage of nodes initially infected with misinformation."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.initialInfectionRate]}
                  onValueChange={([value]) => updateConfig('initialInfectionRate', value)}
                  min={0}
                  max={0.5}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.initialInfectionRate * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Transmission Probability"
              tooltip="Base probability of information transmission between connected nodes."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.transmissionProbability]}
                  onValueChange={([value]) => updateConfig('transmissionProbability', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.transmissionProbability * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Recovery Rate"
              tooltip="Rate at which nodes recover from misinformation exposure."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.recoveryRate]}
                  onValueChange={([value]) => updateConfig('recoveryRate', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.recoveryRate * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Information Decay"
              tooltip="Rate at which information loses potency over time."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.informationDecay]}
                  onValueChange={([value]) => updateConfig('informationDecay', value)}
                  min={0}
                  max={0.1}
                  step={0.001}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.informationDecay * 100).toFixed(2)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Enable Reinfection"
              tooltip="Allow nodes to be infected multiple times during the simulation."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.enableReinfection}
                  onCheckedChange={(checked) => updateConfig('enableReinfection', checked)}
                />
              </div>
            </FormField>
          </FormSection>
        </TabsContent>

        <TabsContent value="agents" className="space-y-6">
          <FormSection
            title="Agent Configuration"
            description="Define the types and behaviors of agents in the simulation."
          >
            <FormField
              label="Agent Types"
              tooltip="Types of agents to include in the simulation."
            >
              <Select
                value={config.agentTypes}
                onValueChange={(value) => updateConfig('agentTypes', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mixed">Mixed Population</SelectItem>
                  <SelectItem value="spreaders-only">Spreaders Only</SelectItem>
                  <SelectItem value="moderators-only">Moderators Only</SelectItem>
                  <SelectItem value="users-only">Regular Users Only</SelectItem>
                </SelectContent>
              </Select>
            </FormField>

            <FormField
              label="Spreader Ratio"
              tooltip="Percentage of agents that are misinformation spreaders."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.spreaderRatio]}
                  onValueChange={([value]) => updateConfig('spreaderRatio', value)}
                  min={0}
                  max={0.3}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.spreaderRatio * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Moderator Ratio"
              tooltip="Percentage of agents that are content moderators."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.moderatorRatio]}
                  onValueChange={([value]) => updateConfig('moderatorRatio', value)}
                  min={0}
                  max={0.2}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.moderatorRatio * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Agent Learning"
              tooltip="Enable agents to learn and adapt their strategies."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.enableAgentLearning}
                  onCheckedChange={(checked) => updateConfig('enableAgentLearning', checked)}
                />
              </div>
            </FormField>

            <FormField
              label="Strategy Adaptation Rate"
              tooltip="How quickly agents adapt their strategies based on outcomes."
            >
              <div className="space-y-2">
                <Slider
                  value={[config.strategyAdaptationRate]}
                  onValueChange={([value]) => updateConfig('strategyAdaptationRate', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(config.strategyAdaptationRate * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>
          </FormSection>
        </TabsContent>

        <TabsContent value="output" className="space-y-6">
          <FormSection
            title="Output Configuration"
            description="Configure what data to collect and how often to save results."
          >
            <FormField
              label="Save Frequency"
              tooltip="How often to save simulation state (every N time steps)."
            >
              <Input
                type="number"
                min={1}
                max={1000}
                value={config.saveFrequency}
                onChange={(e) => updateConfig('saveFrequency', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Collect Network Snapshots"
              tooltip="Save complete network state at specified intervals."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.collectNetworkSnapshots}
                  onCheckedChange={(checked) => updateConfig('collectNetworkSnapshots', checked)}
                />
              </div>
            </FormField>

            <FormField
              label="Collect Agent Trajectories"
              tooltip="Track individual agent states throughout the simulation."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.collectAgentTrajectories}
                  onCheckedChange={(checked) => updateConfig('collectAgentTrajectories', checked)}
                />
              </div>
            </FormField>

            <FormField
              label="Collect Propagation Paths"
              tooltip="Record the paths through which information spreads."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.collectPropagationPaths}
                  onCheckedChange={(checked) => updateConfig('collectPropagationPaths', checked)}
                />
              </div>
            </FormField>

            <FormField
              label="Output Format"
              tooltip="Format for saving simulation results."
            >
              <Select
                value={config.outputFormat}
                onValueChange={(value) => updateConfig('outputFormat', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                  <SelectItem value="parquet">Parquet</SelectItem>
                  <SelectItem value="hdf5">HDF5</SelectItem>
                </SelectContent>
              </Select>
            </FormField>

            <FormField
              label="Compression"
              tooltip="Enable compression for output files."
            >
              <div className="flex justify-end">
                <Switch
                  checked={config.enableCompression}
                  onCheckedChange={(checked) => updateConfig('enableCompression', checked)}
                />
              </div>
            </FormField>
          </FormSection>
        </TabsContent>
      </Tabs>

      {/* Action Buttons */}
      <div className="flex justify-between items-center pt-6 border-t">
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

        <div className="flex gap-3">
          <Button
            type="button"
            variant="outline"
            disabled={isLoading}
          >
            Save Configuration
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={isLoading}
          >
            Load Preset
          </Button>
          <Button
            type="submit"
            size="lg"
            disabled={isLoading}
            className="min-w-[140px] flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            {isLoading ? 'Starting...' : 'Start Simulation'}
          </Button>
        </div>
      </div>
    </form>
  );
};