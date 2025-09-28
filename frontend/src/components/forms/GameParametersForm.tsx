// frontend/src/components/forms/GameParametersForm.tsx

"use client";

import React, { useState } from 'react';
import { Info, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { GameParameters } from '@/types/gameTheory';
import { DEFAULT_GAME_PARAMS } from '@/lib/constants';

interface GameParametersFormProps {
  initialValues?: Partial<GameParameters>;
  onSubmit: (parameters: GameParameters) => void;
  onReset?: () => void;
  isLoading?: boolean;
}

// Reusable form section component
const FormSection: React.FC<{
  title: string;
  description?: string;
  children: React.ReactNode;
}> = ({ title, description, children }) => (
  <Card className="mb-6">
    <CardHeader>
      <CardTitle className="text-lg">{title}</CardTitle>
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

export const GameParametersForm: React.FC<GameParametersFormProps> = ({
  initialValues = {},
  onSubmit,
  onReset,
  isLoading = false,
}) => {
  const [parameters, setParameters] = useState<GameParameters>({
    ...DEFAULT_GAME_PARAMS,
    ...initialValues,
  });

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
    setParameters({ ...DEFAULT_GAME_PARAMS, ...initialValues });
    if (onReset) onReset();
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-4xl mx-auto space-y-6">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-50 mb-2">
          Game Theory Parameters
        </h2>
        <p className="text-slate-600 dark:text-slate-400">
          Configure the strategic interactions, payoff structures, and network dynamics for your simulation.
        </p>
      </div>

      <Tabs defaultValue="network" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="payoffs">Payoffs</TabsTrigger>
          <TabsTrigger value="dynamics">Dynamics</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>

        <TabsContent value="network" className="space-y-6">
          <FormSection
            title="Network Configuration"
            description="Define the structure and properties of the social network."
          >
            <FormField
              label="Network Size"
              tooltip="Total number of nodes (users) in the simulation network."
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
              tooltip="The topology structure of the network connections."
            >
              <Select
                value={parameters.networkType}
                onValueChange={(value) => updateParameter('networkType', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="scale-free">Scale-Free</SelectItem>
                  <SelectItem value="small-world">Small World</SelectItem>
                  <SelectItem value="random">Random</SelectItem>
                  <SelectItem value="grid">Grid</SelectItem>
                </SelectContent>
              </Select>
            </FormField>

            <FormField
              label="Base Propagation Rate"
              tooltip="The base probability that information spreads between connected nodes."
            >
              <div className="space-y-2">
                <Slider
                  value={[parameters.basePropagationRate]}
                  onValueChange={([value]) => updateParameter('basePropagationRate', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(parameters.basePropagationRate * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Average Degree"
              tooltip="Average number of connections per node in the network."
            >
              <Input
                type="number"
                min={1}
                max={50}
                value={parameters.averageDegree}
                onChange={(e) => updateParameter('averageDegree', parseInt(e.target.value))}
              />
            </FormField>
          </FormSection>
        </TabsContent>

        <TabsContent value="payoffs" className="space-y-6">
          <FormSection
            title="Spreader Payoffs"
            description="Configure payoff structure for misinformation spreaders."
          >
            <FormField
              label="Reach Weight"
              tooltip="How much spreaders value the reach of their content."
            >
              <Input
                type="number"
                step="0.1"
                value={parameters.spreaderReachWeight}
                onChange={(e) => updateParameter('spreaderReachWeight', parseFloat(e.target.value))}
              />
            </FormField>

            <FormField
              label="Detection Penalty"
              tooltip="The negative payoff when spreaders are detected by moderators."
            >
              <Input
                type="number"
                step="0.1"
                value={parameters.spreaderDetectionPenalty}
                onChange={(e) => updateParameter('spreaderDetectionPenalty', parseFloat(e.target.value))}
              />
            </FormField>

            <FormField
              label="Sophistication Bonus"
              tooltip="Additional payoff for using sophisticated spreading strategies."
            >
              <Input
                type="number"
                step="0.1"
                value={parameters.spreaderSophisticationBonus}
                onChange={(e) => updateParameter('spreaderSophisticationBonus', parseFloat(e.target.value))}
              />
            </FormField>
          </FormSection>

          <FormSection
            title="Moderator Payoffs"
            description="Configure payoff structure for content moderators."
          >
            <FormField
              label="Detection Reward"
              tooltip="Positive payoff for successfully detecting misinformation."
            >
              <Input
                type="number"
                step="0.1"
                value={parameters.moderatorDetectionReward}
                onChange={(e) => updateParameter('moderatorDetectionReward', parseFloat(e.target.value))}
              />
            </FormField>

            <FormField
              label="False Positive Penalty"
              tooltip="Negative payoff for incorrectly flagging legitimate content."
            >
              <Input
                type="number"
                step="0.1"
                value={parameters.moderatorFalsePositivePenalty}
                onChange={(e) => updateParameter('moderatorFalsePositivePenalty', parseFloat(e.target.value))}
              />
            </FormField>

            <FormField
              label="Efficiency Bonus"
              tooltip="Additional payoff for efficient moderation strategies."
            >
              <Input
                type="number"
                step="0.1"
                value={parameters.moderatorEfficiencyBonus}
                onChange={(e) => updateParameter('moderatorEfficiencyBonus', parseFloat(e.target.value))}
              />
            </FormField>
          </FormSection>
        </TabsContent>

        <TabsContent value="dynamics" className="space-y-6">
          <FormSection
            title="Game Dynamics"
            description="Configure how the game evolves over time."
          >
            <FormField
              label="Time Horizon"
              tooltip="Number of rounds in the simulation."
            >
              <Input
                type="number"
                min={1}
                max={1000}
                value={parameters.timeHorizon}
                onChange={(e) => updateParameter('timeHorizon', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Learning Rate"
              tooltip="How quickly players adapt their strategies based on outcomes."
            >
              <div className="space-y-2">
                <Slider
                  value={[parameters.learningRate]}
                  onValueChange={([value]) => updateParameter('learningRate', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(parameters.learningRate * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Adaptation Frequency"
              tooltip="How often players can change their strategies (rounds)."
            >
              <Input
                type="number"
                min={1}
                max={50}
                value={parameters.adaptationFrequency}
                onChange={(e) => updateParameter('adaptationFrequency', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Enable Learning"
              tooltip="Allow players to adapt strategies over time based on payoffs."
            >
              <div className="flex justify-end">
                <Switch
                  checked={parameters.enableLearning}
                  onCheckedChange={(checked) => updateParameter('enableLearning', checked)}
                />
              </div>
            </FormField>
          </FormSection>
        </TabsContent>

        <TabsContent value="advanced" className="space-y-6">
          <FormSection
            title="Advanced Settings"
            description="Fine-tune advanced simulation parameters."
          >
            <FormField
              label="Random Seed"
              tooltip="Seed for random number generation (for reproducible results)."
            >
              <Input
                type="number"
                value={parameters.randomSeed || ''}
                onChange={(e) => updateParameter('randomSeed', e.target.value ? parseInt(e.target.value) : undefined)}
                placeholder="Leave empty for random"
              />
            </FormField>

            <FormField
              label="Noise Level"
              tooltip="Amount of random noise in player decisions."
            >
              <div className="space-y-2">
                <Slider
                  value={[parameters.noiseLevel]}
                  onValueChange={([value]) => updateParameter('noiseLevel', value)}
                  min={0}
                  max={0.5}
                  step={0.01}
                  className="w-full"
                />
                <div className="text-sm text-slate-500 text-right">
                  {(parameters.noiseLevel * 100).toFixed(1)}%
                </div>
              </div>
            </FormField>

            <FormField
              label="Memory Length"
              tooltip="How many past rounds players remember when making decisions."
            >
              <Input
                type="number"
                min={1}
                max={20}
                value={parameters.memoryLength}
                onChange={(e) => updateParameter('memoryLength', parseInt(e.target.value))}
              />
            </FormField>

            <FormField
              label="Enable Network Evolution"
              tooltip="Allow the network structure to change during simulation."
            >
              <div className="flex justify-end">
                <Switch
                  checked={parameters.enableNetworkEvolution}
                  onCheckedChange={(checked) => updateParameter('enableNetworkEvolution', checked)}
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
            Save as Preset
          </Button>
          <Button
            type="submit"
            size="lg"
            disabled={isLoading}
            className="min-w-[140px]"
          >
            {isLoading ? 'Running...' : 'Run Simulation'}
          </Button>
        </div>
      </div>
    </form>
  );
};