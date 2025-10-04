// frontend/src/app/(dashboard)/equilibrium/page.tsx

"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Calculator,
  Target,
  CheckCircle2,
  Info,
  Sparkles,
  BarChart3,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { PayoffMatrix } from '@/components/game-theory/PayoffMatrix';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { PayoffMatrixData } from '@/types/gameTheory';

// ================================================================
// Scenario Configurations
// ================================================================

interface GameScenario {
  id: string;
  name: string;
  description: string;
  parameters: GameParameters;
}

interface GameParameters {
  detectionPenalty: number;
  verificationCost: number;
  engagementRevenue: number;
  networkDensity: number;
  learningRate: number;
}

const SCENARIOS: GameScenario[] = [
  {
    id: 'baseline',
    name: 'Baseline Scenario',
    description: 'Standard parameters representing typical social media dynamics',
    parameters: {
      detectionPenalty: 50,
      verificationCost: 30,
      engagementRevenue: 60,
      networkDensity: 50,
      learningRate: 40,
    },
  },
  {
    id: 'high_detection',
    name: 'High Detection Scenario',
    description: 'Strong content moderation with high penalties for fake news',
    parameters: {
      detectionPenalty: 90,
      verificationCost: 30,
      engagementRevenue: 50,
      networkDensity: 50,
      learningRate: 60,
    },
  },
  {
    id: 'low_moderation',
    name: 'Low Moderation Scenario',
    description: 'Minimal platform intervention with weak detection',
    parameters: {
      detectionPenalty: 20,
      verificationCost: 40,
      engagementRevenue: 80,
      networkDensity: 60,
      learningRate: 30,
    },
  },
  {
    id: 'high_engagement',
    name: 'High Engagement Scenario',
    description: 'Platforms prioritize engagement over content quality',
    parameters: {
      detectionPenalty: 30,
      verificationCost: 50,
      engagementRevenue: 95,
      networkDensity: 70,
      learningRate: 35,
    },
  },
];

// Player interaction options
const PLAYER_INTERACTIONS = [
  { value: 'spreader_moderator', label: 'Spreader vs. Moderator', players: ['Spreader', 'Moderator'] },
  { value: 'spreader_platform', label: 'Spreader vs. Platform', players: ['Spreader', 'Platform'] },
  { value: 'platform_factchecker', label: 'Platform vs. Fact-Checker', players: ['Platform', 'Fact-Checker'] },
  { value: 'moderator_user', label: 'Moderator vs. User', players: ['Moderator', 'User'] },
];

// ================================================================
// Helper Functions
// ================================================================

function generatePayoffMatrix(
  playerInteraction: string,
  parameters: GameParameters
): PayoffMatrixData {
  const { detectionPenalty, verificationCost, engagementRevenue } = parameters;

  // Scale parameters to payoff values
  const detectScale = detectionPenalty / 50;
  const costScale = verificationCost / 50;
  const revenueScale = engagementRevenue / 50;

  if (playerInteraction === 'spreader_moderator') {
    return {
      players: ['Spreader', 'Moderator'] as [string, string],
      strategies: {
        Spreader: ['Aggressive', 'Conservative'],
        Moderator: ['Strict', 'Lenient'],
      },
      payoffs: [
        [
          { Spreader: 2.5 * revenueScale - 1.0 * detectScale, Moderator: -1.8 * costScale },
          { Spreader: 1.5 * revenueScale, Moderator: 1.2 - 0.5 * costScale },
        ],
        [
          { Spreader: 1.0 * revenueScale - 0.3 * detectScale, Moderator: -0.8 * costScale },
          { Spreader: 0.8 * revenueScale, Moderator: 0.5 },
        ],
      ],
      equilibrium: {
        strategies: [1, 1] as [number, number],
        payoffs: {
          Spreader: 0.8 * revenueScale,
          Moderator: 0.5,
        },
        type: 'pure',
        stability: 0.85,
        classification: 'strict',
      },
    };
  }

  if (playerInteraction === 'spreader_platform') {
    return {
      players: ['Spreader', 'Platform'] as [string, string],
      strategies: {
        Spreader: ['Post Fake', 'Post Mixed'],
        Platform: ['Strict Policy', 'Lenient Policy'],
      },
      payoffs: [
        [
          { Spreader: 1.2 * revenueScale - 2.0 * detectScale, Platform: 0.5 * revenueScale - 1.5 },
          { Spreader: 2.8 * revenueScale - 0.5 * detectScale, Platform: 1.8 * revenueScale },
        ],
        [
          { Spreader: 0.9 * revenueScale - 0.8 * detectScale, Platform: 1.2 * revenueScale - 0.8 },
          { Spreader: 1.5 * revenueScale, Platform: 1.6 * revenueScale - 0.3 },
        ],
      ],
      equilibrium: {
        strategies: [1, 0] as [number, number],
        payoffs: {
          Spreader: 0.9 * revenueScale - 0.8 * detectScale,
          Platform: 1.2 * revenueScale - 0.8,
        },
        type: 'pure',
        stability: 0.78,
        classification: 'weak',
      },
    };
  }

  if (playerInteraction === 'platform_factchecker') {
    return {
      players: ['Platform', 'Fact-Checker'] as [string, string],
      strategies: {
        Platform: ['Active Monitoring', 'Passive Monitoring'],
        'Fact-Checker': ['Comprehensive', 'Selective'],
      },
      payoffs: [
        [
          { Platform: 1.8 - 1.2 * costScale, 'Fact-Checker': 2.5 - 1.5 * costScale },
          { Platform: 1.2 - 0.8 * costScale, 'Fact-Checker': 1.8 - 0.8 * costScale },
        ],
        [
          { Platform: 1.5 - 0.5 * costScale, 'Fact-Checker': 2.0 - 1.0 * costScale },
          { Platform: 0.9, 'Fact-Checker': 1.2 - 0.4 * costScale },
        ],
      ],
      equilibrium: {
        strategies: [0, 1] as [number, number],
        payoffs: {
          Platform: 1.2 - 0.8 * costScale,
          'Fact-Checker': 1.8 - 0.8 * costScale,
        },
        type: 'pure',
        stability: 0.88,
        classification: 'strict',
      },
    };
  }

  // Default: moderator_user
  return {
    players: ['Moderator', 'User'] as [string, string],
    strategies: {
      Moderator: ['Proactive', 'Reactive'],
      User: ['Report Content', 'Ignore'],
    },
    payoffs: [
      [
        { Moderator: 1.5, User: 1.2 },
        { Moderator: 0.8 - 0.3 * costScale, User: -0.5 },
      ],
      [
        { Moderator: 0.9, User: 0.8 },
        { Moderator: 0.3, User: 0.1 },
      ],
    ],
    equilibrium: {
      strategies: [0, 0] as [number, number],
      payoffs: { Moderator: 1.5, User: 1.2 },
      type: 'pure',
      stability: 0.92,
      classification: 'strict',
    },
  };
}

// Animated counter hook
function useCountUp(end: number, duration: number = 1000) {
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

// ================================================================
// Main Component
// ================================================================

export default function EquilibriumPage() {
  // State
  const [selectedScenario, setSelectedScenario] = useState<string>('baseline');
  const [selectedInteraction, setSelectedInteraction] = useState<string>('spreader_moderator');
  const [parameters, setParameters] = useState<GameParameters>(SCENARIOS[0].parameters);
  const [isCalculating, setIsCalculating] = useState(false);
  const [hasManualChanges, setHasManualChanges] = useState(false);
  const [payoffMatrix, setPayoffMatrix] = useState<PayoffMatrixData>(
    generatePayoffMatrix('spreader_moderator', SCENARIOS[0].parameters)
  );

  // Handle scenario change
  const handleScenarioChange = (scenarioId: string) => {
    const scenario = SCENARIOS.find((s) => s.id === scenarioId);
    if (scenario) {
      setSelectedScenario(scenarioId);
      setParameters(scenario.parameters);
      setHasManualChanges(false);
      calculateEquilibrium(selectedInteraction, scenario.parameters);
    }
  };

  // Handle parameter change
  const handleParameterChange = (param: keyof GameParameters, value: number) => {
    const newParams = { ...parameters, [param]: value };
    setParameters(newParams);
    setHasManualChanges(true);
  };

  // Calculate equilibrium
  const calculateEquilibrium = async (interaction: string, params: GameParameters) => {
    setIsCalculating(true);

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));

    const newMatrix = generatePayoffMatrix(interaction, params);
    setPayoffMatrix(newMatrix);
    setIsCalculating(false);
  };

  // Handle calculate button
  const handleCalculate = () => {
    calculateEquilibrium(selectedInteraction, parameters);
    setHasManualChanges(false);
  };

  // Handle interaction change
  const handleInteractionChange = (interaction: string) => {
    setSelectedInteraction(interaction);
    calculateEquilibrium(interaction, parameters);
  };

  // Get current scenario
  const currentScenario = SCENARIOS.find((s) => s.id === selectedScenario) || SCENARIOS[0];
  const currentInteraction = PLAYER_INTERACTIONS.find((i) => i.value === selectedInteraction);

  // Animated values for expected payoffs
  const animatedPayoff1 = useCountUp(payoffMatrix.equilibrium?.payoffs[payoffMatrix.players[0]] || 0);
  const animatedPayoff2 = useCountUp(payoffMatrix.equilibrium?.payoffs[payoffMatrix.players[1]] || 0);

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="p-3 bg-purple-100 rounded-lg">
            <Target className="h-6 w-6 text-purple-600" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
            Nash Equilibrium Analysis
          </h1>
        </div>
        <p className="text-gray-600 text-lg">
          Explore the strategic outcomes of the misinformation game by adjusting key parameters and scenarios
        </p>
      </motion.div>

      {/* Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Interactive Controls */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="lg:col-span-1 space-y-6"
        >
          <Card className="sticky top-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-purple-600" />
                Controls
              </CardTitle>
              <CardDescription>Configure game parameters and scenarios</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Scenario Selector */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Scenario
                </label>
                <Select value={selectedScenario} onValueChange={handleScenarioChange}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SCENARIOS.map((scenario) => (
                      <SelectItem key={scenario.id} value={scenario.id}>
                        {scenario.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500 mt-2">{currentScenario.description}</p>
              </div>

              {/* Player Interaction Selector */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">
                  Player Interaction
                </label>
                <Select value={selectedInteraction} onValueChange={handleInteractionChange}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PLAYER_INTERACTIONS.map((interaction) => (
                      <SelectItem key={interaction.value} value={interaction.value}>
                        {interaction.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="border-t pt-4">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">
                  Game Parameters
                </h3>

                {/* Detection Penalty Slider */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm text-gray-700">
                      Detection Penalty
                    </label>
                    <Badge variant="outline">{parameters.detectionPenalty}%</Badge>
                  </div>
                  <Slider
                    min={0}
                    max={100}
                    step={5}
                    value={[parameters.detectionPenalty]}
                    onValueChange={(value) => handleParameterChange('detectionPenalty', value[0])}
                  />
                  <p className="text-xs text-gray-500">Cost for spreaders when fake news is detected</p>
                </div>

                {/* Verification Cost Slider */}
                <div className="space-y-3 mt-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm text-gray-700">
                      Verification Cost
                    </label>
                    <Badge variant="outline">{parameters.verificationCost}%</Badge>
                  </div>
                  <Slider
                    min={0}
                    max={100}
                    step={5}
                    value={[parameters.verificationCost]}
                    onValueChange={(value) => handleParameterChange('verificationCost', value[0])}
                  />
                  <p className="text-xs text-gray-500">Resource cost for fact-checkers to verify content</p>
                </div>

                {/* Engagement Revenue Slider */}
                <div className="space-y-3 mt-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm text-gray-700">
                      Engagement Revenue
                    </label>
                    <Badge variant="outline">{parameters.engagementRevenue}%</Badge>
                  </div>
                  <Slider
                    min={0}
                    max={100}
                    step={5}
                    value={[parameters.engagementRevenue]}
                    onValueChange={(value) => handleParameterChange('engagementRevenue', value[0])}
                  />
                  <p className="text-xs text-gray-500">Platform revenue from user engagement</p>
                </div>

                {/* Network Density Slider */}
                <div className="space-y-3 mt-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm text-gray-700">
                      Network Density
                    </label>
                    <Badge variant="outline">{parameters.networkDensity}%</Badge>
                  </div>
                  <Slider
                    min={0}
                    max={100}
                    step={5}
                    value={[parameters.networkDensity]}
                    onValueChange={(value) => handleParameterChange('networkDensity', value[0])}
                  />
                  <p className="text-xs text-gray-500">Connectivity of the social network</p>
                </div>

                {/* Learning Rate Slider */}
                <div className="space-y-3 mt-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm text-gray-700">
                      Learning Rate
                    </label>
                    <Badge variant="outline">{parameters.learningRate}%</Badge>
                  </div>
                  <Slider
                    min={0}
                    max={100}
                    step={5}
                    value={[parameters.learningRate]}
                    onValueChange={(value) => handleParameterChange('learningRate', value[0])}
                  />
                  <p className="text-xs text-gray-500">Speed at which players adapt strategies</p>
                </div>
              </div>

              {/* Calculate Button */}
              <Button
                onClick={handleCalculate}
                disabled={!hasManualChanges || isCalculating}
                size="lg"
                className="w-full"
              >
                {isCalculating ? (
                  <>
                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    Calculating...
                  </>
                ) : (
                  <>
                    <Calculator className="mr-2 h-4 w-4" />
                    Calculate Equilibrium
                  </>
                )}
              </Button>

              {!hasManualChanges && (
                <p className="text-xs text-center text-gray-500">
                  Adjust sliders to enable calculation
                </p>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Right Column: Results Display */}
        <div className="lg:col-span-2 space-y-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={`${selectedInteraction}-${JSON.stringify(parameters)}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              className="space-y-6"
            >
              {/* Payoff Matrix */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-blue-600" />
                    Payoff Matrix
                  </CardTitle>
                  <CardDescription>
                    Strategic payoffs for {currentInteraction?.players.join(' vs. ')} interaction
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <PayoffMatrix
                    data={payoffMatrix}
                    title={currentInteraction?.label || ''}
                    highlightEquilibrium={true}
                  />
                </CardContent>
              </Card>

              {/* Equilibrium Summary */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5 text-purple-600" />
                    Equilibrium Analysis
                  </CardTitle>
                  <CardDescription>Detailed analysis of the Nash equilibrium</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Pure Strategy Equilibrium */}
                  {payoffMatrix.equilibrium && (
                    <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                      <div className="flex items-center gap-2 mb-3">
                        <CheckCircle2 className="h-5 w-5 text-purple-600" />
                        <h4 className="font-semibold text-gray-900">
                          Pure Strategy Equilibrium
                        </h4>
                      </div>
                      <div className="space-y-2 text-sm">
                        <p className="text-gray-700">
                          <span className="font-medium">{payoffMatrix.players[0]}:</span>{' '}
                          {
                            payoffMatrix.strategies[payoffMatrix.players[0]][
                              payoffMatrix.equilibrium.strategies[0]
                            ]
                          }
                        </p>
                        <p className="text-gray-700">
                          <span className="font-medium">{payoffMatrix.players[1]}:</span>{' '}
                          {
                            payoffMatrix.strategies[payoffMatrix.players[1]][
                              payoffMatrix.equilibrium.strategies[1]
                            ]
                          }
                        </p>
                        <div className="pt-2 mt-2 border-t border-purple-200">
                          <div className="flex items-center gap-2">
                            <Badge
                              variant="outline"
                              className="border-purple-500 text-purple-700"
                            >
                              {payoffMatrix.equilibrium.type === 'pure' ? 'Pure Strategy' : 'Mixed Strategy'}
                            </Badge>
                            <Badge
                              variant="outline"
                              className="border-green-500 text-green-700"
                            >
                              Stability: {(payoffMatrix.equilibrium.stability * 100).toFixed(0)}%
                            </Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Expected Payoffs */}
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-4">
                      Expected Payoffs at Equilibrium
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                      <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className="p-4 bg-blue-50 rounded-lg border border-blue-200"
                      >
                        <div className="text-sm text-gray-600 mb-1">
                          {payoffMatrix.players[0]}
                        </div>
                        <div className="text-2xl font-bold text-blue-700">
                          {animatedPayoff1.toFixed(2)}
                        </div>
                      </motion.div>

                      <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5, delay: 0.3 }}
                        className="p-4 bg-red-50 rounded-lg border border-red-200"
                      >
                        <div className="text-sm text-gray-600 mb-1">
                          {payoffMatrix.players[1]}
                        </div>
                        <div className="text-2xl font-bold text-red-700">
                          {animatedPayoff2.toFixed(2)}
                        </div>
                      </motion.div>
                    </div>
                  </div>

                  {/* Analysis and Insights */}
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertTitle>Strategic Insight</AlertTitle>
                    <AlertDescription>
                      {parameters.detectionPenalty > 70 ? (
                        <>
                          Under high detection penalties, spreaders are incentivized to shift towards more
                          truthful content to avoid significant reputation costs. This creates a more stable
                          equilibrium where misinformation is naturally suppressed.
                        </>
                      ) : parameters.engagementRevenue > 70 ? (
                        <>
                          With high engagement revenue, platforms face conflicting incentives between content
                          moderation and profit maximization. This can lead to suboptimal equilibria where
                          misinformation persists despite available detection tools.
                        </>
                      ) : parameters.verificationCost > 60 ? (
                        <>
                          High verification costs discourage comprehensive fact-checking, creating
                          opportunities for sophisticated misinformation to spread. Reducing these costs
                          through automation or crowdsourcing could improve equilibrium outcomes.
                        </>
                      ) : (
                        <>
                          The current parameter configuration represents a balanced scenario where multiple
                          strategic forces interact. Players must carefully weigh the trade-offs between
                          aggressive strategies (higher payoff potential) and conservative approaches (lower
                          risk).
                        </>
                      )}
                    </AlertDescription>
                  </Alert>

                  {/* Additional Metrics */}
                  <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                    <div>
                      <div className="text-sm text-gray-600 mb-1">
                        Equilibrium Type
                      </div>
                      <div className="font-semibold text-gray-900">
                        {payoffMatrix.equilibrium?.classification === 'strict'
                          ? 'Strict Nash'
                          : payoffMatrix.equilibrium?.classification === 'weak'
                          ? 'Weak Nash'
                          : 'Trembling Hand Perfect'}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600 mb-1">
                        Social Welfare
                      </div>
                      <div className="font-semibold text-gray-900">
                        {(
                          (animatedPayoff1 + animatedPayoff2) /
                          2
                        ).toFixed(2)}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}