import { z } from 'zod';
import { VALIDATION_LIMITS } from './constants';

export const simulationParamsSchema = z.object({
  metadata: z.object({
    name: z.string().min(1, "Simulation name is required.").max(100, "Name cannot exceed 100 characters."),
    description: z.string().optional(),
    tags: z.array(z.string()).default([]),
    priority: z.enum(['low', 'normal', 'high', 'urgent']).default('normal')
  }),

  numPlayers: z.object({
    spreaders: z.number()
      .min(VALIDATION_LIMITS.players.spreaders.min, `Must have at least ${VALIDATION_LIMITS.players.spreaders.min} spreader.`)
      .max(VALIDATION_LIMITS.players.spreaders.max, `Cannot exceed ${VALIDATION_LIMITS.players.spreaders.max} spreaders.`),
    factCheckers: z.number()
      .min(VALIDATION_LIMITS.players.factCheckers.min, "Fact checkers cannot be negative.")
      .max(VALIDATION_LIMITS.players.factCheckers.max, `Cannot exceed ${VALIDATION_LIMITS.players.factCheckers.max} fact checkers.`),
    platforms: z.number()
      .min(VALIDATION_LIMITS.players.platforms.min, `Must have at least ${VALIDATION_LIMITS.players.platforms.min} platform.`)
      .max(VALIDATION_LIMITS.players.platforms.max, `Cannot exceed ${VALIDATION_LIMITS.players.platforms.max} platforms.`),
    users: z.number().min(1, "Must have at least 1 user.").optional()
  }),

  network: z.object({
    size: z.number()
      .min(VALIDATION_LIMITS.networkSize.min, `Network size must be at least ${VALIDATION_LIMITS.networkSize.min}.`)
      .max(VALIDATION_LIMITS.networkSize.max, `Network size cannot exceed ${VALIDATION_LIMITS.networkSize.max}.`),
    type: z.enum(['random', 'small_world', 'scale_free', 'complete', 'custom']),
    parameters: z.object({
      connectionProbability: z.number().min(0).max(1).optional(),
      averageDegree: z.number().min(1).optional(),
      clusteringCoefficient: z.number().min(0).max(1).optional(),
      powerLawExponent: z.number().min(1).optional(),
      rewireProbability: z.number().min(0).max(1).optional()
    })
  }),

  dynamics: z.object({
    timeHorizon: z.number()
      .min(VALIDATION_LIMITS.timeHorizon.min, `Time horizon must be at least ${VALIDATION_LIMITS.timeHorizon.min} rounds.`)
      .max(VALIDATION_LIMITS.timeHorizon.max, `Time horizon cannot exceed ${VALIDATION_LIMITS.timeHorizon.max} rounds.`),
    timeStep: z.number().min(0.1).max(10),
    maxRounds: z.number().min(1).optional(),

    propagationRate: z.number()
      .min(VALIDATION_LIMITS.propagationRate.min, `Propagation rate must be at least ${VALIDATION_LIMITS.propagationRate.min}.`)
      .max(VALIDATION_LIMITS.propagationRate.max, `Propagation rate cannot exceed ${VALIDATION_LIMITS.propagationRate.max}.`),
    decayRate: z.number().min(0).max(1),
    viralThreshold: z.number().min(0).max(1),

    detectionAccuracy: z.number()
      .min(VALIDATION_LIMITS.detectionAccuracy.min, `Detection accuracy must be at least ${VALIDATION_LIMITS.detectionAccuracy.min}.`)
      .max(VALIDATION_LIMITS.detectionAccuracy.max, `Detection accuracy cannot exceed ${VALIDATION_LIMITS.detectionAccuracy.max}.`),
    detectionDelay: z.number().min(0).max(100),
    factCheckingEffectiveness: z.number().min(0).max(1),

    learningRate: z.number().min(0).max(1),
    memoryLength: z.number().min(1).max(100),
    adaptationThreshold: z.number().min(0).max(1)
  }),

  payoffWeights: z.object({
    spreaders: z.object({
      reach: z.number(),
      detection: z.number(),
      reputation: z.number(),
      cost: z.number()
    }),
    factCheckers: z.object({
      accuracy: z.number(),
      effort: z.number(),
      impact: z.number(),
      reputation: z.number()
    }),
    platforms: z.object({
      engagement: z.number(),
      reputation: z.number(),
      cost: z.number(),
      regulation: z.number()
    })
  })
});

// Simplified schema for basic simulation form
export const basicSimulationSchema = z.object({
  networkSize: z.number()
    .min(VALIDATION_LIMITS.networkSize.min, `Network size must be at least ${VALIDATION_LIMITS.networkSize.min}.`)
    .max(VALIDATION_LIMITS.networkSize.max, `Network size cannot exceed ${VALIDATION_LIMITS.networkSize.max}.`),
  timeHorizon: z.number()
    .min(VALIDATION_LIMITS.timeHorizon.min, `Time horizon must be at least ${VALIDATION_LIMITS.timeHorizon.min} rounds.`)
    .max(VALIDATION_LIMITS.timeHorizon.max, `Time horizon cannot exceed ${VALIDATION_LIMITS.timeHorizon.max} rounds.`),
  propagationRate: z.number()
    .min(VALIDATION_LIMITS.propagationRate.min, `Propagation rate must be at least ${VALIDATION_LIMITS.propagationRate.min}.`)
    .max(VALIDATION_LIMITS.propagationRate.max, `Propagation rate cannot exceed ${VALIDATION_LIMITS.propagationRate.max}.`),
  detectionAccuracy: z.number()
    .min(VALIDATION_LIMITS.detectionAccuracy.min, `Detection accuracy must be at least ${VALIDATION_LIMITS.detectionAccuracy.min}.`)
    .max(VALIDATION_LIMITS.detectionAccuracy.max, `Detection accuracy cannot exceed ${VALIDATION_LIMITS.detectionAccuracy.max}.`),
  spreaders: z.number()
    .min(VALIDATION_LIMITS.players.spreaders.min, `Must have at least ${VALIDATION_LIMITS.players.spreaders.min} spreader.`)
    .max(VALIDATION_LIMITS.players.spreaders.max, `Cannot exceed ${VALIDATION_LIMITS.players.spreaders.max} spreaders.`),
  factCheckers: z.number()
    .min(VALIDATION_LIMITS.players.factCheckers.min, "Fact checkers cannot be negative.")
    .max(VALIDATION_LIMITS.players.factCheckers.max, `Cannot exceed ${VALIDATION_LIMITS.players.factCheckers.max} fact checkers.`),
  platforms: z.number()
    .min(VALIDATION_LIMITS.players.platforms.min, `Must have at least ${VALIDATION_LIMITS.players.platforms.min} platform.`)
    .max(VALIDATION_LIMITS.players.platforms.max, `Cannot exceed ${VALIDATION_LIMITS.players.platforms.max} platforms.`)
});

// Schema for game theory analysis input
export const gameTheoryAnalysisSchema = z.object({
  playerTypes: z.array(z.enum(['spreader', 'fact_checker', 'platform'])).min(2, "Need at least 2 player types for analysis."),
  strategies: z.record(z.array(z.string().min(1, "Strategy name cannot be empty."))),
  payoffMatrix: z.array(z.array(z.record(z.string(), z.number()))),
  analysisType: z.enum(['nash_equilibrium', 'dominant_strategies', 'pareto_optimal', 'full_analysis']),
  parameters: z.object({
    convergenceTolerance: z.number().min(0.001).max(0.1).optional(),
    maxIterations: z.number().min(10).max(10000).optional(),
    stabilityThreshold: z.number().min(0).max(1).optional()
  }).optional()
});

// Schema for classifier input
export const classifierInputSchema = z.object({
  text: z.string().min(1, "Text content is required.").max(10000, "Text cannot exceed 10,000 characters."),
  model: z.enum(['bert', 'roberta', 'distilbert']).default('bert'),
  parameters: z.object({
    confidence_threshold: z.number().min(0).max(1).default(0.5),
    max_length: z.number().min(1).max(512).default(512),
    return_probabilities: z.boolean().default(true)
  }).optional()
});

// Schema for network analysis
export const networkAnalysisSchema = z.object({
  networkData: z.object({
    nodes: z.array(z.object({
      id: z.string(),
      type: z.string(),
      attributes: z.record(z.unknown()).optional()
    })),
    edges: z.array(z.object({
      source: z.string(),
      target: z.string(),
      weight: z.number().optional(),
      type: z.string().optional()
    }))
  }),
  analysisType: z.enum(['centrality', 'clustering', 'communities', 'paths', 'full']),
  parameters: z.object({
    directed: z.boolean().default(false),
    weighted: z.boolean().default(false),
    algorithm: z.string().optional()
  }).optional()
});

// Export inferred types for use in components
export type SimulationParamsInput = z.infer<typeof simulationParamsSchema>;
export type BasicSimulationInput = z.infer<typeof basicSimulationSchema>;
export type GameTheoryAnalysisInput = z.infer<typeof gameTheoryAnalysisSchema>;
export type ClassifierInput = z.infer<typeof classifierInputSchema>;
export type NetworkAnalysisInput = z.infer<typeof networkAnalysisSchema>;