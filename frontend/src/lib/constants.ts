import { GameParameters } from '@/types/simulation';

export const API_ROUTES = {
  CLASSIFIER_PREDICT: '/classifier/predict',
  SIMULATION_RUN: '/simulation/run',
  SIMULATION_STATUS: (id: string) => `/simulation/status/${id}`,
  SIMULATION_RESULTS: (id: string) => `/simulation/results/${id}`,
  NETWORK_METRICS: '/analysis/network/metrics',
  EQUILIBRIUM_CALCULATE: '/equilibrium/calculate',
  GAME_THEORY_ANALYSIS: '/game-theory/analyze',
  PAYOFF_MATRIX: '/game-theory/payoff-matrix',
  STRATEGY_EVOLUTION: '/simulation/strategy-evolution',
} as const;

export const DEFAULT_SIMULATION_PARAMS: GameParameters = {
  metadata: {
    name: 'Default Simulation',
    description: 'A basic simulation with default parameters',
    tags: ['default', 'basic'],
    priority: 'normal'
  },
  numPlayers: {
    spreaders: 10,
    factCheckers: 5,
    platforms: 1,
    users: 100
  },
  network: {
    size: 1000,
    type: 'small_world',
    parameters: {
      connectionProbability: 0.1,
      averageDegree: 6,
      clusteringCoefficient: 0.3,
      rewireProbability: 0.1
    }
  },
  dynamics: {
    timeHorizon: 50,
    timeStep: 1,
    maxRounds: 100,
    propagationRate: 0.1,
    decayRate: 0.05,
    viralThreshold: 0.3,
    detectionAccuracy: 0.8,
    detectionDelay: 2,
    factCheckingEffectiveness: 0.9,
    learningRate: 0.1,
    memoryLength: 10,
    adaptationThreshold: 0.2
  },
  payoffWeights: {
    spreaders: {
      reach: 1.0,
      detection: -0.5,
      reputation: -0.3,
      cost: -0.1
    },
    factCheckers: {
      accuracy: 1.0,
      effort: -0.2,
      impact: 0.5,
      reputation: 0.3
    },
    platforms: {
      engagement: 0.8,
      reputation: -0.6,
      cost: -0.1,
      regulation: -0.4
    },
  },
  environment: {
    events: [],
    cycles: {
      newsVolume: Array.from({ length: 24 }, (_, i) => 0.5 + 0.3 * Math.sin(i * Math.PI / 12)),
      userActivity: Array.from({ length: 24 }, (_, i) => 0.6 + 0.4 * Math.sin((i - 6) * Math.PI / 12)),
      platformAttention: Array.from({ length: 24 }, (_, i) => 0.7 + 0.3 * Math.cos(i * Math.PI / 12))
    },
    uncertainty: {
      payoffNoise: 0.1,
      informationNoise: 0.05,
      behaviorNoise: 0.02
    }
  },
  advanced: {
    randomSeed: 42,
    parallelization: {
      enabled: false,
      workers: 1,
      batchSize: 100
    },
    output: {
      saveInterval: 10,
      saveDetailedHistory: true,
      saveNetworkStates: false,
      savePlayerStates: true,
      compression: false
    },
    stoppingCriteria: {
      convergenceThreshold: 0.001,
      maxIterationsWithoutChange: 100,
      minPayoffImprovement: 0.01
    }
  }
};

export const NODE_COLORS: Record<string, string> = {
  spreader: '#FF6B6B',
  fact_checker: '#4ECDC4',
  platform: '#F9CA24',
  user: '#95A5A6',
  default: '#45B7D1',
} as const;

export const EDGE_COLORS: Record<string, string> = {
  information: '#3498DB',
  misinformation: '#E74C3C',
  fact_check: '#2ECC71',
  default: '#BDC3C7',
} as const;

export const SIMULATION_STATUS_COLORS: Record<string, string> = {
  idle: '#95A5A6',
  configuring: '#3498DB',
  starting: '#F39C12',
  running: '#2ECC71',
  paused: '#F1C40F',
  completed: '#27AE60',
  error: '#E74C3C',
  cancelled: '#95A5A6',
} as const;

export const PLAYER_STRATEGY_COLORS: Record<string, string> = {
  // Spreader strategies
  aggressive: '#E74C3C',
  moderate: '#F39C12',
  cautious: '#F1C40F',
  inactive: '#95A5A6',

  // Fact checker strategies
  active: '#2ECC71',
  reactive: '#27AE60',
  selective: '#16A085',
  passive: '#95A5A6',

  // Platform strategies
  strict_moderation: '#8E44AD',
  moderate_moderation: '#9B59B6',
  minimal_moderation: '#BDC3C7',
  no_moderation: '#95A5A6',
} as const;

export const VISUALIZATION_CONFIG = {
  network: {
    nodeSize: {
      min: 5,
      max: 20,
      default: 10
    },
    linkDistance: {
      min: 30,
      max: 100,
      default: 50
    },
    charge: {
      min: -300,
      max: -30,
      default: -100
    }
  },
  simulation: {
    fps: 60,
    animationDuration: 1000,
    updateInterval: 100
  },
  chart: {
    margins: {
      top: 20,
      right: 30,
      bottom: 40,
      left: 40
    },
    colors: {
      primary: '#3498DB',
      secondary: '#2ECC71',
      accent: '#F39C12',
      warning: '#E74C3C'
    }
  }
} as const;

export const DEFAULT_FORM_VALUES = {
  simulation: {
    networkSize: 1000,
    timeHorizon: 50,
    propagationRate: 0.1,
    detectionAccuracy: 0.8
  },
  players: {
    spreaders: 10,
    factCheckers: 5,
    platforms: 1
  }
} as const;

export const VALIDATION_LIMITS = {
  networkSize: { min: 10, max: 10000 },
  timeHorizon: { min: 5, max: 1000 },
  propagationRate: { min: 0.01, max: 1.0 },
  detectionAccuracy: { min: 0.1, max: 1.0 },
  players: {
    spreaders: { min: 1, max: 100 },
    factCheckers: { min: 0, max: 100 },
    platforms: { min: 1, max: 10 }
  }
} as const;