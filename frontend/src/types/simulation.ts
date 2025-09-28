// frontend/src/types/simulation.ts

/**
 * Types for the simulation workflow, from user-configurable parameters
 * to final results. Used extensively in simulation state management
 * and the simulation dashboard.
 */

import type { PlayerType, Strategy, Equilibrium, StrategyEvolution } from './gameTheory';
import type { NetworkData } from './network';

/**
 * Current state of a simulation
 */
export type SimulationState = 'idle' | 'configuring' | 'starting' | 'running' | 'paused' | 'completed' | 'error' | 'cancelled';

/**
 * Simulation priority levels
 */
export type SimulationPriority = 'low' | 'normal' | 'high' | 'urgent';

/**
 * Game parameters for configuring a simulation
 */
export interface GameParameters {
  /** Simulation metadata */
  metadata: {
    name: string;
    description?: string;
    tags: string[];
    priority: SimulationPriority;
  };

  /** Player configuration */
  numPlayers: {
    spreaders: number;
    factCheckers: number;
    platforms: number;
    users?: number;
  };

  /** Network configuration */
  network: {
    size: number;
    type: 'random' | 'small_world' | 'scale_free' | 'complete' | 'custom';
    parameters: {
      connectionProbability?: number;
      averageDegree?: number;
      clusteringCoefficient?: number;
      powerLawExponent?: number;
      rewireProbability?: number;
    };
    seedNetwork?: NetworkData;
  };

  /** Game dynamics */
  dynamics: {
    /** Simulation time parameters */
    timeHorizon: number;
    timeStep: number;
    maxRounds?: number;

    /** Information propagation */
    propagationRate: number;
    decayRate: number;
    viralThreshold: number;

    /** Detection and fact-checking */
    detectionAccuracy: number;
    detectionDelay: number;
    factCheckingEffectiveness: number;

    /** Learning and adaptation */
    learningRate: number;
    memoryLength: number;
    adaptationThreshold: number;
  };

  /** Payoff weights for different player types */
  payoffWeights: {
    spreaders: {
      reach: number;
      detection: number;
      reputation: number;
      cost: number;
    };
    factCheckers: {
      accuracy: number;
      effort: number;
      impact: number;
      reputation: number;
    };
    platforms: {
      engagement: number;
      reputation: number;
      cost: number;
      regulation: number;
    };
  };

  /** Environmental factors */
  environment: {
    /** External events that affect the game */
    events: Array<{
      type: 'crisis' | 'regulation' | 'technology' | 'social';
      timing: number;
      duration: number;
      impact: Record<string, number>;
    }>;

    /** Seasonal or cyclical effects */
    cycles: {
      newsVolume: number[];
      userActivity: number[];
      platformAttention: number[];
    };

    /** Noise and uncertainty */
    uncertainty: {
      payoffNoise: number;
      informationNoise: number;
      behaviorNoise: number;
    };
  };

  /** Advanced configuration */
  advanced: {
    /** Random seed for reproducibility */
    randomSeed?: number;

    /** Parallel processing options */
    parallelization: {
      enabled: boolean;
      workers?: number;
      batchSize?: number;
    };

    /** Output configuration */
    output: {
      saveInterval: number;
      saveDetailedHistory: boolean;
      saveNetworkStates: boolean;
      savePlayerStates: boolean;
      compression: boolean;
    };

    /** Early stopping conditions */
    stoppingCriteria: {
      convergenceThreshold: number;
      maxIterationsWithoutChange: number;
      minPayoffImprovement: number;
    };
  };
}

/**
 * Simulation results after completion
 */
export interface SimulationResults {
  /** Simulation metadata */
  simulationId: string;
  parameters: GameParameters;
  status: SimulationState;

  /** Timing information */
  timing: {
    startTime: string;
    endTime: string;
    duration: number; // in seconds
    realTimeRatio: number; // simulation time / real time
  };

  /** Final equilibrium state */
  equilibrium: {
    strategies: Record<PlayerType, Record<Strategy, number>>;
    payoffs: Record<PlayerType, number>;
    stability: number;
    convergenceRound: number;
    isNashEquilibrium: boolean;
  };

  /** Propagation statistics */
  propagationStats: {
    /** Total content items */
    totalContent: number;
    fakeNewsItems: number;
    realNewsItems: number;

    /** Spread metrics */
    averageReach: number;
    maxReach: number;
    totalViews: number;

    /** Detection metrics */
    detectionRate: number;
    falsePositiveRate: number;
    falseNegativeRate: number;
    averageDetectionTime: number;

    /** Platform intervention */
    contentRemoved: number;
    accountsSuspended: number;
    warningsIssued: number;
  };

  /** Player performance */
  playerPerformance: {
    /** Performance by player type */
    byType: Record<PlayerType, {
      averagePayoff: number;
      totalPayoff: number;
      payoffVariance: number;
      strategyDistribution: Record<Strategy, number>;
      learningCurve: number[];
    }>;

    /** Individual player rankings */
    rankings: Array<{
      playerId: string;
      playerType: PlayerType;
      finalPayoff: number;
      rank: number;
      percentile: number;
    }>;
  };

  /** Network evolution */
  networkEvolution: {
    /** Network metrics over time */
    metrics: Array<{
      round: number;
      density: number;
      clustering: number;
      pathLength: number;
      modularity: number;
      centralization: number;
    }>;

    /** Information flow patterns */
    informationFlow: {
      mainChannels: Array<{
        source: string;
        target: string;
        strength: number;
        contentType: 'fake' | 'real' | 'mixed';
      }>;
      bottlenecks: string[];
      influencers: string[];
    };
  };

  /** Strategy evolution over time */
  strategyEvolution: StrategyEvolution[];

  /** Welfare analysis */
  welfare: {
    totalWelfare: number;
    paretoEfficiency: number;
    inequalityIndex: number;
    socialOptimum: number;
    lossFromNonCooperation: number;
  };

  /** Statistical analysis */
  statistics: {
    /** Descriptive statistics */
    descriptive: {
      payoffMean: number;
      payoffStd: number;
      payoffSkewness: number;
      payoffKurtosis: number;
    };

    /** Hypothesis tests */
    tests: {
      equilibriumTest: {
        statistic: number;
        pValue: number;
        conclusion: string;
      };
      strategyIndependence: {
        chiSquare: number;
        pValue: number;
        degreesOfFreedom: number;
      };
    };

    /** Confidence intervals */
    confidenceIntervals: {
      payoffs: Record<PlayerType, { lower: number; upper: number; confidence: number }>;
      strategies: Record<Strategy, { lower: number; upper: number; confidence: number }>;
    };
  };

  /** Sensitivity analysis */
  sensitivity: {
    parameterSensitivity: Record<string, {
      elasticity: number;
      criticalValues: number[];
      robustness: number;
    }>;
  };
}

/**
 * Real-time simulation state during execution
 */
export interface SimulationRealTimeState {
  /** Current simulation step */
  currentRound: number;
  totalRounds: number;
  progress: number; // 0-1

  /** Performance metrics */
  performance: {
    roundsPerSecond: number;
    estimatedTimeRemaining: number;
    memoryUsage: number;
    cpuUsage: number;
  };

  /** Current state snapshot */
  currentState: {
    strategies: Record<PlayerType, Record<Strategy, number>>;
    payoffs: Record<PlayerType, number>;
    networkState: {
      activeConnections: number;
      informationFlow: number;
      contentItems: number;
    };
  };

  /** Recent events */
  recentEvents: Array<{
    round: number;
    type: 'strategy_change' | 'content_posted' | 'content_detected' | 'platform_action';
    description: string;
    impact: 'low' | 'medium' | 'high';
    players?: string[];
  }>;

  /** Convergence indicators */
  convergence: {
    strategyStability: number;
    payoffStability: number;
    networkStability: number;
    convergenceScore: number;
    hasConverged: boolean;
  };
}

/**
 * Simulation history entry
 */
export interface SimulationHistoryEntry {
  simulationId: string;
  name: string;
  description?: string;
  parameters: Partial<GameParameters>;
  results?: Partial<SimulationResults>;
  status: SimulationState;
  createdAt: string;
  completedAt?: string;
  duration?: number;
  tags: string[];
  favorite: boolean;
  shared: boolean;
}

/**
 * Simulation comparison data
 */
export interface SimulationComparison {
  simulations: SimulationHistoryEntry[];
  comparison: {
    /** Parameter differences */
    parameterDiffs: Record<string, {
      values: unknown[];
      significant: boolean;
    }>;

    /** Result differences */
    resultDiffs: {
      equilibria: Equilibrium[];
      welfareComparison: number[];
      strategyDistributions: Record<string, Record<Strategy, number>[]>;
    };

    /** Statistical comparison */
    statistics: {
      significantDifferences: string[];
      effectSizes: Record<string, number>;
      recommendations: string[];
    };
  };
}

/**
 * Simulation batch job configuration
 */
export interface BatchSimulationConfig {
  /** Batch metadata */
  batchId: string;
  name: string;
  description?: string;

  /** Base parameters */
  baseParameters: GameParameters;

  /** Parameter variations */
  variations: Array<{
    name: string;
    parameter: string;
    values: unknown[];
    type: 'sweep' | 'random' | 'grid';
  }>;

  /** Execution configuration */
  execution: {
    maxConcurrent: number;
    priority: SimulationPriority;
    timeout: number;
    retries: number;
    saveResults: boolean;
  };

  /** Analysis configuration */
  analysis: {
    autoAnalyze: boolean;
    comparisons: string[];
    generateReport: boolean;
    notifyOnCompletion: boolean;
  };
}

/**
 * Simulation template for common configurations
 */
export interface SimulationTemplate {
  id: string;
  name: string;
  description: string;
  category: 'basic' | 'advanced' | 'research' | 'educational';
  parameters: GameParameters;
  metadata: {
    author: string;
    version: string;
    createdAt: string;
    updatedAt: string;
    downloads: number;
    rating: number;
    tags: string[];
  };
  validation: {
    tested: boolean;
    expectedResults?: Partial<SimulationResults>;
    notes?: string;
  };
}

/**
 * Simulation queue entry
 */
export interface SimulationQueueEntry {
  id: string;
  simulationId: string;
  parameters: GameParameters;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: SimulationPriority;
  queuedAt: string;
  startedAt?: string;
  estimatedDuration: number;
  actualDuration?: number;
  position: number;
  resourceRequirements: {
    cpu: number;
    memory: number;
    storage: number;
  };
}

/**
 * Simulation metrics for monitoring and analytics
 */
export interface SimulationMetrics {
  /** System metrics */
  system: {
    totalSimulations: number;
    activeSimulations: number;
    queuedSimulations: number;
    completedSimulations: number;
    failedSimulations: number;
  };

  /** Performance metrics */
  performance: {
    averageDuration: number;
    averageRoundsPerSecond: number;
    resourceUtilization: {
      cpu: number;
      memory: number;
      storage: number;
    };
  };

  /** Usage metrics */
  usage: {
    popularTemplates: string[];
    commonParameters: Record<string, unknown>;
    userActivity: {
      dailyActiveUsers: number;
      simulationsPerUser: number;
    };
  };

  /** Quality metrics */
  quality: {
    convergenceRate: number;
    averageAccuracy: number;
    userSatisfaction: number;
    errorRate: number;
  };
}