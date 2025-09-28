// frontend/src/types/gameTheory.ts

/**
 * TypeScript interfaces for game theory concepts, payoff matrices,
 * equilibrium calculations, and strategy analysis.
 * These types ensure proper data structure for game theory components.
 */

/**
 * Player types in the fake news ecosystem
 */
export type PlayerType = 'spreader' | 'fact_checker' | 'platform' | 'user';

/**
 * Strategy types for different players
 */
export type SpreaderStrategy = 'aggressive' | 'moderate' | 'cautious' | 'inactive';
export type FactCheckerStrategy = 'active' | 'reactive' | 'selective' | 'passive';
export type PlatformStrategy = 'strict_moderation' | 'moderate_moderation' | 'minimal_moderation' | 'no_moderation';

/**
 * All possible strategies in the game
 */
export type Strategy = SpreaderStrategy | FactCheckerStrategy | PlatformStrategy;

/**
 * Nash Equilibrium definition
 * Represents a state where no player can improve their payoff by unilaterally changing strategy
 */
export interface Equilibrium {
  /** Strategy indices for each player [player1_strategy_index, player2_strategy_index] */
  strategies: [number, number];
  /** Payoffs for each player at this equilibrium */
  payoffs: Record<string, number>;
  /** Whether this is a pure or mixed strategy equilibrium */
  type: 'pure' | 'mixed';
  /** Stability measure of this equilibrium */
  stability: number;
  /** Nash equilibrium type classification */
  classification: 'strict' | 'weak' | 'trembling_hand_perfect';
}

/**
 * Mixed strategy equilibrium with probabilities
 */
export interface MixedStrategyEquilibrium {
  /** Player name */
  player: string;
  /** Strategy probabilities (must sum to 1) */
  probabilities: number[];
  /** Expected payoff for this mixed strategy */
  expectedPayoff: number;
  /** Variance in payoffs */
  payoffVariance: number;
}

/**
 * Payoff matrix data structure for visualization
 * Used by PayoffMatrix.tsx component
 */
export interface PayoffMatrixData {
  /** Names of the two players */
  players: [string, string];
  /** Available strategies for each player */
  strategies: Record<string, string[]>;
  /** Payoff values for each strategy combination */
  payoffs: { [player: string]: number }[][];
  /** Nash equilibrium to highlight in the matrix */
  equilibrium?: Equilibrium;
  /** Additional metadata */
  metadata?: {
    gameType: string;
    description?: string;
    createdAt: string;
    parameters?: Record<string, unknown>;
  };
}

/**
 * Multi-player payoff matrix for games with more than 2 players
 */
export interface MultiPlayerPayoffMatrix {
  /** Names of all players */
  players: string[];
  /** Available strategies for each player */
  strategies: Record<string, string[]>;
  /** N-dimensional payoff array */
  payoffs: number[][][]; // [strategy_combination][player][payoff]
  /** Multiple equilibria possible in multi-player games */
  equilibria: Equilibrium[];
}

/**
 * Game theory analysis results
 */
export interface GameAnalysis {
  /** All Nash equilibria found */
  equilibria: Equilibrium[];
  /** Mixed strategy equilibria */
  mixedEquilibria: MixedStrategyEquilibrium[];
  /** Dominant strategies for each player */
  dominantStrategies: Record<string, {
    strategy: string;
    type: 'strictly_dominant' | 'weakly_dominant';
  }>;
  /** Dominated strategies that can be eliminated */
  dominatedStrategies: Record<string, string[]>;
  /** Pareto optimal outcomes */
  paretoOptimal: Array<{
    strategies: number[];
    payoffs: Record<string, number>;
    efficiency: number;
  }>;
  /** Social welfare analysis */
  socialWelfare: {
    maximum: number;
    minimum: number;
    equilibriumWelfare: number;
    efficiency: number; // ratio of equilibrium welfare to maximum
  };
}

/**
 * Player behavior and strategy preferences
 */
export interface PlayerProfile {
  /** Player identifier */
  id: string;
  /** Player type */
  type: PlayerType;
  /** Risk preference */
  riskPreference: 'risk_averse' | 'risk_neutral' | 'risk_seeking';
  /** Learning parameters */
  learning: {
    rate: number;
    memory: number; // how many past rounds to remember
    adaptability: number; // how quickly to change strategies
  };
  /** Current strategy */
  currentStrategy: Strategy;
  /** Strategy history */
  strategyHistory: Array<{
    round: number;
    strategy: Strategy;
    payoff: number;
    reason?: string;
  }>;
  /** Performance metrics */
  performance: {
    averagePayoff: number;
    totalPayoff: number;
    winRate: number;
    strategySwitches: number;
  };
}

/**
 * Game configuration and parameters
 */
export interface GameConfiguration {
  /** Game identifier */
  id: string;
  /** Game name/description */
  name: string;
  /** Game type */
  type: 'symmetric' | 'asymmetric' | 'zero_sum' | 'non_zero_sum';
  /** Number of players */
  playerCount: number;
  /** Player types and their counts */
  playerTypes: Record<PlayerType, number>;
  /** Payoff structure */
  payoffStructure: {
    /** Base payoffs for different outcomes */
    basePayoffs: Record<string, Record<string, number>>;
    /** Dynamic payoff modifiers */
    modifiers: {
      networkEffect: number;
      reputationEffect: number;
      learningEffect: number;
      timeDecay: number;
    };
  };
  /** Game rules and constraints */
  rules: {
    simultaneousPlay: boolean;
    informationStructure: 'complete' | 'incomplete' | 'imperfect';
    repetition: {
      finite: boolean;
      rounds?: number;
      discountFactor?: number;
    };
  };
}

/**
 * Strategy evolution over time
 */
export interface StrategyEvolution {
  /** Round number */
  round: number;
  /** Strategy distributions for each player type */
  distributions: Record<PlayerType, Record<Strategy, number>>;
  /** Average payoffs by strategy */
  averagePayoffs: Record<Strategy, number>;
  /** Convergence metrics */
  convergence: {
    /** Convergence to equilibrium measure */
    equilibriumDistance: number;
    /** Strategy variance (measure of stability) */
    strategyVariance: number;
    /** Whether strategies have converged */
    hasConverged: boolean;
  };
}

/**
 * Comparative statics analysis
 */
export interface ComparativeStatics {
  /** Parameter being varied */
  parameter: string;
  /** Parameter values tested */
  parameterValues: number[];
  /** Equilibria for each parameter value */
  equilibriaByParameter: Equilibrium[][];
  /** Welfare analysis for each parameter value */
  welfareByParameter: number[];
  /** Sensitivity measures */
  sensitivity: {
    /** How much equilibrium strategies change */
    strategyElasticity: number[];
    /** How much payoffs change */
    payoffElasticity: number[];
    /** Critical parameter values where equilibria change */
    criticalPoints: number[];
  };
}

/**
 * Robustness analysis results
 */
export interface RobustnessAnalysis {
  /** Base equilibrium being tested */
  baseEquilibrium: Equilibrium;
  /** Perturbations tested */
  perturbations: Array<{
    type: 'payoff' | 'strategy' | 'player_count';
    magnitude: number;
    description: string;
  }>;
  /** Stability measures */
  stability: {
    /** Trembling hand perfect */
    tremblingHandPerfect: boolean;
    /** Evolutionarily stable */
    evolutionarilyStable: boolean;
    /** Risk dominance */
    riskDominant: boolean;
    /** Payoff dominance */
    payoffDominant: boolean;
  };
  /** Robustness scores */
  robustnessScores: {
    overall: number;
    structural: number;
    parametric: number;
    behavioral: number;
  };
}

/**
 * Mechanism design elements
 */
export interface MechanismDesign {
  /** Mechanism type */
  type: 'auction' | 'voting' | 'matching' | 'regulation';
  /** Objectives */
  objectives: Array<'efficiency' | 'fairness' | 'revenue' | 'participation'>;
  /** Design parameters */
  parameters: {
    /** Information requirements */
    informationRequirement: 'truthful' | 'strategic';
    /** Participation constraints */
    participationConstraints: Record<PlayerType, number>;
    /** Budget balance */
    budgetBalance: boolean;
    /** Individual rationality */
    individualRationality: boolean;
  };
  /** Performance metrics */
  performance: {
    efficiency: number;
    fairness: number;
    revenue: number;
    participation: number;
  };
}

/**
 * Experimental economics data
 */
export interface ExperimentalData {
  /** Experiment identifier */
  experimentId: string;
  /** Treatment conditions */
  treatment: string;
  /** Participant data */
  participants: Array<{
    id: string;
    demographics: Record<string, unknown>;
    strategies: Strategy[];
    payoffs: number[];
    surveyResponses?: Record<string, unknown>;
  }>;
  /** Aggregate results */
  aggregateResults: {
    convergenceTime: number;
    finalDistribution: Record<Strategy, number>;
    averagePayoff: number;
    efficiency: number;
  };
  /** Statistical tests */
  statisticalTests: {
    equilibriumTest: {
      pValue: number;
      rejected: boolean;
    };
    learningTest: {
      pValue: number;
      rejected: boolean;
    };
    treatmentEffects: Record<string, {
      estimate: number;
      standardError: number;
      pValue: number;
    }>;
  };
}