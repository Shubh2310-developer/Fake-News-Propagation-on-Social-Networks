// frontend/tests/utils/gameTheory.test.ts

import {
  formatMixedStrategyEquilibrium,
  findBestResponse,
  strictlyDominates,
  weaklyDominates,
  calculateExpectedPayoff,
  isNashEquilibrium,
  formatPayoff,
  calculateSocialWelfare,
  classifyGame,
  strategyNamesToIndices,
  strategyIndicesToNames,
  validateMixedStrategy,
} from '@/lib/gameTheory';

describe('Game Theory Utilities', () => {
  describe('formatMixedStrategyEquilibrium', () => {
    it('formats mixed strategy equilibrium correctly', () => {
      const equilibrium: [number[], number[]] = [
        [0.6, 0.4],
        [0.3, 0.7]
      ];
      const strategies = {
        spreader: ['Aggressive', 'Moderate'],
        fact_checker: ['Active', 'Passive']
      };

      const result = formatMixedStrategyEquilibrium(equilibrium, strategies);
      expect(result).toBe('Player 1 plays (60.0% Aggressive, 40.0% Moderate) and Player 2 plays (30.0% Active, 70.0% Passive).');
    });

    it('handles missing strategy names gracefully', () => {
      const equilibrium: [number[], number[]] = [
        [0.5, 0.5],
        [0.8, 0.2]
      ];
      const strategies = {};

      const result = formatMixedStrategyEquilibrium(equilibrium, strategies);
      expect(result).toBe('Player 1 plays (50.0% Strategy 1, 50.0% Strategy 2) and Player 2 plays (80.0% Strategy 1, 20.0% Strategy 2).');
    });

    it('uses fallback player names when standard names not found', () => {
      const equilibrium: [number[], number[]] = [
        [1.0, 0.0],
        [0.0, 1.0]
      ];
      const strategies = {
        player1: ['Strategy A', 'Strategy B'],
        player2: ['Strategy X', 'Strategy Y']
      };

      const result = formatMixedStrategyEquilibrium(equilibrium, strategies);
      expect(result).toBe('Player 1 plays (100.0% Strategy A, 0.0% Strategy B) and Player 2 plays (0.0% Strategy X, 100.0% Strategy Y).');
    });
  });

  describe('findBestResponse', () => {
    it('finds best response for given opponent strategy', () => {
      const payoffMatrix = [
        [3, 0],
        [5, 1],
        [6, 4]
      ];

      // If opponent plays strategy 0, best response is strategy 2 (payoff 6)
      expect(findBestResponse(payoffMatrix, 0)).toBe(2);

      // If opponent plays strategy 1, best response is strategy 2 (payoff 4)
      expect(findBestResponse(payoffMatrix, 1)).toBe(2);
    });

    it('returns first strategy when all payoffs are equal', () => {
      const payoffMatrix = [
        [2, 2],
        [2, 2],
        [2, 2]
      ];

      expect(findBestResponse(payoffMatrix, 0)).toBe(0);
      expect(findBestResponse(payoffMatrix, 1)).toBe(0);
    });

    it('handles single strategy case', () => {
      const payoffMatrix = [[5, 3]];

      expect(findBestResponse(payoffMatrix, 0)).toBe(0);
      expect(findBestResponse(payoffMatrix, 1)).toBe(0);
    });
  });

  describe('strictlyDominates', () => {
    it('returns true when strategy1 strictly dominates strategy2', () => {
      const strategy1 = [3, 4, 5];
      const strategy2 = [2, 3, 4];

      expect(strictlyDominates(strategy1, strategy2)).toBe(true);
    });

    it('returns false when strategies are equal', () => {
      const strategy1 = [3, 4, 5];
      const strategy2 = [3, 4, 5];

      expect(strictlyDominates(strategy1, strategy2)).toBe(false);
    });

    it('returns false when strategy2 is better in some outcomes', () => {
      const strategy1 = [3, 2, 5];
      const strategy2 = [2, 4, 4];

      expect(strictlyDominates(strategy1, strategy2)).toBe(false);
    });

    it('returns false for different length arrays', () => {
      const strategy1 = [3, 4];
      const strategy2 = [2, 3, 4];

      expect(strictlyDominates(strategy1, strategy2)).toBe(false);
    });

    it('returns false when strategy1 only weakly dominates', () => {
      const strategy1 = [3, 4, 5];
      const strategy2 = [3, 4, 5]; // Equal - weakly dominates but not strictly

      expect(strictlyDominates(strategy1, strategy2)).toBe(false);
    });
  });

  describe('weaklyDominates', () => {
    it('returns true when strategy1 weakly dominates strategy2', () => {
      const strategy1 = [3, 4, 5];
      const strategy2 = [3, 3, 4];

      expect(weaklyDominates(strategy1, strategy2)).toBe(true);
    });

    it('returns true when strategies are equal', () => {
      const strategy1 = [3, 4, 5];
      const strategy2 = [3, 4, 5];

      expect(weaklyDominates(strategy1, strategy2)).toBe(true);
    });

    it('returns false when strategy2 is better in any outcome', () => {
      const strategy1 = [3, 2, 5];
      const strategy2 = [2, 4, 4];

      expect(weaklyDominates(strategy1, strategy2)).toBe(false);
    });

    it('returns false for different length arrays', () => {
      const strategy1 = [3, 4];
      const strategy2 = [2, 3, 4];

      expect(weaklyDominates(strategy1, strategy2)).toBe(false);
    });
  });

  describe('calculateExpectedPayoff', () => {
    it('calculates expected payoff against pure strategy', () => {
      const mixedStrategy = [0.4, 0.6];
      const payoffMatrix = [
        [3, 1],
        [0, 2]
      ];
      const opponentStrategy = 0;

      const expected = 0.4 * 3 + 0.6 * 0; // 1.2
      expect(calculateExpectedPayoff(mixedStrategy, payoffMatrix, opponentStrategy)).toBe(expected);
    });

    it('calculates expected payoff against mixed strategy', () => {
      const mixedStrategy = [0.5, 0.5];
      const payoffMatrix = [
        [3, 1],
        [0, 2]
      ];
      const opponentMixedStrategy = [0.3, 0.7];

      // Expected payoff = 0.5 * 0.3 * 3 + 0.5 * 0.7 * 1 + 0.5 * 0.3 * 0 + 0.5 * 0.7 * 2
      const expected = 0.5 * 0.3 * 3 + 0.5 * 0.7 * 1 + 0.5 * 0.3 * 0 + 0.5 * 0.7 * 2;
      expect(calculateExpectedPayoff(mixedStrategy, payoffMatrix, opponentMixedStrategy)).toBe(expected);
    });

    it('handles edge case with zero probabilities', () => {
      const mixedStrategy = [1.0, 0.0];
      const payoffMatrix = [
        [4, 2],
        [1, 3]
      ];
      const opponentStrategy = 1;

      expect(calculateExpectedPayoff(mixedStrategy, payoffMatrix, opponentStrategy)).toBe(2);
    });
  });

  describe('isNashEquilibrium', () => {
    it('identifies valid Nash equilibrium', () => {
      // Classic coordination game - (1,1) is Nash equilibrium
      const strategyProfile = [1, 1]; // Both choose strategy 1
      const payoffMatrices = [
        [
          [1, 0], // Strategy 0: (1 vs 0, 0 vs 1)
          [0, 2]  // Strategy 1: (0 vs 0, 2 vs 1)
        ],
        [
          [1, 0], // Strategy 0: (1 vs 0, 0 vs 1)
          [0, 2]  // Strategy 1: (0 vs 0, 2 vs 1)
        ]
      ];

      expect(isNashEquilibrium(strategyProfile, payoffMatrices)).toBe(true);
    });

    it('identifies non-Nash equilibrium', () => {
      // (Cooperate, Cooperate) is not Nash equilibrium in prisoners' dilemma
      const strategyProfile = [0, 0]; // Both cooperate
      const payoffMatrices = [
        [
          [3, 0],
          [5, 1]
        ],
        [
          [3, 5],
          [0, 1]
        ]
      ];

      expect(isNashEquilibrium(strategyProfile, payoffMatrices)).toBe(false);
    });

    it('handles coordination game Nash equilibrium', () => {
      // Coordination game - (1,1) is Nash equilibrium
      const strategyProfile = [1, 1];
      const payoffMatrices = [
        [
          [1, 0],
          [0, 2]
        ],
        [
          [1, 0],
          [0, 2]
        ]
      ];

      expect(isNashEquilibrium(strategyProfile, payoffMatrices)).toBe(true);
    });
  });

  describe('formatPayoff', () => {
    it('formats normal payoffs correctly', () => {
      expect(formatPayoff(3.14159, 2)).toBe('3.14');
      expect(formatPayoff(1.5, 1)).toBe('1.5');
      expect(formatPayoff(-2.75, 2)).toBe('-2.75');
    });

    it('formats very small values as zero', () => {
      expect(formatPayoff(0.005)).toBe('0');
      expect(formatPayoff(-0.009)).toBe('0');
      expect(formatPayoff(0.0001)).toBe('0');
    });

    it('uses default precision when not specified', () => {
      expect(formatPayoff(3.14159)).toBe('3.14');
      expect(formatPayoff(2.7)).toBe('2.70');
    });

    it('handles zero correctly', () => {
      expect(formatPayoff(0)).toBe('0');
      expect(formatPayoff(0.0)).toBe('0');
    });

    it('handles large numbers', () => {
      expect(formatPayoff(1000.5678, 2)).toBe('1000.57');
      expect(formatPayoff(-999.12345, 3)).toBe('-999.123');
    });
  });

  describe('calculateSocialWelfare', () => {
    it('calculates total welfare for two-player game', () => {
      const strategyProfile = [0, 1];
      const payoffMatrices = [
        [
          [3, 1],
          [0, 2]
        ],
        [
          [2, 0],
          [1, 3]
        ]
      ];

      // Player 1 payoff at (0,1): 1
      // Player 2 payoff at (0,1): 1 (from second matrix [1,3], player 2's perspective)
      // Total welfare: 2
      expect(calculateSocialWelfare(strategyProfile, payoffMatrices)).toBe(2);
    });

    it('calculates welfare for different strategy combinations', () => {
      const strategyProfile = [1, 0];
      const payoffMatrices = [
        [
          [3, 1],
          [4, 2]
        ],
        [
          [3, 4],
          [1, 2]
        ]
      ];

      // Player 1 payoff at (1,0): 4
      // Player 2 payoff at (1,0): 4
      // Total welfare: 8
      expect(calculateSocialWelfare(strategyProfile, payoffMatrices)).toBe(8);
    });

    it('handles negative payoffs', () => {
      const strategyProfile = [0, 0];
      const payoffMatrices = [
        [
          [-1, 2],
          [0, 1]
        ],
        [
          [-2, 1],
          [1, 0]
        ]
      ];

      // Player 1 payoff at (0,0): -1
      // Player 2 payoff at (0,0): -2
      // Total welfare: -3
      expect(calculateSocialWelfare(strategyProfile, payoffMatrices)).toBe(-3);
    });
  });

  describe('classifyGame', () => {
    it('identifies zero-sum game', () => {
      const payoffMatrices = [
        [
          [1, -1],
          [-1, 1]
        ],
        [
          [-1, 1],
          [1, -1]
        ]
      ];

      expect(classifyGame(payoffMatrices)).toBe('zero-sum');
    });

    it('identifies symmetric zero-sum game', () => {
      const payoffMatrices = [
        [
          [0, -1],
          [1, 0]
        ],
        [
          [0, 1],
          [-1, 0]
        ]
      ];

      expect(classifyGame(payoffMatrices)).toBe('symmetric zero-sum');
    });

    it('identifies symmetric non-zero-sum game', () => {
      const payoffMatrices = [
        [
          [3, 0],
          [5, 1]
        ],
        [
          [3, 5],
          [0, 1]
        ]
      ];

      expect(classifyGame(payoffMatrices)).toBe('symmetric');
    });

    it('identifies asymmetric game', () => {
      const payoffMatrices = [
        [
          [2, 1],
          [0, 3]
        ],
        [
          [1, 0],
          [2, 1]
        ]
      ];

      expect(classifyGame(payoffMatrices)).toBe('asymmetric');
    });

    it('identifies multi-player game', () => {
      const payoffMatrices = [
        [[[1, 2]], [[0, 1]]],
        [[[2, 1]], [[1, 0]]],
        [[[1, 1]], [[2, 2]]]
      ];

      expect(classifyGame(payoffMatrices)).toBe('multi-player');
    });
  });

  describe('strategyNamesToIndices', () => {
    it('converts strategy names to indices', () => {
      const strategyNames = ['Aggressive', 'Passive', 'Moderate'];
      const availableStrategies = {
        'Aggressive': 0,
        'Moderate': 1,
        'Passive': 2,
        'Defensive': 3
      };

      expect(strategyNamesToIndices(strategyNames, availableStrategies)).toEqual([0, 2, 1]);
    });

    it('uses default index 0 for unknown strategies', () => {
      const strategyNames = ['Unknown', 'Aggressive'];
      const availableStrategies = {
        'Aggressive': 1,
        'Moderate': 2
      };

      expect(strategyNamesToIndices(strategyNames, availableStrategies)).toEqual([0, 1]);
    });

    it('handles empty arrays', () => {
      expect(strategyNamesToIndices([], {})).toEqual([]);
    });
  });

  describe('strategyIndicesToNames', () => {
    it('converts strategy indices to names', () => {
      const strategyIndices = [0, 2, 1];
      const availableStrategies = ['Aggressive', 'Moderate', 'Passive', 'Defensive'];

      expect(strategyIndicesToNames(strategyIndices, availableStrategies)).toEqual(['Aggressive', 'Passive', 'Moderate']);
    });

    it('uses "Unknown" for invalid indices', () => {
      const strategyIndices = [0, 5, 1];
      const availableStrategies = ['Aggressive', 'Moderate'];

      expect(strategyIndicesToNames(strategyIndices, availableStrategies)).toEqual(['Aggressive', 'Unknown', 'Moderate']);
    });

    it('handles empty arrays', () => {
      expect(strategyIndicesToNames([], [])).toEqual([]);
    });
  });

  describe('validateMixedStrategy', () => {
    it('validates correct mixed strategy', () => {
      expect(validateMixedStrategy([0.3, 0.7])).toBe(true);
      expect(validateMixedStrategy([0.25, 0.25, 0.25, 0.25])).toBe(true);
      expect(validateMixedStrategy([1.0])).toBe(true);
    });

    it('rejects strategies that do not sum to 1', () => {
      expect(validateMixedStrategy([0.3, 0.6])).toBe(false);
      expect(validateMixedStrategy([0.4, 0.7])).toBe(false);
      expect(validateMixedStrategy([0.5, 0.5, 0.1])).toBe(false);
    });

    it('rejects negative probabilities', () => {
      expect(validateMixedStrategy([-0.1, 1.1])).toBe(false);
      expect(validateMixedStrategy([0.5, -0.5, 1.0])).toBe(false);
    });

    it('rejects probabilities greater than 1', () => {
      expect(validateMixedStrategy([1.5, -0.5])).toBe(false);
      expect(validateMixedStrategy([0.3, 1.2, -0.5])).toBe(false);
    });

    it('accepts strategies within tolerance', () => {
      // Sum is 0.999, within default tolerance of 0.001
      expect(validateMixedStrategy([0.4, 0.5999])).toBe(true);

      // Sum is 1.002, outside default tolerance
      expect(validateMixedStrategy([0.4, 0.602])).toBe(false);
    });

    it('respects custom tolerance', () => {
      // Sum is 0.99, within tolerance of 0.02
      expect(validateMixedStrategy([0.4, 0.59], 0.02)).toBe(true);

      // Sum is 0.99, outside tolerance of 0.005
      expect(validateMixedStrategy([0.4, 0.59], 0.005)).toBe(false);
    });

    it('handles edge cases', () => {
      expect(validateMixedStrategy([0, 0, 1])).toBe(true);
      expect(validateMixedStrategy([0.3333, 0.3333, 0.3334])).toBe(true); // Close to 1/3 each
      expect(validateMixedStrategy([1.0])).toBe(true); // Pure strategy
    });
  });
});