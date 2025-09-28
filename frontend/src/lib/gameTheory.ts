// Game theory utility functions for the fake news game theory platform

/**
 * Formats a mixed strategy equilibrium into a human-readable string.
 * @param equilibrium - The raw equilibrium data from the API.
 * @param strategies - The strategy definitions.
 * @returns A formatted string describing the mixed strategy.
 */
export function formatMixedStrategyEquilibrium(
  equilibrium: [number[], number[]],
  strategies: Record<string, string[]>
): string {
  const [p1Probs, p2Probs] = equilibrium;
  const player1Strategies = strategies.player1 || strategies.spreader || [];
  const player2Strategies = strategies.player2 || strategies.fact_checker || [];

  const p1Strategy = p1Probs
    .map((prob, i) => `${(prob * 100).toFixed(1)}% ${player1Strategies[i] || `Strategy ${i + 1}`}`)
    .join(', ');

  const p2Strategy = p2Probs
    .map((prob, i) => `${(prob * 100).toFixed(1)}% ${player2Strategies[i] || `Strategy ${i + 1}`}`)
    .join(', ');

  return `Player 1 plays (${p1Strategy}) and Player 2 plays (${p2Strategy}).`;
}

/**
 * Finds the best response strategy for a player given opponent's strategy.
 * @param payoffMatrix - The payoff matrix for the current player.
 * @param opponentStrategy - The opponent's strategy index.
 * @returns The index of the best response strategy.
 */
export function findBestResponse(
  payoffMatrix: number[][],
  opponentStrategy: number
): number {
  let bestResponse = 0;
  let maxPayoff = payoffMatrix[0][opponentStrategy];

  for (let i = 1; i < payoffMatrix.length; i++) {
    if (payoffMatrix[i][opponentStrategy] > maxPayoff) {
      maxPayoff = payoffMatrix[i][opponentStrategy];
      bestResponse = i;
    }
  }

  return bestResponse;
}

/**
 * Determines if a strategy strictly dominates another strategy.
 * @param strategy1 - First strategy's payoffs.
 * @param strategy2 - Second strategy's payoffs.
 * @returns True if strategy1 strictly dominates strategy2.
 */
export function strictlyDominates(strategy1: number[], strategy2: number[]): boolean {
  if (strategy1.length !== strategy2.length) return false;

  let hasStrictAdvantage = false;
  for (let i = 0; i < strategy1.length; i++) {
    if (strategy1[i] < strategy2[i]) {
      return false;
    }
    if (strategy1[i] > strategy2[i]) {
      hasStrictAdvantage = true;
    }
  }

  return hasStrictAdvantage;
}

/**
 * Determines if a strategy weakly dominates another strategy.
 * @param strategy1 - First strategy's payoffs.
 * @param strategy2 - Second strategy's payoffs.
 * @returns True if strategy1 weakly dominates strategy2.
 */
export function weaklyDominates(strategy1: number[], strategy2: number[]): boolean {
  if (strategy1.length !== strategy2.length) return false;

  for (let i = 0; i < strategy1.length; i++) {
    if (strategy1[i] < strategy2[i]) {
      return false;
    }
  }

  return true;
}

/**
 * Calculates the expected payoff for a mixed strategy.
 * @param mixedStrategy - Array of probabilities for each pure strategy.
 * @param payoffMatrix - Payoff matrix for the player.
 * @param opponentStrategy - Opponent's strategy (pure or mixed).
 * @returns The expected payoff.
 */
export function calculateExpectedPayoff(
  mixedStrategy: number[],
  payoffMatrix: number[][],
  opponentStrategy: number[] | number
): number {
  let expectedPayoff = 0;

  if (typeof opponentStrategy === 'number') {
    // Opponent plays pure strategy
    for (let i = 0; i < mixedStrategy.length; i++) {
      expectedPayoff += mixedStrategy[i] * payoffMatrix[i][opponentStrategy];
    }
  } else {
    // Opponent plays mixed strategy
    for (let i = 0; i < mixedStrategy.length; i++) {
      for (let j = 0; j < opponentStrategy.length; j++) {
        expectedPayoff += mixedStrategy[i] * opponentStrategy[j] * payoffMatrix[i][j];
      }
    }
  }

  return expectedPayoff;
}

/**
 * Checks if a strategy profile is a Nash equilibrium.
 * @param strategyProfile - Array of strategy indices for each player.
 * @param payoffMatrices - Payoff matrices for each player.
 * @returns True if the strategy profile is a Nash equilibrium.
 */
export function isNashEquilibrium(
  strategyProfile: number[],
  payoffMatrices: number[][][]
): boolean {
  for (let player = 0; player < strategyProfile.length; player++) {
    const currentStrategy = strategyProfile[player];
    const playerPayoffs = payoffMatrices[player];

    // For two-player games, we only need the opponent's strategy

    // Calculate current payoff
    let currentPayoff: number;
    if (strategyProfile.length === 2) {
      const opponentStrategy = strategyProfile[1 - player];
      currentPayoff = playerPayoffs[currentStrategy][opponentStrategy];
    } else {
      // For multi-player games, this would need more complex indexing
      currentPayoff = 0; // Simplified for now
    }

    // Check if player can improve by unilateral deviation
    for (let altStrategy = 0; altStrategy < playerPayoffs.length; altStrategy++) {
      if (altStrategy === currentStrategy) continue;

      let altPayoff: number;
      if (strategyProfile.length === 2) {
        const opponentStrategy = strategyProfile[1 - player];
        altPayoff = playerPayoffs[altStrategy][opponentStrategy];
      } else {
        altPayoff = 0; // Simplified for now
      }

      if (altPayoff > currentPayoff) {
        return false; // Player can improve, not Nash equilibrium
      }
    }
  }

  return true;
}

/**
 * Formats payoff values for display in the UI.
 * @param payoff - The payoff value to format.
 * @param precision - Number of decimal places.
 * @returns Formatted string.
 */
export function formatPayoff(payoff: number, precision: number = 2): string {
  if (Math.abs(payoff) < 0.01) {
    return '0';
  }
  return payoff.toFixed(precision);
}

/**
 * Calculates social welfare for a given strategy profile.
 * @param strategyProfile - Array of strategy indices.
 * @param payoffMatrices - Payoff matrices for each player.
 * @returns Total social welfare.
 */
export function calculateSocialWelfare(
  strategyProfile: number[],
  payoffMatrices: number[][][]
): number {
  let totalWelfare = 0;

  for (let player = 0; player < strategyProfile.length; player++) {
    const strategy = strategyProfile[player];
    const playerPayoffs = payoffMatrices[player];

    if (strategyProfile.length === 2) {
      const opponentStrategy = strategyProfile[1 - player];
      totalWelfare += playerPayoffs[strategy][opponentStrategy];
    }
    // Multi-player case would need more complex calculation
  }

  return totalWelfare;
}

/**
 * Determines the type of game based on payoff structure.
 * @param payoffMatrices - Payoff matrices for all players.
 * @returns Game classification.
 */
export function classifyGame(payoffMatrices: number[][][]): string {
  if (payoffMatrices.length !== 2) {
    return 'multi-player';
  }

  const [p1Matrix, p2Matrix] = payoffMatrices;
  let isZeroSum = true;
  let isSymmetric = true;

  // Check if zero-sum
  for (let i = 0; i < p1Matrix.length; i++) {
    for (let j = 0; j < p1Matrix[i].length; j++) {
      if (Math.abs(p1Matrix[i][j] + p2Matrix[i][j]) > 0.001) {
        isZeroSum = false;
        break;
      }
    }
    if (!isZeroSum) break;
  }

  // Check if symmetric
  if (p1Matrix.length === p2Matrix.length && p1Matrix[0].length === p2Matrix[0].length) {
    for (let i = 0; i < p1Matrix.length; i++) {
      for (let j = 0; j < p1Matrix[i].length; j++) {
        if (Math.abs(p1Matrix[i][j] - p2Matrix[j][i]) > 0.001) {
          isSymmetric = false;
          break;
        }
      }
      if (!isSymmetric) break;
    }
  } else {
    isSymmetric = false;
  }

  if (isZeroSum) {
    return isSymmetric ? 'symmetric zero-sum' : 'zero-sum';
  }

  return isSymmetric ? 'symmetric' : 'asymmetric';
}

/**
 * Converts strategy names to indices for API calls.
 * @param strategyNames - Array of strategy names.
 * @param availableStrategies - Map of strategy names to indices.
 * @returns Array of strategy indices.
 */
export function strategyNamesToIndices(
  strategyNames: string[],
  availableStrategies: Record<string, number>
): number[] {
  return strategyNames.map(name => availableStrategies[name] ?? 0);
}

/**
 * Converts strategy indices to names for display.
 * @param strategyIndices - Array of strategy indices.
 * @param availableStrategies - Array of strategy names.
 * @returns Array of strategy names.
 */
export function strategyIndicesToNames(
  strategyIndices: number[],
  availableStrategies: string[]
): string[] {
  return strategyIndices.map(index => availableStrategies[index] ?? 'Unknown');
}

/**
 * Validates that probabilities in a mixed strategy sum to 1.
 * @param probabilities - Array of probabilities.
 * @param tolerance - Tolerance for floating point comparison.
 * @returns True if probabilities are valid.
 */
export function validateMixedStrategy(probabilities: number[], tolerance: number = 0.001): boolean {
  const sum = probabilities.reduce((total, prob) => total + prob, 0);
  return Math.abs(sum - 1) <= tolerance && probabilities.every(prob => prob >= 0 && prob <= 1);
}