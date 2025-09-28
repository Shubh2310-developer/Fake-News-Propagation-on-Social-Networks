// frontend/src/components/game-theory/EquilibriumVisualizer.tsx

import React from 'react';
import { BarChart } from '@/components/charts/BarChart';
import { Equilibrium, PayoffMatrixData } from '@/types';

interface EquilibriumVisualizerProps {
  equilibrium: Equilibrium;
  strategies: PayoffMatrixData['strategies'];
  players: PayoffMatrixData['players'];
  title?: string;
}

export const EquilibriumVisualizer: React.FC<EquilibriumVisualizerProps> = ({
  equilibrium,
  strategies,
  players,
  title = 'Mixed Strategy Equilibrium',
}) => {
  // Check if equilibrium has probability data (mixed strategy)
  const hasProbabilities = equilibrium.probabilities && equilibrium.probabilities.length === 2;

  if (!hasProbabilities) {
    // Pure strategy equilibrium - show simple text display
    return (
      <div className="rounded-lg border bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950">
        <h3 className="mb-4 text-lg font-semibold">{title}</h3>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="rounded-lg bg-slate-50 p-4 dark:bg-slate-800/50">
            <h4 className="mb-2 font-medium text-blue-600 dark:text-blue-400">
              {players[0]}'s Strategy
            </h4>
            <p className="text-lg font-semibold">
              {strategies[players[0]][equilibrium.strategies[0]]}
            </p>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Pure Strategy (100% probability)
            </p>
          </div>
          <div className="rounded-lg bg-slate-50 p-4 dark:bg-slate-800/50">
            <h4 className="mb-2 font-medium text-red-600 dark:text-red-400">
              {players[1]}'s Strategy
            </h4>
            <p className="text-lg font-semibold">
              {strategies[players[1]][equilibrium.strategies[1]]}
            </p>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Pure Strategy (100% probability)
            </p>
          </div>
        </div>
        <div className="mt-4 rounded-lg bg-slate-50 p-4 dark:bg-slate-800/50">
          <h4 className="mb-2 font-medium">Expected Payoffs</h4>
          <div className="flex gap-4">
            <span className="text-blue-600 dark:text-blue-400">
              {players[0]}: {equilibrium.payoffs[players[0]].toFixed(2)}
            </span>
            <span className="text-red-600 dark:text-red-400">
              {players[1]}: {equilibrium.payoffs[players[1]].toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Mixed strategy equilibrium - show probability charts
  const player1ChartData = strategies[players[0]].map((strategyName, index) => ({
    name: strategyName,
    probability: (equilibrium.probabilities[0][index] * 100).toFixed(1),
  }));

  const player2ChartData = strategies[players[1]].map((strategyName, index) => ({
    name: strategyName,
    probability: (equilibrium.probabilities[1][index] * 100).toFixed(1),
  }));

  const player1Series = [
    { dataKey: 'probability', name: 'Probability (%)', color: '#3b82f6' }
  ];

  const player2Series = [
    { dataKey: 'probability', name: 'Probability (%)', color: '#ef4444' }
  ];

  return (
    <div className="rounded-lg border bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950">
      <h3 className="mb-6 text-lg font-semibold">{title}</h3>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
        <div>
          <h4 className="mb-4 text-center font-semibold text-blue-600 dark:text-blue-400">
            {players[0]}'s Mixed Strategy
          </h4>
          <BarChart
            data={player1ChartData}
            xAxisKey="name"
            series={player1Series}
          />
          <div className="mt-2 text-center text-sm text-slate-600 dark:text-slate-400">
            Expected Payoff: {equilibrium.payoffs[players[0]].toFixed(2)}
          </div>
        </div>

        <div>
          <h4 className="mb-4 text-center font-semibold text-red-600 dark:text-red-400">
            {players[1]}'s Mixed Strategy
          </h4>
          <BarChart
            data={player2ChartData}
            xAxisKey="name"
            series={player2Series}
          />
          <div className="mt-2 text-center text-sm text-slate-600 dark:text-slate-400">
            Expected Payoff: {equilibrium.payoffs[players[1]].toFixed(2)}
          </div>
        </div>
      </div>

      {/* Strategy Details */}
      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-lg bg-blue-50 p-4 dark:bg-blue-900/20">
          <h5 className="mb-2 font-medium text-blue-800 dark:text-blue-300">
            {players[0]} Strategy Probabilities
          </h5>
          {strategies[players[0]].map((strategy, index) => (
            <div key={index} className="flex justify-between text-sm">
              <span>{strategy}:</span>
              <span className="font-medium">
                {(equilibrium.probabilities[0][index] * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>

        <div className="rounded-lg bg-red-50 p-4 dark:bg-red-900/20">
          <h5 className="mb-2 font-medium text-red-800 dark:text-red-300">
            {players[1]} Strategy Probabilities
          </h5>
          {strategies[players[1]].map((strategy, index) => (
            <div key={index} className="flex justify-between text-sm">
              <span>{strategy}:</span>
              <span className="font-medium">
                {(equilibrium.probabilities[1][index] * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};