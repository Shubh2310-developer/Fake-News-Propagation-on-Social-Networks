// frontend/src/components/game-theory/PayoffMatrix.tsx

import React from 'react';
import { PayoffMatrixData } from '@/types';
import { cn } from '@/lib/utils';

interface PayoffMatrixProps {
  data: PayoffMatrixData;
  title: string;
  highlightEquilibrium?: boolean;
  selectedCell?: { row: number; col: number };
}

export const PayoffMatrix: React.FC<PayoffMatrixProps> = ({
  data,
  title,
  highlightEquilibrium = false,
  selectedCell,
}) => {
  const { players, strategies, payoffs, equilibrium } = data;

  const isEquilibrium = (rowIndex: number, colIndex: number): boolean => {
    if (!highlightEquilibrium || !equilibrium) return false;
    return equilibrium.strategies[0] === rowIndex && equilibrium.strategies[1] === colIndex;
  };

  const isSelected = (rowIndex: number, colIndex: number): boolean => {
    if (!selectedCell) return false;
    return selectedCell.row === rowIndex && selectedCell.col === colIndex;
  };

  return (
    <div className="rounded-lg border bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-950">
      <h3 className="mb-4 text-lg font-semibold">{title}</h3>
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse border border-slate-300 dark:border-slate-700">
          <thead>
            <tr>
              <th className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium dark:border-slate-700 dark:bg-slate-800">
                {/* Empty cell for top-left corner */}
              </th>
              <th
                colSpan={strategies[players[1]].length}
                className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium dark:border-slate-700 dark:bg-slate-800"
              >
                {players[1]}
              </th>
            </tr>
            <tr>
              <th className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium dark:border-slate-700 dark:bg-slate-800">
                {players[0]}
              </th>
              {strategies[players[1]].map((colStrategy, colIndex) => (
                <th
                  key={colIndex}
                  className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium dark:border-slate-700 dark:bg-slate-800"
                >
                  {colStrategy}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {strategies[players[0]].map((rowStrategy, rowIndex) => (
              <tr key={rowIndex}>
                <th className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium dark:border-slate-700 dark:bg-slate-800">
                  {rowStrategy}
                </th>
                {strategies[players[1]].map((colStrategy, colIndex) => (
                  <td
                    key={colIndex}
                    className={cn(
                      'border border-slate-300 p-3 text-center dark:border-slate-700',
                      isEquilibrium(rowIndex, colIndex) && 'bg-yellow-200 dark:bg-yellow-800/50',
                      isSelected(rowIndex, colIndex) && 'bg-blue-100 dark:bg-blue-800/50'
                    )}
                  >
                    <div className="flex flex-col gap-1">
                      <span className="text-sm font-semibold text-blue-600 dark:text-blue-400">
                        {payoffs[rowIndex][colIndex][players[0]].toFixed(2)}
                      </span>
                      <span className="text-sm font-semibold text-red-600 dark:text-red-400">
                        {payoffs[rowIndex][colIndex][players[1]].toFixed(2)}
                      </span>
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-blue-600 dark:text-blue-400">Blue:</span>
          <span>{players[0]} Payoff</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-semibold text-red-600 dark:text-red-400">Red:</span>
          <span>{players[1]} Payoff</span>
        </div>
        {highlightEquilibrium && equilibrium && (
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded bg-yellow-200 dark:bg-yellow-800/50"></div>
            <span>Nash Equilibrium</span>
          </div>
        )}
        {selectedCell && (
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded bg-blue-100 dark:bg-blue-800/50"></div>
            <span>Selected Strategy</span>
          </div>
        )}
      </div>

      {/* Equilibrium Details */}
      {highlightEquilibrium && equilibrium && (
        <div className="mt-4 rounded-lg bg-slate-50 p-4 dark:bg-slate-800/50">
          <h4 className="mb-2 font-medium">Nash Equilibrium</h4>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            {players[0]}: {strategies[players[0]][equilibrium.strategies[0]]} | {' '}
            {players[1]}: {strategies[players[1]][equilibrium.strategies[1]]}
          </p>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Payoffs: ({equilibrium.payoffs[players[0]].toFixed(2)}, {equilibrium.payoffs[players[1]].toFixed(2)})
          </p>
        </div>
      )}
    </div>
  );
};