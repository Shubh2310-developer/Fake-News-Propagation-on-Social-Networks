// frontend/src/components/game-theory/PayoffMatrix.tsx

"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PayoffMatrixData } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface PayoffMatrixProps {
  data: PayoffMatrixData;
  title: string;
  highlightEquilibrium?: boolean;
  selectedCell?: { row: number; col: number };
  onCellClick?: (row: number, col: number) => void;
}

/**
 * Animated counter hook for smooth number transitions
 */
function useAnimatedValue(endValue: number, duration: number = 500) {
  const [value, setValue] = useState(endValue);
  const previousValue = useRef(endValue);

  useEffect(() => {
    const startValue = previousValue.current;
    const startTime = Date.now();

    const animate = () => {
      const now = Date.now();
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Easing function (ease-out)
      const eased = 1 - Math.pow(1 - progress, 3);
      const currentValue = startValue + (endValue - startValue) * eased;

      setValue(currentValue);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        previousValue.current = endValue;
      }
    };

    animate();
  }, [endValue, duration]);

  return value;
}

/**
 * Animated payoff cell component
 */
const AnimatedPayoff: React.FC<{
  value: number;
  color: string;
  isEquilibrium: boolean;
}> = ({ value, color, isEquilibrium }) => {
  const animatedValue = useAnimatedValue(value);

  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "text-sm font-semibold transition-all duration-300",
        color,
        isEquilibrium && "font-bold"
      )}
    >
      {animatedValue.toFixed(2)}
    </motion.span>
  );
};

/**
 * PayoffMatrix: Reusable component for displaying 2-player game theory payoff matrices
 *
 * Features:
 * - Clear tabular display with player names and strategies
 * - Color-coded payoffs for each player (blue/red)
 * - Nash equilibrium highlighting with animation
 * - Hover tooltips with contextual information
 * - Smooth animations for data updates
 * - Professional Card-based layout
 */
export const PayoffMatrix: React.FC<PayoffMatrixProps> = ({
  data,
  title,
  highlightEquilibrium = false,
  selectedCell,
  onCellClick,
}) => {
  const { players, strategies, payoffs, equilibrium } = data;
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  const isEquilibrium = (rowIndex: number, colIndex: number): boolean => {
    if (!highlightEquilibrium || !equilibrium) return false;
    return equilibrium.strategies[0] === rowIndex && equilibrium.strategies[1] === colIndex;
  };

  const isSelected = (rowIndex: number, colIndex: number): boolean => {
    if (!selectedCell) return false;
    return selectedCell.row === rowIndex && selectedCell.col === colIndex;
  };

  const handleCellHover = (
    rowIndex: number,
    colIndex: number,
    event: React.MouseEvent
  ) => {
    setHoveredCell({ row: rowIndex, col: colIndex });
    setTooltipPosition({ x: event.clientX, y: event.clientY });
  };

  const handleCellLeave = () => {
    setHoveredCell(null);
  };

  const handleCellClickInternal = (rowIndex: number, colIndex: number) => {
    onCellClick?.(rowIndex, colIndex);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Card className="shadow-md">
        <CardHeader>
          <CardTitle className="text-xl">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse border border-slate-300">
              <thead>
                <tr>
                  <th className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium text-gray-900">
                    {/* Empty cell for top-left corner */}
                  </th>
                  <th
                    colSpan={strategies[players[1]].length}
                    className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium text-gray-900"
                  >
                    {players[1]}
                  </th>
                </tr>
                <tr>
                  <th className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium text-gray-900">
                    {players[0]}
                  </th>
                  {strategies[players[1]].map((colStrategy, colIndex) => (
                    <th
                      key={colIndex}
                      className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium text-gray-900"
                    >
                      {colStrategy}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {strategies[players[0]].map((rowStrategy, rowIndex) => (
                  <tr key={rowIndex}>
                    <th className="border border-slate-300 bg-slate-50 p-3 text-sm font-medium text-gray-900">
                      {rowStrategy}
                    </th>
                    {strategies[players[1]].map((colStrategy, colIndex) => {
                      const isEq = isEquilibrium(rowIndex, colIndex);
                      const isSel = isSelected(rowIndex, colIndex);
                      const isHovered = hoveredCell?.row === rowIndex && hoveredCell?.col === colIndex;

                      return (
                        <motion.td
                          key={colIndex}
                          className={cn(
                            'border border-slate-300 p-3 text-center cursor-pointer',
                            'transition-all duration-300 relative bg-white',
                            isEq && 'bg-yellow-100 ring-2 ring-yellow-400 ring-inset',
                            isSel && 'bg-blue-100 ring-2 ring-blue-500 ring-inset',
                            isHovered && 'bg-slate-100 scale-105'
                          )}
                          animate={isEq ? {
                            boxShadow: [
                              '0 0 0px rgba(251, 191, 36, 0)',
                              '0 0 20px rgba(251, 191, 36, 0.6)',
                              '0 0 0px rgba(251, 191, 36, 0)',
                            ],
                          } : {}}
                          transition={isEq ? {
                            duration: 2,
                            repeat: Infinity,
                            repeatType: 'loop',
                          } : {}}
                          onMouseEnter={(e) => handleCellHover(rowIndex, colIndex, e)}
                          onMouseLeave={handleCellLeave}
                          onClick={() => handleCellClickInternal(rowIndex, colIndex)}
                        >
                          <div className="flex flex-col gap-1">
                            <AnimatedPayoff
                              value={payoffs[rowIndex][colIndex][players[0]]}
                              color="text-blue-600"
                              isEquilibrium={isEq}
                            />
                            <AnimatedPayoff
                              value={payoffs[rowIndex][colIndex][players[1]]}
                              color="text-red-600"
                              isEquilibrium={isEq}
                            />
                          </div>
                        </motion.td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Hover Tooltip */}
          <AnimatePresence>
            {hoveredCell && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.15 }}
                className="fixed z-50 pointer-events-none"
                style={{
                  left: tooltipPosition.x + 15,
                  top: tooltipPosition.y + 15,
                }}
              >
                <div className="px-4 py-3 bg-gray-900 dark:bg-gray-800 text-white rounded-lg shadow-xl border border-gray-700 max-w-sm">
                  <p className="text-sm font-medium mb-2">Outcome</p>
                  <p className="text-xs text-gray-300 leading-relaxed">
                    If <span className="font-semibold text-blue-400">{players[0]}</span> plays{' '}
                    <span className="font-semibold">&quot;{strategies[players[0]][hoveredCell.row]}&quot;</span> and{' '}
                    <span className="font-semibold text-red-400">{players[1]}</span> plays{' '}
                    <span className="font-semibold">&quot;{strategies[players[1]][hoveredCell.col]}&quot;</span>,{' '}
                    the <span className="text-blue-400">{players[0]}&apos;s</span> payoff is{' '}
                    <span className="font-bold">{payoffs[hoveredCell.row][hoveredCell.col][players[0]].toFixed(2)}</span>{' '}
                    and the <span className="text-red-400">{players[1]}&apos;s</span> payoff is{' '}
                    <span className="font-bold">{payoffs[hoveredCell.row][hoveredCell.col][players[1]].toFixed(2)}</span>.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Legend */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="mt-4 flex flex-wrap gap-4 text-sm"
          >
            <div className="flex items-center gap-2">
              <span className="font-semibold text-blue-600">Blue:</span>
              <span className="text-slate-700">{players[0]} Payoff</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-red-600">Red:</span>
              <span className="text-slate-700">{players[1]} Payoff</span>
            </div>
            {highlightEquilibrium && equilibrium && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 }}
                className="flex items-center gap-2"
              >
                <div className="h-3 w-3 rounded bg-yellow-200 ring-2 ring-yellow-400"></div>
                <span className="text-slate-700">Nash Equilibrium</span>
              </motion.div>
            )}
            {selectedCell && (
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded bg-blue-100 ring-2 ring-blue-500"></div>
                <span className="text-slate-700">Selected Strategy</span>
              </div>
            )}
          </motion.div>

          {/* Equilibrium Summary */}
          <AnimatePresence>
            {highlightEquilibrium && equilibrium && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.4, delay: 0.2 }}
                className="mt-4 overflow-hidden"
              >
                <div className="rounded-lg bg-gradient-to-r from-yellow-50 to-amber-50 p-4 border border-yellow-200">
                  <h4 className="mb-2 font-semibold text-yellow-900 flex items-center gap-2">
                    <span className="inline-block w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></span>
                    Nash Equilibrium
                  </h4>
                  <p className="text-sm text-slate-700 leading-relaxed">
                    At equilibrium, <span className="font-semibold text-blue-700">{players[0]}</span> chooses{' '}
                    <span className="font-bold">&quot;{strategies[players[0]][equilibrium.strategies[0]]}&quot;</span> and{' '}
                    <span className="font-semibold text-red-700">{players[1]}</span> chooses{' '}
                    <span className="font-bold">&quot;{strategies[players[1]][equilibrium.strategies[1]]}&quot;</span>.{' '}
                    This results in payoffs of{' '}
                    <span className="font-bold text-blue-700">
                      {equilibrium.payoffs[players[0]].toFixed(2)}
                    </span>{' '}
                    for {players[0]} and{' '}
                    <span className="font-bold text-red-700">
                      {equilibrium.payoffs[players[1]].toFixed(2)}
                    </span>{' '}
                    for {players[1]}.
                  </p>
                  {equilibrium.stability !== undefined && (
                    <div className="mt-3 pt-3 border-t border-yellow-200">
                      <p className="text-xs text-slate-600">
                        Stability: <span className="font-semibold">{(equilibrium.stability * 100).toFixed(1)}%</span>
                        {' • '}
                        Type: <span className="font-semibold capitalize">{equilibrium.type}</span>
                        {' • '}
                        Classification: <span className="font-semibold capitalize">{equilibrium.classification.replace(/_/g, ' ')}</span>
                      </p>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  );
};