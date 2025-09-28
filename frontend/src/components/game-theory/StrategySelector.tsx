// frontend/src/components/game-theory/StrategySelector.tsx

import React, { useState, useEffect } from 'react';
import { PayoffMatrixData } from '@/types';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface StrategySelectorProps {
  players: PayoffMatrixData['players'];
  strategies: PayoffMatrixData['strategies'];
  payoffs: PayoffMatrixData['payoffs'];
  onSelectionChange: (selection: { player1Index: number; player2Index: number }) => void;
  initialSelection?: { player1Index: number; player2Index: number };
  variant?: 'tabs' | 'select';
}

export const StrategySelector: React.FC<StrategySelectorProps> = ({
  players,
  strategies,
  payoffs,
  onSelectionChange,
  initialSelection = { player1Index: 0, player2Index: 0 },
  variant = 'tabs',
}) => {
  const [p1Index, setP1Index] = useState(initialSelection.player1Index);
  const [p2Index, setP2Index] = useState(initialSelection.player2Index);

  useEffect(() => {
    setP1Index(initialSelection.player1Index);
    setP2Index(initialSelection.player2Index);
  }, [initialSelection]);

  const handleP1Change = (newIndex: number) => {
    setP1Index(newIndex);
    onSelectionChange({ player1Index: newIndex, player2Index: p2Index });
  };

  const handleP2Change = (newIndex: number) => {
    setP2Index(newIndex);
    onSelectionChange({ player1Index: p1Index, player2Index: newIndex });
  };

  // Get current payoffs for selected strategies
  const currentPayoffs = payoffs[p1Index] && payoffs[p1Index][p2Index]
    ? payoffs[p1Index][p2Index]
    : null;

  const renderPlayerStrategy = (playerIndex: 0 | 1, selectedIndex: number, onChange: (index: number) => void) => {
    const player = players[playerIndex];
    const playerStrategies = strategies[player];
    const color = playerIndex === 0 ? 'blue' : 'red';

    if (variant === 'select') {
      return (
        <div className="space-y-2">
          <label className={`text-sm font-medium text-${color}-600 dark:text-${color}-400`}>
            {player}'s Strategy
          </label>
          <Select
            value={selectedIndex.toString()}
            onValueChange={(value) => onChange(parseInt(value))}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {playerStrategies.map((strategy, index) => (
                <SelectItem key={index} value={index.toString()}>
                  {strategy}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      );
    }

    // Tabs variant
    return (
      <div className="space-y-2">
        <h4 className={`text-sm font-medium text-${color}-600 dark:text-${color}-400`}>
          {player}'s Strategy
        </h4>
        <Tabs
          value={selectedIndex.toString()}
          onValueChange={(value) => onChange(parseInt(value))}
        >
          <TabsList className="grid w-full grid-cols-2">
            {playerStrategies.map((strategy, index) => (
              <TabsTrigger key={index} value={index.toString()}>
                {strategy}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Strategy Selection</CardTitle>
        <CardDescription>
          Choose strategies for each player to see the resulting payoffs
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Strategy Selectors */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {renderPlayerStrategy(0, p1Index, handleP1Change)}
          {renderPlayerStrategy(1, p2Index, handleP2Change)}
        </div>

        {/* Current Selection Display */}
        <div className="rounded-lg bg-slate-50 p-4 dark:bg-slate-800/50">
          <h4 className="mb-3 font-medium">Current Selection</h4>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-blue-600 dark:text-blue-400">
                {players[0]}:
              </span>
              <span>{strategies[players[0]][p1Index]}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-red-600 dark:text-red-400">
                {players[1]}:
              </span>
              <span>{strategies[players[1]][p2Index]}</span>
            </div>
          </div>
        </div>

        {/* Payoff Display */}
        {currentPayoffs && (
          <div className="rounded-lg border-2 border-dashed border-slate-300 p-4 dark:border-slate-600">
            <h4 className="mb-3 text-center font-medium">Resulting Payoffs</h4>
            <div className="flex items-center justify-center gap-8">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {currentPayoffs[players[0]].toFixed(2)}
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                  {players[0]}
                </div>
              </div>
              <div className="text-xl text-slate-400">vs</div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {currentPayoffs[players[1]].toFixed(2)}
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                  {players[1]}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Strategy Descriptions */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="rounded-lg bg-blue-50 p-3 dark:bg-blue-900/20">
            <h5 className="mb-2 text-sm font-medium text-blue-800 dark:text-blue-300">
              {players[0]}'s Available Strategies
            </h5>
            <ul className="space-y-1 text-sm">
              {strategies[players[0]].map((strategy, index) => (
                <li
                  key={index}
                  className={`${
                    index === p1Index
                      ? 'font-semibold text-blue-900 dark:text-blue-200'
                      : 'text-blue-700 dark:text-blue-400'
                  }`}
                >
                  {index === p1Index ? '’ ' : '   '}
                  {strategy}
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg bg-red-50 p-3 dark:bg-red-900/20">
            <h5 className="mb-2 text-sm font-medium text-red-800 dark:text-red-300">
              {players[1]}'s Available Strategies
            </h5>
            <ul className="space-y-1 text-sm">
              {strategies[players[1]].map((strategy, index) => (
                <li
                  key={index}
                  className={`${
                    index === p2Index
                      ? 'font-semibold text-red-900 dark:text-red-200'
                      : 'text-red-700 dark:text-red-400'
                  }`}
                >
                  {index === p2Index ? '’ ' : '   '}
                  {strategy}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};