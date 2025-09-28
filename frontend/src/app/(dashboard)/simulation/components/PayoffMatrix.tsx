// frontend/src/app/(dashboard)/simulation/components/PayoffMatrix.tsx

"use client";

import React, { useState } from 'react';
import { Info, Download, BarChart3, TrendingUp, Calculator, Trophy } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';

// Types for game theory data
interface PayoffData {
  title: string;
  players: [string, string];
  strategies: [string[], string[]];
  payoffs: number[][][]; // [player1Strategy][player2Strategy][player1Payoff, player2Payoff]
  equilibrium?: {
    strategies: [number, number];
    payoffs: [number, number];
    type: 'Nash' | 'Dominant' | 'Mixed' | 'Correlated';
    probability?: [number, number];
  };
  metadata?: {
    dominantStrategies?: {
      player1?: number;
      player2?: number;
    };
    paretoOptimal?: [number, number][];
    socialWelfare?: number[][];
  };
}

interface PayoffMatrixProps {
  data: PayoffData;
  highlightEquilibrium?: boolean;
  showAnalysis?: boolean;
  onCellClick?: (row: number, col: number, payoffs: [number, number]) => void;
  className?: string;
}

interface AnalysisResult {
  nashEquilibria: Array<{
    strategies: [number, number];
    payoffs: [number, number];
    type: string;
  }>;
  dominantStrategies: {
    player1?: { strategy: number; type: 'strict' | 'weak' };
    player2?: { strategy: number; type: 'strict' | 'weak' };
  };
  paretoFrontier: Array<{
    strategies: [number, number];
    payoffs: [number, number];
  }>;
  socialWelfare: {
    maximum: { strategies: [number, number]; welfare: number };
    current: number;
  };
}

export const PayoffMatrix: React.FC<PayoffMatrixProps> = ({
  data,
  highlightEquilibrium = true,
  showAnalysis = true,
  onCellClick,
  className = ""
}) => {
  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null);
  const [viewMode, setViewMode] = useState<'player1' | 'player2' | 'combined'>('combined');
  const [showHeatmap, setShowHeatmap] = useState(false);

  // Analyze the payoff matrix
  const analyzeMatrix = (): AnalysisResult => {
    const nashEquilibria: AnalysisResult['nashEquilibria'] = [];
    const paretoFrontier: AnalysisResult['paretoFrontier'] = [];

    // Find Nash Equilibria (simplified check for pure strategy equilibria)
    for (let i = 0; i < data.strategies[0].length; i++) {
      for (let j = 0; j < data.strategies[1].length; j++) {
        let isNash = true;

        // Check if player 1 would want to deviate
        for (let k = 0; k < data.strategies[0].length; k++) {
          if (k !== i && data.payoffs[k][j][0] > data.payoffs[i][j][0]) {
            isNash = false;
            break;
          }
        }

        // Check if player 2 would want to deviate
        if (isNash) {
          for (let k = 0; k < data.strategies[1].length; k++) {
            if (k !== j && data.payoffs[i][k][1] > data.payoffs[i][j][1]) {
              isNash = false;
              break;
            }
          }
        }

        if (isNash) {
          nashEquilibria.push({
            strategies: [i, j],
            payoffs: [data.payoffs[i][j][0], data.payoffs[i][j][1]],
            type: 'Pure Strategy Nash'
          });
        }
      }
    }

    // Find dominant strategies
    const dominantStrategies: AnalysisResult['dominantStrategies'] = {};

    // Check for player 1 dominant strategies
    for (let i = 0; i < data.strategies[0].length; i++) {
      let strictlyDominant = true;
      let weaklyDominant = true;

      for (let k = 0; k < data.strategies[0].length; k++) {
        if (k !== i) {
          for (let j = 0; j < data.strategies[1].length; j++) {
            if (data.payoffs[i][j][0] <= data.payoffs[k][j][0]) {
              strictlyDominant = false;
            }
            if (data.payoffs[i][j][0] < data.payoffs[k][j][0]) {
              weaklyDominant = false;
            }
          }
        }
      }

      if (strictlyDominant) {
        dominantStrategies.player1 = { strategy: i, type: 'strict' };
      } else if (weaklyDominant) {
        dominantStrategies.player1 = { strategy: i, type: 'weak' };
      }
    }

    // Check for player 2 dominant strategies
    for (let j = 0; j < data.strategies[1].length; j++) {
      let strictlyDominant = true;
      let weaklyDominant = true;

      for (let k = 0; k < data.strategies[1].length; k++) {
        if (k !== j) {
          for (let i = 0; i < data.strategies[0].length; i++) {
            if (data.payoffs[i][j][1] <= data.payoffs[i][k][1]) {
              strictlyDominant = false;
            }
            if (data.payoffs[i][j][1] < data.payoffs[i][k][1]) {
              weaklyDominant = false;
            }
          }
        }
      }

      if (strictlyDominant) {
        dominantStrategies.player2 = { strategy: j, type: 'strict' };
      } else if (weaklyDominant) {
        dominantStrategies.player2 = { strategy: j, type: 'weak' };
      }
    }

    // Find Pareto frontier
    const allOutcomes: Array<{ strategies: [number, number]; payoffs: [number, number] }> = [];
    for (let i = 0; i < data.strategies[0].length; i++) {
      for (let j = 0; j < data.strategies[1].length; j++) {
        allOutcomes.push({
          strategies: [i, j],
          payoffs: [data.payoffs[i][j][0], data.payoffs[i][j][1]]
        });
      }
    }

    allOutcomes.forEach(outcome => {
      let isPareto = true;
      allOutcomes.forEach(other => {
        if (other !== outcome) {
          if (other.payoffs[0] >= outcome.payoffs[0] &&
              other.payoffs[1] >= outcome.payoffs[1] &&
              (other.payoffs[0] > outcome.payoffs[0] || other.payoffs[1] > outcome.payoffs[1])) {
            isPareto = false;
          }
        }
      });
      if (isPareto) {
        paretoFrontier.push(outcome);
      }
    });

    // Calculate social welfare
    let maxWelfare = -Infinity;
    let maxWelfareStrategies: [number, number] = [0, 0];

    for (let i = 0; i < data.strategies[0].length; i++) {
      for (let j = 0; j < data.strategies[1].length; j++) {
        const welfare = data.payoffs[i][j][0] + data.payoffs[i][j][1];
        if (welfare > maxWelfare) {
          maxWelfare = welfare;
          maxWelfareStrategies = [i, j];
        }
      }
    }

    const currentWelfare = data.equilibrium ?
      data.equilibrium.payoffs[0] + data.equilibrium.payoffs[1] : 0;

    return {
      nashEquilibria,
      dominantStrategies,
      paretoFrontier,
      socialWelfare: {
        maximum: { strategies: maxWelfareStrategies, welfare: maxWelfare },
        current: currentWelfare
      }
    };
  };

  const analysis = analyzeMatrix();

  // Check if a cell is a Nash equilibrium
  const isNashEquilibrium = (row: number, col: number): boolean => {
    if (!highlightEquilibrium) return false;
    return analysis.nashEquilibria.some(eq => eq.strategies[0] === row && eq.strategies[1] === col);
  };

  // Check if a cell is Pareto optimal
  const isParetoOptimal = (row: number, col: number): boolean => {
    return analysis.paretoFrontier.some(pf => pf.strategies[0] === row && pf.strategies[1] === col);
  };

  // Get cell color for heatmap
  const getCellColor = (row: number, col: number): string => {
    if (!showHeatmap) return '';

    const payoff1 = data.payoffs[row][col][0];
    const payoff2 = data.payoffs[row][col][1];

    // Normalize payoffs to 0-1 range for coloring
    const allPayoffs1 = data.payoffs.flat().map(p => p[0]);
    const allPayoffs2 = data.payoffs.flat().map(p => p[1]);

    const min1 = Math.min(...allPayoffs1);
    const max1 = Math.max(...allPayoffs1);
    const min2 = Math.min(...allPayoffs2);
    const max2 = Math.max(...allPayoffs2);

    const norm1 = (payoff1 - min1) / (max1 - min1);
    const norm2 = (payoff2 - min2) / (max2 - min2);

    if (viewMode === 'player1') {
      const intensity = Math.round(norm1 * 255);
      return `rgba(59, 130, 246, ${norm1 * 0.7})`; // Blue gradient
    } else if (viewMode === 'player2') {
      const intensity = Math.round(norm2 * 255);
      return `rgba(239, 68, 68, ${norm2 * 0.7})`; // Red gradient
    } else {
      const combined = (norm1 + norm2) / 2;
      return `rgba(16, 185, 129, ${combined * 0.7})`; // Green gradient
    }
  };

  // Handle cell click
  const handleCellClick = (row: number, col: number) => {
    setSelectedCell([row, col]);
    onCellClick?.(row, col, [data.payoffs[row][col][0], data.payoffs[row][col][1]]);
  };

  // Export matrix data
  const handleExport = () => {
    const exportData = {
      ...data,
      analysis
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `payoff-matrix-${data.title.toLowerCase().replace(/\s+/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              {data.title}
            </CardTitle>
            <CardDescription>
              Strategic payoffs for {data.players[0]} vs {data.players[1]} interactions
            </CardDescription>
          </div>

          <div className="flex items-center gap-2">
            <Select value={viewMode} onValueChange={(value: any) => setViewMode(value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="combined">Combined</SelectItem>
                <SelectItem value="player1">{data.players[0]}</SelectItem>
                <SelectItem value="player2">{data.players[1]}</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" size="sm" onClick={handleExport}>
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <Tabs defaultValue="matrix" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="matrix">Payoff Matrix</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="matrix" className="space-y-4">
            {/* Matrix Table */}
            <div className="overflow-x-auto">
              <Table className="border">
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-32 text-center border-r-2 border-slate-300">
                      <div className="flex flex-col">
                        <span className="text-xs text-slate-500">{data.players[1]}</span>
                        <span className="font-bold">{data.players[0]}</span>
                      </div>
                    </TableHead>
                    {data.strategies[1].map((strategy, index) => (
                      <TableHead key={index} className="text-center min-w-32 border">
                        <div className="font-medium text-sm">{strategy}</div>
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.strategies[0].map((strategy, rowIndex) => (
                    <TableRow key={rowIndex}>
                      <TableCell className="font-medium text-center border-r-2 border-slate-300 bg-slate-50 dark:bg-slate-800">
                        {strategy}
                      </TableCell>
                      {data.strategies[1].map((_, colIndex) => {
                        const payoffs = data.payoffs[rowIndex][colIndex];
                        const isNash = isNashEquilibrium(rowIndex, colIndex);
                        const isPareto = isParetoOptimal(rowIndex, colIndex);
                        const isSelected = selectedCell?.[0] === rowIndex && selectedCell?.[1] === colIndex;

                        return (
                          <TableCell
                            key={colIndex}
                            className={`
                              text-center border cursor-pointer transition-all duration-200 p-3
                              ${isNash ? 'ring-2 ring-yellow-400 bg-yellow-50 dark:bg-yellow-900/20' : ''}
                              ${isSelected ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-900/20' : ''}
                              ${isPareto ? 'border-green-400 border-2' : ''}
                              hover:bg-slate-100 dark:hover:bg-slate-700
                            `}
                            style={{ backgroundColor: showHeatmap ? getCellColor(rowIndex, colIndex) : undefined }}
                            onClick={() => handleCellClick(rowIndex, colIndex)}
                          >
                            <div className="space-y-1">
                              {viewMode === 'combined' ? (
                                <>
                                  <div className="text-sm font-bold text-blue-600 dark:text-blue-400">
                                    {payoffs[0].toFixed(1)}
                                  </div>
                                  <div className="text-sm font-bold text-red-600 dark:text-red-400">
                                    {payoffs[1].toFixed(1)}
                                  </div>
                                </>
                              ) : viewMode === 'player1' ? (
                                <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                                  {payoffs[0].toFixed(1)}
                                </div>
                              ) : (
                                <div className="text-lg font-bold text-red-600 dark:text-red-400">
                                  {payoffs[1].toFixed(1)}
                                </div>
                              )}

                              {/* Indicators */}
                              <div className="flex justify-center gap-1">
                                {isNash && (
                                  <Badge variant="outline" className="text-xs px-1 py-0 bg-yellow-100 text-yellow-800">
                                    N
                                  </Badge>
                                )}
                                {isPareto && (
                                  <Badge variant="outline" className="text-xs px-1 py-0 bg-green-100 text-green-800">
                                    P
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {/* Legend */}
            <div className="flex flex-wrap gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-yellow-400 bg-yellow-50 rounded"></div>
                <span>Nash Equilibrium</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-green-400 rounded"></div>
                <span>Pareto Optimal</span>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs bg-yellow-100 text-yellow-800">N</Badge>
                <span>Nash</span>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs bg-green-100 text-green-800">P</Badge>
                <span>Pareto</span>
              </div>
              {viewMode === 'combined' && (
                <>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-blue-600 rounded"></div>
                    <span>{data.players[0]}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-600 rounded"></div>
                    <span>{data.players[1]}</span>
                  </div>
                </>
              )}
            </div>

            {/* Selected Cell Info */}
            {selectedCell && (
              <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <h4 className="font-medium mb-2">Selected Outcome</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-slate-600 dark:text-slate-400">Strategies:</span>
                    <div className="font-medium">
                      {data.strategies[0][selectedCell[0]]} vs {data.strategies[1][selectedCell[1]]}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-600 dark:text-slate-400">Payoffs:</span>
                    <div className="font-medium">
                      ({data.payoffs[selectedCell[0]][selectedCell[1]][0].toFixed(1)}, {data.payoffs[selectedCell[0]][selectedCell[1]][1].toFixed(1)})
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-600 dark:text-slate-400">Properties:</span>
                    <div className="flex gap-1">
                      {isNashEquilibrium(selectedCell[0], selectedCell[1]) && (
                        <Badge variant="outline" className="text-xs">Nash</Badge>
                      )}
                      {isParetoOptimal(selectedCell[0], selectedCell[1]) && (
                        <Badge variant="outline" className="text-xs">Pareto</Badge>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="analysis" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Nash Equilibria */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Calculator className="w-4 h-4" />
                    Nash Equilibria
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {analysis.nashEquilibria.length > 0 ? (
                    <div className="space-y-2">
                      {analysis.nashEquilibria.map((eq, index) => (
                        <div key={index} className="p-3 bg-slate-50 dark:bg-slate-800 rounded border">
                          <div className="font-medium text-sm">
                            {data.strategies[0][eq.strategies[0]]} vs {data.strategies[1][eq.strategies[1]]}
                          </div>
                          <div className="text-xs text-slate-600 dark:text-slate-400">
                            Payoffs: ({eq.payoffs[0].toFixed(1)}, {eq.payoffs[1].toFixed(1)})
                          </div>
                          <Badge variant="outline" className="text-xs mt-1">
                            {eq.type}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      No pure strategy Nash equilibria found.
                    </p>
                  )}
                </CardContent>
              </Card>

              {/* Dominant Strategies */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Trophy className="w-4 h-4" />
                    Dominant Strategies
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div>
                      <div className="font-medium text-sm">{data.players[0]}</div>
                      {analysis.dominantStrategies.player1 ? (
                        <div className="text-sm">
                          <span className="font-medium">
                            {data.strategies[0][analysis.dominantStrategies.player1.strategy]}
                          </span>
                          <Badge variant="outline" className="ml-2 text-xs">
                            {analysis.dominantStrategies.player1.type}
                          </Badge>
                        </div>
                      ) : (
                        <p className="text-xs text-slate-600 dark:text-slate-400">No dominant strategy</p>
                      )}
                    </div>

                    <Separator />

                    <div>
                      <div className="font-medium text-sm">{data.players[1]}</div>
                      {analysis.dominantStrategies.player2 ? (
                        <div className="text-sm">
                          <span className="font-medium">
                            {data.strategies[1][analysis.dominantStrategies.player2.strategy]}
                          </span>
                          <Badge variant="outline" className="ml-2 text-xs">
                            {analysis.dominantStrategies.player2.type}
                          </Badge>
                        </div>
                      ) : (
                        <p className="text-xs text-slate-600 dark:text-slate-400">No dominant strategy</p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Pareto Frontier */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" />
                    Pareto Optimal Outcomes
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {analysis.paretoFrontier.map((pf, index) => (
                      <div key={index} className="p-3 bg-slate-50 dark:bg-slate-800 rounded border">
                        <div className="font-medium text-sm">
                          {data.strategies[0][pf.strategies[0]]} vs {data.strategies[1][pf.strategies[1]]}
                        </div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                          Payoffs: ({pf.payoffs[0].toFixed(1)}, {pf.payoffs[1].toFixed(1)})
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Social Welfare */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Info className="w-4 h-4" />
                    Social Welfare Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <div className="font-medium text-sm">Maximum Welfare</div>
                    <div className="text-sm">
                      <span className="font-medium">
                        {data.strategies[0][analysis.socialWelfare.maximum.strategies[0]]} vs {data.strategies[1][analysis.socialWelfare.maximum.strategies[1]]}
                      </span>
                      <div className="text-xs text-slate-600 dark:text-slate-400">
                        Total: {analysis.socialWelfare.maximum.welfare.toFixed(1)}
                      </div>
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <div className="font-medium text-sm">Current Welfare</div>
                    <div className="text-sm font-mono">
                      {analysis.socialWelfare.current.toFixed(1)}
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      Efficiency: {((analysis.socialWelfare.current / analysis.socialWelfare.maximum.welfare) * 100).toFixed(1)}%
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="insights" className="space-y-4">
            <div className="grid grid-cols-1 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Strategic Insights</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Equilibrium Analysis */}
                  <div>
                    <h4 className="font-medium text-sm mb-2">Equilibrium Analysis</h4>
                    <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                      {analysis.nashEquilibria.length === 0 && (
                        <p>• No pure strategy Nash equilibria exist - players may need mixed strategies.</p>
                      )}
                      {analysis.nashEquilibria.length === 1 && (
                        <p>• Unique equilibrium provides a clear prediction of rational play.</p>
                      )}
                      {analysis.nashEquilibria.length > 1 && (
                        <p>• Multiple equilibria suggest coordination problems may arise.</p>
                      )}

                      {analysis.dominantStrategies.player1 && (
                        <p>• {data.players[0]} has a dominant strategy, simplifying decision-making.</p>
                      )}
                      {analysis.dominantStrategies.player2 && (
                        <p>• {data.players[1]} has a dominant strategy, simplifying decision-making.</p>
                      )}
                    </div>
                  </div>

                  <Separator />

                  {/* Efficiency Analysis */}
                  <div>
                    <h4 className="font-medium text-sm mb-2">Efficiency Considerations</h4>
                    <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                      {analysis.paretoFrontier.length > 1 && (
                        <p>• Multiple Pareto optimal outcomes exist - room for mutual improvement.</p>
                      )}

                      {(() => {
                        const efficiency = (analysis.socialWelfare.current / analysis.socialWelfare.maximum.welfare) * 100;
                        if (efficiency < 70) {
                          return <p>• Low efficiency suggests significant welfare losses from strategic behavior.</p>;
                        } else if (efficiency > 90) {
                          return <p>• High efficiency indicates that strategic outcomes are near-optimal.</p>;
                        } else {
                          return <p>• Moderate efficiency - some welfare loss from strategic considerations.</p>;
                        }
                      })()}
                    </div>
                  </div>

                  <Separator />

                  {/* Strategic Recommendations */}
                  <div>
                    <h4 className="font-medium text-sm mb-2">Strategic Recommendations</h4>
                    <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                      {analysis.dominantStrategies.player1 ? (
                        <p>• {data.players[0]} should always play {data.strategies[0][analysis.dominantStrategies.player1.strategy]}.</p>
                      ) : (
                        <p>• {data.players[0]} needs to consider {data.players[1]}'s likely strategy.</p>
                      )}

                      {analysis.dominantStrategies.player2 ? (
                        <p>• {data.players[1]} should always play {data.strategies[1][analysis.dominantStrategies.player2.strategy]}.</p>
                      ) : (
                        <p>• {data.players[1]} needs to consider {data.players[0]}'s likely strategy.</p>
                      )}

                      {analysis.nashEquilibria.length > 1 && (
                        <p>• Coordination mechanisms may be needed to select among equilibria.</p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};