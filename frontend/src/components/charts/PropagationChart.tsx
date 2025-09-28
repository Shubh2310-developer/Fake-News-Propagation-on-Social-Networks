// frontend/src/components/charts/PropagationChart.tsx

"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { NetworkVisualization } from './NetworkVisualization';
import { SimulationResults, NetworkData, NodeData } from '@/types';
import { Button } from '@/components/ui/button';
import { Play, Pause, RefreshCw } from 'lucide-react';

interface PropagationChartProps {
  baseNetwork: NetworkData;
  simulationResults: SimulationResults;
  width?: number;
  height?: number;
  onNodeClick?: (node: NodeData) => void;
}

const getNodeColor = (state: string) => {
  switch (state) {
    case 'infected':
      return '#ef4444'; // Red
    case 'recovered':
      return '#22c55e'; // Green
    case 'susceptible':
    default:
      return '#94a3b8'; // Gray
  }
};

export function PropagationChart({
  baseNetwork,
  simulationResults,
  width = 800,
  height = 600,
  onNodeClick
}: PropagationChartProps) {
  const [timeStep, setTimeStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(500); // milliseconds per step

  const maxTimeSteps = simulationResults.propagationSteps?.length || 0;

  // Animate the simulation over time
  useEffect(() => {
    if (!isPlaying || maxTimeSteps === 0) return;

    const interval = setInterval(() => {
      setTimeStep((prev) => {
        const next = prev + 1;
        if (next >= maxTimeSteps) {
          setIsPlaying(false);
          return maxTimeSteps - 1;
        }
        return next;
      });
    }, playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, maxTimeSteps]);

  // Memoize the network data for the current timestep to avoid re-renders
  const currentNetworkData = useMemo(() => {
    if (!simulationResults.propagationSteps || timeStep >= simulationResults.propagationSteps.length) {
      return baseNetwork;
    }

    const currentStep = simulationResults.propagationSteps[timeStep];
    const nodes = baseNetwork.nodes.map(node => {
      let state = 'susceptible';

      if (currentStep.infected?.includes(node.id)) {
        state = 'infected';
      } else if (currentStep.recovered?.includes(node.id)) {
        state = 'recovered';
      }

      return {
        ...node,
        state,
        color: getNodeColor(state),
      };
    });

    return { ...baseNetwork, nodes };
  }, [timeStep, baseNetwork, simulationResults]);

  const handlePlay = () => {
    if (timeStep >= maxTimeSteps - 1) {
      setTimeStep(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setTimeStep(0);
  };

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(event.target.value);
    setTimeStep(value);
    setIsPlaying(false);
  };

  const handleSpeedChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setPlaybackSpeed(parseInt(event.target.value));
  };

  if (maxTimeSteps === 0) {
    return (
      <div className="flex items-center justify-center h-96 text-slate-500">
        No simulation data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <NetworkVisualization
        data={currentNetworkData}
        width={width}
        height={height}
        onNodeClick={onNodeClick}
      />

      {/* Controls */}
      <div className="flex items-center gap-4 p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
        <div className="flex items-center gap-2">
          <Button
            onClick={handlePlay}
            size="icon"
            variant="outline"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>

          <Button
            onClick={handleReset}
            size="icon"
            variant="outline"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>

        {/* Timeline Slider */}
        <div className="flex-1 flex items-center gap-2">
          <span className="text-sm text-slate-600 dark:text-slate-400">
            Step: {timeStep + 1} / {maxTimeSteps}
          </span>
          <input
            type="range"
            min={0}
            max={maxTimeSteps - 1}
            value={timeStep}
            onChange={handleSliderChange}
            className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer dark:bg-slate-700"
          />
        </div>

        {/* Speed Control */}
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600 dark:text-slate-400">
            Speed:
          </label>
          <select
            value={playbackSpeed}
            onChange={handleSpeedChange}
            className="px-2 py-1 text-sm border border-slate-200 rounded dark:border-slate-700 dark:bg-slate-800"
          >
            <option value={1000}>Slow</option>
            <option value={500}>Normal</option>
            <option value={250}>Fast</option>
            <option value={100}>Very Fast</option>
          </select>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-slate-400"></div>
          <span>Susceptible</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <span>Infected</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span>Recovered</span>
        </div>
      </div>
    </div>
  );
}