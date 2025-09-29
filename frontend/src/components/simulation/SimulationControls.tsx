import React from 'react';

export interface SimulationControlsProps {
  onStart?: () => void;
  onStop?: () => void;
  onPause?: () => void;
  onReset?: () => void;
  isRunning?: boolean;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  onStart,
  onStop,
  onPause,
  onReset,
  isRunning
}) => {
  return (
    <div className="p-4 border rounded-lg">
      <h3 className="text-lg font-semibold mb-4">Simulation Controls</h3>
      <div className="flex gap-2">
        <button
          onClick={onStart}
          disabled={isRunning}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          Start
        </button>
        <button
          onClick={onPause}
          disabled={!isRunning}
          className="px-4 py-2 bg-yellow-500 text-white rounded disabled:opacity-50"
        >
          Pause
        </button>
        <button
          onClick={onStop}
          disabled={!isRunning}
          className="px-4 py-2 bg-red-500 text-white rounded disabled:opacity-50"
        >
          Stop
        </button>
        <button
          onClick={onReset}
          className="px-4 py-2 bg-gray-500 text-white rounded"
        >
          Reset
        </button>
      </div>
    </div>
  );
};