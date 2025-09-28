// frontend/src/store/simulationStore.ts

import { create } from 'zustand';
import { GameParameters, SimulationResults, SimulationState } from '@/types/simulation';
import { DEFAULT_SIMULATION_PARAMS, API_ROUTES } from '@/lib/constants';
import apiClient from '@/lib/api';

interface SimulationStore {
  // State
  gameParameters: GameParameters;
  simulationState: SimulationState;
  isRunning: boolean;
  results: SimulationResults | null;
  error: string | null;

  // Actions
  setGameParameters: (params: Partial<GameParameters>) => void;
  startSimulation: () => Promise<void>;
  stopSimulation: () => void;
  resetSimulation: () => void;
}

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  // Initial state
  gameParameters: DEFAULT_SIMULATION_PARAMS,
  simulationState: 'idle',
  isRunning: false,
  results: null,
  error: null,

  // Actions
  setGameParameters: (params) =>
    set((state) => ({
      gameParameters: { ...state.gameParameters, ...params },
    })),

  startSimulation: async () => {
    set({
      isRunning: true,
      simulationState: 'running',
      error: null,
      results: null
    });

    try {
      const response = await apiClient.post(API_ROUTES.SIMULATION_RUN, get().gameParameters);

      // In a real app, you would use the simulation_id from the response
      // to poll for the results. For this example, we'll assume results come back immediately.
      set({
        isRunning: false,
        simulationState: 'completed',
        results: response.data.results || response.data, // Flexible response handling
      });
    } catch (err: unknown) {
      const errorMessage = err instanceof Error
        ? err.message
        : (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        || 'An unexpected error occurred.';

      set({
        isRunning: false,
        simulationState: 'error',
        error: errorMessage,
      });
    }
  },

  stopSimulation: () => {
    set({
      isRunning: false,
      simulationState: 'idle',
      error: null,
    });
  },

  resetSimulation: () =>
    set({
      gameParameters: DEFAULT_SIMULATION_PARAMS,
      results: null,
      error: null,
      simulationState: 'idle',
      isRunning: false,
    }),
}));