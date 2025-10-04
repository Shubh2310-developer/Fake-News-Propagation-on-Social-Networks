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
  currentSimulationId: string | null;
  pollIntervalId: NodeJS.Timeout | null;

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
  currentSimulationId: null,
  pollIntervalId: null,

  // Actions
  setGameParameters: (params) =>
    set((state) => ({
      gameParameters: { ...state.gameParameters, ...params },
    })),

  startSimulation: async () => {
    set({
      isRunning: true,
      simulationState: 'starting',
      error: null,
      results: null
    });

    try {
      const params = get().gameParameters;

      // Transform frontend parameters to backend format
      const backendParams = {
        network_config: {
          num_nodes: params.network?.size || 1000,
          network_type: params.network?.type === 'small_world' ? 'watts_strogatz' : 'barabasi_albert',
          attachment_preference: params.network?.parameters?.averageDegree || 5,
          rewiring_probability: params.network?.parameters?.rewireProbability || 0.1,
          random_seed: params.advanced?.randomSeed
        },
        game_config: {
          num_rounds: params.dynamics?.timeHorizon || 50,
          num_spreaders: params.numPlayers?.spreaders || 10,
          num_fact_checkers: params.numPlayers?.factCheckers || 5,
          num_platforms: params.numPlayers?.platforms || 1,
          learning_rate: params.dynamics?.learningRate || 0.1,
          exploration_rate: 0.1,
          random_seed: params.advanced?.randomSeed
        },
        description: params.metadata?.description || '',
        tags: params.metadata?.tags || [],
        save_detailed_history: params.advanced?.output?.saveDetailedHistory !== false,
        save_network: params.advanced?.output?.saveNetworkStates !== false
      };

      // Start simulation
      const response = await apiClient.post(API_ROUTES.SIMULATION_RUN, backendParams);
      const simulationId = response.data.simulation_id;

      set({ simulationState: 'running', currentSimulationId: simulationId });

      // Poll for results
      const pollInterval: NodeJS.Timeout = setInterval(async () => {
        try {
          const statusResponse = await apiClient.get(API_ROUTES.SIMULATION_STATUS(simulationId));
          const status = statusResponse.data.status;

          if (status === 'completed') {
            clearInterval(pollInterval);

            // Fetch results
            const resultsResponse = await apiClient.get(API_ROUTES.SIMULATION_RESULTS(simulationId));

            set({
              isRunning: false,
              simulationState: 'completed',
              results: resultsResponse.data,
              pollIntervalId: null,
            });
          } else if (status === 'failed') {
            clearInterval(pollInterval);

            set({
              isRunning: false,
              simulationState: 'error',
              error: statusResponse.data.error || 'Simulation failed',
              pollIntervalId: null,
            });
          }
        } catch (pollError) {
          clearInterval(pollInterval);
          const errorMessage = pollError instanceof Error
            ? pollError.message
            : 'Failed to fetch simulation status';

          set({
            isRunning: false,
            simulationState: 'error',
            error: errorMessage,
            pollIntervalId: null,
          });
        }
      }, 2000); // Poll every 2 seconds

      set({ pollIntervalId: pollInterval });

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
    const { pollIntervalId } = get();

    // Clear polling interval if active
    if (pollIntervalId) {
      clearInterval(pollIntervalId);
    }

    set({
      isRunning: false,
      simulationState: 'idle',
      error: null,
      pollIntervalId: null,
    });
  },

  resetSimulation: () => {
    const { pollIntervalId } = get();

    // Clear polling interval if active
    if (pollIntervalId) {
      clearInterval(pollIntervalId);
    }

    set({
      gameParameters: DEFAULT_SIMULATION_PARAMS,
      results: null,
      error: null,
      simulationState: 'idle',
      isRunning: false,
      currentSimulationId: null,
      pollIntervalId: null,
    });
  },
}));