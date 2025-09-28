// frontend/src/hooks/useSimulation.ts

import { useCallback } from 'react';
import { useApi } from './useApi';
import { API_ROUTES } from '@/lib/constants';
import { GameParameters, SimulationResults } from '@/types/simulation';

/**
 * Custom hook for managing the simulation lifecycle via API calls.
 * Provides functions to run simulations, check status, and fetch results.
 */
export function useSimulation() {
  const { request: runRequest, ...runState } = useApi<{ simulation_id: string }>();
  const { request: statusRequest, ...statusState } = useApi<{ status: string; progress: number }>();
  const { request: resultsRequest, ...resultsState } = useApi<SimulationResults>();

  const runSimulation = useCallback(
    async (params: GameParameters) => {
      return runRequest(API_ROUTES.SIMULATION_RUN, 'POST', params);
    },
    [runRequest]
  );

  const checkStatus = useCallback(
    async (simulationId: string) => {
      return statusRequest(API_ROUTES.SIMULATION_STATUS(simulationId));
    },
    [statusRequest]
  );

  const fetchResults = useCallback(
    async (simulationId: string) => {
      return resultsRequest(API_ROUTES.SIMULATION_RESULTS(simulationId));
    },
    [resultsRequest]
  );

  const cancelSimulation = useCallback(
    async (simulationId: string) => {
      return runRequest(`/simulation/cancel/${simulationId}`, 'POST');
    },
    [runRequest]
  );

  const getSimulationHistory = useCallback(
    async (limit = 10, offset = 0) => {
      return runRequest(`/simulation/history?limit=${limit}&offset=${offset}`, 'GET');
    },
    [runRequest]
  );

  const downloadSimulationData = useCallback(
    async (simulationId: string, format: 'json' | 'csv' = 'json') => {
      return runRequest(`/simulation/download/${simulationId}?format=${format}`, 'GET');
    },
    [runRequest]
  );

  return {
    runSimulation,
    checkStatus,
    fetchResults,
    cancelSimulation,
    getSimulationHistory,
    downloadSimulationData,
    // Expose states for each operation
    runState,
    statusState,
    resultsState,
  };
}