// frontend/src/hooks/useApi.ts

import { useState, useCallback } from 'react';
import apiClient from '@/lib/api';
import { ApiState } from '@/types/api';

/**
 * A generic custom hook for making API requests.
 * Manages loading, error, and data states automatically.
 * @template T The expected data type of the API response.
 * @param {string} baseURL - The base URL for the endpoint (e.g., '/api/v1').
 * @returns The current API state and a request function.
 */
export function useApi<T = unknown>(baseURL: string = '/api/v1') {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const request = useCallback(
    async (endpoint: string, method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'GET', data?: unknown) => {
      setState((prev) => ({ ...prev, loading: true, error: null }));
      try {
        const response = await apiClient({
          url: `${baseURL}${endpoint}`,
          method,
          data,
        });
        setState({ data: response.data, loading: false, error: null });
        return response.data;
      } catch (err: unknown) {
        const errorMessage = err instanceof Error
          ? err.message
          : (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
          || 'An unexpected error occurred.';

        setState({ data: null, loading: false, error: errorMessage });
        throw new Error(errorMessage);
      }
    },
    [baseURL]
  );

  return { ...state, request };
}