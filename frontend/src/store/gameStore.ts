// frontend/src/store/gameStore.ts

import { create } from 'zustand';
import { PayoffMatrixData } from '@/types/gameTheory';
import apiClient from '@/lib/api';
import { API_ROUTES } from '@/lib/constants';

interface GameStore {
  // State
  isLoading: boolean;
  payoffMatrix: PayoffMatrixData | null;
  error: string | null;

  // Actions
  calculateEquilibrium: (params: Record<string, unknown>) => Promise<void>;
  clearMatrix: () => void;
}

export const useGameStore = create<GameStore>((set) => ({
  // Initial state
  isLoading: false,
  payoffMatrix: null,
  error: null,

  // Actions
  calculateEquilibrium: async (params) => {
    set({ isLoading: true, error: null });

    try {
      const response = await apiClient.post<PayoffMatrixData>(
        API_ROUTES.EQUILIBRIUM_CALCULATE,
        params
      );

      set({
        payoffMatrix: response.data,
        isLoading: false
      });
    } catch (err: unknown) {
      const errorMessage = err instanceof Error
        ? err.message
        : (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        || 'Failed to calculate equilibrium.';

      set({
        error: errorMessage,
        isLoading: false,
        payoffMatrix: null,
      });
    }
  },

  clearMatrix: () => set({
    payoffMatrix: null,
    error: null
  }),
}));