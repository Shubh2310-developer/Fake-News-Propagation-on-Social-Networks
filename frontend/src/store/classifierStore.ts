// frontend/src/store/classifierStore.ts

import { create } from 'zustand';
import { ClassificationResponse } from '@/types/classifier';
import apiClient from '@/lib/api';
import { API_ROUTES } from '@/lib/constants';

interface ClassifierStore {
  // State
  inputText: string;
  isLoading: boolean;
  result: ClassificationResponse | null;
  error: string | null;

  // Actions
  setInputText: (text: string) => void;
  classifyText: () => Promise<void>;
  clearResult: () => void;
}

export const useClassifierStore = create<ClassifierStore>((set, get) => ({
  // Initial state
  inputText: 'Scientists have discovered a new species of glowing frog in the Amazon rainforest.',
  isLoading: false,
  result: null,
  error: null,

  // Actions
  setInputText: (text) => set({ inputText: text }),

  classifyText: async () => {
    const text = get().inputText;
    if (!text.trim()) {
      set({ error: 'Please enter some text to classify.' });
      return;
    }

    set({ isLoading: true, error: null, result: null });

    try {
      const response = await apiClient.post<ClassificationResponse>(
        API_ROUTES.CLASSIFIER_PREDICT,
        {
          text,
          model_type: 'ensemble', // Use ensemble model by default
          explain: true, // Request explanations
          confidence_threshold: 0.5
        }
      );

      set({
        result: response.data,
        isLoading: false
      });
    } catch (err: unknown) {
      const errorMessage = err instanceof Error
        ? err.message
        : (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        || 'Failed to get classification.';

      set({
        error: errorMessage,
        isLoading: false,
      });
    }
  },

  clearResult: () => set({
    result: null,
    error: null
  }),
}));