// frontend/src/hooks/useClassifier.ts

import { useCallback } from 'react';
import { useApi } from './useApi';
import { API_ROUTES } from '@/lib/constants';
import { ClassificationResponse, ClassificationRequest } from '@/types/classifier';

/**
 * Custom hook for all classifier-related API operations.
 * Provides a clean interface for text classification functionality.
 */
export function useClassifier() {
  const { data, loading, error, request } = useApi<ClassificationResponse>('');

  const classifyText = useCallback(
    async (text: string, options?: Partial<ClassificationRequest>) => {
      const payload: ClassificationRequest = {
        text,
        model_type: 'ensemble',
        explain: true,
        confidence_threshold: 0.5,
        ...options,
      };

      return request(API_ROUTES.CLASSIFIER_PREDICT, 'POST', payload);
    },
    [request]
  );

  const batchClassifyText = useCallback(
    async (texts: string[], options?: { model_type?: string; batch_size?: number }) => {
      const payload = {
        texts,
        model_type: 'ensemble',
        batch_size: 32,
        explain: false,
        ...options,
      };

      return request('/classifier/batch-predict', 'POST', payload);
    },
    [request]
  );

  const getModelMetrics = useCallback(
    async (modelType?: string) => {
      const endpoint = modelType
        ? `/classifier/metrics?model=${modelType}`
        : '/classifier/metrics';

      return request(endpoint, 'GET');
    },
    [request]
  );

  const getAvailableModels = useCallback(
    async () => {
      return request('/classifier/models', 'GET');
    },
    [request]
  );

  return {
    classificationResult: data,
    isLoading: loading,
    error,
    classifyText,
    batchClassifyText,
    getModelMetrics,
    getAvailableModels,
  };
}