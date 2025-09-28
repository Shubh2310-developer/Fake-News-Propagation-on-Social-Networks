// frontend/src/hooks/useDataUpload.ts

import { useState, useCallback } from 'react';
import apiClient from '@/lib/api';

interface UploadState {
  progress: number;
  isUploading: boolean;
  error: string | null;
  data: unknown | null;
  uploadId: string | null;
}

interface UploadOptions {
  onProgress?: (progress: number) => void;
  onSuccess?: (data: unknown) => void;
  onError?: (error: string) => void;
  headers?: Record<string, string>;
  fieldName?: string;
}

/**
 * Custom hook to handle file uploads with progress tracking.
 * @param {string} endpoint - The API endpoint to upload the file to.
 */
export function useDataUpload(endpoint: string) {
  const [state, setState] = useState<UploadState>({
    progress: 0,
    isUploading: false,
    error: null,
    data: null,
    uploadId: null,
  });

  const uploadFile = useCallback(
    async (file: File, options: UploadOptions = {}) => {
      const {
        onProgress,
        onSuccess,
        onError,
        headers = {},
        fieldName = 'file'
      } = options;

      // Validate file
      if (!file) {
        const error = 'No file provided';
        setState(prev => ({ ...prev, error }));
        onError?.(error);
        return;
      }

      const formData = new FormData();
      formData.append(fieldName, file);

      // Generate upload ID for tracking
      const uploadId = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      setState({
        progress: 0,
        isUploading: true,
        error: null,
        data: null,
        uploadId,
      });

      try {
        const response = await apiClient.post(endpoint, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            ...headers,
          },
          onUploadProgress: (progressEvent) => {
            if (progressEvent.total) {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );

              setState(prev => ({ ...prev, progress: percentCompleted }));
              onProgress?.(percentCompleted);
            }
          },
        });

        setState(prev => ({
          ...prev,
          isUploading: false,
          data: response.data,
        }));

        onSuccess?.(response.data);
        return response.data;
      } catch (err: unknown) {
        const errorMessage = err instanceof Error
          ? err.message
          : (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
          || 'File upload failed.';

        setState(prev => ({
          ...prev,
          isUploading: false,
          error: errorMessage,
        }));

        onError?.(errorMessage);
        throw new Error(errorMessage);
      }
    },
    [endpoint]
  );

  const uploadMultipleFiles = useCallback(
    async (files: File[], options: UploadOptions = {}) => {
      const results: unknown[] = [];
      const errors: string[] = [];

      for (let i = 0; i < files.length; i++) {
        try {
          const result = await uploadFile(files[i], {
            ...options,
            onProgress: (progress) => {
              const totalProgress = ((i * 100) + progress) / files.length;
              options.onProgress?.(Math.round(totalProgress));
            },
          });
          results.push(result);
        } catch (error) {
          errors.push(error instanceof Error ? error.message : 'Upload failed');
        }
      }

      return { results, errors };
    },
    [uploadFile]
  );

  const cancelUpload = useCallback(() => {
    // Note: This is a simplified implementation
    // In a real app, you'd need to cancel the actual request
    setState(prev => ({
      ...prev,
      isUploading: false,
      error: 'Upload cancelled',
    }));
  }, []);

  const resetUpload = useCallback(() => {
    setState({
      progress: 0,
      isUploading: false,
      error: null,
      data: null,
      uploadId: null,
    });
  }, []);

  return {
    ...state,
    uploadFile,
    uploadMultipleFiles,
    cancelUpload,
    resetUpload,
  };
}