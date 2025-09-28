// frontend/src/types/index.ts

/**
 * Barrel file for all type definitions in the fake news game theory application.
 * This file acts as a single entry point for importing any type definition,
 * making imports cleaner and more maintainable across the application.
 *
 * Usage:
 * import { SimulationState, NetworkData, ClassificationResponse } from '@/types';
 */

// Re-export all types from the other files in this directory
export * from './api';
export * from './classifier';
export * from './gameTheory';
export * from './network';
export * from './simulation';

// Additional utility types that might be used across the application
export type ID = string | number;

export type Status = 'idle' | 'loading' | 'success' | 'error';

export type Theme = 'light' | 'dark' | 'system';

export type Language = 'en' | 'es' | 'fr';

/**
 * Generic response wrapper for API calls
 */
export interface ApiResponse<T = unknown> {
  data: T;
  message?: string;
  status: 'success' | 'error';
  timestamp: string;
}

/**
 * Generic pagination parameters
 */
export interface PaginationParams {
  page: number;
  limit: number;
  offset?: number;
}

/**
 * Generic pagination response wrapper
 */
export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

/**
 * Generic filter parameters
 */
export interface FilterParams {
  search?: string;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  dateFrom?: string;
  dateTo?: string;
  [key: string]: unknown;
}

/**
 * User preferences for the application
 */
export interface UserPreferences {
  theme: Theme;
  language: Language;
  notifications: {
    email: boolean;
    push: boolean;
    simulation: boolean;
    updates: boolean;
  };
  dashboard: {
    autoRefresh: boolean;
    refreshInterval: number;
    defaultView: 'grid' | 'list';
  };
  accessibility: {
    reducedMotion: boolean;
    highContrast: boolean;
    fontSize: 'small' | 'medium' | 'large';
  };
}

/**
 * Application configuration
 */
export interface AppConfig {
  apiUrl: string;
  wsUrl: string;
  environment: 'development' | 'staging' | 'production';
  version: string;
  features: {
    analytics: boolean;
    debugging: boolean;
    performanceMonitoring: boolean;
    errorReporting: boolean;
  };
  limits: {
    maxSimulationSize: number;
    maxSimulationRounds: number;
    maxChartDataPoints: number;
  };
}

/**
 * Error types for consistent error handling
 */
export interface AppError {
  code: string;
  message: string;
  details?: unknown;
  timestamp: string;
  userMessage?: string;
}

/**
 * Metrics and analytics data
 */
export interface MetricsData {
  eventName: string;
  properties: Record<string, unknown>;
  timestamp: string;
  userId?: string;
  sessionId?: string;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  loadTime: number;
  renderTime: number;
  interactionTime: number;
  memoryUsage: number;
  networkLatency: number;
}