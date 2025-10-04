/**
 * API Configuration
 * Central configuration for API endpoints and settings
 */

export const API_CONFIG = {
  // Base URLs
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  apiVersion: process.env.NEXT_PUBLIC_API_VERSION || 'v1',
  wsURL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',

  // Timeouts (in milliseconds)
  timeout: {
    default: 10000,
    upload: 120000,
    training: 300000,
    simulation: 180000,
  },

  // Rate limiting
  rateLimits: {
    predictions: 100, // per minute
    uploads: 10, // per hour
    simulations: 5, // concurrent
  },

  // File upload limits
  upload: {
    maxSize: 104857600, // 100MB in bytes
    allowedFormats: ['csv', 'json'],
    maxBatchSize: 100,
  },

  // Retry configuration
  retry: {
    maxAttempts: 3,
    backoffMultiplier: 2,
    initialDelay: 1000,
  },

  // Feature flags
  features: {
    websockets: false, // Not implemented yet
    realTimeUpdates: false,
    advancedVisualization: true,
    exportResults: true,
  },
} as const;

export const API_ENDPOINTS = {
  // Classifier endpoints
  classifier: {
    predict: '/classifier/predict',
    predictBatch: '/classifier/predict/batch',
    train: '/classifier/train',
    metrics: '/classifier/metrics',
    info: '/classifier/info',
    reload: (modelType: string) => `/classifier/reload/${modelType}`,
    available: '/classifier/models/available',
  },

  // Simulation endpoints
  simulation: {
    run: '/simulation/run',
    status: (id: string) => `/simulation/status/${id}`,
    results: (id: string) => `/simulation/results/${id}`,
    list: '/simulation/list',
    cancel: (id: string) => `/simulation/cancel/${id}`,
    delete: (id: string) => `/simulation/delete/${id}`,
    statistics: '/simulation/statistics',
  },

  // Equilibrium endpoints
  equilibrium: {
    calculate: '/equilibrium/calculate',
    analyzeSensitivity: '/equilibrium/analyze/sensitivity',
    compareScenarios: '/equilibrium/compare/scenarios',
    strategySpace: '/equilibrium/strategy-space',
    payoffWeights: '/equilibrium/payoff-weights',
    quickAnalysis: '/equilibrium/quick-analysis',
  },

  // Network analysis endpoints
  analysis: {
    generateNetwork: '/analysis/network/generate',
    networkMetrics: '/analysis/network/metrics',
    networkVisualize: '/analysis/network/visualize',
    simulatePropagation: '/analysis/propagation/simulate',
    propagationVisualize: '/analysis/propagation/visualize',
    communityVisualize: '/analysis/network/community/visualize',
    compareNetworks: '/analysis/network/compare',
    listNetworks: '/analysis/network/list',
    deleteNetwork: (id: string) => `/analysis/network/${id}`,
    networkStats: '/analysis/network/statistics',
  },

  // Data management endpoints
  data: {
    uploadDataset: '/data/upload/dataset',
    generateSynthetic: '/data/generate/synthetic',
    getDataset: (id: string) => `/data/dataset/${id}`,
    getStatistics: (id: string) => `/data/dataset/${id}/statistics`,
    processTraining: '/data/process/training',
    crossValidation: '/data/cross-validation/splits',
    listDatasets: '/data/datasets/list',
    deleteDataset: (id: string) => `/data/dataset/${id}`,
    exportResults: (id: string, format: string) => `/data/export/results/${id}?format=${format}`,
    exportDataset: (id: string, format: string) => `/data/export/dataset/${id}?format=${format}`,
  },

  // System endpoints
  system: {
    health: '/health',
  },
} as const;

// Helper function to build full API URL
export function buildApiUrl(endpoint: string): string {
  const base = API_CONFIG.baseURL;
  const version = API_CONFIG.apiVersion;
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${base}/api/${version}${cleanEndpoint}`;
}

// Helper function to get timeout for specific operation
export function getTimeout(operation: keyof typeof API_CONFIG.timeout = 'default'): number {
  return API_CONFIG.timeout[operation];
}
