/**
 * ML Models Configuration
 * Configuration for machine learning models and their parameters
 */

export const ML_CONFIG = {
  // Available models
  models: {
    logisticRegression: {
      id: 'logistic_regression',
      name: 'Logistic Regression',
      description: 'Fast linear classifier for binary classification',
      accuracy: 0.77,
      trainingTime: 'Fast',
      features: 2031,
      type: 'traditional' as const,
    },
    randomForest: {
      id: 'random_forest',
      name: 'Random Forest',
      description: 'Ensemble method with best overall performance',
      accuracy: 0.878,
      trainingTime: 'Medium',
      features: 2031,
      type: 'traditional' as const,
    },
    gradientBoosting: {
      id: 'gradient_boosting',
      name: 'Gradient Boosting',
      description: 'Powerful boosting algorithm',
      accuracy: 0.869,
      trainingTime: 'Slow',
      features: 2031,
      type: 'traditional' as const,
    },
    svm: {
      id: 'svm',
      name: 'Support Vector Machine',
      description: 'Kernel-based classifier',
      accuracy: 0.82,
      trainingTime: 'Medium',
      features: 2031,
      type: 'traditional' as const,
    },
    naiveBayes: {
      id: 'naive_bayes',
      name: 'Naive Bayes',
      description: 'Probabilistic classifier based on Bayes theorem',
      accuracy: 0.768,
      trainingTime: 'Fast',
      features: 2031,
      type: 'traditional' as const,
    },
    bert: {
      id: 'bert',
      name: 'BERT',
      description: 'Transformer-based deep learning model',
      accuracy: 0.85,
      trainingTime: 'Very Slow',
      maxSequenceLength: 512,
      model: 'bert-base-uncased',
      type: 'deep_learning' as const,
    },
    lstm: {
      id: 'lstm',
      name: 'LSTM',
      description: 'Recurrent neural network for sequential data',
      accuracy: 0.83,
      trainingTime: 'Slow',
      embeddingDim: 100,
      hiddenDim: 128,
      type: 'deep_learning' as const,
    },
    dnn: {
      id: 'dnn',
      name: 'Deep Neural Network',
      description: 'Multi-layer feedforward neural network',
      accuracy: 0.808,
      trainingTime: 'Medium',
      layers: [512, 256, 128],
      type: 'deep_learning' as const,
    },
    ensemble: {
      id: 'ensemble',
      name: 'Ensemble Model',
      description: 'Combines multiple models for robust predictions',
      accuracy: 0.874,
      trainingTime: 'Slow',
      models: ['random_forest', 'gradient_boosting', 'bert'],
      type: 'ensemble' as const,
    },
  },

  // Feature extraction settings
  features: {
    tfidf: {
      maxFeatures: 2000,
      ngramRange: [1, 2] as [number, number],
      minDf: 2,
      maxDf: 0.95,
    },
    linguistic: {
      enabled: true,
      features: [
        'word_count',
        'char_count',
        'avg_word_length',
        'stopword_ratio',
        'punctuation_ratio',
        'uppercase_ratio',
        'sentiment_score',
        'readability_score',
      ],
    },
    metadata: {
      enabled: true,
      features: ['source', 'timestamp', 'author'],
    },
  },

  // Training configuration
  training: {
    defaultBatchSize: 32,
    defaultEpochs: 10,
    validationSplit: 0.2,
    testSplit: 0.1,
    randomState: 42,
    crossValidationFolds: 5,
    earlyStoppingPatience: 3,
    learningRate: 0.001,
  },

  // Prediction settings
  prediction: {
    confidenceThreshold: 0.5,
    batchSize: 100,
    maxTextLength: 10000,
    minTextLength: 10,
  },

  // Evaluation metrics
  metrics: {
    primary: ['accuracy', 'precision', 'recall', 'f1_score'],
    advanced: ['auc_roc', 'confusion_matrix', 'classification_report'],
    threshold: {
      acceptable: 0.75,
      good: 0.80,
      excellent: 0.85,
    },
  },

  // Model paths (backend)
  paths: {
    models: './ml_models/saved/',
    cache: './ml_models/cache/',
    training: './data/training/',
    results: './results/models/',
  },
} as const;

// Helper types
export type ModelId = keyof typeof ML_CONFIG.models;
export type ModelType = 'traditional' | 'deep_learning' | 'ensemble';

// Helper function to get model by ID
export function getModelConfig(modelId: ModelId) {
  return ML_CONFIG.models[modelId];
}

// Helper function to get all models of a specific type
export function getModelsByType(type: ModelType) {
  return Object.values(ML_CONFIG.models).filter((model) => model.type === type);
}

// Helper function to get recommended model
export function getRecommendedModel() {
  return ML_CONFIG.models.randomForest; // Best accuracy
}

// Helper function to get fastest model
export function getFastestModel() {
  return ML_CONFIG.models.logisticRegression;
}

// Color scheme for models (for UI visualization)
export const MODEL_COLORS = {
  logistic_regression: '#3b82f6',
  random_forest: '#10b981',
  gradient_boosting: '#8b5cf6',
  svm: '#f59e0b',
  naive_bayes: '#ef4444',
  bert: '#06b6d4',
  lstm: '#ec4899',
  dnn: '#6366f1',
  ensemble: '#14b8a6',
} as const;
