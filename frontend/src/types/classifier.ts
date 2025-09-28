// frontend/src/types/classifier.ts

/**
 * TypeScript definitions for machine learning classifier interactions.
 * These types mirror the Pydantic models used in the FastAPI backend
 * and ensure type safety for all classifier-related API calls.
 */

/**
 * Available classifier model types
 */
export type ClassifierModelType = 'ensemble' | 'bert' | 'lstm' | 'logistic_regression' | 'naive_bayes' | 'svm' | 'random_forest';

/**
 * Classification labels
 */
export type ClassificationLabel = 'fake' | 'real';

/**
 * Text preprocessing options
 */
export interface PreprocessingOptions {
  /** Whether to convert text to lowercase */
  lowercase: boolean;
  /** Whether to remove URLs */
  removeUrls: boolean;
  /** Whether to remove mentions (@username) */
  removeMentions: boolean;
  /** Whether to remove hashtags */
  removeHashtags: boolean;
  /** Whether to remove punctuation */
  removePunctuation: boolean;
  /** Whether to remove extra whitespace */
  removeWhitespace: boolean;
  /** Whether to expand contractions */
  expandContractions: boolean;
  /** Whether to remove stop words */
  removeStopWords: boolean;
  /** Language for stop words */
  language: string;
}

/**
 * Classification request payload
 */
export interface ClassificationRequest {
  /** Text content to be classified */
  text: string;
  /** Model type to use for classification */
  model_type?: ClassifierModelType;
  /** Preprocessing options */
  preprocessing?: Partial<PreprocessingOptions>;
  /** Whether to return explanation/interpretation */
  explain?: boolean;
  /** Confidence threshold for predictions */
  confidence_threshold?: number;
}

/**
 * Batch classification request
 */
export interface BatchClassificationRequest {
  /** Array of text content to be classified */
  texts: string[];
  /** Model type to use for classification */
  model_type?: ClassifierModelType;
  /** Preprocessing options */
  preprocessing?: Partial<PreprocessingOptions>;
  /** Whether to return explanations */
  explain?: boolean;
  /** Maximum batch size for processing */
  batch_size?: number;
}

/**
 * Model explanation/interpretation data
 */
export interface ModelExplanation {
  /** Feature importance scores */
  feature_importance: Array<{
    feature: string;
    importance: number;
    type: 'positive' | 'negative';
  }>;
  /** SHAP values if available */
  shap_values?: number[];
  /** LIME explanation if available */
  lime_explanation?: {
    words: string[];
    scores: number[];
  };
  /** Attention weights for transformer models */
  attention_weights?: Array<{
    token: string;
    weight: number;
    position: number;
  }>;
  /** Top contributing phrases */
  top_phrases: Array<{
    phrase: string;
    contribution: number;
    type: 'positive' | 'negative';
  }>;
}

/**
 * Classification response from the API
 */
export interface ClassificationResponse {
  /** Original text that was classified */
  text: string;
  /** Predicted label */
  prediction: ClassificationLabel;
  /** Confidence score (0-1) */
  confidence: number;
  /** Probability distribution over all classes */
  probabilities: {
    real: number;
    fake: number;
  };
  /** Model used for prediction */
  model_used: string;
  /** Processing time in milliseconds */
  processing_time: number;
  /** Model explanation if requested */
  explanation?: ModelExplanation;
  /** Additional metadata */
  metadata?: {
    preprocessed_text?: string;
    text_length: number;
    language?: string;
    sentiment?: 'positive' | 'negative' | 'neutral';
    toxicity_score?: number;
  };
}

/**
 * Batch classification response
 */
export interface BatchClassificationResponse {
  /** Array of classification results */
  results: Array<ClassificationResponse & {
    /** Index in the original batch */
    batch_index: number;
  }>;
  /** Total number of texts processed */
  total_processed: number;
  /** Number of successful classifications */
  successful: number;
  /** Number of failed classifications */
  failed: number;
  /** Model used for predictions */
  model_used: string;
  /** Total processing time */
  total_processing_time: number;
  /** Average processing time per text */
  average_processing_time: number;
  /** Any errors that occurred */
  errors?: Array<{
    batch_index: number;
    error: string;
  }>;
}

/**
 * Model training configuration
 */
export interface ModelTrainingConfig {
  /** Model type to train */
  model_type: ClassifierModelType;
  /** Training parameters */
  parameters: {
    /** Learning rate */
    learning_rate: number;
    /** Batch size */
    batch_size: number;
    /** Number of epochs */
    epochs: number;
    /** Validation split ratio */
    validation_split: number;
    /** Early stopping patience */
    early_stopping_patience?: number;
    /** Regularization strength */
    regularization?: number;
    /** Dropout rate for neural networks */
    dropout_rate?: number;
  };
  /** Cross-validation configuration */
  cross_validation?: {
    folds: number;
    shuffle: boolean;
    random_state: number;
  };
  /** Feature selection options */
  feature_selection?: {
    method: 'chi2' | 'mutual_info' | 'f_test' | 'lasso';
    max_features: number;
  };
}

/**
 * Training data structure
 */
export interface TrainingData {
  /** Training texts */
  X_train: string[];
  /** Training labels */
  y_train: ClassificationLabel[];
  /** Validation texts */
  X_val?: string[];
  /** Validation labels */
  y_val?: ClassificationLabel[];
  /** Test texts */
  X_test?: string[];
  /** Test labels */
  y_test?: ClassificationLabel[];
}

/**
 * Model training request
 */
export interface ModelTrainingRequest {
  /** Training configuration */
  config: ModelTrainingConfig;
  /** Training data */
  data: TrainingData;
  /** Model name/identifier */
  model_name?: string;
  /** Description of the training run */
  description?: string;
  /** Tags for organization */
  tags?: string[];
}

/**
 * Model training response
 */
export interface ModelTrainingResponse {
  /** Training job ID */
  training_id: string;
  /** Current status */
  status: 'started' | 'running' | 'completed' | 'failed';
  /** Progress message */
  message: string;
  /** Model type being trained */
  model_type: ClassifierModelType;
  /** Estimated completion time */
  estimated_completion?: string;
}

/**
 * Model training status
 */
export interface ModelTrainingStatus {
  /** Training job ID */
  training_id: string;
  /** Current status */
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  /** Progress percentage (0-100) */
  progress: number;
  /** Current epoch/iteration */
  current_epoch?: number;
  /** Total epochs */
  total_epochs?: number;
  /** Current metrics */
  current_metrics?: {
    accuracy: number;
    loss: number;
    val_accuracy?: number;
    val_loss?: number;
  };
  /** Training logs */
  logs?: string[];
  /** Error message if failed */
  error_message?: string;
  /** Start time */
  start_time: string;
  /** End time if completed */
  end_time?: string;
}

/**
 * Model performance metrics
 */
export interface ModelMetrics {
  /** Model identifier */
  model_id: string;
  /** Model type */
  model_type: ClassifierModelType;
  /** Whether metrics are available */
  metrics_available: boolean;
  /** Classification metrics */
  classification_metrics?: {
    /** Overall accuracy */
    accuracy: number;
    /** Precision scores */
    precision: {
      macro: number;
      micro: number;
      fake: number;
      real: number;
    };
    /** Recall scores */
    recall: {
      macro: number;
      micro: number;
      fake: number;
      real: number;
    };
    /** F1 scores */
    f1_score: {
      macro: number;
      micro: number;
      fake: number;
      real: number;
    };
    /** Area under ROC curve */
    auc_roc: number;
    /** Area under precision-recall curve */
    auc_pr: number;
    /** Matthews correlation coefficient */
    mcc: number;
  };
  /** Confusion matrix */
  confusion_matrix?: {
    true_positive: number;
    true_negative: number;
    false_positive: number;
    false_negative: number;
  };
  /** Cross-validation results */
  cross_validation?: {
    mean_accuracy: number;
    std_accuracy: number;
    scores: number[];
  };
  /** Training history */
  training_history?: {
    epochs: number[];
    train_accuracy: number[];
    val_accuracy: number[];
    train_loss: number[];
    val_loss: number[];
  };
}

/**
 * Model information and metadata
 */
export interface ModelInfo {
  /** Model identifier */
  model_id: string;
  /** Model name */
  name: string;
  /** Model type */
  type: ClassifierModelType;
  /** Model description */
  description?: string;
  /** Model version */
  version: string;
  /** Whether model is trained and ready */
  is_trained: boolean;
  /** Training date */
  trained_at?: string;
  /** Model size in bytes */
  size_bytes: number;
  /** Model parameters count */
  parameter_count?: number;
  /** Training dataset info */
  training_data_info?: {
    samples_count: number;
    features_count: number;
    class_distribution: Record<ClassificationLabel, number>;
  };
  /** Model performance summary */
  performance_summary?: {
    accuracy: number;
    f1_score: number;
    training_time: number;
  };
  /** Model tags */
  tags: string[];
  /** Model status */
  status: 'training' | 'ready' | 'deprecated' | 'error';
}

/**
 * Available models response
 */
export interface AvailableModelsResponse {
  /** List of available models */
  models: ModelInfo[];
  /** Default model */
  default_model: string;
  /** Total count */
  total_count: number;
}

/**
 * Model comparison data
 */
export interface ModelComparison {
  /** Models being compared */
  models: ModelInfo[];
  /** Comparison metrics */
  comparison: {
    /** Performance comparison */
    performance: Record<string, {
      accuracy: number;
      f1_score: number;
      precision: number;
      recall: number;
      auc_roc: number;
    }>;
    /** Speed comparison */
    speed: Record<string, {
      training_time: number;
      inference_time: number;
      throughput: number;
    }>;
    /** Resource usage */
    resources: Record<string, {
      memory_usage: number;
      model_size: number;
      cpu_usage: number;
    }>;
  };
  /** Recommendations */
  recommendations: {
    best_accuracy: string;
    best_speed: string;
    best_balance: string;
    use_cases: Record<string, string[]>;
  };
}

/**
 * Text analysis features
 */
export interface TextFeatures {
  /** Basic text statistics */
  basic_stats: {
    character_count: number;
    word_count: number;
    sentence_count: number;
    paragraph_count: number;
    avg_word_length: number;
    avg_sentence_length: number;
  };
  /** Linguistic features */
  linguistic: {
    language: string;
    sentiment_score: number;
    subjectivity_score: number;
    readability_score: number;
    lexical_diversity: number;
  };
  /** Structural features */
  structural: {
    has_urls: boolean;
    has_mentions: boolean;
    has_hashtags: boolean;
    has_numbers: boolean;
    capitalization_ratio: number;
    punctuation_ratio: number;
  };
  /** Content features */
  content: {
    topic_keywords: string[];
    named_entities: Array<{
      text: string;
      label: string;
      confidence: number;
    }>;
    key_phrases: string[];
    emotional_indicators: string[];
  };
}

/**
 * Classifier configuration
 */
export interface ClassifierConfig {
  /** Default model to use */
  default_model: ClassifierModelType;
  /** Confidence threshold for predictions */
  confidence_threshold: number;
  /** Whether to enable model explanations */
  enable_explanations: boolean;
  /** Cache configuration */
  cache: {
    enabled: boolean;
    ttl_seconds: number;
    max_size: number;
  };
  /** Preprocessing defaults */
  preprocessing_defaults: PreprocessingOptions;
  /** Rate limiting */
  rate_limit: {
    requests_per_minute: number;
    batch_size_limit: number;
  };
}

/**
 * Classifier analytics data
 */
export interface ClassifierAnalytics {
  /** Usage statistics */
  usage: {
    total_predictions: number;
    predictions_today: number;
    unique_users: number;
    avg_predictions_per_user: number;
  };
  /** Performance statistics */
  performance: {
    avg_response_time: number;
    success_rate: number;
    error_rate: number;
    cache_hit_rate: number;
  };
  /** Model usage distribution */
  model_usage: Record<ClassifierModelType, {
    usage_count: number;
    percentage: number;
    avg_confidence: number;
  }>;
  /** Prediction distribution */
  prediction_distribution: {
    fake_percentage: number;
    real_percentage: number;
    high_confidence_percentage: number;
    low_confidence_percentage: number;
  };
  /** Error analysis */
  errors: {
    common_errors: Array<{
      error_type: string;
      count: number;
      percentage: number;
    }>;
    error_trends: Array<{
      date: string;
      error_count: number;
    }>;
  };
}