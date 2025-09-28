// frontend/src/app/api/classifier/metrics/route.ts

import { NextRequest, NextResponse } from 'next/server';

// Types for metrics response
interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  matthews_correlation_coefficient: number;
  confusion_matrix: {
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
  };
  class_metrics: {
    real: {
      precision: number;
      recall: number;
      f1_score: number;
      support: number;
    };
    fake: {
      precision: number;
      recall: number;
      f1_score: number;
      support: number;
    };
  };
  model_info: {
    model_version: string;
    model_type: string;
    training_date: string;
    last_updated: string;
    dataset_size: number;
    validation_size: number;
  };
  performance_history: Array<{
    date: string;
    accuracy: number;
    f1_score: number;
    dataset_version: string;
  }>;
}

interface ErrorResponse {
  error: string;
  code: string;
  details?: string;
}

// Mock model metrics data
const getCurrentModelMetrics = (): ModelMetrics => {
  // Simulate some realistic but randomized metrics
  const baseAccuracy = 0.94 + (Math.random() - 0.5) * 0.02;
  const basePrecision = 0.92 + (Math.random() - 0.5) * 0.03;
  const baseRecall = 0.89 + (Math.random() - 0.5) * 0.03;

  const f1Score = 2 * (basePrecision * baseRecall) / (basePrecision + baseRecall);
  const aucRoc = 0.96 + (Math.random() - 0.5) * 0.02;
  const mcc = 0.88 + (Math.random() - 0.5) * 0.03;

  // Generate confusion matrix values
  const totalSamples = 10000;
  const truePositives = Math.round(totalSamples * 0.45 * baseRecall);
  const falseNegatives = Math.round(totalSamples * 0.45) - truePositives;
  const trueNegatives = Math.round(totalSamples * 0.55 * (basePrecision * 0.95));
  const falsePositives = Math.round(totalSamples * 0.55) - trueNegatives;

  return {
    accuracy: Math.round(baseAccuracy * 1000) / 1000,
    precision: Math.round(basePrecision * 1000) / 1000,
    recall: Math.round(baseRecall * 1000) / 1000,
    f1_score: Math.round(f1Score * 1000) / 1000,
    auc_roc: Math.round(aucRoc * 1000) / 1000,
    matthews_correlation_coefficient: Math.round(mcc * 1000) / 1000,

    confusion_matrix: {
      true_positives: truePositives,
      true_negatives: trueNegatives,
      false_positives: falsePositives,
      false_negatives: falseNegatives
    },

    class_metrics: {
      real: {
        precision: Math.round((trueNegatives / (trueNegatives + falseNegatives)) * 1000) / 1000,
        recall: Math.round((trueNegatives / (trueNegatives + falsePositives)) * 1000) / 1000,
        f1_score: Math.round(((2 * trueNegatives) / (2 * trueNegatives + falsePositives + falseNegatives)) * 1000) / 1000,
        support: trueNegatives + falsePositives
      },
      fake: {
        precision: Math.round((truePositives / (truePositives + falsePositives)) * 1000) / 1000,
        recall: Math.round((truePositives / (truePositives + falseNegatives)) * 1000) / 1000,
        f1_score: Math.round(((2 * truePositives) / (2 * truePositives + falsePositives + falseNegatives)) * 1000) / 1000,
        support: truePositives + falseNegatives
      }
    },

    model_info: {
      model_version: 'bert-large-v2.1',
      model_type: 'BERT Large (Fine-tuned)',
      training_date: '2024-09-25T10:30:00Z',
      last_updated: new Date().toISOString(),
      dataset_size: 75000,
      validation_size: 15000
    },

    performance_history: [
      {
        date: '2024-09-20T00:00:00Z',
        accuracy: 0.912,
        f1_score: 0.895,
        dataset_version: 'v1.0'
      },
      {
        date: '2024-09-22T00:00:00Z',
        accuracy: 0.923,
        f1_score: 0.908,
        dataset_version: 'v1.1'
      },
      {
        date: '2024-09-24T00:00:00Z',
        accuracy: 0.935,
        f1_score: 0.921,
        dataset_version: 'v1.2'
      },
      {
        date: '2024-09-25T00:00:00Z',
        accuracy: Math.round(baseAccuracy * 1000) / 1000,
        f1_score: Math.round(f1Score * 1000) / 1000,
        dataset_version: 'v2.0'
      }
    ]
  };
};

export async function GET(request: NextRequest) {
  try {
    // Optional query parameters for filtering metrics
    const { searchParams } = new URL(request.url);
    const includeHistory = searchParams.get('include_history') !== 'false';
    const includeConfusionMatrix = searchParams.get('include_confusion_matrix') !== 'false';
    const modelVersion = searchParams.get('model_version');

    // Simulate some processing delay
    await new Promise(resolve => setTimeout(resolve, 100));

    // Get current model metrics
    let metrics = getCurrentModelMetrics();

    // Filter by model version if specified
    if (modelVersion && modelVersion !== metrics.model_info.model_version) {
      return NextResponse.json(
        {
          error: 'Model version not found',
          code: 'MODEL_VERSION_NOT_FOUND',
          details: `No metrics available for model version: ${modelVersion}`
        } as ErrorResponse,
        { status: 404 }
      );
    }

    // Apply filtering based on query parameters
    if (!includeHistory) {
      delete (metrics as any).performance_history;
    }

    if (!includeConfusionMatrix) {
      delete (metrics as any).confusion_matrix;
    }

    // Add response metadata
    const response = {
      ...metrics,
      metadata: {
        retrieved_at: new Date().toISOString(),
        cache_ttl_seconds: 300, // 5 minutes
        api_version: '1.0'
      }
    };

    // Log metrics retrieval
    console.log(`[CLASSIFIER] Metrics retrieved for model ${metrics.model_info.model_version}`);

    return NextResponse.json(response, {
      status: 200,
      headers: {
        'Cache-Control': 'public, max-age=300', // Cache for 5 minutes
        'Content-Type': 'application/json'
      }
    });

  } catch (error) {
    console.error('[CLASSIFIER] Metrics retrieval error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error during metrics retrieval',
        code: 'METRICS_RETRIEVAL_FAILED',
        details: 'An unexpected error occurred while retrieving model metrics'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

// Handle model metrics comparison
export async function POST(request: NextRequest) {
  try {
    let body: { modelVersions: string[] };

    try {
      body = await request.json();
    } catch (error) {
      return NextResponse.json(
        {
          error: 'Invalid JSON in request body',
          code: 'INVALID_JSON',
          details: 'Request body must be valid JSON'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (!body.modelVersions || !Array.isArray(body.modelVersions)) {
      return NextResponse.json(
        {
          error: 'Invalid request body',
          code: 'INVALID_MODEL_VERSIONS',
          details: 'modelVersions must be an array of model version strings'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (body.modelVersions.length > 5) {
      return NextResponse.json(
        {
          error: 'Too many models requested',
          code: 'TOO_MANY_MODELS',
          details: 'Cannot compare more than 5 models at once'
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Mock comparison data
    const availableModels = [
      'bert-large-v2.1',
      'bert-large-v2.0',
      'roberta-large-v1.3',
      'distilbert-v1.1'
    ];

    const invalidModels = body.modelVersions.filter(v => !availableModels.includes(v));
    if (invalidModels.length > 0) {
      return NextResponse.json(
        {
          error: 'Invalid model versions',
          code: 'INVALID_MODEL_VERSIONS',
          details: `The following model versions are not available: ${invalidModels.join(', ')}`
        } as ErrorResponse,
        { status: 404 }
      );
    }

    // Generate comparison metrics
    const comparison = body.modelVersions.map(version => {
      const baseMetrics = getCurrentModelMetrics();

      // Adjust metrics slightly for different versions
      const versionMultiplier = version.includes('v2.1') ? 1.0 :
                               version.includes('v2.0') ? 0.98 :
                               version.includes('v1.3') ? 0.95 : 0.92;

      return {
        model_version: version,
        accuracy: Math.round(baseMetrics.accuracy * versionMultiplier * 1000) / 1000,
        precision: Math.round(baseMetrics.precision * versionMultiplier * 1000) / 1000,
        recall: Math.round(baseMetrics.recall * versionMultiplier * 1000) / 1000,
        f1_score: Math.round(baseMetrics.f1_score * versionMultiplier * 1000) / 1000,
        auc_roc: Math.round(baseMetrics.auc_roc * versionMultiplier * 1000) / 1000,
        training_date: baseMetrics.model_info.training_date,
        dataset_size: baseMetrics.model_info.dataset_size
      };
    });

    return NextResponse.json({
      comparison,
      metadata: {
        compared_at: new Date().toISOString(),
        models_count: comparison.length
      }
    });

  } catch (error) {
    console.error('[CLASSIFIER] Metrics comparison error:', error);

    return NextResponse.json(
      {
        error: 'Internal server error during metrics comparison',
        code: 'METRICS_COMPARISON_FAILED',
        details: 'An unexpected error occurred while comparing model metrics'
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

// Handle unsupported methods
export async function PUT() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint supports GET (retrieve metrics) and POST (compare models) requests'
    } as ErrorResponse,
    { status: 405 }
  );
}

export async function DELETE() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint supports GET (retrieve metrics) and POST (compare models) requests'
    } as ErrorResponse,
    { status: 405 }
  );
}