// frontend/src/app/api/classifier/train/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';
import crypto from 'crypto';

/**
 * ================================================================
 * TYPE DEFINITIONS
 * ================================================================
 */

interface TrainRequest {
  training_data: {
    dataset_id: string;
    train_split?: number;
    validation_split?: number;
    test_split?: number;
  };
  model_config: {
    model_type: string;
    learning_rate?: number;
    batch_size?: number;
    epochs?: number;
    max_length?: number;
    dropout_rate?: number;
    optimizer?: string;
    scheduler?: string;
    early_stopping_patience?: number;
    save_strategy?: string;
  };
  experiment_name?: string;
  description?: string;
  tags?: string[];
}

interface TrainResponse {
  training_id: string;
  status: 'started' | 'queued';
  message: string;
  job_info: {
    created_at: string;
    estimated_duration_minutes?: number;
    priority?: string;
  };
  experiment?: {
    name: string;
    description?: string;
    tags: string[];
  };
  _links: {
    self: string;
    status: string;
    cancel: string;
  };
}

interface ErrorResponse {
  error: string;
  code: string;
  details?: string;
  timestamp: string;
}

/**
 * ================================================================
 * VALIDATION HELPERS
 * ================================================================
 */

/**
 * Validates training data configuration
 */
function validateTrainingData(training_data: any): string | null {
  if (!training_data) {
    return 'training_data is required';
  }

  if (typeof training_data !== 'object') {
    return 'training_data must be an object';
  }

  if (!training_data.dataset_id) {
    return 'training_data.dataset_id is required';
  }

  if (typeof training_data.dataset_id !== 'string') {
    return 'training_data.dataset_id must be a string';
  }

  // Validate splits if provided
  if (training_data.train_split !== undefined) {
    if (typeof training_data.train_split !== 'number' || training_data.train_split <= 0 || training_data.train_split > 1) {
      return 'training_data.train_split must be a number between 0 and 1';
    }
  }

  if (training_data.validation_split !== undefined) {
    if (typeof training_data.validation_split !== 'number' || training_data.validation_split < 0 || training_data.validation_split > 1) {
      return 'training_data.validation_split must be a number between 0 and 1';
    }
  }

  if (training_data.test_split !== undefined) {
    if (typeof training_data.test_split !== 'number' || training_data.test_split < 0 || training_data.test_split > 1) {
      return 'training_data.test_split must be a number between 0 and 1';
    }
  }

  // Validate that splits sum to <= 1
  const totalSplit = (training_data.train_split || 0.7) +
                     (training_data.validation_split || 0.15) +
                     (training_data.test_split || 0.15);

  if (totalSplit > 1.01) { // Allow small floating point error
    return 'training_data splits must sum to 1.0 or less';
  }

  return null;
}

/**
 * Validates model configuration
 */
function validateModelConfig(model_config: any): string | null {
  if (!model_config) {
    return 'model_config is required';
  }

  if (typeof model_config !== 'object') {
    return 'model_config must be an object';
  }

  if (!model_config.model_type) {
    return 'model_config.model_type is required';
  }

  if (typeof model_config.model_type !== 'string') {
    return 'model_config.model_type must be a string';
  }

  // Validate model type
  const validModelTypes = [
    'bert',
    'roberta',
    'distilbert',
    'albert',
    'xlnet',
    'electra',
    'lstm',
    'gru',
    'random_forest',
    'gradient_boosting',
    'svm',
    'naive_bayes',
    'logistic_regression',
    'ensemble',
  ];

  if (!validModelTypes.includes(model_config.model_type)) {
    return `model_config.model_type must be one of: ${validModelTypes.join(', ')}`;
  }

  // Validate numeric parameters
  if (model_config.learning_rate !== undefined) {
    if (typeof model_config.learning_rate !== 'number' || model_config.learning_rate <= 0 || model_config.learning_rate > 1) {
      return 'model_config.learning_rate must be a number between 0 and 1';
    }
  }

  if (model_config.batch_size !== undefined) {
    if (typeof model_config.batch_size !== 'number' || model_config.batch_size < 1 || model_config.batch_size > 512) {
      return 'model_config.batch_size must be a number between 1 and 512';
    }
  }

  if (model_config.epochs !== undefined) {
    if (typeof model_config.epochs !== 'number' || model_config.epochs < 1 || model_config.epochs > 1000) {
      return 'model_config.epochs must be a number between 1 and 1000';
    }
  }

  if (model_config.max_length !== undefined) {
    if (typeof model_config.max_length !== 'number' || model_config.max_length < 1 || model_config.max_length > 2048) {
      return 'model_config.max_length must be a number between 1 and 2048';
    }
  }

  if (model_config.dropout_rate !== undefined) {
    if (typeof model_config.dropout_rate !== 'number' || model_config.dropout_rate < 0 || model_config.dropout_rate > 1) {
      return 'model_config.dropout_rate must be a number between 0 and 1';
    }
  }

  if (model_config.early_stopping_patience !== undefined) {
    if (typeof model_config.early_stopping_patience !== 'number' || model_config.early_stopping_patience < 1) {
      return 'model_config.early_stopping_patience must be a positive number';
    }
  }

  return null;
}

/**
 * ================================================================
 * POST /api/classifier/train
 *
 * Initiates a background model training job on the Python backend.
 *
 * Security:
 * - NextAuth.js authentication required (401 if not authenticated)
 * - Role-based authorization: only 'admin' or 'researcher' roles allowed (403 if unauthorized)
 *
 * Features:
 * - Input validation (training_data, model_config)
 * - Asynchronous job initiation (non-blocking)
 * - Backend proxy with JWT forwarding
 * - Immediate response with training_id (202 Accepted)
 * - Comprehensive error handling
 * ================================================================
 */
export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    // ================================================================
    // 1. AUTHENTICATION
    // ================================================================
    const session = await getServerSession(authOptions);

    if (!session || !session.user) {
      return NextResponse.json(
        {
          error: 'Authentication required',
          code: 'UNAUTHORIZED',
          details: 'You must be authenticated to initiate model training',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 401 }
      );
    }

    // ================================================================
    // 2. AUTHORIZATION (Role-Based Access Control)
    // ================================================================
    const userRole = session.user.role;
    const authorizedRoles = ['admin', 'researcher'];

    if (!authorizedRoles.includes(userRole)) {
      console.warn(
        `[TRAIN] Authorization denied for user ${session.user.id} with role '${userRole}'`
      );

      return NextResponse.json(
        {
          error: 'Insufficient permissions',
          code: 'FORBIDDEN',
          details: `Training models requires one of the following roles: ${authorizedRoles.join(', ')}. Your role: ${userRole}`,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 403 }
      );
    }

    console.log(
      `[TRAIN] Training request authorized for user ${session.user.id} (${userRole})`
    );

    // ================================================================
    // 3. REQUEST BODY PARSING
    // ================================================================
    let body: TrainRequest;

    try {
      body = await request.json();
    } catch (error) {
      return NextResponse.json(
        {
          error: 'Invalid JSON',
          code: 'INVALID_JSON',
          details: 'Request body must be valid JSON',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // ================================================================
    // 4. INPUT VALIDATION
    // ================================================================

    // Validate training_data
    const trainingDataError = validateTrainingData(body.training_data);
    if (trainingDataError) {
      return NextResponse.json(
        {
          error: 'Invalid training_data',
          code: 'VALIDATION_ERROR',
          details: trainingDataError,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate model_config
    const modelConfigError = validateModelConfig(body.model_config);
    if (modelConfigError) {
      return NextResponse.json(
        {
          error: 'Invalid model_config',
          code: 'VALIDATION_ERROR',
          details: modelConfigError,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate optional fields
    if (body.experiment_name && typeof body.experiment_name !== 'string') {
      return NextResponse.json(
        {
          error: 'Invalid experiment_name',
          code: 'VALIDATION_ERROR',
          details: 'experiment_name must be a string',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (body.description && typeof body.description !== 'string') {
      return NextResponse.json(
        {
          error: 'Invalid description',
          code: 'VALIDATION_ERROR',
          details: 'description must be a string',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (body.tags && !Array.isArray(body.tags)) {
      return NextResponse.json(
        {
          error: 'Invalid tags',
          code: 'VALIDATION_ERROR',
          details: 'tags must be an array',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // ================================================================
    // 5. BACKEND API REQUEST (Asynchronous Job Initiation)
    // ================================================================
    const backendApiUrl = process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL;

    if (!backendApiUrl) {
      console.error('[TRAIN] Backend API URL not configured');
      return NextResponse.json(
        {
          error: 'Backend configuration error',
          code: 'BACKEND_URL_MISSING',
          details: 'Backend API URL is not configured',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 500 }
      );
    }

    const backendUrl = `${backendApiUrl}/api/v1/classifier/train`;
    const token = session.user.id; // In production, use actual JWT token

    console.log(`[TRAIN] Initiating training job on backend: ${backendUrl}`);
    console.log(`[TRAIN] Dataset: ${body.training_data.dataset_id}, Model: ${body.model_config.model_type}`);

    // Forward request to Python backend
    const backendResponse = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Authorization': token ? `Bearer ${token}` : '',
        'Content-Type': 'application/json',
        'User-Agent': 'NextJS-BFF/1.0',
        'X-Request-ID': crypto.randomUUID(),
        'X-User-ID': session.user.id,
        'X-User-Role': session.user.role,
      },
      body: JSON.stringify({
        training_data: body.training_data,
        model_config: body.model_config,
        experiment_name: body.experiment_name,
        description: body.description,
        tags: body.tags || [],
        user_id: session.user.id,
      }),
      // Timeout after 30 seconds (job initiation should be fast)
      signal: AbortSignal.timeout(30000),
    });

    // ================================================================
    // 6. HANDLE BACKEND ERRORS
    // ================================================================
    if (!backendResponse.ok) {
      const errorBody = await backendResponse.json().catch(() => ({
        error: 'Unknown backend error',
      }));

      console.error(
        `[TRAIN] Backend error: ${backendResponse.status} ${backendResponse.statusText}`,
        errorBody
      );

      if (backendResponse.status === 400) {
        return NextResponse.json(
          {
            error: 'Backend validation error',
            code: 'BACKEND_VALIDATION_ERROR',
            details: errorBody.error || errorBody.detail || 'The backend service rejected the training request',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 400 }
        );
      }

      if (backendResponse.status === 401 || backendResponse.status === 403) {
        return NextResponse.json(
          {
            error: 'Backend authentication failed',
            code: 'BACKEND_AUTH_FAILED',
            details: 'Failed to authenticate with backend training service',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 502 }
        );
      }

      if (backendResponse.status === 404) {
        return NextResponse.json(
          {
            error: 'Dataset not found',
            code: 'DATASET_NOT_FOUND',
            details: errorBody.error || 'The specified dataset does not exist on the backend',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 404 }
        );
      }

      if (backendResponse.status === 409) {
        return NextResponse.json(
          {
            error: 'Training job conflict',
            code: 'TRAINING_CONFLICT',
            details: errorBody.error || 'A training job with similar parameters is already running',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 409 }
        );
      }

      if (backendResponse.status === 503) {
        return NextResponse.json(
          {
            error: 'Training service unavailable',
            code: 'SERVICE_UNAVAILABLE',
            details: 'The training service is temporarily unavailable. Please try again later',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 503 }
        );
      }

      // Default to 502 Bad Gateway
      return NextResponse.json(
        {
          error: 'Backend service error',
          code: 'BACKEND_ERROR',
          details: errorBody.error || errorBody.detail || 'The backend training service encountered an error',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 502 }
      );
    }

    // ================================================================
    // 7. PARSE RESPONSE & RETURN IMMEDIATELY (202 Accepted)
    // ================================================================
    const trainingData = await backendResponse.json();

    // Extract training_id from backend response
    const trainingId = trainingData.training_id || trainingData.job_id || trainingData.id;

    if (!trainingId) {
      console.error('[TRAIN] Backend response missing training_id:', trainingData);
      return NextResponse.json(
        {
          error: 'Invalid backend response',
          code: 'INVALID_BACKEND_RESPONSE',
          details: 'Backend did not return a training job ID',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 502 }
      );
    }

    // Construct frontend API URLs for job management
    const baseUrl = process.env.NEXTAUTH_URL || 'http://localhost:3000';

    const response: TrainResponse = {
      training_id: trainingId,
      status: trainingData.status || 'started',
      message: trainingData.message || 'Training job has been started successfully in the background',
      job_info: {
        created_at: trainingData.created_at || new Date().toISOString(),
        estimated_duration_minutes: trainingData.estimated_duration_minutes,
        priority: trainingData.priority || 'normal',
      },
      experiment: body.experiment_name || body.description || body.tags ? {
        name: body.experiment_name || `training_${trainingId}`,
        description: body.description,
        tags: body.tags || [],
      } : undefined,
      _links: {
        self: `${baseUrl}/api/classifier/train/${trainingId}`,
        status: `${baseUrl}/api/classifier/train/${trainingId}/status`,
        cancel: `${baseUrl}/api/classifier/train/${trainingId}/cancel`,
      },
    };

    const processingTime = Date.now() - startTime;

    console.log(
      `[TRAIN] Training job initiated successfully: ${trainingId} (${processingTime}ms)`
    );

    // ================================================================
    // 8. RETURN 202 ACCEPTED (Job accepted, processing in background)
    // ================================================================
    return NextResponse.json(response, {
      status: 202,
      headers: {
        'Location': response._links.self,
        'X-Training-ID': trainingId,
        'X-Processing-Time': processingTime.toString(),
      },
    });

  } catch (error) {
    // ================================================================
    // 9. HANDLE UNEXPECTED ERRORS
    // ================================================================
    console.error('[TRAIN] Unexpected error:', error);

    // Handle timeout errors
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        {
          error: 'Backend request timeout',
          code: 'REQUEST_TIMEOUT',
          details: 'The training service did not respond in time (30s timeout)',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 504 }
      );
    }

    // Handle network errors
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        {
          error: 'Backend connection failed',
          code: 'CONNECTION_FAILED',
          details: 'Unable to connect to the training service',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 503 }
      );
    }

    // Generic error
    return NextResponse.json(
      {
        error: 'Internal server error',
        code: 'INTERNAL_SERVER_ERROR',
        details: 'An unexpected error occurred during training job initiation',
        timestamp: new Date().toISOString(),
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

/**
 * ================================================================
 * REJECT OTHER HTTP METHODS
 * ================================================================
 */

export async function GET() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports POST requests for initiating training jobs',
      timestamp: new Date().toISOString(),
    } as ErrorResponse,
    {
      status: 405,
      headers: {
        Allow: 'POST',
      },
    }
  );
}

export async function PUT() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports POST requests for initiating training jobs',
      timestamp: new Date().toISOString(),
    } as ErrorResponse,
    {
      status: 405,
      headers: {
        Allow: 'POST',
      },
    }
  );
}

export async function DELETE() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports POST requests for initiating training jobs',
      timestamp: new Date().toISOString(),
    } as ErrorResponse,
    {
      status: 405,
      headers: {
        Allow: 'POST',
      },
    }
  );
}

export async function PATCH() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports POST requests for initiating training jobs',
      timestamp: new Date().toISOString(),
    } as ErrorResponse,
    {
      status: 405,
      headers: {
        Allow: 'POST',
      },
    }
  );
}
