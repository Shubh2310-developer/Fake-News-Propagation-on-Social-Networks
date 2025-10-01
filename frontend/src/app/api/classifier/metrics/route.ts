// frontend/src/app/api/classifier/metrics/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';

/**
 * Server-side cache for model metrics
 * Key format: `metrics:${model_type}`
 */
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

class MetricsCache {
  private cache = new Map<string, CacheEntry<any>>();
  private defaultTTL = 60 * 60 * 1000; // 1 hour in milliseconds

  set<T>(key: string, data: T, ttl?: number): void {
    const now = Date.now();
    this.cache.set(key, {
      data,
      timestamp: now,
      expiresAt: now + (ttl || this.defaultTTL),
    });
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);

    if (!entry) {
      return null;
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return entry.data as T;
  }

  has(key: string): boolean {
    const entry = this.cache.get(key);

    if (!entry) {
      return false;
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    // Clean expired entries first
    const now = Date.now();
    const keysToDelete: string[] = [];

    this.cache.forEach((entry, key) => {
      if (now > entry.expiresAt) {
        keysToDelete.push(key);
      }
    });

    keysToDelete.forEach(key => this.cache.delete(key));

    return this.cache.size;
  }
}

// Global cache instance
const metricsCache = new MetricsCache();

/**
 * Type definitions for classifier metrics
 */
interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  matthews_correlation_coefficient?: number;
  confusion_matrix?: {
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
  };
  class_metrics?: {
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
  model_info?: {
    model_version: string;
    model_type: string;
    training_date: string;
    last_updated: string;
    dataset_size: number;
    validation_size: number;
  };
  performance_history?: Array<{
    date: string;
    accuracy: number;
    f1_score: number;
    dataset_version: string;
  }>;
}

interface ErrorResponse {
  error: string;
  code?: string;
  details?: string;
  timestamp: string;
}

/**
 * GET /api/classifier/metrics
 *
 * Backend-for-Frontend (BFF) proxy for classifier metrics
 *
 * Features:
 * - Authentication via NextAuth.js session
 * - JWT token forwarding to Python backend
 * - Server-side caching with configurable TTL
 * - Robust error handling
 * - Query parameter support for model filtering
 *
 * Query Parameters:
 * - model_type: Filter by model type (e.g., 'ensemble', 'bert', 'random_forest')
 * - include_history: Include performance history (default: true)
 * - include_confusion_matrix: Include confusion matrix (default: true)
 */
export async function GET(request: NextRequest) {
  try {
    // ================================================================
    // 1. Authentication Check
    // ================================================================
    const session = await getServerSession(authOptions);

    if (!session || !session.user) {
      return NextResponse.json(
        {
          error: 'Unauthorized',
          code: 'AUTHENTICATION_REQUIRED',
          details: 'You must be authenticated to access this resource',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 401 }
      );
    }

    // ================================================================
    // 2. Extract and Validate Query Parameters
    // ================================================================
    const { searchParams } = new URL(request.url);
    const modelType = searchParams.get('model_type') || 'ensemble';
    const includeHistory = searchParams.get('include_history') !== 'false';
    const includeConfusionMatrix = searchParams.get('include_confusion_matrix') !== 'false';

    // Validate model_type parameter
    const validModelTypes = [
      'ensemble',
      'random_forest',
      'gradient_boosting',
      'svm',
      'naive_bayes',
      'logistic_regression',
      'bert',
      'lstm',
    ];

    if (!validModelTypes.includes(modelType)) {
      return NextResponse.json(
        {
          error: 'Invalid model type',
          code: 'INVALID_MODEL_TYPE',
          details: `Model type must be one of: ${validModelTypes.join(', ')}`,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // ================================================================
    // 3. Check Cache
    // ================================================================
    const cacheKey = `metrics:${modelType}:${includeHistory}:${includeConfusionMatrix}`;
    const cachedData = metricsCache.get<ModelMetrics>(cacheKey);

    if (cachedData) {
      console.log(`[METRICS] Cache HIT for ${cacheKey}`);

      return NextResponse.json(
        {
          ...cachedData,
          metadata: {
            source: 'cache',
            cached_at: new Date().toISOString(),
            cache_key: cacheKey,
          },
        },
        {
          status: 200,
          headers: {
            'X-Cache-Status': 'HIT',
            'Cache-Control': 'private, max-age=3600',
            'Content-Type': 'application/json',
          },
        }
      );
    }

    console.log(`[METRICS] Cache MISS for ${cacheKey}`);

    // ================================================================
    // 4. Fetch from Python Backend
    // ================================================================
    const backendApiUrl = process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL;

    if (!backendApiUrl) {
      console.error('[METRICS] Backend API URL not configured');
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

    // Construct backend URL with query parameters
    const backendUrl = new URL(`${backendApiUrl}/api/v1/classifier/metrics`);
    backendUrl.searchParams.set('model_type', modelType);
    if (!includeHistory) backendUrl.searchParams.set('include_history', 'false');
    if (!includeConfusionMatrix) backendUrl.searchParams.set('include_confusion_matrix', 'false');

    // Get JWT token from session (if available)
    const token = session.user.id; // In production, use actual JWT token

    console.log(`[METRICS] Fetching from backend: ${backendUrl.toString()}`);

    // Make request to Python backend
    const backendResponse = await fetch(backendUrl.toString(), {
      method: 'GET',
      headers: {
        'Authorization': token ? `Bearer ${token}` : '',
        'Content-Type': 'application/json',
        'User-Agent': 'NextJS-BFF/1.0',
        'X-Request-ID': crypto.randomUUID(),
      },
      // Timeout after 30 seconds
      signal: AbortSignal.timeout(30000),
    });

    // ================================================================
    // 5. Handle Backend Errors
    // ================================================================
    if (!backendResponse.ok) {
      const errorBody = await backendResponse.json().catch(() => ({
        error: 'Unknown backend error',
      }));

      console.error(
        `[METRICS] Backend error: ${backendResponse.status} ${backendResponse.statusText}`,
        errorBody
      );

      // Map backend status codes to appropriate client responses
      if (backendResponse.status === 404) {
        return NextResponse.json(
          {
            error: 'Model not found',
            code: 'MODEL_NOT_FOUND',
            details: errorBody.error || `No metrics found for model type: ${modelType}`,
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 404 }
        );
      }

      if (backendResponse.status === 401 || backendResponse.status === 403) {
        return NextResponse.json(
          {
            error: 'Backend authentication failed',
            code: 'BACKEND_AUTH_FAILED',
            details: 'Failed to authenticate with backend service',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 502 }
        );
      }

      // Default to 502 Bad Gateway for other backend errors
      return NextResponse.json(
        {
          error: 'Backend service error',
          code: 'BACKEND_ERROR',
          details: errorBody.error || 'The backend service encountered an error',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 502 }
      );
    }

    // ================================================================
    // 6. Parse and Cache Response
    // ================================================================
    const metricsData: ModelMetrics = await backendResponse.json();

    // Cache the successful response (1 hour TTL)
    metricsCache.set(cacheKey, metricsData, 60 * 60 * 1000);

    console.log(`[METRICS] Successfully fetched and cached metrics for ${modelType}`);

    // ================================================================
    // 7. Return Successful Response
    // ================================================================
    return NextResponse.json(
      {
        ...metricsData,
        metadata: {
          source: 'backend',
          retrieved_at: new Date().toISOString(),
          cache_ttl_seconds: 3600,
          model_type: modelType,
        },
      },
      {
        status: 200,
        headers: {
          'X-Cache-Status': 'MISS',
          'Cache-Control': 'private, max-age=3600',
          'Content-Type': 'application/json',
        },
      }
    );
  } catch (error) {
    // ================================================================
    // 8. Handle Unexpected Errors
    // ================================================================
    console.error('[METRICS] Unexpected error:', error);

    // Handle timeout errors
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        {
          error: 'Backend request timeout',
          code: 'REQUEST_TIMEOUT',
          details: 'The backend service did not respond in time',
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
          details: 'Unable to connect to the backend service',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 503 }
      );
    }

    // Generic error response
    return NextResponse.json(
      {
        error: 'Internal server error',
        code: 'INTERNAL_SERVER_ERROR',
        details: 'An unexpected error occurred while retrieving metrics',
        timestamp: new Date().toISOString(),
      } as ErrorResponse,
      { status: 500 }
    );
  }
}

/**
 * Reject all other HTTP methods
 */
export async function POST() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports GET requests',
      timestamp: new Date().toISOString(),
    } as ErrorResponse,
    {
      status: 405,
      headers: {
        Allow: 'GET',
      },
    }
  );
}

export async function PUT() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports GET requests',
      timestamp: new Date().toISOString(),
    } as ErrorResponse,
    {
      status: 405,
      headers: {
        Allow: 'GET',
      },
    }
  );
}

export async function DELETE() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports GET requests',
      timestamp: new Date().toISOString(),
    } as ErrorResponse,
    {
      status: 405,
      headers: {
        Allow: 'GET',
      },
    }
  );
}

export async function PATCH() {
  return NextResponse.json(
    {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
      details: 'This endpoint only supports GET requests',
      timestamp: new Date().toISOString(),
    } as ErrorResponse,
    {
      status: 405,
      headers: {
        Allow: 'GET',
      },
    }
  );
}
