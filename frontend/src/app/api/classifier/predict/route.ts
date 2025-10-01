// frontend/src/app/api/classifier/predict/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';
import crypto from 'crypto';

/**
 * ================================================================
 * TYPE DEFINITIONS
 * ================================================================
 */

interface PredictRequest {
  text: string;
  model_type?: string;
  explain_prediction?: boolean;
  include_features?: boolean;
}

interface PredictResponse {
  prediction: 'real' | 'fake';
  confidence: number;
  probabilities: {
    real: number;
    fake: number;
  };
  model_info: {
    model_type: string;
    version: string;
  };
  features?: Record<string, any>;
  explanation?: {
    top_features: Array<{
      name: string;
      importance: number;
      contribution: 'positive' | 'negative';
    }>;
    confidence_factors: string[];
  };
  metadata: {
    processing_time_ms: number;
    timestamp: string;
    cached: boolean;
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
 * CACHING IMPLEMENTATION
 * ================================================================
 */

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

class PredictionCache {
  private cache = new Map<string, CacheEntry<any>>();
  private defaultTTL = 24 * 60 * 60 * 1000; // 24 hours

  /**
   * Generate cache key from request parameters
   */
  generateKey(text: string, modelType: string, explainPrediction: boolean): string {
    const hash = crypto
      .createHash('sha256')
      .update(`${text}:${modelType}:${explainPrediction}`)
      .digest('hex');

    return `predict:${hash}`;
  }

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

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return entry.data as T;
  }

  has(key: string): boolean {
    const entry = this.cache.get(key);

    if (!entry || Date.now() > entry.expiresAt) {
      if (entry) this.cache.delete(key);
      return false;
    }

    return true;
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
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
const predictionCache = new PredictionCache();

/**
 * ================================================================
 * RATE LIMITING IMPLEMENTATION
 * ================================================================
 */

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

class RateLimiter {
  private limits = new Map<string, RateLimitEntry>();
  private maxRequests = 100; // Max requests per window
  private windowMs = 60 * 60 * 1000; // 1 hour window

  check(userId: string): { allowed: boolean; remaining: number; resetAt: number } {
    const now = Date.now();
    const key = `ratelimit:${userId}`;
    const entry = this.limits.get(key);

    // If no entry or expired, create new
    if (!entry || now > entry.resetAt) {
      const newEntry: RateLimitEntry = {
        count: 1,
        resetAt: now + this.windowMs,
      };
      this.limits.set(key, newEntry);

      return {
        allowed: true,
        remaining: this.maxRequests - 1,
        resetAt: newEntry.resetAt,
      };
    }

    // Check if limit exceeded
    if (entry.count >= this.maxRequests) {
      return {
        allowed: false,
        remaining: 0,
        resetAt: entry.resetAt,
      };
    }

    // Increment counter
    entry.count++;
    this.limits.set(key, entry);

    return {
      allowed: true,
      remaining: this.maxRequests - entry.count,
      resetAt: entry.resetAt,
    };
  }

  cleanup(): void {
    const now = Date.now();
    const keysToDelete: string[] = [];

    this.limits.forEach((entry, key) => {
      if (now > entry.resetAt) {
        keysToDelete.push(key);
      }
    });

    keysToDelete.forEach(key => this.limits.delete(key));
  }
}

// Global rate limiter instance
const rateLimiter = new RateLimiter();

// Cleanup expired rate limits every 5 minutes
if (typeof setInterval !== 'undefined') {
  setInterval(() => rateLimiter.cleanup(), 5 * 60 * 1000);
}

/**
 * ================================================================
 * POST /api/classifier/predict
 *
 * Backend-for-Frontend (BFF) proxy for text classification
 *
 * Features:
 * - NextAuth.js authentication
 * - Input validation (text, model_type, options)
 * - SHA-256 based caching (24hr TTL)
 * - Rate limiting (100 requests/hour per user)
 * - Backend proxy with JWT forwarding
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
          error: 'Unauthorized',
          code: 'AUTHENTICATION_REQUIRED',
          details: 'You must be authenticated to use the prediction service',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 401 }
      );
    }

    const userId = session.user.id;

    // ================================================================
    // 2. RATE LIMITING
    // ================================================================
    const rateLimit = rateLimiter.check(userId);

    if (!rateLimit.allowed) {
      const resetDate = new Date(rateLimit.resetAt).toISOString();

      return NextResponse.json(
        {
          error: 'Rate limit exceeded',
          code: 'RATE_LIMIT_EXCEEDED',
          details: `Maximum requests exceeded. Limit resets at ${resetDate}`,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        {
          status: 429,
          headers: {
            'X-RateLimit-Limit': '100',
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': rateLimit.resetAt.toString(),
            'Retry-After': Math.ceil((rateLimit.resetAt - Date.now()) / 1000).toString(),
          },
        }
      );
    }

    // ================================================================
    // 3. REQUEST BODY PARSING & VALIDATION
    // ================================================================
    let body: PredictRequest;

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

    // Validate required field: text
    if (!body.text) {
      return NextResponse.json(
        {
          error: 'Missing required field',
          code: 'MISSING_TEXT',
          details: 'The "text" field is required',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (typeof body.text !== 'string') {
      return NextResponse.json(
        {
          error: 'Invalid field type',
          code: 'INVALID_TEXT_TYPE',
          details: 'The "text" field must be a string',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate text content
    const trimmedText = body.text.trim();

    if (trimmedText.length === 0) {
      return NextResponse.json(
        {
          error: 'Empty text',
          code: 'EMPTY_TEXT',
          details: 'The text field cannot be empty or contain only whitespace',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (trimmedText.length < 10) {
      return NextResponse.json(
        {
          error: 'Text too short',
          code: 'TEXT_TOO_SHORT',
          details: 'Text must be at least 10 characters long',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    if (trimmedText.length > 10000) {
      return NextResponse.json(
        {
          error: 'Text too long',
          code: 'TEXT_TOO_LONG',
          details: 'Text must be 10,000 characters or less',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate optional parameters
    const modelType = body.model_type || 'ensemble';
    const explainPrediction = body.explain_prediction ?? false;
    const includeFeatures = body.include_features ?? true;

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
    // 4. CHECK CACHE
    // ================================================================
    const cacheKey = predictionCache.generateKey(trimmedText, modelType, explainPrediction);
    const cachedResult = predictionCache.get<PredictResponse>(cacheKey);

    if (cachedResult) {
      console.log(`[PREDICT] Cache HIT for key: ${cacheKey}`);

      // Update metadata for cached response
      const responseWithMetadata = {
        ...cachedResult,
        metadata: {
          ...cachedResult.metadata,
          cached: true,
          processing_time_ms: Date.now() - startTime,
        },
      };

      return NextResponse.json(responseWithMetadata, {
        status: 200,
        headers: {
          'X-Cache-Status': 'HIT',
          'X-RateLimit-Remaining': rateLimit.remaining.toString(),
          'Cache-Control': 'private, max-age=86400',
        },
      });
    }

    console.log(`[PREDICT] Cache MISS for key: ${cacheKey}`);

    // ================================================================
    // 5. BACKEND API REQUEST
    // ================================================================
    const backendApiUrl = process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL;

    if (!backendApiUrl) {
      console.error('[PREDICT] Backend API URL not configured');
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

    const backendUrl = `${backendApiUrl}/api/v1/classifier/predict`;
    const token = session.user.id; // In production, use actual JWT token

    console.log(`[PREDICT] Forwarding to backend: ${backendUrl}`);

    // Make request to Python backend
    const backendResponse = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Authorization': token ? `Bearer ${token}` : '',
        'Content-Type': 'application/json',
        'User-Agent': 'NextJS-BFF/1.0',
        'X-Request-ID': crypto.randomUUID(),
      },
      body: JSON.stringify({
        text: trimmedText,
        model_type: modelType,
        explain_prediction: explainPrediction,
        include_features: includeFeatures,
      }),
      // Timeout after 60 seconds (predictions can take longer)
      signal: AbortSignal.timeout(60000),
    });

    // ================================================================
    // 6. HANDLE BACKEND ERRORS
    // ================================================================
    if (!backendResponse.ok) {
      const errorBody = await backendResponse.json().catch(() => ({
        error: 'Unknown backend error',
      }));

      console.error(
        `[PREDICT] Backend error: ${backendResponse.status} ${backendResponse.statusText}`,
        errorBody
      );

      if (backendResponse.status === 400) {
        return NextResponse.json(
          {
            error: 'Backend validation error',
            code: 'BACKEND_VALIDATION_ERROR',
            details: errorBody.error || 'The backend service rejected the request',
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
            details: 'Failed to authenticate with backend service',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 502 }
        );
      }

      // Default to 502 Bad Gateway
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
    // 7. PARSE RESPONSE & CACHE
    // ================================================================
    const predictionData: PredictResponse = await backendResponse.json();

    // Add processing metadata
    const responseWithMetadata: PredictResponse = {
      ...predictionData,
      metadata: {
        processing_time_ms: Date.now() - startTime,
        timestamp: new Date().toISOString(),
        cached: false,
      },
    };

    // Cache successful prediction (24 hours)
    predictionCache.set(cacheKey, responseWithMetadata, 24 * 60 * 60 * 1000);

    console.log(
      `[PREDICT] Prediction completed: ${predictionData.prediction} (${predictionData.confidence})`
    );

    // ================================================================
    // 8. RETURN SUCCESS RESPONSE
    // ================================================================
    return NextResponse.json(responseWithMetadata, {
      status: 200,
      headers: {
        'X-Cache-Status': 'MISS',
        'X-RateLimit-Remaining': rateLimit.remaining.toString(),
        'Cache-Control': 'private, max-age=86400',
      },
    });
  } catch (error) {
    // ================================================================
    // 9. HANDLE UNEXPECTED ERRORS
    // ================================================================
    console.error('[PREDICT] Unexpected error:', error);

    // Handle timeout errors
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        {
          error: 'Backend request timeout',
          code: 'REQUEST_TIMEOUT',
          details: 'The prediction service did not respond in time (60s timeout)',
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
          details: 'Unable to connect to the prediction service',
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
        details: 'An unexpected error occurred during prediction',
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
      details: 'This endpoint only supports POST requests',
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
      details: 'This endpoint only supports POST requests',
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
      details: 'This endpoint only supports POST requests',
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
      details: 'This endpoint only supports POST requests',
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
