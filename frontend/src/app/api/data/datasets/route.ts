// frontend/src/app/api/data/datasets/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';

/**
 * ================================================================
 * TYPE DEFINITIONS
 * ================================================================
 */

interface Dataset {
  id: string;
  name: string;
  description: string;
  size: string;
  features: string[];
  usage: string;
  source?: string;
  format?: string;
  lastUpdated?: string;
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

class DatasetCache {
  private cache = new Map<string, CacheEntry<any>>();
  private defaultTTL = 24 * 60 * 60 * 1000; // 24 hours

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

  /**
   * Get cache age in seconds
   */
  getAge(key: string): number | null {
    const entry = this.cache.get(key);

    if (!entry) {
      return null;
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return Math.floor((Date.now() - entry.timestamp) / 1000);
  }
}

// Global cache instance
const datasetCache = new DatasetCache();

/**
 * ================================================================
 * GET /api/data/datasets
 *
 * Backend-for-Frontend (BFF) proxy for retrieving dataset metadata.
 *
 * Security:
 * - NextAuth.js authentication required (401 if not authenticated)
 * - JWT forwarding to backend
 *
 * Features:
 * - Aggressive server-side caching (24hr TTL)
 * - Cache-first strategy for instant responses
 * - Backend proxy with comprehensive error handling
 * - Structured dataset metadata response
 * ================================================================
 */
export async function GET(request: NextRequest) {
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
          details: 'You must be authenticated to access dataset information',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 401 }
      );
    }

    console.log(`[DATASETS] Request from user ${session.user.id}`);

    // ================================================================
    // 2. CHECK CACHE (Cache-First Strategy)
    // ================================================================
    const cacheKey = 'datasets:list';
    const cachedData = datasetCache.get<Dataset[]>(cacheKey);

    if (cachedData) {
      const cacheAge = datasetCache.getAge(cacheKey);
      console.log(`[DATASETS] Cache HIT (age: ${cacheAge}s)`);

      return NextResponse.json(cachedData, {
        status: 200,
        headers: {
          'X-Cache-Status': 'HIT',
          'X-Cache-Age': cacheAge?.toString() || '0',
          'Cache-Control': 'private, max-age=86400', // 24 hours
          'X-Processing-Time': (Date.now() - startTime).toString(),
        },
      });
    }

    console.log(`[DATASETS] Cache MISS - fetching from backend`);

    // ================================================================
    // 3. BACKEND API REQUEST
    // ================================================================
    const backendApiUrl = process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL;

    if (!backendApiUrl) {
      console.error('[DATASETS] Backend API URL not configured');
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

    const backendUrl = `${backendApiUrl}/api/v1/data/datasets`;
    const token = session.user.id; // In production, use actual JWT token

    console.log(`[DATASETS] Fetching from backend: ${backendUrl}`);

    // Make request to Python backend
    const backendResponse = await fetch(backendUrl, {
      method: 'GET',
      headers: {
        'Authorization': token ? `Bearer ${token}` : '',
        'Content-Type': 'application/json',
        'User-Agent': 'NextJS-BFF/1.0',
      },
      // Timeout after 10 seconds
      signal: AbortSignal.timeout(10000),
    });

    // ================================================================
    // 4. HANDLE BACKEND ERRORS
    // ================================================================
    if (!backendResponse.ok) {
      const errorBody = await backendResponse.json().catch(() => ({
        error: 'Unknown backend error',
      }));

      console.error(
        `[DATASETS] Backend error: ${backendResponse.status} ${backendResponse.statusText}`,
        errorBody
      );

      if (backendResponse.status === 401 || backendResponse.status === 403) {
        return NextResponse.json(
          {
            error: 'Backend authentication failed',
            code: 'BACKEND_AUTH_FAILED',
            details: 'Failed to authenticate with backend data service',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 502 }
        );
      }

      if (backendResponse.status === 404) {
        return NextResponse.json(
          {
            error: 'Datasets endpoint not found',
            code: 'ENDPOINT_NOT_FOUND',
            details: 'The backend datasets endpoint is not available',
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
          details: errorBody.error || errorBody.detail || 'The backend data service encountered an error',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 502 }
      );
    }

    // ================================================================
    // 5. PARSE RESPONSE & CACHE
    // ================================================================
    let datasets: Dataset[];

    try {
      datasets = await backendResponse.json();

      // Validate response is an array
      if (!Array.isArray(datasets)) {
        console.error('[DATASETS] Backend returned non-array response:', datasets);

        // Try to extract datasets array from nested response
        if (datasets && typeof datasets === 'object' && 'datasets' in datasets) {
          datasets = (datasets as any).datasets;
        } else {
          throw new Error('Invalid response format');
        }
      }

      // Validate dataset objects
      datasets = datasets.map((dataset: any) => ({
        id: dataset.id || dataset.dataset_id || 'unknown',
        name: dataset.name || 'Unknown Dataset',
        description: dataset.description || '',
        size: dataset.size || dataset.num_samples || 'Unknown',
        features: Array.isArray(dataset.features) ? dataset.features : [],
        usage: dataset.usage || dataset.use_case || '',
        source: dataset.source,
        format: dataset.format || dataset.file_format,
        lastUpdated: dataset.lastUpdated || dataset.last_updated || dataset.updated_at,
      }));

    } catch (error) {
      console.error('[DATASETS] Failed to parse backend response:', error);
      return NextResponse.json(
        {
          error: 'Invalid backend response',
          code: 'INVALID_BACKEND_RESPONSE',
          details: 'Failed to parse datasets from backend response',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 502 }
      );
    }

    // Cache successful response (24 hours)
    datasetCache.set(cacheKey, datasets, 24 * 60 * 60 * 1000);

    console.log(`[DATASETS] Retrieved ${datasets.length} datasets (${Date.now() - startTime}ms)`);

    // ================================================================
    // 6. RETURN SUCCESS RESPONSE
    // ================================================================
    return NextResponse.json(datasets, {
      status: 200,
      headers: {
        'X-Cache-Status': 'MISS',
        'X-Cache-Age': '0',
        'Cache-Control': 'private, max-age=86400', // 24 hours
        'X-Processing-Time': (Date.now() - startTime).toString(),
        'X-Dataset-Count': datasets.length.toString(),
      },
    });

  } catch (error) {
    // ================================================================
    // 7. HANDLE UNEXPECTED ERRORS
    // ================================================================
    console.error('[DATASETS] Unexpected error:', error);

    // Handle timeout errors
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        {
          error: 'Backend request timeout',
          code: 'REQUEST_TIMEOUT',
          details: 'The data service did not respond in time (10s timeout)',
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
          details: 'Unable to connect to the data service',
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
        details: 'An unexpected error occurred while retrieving datasets',
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
