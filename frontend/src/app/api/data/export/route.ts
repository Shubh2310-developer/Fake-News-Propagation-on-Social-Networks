// frontend/src/app/api/data/export/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';

/**
 * ================================================================
 * TYPE DEFINITIONS
 * ================================================================
 */

interface ErrorResponse {
  error: string;
  code: string;
  details?: string;
  timestamp: string;
}

/**
 * Valid exportable data sources
 */
const VALID_SOURCES = [
  'simulation_results',
  'network_data',
  'processed_features',
  'training_data',
  'model_predictions',
  'game_theory_metrics',
  'propagation_patterns',
] as const;

type ExportSource = typeof VALID_SOURCES[number];

/**
 * Role-based permissions for data export
 */
const EXPORT_PERMISSIONS: Record<ExportSource, string[]> = {
  simulation_results: ['user', 'researcher', 'admin'],
  network_data: ['researcher', 'admin'],
  processed_features: ['researcher', 'admin'],
  training_data: ['researcher', 'admin'],
  model_predictions: ['user', 'researcher', 'admin'],
  game_theory_metrics: ['researcher', 'admin'],
  propagation_patterns: ['researcher', 'admin'],
};

/**
 * MIME type mapping for export formats
 */
const CONTENT_TYPES: Record<string, string> = {
  csv: 'text/csv',
  json: 'application/json',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  txt: 'text/plain',
};

/**
 * ================================================================
 * VALIDATION HELPERS
 * ================================================================
 */

/**
 * Validates export source parameter
 */
function validateSource(source: string | null): { valid: boolean; error?: string } {
  if (!source) {
    return { valid: false, error: 'Missing required parameter: source' };
  }

  if (!VALID_SOURCES.includes(source as ExportSource)) {
    return {
      valid: false,
      error: `Invalid source. Must be one of: ${VALID_SOURCES.join(', ')}`,
    };
  }

  return { valid: true };
}

/**
 * Validates that required ID is present for sources that need it
 */
function validateId(source: ExportSource, id: string | null): { valid: boolean; error?: string } {
  const sourcesRequiringId: ExportSource[] = ['simulation_results', 'model_predictions'];

  if (sourcesRequiringId.includes(source) && !id) {
    return {
      valid: false,
      error: `Parameter 'id' is required when exporting ${source}`,
    };
  }

  return { valid: true };
}

/**
 * Checks if user has permission to export the specified source
 */
function checkExportPermission(
  source: ExportSource,
  userRole: string
): { allowed: boolean; error?: string } {
  const allowedRoles = EXPORT_PERMISSIONS[source];

  if (!allowedRoles.includes(userRole)) {
    return {
      allowed: false,
      error: `Your role '${userRole}' does not have permission to export ${source}. Required roles: ${allowedRoles.join(', ')}`,
    };
  }

  return { allowed: true };
}

/**
 * Generates dynamic filename for export
 */
function generateFilename(
  source: ExportSource,
  format: string,
  id?: string | null,
  userId?: string
): string {
  const timestamp = new Date().toISOString().split('T')[0].replace(/-/g, '');

  if (id) {
    return `${source}_${id}_${timestamp}.${format}`;
  }

  if (userId) {
    return `${source}_user${userId.substring(0, 8)}_${timestamp}.${format}`;
  }

  return `${source}_${timestamp}.${format}`;
}

/**
 * ================================================================
 * GET /api/data/export
 *
 * Streams data exports directly from the Python backend to the client.
 *
 * Query Parameters:
 * - source: Required. The data source to export (e.g., 'simulation_results')
 * - id: Required for certain sources (e.g., simulation ID)
 * - format: Optional. Export format (default: 'csv')
 * - filters: Optional. JSON-encoded filter criteria
 *
 * Security:
 * - NextAuth.js authentication required (401 if not authenticated)
 * - Role-based authorization per data source (403 if unauthorized)
 * - User can only export their own data unless they have researcher/admin role
 *
 * Features:
 * - Streaming response (no server-side buffering)
 * - Dynamic filename generation
 * - Proper Content-Type and Content-Disposition headers
 * - Comprehensive validation and error handling
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
          details: 'You must be authenticated to export data',
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 401 }
      );
    }

    console.log(`[EXPORT] Request from user ${session.user.id} (${session.user.role})`);

    // ================================================================
    // 2. EXTRACT AND VALIDATE QUERY PARAMETERS
    // ================================================================
    const { searchParams } = new URL(request.url);

    const source = searchParams.get('source');
    const id = searchParams.get('id');
    const format = searchParams.get('format') || 'csv';
    const filters = searchParams.get('filters');

    // Validate source
    const sourceValidation = validateSource(source);
    if (!sourceValidation.valid) {
      return NextResponse.json(
        {
          error: 'Invalid source parameter',
          code: 'VALIDATION_ERROR',
          details: sourceValidation.error,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    const validSource = source as ExportSource;

    // Validate ID if required
    const idValidation = validateId(validSource, id);
    if (!idValidation.valid) {
      return NextResponse.json(
        {
          error: 'Missing required parameter',
          code: 'VALIDATION_ERROR',
          details: idValidation.error,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate format
    const validFormats = ['csv', 'json', 'xlsx', 'txt'];
    if (!validFormats.includes(format)) {
      return NextResponse.json(
        {
          error: 'Invalid format parameter',
          code: 'VALIDATION_ERROR',
          details: `Format must be one of: ${validFormats.join(', ')}`,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Validate filters if provided
    let parsedFilters: any = null;
    if (filters) {
      try {
        parsedFilters = JSON.parse(filters);
      } catch (error) {
        return NextResponse.json(
          {
            error: 'Invalid filters parameter',
            code: 'VALIDATION_ERROR',
            details: 'Filters must be valid JSON',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 400 }
        );
      }
    }

    console.log(`[EXPORT] Source: ${validSource}, ID: ${id || 'none'}, Format: ${format}`);

    // ================================================================
    // 3. AUTHORIZATION (Role-Based Access Control)
    // ================================================================
    const userRole = session.user.role;
    const permissionCheck = checkExportPermission(validSource, userRole);

    if (!permissionCheck.allowed) {
      console.warn(
        `[EXPORT] Authorization denied for user ${session.user.id} (${userRole}) to export ${validSource}`
      );

      return NextResponse.json(
        {
          error: 'Insufficient permissions',
          code: 'FORBIDDEN',
          details: permissionCheck.error,
          timestamp: new Date().toISOString(),
        } as ErrorResponse,
        { status: 403 }
      );
    }

    console.log(`[EXPORT] Authorization granted for ${validSource}`);

    // ================================================================
    // 4. OWNERSHIP VALIDATION (for user role)
    // ================================================================
    // Users can only export their own data unless they're researcher/admin
    if (userRole === 'user' && id) {
      // In production, verify that the resource (simulation, prediction, etc.)
      // belongs to the authenticated user. For now, we'll add the user_id to filters
      if (!parsedFilters) {
        parsedFilters = {};
      }
      parsedFilters.user_id = session.user.id;
    }

    // ================================================================
    // 5. BACKEND API REQUEST (Streaming)
    // ================================================================
    const backendApiUrl = process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL;

    if (!backendApiUrl) {
      console.error('[EXPORT] Backend API URL not configured');
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

    // Build backend URL with query parameters
    const backendUrl = new URL(`${backendApiUrl}/api/v1/data/export`);
    backendUrl.searchParams.set('source', validSource);
    if (id) backendUrl.searchParams.set('id', id);
    backendUrl.searchParams.set('format', format);
    if (parsedFilters) backendUrl.searchParams.set('filters', JSON.stringify(parsedFilters));

    const token = session.user.id; // In production, use actual JWT token

    console.log(`[EXPORT] Streaming from backend: ${backendUrl.toString()}`);

    // Make streaming request to Python backend
    const backendResponse = await fetch(backendUrl.toString(), {
      method: 'GET',
      headers: {
        'Authorization': token ? `Bearer ${token}` : '',
        'User-Agent': 'NextJS-BFF/1.0',
        'X-User-ID': session.user.id,
        'X-User-Role': session.user.role,
      },
      // No timeout for streaming responses (can be large files)
    });

    // ================================================================
    // 6. HANDLE BACKEND ERRORS (Before Streaming)
    // ================================================================
    if (!backendResponse.ok) {
      // For error responses, read the body as JSON
      const errorBody = await backendResponse.json().catch(() => ({
        error: 'Unknown backend error',
      }));

      console.error(
        `[EXPORT] Backend error: ${backendResponse.status} ${backendResponse.statusText}`,
        errorBody
      );

      if (backendResponse.status === 400) {
        return NextResponse.json(
          {
            error: 'Backend validation error',
            code: 'BACKEND_VALIDATION_ERROR',
            details: errorBody.error || errorBody.detail || 'The backend service rejected the export request',
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
            details: 'Failed to authenticate with backend data service',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 502 }
        );
      }

      if (backendResponse.status === 404) {
        return NextResponse.json(
          {
            error: 'Data not found',
            code: 'DATA_NOT_FOUND',
            details: errorBody.error || 'The requested data does not exist or has been deleted',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 404 }
        );
      }

      if (backendResponse.status === 413) {
        return NextResponse.json(
          {
            error: 'Export too large',
            code: 'EXPORT_TOO_LARGE',
            details: 'The requested export exceeds size limits. Try filtering or exporting in smaller chunks',
            timestamp: new Date().toISOString(),
          } as ErrorResponse,
          { status: 413 }
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
    // 7. STREAM RESPONSE TO CLIENT
    // ================================================================

    // Generate dynamic filename
    const filename = generateFilename(validSource, format, id, session.user.id);

    // Get content type from backend or use default
    const contentType =
      backendResponse.headers.get('content-type') || CONTENT_TYPES[format] || 'application/octet-stream';

    // Get content length if available (for progress tracking)
    const contentLength = backendResponse.headers.get('content-length');

    console.log(
      `[EXPORT] Streaming ${filename} (${contentLength ? contentLength + ' bytes' : 'unknown size'})`
    );

    // Create streaming response
    const headers = new Headers({
      'Content-Type': contentType,
      'Content-Disposition': `attachment; filename="${filename}"`,
      'X-Export-Source': validSource,
      'X-Export-Format': format,
      'X-Processing-Time': (Date.now() - startTime).toString(),
    });

    // Add content length if available
    if (contentLength) {
      headers.set('Content-Length', contentLength);
    }

    // Stream the response body directly from backend to client
    // This is the key feature - no buffering in server memory!
    return new Response(backendResponse.body, {
      status: 200,
      headers,
    });

  } catch (error) {
    // ================================================================
    // 8. HANDLE UNEXPECTED ERRORS
    // ================================================================
    console.error('[EXPORT] Unexpected error:', error);

    // Handle timeout errors (if timeout was set)
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        {
          error: 'Backend request timeout',
          code: 'REQUEST_TIMEOUT',
          details: 'The export service did not respond in time',
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
          details: 'Unable to connect to the export service',
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
        details: 'An unexpected error occurred during data export',
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
      details: 'This endpoint only supports GET requests for data export',
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
      details: 'This endpoint only supports GET requests for data export',
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
      details: 'This endpoint only supports GET requests for data export',
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
      details: 'This endpoint only supports GET requests for data export',
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
