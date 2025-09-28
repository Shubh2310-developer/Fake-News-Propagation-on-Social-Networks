// frontend/src/types/api.ts

/**
 * Generic types related to API calls and HTTP requests.
 * These types provide a consistent way to handle loading states,
 * errors, and data across all API interactions in the application.
 */

/**
 * Represents the generic state of an API request.
 * This interface is used by custom hooks like useApi to provide
 * consistent state management for all API calls.
 *
 * @template T The type of the data returned on success.
 */
export interface ApiState<T> {
  /** The data returned from the API on success, null otherwise */
  data: T | null;
  /** Whether a request is currently in progress */
  loading: boolean;
  /** Error message if the request failed, null otherwise */
  error: string | null;
}

/**
 * Extended API state that includes additional metadata
 */
export interface ExtendedApiState<T> extends ApiState<T> {
  /** Whether the data has been fetched at least once */
  initialized: boolean;
  /** Timestamp of the last successful fetch */
  lastFetch: Date | null;
  /** Whether the data is stale and should be refetched */
  isStale: boolean;
  /** Number of retry attempts made */
  retryCount: number;
}

/**
 * HTTP methods supported by the API
 */
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

/**
 * HTTP status codes
 */
export type HttpStatusCode = 200 | 201 | 204 | 400 | 401 | 403 | 404 | 409 | 422 | 429 | 500 | 502 | 503;

/**
 * Generic request configuration
 */
export interface RequestConfig {
  method?: HttpMethod;
  headers?: Record<string, string>;
  params?: Record<string, string | number | boolean>;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
}

/**
 * API endpoint information
 */
export interface ApiEndpoint {
  path: string;
  method: HttpMethod;
  description?: string;
  requiresAuth?: boolean;
  rateLimit?: {
    requests: number;
    window: number; // in seconds
  };
}

/**
 * API error response structure
 */
export interface ApiError {
  /** Error code */
  code: string;
  /** Human-readable error message */
  message: string;
  /** Additional error details */
  details?: Record<string, unknown>;
  /** Field-specific validation errors */
  fieldErrors?: Record<string, string[]>;
  /** HTTP status code */
  status: HttpStatusCode;
  /** Timestamp when the error occurred */
  timestamp: string;
  /** Request ID for debugging */
  requestId?: string;
}

/**
 * Validation error for form fields
 */
export interface ValidationError {
  field: string;
  message: string;
  code: string;
  value?: unknown;
}

/**
 * Batch operation request
 */
export interface BatchRequest<T> {
  operations: Array<{
    id: string;
    method: HttpMethod;
    path: string;
    data?: T;
  }>;
}

/**
 * Batch operation response
 */
export interface BatchResponse<T> {
  results: Array<{
    id: string;
    status: HttpStatusCode;
    data?: T;
    error?: ApiError;
  }>;
  summary: {
    total: number;
    successful: number;
    failed: number;
  };
}

/**
 * Cache configuration for API requests
 */
export interface CacheConfig {
  /** Cache key */
  key: string;
  /** Time to live in seconds */
  ttl: number;
  /** Whether to use stale data while revalidating */
  staleWhileRevalidate: boolean;
  /** Tags for cache invalidation */
  tags?: string[];
}

/**
 * Retry configuration for failed requests
 */
export interface RetryConfig {
  /** Maximum number of retry attempts */
  maxAttempts: number;
  /** Delay between retries in milliseconds */
  delay: number;
  /** Whether to use exponential backoff */
  exponentialBackoff: boolean;
  /** Maximum delay between retries */
  maxDelay: number;
  /** HTTP status codes that should trigger a retry */
  retryOn: HttpStatusCode[];
}

/**
 * Request/response interceptor function types
 */
export type RequestInterceptor = (config: RequestConfig) => RequestConfig | Promise<RequestConfig>;
export type ResponseInterceptor<T> = (response: T) => T | Promise<T>;
export type ErrorInterceptor = (error: ApiError) => ApiError | Promise<ApiError>;

/**
 * API client configuration
 */
export interface ApiClientConfig {
  /** Base URL for all requests */
  baseURL: string;
  /** Default timeout for requests */
  timeout: number;
  /** Default headers to include with all requests */
  defaultHeaders: Record<string, string>;
  /** Request interceptors */
  requestInterceptors: RequestInterceptor[];
  /** Response interceptors */
  responseInterceptors: ResponseInterceptor<unknown>[];
  /** Error interceptors */
  errorInterceptors: ErrorInterceptor[];
  /** Default retry configuration */
  retryConfig: RetryConfig;
  /** Default cache configuration */
  cacheConfig: Partial<CacheConfig>;
}

/**
 * WebSocket connection state
 */
export interface WebSocketState {
  /** Whether the WebSocket is connected */
  connected: boolean;
  /** Whether the connection is in progress */
  connecting: boolean;
  /** Connection error if any */
  error: string | null;
  /** Number of reconnection attempts */
  reconnectAttempts: number;
  /** Last message received timestamp */
  lastActivity: Date | null;
}

/**
 * WebSocket message structure
 */
export interface WebSocketMessage<T = unknown> {
  /** Message type */
  type: string;
  /** Message payload */
  data: T;
  /** Message ID for tracking */
  id?: string;
  /** Timestamp when message was sent */
  timestamp: string;
}

/**
 * Server-sent events state
 */
export interface SSEState<T> {
  /** Whether the connection is active */
  connected: boolean;
  /** Latest data received */
  data: T | null;
  /** Connection error if any */
  error: string | null;
  /** Number of reconnection attempts */
  reconnectAttempts: number;
}

/**
 * Upload progress information
 */
export interface UploadProgress {
  /** Bytes uploaded */
  loaded: number;
  /** Total bytes to upload */
  total: number;
  /** Upload percentage (0-100) */
  percentage: number;
  /** Upload speed in bytes per second */
  speed: number;
  /** Estimated time remaining in seconds */
  remainingTime: number;
}

/**
 * File upload state
 */
export interface UploadState {
  /** Upload progress */
  progress: UploadProgress | null;
  /** Whether upload is in progress */
  uploading: boolean;
  /** Upload error if any */
  error: string | null;
  /** Whether upload was completed successfully */
  completed: boolean;
  /** Uploaded file information */
  fileInfo: {
    name: string;
    size: number;
    type: string;
    url?: string;
  } | null;
}