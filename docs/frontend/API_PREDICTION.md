# Prediction API Documentation

## Overview

The Prediction API provides a secure, high-performance endpoint for text classification. This BFF (Backend-for-Frontend) proxy handles authentication, input validation, intelligent caching, rate limiting, and secure communication with the Python ML backend.

## Endpoint

```
POST /api/classifier/predict
```

## Features

✅ **Security**
- NextAuth.js session authentication
- JWT token forwarding to backend
- Rate limiting (100 requests/hour per user)

✅ **Performance**
- SHA-256 based intelligent caching
- 24-hour cache TTL
- Cache hit rate tracking
- Sub-second response for cached predictions

✅ **Robustness**
- Comprehensive input validation
- 60-second timeout handling
- Graceful error responses
- Request/response logging

## Authentication

**Required**: Yes

Must have a valid NextAuth.js session. Unauthenticated requests return `401 Unauthorized`.

## Request Format

### Request Body

```typescript
{
  "text": string,              // Required: Text to classify (10-10,000 chars)
  "model_type"?: string,       // Optional: Model to use (default: "ensemble")
  "explain_prediction"?: boolean, // Optional: Include explanation (default: false)
  "include_features"?: boolean    // Optional: Include extracted features (default: true)
}
```

### Request Examples

#### Basic Prediction

```bash
curl -X POST http://localhost:3000/api/classifier/predict \
  -H 'Content-Type: application/json' \
  -H 'Cookie: next-auth.session-token=<token>' \
  -d '{
    "text": "Breaking news: Scientists discover shocking truth about climate change!"
  }'
```

#### Prediction with Explanation

```bash
curl -X POST http://localhost:3000/api/classifier/predict \
  -H 'Content-Type: application/json' \
  -H 'Cookie: next-auth.session-token=<token>' \
  -d '{
    "text": "New study reveals surprising health benefits",
    "model_type": "bert",
    "explain_prediction": true
  }'
```

#### Client-Side Usage (React)

```typescript
import { useSession } from 'next-auth/react';

async function predictText(text: string, modelType = 'ensemble') {
  const response = await fetch('/api/classifier/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      model_type: modelType,
      explain_prediction: true,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.details || error.error);
  }

  return response.json();
}

// Usage in component
const { data: session } = useSession();

if (session) {
  const result = await predictText(
    "This is a news article to classify"
  );
  console.log(result.prediction); // 'real' or 'fake'
  console.log(result.confidence); // 0.85
}
```

## Response Format

### Success Response (200 OK)

```json
{
  "prediction": "fake",
  "confidence": 0.87,
  "probabilities": {
    "real": 0.13,
    "fake": 0.87
  },
  "model_info": {
    "model_type": "ensemble",
    "version": "v2.1.0"
  },
  "features": {
    "word_count": 156,
    "sentiment_score": 0.65,
    "readability_score": 72,
    "clickbait_indicators": ["shocking", "you won't believe"],
    "entity_count": 12,
    "url_count": 2
  },
  "explanation": {
    "top_features": [
      {
        "name": "sentiment_extremity",
        "importance": 0.32,
        "contribution": "positive"
      },
      {
        "name": "clickbait_score",
        "importance": 0.28,
        "contribution": "positive"
      },
      {
        "name": "source_credibility",
        "importance": 0.15,
        "contribution": "negative"
      }
    ],
    "confidence_factors": [
      "High emotional language detected",
      "Multiple clickbait patterns found",
      "Low source credibility score"
    ]
  },
  "metadata": {
    "processing_time_ms": 234,
    "timestamp": "2024-10-01T10:30:00Z",
    "cached": false
  }
}
```

### Response Headers

```
X-Cache-Status: HIT | MISS
X-RateLimit-Remaining: 95
Cache-Control: private, max-age=86400
Content-Type: application/json
```

## Validation Rules

### Text Field

| Rule | Value | Error Code |
|------|-------|------------|
| Required | Yes | `MISSING_TEXT` |
| Type | String | `INVALID_TEXT_TYPE` |
| Min Length | 10 characters | `TEXT_TOO_SHORT` |
| Max Length | 10,000 characters | `TEXT_TOO_LONG` |
| Cannot be | Empty or whitespace only | `EMPTY_TEXT` |

### Model Type

**Valid Values:**
- `ensemble` (default)
- `random_forest`
- `gradient_boosting`
- `svm`
- `naive_bayes`
- `logistic_regression`
- `bert`
- `lstm`

Invalid values return `INVALID_MODEL_TYPE` (400).

## Error Responses

### Standard Error Format

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": "Detailed description",
  "timestamp": "2024-10-01T10:30:00Z"
}
```

### Error Codes

#### 400 Bad Request

**INVALID_JSON**
```json
{
  "error": "Invalid JSON",
  "code": "INVALID_JSON",
  "details": "Request body must be valid JSON"
}
```

**MISSING_TEXT**
```json
{
  "error": "Missing required field",
  "code": "MISSING_TEXT",
  "details": "The \"text\" field is required"
}
```

**TEXT_TOO_SHORT**
```json
{
  "error": "Text too short",
  "code": "TEXT_TOO_SHORT",
  "details": "Text must be at least 10 characters long"
}
```

**TEXT_TOO_LONG**
```json
{
  "error": "Text too long",
  "code": "TEXT_TOO_LONG",
  "details": "Text must be 10,000 characters or less"
}
```

**INVALID_MODEL_TYPE**
```json
{
  "error": "Invalid model type",
  "code": "INVALID_MODEL_TYPE",
  "details": "Model type must be one of: ensemble, random_forest, ..."
}
```

#### 401 Unauthorized

**AUTHENTICATION_REQUIRED**
```json
{
  "error": "Unauthorized",
  "code": "AUTHENTICATION_REQUIRED",
  "details": "You must be authenticated to use the prediction service"
}
```

#### 405 Method Not Allowed

**METHOD_NOT_ALLOWED**
```json
{
  "error": "Method not allowed",
  "code": "METHOD_NOT_ALLOWED",
  "details": "This endpoint only supports POST requests"
}
```

**Headers:**
```
Allow: POST
```

#### 429 Too Many Requests

**RATE_LIMIT_EXCEEDED**
```json
{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "details": "Maximum requests exceeded. Limit resets at 2024-10-01T11:30:00Z"
}
```

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1727784600000
Retry-After: 3600
```

#### 500 Internal Server Error

**BACKEND_URL_MISSING**
```json
{
  "error": "Backend configuration error",
  "code": "BACKEND_URL_MISSING",
  "details": "Backend API URL is not configured"
}
```

#### 502 Bad Gateway

**BACKEND_ERROR**
```json
{
  "error": "Backend service error",
  "code": "BACKEND_ERROR",
  "details": "The backend service encountered an error"
}
```

**BACKEND_AUTH_FAILED**
```json
{
  "error": "Backend authentication failed",
  "code": "BACKEND_AUTH_FAILED",
  "details": "Failed to authenticate with backend service"
}
```

#### 503 Service Unavailable

**CONNECTION_FAILED**
```json
{
  "error": "Backend connection failed",
  "code": "CONNECTION_FAILED",
  "details": "Unable to connect to the prediction service"
}
```

#### 504 Gateway Timeout

**REQUEST_TIMEOUT**
```json
{
  "error": "Backend request timeout",
  "code": "REQUEST_TIMEOUT",
  "details": "The prediction service did not respond in time (60s timeout)"
}
```

## Caching Strategy

### Cache Key Generation

Cache keys are generated using SHA-256 hash of:
```
hash(text + model_type + explain_prediction)
```

**Example:**
```typescript
// These create the same cache key (same result)
{ text: "Hello", model_type: "ensemble", explain_prediction: false }
{ text: "Hello", model_type: "ensemble", explain_prediction: false }

// These create different cache keys
{ text: "Hello", model_type: "bert", explain_prediction: false }
{ text: "Hello world", model_type: "ensemble", explain_prediction: false }
```

### Cache Behavior

**TTL:** 24 hours (86,400 seconds)

**Cache Hit:**
- Returns cached prediction
- `X-Cache-Status: HIT` header
- `metadata.cached: true`
- Near-instant response (~5-10ms)

**Cache Miss:**
- Fetches from Python backend
- Caches successful response
- `X-Cache-Status: MISS` header
- `metadata.cached: false`
- Response time depends on model (~200-500ms)

### Cache Benefits

- **Performance**: 98% faster for cached predictions
- **Cost Reduction**: Reduces backend load by ~90%
- **User Experience**: Instant predictions for repeated queries
- **Reliability**: Continues serving cached data if backend is slow

## Rate Limiting

### Limits

- **Requests per Hour**: 100
- **Window**: Rolling 1-hour window per user
- **Identification**: By NextAuth user ID

### Rate Limit Headers

All responses include:
```
X-RateLimit-Remaining: <number>
```

When rate limited (429):
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: <timestamp>
Retry-After: <seconds>
```

### Rate Limit Behavior

1. **First Request**: Counter starts at 1, 99 remaining
2. **Subsequent Requests**: Counter increments
3. **Limit Reached**: Returns 429, includes reset time
4. **After Reset**: Counter resets to 0, new window begins

### Bypassing Rate Limits

For testing or special users:
```typescript
// In production, implement role-based limits
if (session.user.role === 'admin') {
  // Skip rate limiting
}
```

## Performance Metrics

### Without Caching
- Average: ~300-500ms
- Backend calls: 100%
- Network: ~150ms
- ML inference: ~200ms

### With Caching
- Cache hit: ~5-10ms (98% faster)
- Cache hit rate: ~85% (typical)
- Backend calls: ~15%
- Memory: ~5KB per cached prediction

## Security

### Request Flow

```
1. Client Request
   ↓
2. Session Validation (401 if invalid)
   ↓
3. Rate Limit Check (429 if exceeded)
   ↓
4. Input Validation (400 if invalid)
   ↓
5. Cache Check
   ├─ HIT → Return cached
   └─ MISS ↓
6. Backend Request (with JWT)
   ↓
7. Cache & Return
```

### Security Features

- ✅ NextAuth session validation
- ✅ JWT token forwarding
- ✅ Per-user rate limiting
- ✅ Input sanitization
- ✅ Request timeout protection
- ✅ Error message sanitization
- ✅ No sensitive data in logs

## Testing

### Unit Test Example

```typescript
import { POST } from '@/app/api/classifier/predict/route';
import { NextRequest } from 'next/server';

describe('POST /api/classifier/predict', () => {
  it('requires authentication', async () => {
    const request = new NextRequest('http://localhost:3000/api/classifier/predict', {
      method: 'POST',
      body: JSON.stringify({ text: 'Test' }),
    });

    const response = await POST(request);
    expect(response.status).toBe(401);
  });

  it('validates text length', async () => {
    // Mock session...
    const request = new NextRequest('http://localhost:3000/api/classifier/predict', {
      method: 'POST',
      body: JSON.stringify({ text: 'Too short' }),
    });

    const response = await POST(request);
    expect(response.status).toBe(400);
  });
});
```

### Integration Test

```bash
# Test basic prediction
curl -X POST http://localhost:3000/api/classifier/predict \
  -H 'Content-Type: application/json' \
  -H 'Cookie: next-auth.session-token=<token>' \
  -d '{"text":"This is a test article about climate change and its effects"}' \
  -v

# Check cache behavior (second request should be cached)
curl -X POST http://localhost:3000/api/classifier/predict \
  -H 'Content-Type: application/json' \
  -H 'Cookie: next-auth.session-token=<token>' \
  -d '{"text":"This is a test article about climate change and its effects"}' \
  -v | grep 'X-Cache-Status'

# Test rate limiting (make 101 requests)
for i in {1..101}; do
  curl -X POST http://localhost:3000/api/classifier/predict \
    -H 'Content-Type: application/json' \
    -H 'Cookie: next-auth.session-token=<token>' \
    -d "{\"text\":\"Test message $i\"}" \
    -s -o /dev/null -w "%{http_code}\n"
done
```

## Best Practices

### For Frontend Developers

1. **Always handle rate limits**
   ```typescript
   if (response.status === 429) {
     const retryAfter = response.headers.get('Retry-After');
     // Wait before retrying
   }
   ```

2. **Check cache status for UX**
   ```typescript
   const cacheStatus = response.headers.get('X-Cache-Status');
   if (cacheStatus === 'HIT') {
     // Show "instant results" badge
   }
   ```

3. **Validate input before sending**
   ```typescript
   if (text.trim().length < 10 || text.length > 10000) {
     // Show error before API call
   }
   ```

### For Backend Integration

1. **Maintain consistent response format**
2. **Return proper HTTP status codes**
3. **Include detailed error messages**
4. **Support JWT authentication**
5. **Handle timeouts gracefully**

## Troubleshooting

### Issue: 401 Unauthorized

**Cause**: No valid session

**Solution**: Ensure user is signed in
```typescript
const { data: session } = useSession();
if (!session) {
  router.push('/auth/signin');
}
```

### Issue: 429 Rate Limit

**Cause**: Too many requests

**Solution**: Implement retry logic
```typescript
const retryAfter = parseInt(response.headers.get('Retry-After') || '60');
await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
// Retry request
```

### Issue: Cache not working

**Cause**: Different request parameters

**Solution**: Use consistent parameters
```typescript
// Normalize text before sending
const normalizedText = text.trim();
```

### Issue: 504 Timeout

**Cause**: Backend processing takes >60s

**Solution**:
- For long texts, split into chunks
- Use async processing with webhooks
- Increase timeout (not recommended)

## Monitoring

### Key Metrics

1. **Cache Hit Rate**: Target >80%
2. **Response Time**: <50ms (cached), <500ms (uncached)
3. **Error Rate**: <1%
4. **Rate Limit Hits**: <5% of requests

### Logging

```
[PREDICT] Cache HIT for key: predict:abc123...
[PREDICT] Cache MISS for key: predict:def456...
[PREDICT] Forwarding to backend: http://backend:8000/api/v1/classifier/predict
[PREDICT] Prediction completed: fake (0.87)
```

---

**Last Updated**: October 1, 2024
**API Version**: 1.0
**Maintainer**: Frontend Team
