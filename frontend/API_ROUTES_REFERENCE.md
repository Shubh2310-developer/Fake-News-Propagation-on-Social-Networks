# API Routes Quick Reference

## Classifier Metrics Endpoint

### Overview

**Endpoint:** `GET /api/classifier/metrics`

**Purpose:** BFF proxy for fetching ML model performance metrics

**Authentication:** ✅ Required (NextAuth session)

**Caching:** ✅ Server-side (1 hour TTL)

### Quick Start

```typescript
// Client-side usage (React component)
async function getMetrics(modelType: string) {
  const response = await fetch(
    `/api/classifier/metrics?model_type=${modelType}`
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.details);
  }

  return response.json();
}

// Usage
const metrics = await getMetrics('ensemble');
console.log(metrics.accuracy); // 0.878
```

### Query Parameters

| Parameter | Values | Default |
|-----------|--------|---------|
| `model_type` | `ensemble`, `random_forest`, `bert`, `lstm`, etc. | `ensemble` |
| `include_history` | `true`, `false` | `true` |
| `include_confusion_matrix` | `true`, `false` | `true` |

### Examples

```bash
# Get ensemble metrics
curl http://localhost:3000/api/classifier/metrics?model_type=ensemble

# Get BERT metrics without history
curl http://localhost:3000/api/classifier/metrics?model_type=bert&include_history=false

# Get Random Forest with minimal data
curl http://localhost:3000/api/classifier/metrics?model_type=random_forest&include_history=false&include_confusion_matrix=false
```

### Response Structure

```typescript
interface MetricsResponse {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  confusion_matrix?: {
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
  };
  metadata: {
    source: 'cache' | 'backend';
    retrieved_at: string;
    cache_ttl_seconds: number;
  };
}
```

### Error Codes

| Code | Status | Meaning |
|------|--------|---------|
| `AUTHENTICATION_REQUIRED` | 401 | Not signed in |
| `INVALID_MODEL_TYPE` | 400 | Invalid model parameter |
| `MODEL_NOT_FOUND` | 404 | Model doesn't exist |
| `METHOD_NOT_ALLOWED` | 405 | Wrong HTTP method |
| `BACKEND_URL_MISSING` | 500 | Config error |
| `BACKEND_ERROR` | 502 | Backend failed |
| `CONNECTION_FAILED` | 503 | Can't reach backend |
| `REQUEST_TIMEOUT` | 504 | Backend timeout (30s) |

### Architecture

```
Client → Next.js BFF → Python Backend
           ↓
       [Cache: 1hr]
```

**Features:**
- ✅ NextAuth session validation
- ✅ JWT token forwarding
- ✅ Server-side caching (1hr TTL)
- ✅ Robust error handling
- ✅ 30-second timeout
- ✅ Cache hit/miss tracking

### Environment Setup

```env
# .env.local
INTERNAL_API_URL=http://localhost:8000
NEXTAUTH_SECRET=<your-secret>
```

### Cache Behavior

**Cache Key:** `metrics:${model_type}:${include_history}:${include_confusion_matrix}`

**TTL:** 1 hour

**Headers:**
- `X-Cache-Status: HIT` - Served from cache
- `X-Cache-Status: MISS` - Fetched from backend

### Testing

```typescript
// Test with useSession hook
import { useSession } from 'next-auth/react';

function MetricsComponent() {
  const { data: session } = useSession();
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    if (session) {
      fetch('/api/classifier/metrics?model_type=ensemble')
        .then(res => res.json())
        .then(data => setMetrics(data));
    }
  }, [session]);

  return <div>Accuracy: {metrics?.accuracy}</div>;
}
```

### Troubleshooting

**401 Unauthorized?**
→ User not signed in, redirect to `/auth/signin`

**502 Bad Gateway?**
→ Backend down, check `INTERNAL_API_URL`

**504 Timeout?**
→ Backend slow, check Python API performance

**Cache not working?**
→ Different query params create different cache keys

---

## Documentation

- **Full API Docs:** [API_BFF_PROXY.md](./docs/API_BFF_PROXY.md)
- **Auth Setup:** [AUTHENTICATION.md](./docs/AUTHENTICATION.md)
- **Environment:** [.env.example](./.env.example)

## Implementation Summary

✅ **Completed Features:**
- GET handler with authentication
- NextAuth session validation
- Query parameter validation
- Server-side caching (1hr TTL)
- Backend proxy with JWT forwarding
- Comprehensive error handling
- Method restriction (GET only)
- Cache hit/miss tracking
- Request timeout (30s)
- Error response standardization

🔧 **Configuration:**
- Environment variables documented
- Cache implementation complete
- Error codes standardized
- TypeScript types defined

📚 **Documentation:**
- API reference guide
- Quick reference card
- Architecture diagram
- Testing examples
