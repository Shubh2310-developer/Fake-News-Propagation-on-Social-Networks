# API Integration Fix Summary

## Issues Found & Fixed

### ❌ Problem: 404 Error on Simulation Endpoint

**Error Message:**
```
POST http://localhost:8000/simulation/run 404 (Not Found)
```

**Root Cause:**
The frontend was calling `/simulation/run` but the backend API is at `/api/v1/simulation/run`

### ✅ Solution Applied

**Fixed Files:**

1. **`frontend/src/lib/api.ts`**
   - Changed: `baseURL: 'http://localhost:8000/api/v1'`
   - To: `baseURL: 'http://localhost:8000'`
   - This prevents path duplication

2. **`frontend/src/lib/constants.ts`**
   - Updated all API routes to include `/api/v1/` prefix
   - Example: `/simulation/run` → `/api/v1/simulation/run`

## Complete API Route Mapping

### Before (Incorrect)
```javascript
SIMULATION_RUN: '/simulation/run'
// With baseURL: 'http://localhost:8000/api/v1'
// Results in: http://localhost:8000/api/v1/simulation/run ❌ (duplicate)
```

### After (Correct)
```javascript
baseURL: 'http://localhost:8000'
SIMULATION_RUN: '/api/v1/simulation/run'
// Results in: http://localhost:8000/api/v1/simulation/run ✅
```

## Updated API Endpoints

All endpoints now correctly prefixed with `/api/v1/`:

```javascript
{
  CLASSIFIER_PREDICT: '/api/v1/classifier/predict',
  SIMULATION_RUN: '/api/v1/simulation/run',
  SIMULATION_STATUS: (id) => `/api/v1/simulation/status/${id}`,
  SIMULATION_RESULTS: (id) => `/api/v1/simulation/results/${id}`,
  NETWORK_METRICS: '/api/v1/analysis/network/metrics',
  EQUILIBRIUM_CALCULATE: '/api/v1/equilibrium/calculate',
}
```

## Backend API Structure

The backend routes are organized as:

```
/api/v1/
├── classifier/
│   ├── predict
│   └── ...
├── simulation/
│   ├── run            ← Main endpoint
│   ├── status/{id}
│   ├── results/{id}
│   └── list
├── equilibrium/
│   └── calculate
└── analysis/
    └── network/metrics
```

## Testing the Fix

### 1. Start Both Services
```bash
./start.sh
```

### 2. Test Backend Directly
```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

### 3. Test Simulation Endpoint
```bash
curl -X POST http://localhost:8000/api/v1/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "network_config": {
      "num_nodes": 100,
      "network_type": "watts_strogatz",
      "rewiring_probability": 0.1,
      "attachment_preference": 5
    },
    "game_config": {
      "num_rounds": 50,
      "num_spreaders": 10,
      "num_fact_checkers": 5,
      "num_platforms": 1,
      "learning_rate": 0.1,
      "exploration_rate": 0.1
    },
    "save_detailed_history": true
  }'
```

Expected Response:
```json
{
  "simulation_id": "uuid-here",
  "status": "started",
  "message": "Simulation started successfully",
  "created_at": "2025-10-02T..."
}
```

### 4. Test Frontend Integration

1. Open http://localhost:3000/simulation
2. Configure parameters (or use defaults)
3. Click "Start Simulation"
4. Should see status change from "starting" → "running" → "completed"
5. Results should display automatically

## Environment Configuration

### Backend `.env`
```env
# No changes needed - already correct
DATABASE_URL=postgresql://user:pass@localhost:5432/fake_news_db
REDIS_URL=redis://localhost:6379/0
```

### Frontend `.env.local`
```env
# Set base URL without /api/v1
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_VERSION=v1
```

## Verification Checklist

- [x] ✅ API base URL corrected in `lib/api.ts`
- [x] ✅ All routes updated with `/api/v1/` prefix in `lib/constants.ts`
- [x] ✅ Backend routes verified at `/api/v1/simulation/*`
- [x] ✅ Store updated to use correct endpoints
- [ ] ⏳ Test simulation run from frontend (requires restart)
- [ ] ⏳ Verify status polling works
- [ ] ⏳ Verify results retrieval works

## Next Steps

1. **Restart Frontend** (to pick up changes):
   ```bash
   # Stop current servers (Ctrl+C)
   # Then restart:
   ./start.sh
   ```

2. **Test Simulation Flow**:
   - Go to http://localhost:3000/simulation
   - Click "Start Simulation"
   - Watch for successful API calls in browser console
   - Verify results display

3. **Monitor for Errors**:
   ```bash
   # Watch logs
   tail -f logs/backend.log logs/frontend.log
   ```

## Common Issues & Solutions

### Issue: Still getting 404
**Solution:** Hard refresh the browser (`Ctrl+Shift+R`) or clear Next.js cache:
```bash
cd frontend
rm -rf .next
npm run dev
```

### Issue: CORS errors
**Solution:** Check backend CORS settings in `backend/.env`:
```env
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

### Issue: Timeout errors
**Solution:** Increase timeout in `frontend/src/lib/api.ts`:
```javascript
timeout: 30000, // 30 seconds
```

## Files Changed

1. ✅ `frontend/src/lib/api.ts` - Fixed baseURL
2. ✅ `frontend/src/lib/constants.ts` - Updated all API routes
3. ℹ️ No backend changes needed (already correct)

## Success Indicators

When working correctly, you should see:

**Browser Console:**
```
POST http://localhost:8000/api/v1/simulation/run 200 OK
GET http://localhost:8000/api/v1/simulation/status/{id} 200 OK
GET http://localhost:8000/api/v1/simulation/results/{id} 200 OK
```

**Backend Logs:**
```
INFO: Simulation {id} started
INFO: Running simulation with 100 nodes, 50 rounds
INFO: Simulation {id} completed successfully
```

**Frontend UI:**
- Status indicator shows "Running"
- Progress updates every 2 seconds
- Results display when complete
- Network graph updates
- Payoff charts render

---

**Status:** ✅ Fix Applied - Ready for Testing

**Last Updated:** 2025-10-02
