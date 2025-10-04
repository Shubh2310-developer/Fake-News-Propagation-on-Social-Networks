# 🔄 Restart Instructions

## What Was Fixed

### ✅ Database Connection
- **Issue**: Password authentication failed
- **Fixed**: Updated `.env` with correct password `Meghal0987@23`
- **Created**: Database `gtds_db`

### ✅ API Routes
- **Issue**: 404 errors on `/simulation/run`
- **Fixed**: Corrected API paths to `/api/v1/simulation/run`

## 🚀 How to Restart

### Option 1: Quick Restart (Recommended)

```bash
# Stop current servers (press Ctrl+C in the terminal where they're running)
# OR
pkill -f "uvicorn app.main:app" && pkill -f "next dev"

# Then restart
./start.sh
```

### Option 2: Full Clean Restart

```bash
# 1. Stop all services
pkill -f "uvicorn app.main:app"
pkill -f "next dev"

# 2. Clear Next.js cache (optional, for frontend issues)
cd frontend
rm -rf .next
cd ..

# 3. Start fresh
./start.sh
```

## ✅ Verification Checklist

After restarting, verify everything works:

### 1. Check Backend Connection
```bash
# Should show database connected
tail -20 logs/backend.log
```

**Look for:**
```
✅ Database connected successfully
✅ Redis connected successfully
✅ Application startup complete
```

### 2. Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# API docs
curl http://localhost:8000/docs
```

### 3. Test Frontend
Open in browser:
- Frontend: http://localhost:3000
- Simulation: http://localhost:3000/simulation

### 4. Run a Simulation
1. Go to http://localhost:3000/simulation
2. Click "Start Simulation" (with default parameters)
3. Watch for status changes:
   - "Starting" → "Running" → "Completed"
4. Verify results display

## 📊 Expected Behavior

### Backend Logs (Success)
```
[INFO] Starting application...
✅ Database connected successfully
 Redis connected successfully
[INFO] ✅ Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Frontend (Success)
```
✓ Ready in 1500ms
○ Compiling /simulation ...
✓ Compiled /simulation in 2.6s
GET /simulation 200 in 2800ms
```

### API Calls (Success)
Browser console should show:
```
POST http://localhost:8000/api/v1/simulation/run 200 OK
GET http://localhost:8000/api/v1/simulation/status/{id} 200 OK
GET http://localhost:8000/api/v1/simulation/results/{id} 200 OK
```

## 🐛 If Issues Persist

### Database Still Not Connecting

**Re-run database initialization:**
```bash
./scripts/init-db.sh
```

**Or manually:**
```bash
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost -c "DROP DATABASE IF EXISTS gtds_db;"
PGPASSWORD='Meghal0987@23' psql -U postgres -h localhost -c "CREATE DATABASE gtds_db;"
```

### API Still Returning 404

**Hard refresh frontend:**
```bash
cd frontend
rm -rf .next node_modules/.cache
npm run dev
```

### Backend Won't Start

**Check logs:**
```bash
tail -50 logs/backend.log
```

**Reinstall dependencies:**
```bash
cd backend
conda activate fake_news
pip install -r requirements.txt --force-reinstall
```

### Frontend Won't Start

**Check logs:**
```bash
tail -50 logs/frontend.log
```

**Reinstall dependencies:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## 📝 What's Changed

### Files Updated:
1. ✅ `backend/.env` - Database password updated
2. ✅ `frontend/src/lib/api.ts` - Base URL corrected
3. ✅ `frontend/src/lib/constants.ts` - API routes updated

### Files Created:
1. ✅ `scripts/init-db.sh` - Database initialization script
2. ✅ `DATABASE_SETUP.md` - Database setup guide
3. ✅ `API_FIX_SUMMARY.md` - API fix documentation
4. ✅ This file - Restart instructions

## 🎯 Success Indicators

### ✅ Backend Success
- No database connection errors in logs
- API responds at http://localhost:8000/health
- Documentation loads at http://localhost:8000/docs

### ✅ Frontend Success
- Page loads at http://localhost:3000
- Simulation page accessible
- No console errors

### ✅ Integration Success
- Simulation can be started
- Status polling works (updates every 2s)
- Results display when complete
- No 404 errors in console

## 📚 Quick Reference

### Start Everything
```bash
./start.sh
```

### Stop Everything
```bash
# Press Ctrl+C in terminal
# OR
pkill -f "uvicorn\|next dev"
```

### View Logs
```bash
tail -f logs/backend.log logs/frontend.log
```

### Access Points
- 🌐 Frontend: http://localhost:3000
- 🔧 Backend: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs
- 📊 Simulation: http://localhost:3000/simulation

---

## 🚀 Ready to Go!

**Just run:**
```bash
./start.sh
```

**Then open:**
http://localhost:3000/simulation

**And start simulating!** 🎉

---

**Need help?** Check these guides:
- `START_HERE.md` - Quick start
- `DATABASE_SETUP.md` - Database issues
- `API_FIX_SUMMARY.md` - API issues
- `QUICKSTART.md` - Detailed setup
