# Quick Start Guide

## Starting the Application

### Option 1: Quick Start (Recommended)
```bash
./start.sh
```

### Option 2: Using the development script directly
```bash
./scripts/run-dev.sh
```

### Option 3: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
conda activate fake_news
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Access Points

Once both services are running, you can access:

- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API Explorer**: http://localhost:8000/redoc

### Main Pages

- **Simulation Page**: http://localhost:3000/simulation
- **Equilibrium Analysis**: http://localhost:3000/equilibrium
- **Network Analysis**: http://localhost:3000/network
- **Classifier**: http://localhost:3000/classifier

## First Time Setup

If this is your first time running the application:

1. **Run the setup script:**
   ```bash
   ./scripts/setup.sh
   ```

2. **Activate conda environment (backend):**
   ```bash
   conda activate fake_news
   ```

3. **Start the application:**
   ```bash
   ./start.sh
   ```

## Stopping the Application

- If using `./start.sh` or `./scripts/run-dev.sh`: Press `Ctrl+C`
- If running manually: Press `Ctrl+C` in each terminal window

## Troubleshooting

### Port Already in Use

If you get an error that port 8000 or 3000 is already in use:

**Kill processes on port 8000 (Backend):**
```bash
pkill -f "uvicorn app.main:app"
# OR
lsof -ti:8000 | xargs kill -9
```

**Kill processes on port 3000 (Frontend):**
```bash
pkill -f "next dev"
# OR
lsof -ti:3000 | xargs kill -9
```

### Conda Environment Not Found

If the `fake_news` conda environment doesn't exist:

```bash
conda create -n fake_news python=3.10
conda activate fake_news
cd backend
pip install -r requirements.txt
```

### Frontend Dependencies Missing

If frontend won't start:

```bash
cd frontend
npm install
```

### Database Connection Issues

If the backend fails to connect to the database:

1. Check if PostgreSQL is running:
   ```bash
   sudo service postgresql status
   ```

2. Check your `.env` file in the `backend` directory

3. Initialize the database:
   ```bash
   cd backend
   conda activate fake_news
   python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
   ```

### Redis Connection Issues

If Redis connection fails:

1. Start Redis:
   ```bash
   sudo service redis-server start
   ```

2. Or disable Redis in backend `.env`:
   ```
   USE_REDIS=false
   ```

## Viewing Logs

The development script creates log files in the `logs/` directory:

**View all logs:**
```bash
tail -f logs/backend.log logs/frontend.log
```

**Backend logs only:**
```bash
tail -f logs/backend.log
```

**Frontend logs only:**
```bash
tail -f logs/frontend.log
```

## Environment Variables

### Backend (.env)

Create `backend/.env` with:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/fake_news_db

# Redis
REDIS_URL=redis://localhost:6379/0
USE_REDIS=true

# API
API_VERSION=v1
DEBUG=true
SECRET_KEY=your-secret-key-here

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# ML Models
MODEL_PATH=models/
```

### Frontend (.env.local)

Create `frontend/.env.local` with:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_VERSION=v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## Testing the Integration

### 1. Test Backend API
```bash
curl http://localhost:8000/health
```

Expected response: `{"status":"healthy"}`

### 2. Test Frontend
Open http://localhost:3000 in your browser

### 3. Run a Simulation

1. Navigate to http://localhost:3000/simulation
2. Configure parameters
3. Click "Start Simulation"
4. Watch the real-time results

## Common Issues

### Issue: "Module not found" errors in frontend
**Solution:** Clear Next.js cache and reinstall
```bash
cd frontend
rm -rf .next node_modules package-lock.json
npm install
npm run dev
```

### Issue: Backend fails with import errors
**Solution:** Reinstall backend dependencies
```bash
conda activate fake_news
cd backend
pip install -r requirements.txt --force-reinstall
```

### Issue: Simulation returns 500 error
**Solution:** Check backend logs
```bash
tail -50 logs/backend.log
```

Common causes:
- Database not initialized
- Missing ML models
- Network generation timeout

## Performance Tips

1. **Reduce network size** in simulations (start with 100 nodes)
2. **Enable Redis** for faster repeated requests
3. **Use production build** for better frontend performance:
   ```bash
   cd frontend
   npm run build
   npm start
   ```

## Development Tips

### Hot Reload

Both frontend and backend support hot reload:
- **Backend**: Auto-reloads on file changes (uvicorn --reload)
- **Frontend**: Fast Refresh for React components

### API Testing

Use the interactive API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Database Inspection

```bash
# Using psql
psql -U your_user -d fake_news_db

# List tables
\dt

# Query simulations
SELECT * FROM simulations LIMIT 10;
```

## Next Steps

1. ✅ Start the application
2. ✅ Access the frontend
3. ✅ Run a simulation
4. 📚 Read the full documentation in `docs/`
5. 🧪 Explore different game theory scenarios
6. 📊 Analyze network propagation patterns

## Support

- **Documentation**: See `README.md` and `WORKFLOW.md`
- **API Reference**: http://localhost:8000/docs
- **Integration Status**: See `INTEGRATION_STATUS.md`

---

**Happy researching! 🚀**
