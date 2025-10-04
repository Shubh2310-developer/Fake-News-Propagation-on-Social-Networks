# 🚀 START HERE - Quick Launch Guide

## One-Command Start

```bash
./start.sh
```

That's it! This will:
- ✅ Start the backend API on port 8000
- ✅ Start the frontend on port 3000
- ✅ Stream logs from both services
- ✅ Clean up gracefully on Ctrl+C

## What You'll See

```
================================================
Fake News Game Theory - Development Mode
================================================

→ Starting development servers...

Starting Backend Server...
→ Backend will be available at http://localhost:8000
→ API Documentation at http://localhost:8000/docs
→ Using conda environment: fake_news
→ Waiting for backend to initialize...
✓ Backend started successfully (PID: 12345)

Starting Frontend Server...
→ Frontend will be available at http://localhost:3000
→ Waiting for frontend to initialize...
✓ Frontend started successfully (PID: 12346)

================================================
✓ 🚀 All Services Running Successfully!
================================================

Access Points:
  🌐 Frontend:        http://localhost:3000
  🔧 Backend API:     http://localhost:8000
  📚 API Docs:        http://localhost:8000/docs
  📊 Simulation:      http://localhost:3000/simulation
  🎯 Equilibrium:     http://localhost:3000/equilibrium
  🤖 Classifier:      http://localhost:3000/classifier

Process Information:
  Backend PID:        12345
  Frontend PID:       12346

================================================
→ Streaming logs (Ctrl+C to stop)...
================================================
```

## First Time? Run Setup First

If you haven't set up the project yet:

```bash
# 1. Run setup (installs dependencies)
./scripts/setup.sh

# 2. Activate conda environment
conda activate fake_news

# 3. Start the application
./start.sh
```

## Quick Access Links

Once running, click these:

- 🌐 **Main App**: http://localhost:3000
- 📊 **Run Simulation**: http://localhost:3000/simulation
- 🎯 **Equilibrium Analysis**: http://localhost:3000/equilibrium
- 🕸️ **Network Visualization**: http://localhost:3000/network
- 🤖 **Fake News Classifier**: http://localhost:3000/classifier
- 📚 **API Documentation**: http://localhost:8000/docs

## Running a Simulation

1. Open http://localhost:3000/simulation
2. Configure parameters (or use defaults):
   - Network size: 100 nodes
   - Network type: Small World
   - Time horizon: 50 rounds
3. Click "Start Simulation"
4. Watch real-time results appear!

## Stopping the Application

Press `Ctrl+C` in the terminal where you ran `./start.sh`

The script will automatically:
- Stop backend gracefully
- Stop frontend gracefully
- Clean up processes

## Troubleshooting

### "Port already in use"

Kill existing processes:
```bash
# Kill backend on port 8000
pkill -f "uvicorn app.main:app"

# Kill frontend on port 3000
pkill -f "next dev"

# Then restart
./start.sh
```

### "Backend failed to start"

Check the logs:
```bash
cat logs/backend.log
```

Common fixes:
```bash
# Reinstall backend dependencies
conda activate fake_news
cd backend
pip install -r requirements.txt
```

### "Frontend failed to start"

Check the logs:
```bash
cat logs/frontend.log
```

Common fixes:
```bash
# Clear cache and reinstall
cd frontend
rm -rf .next node_modules
npm install
```

### "Conda environment not found"

Create it:
```bash
conda create -n fake_news python=3.10
conda activate fake_news
cd backend
pip install -r requirements.txt
```

## Alternative: Manual Start

If the automated script doesn't work, start manually:

**Terminal 1 (Backend):**
```bash
cd backend
conda activate fake_news
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

## File Structure

```
fake-news-game-theory/
├── start.sh                 # ← Quick start script (USE THIS!)
├── scripts/
│   ├── run-dev.sh          # Full development script
│   └── setup.sh            # Initial setup
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/v1/         # API endpoints
│   │   ├── services/       # Business logic
│   │   ├── core/           # Database, config, cache
│   │   └── main.py         # Entry point
│   └── requirements.txt
├── frontend/               # Next.js React frontend
│   ├── src/
│   │   ├── app/           # Pages and layouts
│   │   ├── components/    # React components
│   │   ├── store/         # State management
│   │   └── types/         # TypeScript types
│   └── package.json
└── logs/                   # Runtime logs
    ├── backend.log
    └── frontend.log
```

## Key Features

### ✅ Fully Integrated
- Backend API connected to frontend
- Real-time simulation updates
- Live network visualization
- Game theory equilibrium analysis

### ✅ Development Features
- Hot reload on both frontend and backend
- Automatic error detection
- Color-coded log streaming
- Graceful shutdown

### ✅ Production Ready
- Environment-based configuration
- Redis caching support
- PostgreSQL database integration
- CORS configured

## Next Steps

1. ✅ **Start the app**: `./start.sh`
2. 🔍 **Explore the UI**: Visit http://localhost:3000
3. 🎮 **Run a simulation**: Go to /simulation page
4. 📊 **View results**: Analyze equilibrium and network metrics
5. 📖 **Read docs**: Check `QUICKSTART.md` for detailed info
6. 🧪 **Experiment**: Try different game theory parameters

## Documentation

- 📄 **QUICKSTART.md** - Detailed setup and troubleshooting
- 📄 **WORKFLOW.md** - Project workflow and architecture
- 📄 **INTEGRATION_STATUS.md** - Integration status and features
- 📄 **README.md** - Full project documentation

## Testing the Integration

```bash
# 1. Start the application
./start.sh

# 2. In another terminal, test the backend
curl http://localhost:8000/health
# Should return: {"status":"ok",...}

# 3. Test the frontend
curl -I http://localhost:3000
# Should return: HTTP/1.1 200 OK

# 4. Open browser and run a simulation
# Navigate to: http://localhost:3000/simulation
```

## Support & Issues

- 🐛 **Found a bug?** Check logs in `logs/` directory
- ❓ **Have questions?** Read `QUICKSTART.md`
- 🔧 **Need help?** Check the troubleshooting section

---

## 🎉 You're All Set!

Run `./start.sh` and start exploring the Fake News Game Theory platform!

**Happy simulating! 🚀📊🎯**
