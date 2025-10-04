# Scripts Guide

## Overview

This project includes several scripts to make development easier. All scripts are located in the `scripts/` directory.

## Available Scripts

### 🚀 Quick Start Scripts

#### 1. `./start.sh` (Recommended)
**The simplest way to start the application.**

```bash
./start.sh
```

**What it does:**
- Starts both backend and frontend
- Shows colored, real-time logs
- Handles graceful shutdown
- Automatically checks if services are healthy

**When to use:**
- Every time you want to run the application
- Daily development work
- Testing the full stack

---

#### 2. `./scripts/run-dev.sh`
**Full development mode with detailed monitoring.**

```bash
./scripts/run-dev.sh
```

**What it does:**
- Everything `start.sh` does, plus:
- Validates environment setup
- Checks conda/venv existence
- Verifies health endpoints
- Provides detailed progress updates
- Color-coded log streaming

**When to use:**
- When `start.sh` has issues
- When you need detailed startup info
- When debugging startup problems

---

#### 3. `./scripts/setup.sh`
**One-time setup script for initial installation.**

```bash
./scripts/setup.sh
```

**What it does:**
- Creates conda environment
- Installs Python dependencies
- Installs Node.js dependencies
- Sets up database
- Creates necessary directories

**When to use:**
- First time setting up the project
- After cloning the repository
- When dependencies are corrupted

---

### 🛠️ Utility Scripts

#### 4. `./scripts/deploy.sh`
**Production deployment script.**

```bash
./scripts/deploy.sh
```

**What it does:**
- Builds production frontend
- Sets up production environment
- Configures nginx (if needed)
- Runs database migrations

**When to use:**
- Deploying to production server
- Creating production builds
- Setting up staging environment

---

## Script Comparison

| Feature | start.sh | run-dev.sh | Manual Start |
|---------|----------|------------|--------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Setup Required** | None | None | Conda activation |
| **Log Streaming** | ✅ | ✅ | Manual |
| **Health Checks** | ✅ | ✅ | ❌ |
| **Auto Cleanup** | ✅ | ✅ | ❌ |
| **Error Detection** | ✅ | ✅ Advanced | ❌ |
| **Progress Updates** | Basic | Detailed | None |

## How Scripts Work

### Script Flow

```
start.sh
    ↓
Calls run-dev.sh
    ↓
1. Check environment
    ├─ Verify conda/venv exists
    ├─ Verify node_modules exists
    └─ Check project structure
    ↓
2. Start Backend
    ├─ Activate conda environment
    ├─ Run uvicorn in background
    ├─ Wait for health check
    └─ Verify port 8000 listening
    ↓
3. Start Frontend
    ├─ Run npm dev in background
    ├─ Wait for Next.js ready
    └─ Verify port 3000 listening
    ↓
4. Monitor Both
    ├─ Stream logs from both
    ├─ Color-code messages
    └─ Handle Ctrl+C gracefully
```

### Environment Detection

The scripts automatically detect your setup:

**Backend:**
```bash
# Checks in order:
1. Conda environment "fake_news" → Use conda
2. ./backend/venv/ → Use venv
3. Neither → Error and exit
```

**Frontend:**
```bash
# Checks:
1. ./frontend/node_modules/ → OK
2. Missing → Error and exit
```

## Advanced Usage

### Custom Ports

To use custom ports, edit the script or run manually:

**Backend on 8080:**
```bash
cd backend
conda activate fake_news
uvicorn app.main:app --reload --port 8080
```

**Frontend on 3001:**
```bash
cd frontend
PORT=3001 npm run dev
```

### Running in Background

To run without log streaming:

```bash
./scripts/run-dev.sh &
disown
```

View logs later:
```bash
tail -f logs/backend.log logs/frontend.log
```

### Debugging Mode

For maximum verbosity:

```bash
# Backend with debug logs
cd backend
conda activate fake_news
DEBUG=true python -m uvicorn app.main:app --reload --log-level debug

# Frontend with debug info
cd frontend
DEBUG=* npm run dev
```

## Log Files

Scripts create logs in `logs/` directory:

```bash
logs/
├── backend.log    # Backend uvicorn output
└── frontend.log   # Frontend Next.js output
```

**View logs:**
```bash
# Both logs
tail -f logs/backend.log logs/frontend.log

# Only backend
tail -f logs/backend.log

# Only frontend
tail -f logs/frontend.log

# Last 50 lines
tail -50 logs/backend.log

# Search for errors
grep -i error logs/*.log
```

## Environment Variables

### Backend (.env)

Location: `backend/.env`

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/fake_news_db

# Redis
REDIS_URL=redis://localhost:6379/0
USE_REDIS=true

# API
DEBUG=true
SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000

# ML Models
MODEL_PATH=models/
```

### Frontend (.env.local)

Location: `frontend/.env.local`

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_VERSION=v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## Common Issues & Solutions

### Issue: Script says "command not found"

**Solution:**
```bash
chmod +x scripts/*.sh start.sh
```

### Issue: "Conda environment not found"

**Solution:**
```bash
./scripts/setup.sh
# OR manually:
conda create -n fake_news python=3.10
conda activate fake_news
pip install -r backend/requirements.txt
```

### Issue: Backend starts but frontend fails

**Solution:**
```bash
cd frontend
rm -rf .next node_modules package-lock.json
npm install
```

### Issue: Port already in use

**Solution:**
```bash
# Kill all processes
pkill -f "uvicorn app.main:app"
pkill -f "next dev"

# Or specific port
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### Issue: Health check fails

**Backend not responding:**
```bash
# Check if process is running
ps aux | grep uvicorn

# Check logs
tail -50 logs/backend.log

# Common fixes:
cd backend
conda activate fake_news
pip install -r requirements.txt --force-reinstall
```

### Issue: Logs directory doesn't exist

**Solution:**
```bash
mkdir -p logs
./start.sh
```

## Script Customization

### Adding Custom Checks

Edit `scripts/run-dev.sh`:

```bash
# Add after environment checks
if [ ! -f "backend/.env" ]; then
    print_error "Backend .env file missing"
    exit 1
fi
```

### Changing Wait Times

```bash
# In run-dev.sh, find:
sleep 3  # ← Change initial wait
for i in {1..30}; do  # ← Change max wait time
```

### Adding Pre-Start Commands

```bash
# In run-dev.sh, before starting backend:
print_info "Running database migrations..."
cd backend
conda activate fake_news
alembic upgrade head
cd ..
```

## Script Best Practices

### ✅ DO:
- Use `./start.sh` for normal development
- Check logs when something fails
- Run `setup.sh` when dependencies change
- Stop services with Ctrl+C

### ❌ DON'T:
- Modify scripts while they're running
- Kill processes manually (use Ctrl+C)
- Run multiple instances simultaneously
- Delete logs directory while script is running

## Testing Scripts

### Test Backend Only
```bash
cd backend
conda activate fake_news
python -m uvicorn app.main:app --reload
```

### Test Frontend Only
```bash
cd frontend
npm run dev
```

### Test Full Stack
```bash
./start.sh
```

Then in another terminal:
```bash
# Test backend
curl http://localhost:8000/health

# Test frontend
curl http://localhost:3000
```

## Quick Reference

| Task | Command |
|------|---------|
| Start everything | `./start.sh` |
| First-time setup | `./scripts/setup.sh` |
| View backend logs | `tail -f logs/backend.log` |
| View frontend logs | `tail -f logs/frontend.log` |
| Stop everything | `Ctrl+C` |
| Kill hung processes | `pkill -f "uvicorn\|next dev"` |
| Reinstall backend | `cd backend && pip install -r requirements.txt` |
| Reinstall frontend | `cd frontend && npm install` |
| Check if running | `ps aux \| grep "uvicorn\|next"` |
| Check ports | `lsof -i:8000,3000` |

## Support

For issues with scripts:
1. Check logs in `logs/` directory
2. Read `QUICKSTART.md`
3. Try manual start method
4. Check `WORKFLOW.md` for architecture details

---

**Remember:** `./start.sh` is all you need for daily development! 🚀
