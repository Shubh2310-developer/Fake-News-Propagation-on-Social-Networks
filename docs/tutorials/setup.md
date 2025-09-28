# Getting Started - Project Setup Guide

This guide will help you get the entire fake news detection and game theory simulation platform running on your local machine in the shortest possible time.

## Prerequisites Checklist

Before starting, ensure you have the following tools installed:

### Required Software
- **Git** (v2.25+): [Download Git](https://git-scm.com/downloads)
- **Docker** (v20.10+): [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** (v2.0+): Usually included with Docker Desktop
- **Node.js** (v18+): [Download Node.js](https://nodejs.org/)
- **Python** (v3.10+): [Install Python](https://www.python.org/downloads/)
- **Conda** (recommended): [Install Anaconda](https://www.anaconda.com/download)

### Optional Tools
- **VS Code**: [Download VS Code](https://code.visualstudio.com/) with Python and TypeScript extensions
- **Postman**: [Download Postman](https://www.postman.com/downloads/) for API testing

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended for ML training)
- **Storage**: At least 10GB free space
- **OS**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+

## Quick Setup (5-Minute Installation)

### 1. Clone the Repository

```bash
# Clone the project
git clone https://github.com/your-org/fake-news-game-theory.git
cd fake-news-game-theory

# Verify you're in the right directory
ls -la
# You should see: backend/ frontend/ docs/ scripts/ etc.
```

### 2. Environment Setup

```bash
# Copy environment files
cp frontend/.env.example frontend/.env
cp backend/.env.example backend/.env

# Review and edit environment variables
# Edit frontend/.env and backend/.env with your preferred editor
```

#### Key Environment Variables to Configure

**Frontend (.env)**:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME="Fake News Game Theory Platform"
```

**Backend (.env)**:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/fake_news_db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here
API_TOKEN_EXPIRE_HOURS=24
```

### 3. One-Command Installation

```bash
# Install all dependencies and start services
docker-compose up --build
```

**Alternative: Manual Installation**

If you prefer not to use Docker:

```bash
# Install backend dependencies
cd backend
conda create -n fake_news python=3.10
conda activate fake_news
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install

# Start services manually
# Terminal 1: Backend
cd backend
conda activate fake_news
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Database (if not using Docker)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:14
```

## Verification Steps

### 1. Check Services Status

```bash
# Check running containers
docker-compose ps

# Expected output:
# NAME                COMMAND             SERVICE     STATUS     PORTS
# fake-news-backend   "uvicorn main:app"  backend     Up         8000/tcp
# fake-news-frontend  "npm run dev"       frontend    Up         3000/tcp
# fake-news-db        "postgres"          postgres    Up         5432/tcp
# fake-news-redis     "redis-server"      redis       Up         6379/tcp
```

### 2. Access the Application

1. **Frontend Dashboard**: http://localhost:3000
   - Should display the main dashboard
   - Navigation menu should be functional

2. **Backend API**: http://localhost:8000
   - Should show FastAPI welcome page
   - Access interactive docs at http://localhost:8000/docs

3. **Health Check**: http://localhost:8000/api/health
   - Should return: `{"status": "healthy", "version": "1.0.0"}`

### 3. Test Core Functionality

#### Test Fake News Classification

```bash
# Using curl
curl -X POST "http://localhost:8000/api/classifier/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover that coffee cures all diseases!"}'

# Expected response:
# {
#   "success": true,
#   "data": {
#     "prediction": "fake",
#     "confidence": 0.85
#   }
# }
```

#### Test Game Theory Simulation

```bash
# Start a simple simulation
curl -X POST "http://localhost:8000/api/simulation/run" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "network_size": 100,
      "network_type": "scale_free",
      "detection_rate": 0.7,
      "simulation_steps": 50
    }
  }'

# Should return simulation ID and status
```

## Troubleshooting Common Issues

### Port Conflicts

```bash
# Check if ports are already in use
lsof -i :3000  # Frontend
lsof -i :8000  # Backend
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# Kill processes using ports
sudo kill -9 <PID>

# Or use different ports in docker-compose.yml
```

### Docker Issues

```bash
# Clean up Docker containers and volumes
docker-compose down -v
docker system prune -f

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up
```

### Memory Issues

```bash
# Increase Docker memory allocation
# Docker Desktop > Settings > Resources > Advanced
# Set Memory to at least 8GB

# For Linux, check available memory
free -h
```

### Python Environment Issues

```bash
# Reset conda environment
conda deactivate
conda remove -n fake_news --all
conda create -n fake_news python=3.10
conda activate fake_news
pip install -r requirements.txt
```

### Database Connection Issues

```bash
# Check database connection
docker-compose exec postgres psql -U postgres -d fake_news_db -c "\dt"

# Reset database
docker-compose down -v
docker-compose up postgres -d
# Wait 30 seconds for initialization
docker-compose up
```

## Development Workflow

### Git Hooks (Optional)

```bash
# Install pre-commit hooks for code quality
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### IDE Setup

**VS Code Extensions**:
- Python (Microsoft)
- TypeScript and JavaScript Language Features
- Docker (Microsoft)
- PostgreSQL (Chris Kolkman)

**VS Code Settings** (add to `.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "./backend/.venv/bin/python",
  "typescript.preferences.importModuleSpecifier": "relative",
  "eslint.workingDirectories": ["frontend"]
}
```

### Hot Reloading

- **Frontend**: Automatically reloads on file changes (Next.js)
- **Backend**: Automatically reloads on file changes (uvicorn --reload)
- **Database Schema**: Use Alembic for migrations

```bash
# Generate migration after model changes
cd backend
alembic revision --autogenerate -m "Add new table"
alembic upgrade head
```

## Next Steps

Once your setup is complete:

1. **Explore the Dashboard**: Navigate through the web interface
2. **Run Your First Simulation**: Use the simulation interface
3. **Test the API**: Use the interactive docs at `/docs`
4. **Review Documentation**: Read the methodology and tutorial docs
5. **Train Models**: Follow the model training tutorial

## Getting Help

If you encounter issues:

1. **Check Logs**:
   ```bash
   docker-compose logs backend
   docker-compose logs frontend
   ```

2. **Search Issues**: Check the GitHub issues page

3. **Create Issue**: Report bugs with detailed information:
   - OS and version
   - Docker version
   - Error logs
   - Steps to reproduce

4. **Discord/Slack**: Join our developer community (if available)

## Success! <‰

You now have a fully functional fake news detection and game theory simulation platform running locally. The system includes:

-  Web dashboard for interactive simulations
-  REST API for programmatic access
-  Machine learning models for fake news detection
-  Game theory simulation engine
-  Database for persistent storage
-  Redis for caching and session management

Proceed to the [Running Simulations Tutorial](./running_simulations.md) to learn how to use the platform effectively.