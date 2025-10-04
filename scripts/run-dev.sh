#!/bin/bash

# ==============================================
# Fake News Game Theory - Development Runner
# ==============================================

set -e

echo "================================================"
echo "Fake News Game Theory - Development Mode"
echo "================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${YELLOW}→ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_header() { echo -e "${BLUE}$1${NC}"; }

cleanup() {
    echo ""
    print_info "Shutting down services..."
    [ ! -z "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null || true
    [ ! -z "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null || true
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    print_success "Services stopped"
    exit 0
}

trap cleanup EXIT INT TERM

# Check if we're in the project root
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Verify setup (check for conda env or venv, and node_modules)
CONDA_ENV_EXISTS=$(conda env list | grep -c "fake_news" || echo "0")
if [ "$CONDA_ENV_EXISTS" = "0" ] && [ ! -d "backend/venv" ]; then
    print_error "Backend environment not found. Run ./scripts/setup.sh first"
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    print_error "Frontend dependencies not installed. Run ./scripts/setup.sh first"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo ""
print_info "Starting development servers..."
echo ""

# Start backend
print_header "Starting Backend Server..."
print_info "Backend will be available at http://localhost:8000"
print_info "API Documentation at http://localhost:8000/docs"

cd backend

# Activate conda environment if it exists, otherwise use venv
if [ "$CONDA_ENV_EXISTS" != "0" ]; then
    print_info "Using conda environment: fake_news"
    eval "$(conda shell.bash hook)"
    conda activate fake_news
else
    print_info "Using virtual environment"
    source venv/bin/activate
fi

# Start uvicorn
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait and verify backend started
print_info "Waiting for backend to initialize..."
sleep 3

# Wait up to 30 seconds for backend to be ready
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1 || \
       curl -s http://localhost:8000/ >/dev/null 2>&1; then
        print_success "Backend started successfully (PID: $BACKEND_PID)"
        break
    fi

    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend process died. Check logs/backend.log"
        tail -30 logs/backend.log
        exit 1
    fi

    if [ $i -eq 30 ]; then
        print_error "Backend failed to respond after 30 seconds"
        print_info "Process is running but may have issues. Check logs/backend.log"
        tail -30 logs/backend.log
        exit 1
    fi

    sleep 1
done

# Start frontend
echo ""
print_header "Starting Frontend Server..."
print_info "Frontend will be available at http://localhost:3000"

cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait and verify frontend started
print_info "Waiting for frontend to initialize..."
sleep 3

# Wait up to 45 seconds for frontend to be ready (Next.js can take time)
for i in {1..45}; do
    if curl -s http://localhost:3000 >/dev/null 2>&1; then
        print_success "Frontend started successfully (PID: $FRONTEND_PID)"
        break
    fi

    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend process died. Check logs/frontend.log"
        tail -30 logs/frontend.log
        exit 1
    fi

    if [ $i -eq 45 ]; then
        print_error "Frontend failed to respond after 45 seconds"
        print_info "Process is running but may have issues. Check logs/frontend.log"
        tail -30 logs/frontend.log
        # Don't exit - frontend might still be compiling
        print_info "Continuing anyway - frontend may still be starting..."
        break
    fi

    # Show progress every 5 seconds
    if [ $((i % 5)) -eq 0 ]; then
        print_info "Still waiting for frontend... ($i/45s)"
    fi

    sleep 1
done

echo ""
echo "================================================"
print_success "🚀 All Services Running Successfully!"
echo "================================================"
echo ""
print_header "Access Points:"
echo "  🌐 Frontend:        http://localhost:3000"
echo "  🔧 Backend API:     http://localhost:8000"
echo "  📚 API Docs:        http://localhost:8000/docs"
echo "  📊 Simulation:      http://localhost:3000/simulation"
echo "  🎯 Equilibrium:     http://localhost:3000/equilibrium"
echo "  🤖 Classifier:      http://localhost:3000/classifier"
echo ""
print_header "Process Information:"
echo "  Backend PID:        $BACKEND_PID"
echo "  Frontend PID:       $FRONTEND_PID"
echo ""
print_header "Useful Commands:"
echo "  View logs:          tail -f logs/backend.log logs/frontend.log"
echo "  Backend log only:   tail -f logs/backend.log"
echo "  Frontend log only:  tail -f logs/frontend.log"
echo "  Stop servers:       Press Ctrl+C"
echo ""
echo "================================================"
print_info "Streaming logs (Ctrl+C to stop)..."
echo "================================================"
echo ""

# Stream logs with color coding
tail -f logs/backend.log logs/frontend.log | while read line; do
    if echo "$line" | grep -qi "error\|failed\|exception"; then
        echo -e "${RED}$line${NC}"
    elif echo "$line" | grep -qi "warning\|warn"; then
        echo -e "${YELLOW}$line${NC}"
    elif echo "$line" | grep -qi "success\|ready\|compiled\|started"; then
        echo -e "${GREEN}$line${NC}"
    else
        echo "$line"
    fi
done
