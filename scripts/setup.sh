#!/bin/bash

# ==============================================
# Fake News Game Theory - Setup Script
# ==============================================

set -e

echo "================================================"
echo "Fake News Game Theory - Environment Setup"
echo "================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${YELLOW}→ $1${NC}"; }

# Check prerequisites
echo ""
print_info "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi
print_success "Python $(python3 --version | cut -d' ' -f2) found"

if ! command -v node &> /dev/null; then
    print_error "Node.js not found. Please install Node.js 18+"
    exit 1
fi
print_success "Node.js $(node --version) found"

# Backend setup
echo ""
print_info "Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
print_success "Python dependencies installed"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    print_success "Created .env file"
fi

mkdir -p logs uploads ml_models/saved networks/propagations
cd ..

# Frontend setup
echo ""
print_info "Setting up frontend..."
cd frontend
npm install -q
print_success "Node dependencies installed"

if [ ! -f ".env.local" ] && [ -f ".env.example" ]; then
    cp .env.example .env.local
    print_success "Created .env.local file"
fi
cd ..

# Create logs directory
mkdir -p logs

echo ""
echo "================================================"
print_success "Setup completed!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Update environment variables in backend/.env and frontend/.env.local"
echo "2. Start services:"
echo "   - Manual: ./scripts/run-dev.sh"
echo "   - Docker: cd config && docker-compose up"
echo ""
echo "Access:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
