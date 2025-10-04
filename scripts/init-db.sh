#!/bin/bash

# ==============================================
# Database Initialization Script
# ==============================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${YELLOW}→ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

echo "================================================"
echo "Database Initialization"
echo "================================================"
echo ""

# Load environment variables
if [ -f "backend/.env" ]; then
    export $(cat backend/.env | grep -v '^#' | xargs)
    print_success "Loaded environment variables from backend/.env"
else
    print_error "backend/.env file not found"
    exit 1
fi

# Check PostgreSQL connection
print_info "Checking PostgreSQL connection..."
if PGPASSWORD="$DB_PASSWORD" psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c '\l' > /dev/null 2>&1; then
    print_success "PostgreSQL is accessible"
else
    print_error "Cannot connect to PostgreSQL"
    echo "Please ensure PostgreSQL is running and credentials are correct"
    exit 1
fi

# Create database if it doesn't exist
print_info "Checking if database '$DB_NAME' exists..."
if PGPASSWORD="$DB_PASSWORD" psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    print_success "Database '$DB_NAME' already exists"
else
    print_info "Creating database '$DB_NAME'..."
    PGPASSWORD="$DB_PASSWORD" psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -c "CREATE DATABASE $DB_NAME;"
    print_success "Database '$DB_NAME' created successfully"
fi

# Run database migrations (if using Alembic)
if [ -d "backend/alembic" ]; then
    print_info "Running database migrations..."
    cd backend

    # Activate conda environment if it exists
    if conda env list | grep -q "fake_news"; then
        eval "$(conda shell.bash hook)"
        conda activate fake_news
    else
        source venv/bin/activate
    fi

    alembic upgrade head
    cd ..
    print_success "Database migrations completed"
else
    print_info "No alembic directory found - skipping migrations"
fi

# Initialize tables from models (if no migrations)
print_info "Initializing database tables..."
cd backend

# Activate environment
if conda env list | grep -q "fake_news"; then
    eval "$(conda shell.bash hook)"
    conda activate fake_news
else
    source venv/bin/activate
fi

# Run initialization script
python -c "
import asyncio
from app.core.database import init_db

async def main():
    await init_db()
    print('Database tables initialized')

asyncio.run(main())
" 2>&1 | grep -v "WARNING" || true

cd ..
print_success "Database initialization completed"

echo ""
echo "================================================"
print_success "Database ready to use!"
echo "================================================"
echo ""
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo "Host: $DB_HOST:$DB_PORT"
echo ""
