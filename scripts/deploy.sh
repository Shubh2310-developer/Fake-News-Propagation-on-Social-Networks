#!/bin/bash

# ==============================================
# Fake News Game Theory - Deployment Script
# ==============================================

set -e

echo "================================================"
echo "Fake News Game Theory - Deployment"
echo "================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${YELLOW}→ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

# Check Docker
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    print_error "Docker/docker-compose not found"
    exit 1
fi

ENVIRONMENT=${1:-production}
ACTION=${2:-up}

print_info "Environment: $ENVIRONMENT"
print_info "Action: $ACTION"

case $ENVIRONMENT in
    dev|development)
        COMPOSE_FILE="config/docker-compose.yml:config/docker-compose.dev.yml"
        ;;
    prod|production)
        COMPOSE_FILE="config/docker-compose.yml:config/docker-compose.prod.yml"
        ;;
    test)
        COMPOSE_FILE="config/docker-compose.yml:config/docker-compose.test.yml"
        ;;
    *)
        print_error "Unknown environment: $ENVIRONMENT"
        echo "Usage: $0 <dev|prod|test> <up|down|restart|logs>"
        exit 1
        ;;
esac

case $ACTION in
    up)
        print_info "Building and starting containers..."
        docker-compose -f $(echo $COMPOSE_FILE | tr ':' ' ' | awk '{print "-f " $1 " -f " $2}') up --build -d
        print_success "Containers started"
        echo ""
        echo "Access: http://localhost:3000 (frontend), http://localhost:8000 (backend)"
        ;;
    down)
        print_info "Stopping containers..."
        docker-compose -f $(echo $COMPOSE_FILE | tr ':' ' ' | awk '{print "-f " $1 " -f " $2}') down
        print_success "Containers stopped"
        ;;
    restart)
        print_info "Restarting containers..."
        docker-compose -f $(echo $COMPOSE_FILE | tr ':' ' ' | awk '{print "-f " $1 " -f " $2}') restart
        print_success "Containers restarted"
        ;;
    logs)
        docker-compose -f $(echo $COMPOSE_FILE | tr ':' ' ' | awk '{print "-f " $1 " -f " $2}') logs -f
        ;;
    ps|status)
        docker-compose -f $(echo $COMPOSE_FILE | tr ':' ' ' | awk '{print "-f " $1 " -f " $2}') ps
        ;;
    *)
        print_error "Unknown action: $ACTION"
        exit 1
        ;;
esac
