#!/bin/bash
# Quick deployment helper for Render.com

echo "=========================================="
echo "Fake News Detection - Render Deployment"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Prerequisites:${NC}"
echo "1. Render account: https://render.com"
echo "2. GitHub repository connected to Render"
echo "3. Environment variables ready"
echo ""

read -p "Have you completed the prerequisites? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${RED}Please complete prerequisites first${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 1: Generate Secret Key${NC}"
echo "Copy this secret key for your .env:"
SECRET_KEY=$(openssl rand -hex 32)
echo -e "${YELLOW}SECRET_KEY=${SECRET_KEY}${NC}"
echo ""

echo -e "${GREEN}Step 2: Environment Variables Checklist${NC}"
echo "Add these to your Render Web Service:"
echo ""
echo "APPLICATION:"
echo "  APP_NAME=GTDS Fake News Detection"
echo "  ENVIRONMENT=production"
echo "  DEBUG=false"
echo ""
echo "SECURITY:"
echo "  SECRET_KEY=${SECRET_KEY}"
echo "  ALGORITHM=HS256"
echo ""
echo "DATABASE (from Render PostgreSQL):"
echo "  DATABASE_URL=<Internal Database URL>"
echo ""
echo "REDIS (from Render Redis):"
echo "  REDIS_URL=<Redis URL>"
echo ""
echo "CORS (update with your frontend domain):"
echo "  CORS_ORIGINS=[\"https://your-app.vercel.app\"]"
echo ""

echo -e "${GREEN}Step 3: Render Web Service Configuration${NC}"
echo ""
echo "Build Command:"
echo "  pip install -r requirements.txt"
echo ""
echo "Start Command:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port \$PORT"
echo ""
echo "Root Directory:"
echo "  backend"
echo ""

echo -e "${YELLOW}For detailed instructions, see: docs/DEPLOYMENT_GUIDE.md${NC}"
echo ""
echo "=========================================="
