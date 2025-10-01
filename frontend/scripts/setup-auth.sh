#!/bin/bash

# Authentication Setup Script
# This script helps set up the authentication system for the Fake News Game Theory platform

set -e

echo "🔐 Setting up Authentication System..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env.local exists
if [ ! -f .env.local ]; then
    echo -e "${YELLOW}⚠️  .env.local not found${NC}"
    echo "Creating .env.local from .env.example..."
    cp .env.example .env.local
    echo -e "${GREEN}✓ Created .env.local${NC}"
    echo ""
fi

# Generate NextAuth secret if not set
if ! grep -q "NEXTAUTH_SECRET=" .env.local || grep -q "NEXTAUTH_SECRET=your-super-secret-key" .env.local; then
    echo "🔑 Generating secure NEXTAUTH_SECRET..."
    SECRET=$(openssl rand -base64 32)

    # Update or add NEXTAUTH_SECRET in .env.local
    if grep -q "NEXTAUTH_SECRET=" .env.local; then
        sed -i.bak "s|NEXTAUTH_SECRET=.*|NEXTAUTH_SECRET=${SECRET}|" .env.local
        rm .env.local.bak
    else
        echo "NEXTAUTH_SECRET=${SECRET}" >> .env.local
    fi

    echo -e "${GREEN}✓ Generated and saved NEXTAUTH_SECRET${NC}"
    echo ""
fi

# Check for required environment variables
echo "📋 Checking required environment variables..."
MISSING_VARS=()

if ! grep -q "DATABASE_URL=postgresql://" .env.local; then
    MISSING_VARS+=("DATABASE_URL")
fi

if ! grep -q "GOOGLE_CLIENT_ID=.*apps.googleusercontent.com" .env.local; then
    MISSING_VARS+=("GOOGLE_CLIENT_ID")
fi

if ! grep -q "GOOGLE_CLIENT_SECRET=.*" .env.local && [ "${#MISSING_VARS[@]}" -eq 0 ]; then
    MISSING_VARS+=("GOOGLE_CLIENT_SECRET")
fi

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Please update .env.local with these values before continuing."
    echo "See INSTALL_AUTH.md for instructions on obtaining OAuth credentials."
    echo ""
fi

# Initialize Prisma
echo "🗄️  Setting up database..."
if command -v npx &> /dev/null; then
    echo "Generating Prisma client..."
    npx prisma generate
    echo -e "${GREEN}✓ Prisma client generated${NC}"
    echo ""

    # Ask if user wants to run migrations
    read -p "Do you want to run database migrations now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running database migrations..."
        npx prisma migrate dev --name init_auth
        echo -e "${GREEN}✓ Database migrations completed${NC}"
        echo ""
    else
        echo -e "${YELLOW}⚠️  Skipped migrations. Run 'npx prisma migrate dev' when ready.${NC}"
        echo ""
    fi
else
    echo -e "${RED}✗ npx not found. Please install Node.js and npm.${NC}"
    exit 1
fi

# Summary
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Authentication setup completed!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "📚 Next steps:"
echo ""
echo "1. Configure OAuth providers in .env.local:"
echo "   - Google: https://console.cloud.google.com/"
echo "   - GitHub: https://github.com/settings/developers"
echo ""
echo "2. Start the development server:"
echo "   ${BLUE}npm run dev${NC}"
echo ""
echo "3. Visit the sign-in page:"
echo "   ${BLUE}http://localhost:3000/auth/signin${NC}"
echo ""
echo "4. (Optional) Create an admin user:"
echo "   ${BLUE}npx prisma studio${NC}"
echo "   Navigate to Users table and change role to 'admin'"
echo ""
echo "📖 For detailed instructions, see:"
echo "   - INSTALL_AUTH.md"
echo "   - docs/AUTHENTICATION.md"
echo ""
