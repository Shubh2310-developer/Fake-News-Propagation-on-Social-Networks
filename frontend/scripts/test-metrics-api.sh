#!/bin/bash

# Test script for /api/classifier/metrics endpoint
# This script tests various scenarios of the BFF proxy

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

API_URL="${API_URL:-http://localhost:3000}"
SESSION_TOKEN="${SESSION_TOKEN:-}"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Testing Classifier Metrics API Endpoint${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to make API request
test_request() {
    local description="$1"
    local endpoint="$2"
    local expected_status="$3"
    local auth_header="$4"

    echo -e "${YELLOW}Testing:${NC} $description"
    echo -e "${BLUE}Endpoint:${NC} $endpoint"

    if [ -n "$auth_header" ]; then
        response=$(curl -s -w "\n%{http_code}" -H "$auth_header" "$API_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint")
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    if [ "$http_code" -eq "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC} - Status: $http_code"
    else
        echo -e "${RED}✗ FAIL${NC} - Expected: $expected_status, Got: $http_code"
    fi

    echo -e "${BLUE}Response:${NC}"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
    echo ""
}

# Test 1: Unauthenticated request (should return 401)
test_request \
    "Unauthenticated request" \
    "/api/classifier/metrics" \
    401 \
    ""

# Test 2: Invalid model type (should return 400)
if [ -n "$SESSION_TOKEN" ]; then
    test_request \
        "Invalid model type" \
        "/api/classifier/metrics?model_type=invalid_model" \
        400 \
        "Cookie: next-auth.session-token=$SESSION_TOKEN"
fi

# Test 3: Valid request for ensemble model
if [ -n "$SESSION_TOKEN" ]; then
    test_request \
        "Valid request - ensemble model" \
        "/api/classifier/metrics?model_type=ensemble" \
        200 \
        "Cookie: next-auth.session-token=$SESSION_TOKEN"
fi

# Test 4: Valid request for BERT model without history
if [ -n "$SESSION_TOKEN" ]; then
    test_request \
        "Valid request - BERT without history" \
        "/api/classifier/metrics?model_type=bert&include_history=false" \
        200 \
        "Cookie: next-auth.session-token=$SESSION_TOKEN"
fi

# Test 5: Cache test (second request should be cached)
if [ -n "$SESSION_TOKEN" ]; then
    echo -e "${YELLOW}Testing:${NC} Cache behavior"
    echo -e "${BLUE}First request (cache MISS):${NC}"

    response1=$(curl -s -i -H "Cookie: next-auth.session-token=$SESSION_TOKEN" \
        "$API_URL/api/classifier/metrics?model_type=random_forest")

    cache_status1=$(echo "$response1" | grep -i "X-Cache-Status:" | awk '{print $2}' | tr -d '\r')
    echo "Cache Status: $cache_status1"

    sleep 1

    echo -e "${BLUE}Second request (cache HIT):${NC}"
    response2=$(curl -s -i -H "Cookie: next-auth.session-token=$SESSION_TOKEN" \
        "$API_URL/api/classifier/metrics?model_type=random_forest")

    cache_status2=$(echo "$response2" | grep -i "X-Cache-Status:" | awk '{print $2}' | tr -d '\r')
    echo "Cache Status: $cache_status2"

    if [ "$cache_status2" = "HIT" ]; then
        echo -e "${GREEN}✓ PASS${NC} - Caching working correctly"
    else
        echo -e "${YELLOW}⚠ WARNING${NC} - Expected HIT, got: $cache_status2"
    fi
    echo ""
fi

# Test 6: Method not allowed (POST should return 405)
test_request \
    "POST method (should be rejected)" \
    "/api/classifier/metrics" \
    405 \
    ""

# Test 7: Method not allowed (PUT should return 405)
echo -e "${YELLOW}Testing:${NC} PUT method (should be rejected)"
http_code=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$API_URL/api/classifier/metrics")

if [ "$http_code" -eq 405 ]; then
    echo -e "${GREEN}✓ PASS${NC} - Status: $http_code"
else
    echo -e "${RED}✗ FAIL${NC} - Expected: 405, Got: $http_code"
fi
echo ""

# Test 8: Method not allowed (DELETE should return 405)
echo -e "${YELLOW}Testing:${NC} DELETE method (should be rejected)"
http_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$API_URL/api/classifier/metrics")

if [ "$http_code" -eq 405 ]; then
    echo -e "${GREEN}✓ PASS${NC} - Status: $http_code"
else
    echo -e "${RED}✗ FAIL${NC} - Expected: 405, Got: $http_code"
fi
echo ""

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Testing Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -z "$SESSION_TOKEN" ]; then
    echo -e "${YELLOW}Note:${NC} Some tests were skipped because SESSION_TOKEN is not set."
    echo ""
    echo "To test authenticated requests, set the SESSION_TOKEN environment variable:"
    echo -e "${BLUE}export SESSION_TOKEN=<your-session-token>${NC}"
    echo ""
    echo "You can get your session token from browser dev tools:"
    echo "1. Sign in to the app"
    echo "2. Open browser DevTools (F12)"
    echo "3. Go to Application → Cookies"
    echo "4. Find 'next-auth.session-token'"
    echo "5. Copy the value"
fi

echo ""
echo "For more details, see:"
echo "- API Documentation: docs/API_BFF_PROXY.md"
echo "- Quick Reference: API_ROUTES_REFERENCE.md"
