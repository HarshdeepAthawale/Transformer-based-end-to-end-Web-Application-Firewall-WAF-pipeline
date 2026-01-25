#!/bin/bash
# Quick script to run WAF test

echo "=========================================="
echo "WAF 200 Requests Test"
echo "=========================================="
echo ""
echo "Make sure your API server is running on http://localhost:3001"
echo "Press Enter to continue or Ctrl+C to cancel..."
read

echo ""
echo "Starting test..."
echo ""

python3 scripts/test_waf_200_requests_simple.py \
    --malicious_count 100 \
    --normal_count 100 \
    --api_base http://localhost:3001 \
    --max_workers 10

echo ""
echo "Test completed! Check the logs and reports directory for results."
