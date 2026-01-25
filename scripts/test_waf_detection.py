#!/usr/bin/env python3
"""
Test WAF Detection

Tests the WAF by sending malicious and normal requests through the API
and verifying they are blocked/allowed correctly.
"""
import sys
import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Configure logger
logger.add("logs/test_waf_detection.log", rotation="10 MB", retention="7 days")

# API base URL
API_BASE = "http://localhost:3001"

# Test samples
MALICIOUS_REQUESTS = [
    {
        "method": "GET",
        "path": "/api/test",
        "query_params": {"id": "1' OR '1'='1"},
        "expected_status": 403,
        "description": "SQL Injection in query param"
    },
    {
        "method": "GET",
        "path": "/api/search",
        "query_params": {"q": "<script>alert('XSS')</script>"},
        "expected_status": 403,
        "description": "XSS in query param"
    },
    {
        "method": "POST",
        "path": "/api/login",
        "body": {"username": "admin'--", "password": "test"},
        "expected_status": 403,
        "description": "SQL Injection in POST body"
    },
    {
        "method": "GET",
        "path": "/api/file",
        "query_params": {"path": "../../../etc/passwd"},
        "expected_status": 403,
        "description": "Path traversal"
    },
    {
        "method": "GET",
        "path": "/api/exec",
        "query_params": {"cmd": "; cat /etc/passwd"},
        "expected_status": 403,
        "description": "Command injection"
    },
    {
        "method": "POST",
        "path": "/api/comment",
        "body": {"text": "<img src=x onerror=alert(1)>"},
        "expected_status": 403,
        "description": "XSS in POST body"
    },
]

NORMAL_REQUESTS = [
    {
        "method": "GET",
        "path": "/api/products",
        "query_params": {"page": "1", "limit": "10"},
        "expected_status": [200, 404],  # 404 is OK if endpoint doesn't exist
        "description": "Normal GET request"
    },
    {
        "method": "GET",
        "path": "/api/health",
        "expected_status": [200],
        "description": "Health check"
    },
    {
        "method": "POST",
        "path": "/api/login",
        "body": {"username": "user", "password": "password123"},
        "expected_status": [200, 401, 404],  # 401/404 OK if endpoint doesn't exist
        "description": "Normal POST request"
    },
    {
        "method": "GET",
        "path": "/api/search",
        "query_params": {"q": "apple"},
        "expected_status": [200, 404],
        "description": "Normal search"
    },
]


def test_request(request_config: Dict, api_base: str) -> Dict:
    """Test a single request"""
    method = request_config["method"]
    path = request_config["path"]
    query_params = request_config.get("query_params", {})
    body = request_config.get("body")
    expected_status = request_config.get("expected_status", 200)
    description = request_config.get("description", "")
    
    if not isinstance(expected_status, list):
        expected_status = [expected_status]
    
    url = f"{api_base}{path}"
    
    result = {
        "description": description,
        "method": method,
        "path": path,
        "query_params": query_params,
        "has_body": body is not None,
        "expected_status": expected_status,
        "actual_status": None,
        "success": False,
        "error": None,
        "response_time_ms": 0,
    }
    
    try:
        start_time = time.time()
        
        if method == "GET":
            response = requests.get(url, params=query_params, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=body, params=query_params, timeout=5)
        else:
            response = requests.request(method, url, json=body, params=query_params, timeout=5)
        
        result["actual_status"] = response.status_code
        result["response_time_ms"] = (time.time() - start_time) * 1000
        
        # Check if status matches expected
        result["success"] = response.status_code in expected_status
        
        # Try to get response body for debugging
        try:
            result["response_body"] = response.json()
        except:
            result["response_body"] = response.text[:200]
        
    except requests.exceptions.RequestException as e:
        result["error"] = str(e)
        result["success"] = False
    
    return result


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WAF detection")
    parser.add_argument("--api_base", type=str, default=API_BASE,
                       help="API base URL")
    parser.add_argument("--skip_normal", action="store_true",
                       help="Skip normal request tests")
    parser.add_argument("--skip_malicious", action="store_true",
                       help="Skip malicious request tests")
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("WAF DETECTION TEST")
    logger.info("="*70)
    logger.info(f"API Base: {args.api_base}")
    logger.info("")
    
    # Check API is accessible
    try:
        response = requests.get(f"{args.api_base}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"API health check failed: {response.status_code}")
            return 1
        logger.info("✓ API is accessible")
    except Exception as e:
        logger.error(f"API is not accessible: {e}")
        logger.error("Make sure the API server is running on the specified URL")
        return 1
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "api_base": args.api_base,
        "malicious_results": [],
        "normal_results": [],
    }
    
    # Test malicious requests
    if not args.skip_malicious:
        logger.info("")
        logger.info("="*70)
        logger.info("TESTING MALICIOUS REQUESTS")
        logger.info("="*70)
        
        for i, req_config in enumerate(MALICIOUS_REQUESTS):
            logger.info(f"\nTest {i+1}/{len(MALICIOUS_REQUESTS)}: {req_config['description']}")
            result = test_request(req_config, args.api_base)
            results["malicious_results"].append(result)
            
            if result["success"]:
                logger.info(f"  ✓ PASS: Request blocked (status {result['actual_status']})")
            else:
                logger.error(f"  ✗ FAIL: Expected status {result['expected_status']}, got {result['actual_status']}")
                if result.get("response_body"):
                    logger.error(f"  Response: {result['response_body']}")
    
    # Test normal requests
    if not args.skip_normal:
        logger.info("")
        logger.info("="*70)
        logger.info("TESTING NORMAL REQUESTS")
        logger.info("="*70)
        
        for i, req_config in enumerate(NORMAL_REQUESTS):
            logger.info(f"\nTest {i+1}/{len(NORMAL_REQUESTS)}: {req_config['description']}")
            result = test_request(req_config, args.api_base)
            results["normal_results"].append(result)
            
            if result["success"]:
                logger.info(f"  ✓ PASS: Request allowed (status {result['actual_status']})")
            else:
                logger.warning(f"  ⚠ WARNING: Unexpected status {result['actual_status']} (expected {result['expected_status']})")
                if result.get("response_body"):
                    logger.warning(f"  Response: {result['response_body']}")
    
    # Calculate metrics
    logger.info("")
    logger.info("="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    
    if results["malicious_results"]:
        malicious_blocked = sum(1 for r in results["malicious_results"] if r["success"])
        malicious_total = len(results["malicious_results"])
        malicious_rate = malicious_blocked / max(1, malicious_total)
        
        logger.info(f"Malicious Requests:")
        logger.info(f"  Total: {malicious_total}")
        logger.info(f"  Blocked: {malicious_blocked}")
        logger.info(f"  Block Rate: {malicious_rate:.2%}")
        
        if malicious_rate < 0.8:
            logger.warning("  ⚠ WARNING: Less than 80% of malicious requests were blocked!")
    
    if results["normal_results"]:
        normal_allowed = sum(1 for r in results["normal_results"] if r["success"])
        normal_total = len(results["normal_results"])
        normal_rate = normal_allowed / max(1, normal_total)
        
        logger.info(f"Normal Requests:")
        logger.info(f"  Total: {normal_total}")
        logger.info(f"  Allowed: {normal_allowed}")
        logger.info(f"  Allow Rate: {normal_rate:.2%}")
        
        false_positives = normal_total - normal_allowed
        if false_positives > 0:
            logger.warning(f"  ⚠ WARNING: {false_positives} false positives detected!")
    
    # Save results
    output_file = project_root / "reports" / f"waf_detection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info(f"✓ Results saved to {output_file}")
    
    # Determine overall success
    if results["malicious_results"]:
        malicious_rate = sum(1 for r in results["malicious_results"] if r["success"]) / len(results["malicious_results"])
        if malicious_rate < 0.5:
            logger.error("")
            logger.error("✗ TEST FAILED: Less than 50% of malicious requests were blocked")
            return 1
    
    logger.info("")
    logger.info("✓ WAF detection test completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
