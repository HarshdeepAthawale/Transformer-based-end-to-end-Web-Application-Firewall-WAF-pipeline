#!/usr/bin/env python3
"""
Simple WAF Test: 200 Requests Through API

Sends 200 requests (100 malicious, 100 normal) to the API.
The WAF middleware will intercept and check each request.
"""
import requests
import json
import time
import random
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Configure logger
logger.add("logs/test_waf_200_requests.log", rotation="10 MB", retention="7 days")

API_BASE = "http://localhost:3001"

# Malicious payloads
MALICIOUS_PAYLOADS = [
    # SQL Injection
    "1' OR '1'='1",
    "admin'--",
    "' OR 1=1--",
    "1' UNION SELECT NULL--",
    "'; DROP TABLE users--",
    # XSS
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    "javascript:alert('XSS')",
    # Command Injection
    "; cat /etc/passwd",
    "| ls -la",
    "& whoami",
    # Path Traversal
    "../../../etc/passwd",
    "..\\..\\..\\etc\\passwd",
    "/etc/passwd",
    # File Inclusion
    "php://filter/read=string.rot13/resource=index.php",
    "file:///etc/passwd",
]

# Normal values
NORMAL_VALUES = [
    "apple",
    "orange",
    "test",
    "user",
    "123",
    "product",
    "search",
    "page",
    "1",
    "10",
    "admin",
    "home",
]


def generate_malicious_request() -> Dict:
    """Generate a malicious request"""
    method = random.choice(["GET", "POST"])
    payload = random.choice(MALICIOUS_PAYLOADS)
    
    if method == "GET":
        param_name = random.choice(["id", "q", "search", "user", "file", "path", "cmd"])
        return {
            "method": method,
            "url": f"{API_BASE}/api/test",
            "params": {param_name: payload},
            "json": None,
            "is_malicious": True,
            "payload": payload,
        }
    else:
        return {
            "method": method,
            "url": f"{API_BASE}/api/test",
            "params": {},
            "json": {"input": payload, "text": payload, "query": payload},
            "is_malicious": True,
            "payload": payload,
        }


def generate_normal_request() -> Dict:
    """Generate a normal request"""
    method = random.choice(["GET", "POST"])
    value = random.choice(NORMAL_VALUES)
    
    if method == "GET":
        param_name = random.choice(["id", "q", "search", "user", "page"])
        return {
            "method": method,
            "url": f"{API_BASE}/api/test",
            "params": {param_name: value},
            "json": None,
            "is_malicious": False,
            "payload": None,
        }
    else:
        return {
            "method": method,
            "url": f"{API_BASE}/api/test",
            "params": {},
            "json": {"input": value, "text": value},
            "is_malicious": False,
            "payload": None,
        }


def send_request(request_config: Dict) -> Dict:
    """Send a request and check if WAF blocks it"""
    result = {
        "request_config": request_config,
        "timestamp": datetime.now().isoformat(),
        "status_code": None,
        "blocked_by_waf": False,
        "allowed": False,
        "error": None,
        "response_time_ms": 0,
        "response_body": None,
    }
    
    try:
        start_time = time.time()
        
        if request_config["method"] == "GET":
            response = requests.get(
                request_config["url"],
                params=request_config["params"],
                timeout=5
            )
        else:
            response = requests.post(
                request_config["url"],
                params=request_config["params"],
                json=request_config["json"],
                timeout=5
            )
        
        result["status_code"] = response.status_code
        result["response_time_ms"] = (time.time() - start_time) * 1000
        
        # Check if blocked by WAF (403 with WAF message)
        if response.status_code == 403:
            try:
                response_data = response.json()
                if "blocked by WAF" in response_data.get("message", "").lower() or \
                   "anomaly" in response_data.get("message", "").lower():
                    result["blocked_by_waf"] = True
                    result["response_body"] = response_data
            except:
                result["blocked_by_waf"] = True  # 403 likely means blocked
        elif response.status_code in [200, 201, 404, 405]:
            # 404/405 are OK - endpoint might not exist, but WAF didn't block
            result["allowed"] = True
            try:
                result["response_body"] = response.json()
            except:
                result["response_body"] = response.text[:200]
        
    except requests.exceptions.RequestException as e:
        result["error"] = str(e)
        result["response_time_ms"] = (time.time() - start_time) * 1000
    
    return result


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WAF with 200 requests")
    parser.add_argument("--malicious_count", type=int, default=100,
                       help="Number of malicious requests")
    parser.add_argument("--normal_count", type=int, default=100,
                       help="Number of normal requests")
    parser.add_argument("--api_base", type=str, default=API_BASE,
                       help="API base URL")
    parser.add_argument("--max_workers", type=int, default=10,
                       help="Max concurrent workers")
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("WAF TEST: 200 REQUESTS")
    logger.info("="*70)
    logger.info(f"API Base: {args.api_base}")
    logger.info(f"Malicious requests: {args.malicious_count}")
    logger.info(f"Normal requests: {args.normal_count}")
    logger.info(f"Total requests: {args.malicious_count + args.normal_count}")
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
    
    # Generate requests
    logger.info("Generating test requests...")
    malicious_requests = [generate_malicious_request() for _ in range(args.malicious_count)]
    normal_requests = [generate_normal_request() for _ in range(args.normal_count)]
    all_requests = malicious_requests + normal_requests
    random.shuffle(all_requests)  # Mix them up
    
    logger.info(f"Generated {len(malicious_requests)} malicious and {len(normal_requests)} normal requests")
    logger.info("")
    
    # Send requests
    logger.info("="*70)
    logger.info("SENDING REQUESTS (this may take a while...)")
    logger.info("="*70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "api_base": args.api_base,
        "total_requests": len(all_requests),
        "malicious_count": len(malicious_requests),
        "normal_count": len(normal_requests),
        "results": [],
    }
    
    # Send requests with threading
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_request = {
            executor.submit(send_request, req): req
            for req in all_requests
        }
        
        completed = 0
        for future in as_completed(future_to_request):
            try:
                result = future.result()
                results["results"].append(result)
                completed += 1
                
                if completed % 20 == 0:
                    logger.info(f"Processed {completed}/{len(all_requests)} requests...")
                    
            except Exception as e:
                logger.error(f"Request failed with exception: {e}")
                results["results"].append({
                    "error": str(e),
                    "request_config": future_to_request[future],
                })
    
    # Analyze results
    logger.info("")
    logger.info("="*70)
    logger.info("RESULTS ANALYSIS")
    logger.info("="*70)
    
    malicious_results = [r for r in results["results"] if r.get("request_config", {}).get("is_malicious", False)]
    normal_results = [r for r in results["results"] if not r.get("request_config", {}).get("is_malicious", False)]
    
    # Malicious requests analysis
    malicious_blocked = sum(1 for r in malicious_results if r.get("blocked_by_waf", False))
    malicious_total = len(malicious_results)
    block_rate = malicious_blocked / max(1, malicious_total)
    
    logger.info("Malicious Requests:")
    logger.info(f"  Total: {malicious_total}")
    logger.info(f"  Blocked by WAF (403): {malicious_blocked}")
    logger.info(f"  Block Rate: {block_rate*100:.1f}%")
    logger.info(f"  Missed: {malicious_total - malicious_blocked} ({(1-block_rate)*100:.1f}%)")
    
    if block_rate < 0.8:
        logger.warning(f"  ⚠ WARNING: Block rate is below 80%!")
        logger.warning(f"     Consider optimizing threshold or retraining model")
    else:
        logger.info(f"  ✓ Good block rate!")
    
    # Normal requests analysis
    normal_allowed = sum(1 for r in normal_results if r.get("allowed", False) and not r.get("blocked_by_waf", False))
    normal_blocked = sum(1 for r in normal_results if r.get("blocked_by_waf", False))
    normal_total = len(normal_results)
    false_positive_rate = normal_blocked / max(1, normal_total)
    
    logger.info("")
    logger.info("Normal Requests:")
    logger.info(f"  Total: {normal_total}")
    logger.info(f"  Allowed: {normal_allowed}")
    logger.info(f"  Blocked (False Positives): {normal_blocked}")
    logger.info(f"  False Positive Rate: {false_positive_rate*100:.1f}%")
    
    if false_positive_rate > 0.1:
        logger.warning(f"  ⚠ WARNING: False positive rate is above 10%!")
        logger.warning(f"     Consider adjusting threshold")
    else:
        logger.info(f"  ✓ Low false positive rate!")
    
    # Overall metrics
    logger.info("")
    logger.info("Overall Metrics:")
    logger.info(f"  Total requests: {len(all_requests)}")
    logger.info(f"  Successful responses: {len([r for r in results['results'] if r.get('status_code')])}")
    logger.info(f"  Errors: {len([r for r in results['results'] if r.get('error')])}")
    
    # Response time statistics
    response_times = [r.get("response_time_ms", 0) for r in results["results"] if r.get("response_time_ms")]
    if response_times:
        logger.info("")
        logger.info("Response Times:")
        logger.info(f"  Average: {sum(response_times)/len(response_times):.2f}ms")
        logger.info(f"  Min: {min(response_times):.2f}ms")
        logger.info(f"  Max: {max(response_times):.2f}ms")
    
    # Save results
    output_file = project_root / "reports" / f"waf_200_requests_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info(f"✓ Results saved to {output_file}")
    
    # Final verdict
    logger.info("")
    logger.info("="*70)
    logger.info("FINAL VERDICT")
    logger.info("="*70)
    
    if block_rate >= 0.8 and false_positive_rate <= 0.1:
        logger.info("✓✓✓ WAF IS WORKING WELL! ✓✓✓")
        logger.info(f"  Block rate: {block_rate*100:.1f}% (target: 80%+)")
        logger.info(f"  False positive rate: {false_positive_rate*100:.1f}% (target: <10%)")
        return 0
    elif block_rate >= 0.5:
        logger.warning("⚠ WAF IS PARTIALLY WORKING")
        logger.warning(f"  Block rate: {block_rate*100:.1f}% (target: 80%+)")
        logger.warning(f"  False positive rate: {false_positive_rate*100:.1f}% (target: <10%)")
        logger.warning("  Recommendation: Optimize threshold or retrain model")
        return 1
    else:
        logger.error("✗✗✗ WAF IS NOT WORKING EFFECTIVELY ✗✗✗")
        logger.error(f"  Block rate: {block_rate*100:.1f}% (target: 80%+)")
        logger.error(f"  False positive rate: {false_positive_rate*100:.1f}% (target: <10%)")
        logger.error("  Recommendation: Check model, threshold, or retrain")
        return 1


if __name__ == "__main__":
    sys.exit(main())
