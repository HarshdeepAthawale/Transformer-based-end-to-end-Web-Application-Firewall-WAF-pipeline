#!/usr/bin/env python3
"""
Quick WAF Service Test

Simple test to verify WAF service is working
"""
import requests
import json
import time
import subprocess
import sys
from pathlib import Path

def test_waf_service():
    """Test WAF service functionality"""
    waf_url = "http://127.0.0.1:8888"

    print("Testing WAF Service...")

    # Test 1: Health check
    try:
        response = requests.get(f"{waf_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ Health check passed")
            print(f"  Status: {data.get('status')}")
            print(f"  Model loaded: {data.get('model_loaded')}")
            print(f"  Vocab size: {data.get('vocab_size')}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

    # Test 2: Normal request
    try:
        request_data = {
            "method": "GET",
            "path": "/api/products",
            "query_params": {"page": "1", "limit": "10"},
            "headers": {"user-agent": "test-client"}
        }
        response = requests.post(f"{waf_url}/check", json=request_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✓ Normal request test passed")
            print(f"  Anomaly score: {result.get('anomaly_score', 'N/A')}")
            print(f"  Is anomaly: {result.get('is_anomaly', 'N/A')}")
        else:
            print(f"✗ Normal request test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Normal request test failed: {e}")
        return False

    # Test 3: Suspicious request (SQL injection)
    try:
        request_data = {
            "method": "GET",
            "path": "/api/users",
            "query_params": {"id": "1' OR '1'='1"},
            "headers": {"user-agent": "test-client"}
        }
        response = requests.post(f"{waf_url}/check", json=request_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✓ SQL injection test passed")
            print(f"  Anomaly score: {result.get('anomaly_score', 'N/A')}")
            print(f"  Is anomaly: {result.get('is_anomaly', 'N/A')}")
        else:
            print(f"✗ SQL injection test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ SQL injection test failed: {e}")
        return False

    # Test 4: Metrics
    try:
        response = requests.get(f"{waf_url}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print("✓ Metrics test passed")
            print(f"  Total requests: {metrics.get('total_requests', 'N/A')}")
            print(f"  Anomalies detected: {metrics.get('anomalies_detected', 'N/A')}")
        else:
            print(f"✗ Metrics test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

    print("\n✓ All WAF service tests passed!")
    return True

def main():
    # Start WAF service
    print("Starting WAF service...")
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    cmd = [
        sys.executable, "scripts/start_waf_service.py",
        "--host", "127.0.0.1",
        "--port", "8888",
        "--workers", "1",
        "--log_level", "warning"
    ]

    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for service to start
    print("Waiting for service to start...")
    time.sleep(5)

    try:
        # Run tests
        success = test_waf_service()

        if success:
            print("\n🎉 WAF Integration Phase 6 completed successfully!")
            print("\nNext steps:")
            print("1. Set up Nginx reverse proxy: ./scripts/setup_nginx_waf.sh")
            print("2. Configure your web application on port 8080")
            print("3. Test end-to-end: curl http://localhost/")
            print("4. Monitor: curl http://localhost/waf-metrics")
        else:
            print("\n❌ WAF service tests failed")

        return 0 if success else 1

    finally:
        # Stop service
        print("\nStopping WAF service...")
        process.terminate()
        process.wait(timeout=5)

if __name__ == "__main__":
    sys.exit(main())