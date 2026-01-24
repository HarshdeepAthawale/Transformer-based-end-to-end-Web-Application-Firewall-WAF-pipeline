#!/usr/bin/env python3
"""
WAF Integration End-to-End Test Script

Tests the complete WAF integration pipeline with real model inference
"""
import requests
import json
import time
import subprocess
import signal
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WAFIntegrationTester:
    """End-to-end tester for WAF integration"""

    def __init__(self, waf_url: str = "http://127.0.0.1:8000", nginx_url: str = "http://127.0.0.1"):
        self.waf_url = waf_url.rstrip('/')
        self.nginx_url = nginx_url.rstrip('/')
        self.test_results = []

    def log_test_result(self, test_name: str, success: bool, message: str, details: Optional[Dict] = None):
        """Log a test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': time.time()
        }
        if details:
            result['details'] = details

        self.test_results.append(result)

        if success:
            logger.info(f"✓ {test_name}: {message}")
        else:
            logger.error(f"✗ {test_name}: {message}")
            if details:
                logger.error(f"  Details: {details}")

    def test_waf_service_health(self) -> bool:
        """Test WAF service health check"""
        try:
            response = requests.get(f"{self.waf_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy' and data.get('model_loaded'):
                    self.log_test_result(
                        "WAF Service Health",
                        True,
                        "WAF service is healthy and model is loaded",
                        data
                    )
                    return True
                else:
                    self.log_test_result(
                        "WAF Service Health",
                        False,
                        "WAF service reports unhealthy status",
                        data
                    )
            else:
                self.log_test_result(
                    "WAF Service Health",
                    False,
                    f"WAF service returned status {response.status_code}",
                    {'status_code': response.status_code, 'response': response.text}
                )
        except Exception as e:
            self.log_test_result(
                "WAF Service Health",
                False,
                f"Failed to connect to WAF service: {e}"
            )
        return False

    def test_waf_anomaly_detection(self) -> bool:
        """Test WAF anomaly detection with various requests"""
        test_cases = [
            {
                'name': 'Normal GET request',
                'request': {
                    'method': 'GET',
                    'path': '/api/products',
                    'query_params': {'page': '1', 'limit': '10'},
                    'headers': {'user-agent': 'Mozilla/5.0', 'accept': 'application/json'}
                },
                'expect_anomaly': False
            },
            {
                'name': 'SQL Injection attempt',
                'request': {
                    'method': 'GET',
                    'path': '/api/users',
                    'query_params': {'id': "1' OR '1'='1"},
                    'headers': {'user-agent': 'sqlmap/1.6.5'}
                },
                'expect_anomaly': True
            },
            {
                'name': 'XSS attempt',
                'request': {
                    'method': 'POST',
                    'path': '/api/comments',
                    'body': '<script>alert("xss")</script>',
                    'headers': {'content-type': 'application/json'}
                },
                'expect_anomaly': True
            },
            {
                'name': 'Path traversal attempt',
                'request': {
                    'method': 'GET',
                    'path': '/../../../etc/passwd',
                    'headers': {}
                },
                'expect_anomaly': True
            }
        ]

        all_passed = True

        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.waf_url}/check",
                    json=test_case['request'],
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    is_anomaly = result.get('is_anomaly', False)
                    score = result.get('anomaly_score', 0.0)

                    if is_anomaly == test_case['expect_anomaly']:
                        self.log_test_result(
                            f"WAF Detection - {test_case['name']}",
                            True,
                            f"Correctly detected anomaly={is_anomaly} (score: {score:.4f})",
                            {'score': score, 'is_anomaly': is_anomaly}
                        )
                    else:
                        self.log_test_result(
                            f"WAF Detection - {test_case['name']}",
                            False,
                            f"Expected anomaly={test_case['expect_anomaly']}, got {is_anomaly} (score: {score:.4f})",
                            {'expected': test_case['expect_anomaly'], 'actual': is_anomaly, 'score': score}
                        )
                        all_passed = False
                else:
                    self.log_test_result(
                        f"WAF Detection - {test_case['name']}",
                        False,
                        f"WAF service returned status {response.status_code}",
                        {'status_code': response.status_code}
                    )
                    all_passed = False

            except Exception as e:
                self.log_test_result(
                    f"WAF Detection - {test_case['name']}",
                    False,
                    f"Request failed: {e}"
                )
                all_passed = False

        return all_passed

    def test_waf_metrics(self) -> bool:
        """Test WAF metrics endpoint"""
        try:
            response = requests.get(f"{self.waf_url}/metrics", timeout=5)
            if response.status_code == 200:
                metrics = response.json()
                required_fields = [
                    'total_requests', 'anomalies_detected', 'anomaly_rate',
                    'avg_processing_time_ms', 'uptime_seconds', 'memory_usage_mb'
                ]

                missing_fields = [field for field in required_fields if field not in metrics]
                if not missing_fields:
                    self.log_test_result(
                        "WAF Metrics",
                        True,
                        f"Metrics collected: {metrics['total_requests']} requests, "
                        f"{metrics['anomalies_detected']} anomalies, "
                        f"avg time: {metrics['avg_processing_time_ms']:.2f}ms",
                        metrics
                    )
                    return True
                else:
                    self.log_test_result(
                        "WAF Metrics",
                        False,
                        f"Missing metrics fields: {missing_fields}",
                        metrics
                    )
            else:
                self.log_test_result(
                    "WAF Metrics",
                    False,
                    f"Metrics endpoint returned status {response.status_code}"
                )
        except Exception as e:
            self.log_test_result(
                "WAF Metrics",
                False,
                f"Failed to get metrics: {e}"
            )
        return False

    def test_nginx_proxy_health(self) -> bool:
        """Test Nginx proxy health"""
        try:
            response = requests.get(f"{self.nginx_url}/health", timeout=5)
            if response.status_code == 200 and "healthy" in response.text.lower():
                self.log_test_result(
                    "Nginx Proxy Health",
                    True,
                    "Nginx proxy is responding to health checks"
                )
                return True
            else:
                self.log_test_result(
                    "Nginx Proxy Health",
                    False,
                    f"Nginx health check failed: status {response.status_code}, response: {response.text}"
                )
        except Exception as e:
            self.log_test_result(
                "Nginx Proxy Health",
                False,
                f"Failed to connect to Nginx: {e}"
            )
        return False

    def test_nginx_waf_integration(self) -> bool:
        """Test Nginx-WAF integration by making requests through Nginx"""
        test_cases = [
            {
                'name': 'Normal request through Nginx',
                'path': '/api/test/normal',
                'params': {'param': 'value'},
                'expect_blocked': False
            },
            {
                'name': 'SQL injection through Nginx',
                'path': '/api/test/users',
                'params': {'id': "1' OR '1'='1"},
                'expect_blocked': True
            }
        ]

        all_passed = True

        for test_case in test_cases:
            try:
                url = f"{self.nginx_url}{test_case['path']}"
                response = requests.get(url, params=test_case['params'], timeout=10)

                is_blocked = response.status_code == 403

                if is_blocked == test_case['expect_blocked']:
                    status_desc = "blocked" if is_blocked else "allowed"
                    self.log_test_result(
                        f"Nginx-WAF Integration - {test_case['name']}",
                        True,
                        f"Request correctly {status_desc} (status: {response.status_code})",
                        {
                            'status_code': response.status_code,
                            'headers': dict(response.headers),
                            'expected_blocked': test_case['expect_blocked'],
                            'actual_blocked': is_blocked
                        }
                    )
                else:
                    self.log_test_result(
                        f"Nginx-WAF Integration - {test_case['name']}",
                        False,
                        f"Expected {'blocked' if test_case['expect_blocked'] else 'allowed'}, "
                        f"but got status {response.status_code}",
                        {
                            'expected_blocked': test_case['expect_blocked'],
                            'actual_status': response.status_code,
                            'response': response.text[:200]
                        }
                    )
                    all_passed = False

            except Exception as e:
                self.log_test_result(
                    f"Nginx-WAF Integration - {test_case['name']}",
                    False,
                    f"Request failed: {e}"
                )
                all_passed = False

        return all_passed

    def test_threshold_update(self) -> bool:
        """Test threshold update functionality"""
        try:
            # Get current config
            response = requests.get(f"{self.waf_url}/config", timeout=5)
            if response.status_code != 200:
                self.log_test_result("Threshold Update", False, "Failed to get current config")
                return False

            original_threshold = response.json()['threshold']

            # Update threshold
            new_threshold = 0.3
            update_response = requests.post(
                f"{self.waf_url}/update-threshold",
                json={'threshold': new_threshold},
                timeout=5
            )

            if update_response.status_code != 200:
                self.log_test_result(
                    "Threshold Update",
                    False,
                    f"Failed to update threshold: status {update_response.status_code}"
                )
                return False

            # Verify threshold was updated
            config_response = requests.get(f"{self.waf_url}/config", timeout=5)
            if config_response.status_code == 200:
                current_threshold = config_response.json()['threshold']
                if abs(current_threshold - new_threshold) < 0.001:
                    self.log_test_result(
                        "Threshold Update",
                        True,
                        f"Threshold successfully updated from {original_threshold} to {new_threshold}"
                    )

                    # Restore original threshold
                    requests.post(
                        f"{self.waf_url}/update-threshold",
                        json={'threshold': original_threshold},
                        timeout=5
                    )
                    return True
                else:
                    self.log_test_result(
                        "Threshold Update",
                        False,
                        f"Threshold not updated correctly. Expected {new_threshold}, got {current_threshold}"
                    )
            else:
                self.log_test_result("Threshold Update", False, "Failed to verify threshold update")

        except Exception as e:
            self.log_test_result("Threshold Update", False, f"Threshold update test failed: {e}")

        return False

    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        logger.info("Starting WAF Integration End-to-End Tests")
        logger.info("=" * 50)

        tests = [
            ("WAF Service Health", self.test_waf_service_health),
            ("WAF Anomaly Detection", self.test_waf_anomaly_detection),
            ("WAF Metrics", self.test_waf_metrics),
            ("Threshold Update", self.test_threshold_update),
            ("Nginx Proxy Health", self.test_nginx_proxy_health),
            ("Nginx-WAF Integration", self.test_nginx_waf_integration),
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\nRunning test: {test_name}")
            logger.info("-" * 30)
            if test_func():
                passed_tests += 1

        logger.info("\n" + "=" * 50)
        logger.info(f"Test Results: {passed_tests}/{total_tests} tests passed")

        success_rate = (passed_tests / total_tests) * 100
        if success_rate >= 80:
            logger.info(f"Test Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}% success rate)")
        else:
            logger.warning(f"Test Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}% success rate)")
        return passed_tests == total_tests

    def generate_report(self) -> Dict:
        """Generate test report"""
        passed = sum(1 for result in self.test_results if result['success'])
        total = len(self.test_results)

        return {
            'summary': {
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': total - passed,
                'success_rate': (passed / total * 100) if total > 0 else 0,
                'timestamp': time.time()
            },
            'results': self.test_results,
            'configuration': {
                'waf_url': self.waf_url,
                'nginx_url': self.nginx_url
            }
        }


def main():
    parser = argparse.ArgumentParser(description="WAF Integration End-to-End Tester")
    parser.add_argument("--waf-url", default="http://127.0.0.1:8000", help="WAF service URL")
    parser.add_argument("--nginx-url", default="http://127.0.0.1", help="Nginx proxy URL")
    parser.add_argument("--output", help="Output JSON report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create tester
    tester = WAFIntegrationTester(
        waf_url=args.waf_url,
        nginx_url=args.nginx_url
    )

    # Run tests
    success = tester.run_all_tests()

    # Generate report
    report = tester.generate_report()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.output}")

    # Print summary
    print("\nTest Summary:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed_tests']}")
    print(f"  Failed: {report['summary']['failed_tests']}")
    print(".1f")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()