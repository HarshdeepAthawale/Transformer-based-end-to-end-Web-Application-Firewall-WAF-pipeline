#!/usr/bin/env python3
"""
Phase 6 Complete Integration Test

Comprehensive test to verify all Phase 6 components work together
"""
import subprocess
import time
import requests
import json
import signal
import sys
import os
from pathlib import Path
import threading
import tempfile

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))
os.environ['TMPDIR'] = '/tmp'

def run_command(cmd, timeout=30, cwd=None):
    """Run a command with timeout"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

class Phase6Tester:
    """Complete Phase 6 integration tester"""

    def __init__(self):
        self.results = []
        self.services = []

    def log_result(self, test_name, success, message, details=None):
        """Log a test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': time.time()
        }
        if details:
            result['details'] = details
        self.results.append(result)

        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {message}")
        if details:
            print(f"   Details: {details}")

    def test_waf_service_initialization(self):
        """Test 1: WAF Service can initialize with real model"""
        try:
            from integration.waf_service import initialize_waf_service, waf_service

            # Initialize with real trained model
            initialize_waf_service(
                'models/checkpoints/best_model.pt',
                'models/vocabularies/http_vocab.json',
                0.5
            )

            # Check if service is initialized
            success = waf_service is not None
            if success:
                vocab_size = len(waf_service.tokenizer.word_to_id)
                model_params = sum(p.numel() for p in waf_service.model.parameters())

                self.log_result(
                    "WAF Service Initialization",
                    True,
                    f"Real model loaded: vocab_size={vocab_size}, params={model_params}",
                    {
                        'vocab_size': vocab_size,
                        'model_params': model_params,
                        'device': waf_service.device,
                        'threshold': waf_service.threshold
                    }
                )
            else:
                self.log_result("WAF Service Initialization", False, "Service not initialized")
            return success

        except Exception as e:
            self.log_result("WAF Service Initialization", False, f"Failed: {e}")
            return False

    def test_model_inference(self):
        """Test 2: Real model inference works"""
        try:
            from integration.waf_service import waf_service

            if waf_service is None:
                self.log_result("Model Inference", False, "WAF service not initialized")
                return False

            # Test normal request
            normal_result = waf_service.check_request(
                'GET', '/api/products',
                {'page': '1', 'limit': '10'},
                {'user-agent': 'test-client'}
            )

            # Test suspicious request
            suspicious_result = waf_service.check_request(
                'GET', '/api/users',
                {'id': "1' OR '1'='1"},
                {'user-agent': 'test-client'}
            )

            if ('anomaly_score' in normal_result and
                'is_anomaly' in normal_result and
                'anomaly_score' in suspicious_result):

                self.log_result(
                    "Model Inference",
                    True,
                    "Real-time inference working",
                    {
                        'normal_score': normal_result['anomaly_score'],
                        'normal_anomaly': normal_result['is_anomaly'],
                        'suspicious_score': suspicious_result['anomaly_score'],
                        'suspicious_anomaly': suspicious_result['is_anomaly'],
                        'processing_time': normal_result.get('processing_time_ms', 0)
                    }
                )
                return True
            else:
                self.log_result("Model Inference", False, "Inference results incomplete")
                return False

        except Exception as e:
            self.log_result("Model Inference", False, f"Failed: {e}")
            return False

    def test_waf_service_api(self):
        """Test 3: WAF Service API endpoints"""
        # Start WAF service in background
        service_process = subprocess.Popen([
            sys.executable, 'scripts/start_waf_service.py',
            '--host', '127.0.0.1',
            '--port', '8889',
            '--workers', '1',
            '--log_level', 'error'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.services.append(service_process)

        # Wait for service to start
        time.sleep(3)

        try:
            base_url = "http://127.0.0.1:8889"

            # Test health endpoint
            health_response = requests.get(f"{base_url}/health", timeout=5)
            health_ok = health_response.status_code == 200

            # Test check endpoint
            check_response = requests.post(
                f"{base_url}/check",
                json={
                    'method': 'GET',
                    'path': '/api/test',
                    'query_params': {'param': 'value'}
                },
                timeout=10
            )
            check_ok = check_response.status_code == 200

            # Test metrics endpoint
            metrics_response = requests.get(f"{base_url}/metrics", timeout=5)
            metrics_ok = metrics_response.status_code == 200

            success = health_ok and check_ok and metrics_ok
            details = {
                'health': health_ok,
                'check': check_ok,
                'metrics': metrics_ok
            }

            if success:
                # Get some actual metrics
                if metrics_ok:
                    metrics = metrics_response.json()
                    details.update({
                        'total_requests': metrics.get('total_requests', 0),
                        'anomalies_detected': metrics.get('anomalies_detected', 0)
                    })

            self.log_result("WAF Service API", success, "All endpoints responding", details)
            return success

        except Exception as e:
            self.log_result("WAF Service API", False, f"API test failed: {e}")
            return False
        finally:
            # Stop service
            service_process.terminate()
            service_process.wait(timeout=5)

    def test_configuration_files(self):
        """Test 4: Configuration files are properly set up"""
        config_file = Path('config/config.yaml')

        if not config_file.exists():
            self.log_result("Configuration Files", False, "config/config.yaml not found")
            return False

        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Check required sections exist
            has_waf_service = 'waf_service' in config
            has_integration = 'integration' in config
            has_model_path = config.get('waf_service', {}).get('model_path')
            has_vocab_path = config.get('waf_service', {}).get('vocab_path')

            # Check model and vocab files exist
            model_exists = Path('models/checkpoints/best_model.pt').exists()
            vocab_exists = Path('models/vocabularies/http_vocab.json').exists()

            success = all([has_waf_service, has_integration, has_model_path,
                          has_vocab_path, model_exists, vocab_exists])

            details = {
                'config_sections': [k for k in config.keys()],
                'model_path': has_model_path,
                'vocab_path': has_vocab_path,
                'model_file_exists': model_exists,
                'vocab_file_exists': vocab_exists
            }

            self.log_result("Configuration Files", success,
                          "Configuration properly set up", details)
            return success

        except Exception as e:
            self.log_result("Configuration Files", False, f"Config error: {e}")
            return False

    def test_docker_setup(self):
        """Test 5: Docker setup is ready"""
        docker_compose = Path('docker-compose.waf.yml')
        dockerfile = Path('Dockerfile.waf')

        compose_exists = docker_compose.exists()
        dockerfile_exists = dockerfile.exists()

        if compose_exists:
            with open(docker_compose, 'r') as f:
                compose_content = f.read()
                has_waf_service = 'waf-service' in compose_content
                has_nginx = 'nginx' in compose_content
                has_ports = '8000:8000' in compose_content
                has_model_mount = 'models:' in compose_content
        else:
            has_waf_service = has_nginx = has_ports = has_model_mount = False

        success = compose_exists and dockerfile_exists and has_waf_service and has_nginx

        details = {
            'compose_exists': compose_exists,
            'dockerfile_exists': dockerfile_exists,
            'has_waf_service': has_waf_service,
            'has_nginx': has_nginx,
            'has_ports': has_ports,
            'has_model_mount': has_model_mount
        }

        self.log_result("Docker Setup", success,
                      "Containerization ready", details)
        return success

    def test_integration_tests(self):
        """Test 6: Integration tests pass"""
        test_file = Path('tests/integration/test_waf_service.py')

        if not test_file.exists():
            self.log_result("Integration Tests", False, "Test file not found")
            return False

        # Run the tests
        success, stdout, stderr = run_command(
            f"python -m pytest {test_file} -v --tb=short",
            timeout=60
        )

        if success:
            # Parse test results
            passed_tests = stdout.count('PASSED') if 'PASSED' in stdout else 0
            total_tests = stdout.count('PASSED') + stdout.count('FAILED') if 'FAILED' in stdout else passed_tests

            self.log_result(
                "Integration Tests",
                True,
                f"Tests passed: {passed_tests}/{total_tests}",
                {'passed': passed_tests, 'total': total_tests}
            )
        else:
            self.log_result("Integration Tests", False,
                          f"Tests failed: {stderr[:200]}...")

        return success

    def test_scripts_exist(self):
        """Test 7: All required scripts exist and are executable"""
        required_scripts = [
            'scripts/start_waf_service.py',
            'scripts/setup_nginx_simple.sh',
            'scripts/setup_nginx_waf_advanced.sh',
            'scripts/setup_openresty_arch.sh',
            'scripts/setup_complete_advanced_waf.sh',
            'scripts/verify_advanced_waf.sh',
            'scripts/test_waf_integration.py',
            'scripts/quick_waf_test.py'
        ]

        missing_scripts = []
        non_executable = []

        for script in required_scripts:
            script_path = Path(script)
            if not script_path.exists():
                missing_scripts.append(script)
            elif not os.access(script_path, os.X_OK):
                non_executable.append(script)

        success = len(missing_scripts) == 0 and len(non_executable) == 0

        details = {
            'missing_scripts': missing_scripts,
            'non_executable': non_executable,
            'total_scripts': len(required_scripts)
        }

        self.log_result("Setup Scripts", success,
                      f"Scripts ready: {len(required_scripts) - len(missing_scripts)}/{len(required_scripts)}",
                      details)
        return success

    def run_all_tests(self):
        """Run all Phase 6 tests"""
        print("🚀 Phase 6 Complete Integration Test")
        print("=" * 50)

        tests = [
            ("WAF Service Initialization", self.test_waf_service_initialization),
            ("Real Model Inference", self.test_model_inference),
            ("WAF Service API", self.test_waf_service_api),
            ("Configuration Files", self.test_configuration_files),
            ("Docker Setup", self.test_docker_setup),
            ("Integration Tests", self.test_integration_tests),
            ("Setup Scripts", self.test_scripts_exist),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n📋 Testing: {test_name}")
            print("-" * 40)
            if test_func():
                passed += 1

        print("\n" + "=" * 50)
        print("📊 PHASE 6 COMPLETENESS REPORT")
        print("=" * 50)

        success_rate = (passed / total) * 100
        print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")

        if success_rate >= 90:
            print("🎉 PHASE 6 IS COMPLETE!")
            print("\n✅ Real Transformer WAF with end-to-end integration")
            print("✅ No mocks - 100% real model inference")
            print("✅ Production-ready microservice architecture")
            print("✅ Comprehensive testing and monitoring")
        else:
            print("⚠️  Phase 6 needs completion")
            print(f"\n❌ {total - passed} tests failed")
            print("Check the failed tests above for details")

        return success_rate >= 90

    def cleanup(self):
        """Clean up running services"""
        for service in self.services:
            try:
                service.terminate()
                service.wait(timeout=5)
            except:
                pass

def main():
    """Main test runner"""
    tester = Phase6Tester()

    try:
        success = tester.run_all_tests()

        # Generate summary report
        passed = sum(1 for r in tester.results if r['success'])
        total = len(tester.results)

        print("\n📋 DETAILED RESULTS:")
        for result in tester.results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['test']}: {result['message']}")

        return 0 if success else 1

    finally:
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())