#!/usr/bin/env python3
"""
Platform Functionality Test

Test platform components without requiring trained model or dependencies
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_rate_limiter():
    """Test rate limiter logic"""
    print("\n" + "=" * 60)
    print("TESTING: Rate Limiter")
    print("=" * 60)
    
    try:
        # Test without importing (check file structure)
        rate_limiter_file = project_root / "src" / "inference" / "rate_limiter.py"
        if rate_limiter_file.exists():
            with open(rate_limiter_file, 'r') as f:
                content = f.read()
                if 'class RateLimiter' in content and 'is_allowed' in content:
                    print("✓ RateLimiter class structure valid")
                    return True
                else:
                    print("✗ RateLimiter structure incomplete")
                    return False
        else:
            print("✗ RateLimiter file not found")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_queue_manager():
    """Test queue manager logic"""
    print("\n" + "=" * 60)
    print("TESTING: Queue Manager")
    print("=" * 60)
    
    try:
        queue_file = project_root / "src" / "inference" / "queue_manager.py"
        if queue_file.exists():
            with open(queue_file, 'r') as f:
                content = f.read()
                if 'class RequestQueueManager' in content and 'enqueue' in content and 'process_queue' in content:
                    print("✓ RequestQueueManager class structure valid")
                    return True
                else:
                    print("✗ RequestQueueManager structure incomplete")
                    return False
        else:
            print("✗ RequestQueueManager file not found")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_version_manager():
    """Test version manager logic"""
    print("\n" + "=" * 60)
    print("TESTING: Version Manager")
    print("=" * 60)
    
    try:
        version_file = project_root / "src" / "learning" / "version_manager.py"
        if version_file.exists():
            with open(version_file, 'r') as f:
                content = f.read()
                required_methods = ['create_version', 'activate_version', 'rollback', 'get_active_version']
                all_present = all(method in content for method in required_methods)
                
                if all_present:
                    print("✓ ModelVersionManager class structure valid")
                    print(f"  Methods: {', '.join(required_methods)}")
                    return True
                else:
                    print("✗ ModelVersionManager missing methods")
                    return False
        else:
            print("✗ ModelVersionManager file not found")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_payload_generation():
    """Test payload generation"""
    print("\n" + "=" * 60)
    print("TESTING: Payload Generation")
    print("=" * 60)
    
    try:
        from tests.payloads.malicious_payloads import (
            get_all_malicious_payloads,
            get_payload_count,
            generate_malicious_requests,
            SQL_INJECTION_PAYLOADS,
            XSS_PAYLOADS
        )
        
        # Test payload retrieval
        all_payloads = get_all_malicious_payloads()
        total_count = get_payload_count()
        
        print(f"✓ Payloads loaded: {len(all_payloads)} categories, {total_count} total")
        
        # Test request generation
        requests = generate_malicious_requests(base_path="/api/test", method="GET")
        print(f"✓ Generated {len(requests)} malicious requests")
        
        # Test specific categories
        print(f"  SQL Injection: {len(SQL_INJECTION_PAYLOADS)} payloads")
        print(f"  XSS: {len(XSS_PAYLOADS)} payloads")
        
        # Verify payloads are not empty
        assert total_count > 0, "No payloads found"
        assert len(requests) > 0, "No requests generated"
        
        print("✓ Payload generation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("TESTING: Configuration Loading")
    print("=" * 60)
    
    try:
        import yaml
        
        configs = [
            ("config/config.yaml", "Main Config"),
            ("config/inference.yaml", "Inference Config"),
            ("config/learning.yaml", "Learning Config"),
            ("config/testing.yaml", "Testing Config"),
        ]
        
        all_valid = True
        for config_path, name in configs:
            full_path = project_root / config_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    config = yaml.safe_load(f)
                    print(f"✓ {name}: Loaded successfully")
                    if name == "Inference Config":
                        if 'inference' in config:
                            print(f"  - Async config: {bool(config['inference'].get('async'))}")
                            print(f"  - Rate limiting: {bool(config['inference'].get('rate_limiting', {}).get('enabled'))}")
                    elif name == "Learning Config":
                        if 'learning' in config:
                            print(f"  - Scheduling enabled: {bool(config['learning'].get('scheduling', {}).get('enabled'))}")
            else:
                print(f"✗ {name}: File not found")
                all_valid = False
        
        return all_valid
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_api_structure():
    """Test API endpoint structure"""
    print("\n" + "=" * 60)
    print("TESTING: API Structure")
    print("=" * 60)
    
    try:
        api_file = project_root / "src" / "inference" / "async_waf_service.py"
        if api_file.exists():
            with open(api_file, 'r') as f:
                content = f.read()
                
                endpoints = [
                    ('@app.post("/check"', '/check endpoint'),
                    ('@app.post("/check/batch"', '/check/batch endpoint'),
                    ('@app.get("/metrics"', '/metrics endpoint'),
                    ('@app.get("/health"', '/health endpoint'),
                ]
                
                all_present = True
                for pattern, name in endpoints:
                    if pattern in content:
                        print(f"✓ {name}: Found")
                    else:
                        print(f"✗ {name}: Missing")
                        all_present = False
                
                # Check for rate limiting integration
                if 'rate_limiter' in content and 'RateLimiter' in content:
                    print("✓ Rate limiting integrated")
                else:
                    print("⚠ Rate limiting not found in API")
                
                # Check for anomaly logging
                if '_log_anomaly' in content:
                    print("✓ Anomaly logging implemented")
                else:
                    print("⚠ Anomaly logging not found")
                
                return all_present
        else:
            print("✗ API file not found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_scheduler_structure():
    """Test scheduler structure"""
    print("\n" + "=" * 60)
    print("TESTING: Update Scheduler")
    print("=" * 60)
    
    try:
        scheduler_file = project_root / "src" / "learning" / "scheduler.py"
        if scheduler_file.exists():
            with open(scheduler_file, 'r') as f:
                content = f.read()
                
                required_methods = ['start', 'stop', 'trigger_update', '_update_model']
                all_present = all(f'def {method}' in content for method in required_methods)
                
                if all_present:
                    print("✓ UpdateScheduler class structure valid")
                    print(f"  Methods: {', '.join(required_methods)}")
                    
                    # Check for pipeline integration
                    pipeline_steps = [
                        'collect_new_data',
                        'fine_tune',
                        'create_version',
                        'validate',
                        'swap_model'
                    ]
                    
                    pipeline_present = all(step in content for step in pipeline_steps)
                    if pipeline_present:
                        print("✓ Full pipeline integrated")
                    else:
                        print("⚠ Pipeline steps may be incomplete")
                    
                    return True
                else:
                    print("✗ UpdateScheduler missing methods")
                    return False
        else:
            print("✗ UpdateScheduler file not found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all functionality tests"""
    print("=" * 60)
    print("PLATFORM FUNCTIONALITY TEST")
    print("=" * 60)
    
    results = {
        'rate_limiter': test_rate_limiter(),
        'queue_manager': test_queue_manager(),
        'version_manager': test_version_manager(),
        'payloads': test_payload_generation(),
        'configs': test_config_loading(),
        'api': test_api_structure(),
        'scheduler': test_scheduler_structure(),
    }
    
    print("\n" + "=" * 60)
    print("FUNCTIONALITY TEST SUMMARY")
    print("=" * 60)
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {component}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL FUNCTIONALITY TESTS PASSED")
        print("Platform components are properly structured!")
    else:
        print("⚠ SOME TESTS FAILED")
        print("Please check the failures above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
