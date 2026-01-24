#!/usr/bin/env python3
"""
Platform Verification Script

Verify all components are in place and properly structured
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_file_exists(path: Path, description: str) -> bool:
    """Check if file exists"""
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def check_directory_exists(path: Path, description: str) -> bool:
    """Check if directory exists"""
    exists = path.exists() and path.is_dir()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def verify_phase7():
    """Verify Phase 7 components"""
    print("\n" + "=" * 60)
    print("PHASE 7: Real-Time Non-Blocking Detection")
    print("=" * 60)
    
    components = [
        ("src/inference/async_waf_service.py", "Async WAF Service"),
        ("src/inference/queue_manager.py", "Request Queue Manager"),
        ("src/inference/rate_limiter.py", "Rate Limiter"),
        ("src/inference/optimization.py", "Model Optimization"),
        ("config/inference.yaml", "Inference Configuration"),
        ("scripts/start_async_waf_service.py", "Startup Script"),
    ]
    
    all_exist = True
    for path, desc in components:
        if not check_file_exists(project_root / path, desc):
            all_exist = False
    
    return all_exist


def verify_phase8():
    """Verify Phase 8 components"""
    print("\n" + "=" * 60)
    print("PHASE 8: Continuous Learning & Incremental Updates")
    print("=" * 60)
    
    components = [
        ("src/learning/data_collector.py", "Incremental Data Collector"),
        ("src/learning/fine_tuning.py", "Fine-Tuning Pipeline"),
        ("src/learning/version_manager.py", "Model Version Manager"),
        ("src/learning/validator.py", "Model Validator"),
        ("src/learning/hot_swap.py", "Hot-Swap Manager"),
        ("src/learning/scheduler.py", "Update Scheduler"),
        ("config/learning.yaml", "Learning Configuration"),
        ("scripts/start_continuous_learning.py", "Continuous Learning Script"),
        ("scripts/manual_model_update.py", "Manual Update Script"),
        ("scripts/rollback_model.py", "Rollback Script"),
    ]
    
    all_exist = True
    for path, desc in components:
        if not check_file_exists(project_root / path, desc):
            all_exist = False
    
    return all_exist


def verify_phase9():
    """Verify Phase 9 components"""
    print("\n" + "=" * 60)
    print("PHASE 9: Testing, Validation & Performance Tuning")
    print("=" * 60)
    
    components = [
        ("tests/accuracy/test_detection_accuracy.py", "Accuracy Tests"),
        ("tests/performance/test_throughput.py", "Performance Tests"),
        ("tests/payloads/malicious_payloads.py", "Malicious Payloads"),
        ("scripts/load_test.py", "Load Testing Script"),
        ("scripts/optimize_model.py", "Model Optimization Script"),
        ("scripts/generate_evaluation_report.py", "Report Generator"),
        ("scripts/run_comprehensive_tests.py", "Test Runner"),
        ("config/testing.yaml", "Testing Configuration"),
    ]
    
    all_exist = True
    for path, desc in components:
        if not check_file_exists(project_root / path, desc):
            all_exist = False
    
    return all_exist


def test_payloads():
    """Test malicious payloads loading"""
    print("\n" + "=" * 60)
    print("TESTING: Malicious Payloads")
    print("=" * 60)
    
    try:
        from tests.payloads.malicious_payloads import (
            get_all_malicious_payloads,
            get_payload_count,
            SQL_INJECTION_PAYLOADS,
            XSS_PAYLOADS
        )
        
        all_payloads = get_all_malicious_payloads()
        total_count = get_payload_count()
        
        print(f"✓ Payloads loaded successfully")
        print(f"  Categories: {len(all_payloads)}")
        print(f"  Total payloads: {total_count}")
        print(f"  SQL Injection: {len(SQL_INJECTION_PAYLOADS)} payloads")
        print(f"  XSS: {len(XSS_PAYLOADS)} payloads")
        
        return True
    except Exception as e:
        print(f"✗ Error loading payloads: {e}")
        return False


def test_config_files():
    """Test configuration files"""
    print("\n" + "=" * 60)
    print("TESTING: Configuration Files")
    print("=" * 60)
    
    configs = [
        ("config/config.yaml", "Main Configuration"),
        ("config/inference.yaml", "Inference Configuration"),
        ("config/learning.yaml", "Learning Configuration"),
        ("config/testing.yaml", "Testing Configuration"),
    ]
    
    all_valid = True
    for path, desc in configs:
        config_path = project_root / path
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
                print(f"✓ {desc}: Valid YAML")
            except Exception as e:
                print(f"✗ {desc}: Invalid YAML - {e}")
                all_valid = False
        else:
            print(f"✗ {desc}: Not found")
            all_valid = False
    
    return all_valid


def test_script_syntax():
    """Test script syntax"""
    print("\n" + "=" * 60)
    print("TESTING: Script Syntax")
    print("=" * 60)
    
    scripts = [
        "scripts/start_async_waf_service.py",
        "scripts/start_continuous_learning.py",
        "scripts/manual_model_update.py",
        "scripts/rollback_model.py",
        "scripts/load_test.py",
        "scripts/optimize_model.py",
        "scripts/generate_evaluation_report.py",
        "scripts/run_comprehensive_tests.py",
    ]
    
    all_valid = True
    for script_path in scripts:
        script = project_root / script_path
        if script.exists():
            try:
                with open(script, 'r') as f:
                    compile(f.read(), str(script), 'exec')
                print(f"✓ {script_path}: Valid Python syntax")
            except SyntaxError as e:
                print(f"✗ {script_path}: Syntax error - {e}")
                all_valid = False
            except Exception as e:
                print(f"✗ {script_path}: Error - {e}")
                all_valid = False
        else:
            print(f"✗ {script_path}: Not found")
            all_valid = False
    
    return all_valid


def test_module_structure():
    """Test module structure"""
    print("\n" + "=" * 60)
    print("TESTING: Module Structure")
    print("=" * 60)
    
    modules = [
        ("src/inference", "Inference Module"),
        ("src/learning", "Learning Module"),
        ("tests/accuracy", "Accuracy Tests"),
        ("tests/performance", "Performance Tests"),
        ("tests/payloads", "Payload Tests"),
    ]
    
    all_valid = True
    for module_path, desc in modules:
        module = project_root / module_path
        init_file = module / "__init__.py"
        
        if module.exists() and module.is_dir():
            if init_file.exists():
                print(f"✓ {desc}: Module structure valid")
            else:
                print(f"⚠ {desc}: Missing __init__.py (may be intentional)")
        else:
            print(f"✗ {desc}: Not found")
            all_valid = False
    
    return all_valid


def main():
    """Main verification"""
    print("=" * 60)
    print("PLATFORM VERIFICATION")
    print("=" * 60)
    
    results = {
        'phase7': verify_phase7(),
        'phase8': verify_phase8(),
        'phase9': verify_phase9(),
        'payloads': test_payloads(),
        'configs': test_config_files(),
        'scripts': test_script_syntax(),
        'modules': test_module_structure(),
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {component}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL COMPONENTS VERIFIED")
        print("Platform structure is complete and valid!")
    else:
        print("⚠ SOME COMPONENTS MISSING OR INVALID")
        print("Please check the failures above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
