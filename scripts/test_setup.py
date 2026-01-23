#!/usr/bin/env python3
"""
Test script to verify Phase 1 environment setup
"""
import requests
import os
import sys
import yaml
import time
from pathlib import Path

def print_status(message, status=True):
    """Print status message with checkmark or X"""
    symbol = "✓" if status else "✗"
    color = "\033[0;32m" if status else "\033[0;31m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {message}")

def test_web_applications():
    """Test if all web applications are accessible"""
    print("\n1. Testing Web Applications:")
    apps = [
        ("http://localhost:8080", "App 1"),
        ("http://localhost:8081", "App 2"),
        ("http://localhost:8082", "App 3")
    ]
    
    all_ok = True
    for url, name in apps:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 404]:  # 404 is OK, means server is running
                print_status(f"{name} ({url}): Status {response.status_code}", True)
            else:
                print_status(f"{name} ({url}): Unexpected status {response.status_code}", False)
                all_ok = False
        except requests.exceptions.ConnectionError:
            print_status(f"{name} ({url}): Connection refused - server not running", False)
            all_ok = False
        except Exception as e:
            print_status(f"{name} ({url}): Error - {e}", False)
            all_ok = False
    
    return all_ok

def test_log_files():
    """Verify log files exist and are readable"""
    print("\n2. Testing Log Files:")
    
    # Read config to get log paths
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    log_paths = []
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'web_server' in config:
                    log_paths.append(config['web_server'].get('log_path'))
                    log_paths.append(config['web_server'].get('error_log_path'))
                if 'applications' in config:
                    for app in config['applications'].values():
                        log_paths.append(app.get('log_path'))
        except Exception as e:
            print(f"Warning: Could not read config: {e}")
    
    # Add default paths
    default_paths = [
        "/var/log/nginx/access.log",
        "/var/log/nginx/error.log",
        "/opt/tomcat9/logs/localhost_access_log.txt",
        "/opt/tomcat9/logs/catalina.out"
    ]
    log_paths.extend(default_paths)
    
    all_ok = True
    for log_path in log_paths:
        if not log_path:
            continue
        if os.path.exists(log_path):
            if os.access(log_path, os.R_OK):
                print_status(f"Log file readable: {log_path}", True)
            else:
                print_status(f"Log file not readable: {log_path}", False)
                all_ok = False
        else:
            print_status(f"Log file not found: {log_path} (may be created on first request)", True)
            # This is OK - logs are created on first request
    
    return all_ok

def test_project_structure():
    """Verify project directory structure exists"""
    print("\n3. Testing Project Structure:")
    
    project_root = Path(__file__).parent.parent
    required_dirs = [
        "src/ingestion",
        "src/parsing",
        "src/tokenization",
        "src/model",
        "src/training",
        "src/inference",
        "src/integration",
        "src/learning",
        "data/raw",
        "data/processed",
        "data/normalized",
        "data/training",
        "data/validation",
        "data/test",
        "models/checkpoints",
        "models/vocabularies",
        "models/deployed",
        "config",
        "tests/unit",
        "tests/integration",
        "tests/performance",
        "logs",
        "scripts"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print_status(f"Directory exists: {dir_path}", True)
        else:
            print_status(f"Directory missing: {dir_path}", False)
            all_ok = False
    
    return all_ok

def test_config_files():
    """Verify configuration files exist"""
    print("\n4. Testing Configuration Files:")
    
    project_root = Path(__file__).parent.parent
    config_files = [
        "config/config.yaml",
        "requirements.txt"
    ]
    
    all_ok = True
    for file_path in config_files:
        full_path = project_root / file_path
        if full_path.exists():
            print_status(f"Config file exists: {file_path}", True)
        else:
            print_status(f"Config file missing: {file_path}", False)
            all_ok = False
    
    return all_ok

def test_python_environment():
    """Test Python environment and dependencies"""
    print("\n5. Testing Python Environment:")
    
    try:
        import torch
        print_status(f"PyTorch installed: {torch.__version__}", True)
    except ImportError:
        print_status("PyTorch not installed", False)
        return False
    
    try:
        import transformers
        print_status(f"Transformers installed: {transformers.__version__}", True)
    except ImportError:
        print_status("Transformers not installed", False)
        return False
    
    try:
        import fastapi
        print_status(f"FastAPI installed: {fastapi.__version__}", True)
    except ImportError:
        print_status("FastAPI not installed", False)
        return False
    
    return True

def generate_test_requests():
    """Generate test requests to populate logs"""
    print("\n6. Generating Test Requests:")
    
    apps = [
        ("http://localhost:8080", "App 1"),
        ("http://localhost:8081", "App 2"),
        ("http://localhost:8082", "App 3")
    ]
    
    for url, name in apps:
        try:
            # GET request
            requests.get(f"{url}/test?param=value1", timeout=2)
            # POST request
            requests.post(f"{url}/api/data", json={"data": "test"}, timeout=2)
            print_status(f"Generated requests to {name}", True)
        except Exception as e:
            print_status(f"Could not generate requests to {name}: {e}", False)

if __name__ == "__main__":
    print("=" * 50)
    print("Phase 1 Environment Setup Test")
    print("=" * 50)
    
    results = []
    
    results.append(("Project Structure", test_project_structure()))
    results.append(("Configuration Files", test_config_files()))
    results.append(("Python Environment", test_python_environment()))
    results.append(("Web Applications", test_web_applications()))
    results.append(("Log Files", test_log_files()))
    
    # Generate test requests
    generate_test_requests()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        print("\nNote: Some failures may be expected if services are not running.")
        print("Run 'bash scripts/setup_phase1.sh' to set up the environment.")
        sys.exit(1)
