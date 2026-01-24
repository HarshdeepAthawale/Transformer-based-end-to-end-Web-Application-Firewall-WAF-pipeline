#!/usr/bin/env python3
"""
Comprehensive Test Runner

Run all tests (accuracy, performance) and generate evaluation report
"""
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_test_config():
    """Load testing configuration"""
    config_path = project_root / "config" / "testing.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('testing', {})
    
    return {}


def run_accuracy_tests() -> dict:
    """Run accuracy tests"""
    logger.info("=" * 60)
    logger.info("Running Accuracy Tests")
    logger.info("=" * 60)
    
    test_file = project_root / "tests" / "accuracy" / "test_detection_accuracy.py"
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        logger.info("Accuracy tests output:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        logger.error(f"Error running accuracy tests: {e}")
        return {'success': False, 'error': str(e)}


def run_performance_tests() -> dict:
    """Run performance tests"""
    logger.info("=" * 60)
    logger.info("Running Performance Tests")
    logger.info("=" * 60)
    
    test_file = project_root / "tests" / "performance" / "test_throughput.py"
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        logger.info("Performance tests output:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        logger.error(f"Error running performance tests: {e}")
        return {'success': False, 'error': str(e)}


def run_concurrent_tests() -> dict:
    """Run concurrent processing tests"""
    logger.info("=" * 60)
    logger.info("Running Concurrent Processing Tests")
    logger.info("=" * 60)
    
    test_file = project_root / "tests" / "performance" / "test_concurrent.py"
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        logger.info("Concurrent tests output:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        logger.error(f"Error running concurrent tests: {e}")
        return {'success': False, 'error': str(e)}


def parse_pytest_results(stdout: str) -> dict:
    """Parse pytest output to extract results"""
    # Simple parsing - in production, use pytest-json-report or similar
    results = {
        'passed': 0,
        'failed': 0,
        'total': 0
    }
    
    lines = stdout.split('\n')
    for line in lines:
        if 'passed' in line.lower() and 'failed' in line.lower():
            # Try to extract numbers
            import re
            match = re.search(r'(\d+)\s+passed', line)
            if match:
                results['passed'] = int(match.group(1))
            
            match = re.search(r'(\d+)\s+failed', line)
            if match:
                results['failed'] = int(match.group(1))
            
            results['total'] = results['passed'] + results['failed']
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive tests")
    parser.add_argument("--accuracy-only", action="store_true", help="Run only accuracy tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--skip-load-test", action="store_true", help="Skip load testing")
    parser.add_argument("--output", default="reports/comprehensive_test_results.json", help="Output report path")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE TEST SUITE")
    logger.info("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'accuracy_tests': {},
        'performance_tests': {},
        'concurrent_tests': {},
        'load_test': {},
        'summary': {}
    }
    
    # Run accuracy tests
    if not args.performance_only:
        accuracy_result = run_accuracy_tests()
        results['accuracy_tests'] = accuracy_result
        if accuracy_result.get('stdout'):
            parsed = parse_pytest_results(accuracy_result['stdout'])
            results['accuracy_tests'].update(parsed)
    
    # Run performance tests
    if not args.accuracy_only:
        perf_result = run_performance_tests()
        results['performance_tests'] = perf_result
        if perf_result.get('stdout'):
            parsed = parse_pytest_results(perf_result['stdout'])
            results['performance_tests'].update(parsed)
        
        # Run concurrent tests
        concurrent_result = run_concurrent_tests()
        results['concurrent_tests'] = concurrent_result
        if concurrent_result.get('stdout'):
            parsed = parse_pytest_results(concurrent_result['stdout'])
            results['concurrent_tests'].update(parsed)
    
    # Run load test (if service is running)
    if not args.skip_load_test and not args.accuracy_only:
        logger.info("=" * 60)
        logger.info("Running Load Test")
        logger.info("=" * 60)
        
        try:
            load_test_script = project_root / "scripts" / "load_test.py"
            load_result = subprocess.run(
                [sys.executable, str(load_test_script), "--requests", "500", "--concurrent", "50"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(project_root)
            )
            
            results['load_test'] = {
                'success': load_result.returncode == 0,
                'stdout': load_result.stdout,
                'stderr': load_result.stderr
            }
        except subprocess.TimeoutExpired:
            logger.warning("Load test timed out")
            results['load_test'] = {'success': False, 'error': 'timeout'}
        except Exception as e:
            logger.warning(f"Load test skipped: {e}")
            results['load_test'] = {'success': False, 'error': str(e)}
    
    # Calculate summary
    total_passed = (
        results['accuracy_tests'].get('passed', 0) +
        results['performance_tests'].get('passed', 0) +
        results['concurrent_tests'].get('passed', 0)
    )
    total_failed = (
        results['accuracy_tests'].get('failed', 0) +
        results['performance_tests'].get('failed', 0) +
        results['concurrent_tests'].get('failed', 0)
    )
    total_tests = total_passed + total_failed
    
    results['summary'] = {
        'total_tests': total_tests,
        'passed': total_passed,
        'failed': total_failed,
        'success_rate': total_passed / total_tests if total_tests > 0 else 0.0,
        'all_passed': total_failed == 0
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {results['summary']['success_rate']:.2%}")
    print("=" * 60)
    
    # Generate evaluation report
    try:
        from scripts.generate_evaluation_report import generate_evaluation_report
        report_path = output_path.parent / "evaluation_report.json"
        generate_evaluation_report(results, str(report_path), test_type="comprehensive")
    except Exception as e:
        logger.warning(f"Could not generate evaluation report: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if results['summary']['all_passed'] else 1)


if __name__ == "__main__":
    main()
