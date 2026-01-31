#!/usr/bin/env python3
"""
Master Attack Test Suite Runner
Runs all 10 attack test suites and generates a comprehensive report
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Test scripts in order
TEST_SCRIPTS = [
    ("01_sql_injection.py", "SQL Injection"),
    ("02_xss_attacks.py", "Cross-Site Scripting (XSS)"),
    ("03_command_injection.py", "Command Injection"),
    ("04_path_traversal.py", "Path Traversal / LFI / RFI"),
    ("05_xxe_attacks.py", "XML External Entity (XXE)"),
    ("06_ssrf_attacks.py", "Server-Side Request Forgery (SSRF)"),
    ("07_header_injection.py", "HTTP Header Injection / CRLF"),
    ("08_ldap_xpath_injection.py", "LDAP / XPATH / Template Injection"),
    ("09_dos_patterns.py", "DoS/DDoS Patterns"),
    ("10_mixed_blended.py", "Mixed & Blended Attacks"),
]


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██╗    ██╗ █████╗ ███████╗    ████████╗███████╗███████╗████████╗        ║
║     ██║    ██║██╔══██╗██╔════╝    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝        ║
║     ██║ █╗ ██║███████║█████╗         ██║   █████╗  ███████╗   ██║           ║
║     ██║███╗██║██╔══██║██╔══╝         ██║   ██╔══╝  ╚════██║   ██║           ║
║     ╚███╔███╔╝██║  ██║██║            ██║   ███████╗███████║   ██║           ║
║      ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝            ╚═╝   ╚══════╝╚══════╝   ╚═╝           ║
║                                                                              ║
║           Transformer-based Web Application Firewall Test Suite              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_waf_service():
    """Check if WAF service is available"""
    import requests
    try:
        response = requests.get("http://localhost:3001/health", timeout=5)
        if response.status_code == 200:
            print("\033[92m✓ WAF Backend is running\033[0m")
            return True
    except:
        pass

    print("\033[91m✗ WAF Backend is not available at http://localhost:3001\033[0m")
    print("  Please start the backend first:")
    print("  docker-compose -f docker-compose.full-test.yml up -d")
    return False


def check_model_status():
    """Check if WAF model is loaded"""
    import requests
    try:
        response = requests.get("http://localhost:3001/api/waf/model-info", timeout=5)
        data = response.json()
        if data.get("data", {}).get("model_loaded"):
            threshold = data.get("data", {}).get("threshold", 0.5)
            print(f"\033[92m✓ WAF Model is loaded (threshold: {threshold})\033[0m")
            return True
        else:
            print("\033[93m⚠ WAF Model not loaded - detection may not work\033[0m")
            return False
    except Exception as e:
        print(f"\033[91m✗ Could not check model status: {e}\033[0m")
        return False


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text"""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def run_test(script_path: Path, name: str) -> dict:
    """Run a single test script and capture results"""
    print(f"\n{'─' * 70}")
    print(f"Running: {name}")
    print(f"{'─' * 70}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        elapsed = time.time() - start_time

        # Parse output for summary
        output = result.stdout
        lines = output.split('\n')

        # Find summary lines
        blocked = 0
        missed = 0
        errors = 0
        total = 0

        for line in lines:
            # Strip ANSI codes for parsing
            clean_line = strip_ansi(line)
            if 'Blocked:' in clean_line:
                try:
                    blocked = int(clean_line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'Missed:' in clean_line:
                try:
                    missed = int(clean_line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'Errors:' in clean_line:
                try:
                    errors = int(clean_line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'Total Tests:' in clean_line:
                try:
                    total = int(clean_line.split(':')[1].strip())
                except:
                    pass

        # Print filtered output (just [BLOCKED] and [MISSED] lines)
        for line in lines:
            if '[BLOCKED]' in line or '[MISSED]' in line or '[ERROR]' in line:
                print(line)

        return {
            "name": name,
            "total": total,
            "blocked": blocked,
            "missed": missed,
            "errors": errors,
            "elapsed": elapsed,
            "success": True
        }

    except subprocess.TimeoutExpired:
        return {"name": name, "success": False, "error": "Timeout"}
    except Exception as e:
        return {"name": name, "success": False, "error": str(e)}


def generate_report(results: list):
    """Generate final summary report"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "FINAL TEST REPORT" + " " * 36 + "║")
    print("╠" + "═" * 78 + "╣")

    total_tests = 0
    total_blocked = 0
    total_missed = 0
    total_errors = 0
    total_time = 0

    # Print individual suite results
    print("║ {:<35} {:>8} {:>8} {:>8} {:>8} ║".format(
        "Test Suite", "Total", "Blocked", "Missed", "Rate"))
    print("╠" + "─" * 78 + "╣")

    for r in results:
        if r.get("success"):
            rate = r["blocked"] / (r["total"] - r["errors"]) * 100 if (r["total"] - r["errors"]) > 0 else 0

            # Color code based on rate
            if rate >= 80:
                color = "\033[92m"  # Green
            elif rate >= 50:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red

            print("║ {:<35} {:>8} {:>8} {:>8} {}  {:>5.1f}%\033[0m ║".format(
                r["name"][:35], r["total"], r["blocked"], r["missed"], color, rate))

            total_tests += r["total"]
            total_blocked += r["blocked"]
            total_missed += r["missed"]
            total_errors += r["errors"]
            total_time += r["elapsed"]
        else:
            print("║ {:<35} \033[91m{:>44}\033[0m ║".format(
                r["name"][:35], f"FAILED: {r.get('error', 'Unknown')}"[:44]))

    print("╠" + "═" * 78 + "╣")

    # Overall summary
    overall_rate = total_blocked / (total_tests - total_errors) * 100 if (total_tests - total_errors) > 0 else 0

    if overall_rate >= 80:
        color = "\033[92m"
        status = "EXCELLENT"
    elif overall_rate >= 60:
        color = "\033[93m"
        status = "GOOD"
    elif overall_rate >= 40:
        color = "\033[93m"
        status = "FAIR"
    else:
        color = "\033[91m"
        status = "NEEDS IMPROVEMENT"

    print("║ {:<35} {:>8} {:>8} {:>8} {}  {:>5.1f}%\033[0m ║".format(
        "OVERALL TOTAL", total_tests, total_blocked, total_missed, color, overall_rate))
    print("╠" + "═" * 78 + "╣")
    print("║" + " " * 78 + "║")
    print("║   Total Attack Payloads Tested: {:>6}                                     ║".format(total_tests))
    print("║   {}\033[92mBlocked (Detected):         {:>6}\033[0m                                     ║".format("", total_blocked))
    print("║   {}\033[91mMissed (Not Detected):      {:>6}\033[0m                                     ║".format("", total_missed))
    print("║   {}\033[93mErrors:                     {:>6}\033[0m                                     ║".format("", total_errors))
    print("║" + " " * 78 + "║")
    print("║   Overall Detection Rate: {}{:>6.1f}%\033[0m                                        ║".format(color, overall_rate))
    print("║   Status: {}{}                                                       \033[0m║".format(color, status.ljust(20)))
    print("║   Total Test Time: {:>6.1f} seconds                                          ║".format(total_time))
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")

    # Recommendations
    print("\n\033[1mRecommendations:\033[0m")
    if overall_rate < 60:
        print("  • Consider retraining the model with more attack samples")
        print("  • Review missed attack patterns and add them to training data")
    if total_errors > 0:
        print(f"  • {total_errors} tests had errors - check WAF service stability")
    if overall_rate >= 80:
        print("  • WAF is performing well! Continue monitoring for new attack patterns")

    return overall_rate


def main():
    print_banner()

    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECKS")
    print("=" * 70)

    if not check_waf_service():
        sys.exit(1)

    check_model_status()

    # Get script directory
    script_dir = Path(__file__).parent

    print("\n" + "=" * 70)
    print("STARTING ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Running {len(TEST_SCRIPTS)} test categories...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    for i, (script_name, display_name) in enumerate(TEST_SCRIPTS):
        script_path = script_dir / script_name
        if script_path.exists():
            # Add delay between tests to let the WAF backend recover
            # Longer delays before header injection and LDAP tests
            if i > 0:
                delay = 5 if script_name in ('07_header_injection.py', '08_ldap_xpath_injection.py') else 2
                time.sleep(delay)
            result = run_test(script_path, display_name)
            results.append(result)
        else:
            print(f"\n\033[91m✗ Script not found: {script_name}\033[0m")
            results.append({"name": display_name, "success": False, "error": "Script not found"})

    # Generate final report
    overall_rate = generate_report(results)

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit with appropriate code
    if overall_rate >= 60:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
