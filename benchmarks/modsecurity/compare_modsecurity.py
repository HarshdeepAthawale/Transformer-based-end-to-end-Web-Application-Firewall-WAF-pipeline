#!/usr/bin/env python3
"""
ModSecurity CRS v4 vs Transformer WAF — side-by-side comparison.

Sends all attack payloads through both engines and produces:
  - Per-category detection rate comparison
  - Latency comparison (p50/p95)
  - False positive rate comparison
  - CSV output for further analysis

Usage:
    # 1. Start ModSecurity container
    docker build -t modsec-crs benchmarks/modsecurity/
    docker run -d --name modsec-bench -p 8880:80 modsec-crs

    # 2. Run comparison (WAF model runs locally, ModSec via Docker)
    python benchmarks/modsecurity/compare_modsecurity.py

    # 3. Cleanup
    docker stop modsec-bench && docker rm modsec-bench
"""
import csv
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "attack_tests"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODSEC_URL = os.environ.get("MODSEC_URL", "http://localhost:8880")
WAF_MODEL_PATH = str(PROJECT_ROOT / "models" / "waf-distilbert")
ONNX_PATH = str(PROJECT_ROOT / "models" / "waf-distilbert.onnx")
OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "modsecurity"

# Categories: (module_name, variable_name, category_label, is_attack)
CATEGORIES = [
    ("01_sql_injection", "SQL_INJECTION_ATTACKS", "SQL Injection", True),
    ("02_xss_attacks", "XSS_ATTACKS", "XSS", True),
    ("03_command_injection", "COMMAND_INJECTION_ATTACKS", "Command Injection", True),
    ("04_path_traversal", "PATH_TRAVERSAL_ATTACKS", "Path Traversal", True),
    ("05_xxe_attacks", "XXE_ATTACKS", "XXE", True),
    ("06_ssrf_attacks", "SSRF_ATTACKS", "SSRF", True),
    ("07_header_injection", "HEADER_INJECTION_ATTACKS", "Header Injection", True),
    ("08_ldap_xpath_injection", "INJECTION_ATTACKS", "LDAP/XPath/Template", True),
    ("09_dos_patterns", "DOS_ATTACKS", "DoS Patterns", True),
    ("10_mixed_blended", "MIXED_ATTACKS", "Mixed/Blended", True),
    ("11_fp_regression", "BENIGN_REQUESTS", "Benign (FP test)", False),
]


# ---------------------------------------------------------------------------
# Payload loading
# ---------------------------------------------------------------------------

def load_payloads() -> List[Tuple[Dict[str, Any], str, bool]]:
    """Load all payloads and return list of (payload_dict, category, is_attack)."""
    results = []
    for mod_name, var_name, category, is_attack in CATEGORIES:
        try:
            mod = importlib.import_module(mod_name)
            payloads = getattr(mod, var_name, [])
            for p in payloads:
                results.append((p, category, is_attack))
        except Exception as e:
            print(f"  WARNING: Could not load {mod_name}: {e}")
    return results


def payload_to_url(payload: Dict[str, Any], base_url: str) -> Tuple[str, str, dict]:
    """Convert payload dict to (url, method, kwargs) for requests."""
    path = payload.get("path", "/test")
    query = payload.get("query", {})
    body = payload.get("body", None)
    headers = payload.get("headers", {})
    method = "POST" if body else "GET"

    # Build query string
    if isinstance(query, dict) and query:
        # Handle nested dicts (e.g. MongoDB-style) by converting to string
        flat_query = {}
        for k, v in query.items():
            flat_query[k] = json.dumps(v) if isinstance(v, dict) else str(v)
        qs = urlencode(flat_query)
        url = f"{base_url}{path}?{qs}"
    else:
        url = f"{base_url}{path}"

    kwargs = {"timeout": 5}
    if headers:
        kwargs["headers"] = {
            k: str(v) for k, v in headers.items() if isinstance(k, str)
        }
    if body:
        if isinstance(body, dict):
            kwargs["json"] = body
        else:
            kwargs["data"] = str(body)

    return url, method, kwargs


def payload_to_request_text(payload: Dict[str, Any]) -> str:
    """Convert payload dict to the text format the ML model expects."""
    path = payload.get("path", "/test")
    query = payload.get("query", {})
    body = payload.get("body", None)
    headers = payload.get("headers", {})
    method = "POST" if body else "GET"

    if isinstance(query, dict) and query:
        flat_query = {}
        for k, v in query.items():
            flat_query[k] = json.dumps(v) if isinstance(v, dict) else str(v)
        qs = urlencode(flat_query)
        full_path = f"{path}?{qs}"
    else:
        full_path = path

    lines = [f"{method} {full_path} HTTP/1.1"]
    if isinstance(headers, dict):
        for k, v in headers.items():
            if isinstance(k, str):
                lines.append(f"{k}: {v}")
    if body:
        lines.append("")
        if isinstance(body, dict):
            lines.append(json.dumps(body))
        else:
            lines.append(str(body))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ModSecurity testing
# ---------------------------------------------------------------------------

def test_modsecurity(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send payload to ModSecurity and return result."""
    url, method, kwargs = payload_to_url(payload, MODSEC_URL)
    t0 = time.perf_counter()
    try:
        if method == "POST":
            resp = requests.post(url, **kwargs)
        else:
            resp = requests.get(url, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        # ModSecurity returns 403 for blocked requests
        blocked = resp.status_code == 403
        return {
            "blocked": blocked,
            "status_code": resp.status_code,
            "latency_ms": latency_ms,
        }
    except requests.exceptions.RequestException as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "blocked": False,
            "status_code": 0,
            "latency_ms": latency_ms,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# ML model testing
# ---------------------------------------------------------------------------

def load_ml_classifier():
    """Load the ONNX classifier (or PyTorch fallback)."""
    onnx_file = Path(ONNX_PATH)
    if onnx_file.exists():
        try:
            from backend.ml.onnx_classifier import ONNXWAFClassifier
            clf = ONNXWAFClassifier(
                model_path=WAF_MODEL_PATH,
                onnx_path=ONNX_PATH,
            )
            if clf.is_loaded:
                print("  Using ONNX classifier")
                return clf
        except Exception as e:
            print(f"  ONNX load failed: {e}")

    # Fallback to PyTorch
    from backend.ml.classifier import WAFClassifier
    clf = WAFClassifier(model_path=WAF_MODEL_PATH)
    print("  Using PyTorch classifier")
    return clf


def test_ml_model(classifier, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run payload through the ML classifier."""
    text = payload_to_request_text(payload)
    t0 = time.perf_counter()
    result = classifier.classify(text)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "blocked": result.get("is_malicious", False),
        "confidence": result.get("confidence", 0.0),
        "malicious_score": result.get("malicious_score", 0.0),
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def run_comparison():
    print("\n=== ModSecurity CRS v4 vs Transformer WAF Comparison ===\n")

    # Check ModSecurity is reachable
    print("Checking ModSecurity...")
    try:
        r = requests.get(f"{MODSEC_URL}/", timeout=5)
        print(f"  ModSecurity reachable (status={r.status_code})")
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: ModSecurity not reachable at {MODSEC_URL}")
        print("  Start it with:")
        print("    docker build -t modsec-crs benchmarks/modsecurity/")
        print("    docker run -d --name modsec-bench -p 8880:80 modsec-crs")
        sys.exit(1)

    # Load ML classifier
    print("Loading ML classifier...")
    classifier = load_ml_classifier()

    # Load payloads
    print("Loading payloads...")
    payloads = load_payloads()
    print(f"  Loaded {len(payloads)} payloads across {len(CATEGORIES)} categories\n")

    # Run comparison
    results_per_category = defaultdict(lambda: {
        "modsec_blocked": 0,
        "modsec_missed": 0,
        "ml_blocked": 0,
        "ml_missed": 0,
        "modsec_latencies": [],
        "ml_latencies": [],
        "total": 0,
        "is_attack": True,
        "details": [],
    })

    for i, (payload, category, is_attack) in enumerate(payloads):
        name = payload.get("name", f"payload_{i}")
        cat_data = results_per_category[category]
        cat_data["is_attack"] = is_attack
        cat_data["total"] += 1

        # Test ModSecurity
        modsec_result = test_modsecurity(payload)
        cat_data["modsec_latencies"].append(modsec_result["latency_ms"])

        # Test ML model
        ml_result = test_ml_model(classifier, payload)
        cat_data["ml_latencies"].append(ml_result["latency_ms"])

        if is_attack:
            if modsec_result["blocked"]:
                cat_data["modsec_blocked"] += 1
            else:
                cat_data["modsec_missed"] += 1
            if ml_result["blocked"]:
                cat_data["ml_blocked"] += 1
            else:
                cat_data["ml_missed"] += 1
        else:
            # For benign: "blocked" = false positive
            if modsec_result["blocked"]:
                cat_data["modsec_blocked"] += 1  # FP for benign
            else:
                cat_data["modsec_missed"] += 1   # Correct pass
            if ml_result["blocked"]:
                cat_data["ml_blocked"] += 1       # FP for benign
            else:
                cat_data["ml_missed"] += 1        # Correct pass

        cat_data["details"].append({
            "name": name,
            "category": category,
            "is_attack": is_attack,
            "modsec_blocked": modsec_result["blocked"],
            "modsec_status": modsec_result.get("status_code", 0),
            "modsec_latency_ms": round(modsec_result["latency_ms"], 2),
            "ml_blocked": ml_result["blocked"],
            "ml_score": round(ml_result.get("malicious_score", 0.0), 4),
            "ml_latency_ms": round(ml_result["latency_ms"], 2),
        })

        # Progress
        if (i + 1) % 50 == 0 or i == len(payloads) - 1:
            print(f"  Processed {i + 1}/{len(payloads)} payloads...")

    return results_per_category


def print_results(results_per_category: dict):
    """Print comparison table and save outputs."""
    print("\n" + "=" * 100)
    print("COMPARISON RESULTS: ModSecurity CRS v4 vs Transformer WAF")
    print("=" * 100)

    # Header
    header = (
        f"{'Category':<22} {'Count':>5} "
        f"{'ModSec Det%':>11} {'ML Det%':>9} "
        f"{'ModSec p50':>10} {'ML p50':>8} "
        f"{'Winner':>8}"
    )
    print(header)
    print("-" * 100)

    total_modsec_tp = 0
    total_modsec_fp = 0
    total_ml_tp = 0
    total_ml_fp = 0
    total_attacks = 0
    total_benign = 0

    csv_rows = []

    for cat_label in [c[2] for c in CATEGORIES]:
        data = results_per_category.get(cat_label)
        if not data or data["total"] == 0:
            continue

        is_attack = data["is_attack"]
        total = data["total"]

        if is_attack:
            modsec_rate = data["modsec_blocked"] / total * 100
            ml_rate = data["ml_blocked"] / total * 100
            total_modsec_tp += data["modsec_blocked"]
            total_ml_tp += data["ml_blocked"]
            total_attacks += total
            winner = "ML" if ml_rate > modsec_rate else ("ModSec" if modsec_rate > ml_rate else "Tie")
        else:
            # For benign: lower blocked = better (fewer false positives)
            modsec_rate = (1 - data["modsec_blocked"] / total) * 100  # pass rate
            ml_rate = (1 - data["ml_blocked"] / total) * 100
            total_modsec_fp += data["modsec_blocked"]
            total_ml_fp += data["ml_blocked"]
            total_benign += total
            winner = "ML" if ml_rate > modsec_rate else ("ModSec" if modsec_rate > ml_rate else "Tie")

        modsec_p50 = percentile(data["modsec_latencies"], 50)
        ml_p50 = percentile(data["ml_latencies"], 50)

        rate_label = "Det%" if is_attack else "Pass%"

        print(
            f"{cat_label:<22} {total:>5} "
            f"{modsec_rate:>10.1f}% {ml_rate:>8.1f}% "
            f"{modsec_p50:>8.1f}ms {ml_p50:>6.1f}ms "
            f"{winner:>8}"
        )

        csv_rows.append({
            "category": cat_label,
            "is_attack": is_attack,
            "count": total,
            "modsec_detection_pct": round(modsec_rate, 1) if is_attack else None,
            "ml_detection_pct": round(ml_rate, 1) if is_attack else None,
            "modsec_pass_pct": round(modsec_rate, 1) if not is_attack else None,
            "ml_pass_pct": round(ml_rate, 1) if not is_attack else None,
            "modsec_p50_ms": round(modsec_p50, 2),
            "ml_p50_ms": round(ml_p50, 2),
            "winner": winner,
        })

    # Summary
    print("-" * 100)

    if total_attacks > 0:
        modsec_overall = total_modsec_tp / total_attacks * 100
        ml_overall = total_ml_tp / total_attacks * 100
        print(f"\n  Overall Detection Rate (attacks only):")
        print(f"    ModSecurity CRS v4: {modsec_overall:.1f}% ({total_modsec_tp}/{total_attacks})")
        print(f"    Transformer WAF:    {ml_overall:.1f}% ({total_ml_tp}/{total_attacks})")

    if total_benign > 0:
        modsec_fpr = total_modsec_fp / total_benign * 100
        ml_fpr = total_ml_fp / total_benign * 100
        print(f"\n  False Positive Rate (benign requests blocked):")
        print(f"    ModSecurity CRS v4: {modsec_fpr:.1f}% ({total_modsec_fp}/{total_benign})")
        print(f"    Transformer WAF:    {ml_fpr:.1f}% ({total_ml_fp}/{total_benign})")

    # Save CSV
    csv_path = OUTPUT_DIR / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "is_attack", "count",
            "modsec_detection_pct", "ml_detection_pct",
            "modsec_pass_pct", "ml_pass_pct",
            "modsec_p50_ms", "ml_p50_ms", "winner",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  CSV saved: {csv_path}")

    # Save detailed JSON
    json_path = OUTPUT_DIR / "comparison_details.json"
    all_details = []
    for cat_label in [c[2] for c in CATEGORIES]:
        data = results_per_category.get(cat_label)
        if data:
            all_details.extend(data["details"])
    with open(json_path, "w") as f:
        json.dump(all_details, f, indent=2)
    print(f"  Details saved: {json_path}")

    # Save summary JSON
    summary = {
        "total_payloads": total_attacks + total_benign,
        "total_attacks": total_attacks,
        "total_benign": total_benign,
        "modsec_true_positives": total_modsec_tp,
        "modsec_false_positives": total_modsec_fp,
        "ml_true_positives": total_ml_tp,
        "ml_false_positives": total_ml_fp,
        "modsec_detection_rate": round(total_modsec_tp / total_attacks * 100, 1) if total_attacks else 0,
        "ml_detection_rate": round(total_ml_tp / total_attacks * 100, 1) if total_attacks else 0,
        "modsec_fpr": round(total_modsec_fp / total_benign * 100, 1) if total_benign else 0,
        "ml_fpr": round(total_ml_fp / total_benign * 100, 1) if total_benign else 0,
    }
    summary_path = OUTPUT_DIR / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    results = run_comparison()
    print_results(results)
