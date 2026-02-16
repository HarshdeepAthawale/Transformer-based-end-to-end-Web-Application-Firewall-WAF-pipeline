#!/usr/bin/env python3
"""
WAF Stress Test: 1000 requests (600 benign, 400 malicious).
Malicious payloads are intentionally evasive/hard to detect.
Results inform next plan (retrain, threshold, rules, etc.).

Pools are loaded from scripts/data/benign_pool.json and scripts/data/malicious_pool.json.
Override target URL via STRESS_TEST_BASE_URL (default: http://localhost:3001).

Outputs:
  - Console summary with FP rate & detection rate
  - scripts/missed_malicious_samples.json  (missed malicious requests)
  - scripts/false_positive_samples.json    (benign requests wrongly blocked)
"""
import os
import requests
import random
import time
import json as _json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

BASE_URL = os.getenv("STRESS_TEST_BASE_URL", "http://localhost:3001")


def _load_pool(filename: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}. Create it or run from project root.")
    with open(path) as f:
        return _json.load(f)


def get_benign_pool() -> List[Dict[str, Any]]:
    return _load_pool("benign_pool.json")


def get_malicious_pool() -> List[Dict[str, Any]]:
    return _load_pool("malicious_pool.json")


# Load pools at import so threshold_sweep and finetune_waf_model can import BENIGN_POOL / MALICIOUS_POOL
BENIGN_POOL = get_benign_pool()
MALICIOUS_POOL = get_malicious_pool()

NUM_BENIGN = int(os.getenv("STRESS_TEST_NUM_BENIGN", "600"))
NUM_MALICIOUS = int(os.getenv("STRESS_TEST_NUM_MALICIOUS", "400"))
TOTAL = NUM_BENIGN + NUM_MALICIOUS


def send_request(req: Dict[str, Any]) -> Dict[str, Any]:
    try:
        method = req.get("method", "GET")
        url = BASE_URL + req.get("url", "/test/endpoint")
        params = req.get("params")
        json_data = req.get("json")
        if method == "GET":
            r = requests.get(url, params=params, timeout=8)
        else:
            r = requests.post(url, json=json_data, timeout=8)
        return {"status": r.status_code, "blocked": r.status_code == 403}
    except requests.exceptions.Timeout:
        return {"status": 0, "blocked": False, "error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"status": 0, "blocked": False, "error": "ConnectionError"}
    except Exception as e:
        return {"status": 0, "blocked": False, "error": str(e)}


def _serialize_req(req: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request dict JSON-serialisable."""
    out = {}
    for k, v in req.items():
        if isinstance(v, dict):
            out[k] = {str(dk): str(dv) for dk, dv in v.items()}
        else:
            out[k] = v
    return out


def main():
    random.seed(42)
    # Build 600 benign (repeat from pool)
    benign_list = [random.choice(BENIGN_POOL).copy() for _ in range(NUM_BENIGN)]
    # Build 400 malicious (repeat from pool)
    malicious_list = [random.choice(MALICIOUS_POOL).copy() for _ in range(NUM_MALICIOUS)]
    # Tag and shuffle
    combined: List[tuple] = [(r, "benign") for r in benign_list] + [(r, "malicious") for r in malicious_list]
    random.shuffle(combined)

    print("=" * 70)
    print("  WAF STRESS TEST: 1000 requests (600 benign, 400 evasive malicious)")
    print("=" * 70)
    print(f"  Target: {BASE_URL}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {"benign": {"total": 0, "blocked": 0}, "malicious": {"total": 0, "blocked": 0}, "errors": 0}
    missed_malicious: List[Dict[str, Any]] = []
    false_positives: List[Dict[str, Any]] = []
    start = time.time()

    for i, (req, label) in enumerate(combined):
        out = send_request(req)
        results[label]["total"] += 1
        if out.get("error"):
            results["errors"] += 1
        elif out.get("blocked"):
            results[label]["blocked"] += 1
            # Track false positives (benign requests that were blocked)
            if label == "benign":
                false_positives.append(_serialize_req(req))
        else:
            # Track missed malicious (malicious requests that were NOT blocked)
            if label == "malicious":
                missed_malicious.append(_serialize_req(req))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1:4d}/{TOTAL}] elapsed {elapsed:.1f}s | benign blocked: {results['benign']['blocked']}/{results['benign']['total']} | malicious blocked: {results['malicious']['blocked']}/{results['malicious']['total']}")
        time.sleep(0.02)  # slight throttle to avoid overwhelming

    elapsed = time.time() - start
    b_total = results["benign"]["total"]
    b_blocked = results["benign"]["blocked"]
    m_total = results["malicious"]["total"]
    m_blocked = results["malicious"]["blocked"]

    detection_rate = (m_blocked / m_total * 100) if m_total else 0
    fp_rate = (b_blocked / b_total * 100) if b_total else 0
    missed = m_total - m_blocked

    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Benign:     {b_total} total, {b_blocked} blocked (false positives) -> FP rate: {fp_rate:.1f}%")
    print(f"  Malicious: {m_total} total, {m_blocked} blocked, {missed} missed -> Detection rate: {detection_rate:.1f}%")
    print(f"  Errors:    {results['errors']}")
    print(f"  Time:      {elapsed:.1f}s")
    print("=" * 70)

    # --- Write missed malicious requests to JSON ---
    missed_path = SCRIPT_DIR / "missed_malicious_samples.json"
    # Deduplicate by converting to tuples of sorted items
    seen = set()
    unique_missed = []
    for m in missed_malicious:
        key = _json.dumps(m, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_missed.append(m)
    with open(missed_path, "w") as f:
        _json.dump(unique_missed, f, indent=2)
    print(f"\n  Missed malicious ({len(unique_missed)} unique): {missed_path}")

    # --- Write false positive requests to JSON ---
    fp_path = SCRIPT_DIR / "false_positive_samples.json"
    seen_fp = set()
    unique_fp = []
    for fp in false_positives:
        key = _json.dumps(fp, sort_keys=True)
        if key not in seen_fp:
            seen_fp.add(key)
            unique_fp.append(fp)
    with open(fp_path, "w") as f:
        _json.dump(unique_fp, f, indent=2)
    print(f"  False positives ({len(unique_fp)} unique): {fp_path}")

    # --- Print missed malicious details ---
    if unique_missed:
        print(f"\n  --- Missed malicious payloads ({len(unique_missed)}) ---")
        for idx, m in enumerate(unique_missed, 1):
            payload = m.get("params") or m.get("json") or {}
            print(f"  {idx}. {m.get('method','?')} {m.get('url','?')} | payload: {payload}")

    print()
    print("  Next plan suggestions:")
    if detection_rate < 70:
        print("  - Detection rate is low: consider retraining with more evasive samples or lowering threshold (with FP trade-off).")
    elif detection_rate >= 90:
        print("  - Detection rate is strong; monitor FP rate. If FP is high, consider raising threshold or adding benign variants to training.")
    if fp_rate > 5:
        print("  - False positive rate is high: consider raising WAF_THRESHOLD or adding benign samples to training.")
    print("  - Re-run finetune_waf_model.py with augmented/evasive payloads to improve robustness.")
    print()


if __name__ == "__main__":
    main()
