#!/usr/bin/env python3
"""
Threshold Sweep – test multiple WAF_THRESHOLD values offline.

Loads the WAF model once, then classifies every request from the stress-test
pools at each threshold and prints a summary table.

Thresholds can be set via THRESHOLD_SWEEP env (comma-separated, e.g. 0.4,0.5,0.6)
or --thresholds on the command line.

Usage:
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --thresholds 0.5 0.6 0.7 0.8 0.9
    THRESHOLD_SWEEP=0.4,0.5,0.6,0.7 python scripts/threshold_sweep.py
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `backend.*` imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any

_DEFAULT_THRESHOLDS = "0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"


def _parse_thresholds_env() -> List[float]:
    raw = os.getenv("THRESHOLD_SWEEP", _DEFAULT_THRESHOLDS)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="WAF threshold sweep")
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Threshold values to evaluate (default: from THRESHOLD_SWEEP env or built-in default)",
    )
    p.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "waf-distilbert"),
        help="Path to fine-tuned model",
    )
    args = p.parse_args()
    if args.thresholds is None:
        args.thresholds = _parse_thresholds_env()
    return args


def _score_request(classifier, req: Dict[str, Any]) -> float:
    """Return the raw malicious_score for a single request dict."""
    method = req.get("method", "GET")
    path = req.get("url", "/test/endpoint")
    query_params = req.get("params")
    body = req.get("json")
    result = classifier.check_request({
        "method": method,
        "path": path,
        "query_params": query_params,
        "body": body,
    })
    return result.get("malicious_score", 0.0)


def main():
    args = parse_args()

    # Import the stress-test pools
    from scripts.stress_test_1000_evasive import BENIGN_POOL, MALICIOUS_POOL

    # Load classifier once (threshold doesn't matter for scoring)
    from backend.ml.waf_classifier import WAFClassifier

    print(f"Loading model from {args.model_path} ...")
    classifier = WAFClassifier(model_path=args.model_path, threshold=0.5)
    if not classifier.is_loaded:
        print("ERROR: model not loaded. Run finetune_waf_model.py first.")
        sys.exit(1)

    # Score all unique pool entries once
    print(f"Scoring {len(BENIGN_POOL)} benign pool entries ...")
    benign_scores = [_score_request(classifier, r) for r in BENIGN_POOL]
    print(f"Scoring {len(MALICIOUS_POOL)} malicious pool entries ...")
    malicious_scores = [_score_request(classifier, r) for r in MALICIOUS_POOL]

    # Print table header
    header = f"{'Threshold':>10} | {'FP rate':>8} | {'Det rate':>8} | {'FP count':>8} | {'Missed':>6} | {'Benign OK':>9} | {'Mal blocked':>11}"
    print()
    print("=" * len(header))
    print("  THRESHOLD SWEEP RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    best_thresh = None
    best_score = -1.0

    for thresh in sorted(args.thresholds):
        fp_count = sum(1 for s in benign_scores if s >= thresh)
        det_count = sum(1 for s in malicious_scores if s >= thresh)
        fp_rate = fp_count / len(benign_scores) * 100 if benign_scores else 0
        det_rate = det_count / len(malicious_scores) * 100 if malicious_scores else 0
        missed = len(malicious_scores) - det_count
        benign_ok = len(benign_scores) - fp_count

        print(
            f"{thresh:>10.2f} | {fp_rate:>7.1f}% | {det_rate:>7.1f}% | "
            f"{fp_count:>8} | {missed:>6} | {benign_ok:>9} | {det_count:>11}"
        )

        # Heuristic: best = detection >= 95% with lowest FP rate
        if det_rate >= 95.0:
            score = (100.0 - fp_rate)  # Higher is better
            if score > best_score:
                best_score = score
                best_thresh = thresh

    print("-" * len(header))

    if best_thresh is not None:
        print(f"\n  Recommended threshold (det >= 95%): {best_thresh:.2f}")
    else:
        print("\n  No threshold achieved >= 95% detection. Consider retraining the model.")

    # Also show the threshold where FP < 10% regardless of detection
    for thresh in sorted(args.thresholds):
        fp_count = sum(1 for s in benign_scores if s >= thresh)
        fp_rate = fp_count / len(benign_scores) * 100 if benign_scores else 0
        det_count = sum(1 for s in malicious_scores if s >= thresh)
        det_rate = det_count / len(malicious_scores) * 100 if malicious_scores else 0
        if fp_rate < 10.0:
            print(f"  First threshold with FP < 10%: {thresh:.2f} (FP={fp_rate:.1f}%, Det={det_rate:.1f}%)")
            break

    print()


if __name__ == "__main__":
    main()
