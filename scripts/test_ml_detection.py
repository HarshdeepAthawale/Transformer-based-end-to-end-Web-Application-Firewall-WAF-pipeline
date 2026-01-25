#!/usr/bin/env python3
"""
Test ML Detection on Malicious Payloads

Test the ML model's ability to detect malicious requests.
"""
import argparse
import json
from pathlib import Path
from loguru import logger

from backend.ml.waf_service import WAFService
from tests.payloads.malicious_payloads import MALICIOUS_PAYLOADS, BENIGN_REQUESTS


def test_detection(waf_service: WAFService, payloads: list, expected_anomaly: bool, label: str):
    """Test detection on a set of payloads"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {label} ({len(payloads)} samples)")
    logger.info(f"{'='*60}")
    
    detected = 0
    total = len(payloads)
    
    for i, payload in enumerate(payloads):
        method = payload.get('method', 'GET')
        path = payload.get('path', '/')
        query_params = payload.get('query_params', {})
        headers = payload.get('headers', {})
        body = payload.get('body')
        
        result = waf_service.check_request(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            body=body
        )
        
        is_anomaly = result.get('is_anomaly', False)
        score = result.get('anomaly_score', 0.0)
        
        if is_anomaly == expected_anomaly:
            detected += 1
        
        if i < 5:  # Show first 5 examples
            status = "✓" if is_anomaly == expected_anomaly else "✗"
            logger.info(
                f"{status} [{i+1}/{total}] Score: {score:.4f}, "
                f"Anomaly: {is_anomaly}, Expected: {expected_anomaly}"
            )
            logger.debug(f"  Request: {method} {path}")
    
    accuracy = detected / total if total > 0 else 0.0
    logger.info(f"\nResults: {detected}/{total} correct ({accuracy*100:.2f}%)")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Test ML detection on malicious payloads")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, help="Path to vocabulary file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Try to find default paths
    project_root = Path(__file__).parent.parent
    
    if not args.model:
        default_model = project_root / "models" / "deployed" / "model.pt"
        if default_model.exists():
            args.model = str(default_model)
        else:
            logger.error("No model path provided and default not found")
            return
    
    if not args.vocab:
        default_vocab = project_root / "models" / "vocabularies" / "vocab.json"
        if default_vocab.exists():
            args.vocab = str(default_vocab)
        else:
            logger.error("No vocabulary path provided and default not found")
            return
    
    # Initialize WAF service
    logger.info("Initializing WAF service...")
    waf_service = WAFService(
        model_path=args.model,
        vocab_path=args.vocab,
        threshold=args.threshold,
        device=args.device
    )
    
    if not waf_service.ml_enabled:
        logger.error("WAF service not initialized with ML components")
        return
    
    logger.info(f"WAF service initialized (threshold: {args.threshold})")
    
    # Test malicious payloads
    malicious_accuracy = test_detection(
        waf_service,
        MALICIOUS_PAYLOADS,
        expected_anomaly=True,
        label="MALICIOUS PAYLOADS"
    )
    
    # Test benign requests
    benign_accuracy = test_detection(
        waf_service,
        BENIGN_REQUESTS,
        expected_anomaly=False,
        label="BENIGN REQUESTS"
    )
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Malicious Detection Rate: {malicious_accuracy*100:.2f}%")
    logger.info(f"Benign False Positive Rate: {(1-benign_accuracy)*100:.2f}%")
    logger.info(f"Overall Accuracy: {(malicious_accuracy + benign_accuracy)/2*100:.2f}%")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
