#!/usr/bin/env python3
"""
Version the current WAF model with metadata about what changed.

Creates a timestamped snapshot in models/versions/ using the existing
ModelVersionManager, so you can roll back if accuracy regresses.

Usage:
    python scripts/version_model.py
    python scripts/version_model.py --description "Added header injection training data"
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.learning.version_manager import ModelVersionManager


def main():
    parser = argparse.ArgumentParser(description="Version the current WAF model")
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "waf-distilbert"),
        help="Path to the model to version",
    )
    parser.add_argument(
        "--description",
        default="Security hardening: header injection + SSTI training data",
        help="Description of what changed",
    )
    args = parser.parse_args()

    mgr = ModelVersionManager()

    version_id = mgr.create_version(
        model_path=args.model_path,
        metadata={
            "description": args.description,
            "changes": [
                "Added CRLF/header injection payloads to training data",
                "Added SSTI payloads from PayloadsAllTheThings",
                "Added LDAP/XPATH payloads from attack test 08",
                "Fixed _req_to_text() to serialize headers (training-inference parity)",
                "Expanded benign samples for FP control",
            ],
            "targets": {
                "header_injection_detection": ">80%",
                "ldap_xpath_ssti_detection": ">85%",
                "false_positive_rate": "<1%",
                "overall_detection": ">85%",
            },
        },
    )

    print(f"Created model version: {version_id}")
    print(f"  Path: {mgr.get_version_path(version_id)}")
    print(f"  Current: {mgr.get_current_version()}")

    versions = mgr.list_versions()
    print(f"  Total versions: {len(versions)}")


if __name__ == "__main__":
    main()
