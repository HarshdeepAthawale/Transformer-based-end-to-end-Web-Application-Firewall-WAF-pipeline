"""Validate model before deployment."""

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from backend.ml.waf_classifier import WAFClassifier


class ModelValidator:
    """Validate model detection on known malicious samples before deploy."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        min_detection_rate: float = 0.7,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.min_detection_rate = min_detection_rate

        self._malicious_samples = [
            {"method": "GET", "path": "/api/users", "query_params": {"id": "1' OR '1'='1"}},
            {"method": "GET", "path": "/search", "query_params": {"q": "<script>alert(1)</script>"}},
            {"method": "GET", "path": "/api/file", "query_params": {"file": "../../../etc/passwd"}},
            {"method": "POST", "path": "/api/exec", "body": "; cat /etc/passwd"},
        ]

    def validate(self, model_path: Optional[str] = None) -> Dict:
        """Run validation; return dict with is_valid, detection_rate, details."""
        path = Path(model_path) if model_path else self.model_path
        if not path.exists():
            return {
                "is_valid": False,
                "detection_rate": 0.0,
                "error": f"Model not found: {path}",
            }

        try:
            classifier = WAFClassifier(
                model_path=str(path),
                threshold=self.threshold,
            )
        except Exception as e:
            return {"is_valid": False, "detection_rate": 0.0, "error": str(e)}

        if not classifier.is_loaded:
            return {"is_valid": False, "detection_rate": 0.0, "error": "Model failed to load"}

        detected = 0
        total = len(self._malicious_samples)
        details = []

        for sample in self._malicious_samples:
            result = classifier.check_request(sample)
            is_mal = result.get("is_malicious", False)
            if is_mal:
                detected += 1
            details.append({"sample": str(sample)[:80], "detected": is_mal})

        rate = detected / total if total > 0 else 0.0
        is_valid = rate >= self.min_detection_rate

        return {
            "is_valid": is_valid,
            "detection_rate": rate,
            "detected": detected,
            "total": total,
            "details": details,
        }
