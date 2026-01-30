"""
WAF Classifier using fine-tuned HuggingFace transformer model.
Provides threat detection for HTTP requests.
"""
import asyncio
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger


class WAFClassifier:
    """
    WAF threat classifier using a fine-tuned transformer model.

    Classifies HTTP requests as 'benign' or 'malicious'.
    Provides both sync and async interfaces for integration with FastAPI.
    """

    def __init__(
        self,
        model_path: str = "models/waf-distilbert",
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the WAF classifier.

        Args:
            model_path: Path to the fine-tuned model directory
            threshold: Confidence threshold for malicious classification
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

        # Metrics tracking
        self._metrics = {
            "total_requests": 0,
            "anomalies_detected": 0,
            "total_processing_time_ms": 0.0,
        }
        self._metrics_lock = threading.Lock()

        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Try to load model
        self._load_model()

    def _load_model(self) -> bool:
        """Load the model and tokenizer."""
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}. Run finetune_waf_model.py first.")
            return False

        try:
            logger.info(f"Loading WAF model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"WAF model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load WAF model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a single HTTP request.

        Args:
            text: The HTTP request text

        Returns:
            Dict with 'label', 'confidence', 'is_malicious'
        """
        if not self._loaded:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "is_malicious": False,
                "error": "Model not loaded",
            }

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Get prediction
            malicious_prob = probs[0][1].item()
            benign_prob = probs[0][0].item()

            is_malicious = malicious_prob >= self.threshold
            label = "malicious" if is_malicious else "benign"
            confidence = malicious_prob if is_malicious else benign_prob

            return {
                "label": label,
                "confidence": round(confidence, 4),
                "is_malicious": is_malicious,
                "malicious_score": round(malicious_prob, 4),
                "benign_score": round(benign_prob, 4),
            }
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                "label": "error",
                "confidence": 0.0,
                "is_malicious": False,
                "error": str(e),
            }

    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Classify multiple HTTP requests.

        Args:
            texts: List of HTTP request texts
            batch_size: Batch size for inference

        Returns:
            List of classification results
        """
        if not self._loaded:
            return [{"label": "unknown", "confidence": 0.0, "is_malicious": False, "error": "Model not loaded"}] * len(texts)

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                inputs = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                for j, prob in enumerate(probs):
                    malicious_prob = prob[1].item()
                    benign_prob = prob[0].item()
                    is_malicious = malicious_prob >= self.threshold

                    results.append({
                        "label": "malicious" if is_malicious else "benign",
                        "confidence": round(malicious_prob if is_malicious else benign_prob, 4),
                        "is_malicious": is_malicious,
                        "malicious_score": round(malicious_prob, 4),
                        "benign_score": round(benign_prob, 4),
                    })
            except Exception as e:
                logger.error(f"Batch classification error: {e}")
                results.extend([{"label": "error", "confidence": 0.0, "is_malicious": False, "error": str(e)}] * len(batch))

        return results

    def check_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check an HTTP request for threats.

        Args:
            request_data: Dict containing 'method', 'path', 'headers', 'body', etc.

        Returns:
            Classification result with request reconstruction
        """
        # Reconstruct HTTP request text
        method = request_data.get("method", "GET")
        path = request_data.get("path", "/")
        headers = request_data.get("headers", {})
        body = request_data.get("body", "")

        # Build request string (similar to raw HTTP)
        lines = [f"{method} {path} HTTP/1.1"]
        for key, value in headers.items():
            lines.append(f"{key}: {value}")
        if body:
            lines.append("")
            lines.append(body if isinstance(body, str) else str(body))

        request_text = "\n".join(lines)

        # Classify
        result = self.classify(request_text)
        result["request_text_length"] = len(request_text)

        return result

    def _build_request_text(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
    ) -> str:
        """
        Build HTTP request text from components for classification.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            query_params: Query parameters dict
            headers: Headers dict
            body: Request body (string, dict, or None)

        Returns:
            Reconstructed HTTP request text
        """
        # Build path with query string
        full_path = path
        if query_params:
            query_str = "&".join(f"{k}={v}" for k, v in query_params.items())
            if query_str:
                full_path = f"{path}?{query_str}"

        # Build request string (similar to raw HTTP format)
        lines = [f"{method} {full_path} HTTP/1.1"]

        # Add headers
        if headers:
            for key, value in headers.items():
                # Skip some internal headers
                if key.lower() not in ("host", "content-length"):
                    lines.append(f"{key}: {value}")

        # Add body
        if body:
            lines.append("")
            if isinstance(body, dict):
                import json
                lines.append(json.dumps(body))
            elif isinstance(body, str):
                lines.append(body)
            else:
                lines.append(str(body))

        return "\n".join(lines)

    async def check_request_async(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Async method to check HTTP request for threats.
        Used by middleware and API endpoints.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            query_params: Query parameters dict
            headers: Headers dict
            body: Request body

        Returns:
            Dict with is_anomaly, anomaly_score, threshold, processing_time_ms, label, confidence
        """
        start_time = time.perf_counter()

        # Build request text from components
        request_text = self._build_request_text(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            body=body,
        )

        # Run sync classify in thread pool to avoid blocking event loop
        try:
            classification = await asyncio.to_thread(self.classify, request_text)
        except Exception as e:
            logger.error(f"Async classification error: {e}")
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "threshold": self.threshold,
                "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                "label": "error",
                "confidence": 0.0,
                "error": str(e),
            }

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Determine if this is an anomaly based on malicious score
        malicious_score = classification.get("malicious_score", 0.0)
        is_anomaly = malicious_score >= self.threshold

        # Update metrics
        with self._metrics_lock:
            self._metrics["total_requests"] += 1
            self._metrics["total_processing_time_ms"] += processing_time_ms
            if is_anomaly:
                self._metrics["anomalies_detected"] += 1

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": malicious_score,
            "threshold": self.threshold,
            "processing_time_ms": round(processing_time_ms, 2),
            "label": classification.get("label", "unknown"),
            "confidence": classification.get("confidence", 0.0),
            "malicious_score": malicious_score,
            "benign_score": classification.get("benign_score", 0.0),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get classifier metrics.

        Returns:
            Dict with total_requests, anomalies_detected, average_processing_time_ms
        """
        with self._metrics_lock:
            total = self._metrics["total_requests"]
            avg_time = (
                self._metrics["total_processing_time_ms"] / total
                if total > 0
                else 0.0
            )
            return {
                "total_requests": total,
                "anomalies_detected": self._metrics["anomalies_detected"],
                "average_processing_time_ms": round(avg_time, 2),
                "model_loaded": self._loaded,
                "device": str(self.device),
                "threshold": self.threshold,
            }

    def update_threshold(self, threshold: float) -> None:
        """
        Update the anomaly detection threshold.

        Args:
            threshold: New threshold value (0.0 - 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            logger.info(f"WAF threshold updated to {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
