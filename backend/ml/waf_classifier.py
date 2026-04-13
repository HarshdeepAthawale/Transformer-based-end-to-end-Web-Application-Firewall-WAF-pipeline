"""
WAF Classifier using fine-tuned HuggingFace transformer model.
Provides threat detection for HTTP requests.
"""
import asyncio
import threading
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import parse_qs, urlparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

from backend.parsing import ParsingPipeline

# Multi-class attack categories (index -> label)
# Index 0 is always benign. Indices 1-7 are attack classes.
ATTACK_CLASSES = {
    0: "benign",
    1: "sqli",
    2: "xss",
    3: "rce",
    4: "path_traversal",
    5: "xxe",
    6: "ssrf",
    7: "other_attack",
}

# Sub-score field names for the 3 primary categories WAF tracks
SUB_SCORE_FIELDS = {
    "sqli": "waf_sqli_score",
    "xss": "waf_xss_score",
    "rce": "waf_rce_score",
}


class WAFClassifier:
    """
    WAF threat classifier using a fine-tuned transformer model.

    Classifies HTTP requests as 'benign' or 'malicious'.
    Provides both sync and async interfaces for integration with FastAPI.
    """

    def __init__(
        self,
        model_path: str = "models/waf-distilbert-multiclass",
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
            "cache_hits": 0,
            "cache_misses": 0,
            "prefilter_skips": 0,
        }
        self._metrics_lock = threading.Lock()

        # LRU inference cache (fingerprint -> result)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_maxsize = 10000
        self._cache_order: List[str] = []  # Track insertion order for LRU

        # Parsing pipeline for request normalization
        self._pipeline = ParsingPipeline(
            include_headers=True,
            include_body=True,
        )

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
            self._num_labels = self.model.config.num_labels
            self._is_multiclass = self._num_labels > 2
            logger.info(f"WAF model loaded successfully on {self.device} (num_labels={self._num_labels}, multiclass={self._is_multiclass})")
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

        # LRU cache lookup
        import hashlib
        cache_key = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()
        if cache_key in self._cache:
            with self._metrics_lock:
                self._metrics["cache_hits"] += 1
            return self._cache[cache_key]

        # Ngram pre-filter: skip transformer for obvious cases
        try:
            from backend.ml.ngram_prefilter import quick_score
            prefilter_score = quick_score(text)
            if prefilter_score is not None:
                with self._metrics_lock:
                    self._metrics["prefilter_skips"] += 1
                is_mal = prefilter_score <= 50
                result = {
                    "label": "malicious" if is_mal else "benign",
                    "confidence": 0.95 if prefilter_score <= 10 or prefilter_score >= 90 else 0.7,
                    "is_malicious": is_mal,
                    "malicious_score": round((100 - prefilter_score) / 100, 4),
                    "benign_score": round(prefilter_score / 100, 4),
                    "attack_score": prefilter_score,
                    "prefiltered": True,
                }
                self._cache_put(cache_key, result)
                return result
        except ImportError:
            pass

        with self._metrics_lock:
            self._metrics["cache_misses"] += 1

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # DistilBERT does not use token_type_ids
            inputs.pop("token_type_ids", None)

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Apply isotonic regression calibration if available
            if self._is_multiclass:
                try:
                    from backend.ml.calibration import calibrate_probabilities
                    raw_list = probs[0].tolist()
                    cal_list = calibrate_probabilities(
                        raw_list, str(self.model_path),
                        label_names=list(ATTACK_CLASSES.values()),
                    )
                    probs[0] = torch.tensor(cal_list, device=self.device)
                except Exception:
                    pass  # Fall back to raw probabilities

            if self._is_multiclass:
                result = self._process_multiclass(probs[0])
            else:
                result = self._process_binary(probs[0])

            # Store in cache
            self._cache_put(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                "label": "error",
                "confidence": 0.0,
                "is_malicious": False,
                "error": str(e),
            }

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        """Add an entry to the LRU cache, evicting oldest if over limit."""
        if key in self._cache:
            return
        self._cache[key] = value
        self._cache_order.append(key)
        while len(self._cache) > self._cache_maxsize:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

    def _process_binary(self, probs: torch.Tensor) -> Dict[str, Any]:
        """Process binary classification output (benign vs malicious)."""
        malicious_prob = probs[1].item()
        benign_prob = probs[0].item()
        is_malicious = malicious_prob >= self.threshold
        label = "malicious" if is_malicious else "benign"
        confidence = malicious_prob if is_malicious else benign_prob
        attack_score = max(0, min(100, int(round(malicious_prob * 100))))

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "is_malicious": is_malicious,
            "malicious_score": round(malicious_prob, 4),
            "benign_score": round(benign_prob, 4),
            "attack_score": attack_score,
        }

    def _process_multiclass(self, probs: torch.Tensor) -> Dict[str, Any]:
        """
        Process multi-class output with sub-scores (industry-standard).
        Returns per-category scores where lower = more malicious (1-99 scale).
        """
        benign_prob = probs[0].item()
        # Sum of all attack class probabilities
        malicious_prob = 1.0 - benign_prob

        is_malicious = malicious_prob >= self.threshold

        # Overall attack score: 1-99 where lower = more malicious
        # WAF style: score = 100 - (malicious_probability * 100)
        attack_score = max(1, min(99, int(round((1.0 - malicious_prob) * 100))))

        # Find dominant attack class
        max_attack_idx = int(torch.argmax(probs[1:]).item())
        attack_class = ATTACK_CLASSES.get(max_attack_idx + 1, "other_attack")

        label = attack_class if is_malicious else "benign"
        confidence = malicious_prob if is_malicious else benign_prob

        result = {
            "label": label,
            "confidence": round(confidence, 4),
            "is_malicious": is_malicious,
            "malicious_score": round(malicious_prob, 4),
            "benign_score": round(benign_prob, 4),
            "attack_score": attack_score,
            "attack_class": attack_class if is_malicious else None,
        }

        # Add sub-scores for the 3 primary categories (lower = more malicious)
        for class_name, field_name in SUB_SCORE_FIELDS.items():
            class_idx = [k for k, v in ATTACK_CLASSES.items() if v == class_name]
            if class_idx:
                prob = probs[class_idx[0]].item()
                # Sub-score: 1-99 where lower = this category is more likely
                result[field_name] = max(1, min(99, int(round((1.0 - prob) * 100))))

        return result

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

                # DistilBERT does not use token_type_ids
                inputs.pop("token_type_ids", None)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                for prob in probs:
                    if self._is_multiclass:
                        results.append(self._process_multiclass(prob))
                    else:
                        results.append(self._process_binary(prob))
            except Exception as e:
                logger.error(f"Batch classification error: {e}")
                results.extend([{"label": "error", "confidence": 0.0, "is_malicious": False, "error": str(e)}] * len(batch))

        return results

    def _to_process_dict(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build dict for process_dict, ensuring path and query_params are separate."""
        query_params = dict(query_params) if query_params else {}
        if "?" in path:
            parsed = urlparse(path)
            path = parsed.path or "/"
            if not query_params and parsed.query:
                for k, v in parse_qs(parsed.query).items():
                    query_params[k] = v[0] if v else ""
        body_str = None
        if body is not None:
            body_str = json.dumps(body) if isinstance(body, dict) else str(body)
        return {
            "method": method,
            "path": path,
            "query_params": query_params,
            "headers": headers or {},
            "body": body_str,
        }

    def check_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check an HTTP request for threats.

        Args:
            request_data: Dict containing 'method', 'path', 'headers', 'body', etc.

        Returns:
            Classification result with request reconstruction
        """
        # Use _build_request_text (raw HTTP format) to match training data
        # format.  The model was trained on un-normalised text so we must
        # NOT pass through the normaliser here.
        request_text = self._build_request_text(
            method=request_data.get("method", "GET"),
            path=request_data.get("path", "/"),
            query_params=request_data.get("query_params"),
            headers=request_data.get("headers"),
            body=request_data.get("body"),
        )
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

        # Use _build_request_text (raw HTTP format) to match training data
        # format.  The model was trained on un-normalised text so we must
        # NOT pass through the normaliser here.
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

        attack_score = classification.get("attack_score") or max(0, min(100, int(round(malicious_score * 100))))

        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": malicious_score,
            "threshold": self.threshold,
            "processing_time_ms": round(processing_time_ms, 2),
            "label": classification.get("label", "unknown"),
            "confidence": classification.get("confidence", 0.0),
            "malicious_score": malicious_score,
            "benign_score": classification.get("benign_score", 0.0),
            "attack_score": attack_score,
        }

        # Forward multi-class sub-scores if available
        for field in ("attack_class", "waf_sqli_score", "waf_xss_score", "waf_rce_score"):
            if field in classification:
                result[field] = classification[field]

        return result

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
            cache_hits = self._metrics["cache_hits"]
            cache_misses = self._metrics["cache_misses"]
            cache_total = cache_hits + cache_misses
            cache_hit_ratio = (cache_hits / cache_total * 100) if cache_total > 0 else 0.0

            return {
                "total_requests": total,
                "anomalies_detected": self._metrics["anomalies_detected"],
                "average_processing_time_ms": round(avg_time, 2),
                "model_loaded": self._loaded,
                "device": str(self.device),
                "threshold": self.threshold,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_ratio": round(cache_hit_ratio, 1),
                "cache_size": len(self._cache),
                "prefilter_skips": self._metrics["prefilter_skips"],
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
