"""
ONNX-based WAF Classifier — drop-in replacement for WAFClassifier.
3-5x faster inference via ONNX Runtime. Opt-in via WAF_USE_ONNX=true.
"""
import asyncio
import threading
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger

from backend.parsing import ParsingPipeline

# Multi-class attack categories (index -> label)
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

SUB_SCORE_FIELDS = {
    "sqli": "waf_sqli_score",
    "xss": "waf_xss_score",
    "rce": "waf_rce_score",
}


class ONNXWAFClassifier:
    """
    ONNX Runtime WAF classifier. Same public API as WAFClassifier.

    Uses onnxruntime.InferenceSession with CUDA (if available) then CPU provider.
    Tokenizer is loaded from the same model directory as the PyTorch version.
    """

    # Signal to gateway batching hook that batch inference is available
    _use_batch = True

    def __init__(
        self,
        model_path: str = "models/waf-distilbert-multiclass",
        onnx_path: Optional[str] = None,
        threshold: float = 0.5,
        max_seq_len: int = 256,
    ):
        self.model_path = Path(model_path)
        self.max_seq_len = max_seq_len
        # Default ONNX file: check inside model dir first, then next to it
        if onnx_path:
            self.onnx_path = Path(onnx_path)
        elif (Path(model_path) / "model.onnx").exists():
            self.onnx_path = Path(model_path) / "model.onnx"
        else:
            self.onnx_path = self.model_path.parent / "waf-distilbert.onnx"
        self.threshold = threshold
        self.tokenizer = None
        self.session = None
        self._loaded = False

        self._metrics = {
            "total_requests": 0,
            "anomalies_detected": 0,
            "total_processing_time_ms": 0.0,
        }
        self._metrics_lock = threading.Lock()

        self._pipeline = ParsingPipeline(
            include_headers=True,
            include_body=True,
        )

        self._load_model()

    def _load_model(self) -> bool:
        """Load HuggingFace tokenizer + ONNX Runtime inference session."""
        if not self.onnx_path.exists():
            logger.warning(
                f"ONNX model not found at {self.onnx_path}. "
                "Run: python scripts/export_onnx.py"
            )
            return False

        if not self.model_path.exists():
            logger.warning(f"Tokenizer directory not found at {self.model_path}.")
            return False

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            logger.info(f"Loading ONNX model from {self.onnx_path}")

            # Prefer CUDA; fall back to CPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=providers,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), use_fast=True)
            # Detect num_labels from ONNX output shape
            out_shape = self.session.get_outputs()[0].shape
            self._num_labels = out_shape[-1] if len(out_shape) == 2 and isinstance(out_shape[-1], int) else 2
            self._is_multiclass = self._num_labels > 2
            self._loaded = True
            active_provider = self.session.get_providers()[0]
            logger.info(f"ONNX WAF model loaded — provider: {active_provider}, num_labels={self._num_labels}")
            return True

        except ImportError:
            logger.warning("onnxruntime not installed. pip install onnxruntime>=1.17.0")
            return False
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Internal helpers (mirror WAFClassifier)
    # ------------------------------------------------------------------

    def _build_request_text(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
    ) -> str:
        full_path = path
        if query_params:
            query_str = "&".join(f"{k}={v}" for k, v in query_params.items())
            if query_str:
                full_path = f"{path}?{query_str}"

        lines = [f"{method} {full_path} HTTP/1.1"]
        if headers:
            for key, value in headers.items():
                if key.lower() not in ("host", "content-length"):
                    lines.append(f"{key}: {value}")
        if body:
            lines.append("")
            if isinstance(body, dict):
                lines.append(json.dumps(body))
            elif isinstance(body, str):
                lines.append(body)
            else:
                lines.append(str(body))
        return "\n".join(lines)

    def _run_inference(self, input_ids, attention_mask):
        """Execute ONNX session and return softmax probabilities as list of [benign, malicious]."""
        import numpy as np

        feed = {
            "input_ids": input_ids.astype("int64"),
            "attention_mask": attention_mask.astype("int64"),
        }
        # Some ONNX exports include token_type_ids input
        input_names = {inp.name for inp in self.session.get_inputs()}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype="int64")

        logits = self.session.run(["logits"], feed)[0]  # shape: (batch, 2)
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return probs

    # ------------------------------------------------------------------
    # Public API — identical signatures to WAFClassifier
    # ------------------------------------------------------------------

    def _process_multiclass(self, probs) -> Dict[str, Any]:
        """Process multi-class output (8 classes) into WAF result dict."""
        import numpy as np

        benign_prob = float(probs[0])
        malicious_prob = 1.0 - benign_prob
        is_malicious = malicious_prob >= self.threshold

        attack_score = max(1, min(99, int(round((1.0 - malicious_prob) * 100))))

        max_attack_idx = int(np.argmax(probs[1:]))
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

        for class_name, field_name in SUB_SCORE_FIELDS.items():
            class_idx = [k for k, v in ATTACK_CLASSES.items() if v == class_name]
            if class_idx:
                prob = float(probs[class_idx[0]])
                result[field_name] = max(1, min(99, int(round((1.0 - prob) * 100))))

        return result

    def _process_binary(self, probs) -> Dict[str, Any]:
        """Process binary output (benign vs malicious)."""
        benign_prob = float(probs[0])
        malicious_prob = float(probs[1])
        is_malicious = malicious_prob >= self.threshold
        label = "malicious" if is_malicious else "benign"
        confidence = malicious_prob if is_malicious else benign_prob
        # SECURITY: WAF convention -- lower score = more malicious (1-99 scale)
        # Consistent with _process_multiclass and industry WAF scoring
        attack_score = max(1, min(99, int(round((1.0 - malicious_prob) * 100))))

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "is_malicious": is_malicious,
            "malicious_score": round(malicious_prob, 4),
            "benign_score": round(benign_prob, 4),
            "attack_score": attack_score,
        }

    def classify(self, text: str) -> Dict[str, Any]:
        if not self._loaded:
            return {"label": "unknown", "confidence": 0.0, "is_malicious": False, "error": "Model not loaded"}

        try:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="np",
            )
            probs = self._run_inference(enc["input_ids"], enc["attention_mask"])

            # Apply isotonic regression calibration if available
            if self._is_multiclass:
                try:
                    from backend.ml.calibration import calibrate_probabilities
                    cal_probs = calibrate_probabilities(
                        probs[0].tolist(), str(self.model_path),
                        label_names=list(ATTACK_CLASSES.values()),
                    )
                    probs[0] = cal_probs
                except Exception:
                    pass

            if self._is_multiclass:
                return self._process_multiclass(probs[0])
            return self._process_binary(probs[0])
        except Exception as e:
            logger.error(f"ONNX classification error: {e}")
            return {"label": "error", "confidence": 0.0, "is_malicious": False, "error": str(e)}

    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        if not self._loaded:
            return [{"label": "unknown", "confidence": 0.0, "is_malicious": False, "error": "Model not loaded"}] * len(texts)

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding=True,
                    return_tensors="np",
                )
                probs = self._run_inference(enc["input_ids"], enc["attention_mask"])

                for prob in probs:
                    if self._is_multiclass:
                        results.append(self._process_multiclass(prob))
                    else:
                        results.append(self._process_binary(prob))
            except Exception as e:
                logger.error(f"ONNX batch classification error: {e}")
                results.extend([{"label": "error", "confidence": 0.0, "is_malicious": False, "error": str(e)}] * len(batch))

        return results

    def check_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
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

    async def check_request_async(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        request_text = self._build_request_text(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            body=body,
        )

        try:
            classification = await asyncio.to_thread(self.classify, request_text)
        except Exception as e:
            logger.error(f"ONNX async classification error: {e}")
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "threshold": self.threshold,
                "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                "label": "error",
                "confidence": 0.0,
                "error": str(e),
            }

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        malicious_score = classification.get("malicious_score", 0.0)
        is_anomaly = malicious_score >= self.threshold

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
                "device": self.session.get_providers()[0] if self.session else "none",
                "threshold": self.threshold,
            }

    def update_threshold(self, threshold: float) -> None:
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            logger.info(f"ONNX WAF threshold updated to {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
