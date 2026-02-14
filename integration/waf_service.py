"""
Standalone WAF Service for Nginx Integration

FastAPI service on port 8000. Accepts HTTP requests from Nginx Lua,
runs them through WAFClassifier, returns anomaly score for allow/block.
"""
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Module-level classifier (set by initialize_waf_service)
waf_service = None
_start_time = time.time()


class CheckRequest(BaseModel):
    method: str
    path: str
    query_params: Optional[Dict[str, Any]] = {}
    headers: Optional[Dict[str, str]] = {}
    body: Optional[str] = None


class PlaceholderWAFService:
    """Placeholder WAF service when no model is loaded (for testing)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._metrics = {"total_requests": 0, "anomalies_detected": 0, "total_time_ms": 0.0}

    @property
    def is_loaded(self) -> bool:
        return False

    async def check_request_async(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        body: Optional[Any] = None,
    ) -> Dict[str, Any]:
        query_params = query_params or {}
        headers = headers or {}
        start = time.perf_counter()
        self._metrics["total_requests"] += 1
        # Placeholder: return low score for all (no real detection)
        score = 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._metrics["total_time_ms"] += elapsed_ms
        return {
            "is_anomaly": score >= self.threshold,
            "anomaly_score": score,
            "threshold": self.threshold,
            "processing_time_ms": round(elapsed_ms, 2),
            "label": "benign",
            "confidence": 1.0 - score,
            "malicious_score": score,
            "benign_score": 1.0 - score,
        }

    def check_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync check for compatibility."""
        import asyncio
        return asyncio.run(self.check_request_async(
            method=request_data.get("method", "GET"),
            path=request_data.get("path", "/"),
            query_params=request_data.get("query_params"),
            headers=request_data.get("headers"),
            body=request_data.get("body"),
        ))

    def get_metrics(self) -> Dict[str, Any]:
        total = self._metrics["total_requests"]
        avg_ms = self._metrics["total_time_ms"] / total if total > 0 else 0
        return {
            "total_requests": total,
            "anomalies_detected": self._metrics["anomalies_detected"],
            "average_processing_time_ms": round(avg_ms, 2),
            "model_loaded": False,
            "device": "cpu",
            "threshold": self.threshold,
        }

    def update_threshold(self, threshold: float) -> None:
        self.threshold = threshold


def initialize_waf_service(
    model_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> None:
    """Initialize the WAF service (real model or placeholder)."""
    global waf_service
    vocab_path = vocab_path  # unused; kept for API compatibility
    if model_path is None:
        waf_service = PlaceholderWAFService(threshold=threshold)
        return
    try:
        from backend.ml.waf_classifier import WAFClassifier
        classifier = WAFClassifier(
            model_path=model_path,
            threshold=threshold,
            device=device,
        )
        if classifier.is_loaded:
            waf_service = classifier
        else:
            waf_service = PlaceholderWAFService(threshold=threshold)
    except Exception as e:
        from loguru import logger
        logger.warning(f"WAF model failed to load, using placeholder: {e}")
        waf_service = PlaceholderWAFService(threshold=threshold)


app = FastAPI(title="WAF Service", version="1.0.0")


@app.post("/check")
async def check(payload: CheckRequest):
    """Check request for threats. Used by Nginx Lua."""
    if waf_service is None:
        raise HTTPException(status_code=503, detail="WAF service not initialized")
    result = await waf_service.check_request_async(
        method=payload.method,
        path=payload.path,
        query_params=payload.query_params,
        headers=payload.headers,
        body=payload.body,
    )
    return {
        "is_anomaly": result.get("is_anomaly", False),
        "anomaly_score": result.get("anomaly_score", 0.0),
        "threshold": result.get("threshold", 0.5),
        "processing_time_ms": result.get("processing_time_ms", 0.0),
    }


@app.get("/health")
async def health():
    """Health check for Nginx and orchestration."""
    if waf_service is None:
        return {
            "status": "unhealthy",
            "service": "waf",
            "model_loaded": False,
            "mode": "uninitialized",
            "device": "n/a",
            "threshold": 0.5,
        }
    loaded = getattr(waf_service, "is_loaded", False)
    return {
        "status": "healthy",
        "service": "waf",
        "model_loaded": loaded,
        "mode": "placeholder" if not loaded else "model",
        "device": str(getattr(waf_service, "device", "cpu")),
        "threshold": getattr(waf_service, "threshold", 0.5),
    }


@app.get("/metrics")
async def metrics():
    """WAF metrics for Nginx /waf-metrics proxy."""
    if waf_service is None:
        raise HTTPException(status_code=503, detail="WAF service not initialized")
    m = waf_service.get_metrics()
    total = m.get("total_requests", 0)
    anomaly_rate = m.get("anomalies_detected", 0) / total if total > 0 else 0.0
    return {
        "total_requests": total,
        "anomalies_detected": m.get("anomalies_detected", 0),
        "anomaly_rate": round(anomaly_rate, 4),
        "avg_processing_time_ms": m.get("average_processing_time_ms", 0.0),
        "uptime_seconds": round(time.time() - _start_time, 1),
        "memory_usage_mb": 0.0,  # Optional: use psutil
        "cpu_percent": 0.0,  # Optional: use psutil
        **m,
    }


@app.get("/config")
async def config():
    """Configuration endpoint."""
    if waf_service is None:
        return {
            "threshold": 0.5,
            "device": "n/a",
            "vocab_size": 0,
            "max_batch_size": 32,
            "timeout": 5.0,
        }
    return {
        "threshold": getattr(waf_service, "threshold", 0.5),
        "device": str(getattr(waf_service, "device", "cpu")),
        "vocab_size": 0,
        "max_batch_size": 32,
        "timeout": 5.0,
    }


@app.post("/update-threshold")
async def update_threshold(payload: Dict[str, float]):
    """Update anomaly threshold."""
    if waf_service is None:
        raise HTTPException(status_code=503, detail="WAF service not initialized")
    t = payload.get("threshold")
    if t is None or not (0.0 <= t <= 1.0):
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    if hasattr(waf_service, "update_threshold"):
        waf_service.update_threshold(t)
    else:
        waf_service.threshold = t
    return {"status": "success", "new_threshold": t}
