"""WAF controller. Uses app-state WAF service or core.waf_factory."""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.waf_factory import get_waf_service


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


async def check_request(
    method: str,
    path: str,
    query_params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    body: Optional[str] = None,
    waf_service: Any = None,
) -> Dict:
    svc = waf_service or get_waf_service()
    if svc is None:
        return {"_error": "unavailable", "detail": "WAF service not available. Model files may be missing."}
    result = await svc.check_request_async(
        method=method,
        path=path,
        query_params=query_params or {},
        headers=headers or {},
        body=body,
    )
    return {
        "anomaly_score": result.get("anomaly_score", 0.0),
        "is_anomaly": result.get("is_anomaly", False),
        "threshold": result.get("threshold", svc.threshold),
        "processing_time_ms": result.get("processing_time_ms", 0.0),
        "model_version": "1.0.0",
    }


async def check_batch(
    requests: List[Dict],
    waf_service: Any = None,
) -> Dict:
    svc = waf_service or get_waf_service()
    if svc is None:
        return {"_error": "unavailable", "detail": "WAF service not available"}
    results = []
    for r in requests:
        res = await svc.check_request_async(
            method=r.get("method", "GET"),
            path=r.get("path", "/"),
            query_params=r.get("query_params", {}),
            headers=r.get("headers", {}),
            body=r.get("body"),
        )
        results.append(res)
    return {"success": True, "data": results, "timestamp": datetime.utcnow().isoformat()}


def get_stats(waf_service: Any = None) -> Dict:
    svc = waf_service or get_waf_service()
    if svc is None:
        return {
            "success": True,
            "data": {
                "service_available": False,
                "total_requests": 0,
                "anomalies_detected": 0,
                "average_processing_time_ms": 0.0,
                "threshold": 0.5,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    metrics = svc.get_metrics() if hasattr(svc, "get_metrics") else {}
    return {
        "success": True,
        "data": {
            "service_available": True,
            "total_requests": metrics.get("total_requests", 0),
            "anomalies_detected": metrics.get("anomalies_detected", 0),
            "average_processing_time_ms": metrics.get("average_processing_time_ms", 0.0),
            "threshold": getattr(svc, "threshold", 0.5),
            "model_loaded": True,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_config(waf_service: Any = None) -> Dict:
    svc = waf_service or get_waf_service()
    return {
        "success": True,
        "data": {
            "threshold": svc.threshold if svc else 0.5,
            "model_available": svc is not None,
            "service_enabled": True,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


def update_config(threshold: float, waf_service: Any = None) -> Dict:
    svc = waf_service or get_waf_service()
    if svc is None:
        return {"_error": "unavailable", "detail": "WAF service not available"}
    if hasattr(svc, "update_threshold"):
        svc.update_threshold(threshold)
    else:
        svc.threshold = threshold
    return {
        "success": True,
        "data": {"threshold": threshold, "message": "Configuration updated successfully"},
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_model_info(waf_service: Any = None) -> Dict:
    svc = waf_service or get_waf_service()
    root = _project_root()
    model_path = root / "models" / "checkpoints" / "best_model.pt"
    vocab_path = root / "models" / "vocabularies" / "http_vocab.json"
    init_error = None
    if svc is None:
        if not model_path.exists():
            init_error = f"Model file not found: {model_path}"
        elif not vocab_path.exists():
            init_error = f"Vocab file not found: {vocab_path}"
        else:
            init_error = "WAF service failed to initialize (check logs for details)"
    return {
        "success": True,
        "data": {
            "model_loaded": svc is not None,
            "model_path": str(model_path),
            "vocab_path": str(vocab_path),
            "model_exists": model_path.exists(),
            "vocab_exists": vocab_path.exists(),
            "threshold": svc.threshold if svc else 0.5,
            "version": "1.0.0",
            "initialization_error": init_error,
            "project_root": str(root),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
