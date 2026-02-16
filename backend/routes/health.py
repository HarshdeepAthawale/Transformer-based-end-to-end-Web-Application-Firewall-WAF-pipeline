"""Health check router."""
import time
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    model_loaded = False
    try:
        waf_svc = getattr(request.app.state, "waf_service", None)
        if waf_svc is not None and getattr(waf_svc, "is_loaded", False):
            model_loaded = True
    except Exception:
        pass
    return {
        "status": "healthy",
        "service": "waf-api",
        "model_loaded": model_loaded,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
