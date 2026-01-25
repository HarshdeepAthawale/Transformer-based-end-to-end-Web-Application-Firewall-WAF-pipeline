"""
WAF Service API endpoints. Uses controllers + schemas.
"""
from datetime import datetime
from typing import List
from fastapi import APIRouter, Request, HTTPException

from backend.schemas.waf import CheckRequest, CheckResponse
from backend.controllers import waf as waf_ctrl

router = APIRouter()


def _waf_svc(request: Request):
    return getattr(request.app.state, "waf_service", None)


@router.post("/check", response_model=CheckResponse)
async def check_request(request: Request, payload: CheckRequest):
    waf_svc = _waf_svc(request)
    out = await waf_ctrl.check_request(
        method=payload.method,
        path=payload.path,
        query_params=payload.query_params,
        headers=payload.headers,
        body=payload.body,
        waf_service=waf_svc,
    )
    if "_error" in out:
        raise HTTPException(status_code=503, detail=out.get("detail", "WAF service not available"))
    return CheckResponse(**out)


@router.post("/check/batch")
async def check_batch(request: Request, payload: List[CheckRequest]):
    waf_svc = _waf_svc(request)
    req_dicts = [p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in payload]
    out = await waf_ctrl.check_batch(req_dicts, waf_service=waf_svc)
    if "_error" in out:
        raise HTTPException(status_code=503, detail=out.get("detail", "WAF service not available"))
    return out


@router.get("/stats")
async def get_waf_stats(request: Request):
    return waf_ctrl.get_stats(waf_service=_waf_svc(request))


@router.get("/config")
async def get_waf_config(request: Request):
    return waf_ctrl.get_config(waf_service=_waf_svc(request))


@router.put("/config")
async def update_waf_config(request: Request, threshold: float = 0.5):
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    out = waf_ctrl.update_config(threshold, waf_service=_waf_svc(request))
    if "_error" in out:
        raise HTTPException(status_code=503, detail=out.get("detail", "WAF service not available"))
    return out


@router.get("/model-info")
async def get_model_info(request: Request):
    return waf_ctrl.get_model_info(waf_service=_waf_svc(request))


@router.get("/middleware-metrics")
async def get_waf_middleware_metrics():
    try:
        from backend.middleware.waf_middleware import WAFMiddleware
        if WAFMiddleware._instance is not None:
            metrics = WAFMiddleware._instance.get_metrics()
            return {"success": True, "data": metrics, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("Error getting WAF middleware metrics: %s", e)
    return {"success": False, "message": "WAF middleware metrics not available", "timestamp": datetime.utcnow().isoformat()}
