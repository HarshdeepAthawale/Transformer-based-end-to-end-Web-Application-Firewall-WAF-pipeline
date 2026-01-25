"""Test endpoints for WAF (middleware checks these)."""
import time
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/test")
async def test_get(request: Request):
    return {
        "success": True,
        "message": "Request allowed",
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@router.post("/api/test")
async def test_post(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    return {
        "success": True,
        "message": "Request allowed",
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params),
        "body": body,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
