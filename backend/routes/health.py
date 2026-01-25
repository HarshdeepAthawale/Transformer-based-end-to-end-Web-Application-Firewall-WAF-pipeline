"""Health check router."""
import time
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "waf-api",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
