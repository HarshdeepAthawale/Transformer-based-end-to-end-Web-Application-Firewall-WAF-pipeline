"""DDoS API: adaptive stats for frontend."""

from fastapi import APIRouter

from backend.services.adaptive_ddos_service import get_adaptive_stats

router = APIRouter()


@router.get("/adaptive-stats")
async def get_adaptive_ddos_stats():
    """
    Current adaptive threshold, baseline percentile value, last update time, config.
    Reads from Redis and config; no mocks.
    """
    data = get_adaptive_stats()
    return {"success": True, "data": data}
