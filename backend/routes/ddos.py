"""DDoS API: adaptive stats for frontend."""

from fastapi import APIRouter, Depends

from backend.auth import get_current_tenant
from backend.services.adaptive_ddos_service import get_adaptive_stats

router = APIRouter()


@router.get("/adaptive-stats")
async def get_adaptive_ddos_stats(org_id: int = Depends(get_current_tenant)):
    """
    Current adaptive threshold, baseline percentile value, last update time, config.
    Reads from Redis and config; no mocks.
    """
    data = get_adaptive_stats()
    return {"success": True, "data": data}
