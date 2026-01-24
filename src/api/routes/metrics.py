"""
Metrics API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta
import time

from src.api.database import get_db
from src.api.services.metrics_service import MetricsService
from src.api.models.metrics import Metrics

router = APIRouter()


@router.get("/realtime")
async def get_realtime_metrics(db: Session = Depends(get_db)):
    """Get real-time metrics"""
    service = MetricsService(db)
    metrics = service.get_realtime_metrics()
    
    return {
        "success": True,
        "data": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/historical")
async def get_historical_metrics(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get historical metrics"""
    service = MetricsService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    metrics = service.get_historical_metrics(start_time)
    
    return {
        "success": True,
        "data": [m.to_dict() for m in metrics],
        "timestamp": datetime.utcnow().isoformat()
    }
