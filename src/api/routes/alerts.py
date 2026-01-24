"""
Alerts API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta

from src.api.database import get_db
from src.api.services.alert_service import AlertService

router = APIRouter()


@router.get("/active")
async def get_active_alerts(db: Session = Depends(get_db)):
    """Get active alerts"""
    service = AlertService(db)
    alerts = service.get_active_alerts()
    
    return {
        "success": True,
        "data": [alert.to_dict() for alert in alerts],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/history")
async def get_alert_history(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get alert history"""
    service = AlertService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    alerts = service.get_alert_history(start_time)
    
    return {
        "success": True,
        "data": [alert.to_dict() for alert in alerts],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/{alert_id}/dismiss")
async def dismiss_alert(alert_id: int, db: Session = Depends(get_db)):
    """Dismiss an alert"""
    service = AlertService(db)
    success = service.dismiss_alert(alert_id)
    
    return {
        "success": success,
        "message": "Alert dismissed" if success else "Alert not found",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    """Acknowledge an alert"""
    service = AlertService(db)
    success = service.acknowledge_alert(alert_id)
    
    return {
        "success": success,
        "message": "Alert acknowledged" if success else "Alert not found",
        "timestamp": datetime.utcnow().isoformat()
    }
