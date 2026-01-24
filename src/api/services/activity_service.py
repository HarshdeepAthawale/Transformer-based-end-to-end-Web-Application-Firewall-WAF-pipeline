"""
Activity Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from typing import List

from src.api.models.activities import Activity, ActivityType


class ActivityService:
    """Service for managing activities"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_recent_activities(self, limit: int = 10) -> List[Activity]:
        """Get recent activities"""
        return self.db.query(Activity)\
            .order_by(desc(Activity.timestamp))\
            .limit(limit)\
            .all()
    
    def get_activities_by_range(self, start_time: datetime) -> List[Activity]:
        """Get activities by time range"""
        return self.db.query(Activity)\
            .filter(Activity.timestamp >= start_time)\
            .order_by(desc(Activity.timestamp))\
            .all()
    
    def create_activity(
        self,
        type: ActivityType,
        title: str,
        details: str,
        ip: str = None,
        endpoint: str = None,
        method: str = None,
        threat_type: str = None,
        anomaly_score: float = None
    ) -> Activity:
        """Create a new activity"""
        activity = Activity(
            type=type,
            title=title,
            details=details,
            ip=ip,
            endpoint=endpoint,
            method=method,
            threat_type=threat_type,
            anomaly_score=str(anomaly_score) if anomaly_score is not None else None
        )
        self.db.add(activity)
        self.db.commit()
        self.db.refresh(activity)
        return activity
