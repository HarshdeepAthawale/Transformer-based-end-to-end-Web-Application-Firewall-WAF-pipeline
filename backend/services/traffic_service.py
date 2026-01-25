"""
Traffic Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from typing import List

from backend.models.traffic import TrafficLog


class TrafficService:
    """Service for managing traffic logs"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_recent_traffic(self, limit: int = 50) -> List[TrafficLog]:
        """Get recent traffic logs"""
        return self.db.query(TrafficLog)\
            .order_by(desc(TrafficLog.timestamp))\
            .limit(limit)\
            .all()
    
    def get_traffic_by_range(self, start_time: datetime) -> List[TrafficLog]:
        """Get traffic logs by time range"""
        return self.db.query(TrafficLog)\
            .filter(TrafficLog.timestamp >= start_time)\
            .order_by(desc(TrafficLog.timestamp))\
            .all()
    
    def get_traffic_by_endpoint(self, endpoint: str, start_time: datetime) -> List[TrafficLog]:
        """Get traffic logs for specific endpoint"""
        return self.db.query(TrafficLog)\
            .filter(TrafficLog.endpoint == endpoint)\
            .filter(TrafficLog.timestamp >= start_time)\
            .order_by(desc(TrafficLog.timestamp))\
            .all()
    
    def create_traffic_log(
        self,
        ip: str,
        method: str,
        endpoint: str,
        status_code: int,
        response_size: int = 0,
        user_agent: str = None,
        query_string: str = None,
        request_body: str = None,
        processing_time_ms: int = 0,
        was_blocked: bool = False,
        anomaly_score: float = None,
        country_code: str = None,
        threat_type: str = None
    ) -> TrafficLog:
        """Create a new traffic log"""
        log = TrafficLog(
            ip=ip,
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            response_size=response_size,
            user_agent=user_agent,
            query_string=query_string,
            request_body=request_body[:1000] if request_body else None,  # Truncate large bodies
            processing_time_ms=processing_time_ms,
            was_blocked=1 if was_blocked else 0,
            anomaly_score=str(anomaly_score) if anomaly_score is not None else None,
            country_code=country_code,
            threat_type=threat_type
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log
