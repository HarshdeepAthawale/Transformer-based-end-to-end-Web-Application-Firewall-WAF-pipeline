"""
Traffic Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from typing import List, Optional

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
    
    def get_traffic_by_range(self, start_time: datetime, limit: int = 3000) -> List[TrafficLog]:
        """Get traffic logs by time range (capped to avoid overload under load)."""
        return self.db.query(TrafficLog)\
            .filter(TrafficLog.timestamp >= start_time)\
            .order_by(desc(TrafficLog.timestamp))\
            .limit(limit)\
            .all()
    
    def get_traffic_by_endpoint(self, endpoint: str, start_time: datetime, limit: int = 2000) -> List[TrafficLog]:
        """Get traffic logs for specific endpoint (capped)."""
        return self.db.query(TrafficLog)\
            .filter(TrafficLog.endpoint == endpoint)\
            .filter(TrafficLog.timestamp >= start_time)\
            .order_by(desc(TrafficLog.timestamp))\
            .limit(limit)\
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

    def add_traffic_log_from_ingest_event(
        self,
        event_type: str,
        ip: str,
        method: Optional[str] = None,
        path: Optional[str] = None,
        attack_score: Optional[int] = None,
        bot_score: Optional[int] = None,
        bot_action: Optional[str] = None,
    ) -> TrafficLog:
        """
        Create a TrafficLog from a gateway ingest event and add to session (no commit).
        Used so Request Volume & Threats chart shows gateway traffic.
        """
        was_blocked = event_type != "allow"
        endpoint = (path or "/")[:500]
        method = (method or "GET")[:10]
        # Map event_type to threat_type for blocked events
        threat_type = None
        if was_blocked and event_type:
            et = event_type.lower()
            if "rate_limit" in et:
                threat_type = "rate_limit"
            elif "blacklist" in et:
                threat_type = "blacklist"
            elif "waf" in et:
                threat_type = "waf"
            elif "ddos" in et:
                threat_type = "ddos"
            elif "bot" in et:
                threat_type = "bot"
            else:
                threat_type = event_type[:50]
        anomaly_score = float(attack_score) / 100.0 if attack_score is not None else None
        log = TrafficLog(
            ip=ip,
            method=method,
            endpoint=endpoint,
            status_code=0,
            response_size=0,
            user_agent=None,
            query_string=None,
            request_body=None,
            processing_time_ms=0,
            was_blocked=1 if was_blocked else 0,
            anomaly_score=str(anomaly_score) if anomaly_score is not None else None,
            country_code=None,
            threat_type=threat_type,
        )
        self.db.add(log)
        return log
