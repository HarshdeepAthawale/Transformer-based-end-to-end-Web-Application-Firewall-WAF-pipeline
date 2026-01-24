"""
Threat Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime
from typing import List, Dict

from src.api.models.threats import Threat, ThreatSeverity


class ThreatService:
    """Service for managing threats"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_recent_threats(self, limit: int = 20) -> List[Threat]:
        """Get recent threats"""
        return self.db.query(Threat)\
            .order_by(desc(Threat.timestamp))\
            .limit(limit)\
            .all()
    
    def get_threats_by_range(self, start_time: datetime) -> List[Threat]:
        """Get threats by time range"""
        return self.db.query(Threat)\
            .filter(Threat.timestamp >= start_time)\
            .order_by(desc(Threat.timestamp))\
            .all()
    
    def get_threats_by_type(self, threat_type: str, start_time: datetime) -> List[Threat]:
        """Get threats by type"""
        return self.db.query(Threat)\
            .filter(Threat.type == threat_type)\
            .filter(Threat.timestamp >= start_time)\
            .order_by(desc(Threat.timestamp))\
            .all()
    
    def get_threat_stats(self, start_time: datetime) -> Dict[str, int]:
        """Get threat statistics"""
        stats = self.db.query(
            Threat.type,
            func.count(Threat.id).label('count')
        )\
        .filter(Threat.timestamp >= start_time)\
        .group_by(Threat.type)\
        .all()
        
        return {threat_type: count for threat_type, count in stats}
    
    def create_threat(
        self,
        type: str,
        severity: ThreatSeverity,
        source_ip: str,
        endpoint: str,
        method: str,
        blocked: bool = False,
        anomaly_score: float = None,
        details: str = None,
        payload: str = None,
        user_agent: str = None,
        country_code: str = None,
        processing_time_ms: int = 0
    ) -> Threat:
        """Create a new threat"""
        threat = Threat(
            type=type,
            severity=severity,
            source_ip=source_ip,
            endpoint=endpoint,
            method=method,
            blocked=blocked,
            anomaly_score=str(anomaly_score) if anomaly_score is not None else None,
            details=details,
            payload=payload[:500] if payload else None,  # Truncate large payloads
            user_agent=user_agent,
            country_code=country_code,
            processing_time_ms=processing_time_ms
        )
        self.db.add(threat)
        self.db.commit()
        self.db.refresh(threat)
        return threat
