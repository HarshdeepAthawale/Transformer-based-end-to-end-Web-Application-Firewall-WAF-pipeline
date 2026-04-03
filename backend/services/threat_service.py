"""
Threat Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime
from typing import List, Dict

from backend.models.threats import Threat, ThreatSeverity


class ThreatService:
    """Service for managing threats"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_recent_threats(self, org_id: int, limit: int = 20) -> List[Threat]:
        """Get recent threats for organization"""
        return self.db.query(Threat)\
            .filter(Threat.org_id == org_id)\
            .order_by(desc(Threat.timestamp))\
            .limit(limit)\
            .all()
    
    def get_threats_by_range(self, org_id: int, start_time: datetime) -> List[Threat]:
        """Get threats by time range for organization"""
        return self.db.query(Threat)\
            .filter(Threat.org_id == org_id)\
            .filter(Threat.timestamp >= start_time)\
            .order_by(desc(Threat.timestamp))\
            .all()
    
    def get_threats_by_type(self, org_id: int, threat_type: str, start_time: datetime) -> List[Threat]:
        """Get threats by type for organization"""
        return self.db.query(Threat)\
            .filter(Threat.org_id == org_id)\
            .filter(Threat.type == threat_type)\
            .filter(Threat.timestamp >= start_time)\
            .order_by(desc(Threat.timestamp))\
            .all()
    
    def get_threat_stats(self, org_id: int, start_time: datetime) -> Dict[str, int]:
        """Get threat statistics for organization"""
        stats = self.db.query(
            Threat.type,
            func.count(Threat.id).label('count')
        )\
        .filter(Threat.org_id == org_id)\
        .filter(Threat.timestamp >= start_time)\
        .group_by(Threat.type)\
        .all()

        return {threat_type: count for threat_type, count in stats}
    
    def create_threat(
        self,
        org_id: int,
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
        """Create a new threat for organization"""
        threat = Threat(
            org_id=org_id,
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
