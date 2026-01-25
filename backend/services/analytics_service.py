"""
Analytics Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import Dict, List

from backend.models.metrics import Metrics
from backend.models.traffic import TrafficLog
from backend.models.threats import Threat


class AnalyticsService:
    """Service for analytics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_overview(self, start_time: datetime) -> List[Dict]:
        """Get analytics overview"""
        # Get aggregated metrics
        metrics = self.db.query(
            func.sum(Metrics.total_requests).label('total_requests'),
            func.sum(Metrics.blocked_requests).label('blocked_requests'),
            func.avg(Metrics.attack_rate).label('avg_attack_rate'),
            func.avg(Metrics.avg_response_time).label('avg_response_time')
        )\
        .filter(Metrics.timestamp >= start_time)\
        .first()
        
        # Get threat counts
        threat_counts = self.db.query(
            Threat.type,
            func.count(Threat.id).label('count')
        )\
        .filter(Threat.timestamp >= start_time)\
        .group_by(Threat.type)\
        .all()
        
        return [
            {
                "time": start_time.isoformat(),
                "total_requests": int(metrics.total_requests or 0),
                "blocked_requests": int(metrics.blocked_requests or 0),
                "attack_rate": float(metrics.avg_attack_rate or 0),
                "avg_response_time": float(metrics.avg_response_time or 0),
                "threats": {threat_type: int(count) for threat_type, count in threat_counts}
            }
        ]
    
    def get_trends(self, metric: str, start_time: datetime) -> List[Dict]:
        """Get trends for a specific metric"""
        # Aggregate by hour
        if metric == "requests":
            results = self.db.query(
                func.strftime('%Y-%m-%d %H:00:00', Metrics.timestamp).label('time'),
                func.sum(Metrics.total_requests).label('value')
            )\
            .filter(Metrics.timestamp >= start_time)\
            .group_by('time')\
            .order_by('time')\
            .all()
        elif metric == "threats":
            results = self.db.query(
                func.strftime('%Y-%m-%d %H:00:00', Threat.timestamp).label('time'),
                func.count(Threat.id).label('value')
            )\
            .filter(Threat.timestamp >= start_time)\
            .group_by('time')\
            .order_by('time')\
            .all()
        else:
            results = []
        
        return [
            {
                "time": row.time,
                "value": int(row.value or 0)
            }
            for row in results
        ]
    
    def get_summary(self, start_time: datetime) -> Dict:
        """Get analytics summary"""
        # Total requests
        total_requests = self.db.query(func.count(TrafficLog.id))\
            .filter(TrafficLog.timestamp >= start_time)\
            .scalar() or 0
        
        # Blocked requests
        blocked_requests = self.db.query(func.count(TrafficLog.id))\
            .filter(TrafficLog.timestamp >= start_time)\
            .filter(TrafficLog.was_blocked == 1)\
            .scalar() or 0
        
        # Total threats
        total_threats = self.db.query(func.count(Threat.id))\
            .filter(Threat.timestamp >= start_time)\
            .scalar() or 0
        
        # Threat types
        threat_types = self.db.query(
            Threat.type,
            func.count(Threat.id).label('count')
        )\
        .filter(Threat.timestamp >= start_time)\
        .group_by(Threat.type)\
        .all()
        
        return {
            "total_requests": total_requests,
            "blocked_requests": blocked_requests,
            "allowed_requests": total_requests - blocked_requests,
            "block_rate": (blocked_requests / total_requests * 100) if total_requests > 0 else 0,
            "total_threats": total_threats,
            "threat_types": {threat_type: int(count) for threat_type, count in threat_types},
            "time_range": {
                "start": start_time.isoformat(),
                "end": datetime.utcnow().isoformat()
            }
        }
