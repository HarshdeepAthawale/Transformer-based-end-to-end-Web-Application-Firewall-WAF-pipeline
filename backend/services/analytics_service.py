"""
Analytics Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
from backend.lib.datetime_utils import utc_now
from typing import Dict, List

from backend.models.metrics import Metrics
from backend.models.traffic import TrafficLog
from backend.models.threats import Threat
from backend.services.charts_service import ChartsService


class AnalyticsService:
    """Service for analytics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_overview(self, start_time: datetime) -> List[Dict]:
        """Get analytics overview as time-series (requests, blocked, allowed per time bucket)"""
        charts_service = ChartsService(self.db)
        return charts_service.get_requests_chart_data(start_time)
    
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
        
        threat_map = {t: int(c) for t, c in threat_types}
        sql_injection = sum(int(c) for t, c in threat_types if t and ('sql' in t.lower() or 'injection' in t.lower()))
        xss = sum(int(c) for t, c in threat_types if t and ('xss' in t.lower() or 'cross-site' in t.lower()))
        ddos = sum(int(c) for t, c in threat_types if t and ('ddos' in t.lower() or 'dos' in t.lower()))
        other = sum(int(c) for t, c in threat_types if t and 'sql' not in t.lower() and 'injection' not in t.lower() and 'xss' not in t.lower() and 'cross-site' not in t.lower() and 'ddos' not in t.lower() and 'dos' not in t.lower())
        
        block_rate = (blocked_requests / total_requests * 100) if total_requests > 0 else 0
        return {
            "total_requests": total_requests,
            "blocked_requests": blocked_requests,
            "allowed_requests": total_requests - blocked_requests,
            "block_rate": block_rate,
            "attack_rate": block_rate,
            "total_threats": total_threats,
            "threat_types": threat_map,
            "sql_injection": sql_injection,
            "xss": xss,
            "ddos": ddos,
            "other": other,
            "time_range": {
                "start": start_time.isoformat(),
                "end": utc_now().isoformat()
            }
        }
