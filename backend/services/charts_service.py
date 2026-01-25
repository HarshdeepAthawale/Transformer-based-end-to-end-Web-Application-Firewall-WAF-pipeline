"""
Charts Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import List, Dict

from backend.models.metrics import Metrics
from backend.models.traffic import TrafficLog
from backend.models.threats import Threat


class ChartsService:
    """Service for generating chart data"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_requests_chart_data(self, start_time: datetime) -> List[Dict]:
        """Get requests chart data"""
        # Aggregate by hour
        results = self.db.query(
            func.strftime('%Y-%m-%d %H:00:00', Metrics.timestamp).label('time'),
            func.sum(Metrics.total_requests).label('requests'),
            func.sum(Metrics.blocked_requests).label('blocked'),
            func.sum(Metrics.allowed_requests).label('allowed')
        )\
        .filter(Metrics.timestamp >= start_time)\
        .group_by('time')\
        .order_by('time')\
        .all()
        
        # If no metrics data, try traffic logs
        if not results:
            results = self.db.query(
                func.strftime('%Y-%m-%d %H:00:00', TrafficLog.timestamp).label('time'),
                func.count(TrafficLog.id).label('requests'),
                func.sum(TrafficLog.was_blocked).label('blocked'),
                func.count(TrafficLog.id) - func.sum(TrafficLog.was_blocked).label('allowed')
            )\
            .filter(TrafficLog.timestamp >= start_time)\
            .group_by('time')\
            .order_by('time')\
            .all()
        
        return [
            {
                "time": row.time,
                "requests": int(row.requests or 0),
                "blocked": int(row.blocked or 0),
                "allowed": int(row.allowed or 0)
            }
            for row in results
        ]
    
    def get_threats_chart_data(self, start_time: datetime) -> List[Dict]:
        """Get threats chart data"""
        # Aggregate by hour and threat type
        results = self.db.query(
            func.strftime('%Y-%m-%d %H:00:00', Threat.timestamp).label('time'),
            Threat.type,
            func.count(Threat.id).label('count')
        )\
        .filter(Threat.timestamp >= start_time)\
        .group_by('time', Threat.type)\
        .order_by('time')\
        .all()
        
        # Group by time
        chart_data = {}
        for row in results:
            if row.time not in chart_data:
                chart_data[row.time] = {
                    "time": row.time,
                    "sql": 0,
                    "xss": 0,
                    "ddos": 0,
                    "other": 0
                }
            
            threat_type = row.type.lower()
            if "sql" in threat_type or "injection" in threat_type:
                chart_data[row.time]["sql"] = int(row.count)
            elif "xss" in threat_type or "cross-site" in threat_type:
                chart_data[row.time]["xss"] = int(row.count)
            elif "ddos" in threat_type or "dos" in threat_type:
                chart_data[row.time]["ddos"] = int(row.count)
            else:
                chart_data[row.time]["other"] = int(row.count)
        
        return list(chart_data.values())
    
    def get_performance_chart_data(self, start_time: datetime) -> List[Dict]:
        """Get performance chart data"""
        results = self.db.query(
            func.strftime('%Y-%m-%d %H:00:00', Metrics.timestamp).label('time'),
            func.avg(Metrics.avg_response_time).label('latency'),
            func.avg(Metrics.cpu_usage).label('cpu'),
            func.avg(Metrics.memory_usage).label('memory')
        )\
        .filter(Metrics.timestamp >= start_time)\
        .group_by('time')\
        .order_by('time')\
        .all()
        
        return [
            {
                "time": row.time,
                "latency": float(row.latency or 0),
                "cpu": float(row.cpu or 0),
                "memory": float(row.memory or 0)
            }
            for row in results
        ]
