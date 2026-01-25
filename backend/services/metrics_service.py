"""
Metrics Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import List, Optional
import psutil
import time
from loguru import logger

from backend.models.metrics import Metrics
from backend.models.traffic import TrafficLog
from backend.models.threats import Threat
from backend.services.cache_service import cache_service


class MetricsService:
    """Service for managing metrics"""
    
    def __init__(self, db: Session):
        self.db = db
        self._start_time = time.time()
    
    def get_realtime_metrics(self) -> dict:
        """Get real-time metrics"""
        # Try cache first
        cache_key = "metrics:realtime"
        cached = cache_service.get(cache_key)
        if cached:
            return cached
        
        # Get metrics from last 5 minutes of traffic logs
        five_min_ago = datetime.utcnow() - timedelta(minutes=5)
        
        traffic_stats = self.db.query(
            func.count(TrafficLog.id).label('total'),
            func.sum(TrafficLog.was_blocked).label('blocked'),
            func.avg(TrafficLog.processing_time_ms).label('avg_time')
        ).filter(TrafficLog.timestamp >= five_min_ago).first()
        
        total_requests = int(traffic_stats.total or 0)
        blocked_requests = int(traffic_stats.blocked or 0)
        allowed_requests = total_requests - blocked_requests
        attack_rate = (blocked_requests / total_requests * 100) if total_requests > 0 else 0.0
        avg_processing_time = float(traffic_stats.avg_time or 0)
        
        # Get threat count
        threat_count = self.db.query(func.count(Threat.id))\
            .filter(Threat.timestamp >= five_min_ago)\
            .scalar() or 0
        threats_per_minute = threat_count / 5.0  # Per minute
        
        # Get active WebSocket connections
        try:
            from backend.websocket import manager
            active_connections = manager.get_connection_count()
        except Exception:
            active_connections = 0
        
        result = {
            "requests": total_requests,
            "blocked": blocked_requests,
            "attackRate": round(attack_rate, 2),
            "responseTime": round(avg_processing_time, 2),
            "threatsPerMinute": round(threats_per_minute, 2),
            "uptime": int(time.time() - self._start_time),
            "activeConnections": active_connections,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache for 5 seconds
        cache_service.set(cache_key, result, ttl=5)
        return result
    
    def get_historical_metrics(self, start_time: datetime) -> List[Metrics]:
        """Get historical metrics"""
        return self.db.query(Metrics)\
            .filter(Metrics.timestamp >= start_time)\
            .order_by(Metrics.timestamp)\
            .all()
    
    def create_metrics_snapshot(self, metrics_data: dict) -> Metrics:
        """Create a metrics snapshot"""
        metrics = Metrics(**metrics_data)
        self.db.add(metrics)
        self.db.commit()
        self.db.refresh(metrics)
        return metrics
