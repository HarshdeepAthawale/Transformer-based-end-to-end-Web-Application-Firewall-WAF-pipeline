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

from src.api.models.metrics import Metrics
from src.api.services.cache_service import cache_service


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
        
        # Get latest metrics from database
        latest = self.db.query(Metrics).order_by(desc(Metrics.timestamp)).first()
        
        if latest:
            # Use latest metrics from database
            metrics = latest.to_dict()
        else:
            # Return default metrics if none exist
            metrics = {
                "requests": 0,
                "blocked": 0,
                "attackRate": 0.0,
                "responseTime": 0.0,
                "threatsPerMinute": 0.0,
                "uptime": int(time.time() - self._start_time),
                "activeConnections": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Calculate real-time values
        total_requests = metrics.get("total_requests", 0)
        blocked_requests = metrics.get("blocked_requests", 0)
        allowed_requests = metrics.get("allowed_requests", 0)
        
        attack_rate = (blocked_requests / total_requests * 100) if total_requests > 0 else 0.0
        
        result = {
            "requests": total_requests,
            "blocked": blocked_requests,
            "attackRate": round(attack_rate, 2),
            "responseTime": metrics.get("avg_response_time", 0.0),
            "threatsPerMinute": metrics.get("threats_per_minute", 0.0),
            "uptime": metrics.get("uptime_seconds", int(time.time() - self._start_time)),
            "activeConnections": metrics.get("active_connections", 0),
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
