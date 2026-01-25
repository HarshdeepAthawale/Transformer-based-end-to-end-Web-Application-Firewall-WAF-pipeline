"""
Metrics Aggregator Background Worker
"""
import threading
import time
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import SessionLocal
from backend.models.metrics import Metrics
from backend.models.traffic import TrafficLog
from backend.models.threats import Threat
from backend.services.metrics_service import MetricsService
from backend.services.websocket_service import broadcast_update_sync
import psutil


class MetricsAggregator:
    """Aggregate metrics periodically"""
    
    def __init__(self, interval_seconds: int = 60):
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        self._start_time = time.time()
    
    def start(self):
        """Start metrics aggregator"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Metrics aggregator started")
    
    def stop(self):
        """Stop metrics aggregator"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Metrics aggregator stopped")
    
    def _run(self):
        """Main aggregation loop"""
        while self.running:
            try:
                self._aggregate_metrics()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                time.sleep(self.interval)
    
    def _aggregate_metrics(self):
        """Aggregate metrics from database"""
        db = SessionLocal()
        try:
            # Get time window (last minute)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=1)
            
            # Aggregate from traffic logs
            traffic_stats = db.query(
                func.count(TrafficLog.id).label('total'),
                func.sum(TrafficLog.was_blocked).label('blocked'),
                func.avg(TrafficLog.processing_time_ms).label('avg_processing_time')
            )\
            .filter(TrafficLog.timestamp >= start_time)\
            .filter(TrafficLog.timestamp < end_time)\
            .first()
            
            total_requests = int(traffic_stats.total or 0)
            blocked_requests = int(traffic_stats.blocked or 0)
            allowed_requests = total_requests - blocked_requests
            
            # Get threat count
            threat_count = db.query(func.count(Threat.id))\
                .filter(Threat.timestamp >= start_time)\
                .filter(Threat.timestamp < end_time)\
                .scalar() or 0
            
            # Calculate metrics
            attack_rate = (blocked_requests / total_requests * 100) if total_requests > 0 else 0.0
            threats_per_minute = float(threat_count)
            avg_processing_time = float(traffic_stats.avg_processing_time or 0)
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Create metrics snapshot
            metrics_service = MetricsService(db)
            metrics = metrics_service.create_metrics_snapshot({
                "total_requests": total_requests,
                "blocked_requests": blocked_requests,
                "allowed_requests": allowed_requests,
                "attack_rate": attack_rate,
                "threats_per_minute": threats_per_minute,
                "avg_processing_time": avg_processing_time,
                "avg_response_time": avg_processing_time,  # Use processing time as proxy
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "active_connections": len(psutil.net_connections()),
                "uptime_seconds": int(time.time() - self._start_time)
            })
            
            # Broadcast metrics update
            realtime_metrics = metrics_service.get_realtime_metrics()
            broadcast_update_sync("metrics", realtime_metrics)
            
            # Broadcast performance metrics separately
            performance_metrics = {
                "cpu": cpu_usage,
                "memory": memory_usage,
                "latency": avg_processing_time,
                "requests_per_second": total_requests / 60.0 if total_requests > 0 else 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            broadcast_update_sync("performance", performance_metrics)
            
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
        finally:
            db.close()


