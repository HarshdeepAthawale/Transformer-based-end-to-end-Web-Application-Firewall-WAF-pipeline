"""
IP Reputation Updater Background Worker
"""
import threading
import time
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.services.ip_fencing import IPFencingService
from backend.models.traffic import TrafficLog
from sqlalchemy import func


class IPReputationUpdater:
    """Update IP reputation scores periodically"""
    
    def __init__(self, interval_seconds: int = 3600):  # 1 hour
        self.interval = interval_seconds
        self.running = False
        self.thread = None
    
    def start(self):
        """Start IP reputation updater"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("IP reputation updater started")
    
    def stop(self):
        """Stop IP reputation updater"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("IP reputation updater stopped")
    
    def _run(self):
        """Main update loop"""
        while self.running:
            try:
                self._update_reputations()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"IP reputation update error: {e}")
                time.sleep(self.interval)
    
    def _update_reputations(self):
        """Update IP reputation scores"""
        db = SessionLocal()
        try:
            ip_fencing = IPFencingService(db)
            
            # Get IPs with recent activity (last 24 hours)
            start_time = datetime.utcnow() - timedelta(hours=24)
            
            ips_with_activity = db.query(
                TrafficLog.ip,
                func.count(TrafficLog.id).label('total'),
                func.sum(TrafficLog.was_blocked).label('blocked'),
                func.sum(func.cast(TrafficLog.anomaly_score, func.Float)).label('avg_anomaly')
            )\
            .filter(TrafficLog.timestamp >= start_time)\
            .group_by(TrafficLog.ip)\
            .all()
            
            for ip, total, blocked, avg_anomaly in ips_with_activity:
                # Calculate historical score
                block_rate = (blocked or 0) / total if total > 0 else 0
                anomaly_rate = float(avg_anomaly or 0) / total if total > 0 else 0
                
                historical_score = max(0.0, min(1.0, 1.0 - (block_rate * 0.6 + anomaly_rate * 0.4)))
                
                # Update reputation
                ip_fencing.update_ip_reputation(
                    ip=ip,
                    historical_score=historical_score,
                    recent_activity_score=historical_score  # Use same for now
                )
                
                # Auto-block if reputation is very low
                if historical_score < 0.2:
                    ip_fencing.auto_block_ip(
                        ip=ip,
                        reason=f"Low reputation score: {historical_score:.2f}",
                        duration_hours=24
                    )
            
            # Cleanup expired blocks
            cleaned = ip_fencing.cleanup_expired_blocks()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired IP blocks")
            
            logger.info(f"Updated reputation for {len(ips_with_activity)} IPs")
        
        except Exception as e:
            logger.error(f"Error updating IP reputations: {e}")
        finally:
            db.close()
