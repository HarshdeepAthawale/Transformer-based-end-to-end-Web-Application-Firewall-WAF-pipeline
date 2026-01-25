"""
Metrics database model
"""
from sqlalchemy import Column, Integer, Float, DateTime, String
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime


class Metrics(Base):
    """Metrics table for storing real-time and historical metrics"""
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # Request metrics
    total_requests = Column(Integer, default=0, nullable=False)
    blocked_requests = Column(Integer, default=0, nullable=False)
    allowed_requests = Column(Integer, default=0, nullable=False)
    
    # Attack metrics
    attack_rate = Column(Float, default=0.0, nullable=False)  # Percentage
    threats_per_minute = Column(Float, default=0.0, nullable=False)
    
    # Performance metrics
    avg_response_time = Column(Float, default=0.0, nullable=False)  # milliseconds
    avg_processing_time = Column(Float, default=0.0, nullable=False)  # milliseconds
    
    # System metrics
    cpu_usage = Column(Float, default=0.0, nullable=False)  # Percentage
    memory_usage = Column(Float, default=0.0, nullable=False)  # Percentage
    
    # Connection metrics
    active_connections = Column(Integer, default=0, nullable=False)
    
    # Uptime
    uptime_seconds = Column(Integer, default=0, nullable=False)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "allowed_requests": self.allowed_requests,
            "attack_rate": self.attack_rate,
            "threats_per_minute": self.threats_per_minute,
            "avg_response_time": self.avg_response_time,
            "avg_processing_time": self.avg_processing_time,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "active_connections": self.active_connections,
            "uptime_seconds": self.uptime_seconds,
        }
