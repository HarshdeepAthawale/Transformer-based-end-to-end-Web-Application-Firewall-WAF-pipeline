"""
Activities database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Enum, Text
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime
import enum


class ActivityType(str, enum.Enum):
    BLOCKED = "blocked"
    ALLOWED = "allowed"


class Activity(Base):
    """Activities table for storing activity feed events"""
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    type = Column(Enum(ActivityType), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    details = Column(Text, nullable=False)
    
    # Related information
    ip = Column(String(45), nullable=True, index=True)
    endpoint = Column(String(500), nullable=True)
    method = Column(String(10), nullable=True)
    
    # Additional context
    threat_type = Column(String(50), nullable=True)
    anomaly_score = Column(String(10), nullable=True)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value if self.type else None,
            "title": self.title,
            "details": self.details,
            "time": self.timestamp.strftime("%H:%M:%S") if self.timestamp else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "ip": self.ip,
            "endpoint": self.endpoint,
            "method": self.method,
            "threat_type": self.threat_type,
            "anomaly_score": float(self.anomaly_score) if self.anomaly_score else None,
        }
