"""
Alerts database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime
import enum


class AlertType(str, enum.Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertSeverity(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Alert(Base):
    """Alerts table for storing security alerts"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    type = Column(Enum(AlertType), nullable=False, index=True)
    severity = Column(Enum(AlertSeverity), nullable=False, index=True)
    
    title = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=False)
    
    source = Column(String(100), nullable=False)  # e.g., "waf", "system", "threat_intel"
    icon = Column(String(50), default="alert-circle")
    
    # Alert status
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    is_acknowledged = Column(Boolean, default=False, nullable=False)
    is_dismissed = Column(Boolean, default=False, nullable=False)
    
    # Related data
    related_ip = Column(String(45), nullable=True, index=True)  # IPv4 or IPv6
    related_endpoint = Column(String(500), nullable=True)
    related_threat_id = Column(Integer, nullable=True)
    
    # Actions available
    actions = Column(String(500), nullable=True)  # JSON array of action strings
    
    acknowledged_at = Column(DateTime, nullable=True)
    dismissed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary"""
        import json
        actions_list = json.loads(self.actions) if self.actions else []
        
        return {
            "id": self.id,
            "type": self.type.value if self.type else None,
            "severity": self.severity.value if self.severity else None,
            "title": self.title,
            "description": self.description,
            "time": self.timestamp.strftime("%H:%M:%S") if self.timestamp else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "icon": self.icon,
            "source": self.source,
            "actions": actions_list,
            "is_active": self.is_active,
            "is_acknowledged": self.is_acknowledged,
            "is_dismissed": self.is_dismissed,
            "related_ip": self.related_ip,
            "related_endpoint": self.related_endpoint,
            "related_threat_id": self.related_threat_id,
        }
