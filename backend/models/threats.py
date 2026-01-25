"""
Threats database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum, Text
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime
import enum


class ThreatSeverity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Threat(Base):
    """Threats table for storing detected threats"""
    __tablename__ = "threats"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # Threat classification
    type = Column(String(50), nullable=False, index=True)  # SQL injection, XSS, DDoS, etc.
    severity = Column(Enum(ThreatSeverity), nullable=False, index=True)
    
    # Source information
    source_ip = Column(String(45), nullable=False, index=True)
    endpoint = Column(String(500), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    
    # Detection details
    blocked = Column(Boolean, default=False, nullable=False, index=True)
    anomaly_score = Column(String(10), nullable=True)  # Store as string
    
    # Threat details
    details = Column(Text, nullable=True)  # JSON or text description
    payload = Column(Text, nullable=True)  # The malicious payload detected
    
    # Related data
    user_agent = Column(String(500), nullable=True)
    country_code = Column(String(2), nullable=True, index=True)
    
    # Processing info
    processing_time_ms = Column(Integer, default=0, nullable=False)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity.value if self.severity else None,
            "source": self.source_ip,
            "endpoint": self.endpoint,
            "blocked": self.blocked,
            "time": self.timestamp.strftime("%H:%M:%S") if self.timestamp else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "details": self.details,
            "anomaly_score": float(self.anomaly_score) if self.anomaly_score else None,
            "payload": self.payload[:100] + "..." if self.payload and len(self.payload) > 100 else self.payload,
            "user_agent": self.user_agent,
            "country_code": self.country_code,
        }
