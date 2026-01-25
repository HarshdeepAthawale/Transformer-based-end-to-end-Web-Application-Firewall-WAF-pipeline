"""
IP Reputation database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Index
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime


class IPReputation(Base):
    """IP reputation scores and history"""
    __tablename__ = "ip_reputation"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # IP information
    ip = Column(String(45), nullable=False, index=True)  # IPv4 or IPv6
    
    # Reputation score (0.0 = bad, 1.0 = good)
    reputation_score = Column(Float, nullable=False, index=True, default=0.5)
    
    # Score components
    threat_intel_score = Column(Float, default=0.5, nullable=False)  # From threat intelligence
    historical_score = Column(Float, default=0.5, nullable=False)  # Based on history
    recent_activity_score = Column(Float, default=0.5, nullable=False)  # Recent behavior
    geo_score = Column(Float, default=0.5, nullable=False)  # Geographic risk
    
    # Statistics
    total_requests = Column(Integer, default=0, nullable=False)
    blocked_requests = Column(Integer, default=0, nullable=False)
    anomaly_count = Column(Integer, default=0, nullable=False)
    threat_count = Column(Integer, default=0, nullable=False)
    
    # Time-based metrics
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True, index=True)
    
    # Additional data
    country_code = Column(String(2), nullable=True, index=True)
    asn = Column(String(50), nullable=True)  # Autonomous System Number
    isp = Column(String(200), nullable=True)
    
    # Metadata
    notes = Column(Text, nullable=True)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "ip": self.ip,
            "reputation_score": self.reputation_score,
            "threat_intel_score": self.threat_intel_score,
            "historical_score": self.historical_score,
            "recent_activity_score": self.recent_activity_score,
            "geo_score": self.geo_score,
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "anomaly_count": self.anomaly_count,
            "threat_count": self.threat_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "country_code": self.country_code,
            "asn": self.asn,
            "isp": self.isp,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
