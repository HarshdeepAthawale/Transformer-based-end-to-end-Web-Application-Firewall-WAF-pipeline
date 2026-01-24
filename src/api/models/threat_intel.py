"""
Threat intelligence database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, Index
from sqlalchemy.sql import func
from src.api.database import Base
from datetime import datetime


class ThreatIntel(Base):
    """Threat intelligence data table"""
    __tablename__ = "threat_intel"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # Threat information
    threat_type = Column(String(50), nullable=False, index=True)  # ip, domain, signature, etc.
    value = Column(String(500), nullable=False, index=True)  # IP, domain, signature pattern
    
    # Threat details
    severity = Column(String(20), nullable=False, index=True)  # critical, high, medium, low
    category = Column(String(50), nullable=False)  # malware, phishing, botnet, etc.
    description = Column(Text, nullable=True)
    
    # Source information
    source = Column(String(100), nullable=False, index=True)  # AbuseIPDB, VirusTotal, etc.
    source_url = Column(String(500), nullable=True)
    source_confidence = Column(Float, default=0.5, nullable=False)  # 0.0 to 1.0
    
    # Metadata
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=True, index=True)
    
    # Additional data
    country_code = Column(String(2), nullable=True, index=True)
    asn = Column(String(50), nullable=True)
    tags = Column(Text, nullable=True)  # JSON array of tags
    
    # Status
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    def to_dict(self):
        """Convert to dictionary"""
        import json
        tags_list = json.loads(self.tags) if self.tags else []
        
        return {
            "id": self.id,
            "threat_type": self.threat_type,
            "value": self.value,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "source": self.source,
            "source_url": self.source_url,
            "source_confidence": self.source_confidence,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "country_code": self.country_code,
            "asn": self.asn,
            "tags": tags_list,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
