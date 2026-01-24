"""
Geo-fencing rules database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum, Text
from sqlalchemy.sql import func
from src.api.database import Base
from datetime import datetime
import enum


class GeoRuleType(str, enum.Enum):
    ALLOW = "allow"  # Allow list - only allow these countries
    DENY = "deny"  # Deny list - block these countries


class GeoRule(Base):
    """Geo-fencing rules table"""
    __tablename__ = "geo_rules"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # Rule type
    rule_type = Column(Enum(GeoRuleType), nullable=False, index=True)
    
    # Country information
    country_code = Column(String(2), nullable=False, index=True)  # ISO 3166-1 alpha-2
    country_name = Column(String(100), nullable=False)
    
    # Rule details
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    priority = Column(Integer, default=0, nullable=False)  # Higher priority rules evaluated first
    
    # Exception IPs (IPs that bypass this rule)
    exception_ips = Column(Text, nullable=True)  # JSON array of IPs
    
    # Metadata
    reason = Column(Text, nullable=True)
    created_by = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Statistics
    blocked_requests = Column(Integer, default=0, nullable=False)
    allowed_requests = Column(Integer, default=0, nullable=False)
    last_applied = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary"""
        import json
        exception_ips_list = json.loads(self.exception_ips) if self.exception_ips else []
        
        return {
            "id": self.id,
            "rule_type": self.rule_type.value if self.rule_type else None,
            "country_code": self.country_code,
            "country_name": self.country_name,
            "is_active": self.is_active,
            "priority": self.priority,
            "exception_ips": exception_ips_list,
            "reason": self.reason,
            "created_by": self.created_by,
            "notes": self.notes,
            "blocked_requests": self.blocked_requests,
            "allowed_requests": self.allowed_requests,
            "last_applied": self.last_applied.isoformat() if self.last_applied else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
