"""
IP Blacklist/Whitelist database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum, Text
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime
import enum


class IPListType(str, enum.Enum):
    BLACKLIST = "blacklist"
    WHITELIST = "whitelist"


class IPBlockType(str, enum.Enum):
    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    AUTO = "auto"  # Automatically blocked based on reputation


class IPBlacklist(Base):
    """IP blacklist/whitelist table"""
    __tablename__ = "ip_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # IP information
    ip = Column(String(45), nullable=False, index=True)  # IPv4 or IPv6
    ip_range = Column(String(50), nullable=True, index=True)  # CIDR notation (e.g., 192.168.1.0/24)
    is_range = Column(Boolean, default=False, nullable=False)
    
    # List type
    list_type = Column(Enum(IPListType), nullable=False, index=True)
    block_type = Column(Enum(IPBlockType), nullable=False, default=IPBlockType.PERMANENT)
    
    # Blocking details
    reason = Column(Text, nullable=True)
    source = Column(String(100), nullable=False, default="manual")  # manual, auto, threat_intel, etc.
    
    # Temporary block details
    expires_at = Column(DateTime, nullable=True, index=True)
    auto_unblock = Column(Boolean, default=False, nullable=False)
    
    # Metadata
    created_by = Column(String(100), nullable=True)  # User or system
    notes = Column(Text, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    
    # Statistics
    blocked_requests = Column(Integer, default=0, nullable=False)
    last_seen = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "ip": self.ip,
            "ip_range": self.ip_range,
            "is_range": self.is_range,
            "list_type": self.list_type.value if self.list_type else None,
            "block_type": self.block_type.value if self.block_type else None,
            "reason": self.reason,
            "source": self.source,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "auto_unblock": self.auto_unblock,
            "created_by": self.created_by,
            "notes": self.notes,
            "is_active": self.is_active,
            "blocked_requests": self.blocked_requests,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
