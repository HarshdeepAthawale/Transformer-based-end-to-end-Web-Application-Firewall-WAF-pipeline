"""
Bot signatures database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum, Text
from sqlalchemy.sql import func
from src.api.database import Base
from datetime import datetime
import enum


class BotCategory(str, enum.Enum):
    SEARCH_ENGINE = "search_engine"  # Googlebot, Bingbot, etc.
    SCRAPER = "scraper"  # Web scrapers
    MALICIOUS = "malicious"  # Malicious bots
    MONITORING = "monitoring"  # Uptime monitors
    UNKNOWN = "unknown"


class BotSignature(Base):
    """Bot signatures table"""
    __tablename__ = "bot_signatures"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # Signature details
    user_agent_pattern = Column(String(500), nullable=False, index=True)  # Regex pattern
    name = Column(String(100), nullable=False)
    category = Column(Enum(BotCategory), nullable=False, index=True)
    
    # Behavior patterns
    request_rate_limit = Column(Integer, nullable=True)  # Requests per minute
    requires_javascript = Column(Boolean, default=False, nullable=False)
    requires_cookies = Column(Boolean, default=False, nullable=False)
    
    # Action
    action = Column(String(50), nullable=False, default="block")  # block, allow, challenge, monitor
    is_whitelisted = Column(Boolean, default=False, index=True, nullable=False)
    
    # Metadata
    source = Column(String(100), nullable=False, default="manual")
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    
    # Statistics
    detection_count = Column(Integer, default=0, nullable=False)
    last_detected = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_agent_pattern": self.user_agent_pattern,
            "name": self.name,
            "category": self.category.value if self.category else None,
            "request_rate_limit": self.request_rate_limit,
            "requires_javascript": self.requires_javascript,
            "requires_cookies": self.requires_cookies,
            "action": self.action,
            "is_whitelisted": self.is_whitelisted,
            "source": self.source,
            "notes": self.notes,
            "is_active": self.is_active,
            "detection_count": self.detection_count,
            "last_detected": self.last_detected.isoformat() if self.last_detected else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
