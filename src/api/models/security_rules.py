"""
Security rules database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum, Text
from sqlalchemy.sql import func
from src.api.database import Base
from datetime import datetime
import enum


class RulePriority(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuleAction(str, enum.Enum):
    BLOCK = "block"
    LOG = "log"
    ALERT = "alert"
    REDIRECT = "redirect"
    CHALLENGE = "challenge"


class SecurityRule(Base):
    """Security rules table"""
    __tablename__ = "security_rules"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # Rule information
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    rule_type = Column(String(50), nullable=False, index=True)  # pattern, header, body, ip, geo, etc.
    
    # Rule configuration
    pattern = Column(Text, nullable=True)  # Regex pattern or rule definition (JSON)
    match_conditions = Column(Text, nullable=True)  # JSON object with conditions
    
    # Rule scope
    applies_to = Column(String(50), nullable=False, default="all")  # all, headers, body, query, path
    
    # Action
    action = Column(Enum(RuleAction), nullable=False, default=RuleAction.BLOCK)
    priority = Column(Enum(RulePriority), nullable=False, default=RulePriority.MEDIUM, index=True)
    
    # OWASP Top 10 mapping
    owasp_category = Column(String(50), nullable=True)  # A01, A02, etc.
    cwe_id = Column(String(20), nullable=True)  # CWE identifier
    
    # Status
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    is_system_rule = Column(Boolean, default=False, nullable=False)  # System rules cannot be deleted
    
    # Metadata
    created_by = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Statistics
    match_count = Column(Integer, default=0, nullable=False)
    last_matched = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary"""
        import json
        match_conditions_dict = json.loads(self.match_conditions) if self.match_conditions else {}
        
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type,
            "pattern": self.pattern,
            "match_conditions": match_conditions_dict,
            "applies_to": self.applies_to,
            "action": self.action.value if self.action else None,
            "priority": self.priority.value if self.priority else None,
            "owasp_category": self.owasp_category,
            "cwe_id": self.cwe_id,
            "is_active": self.is_active,
            "is_system_rule": self.is_system_rule,
            "created_by": self.created_by,
            "notes": self.notes,
            "match_count": self.match_count,
            "last_matched": self.last_matched.isoformat() if self.last_matched else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
