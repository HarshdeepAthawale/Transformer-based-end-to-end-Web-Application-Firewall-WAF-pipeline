"""
Audit logging database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Enum, Boolean
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime
import enum


class AuditAction(str, enum.Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VIEW = "view"
    LOGIN = "login"
    LOGOUT = "logout"
    BLOCK = "block"
    UNBLOCK = "unblock"
    CONFIG_CHANGE = "config_change"
    RULE_CHANGE = "rule_change"


class AuditLog(Base):
    """Audit log table for security events"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # User information
    user_id = Column(Integer, nullable=True, index=True)
    username = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True, index=True)
    
    # Action details
    action = Column(Enum(AuditAction), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False, index=True)  # ip, rule, user, config, etc.
    resource_id = Column(Integer, nullable=True)
    
    # Details
    description = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON object with additional details
    
    # Status
    success = Column(Boolean, default=True, nullable=False, index=True)
    error_message = Column(Text, nullable=True)

    def to_dict(self):
        """Convert to dictionary"""
        import json
        details_dict = json.loads(self.details) if self.details else {}
        
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id,
            "username": self.username,
            "ip_address": self.ip_address,
            "action": self.action.value if self.action else None,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "description": self.description,
            "details": details_dict,
            "success": self.success,
            "error_message": self.error_message,
        }
