"""
User authentication and authorization models
"""
from sqlalchemy import ForeignKey, Column, Integer, String, DateTime, Boolean, Enum, Text
from backend.database import Base
from backend.lib.datetime_utils import utc_now
import enum
import hashlib
from passlib.hash import argon2


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class User(Base):
    """Users table for authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, index=True, nullable=False)
    
    # User information
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)  # Hashed password
    salt = Column(String(32), nullable=False)  # Password salt
    
    # Authorization
    role = Column(Enum(UserRole), nullable=False, default=UserRole.VIEWER, index=True)
    
    # Status
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Metadata
    full_name = Column(String(200), nullable=True)
    last_login = Column(DateTime, nullable=True)
    created_by = Column(String(100), nullable=True)
    
    # API Keys
    api_keys = Column(Text, nullable=True)  # JSON array of API keys

    def set_password(self, password: str):
        """Set password using argon2 (salt is embedded in hash)"""
        self.password_hash = argon2.hash(password)
        self.salt = ""  # argon2 embeds salt; no separate salt needed

    def check_password(self, password: str) -> bool:
        """Check password with backward compatibility for legacy SHA-256 hashes"""
        # If salt exists, this is a legacy SHA-256 hash
        if self.salt:
            sha_hash = hashlib.sha256((password + self.salt).encode()).hexdigest()
            if sha_hash == self.password_hash:
                # Auto-upgrade to argon2 on successful login (caller must db.commit())
                self.set_password(password)
                return True
            return False
        # Otherwise use argon2 verification
        try:
            return argon2.verify(password, self.password_hash)
        except Exception:
            return False

    def to_dict(self):
        """Convert to dictionary (without sensitive data)"""
        import json
        api_keys_list = json.loads(self.api_keys) if self.api_keys else []
        
        return {
            "id": self.id,
            "org_id": self.org_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value if self.role else None,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "full_name": self.full_name,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "api_key_count": len(api_keys_list),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "created_at": self.timestamp.isoformat() if self.timestamp else None,
        }
