"""
Traffic logs database model
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Index
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime, timezone


class TrafficLog(Base):
    """Traffic logs table for storing HTTP request logs"""
    __tablename__ = "traffic_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # Request details
    ip = Column(String(45), nullable=False, index=True)  # IPv4 or IPv6
    method = Column(String(10), nullable=False, index=True)  # GET, POST, etc.
    endpoint = Column(String(500), nullable=False, index=True)
    query_string = Column(Text, nullable=True)
    
    # Response details
    status_code = Column(Integer, nullable=False, index=True)
    response_size = Column(Integer, default=0, nullable=False)  # bytes
    
    # Headers
    user_agent = Column(String(500), nullable=True)
    referer = Column(String(500), nullable=True)
    
    # Request body (truncated for large bodies)
    request_body = Column(Text, nullable=True)
    
    # Processing details
    processing_time_ms = Column(Integer, default=0, nullable=False)
    was_blocked = Column(Integer, default=0, nullable=False)  # 0 = allowed, 1 = blocked
    anomaly_score = Column(String(10), nullable=True)  # Store as string to preserve precision
    
    # Additional metadata
    country_code = Column(String(2), nullable=True, index=True)
    threat_type = Column(String(50), nullable=True, index=True)

    def to_dict(self):
        """Convert to dictionary"""
        def format_size(size_bytes):
            """Format bytes to human readable"""
            if size_bytes < 1024:
                return f"{size_bytes}B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f}KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f}MB"
        
        return {
            "id": self.id,
            "ip": self.ip,
            "method": self.method,
            "endpoint": self.endpoint,
            "status": self.status_code,
            "size": format_size(self.response_size),
            "time": self.timestamp.strftime("%H:%M:%S") if self.timestamp else None,
            "timestamp": self.timestamp.replace(tzinfo=timezone.utc).isoformat() if self.timestamp else None,
            "userAgent": self.user_agent,
            "was_blocked": bool(self.was_blocked),
            "anomaly_score": float(self.anomaly_score) if self.anomaly_score else None,
            "threat_type": self.threat_type,
            "country_code": self.country_code,
        }
