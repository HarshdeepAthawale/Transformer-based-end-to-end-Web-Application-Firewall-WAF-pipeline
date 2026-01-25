"""
IP Fencing Service
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import ipaddress
from loguru import logger

from backend.models.ip_blacklist import IPBlacklist, IPListType, IPBlockType
from backend.models.ip_reputation import IPReputation


class IPFencingService:
    """Service for IP fencing and reputation management"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def is_ip_blocked(self, ip: str) -> Tuple[bool, Optional[str]]:
        """
        Check if IP is blocked
        Returns: (is_blocked, reason)
        """
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return False, None
        
        # Check exact IP match
        exact_match = self.db.query(IPBlacklist)\
            .filter(IPBlacklist.ip == ip)\
            .filter(IPBlacklist.is_active == True)\
            .filter(IPBlacklist.list_type == IPListType.BLACKLIST)\
            .first()
        
        if exact_match:
            # Check if temporary block expired
            if exact_match.expires_at and exact_match.expires_at < datetime.utcnow():
                exact_match.is_active = False
                self.db.commit()
                return False, None
            return True, exact_match.reason or "IP is blacklisted"
        
        # Check IP ranges (CIDR)
        ip_ranges = self.db.query(IPBlacklist)\
            .filter(IPBlacklist.is_range == True)\
            .filter(IPBlacklist.is_active == True)\
            .filter(IPBlacklist.list_type == IPListType.BLACKLIST)\
            .all()
        
        for ip_range in ip_ranges:
            try:
                network = ipaddress.ip_network(ip_range.ip_range, strict=False)
                if ip_obj in network:
                    return True, ip_range.reason or "IP is in blacklisted range"
            except ValueError:
                continue
        
        return False, None
    
    def is_ip_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return False
        
        # Check exact IP match
        exact_match = self.db.query(IPBlacklist)\
            .filter(IPBlacklist.ip == ip)\
            .filter(IPBlacklist.is_active == True)\
            .filter(IPBlacklist.list_type == IPListType.WHITELIST)\
            .first()
        
        if exact_match:
            return True
        
        # Check IP ranges
        ip_ranges = self.db.query(IPBlacklist)\
            .filter(IPBlacklist.is_range == True)\
            .filter(IPBlacklist.is_active == True)\
            .filter(IPBlacklist.list_type == IPListType.WHITELIST)\
            .all()
        
        for ip_range in ip_ranges:
            try:
                network = ipaddress.ip_network(ip_range.ip_range, strict=False)
                if ip_obj in network:
                    return True
            except ValueError:
                continue
        
        return False
    
    def get_ip_reputation(self, ip: str) -> Optional[IPReputation]:
        """Get IP reputation"""
        return self.db.query(IPReputation)\
            .filter(IPReputation.ip == ip)\
            .order_by(IPReputation.timestamp.desc())\
            .first()
    
    def update_ip_reputation(
        self,
        ip: str,
        threat_intel_score: float = None,
        historical_score: float = None,
        recent_activity_score: float = None,
        geo_score: float = None,
        country_code: str = None,
        asn: str = None,
        isp: str = None
    ) -> IPReputation:
        """Update or create IP reputation"""
        reputation = self.get_ip_reputation(ip)
        
        if not reputation:
            reputation = IPReputation(
                ip=ip,
                first_seen=datetime.utcnow(),
                country_code=country_code,
                asn=asn,
                isp=isp,
                reputation_score=0.5,
                threat_intel_score=0.5,
                historical_score=0.5,
                recent_activity_score=0.5,
                geo_score=0.5,
                total_requests=0,
                blocked_requests=0,
                anomaly_count=0,
                threat_count=0
            )
            self.db.add(reputation)
            self.db.flush()  # Flush to get the object with defaults applied
        
        # Update scores
        if threat_intel_score is not None:
            reputation.threat_intel_score = threat_intel_score
        if historical_score is not None:
            reputation.historical_score = historical_score
        if recent_activity_score is not None:
            reputation.recent_activity_score = recent_activity_score
        if geo_score is not None:
            reputation.geo_score = geo_score
        
        # Ensure all score fields are numbers (handle None values)
        threat_intel = float(reputation.threat_intel_score or 0.5)
        historical = float(reputation.historical_score or 0.5)
        recent_activity = float(reputation.recent_activity_score or 0.5)
        geo = float(reputation.geo_score or 0.5)
        
        # Calculate overall reputation (weighted average)
        weights = {
            'threat_intel': 0.4,
            'historical': 0.3,
            'recent_activity': 0.2,
            'geo': 0.1
        }
        
        reputation.reputation_score = (
            threat_intel * weights['threat_intel'] +
            historical * weights['historical'] +
            recent_activity * weights['recent_activity'] +
            geo * weights['geo']
        )
        
        reputation.last_seen = datetime.utcnow()
        if country_code:
            reputation.country_code = country_code
        if asn:
            reputation.asn = asn
        if isp:
            reputation.isp = isp
        
        self.db.commit()
        self.db.refresh(reputation)
        return reputation
    
    def increment_ip_stats(self, ip: str, was_blocked: bool = False, is_anomaly: bool = False, is_threat: bool = False):
        """Increment IP statistics"""
        reputation = self.get_ip_reputation(ip)
        
        if not reputation:
            reputation = IPReputation(
                ip=ip,
                first_seen=datetime.utcnow(),
                total_requests=0,
                blocked_requests=0,
                anomaly_count=0,
                threat_count=0,
                reputation_score=0.5,
                threat_intel_score=0.5,
                historical_score=0.5,
                recent_activity_score=0.5,
                geo_score=0.5
            )
            self.db.add(reputation)
            self.db.flush()  # Flush to get the object with defaults applied
        
        # Ensure all fields are initialized (handle None values)
        # This is a safety check in case fields are None (shouldn't happen with defaults, but be safe)
        if reputation.total_requests is None:
            reputation.total_requests = 0
        if reputation.blocked_requests is None:
            reputation.blocked_requests = 0
        if reputation.anomaly_count is None:
            reputation.anomaly_count = 0
        if reputation.threat_count is None:
            reputation.threat_count = 0
        
        # Increment counters (all fields are guaranteed to be integers now)
        reputation.total_requests = (reputation.total_requests or 0) + 1
        if was_blocked:
            reputation.blocked_requests = (reputation.blocked_requests or 0) + 1
        if is_anomaly:
            reputation.anomaly_count = (reputation.anomaly_count or 0) + 1
        if is_threat:
            reputation.threat_count = (reputation.threat_count or 0) + 1
        
        reputation.last_seen = datetime.utcnow()
        
        # Update recent activity score based on behavior
        if reputation.total_requests > 0:
            block_rate = reputation.blocked_requests / reputation.total_requests
            anomaly_rate = reputation.anomaly_count / reputation.total_requests
            
            # Lower score for higher block/anomaly rates
            reputation.recent_activity_score = max(0.0, min(1.0, 1.0 - (block_rate * 0.5 + anomaly_rate * 0.5)))
        
        self.db.commit()
    
    def auto_block_ip(self, ip: str, reason: str, duration_hours: int = 24) -> Optional[IPBlacklist]:
        """Automatically block IP based on reputation"""
        reputation = self.get_ip_reputation(ip)
        
        if not reputation or reputation.reputation_score >= 0.3:
            return None  # Don't auto-block if reputation is acceptable
        
        # Check if already blocked
        existing = self.db.query(IPBlacklist)\
            .filter(IPBlacklist.ip == ip)\
            .filter(IPBlacklist.is_active == True)\
            .filter(IPBlacklist.list_type == IPListType.BLACKLIST)\
            .first()
        
        if existing:
            return existing
        
        # Create auto-block
        expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
        block = IPBlacklist(
            ip=ip,
            list_type=IPListType.BLACKLIST,
            block_type=IPBlockType.AUTO,
            reason=reason,
            source="auto",
            expires_at=expires_at,
            auto_unblock=True
        )
        
        self.db.add(block)
        self.db.commit()
        self.db.refresh(block)
        
        logger.info(f"Auto-blocked IP {ip} for {duration_hours} hours: {reason}")
        return block
    
    def add_to_blacklist(
        self,
        ip: str,
        reason: str,
        source: str = "manual",
        duration_hours: Optional[int] = None,
        created_by: str = None
    ) -> IPBlacklist:
        """Add IP to blacklist"""
        expires_at = None
        if duration_hours:
            expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
        
        block = IPBlacklist(
            ip=ip,
            list_type=IPListType.BLACKLIST,
            block_type=IPBlockType.TEMPORARY if duration_hours else IPBlockType.PERMANENT,
            reason=reason,
            source=source,
            expires_at=expires_at,
            auto_unblock=bool(duration_hours),
            created_by=created_by
        )
        
        self.db.add(block)
        self.db.commit()
        self.db.refresh(block)
        return block
    
    def add_to_whitelist(self, ip: str, reason: str, created_by: str = None) -> IPBlacklist:
        """Add IP to whitelist"""
        whitelist = IPBlacklist(
            ip=ip,
            list_type=IPListType.WHITELIST,
            block_type=IPBlockType.PERMANENT,
            reason=reason,
            source="manual",
            created_by=created_by
        )
        
        self.db.add(whitelist)
        self.db.commit()
        self.db.refresh(whitelist)
        return whitelist
    
    def remove_from_list(self, ip: str, list_type: IPListType) -> bool:
        """Remove IP from blacklist or whitelist"""
        entries = self.db.query(IPBlacklist)\
            .filter(IPBlacklist.ip == ip)\
            .filter(IPBlacklist.list_type == list_type)\
            .filter(IPBlacklist.is_active == True)\
            .all()
        
        for entry in entries:
            entry.is_active = False
        
        self.db.commit()
        return len(entries) > 0
    
    def get_blacklist(self, limit: int = 100) -> List[IPBlacklist]:
        """Get active blacklist entries"""
        return self.db.query(IPBlacklist)\
            .filter(IPBlacklist.list_type == IPListType.BLACKLIST)\
            .filter(IPBlacklist.is_active == True)\
            .order_by(IPBlacklist.timestamp.desc())\
            .limit(limit)\
            .all()
    
    def get_whitelist(self, limit: int = 100) -> List[IPBlacklist]:
        """Get active whitelist entries"""
        return self.db.query(IPBlacklist)\
            .filter(IPBlacklist.list_type == IPListType.WHITELIST)\
            .filter(IPBlacklist.is_active == True)\
            .order_by(IPBlacklist.timestamp.desc())\
            .limit(limit)\
            .all()
    
    def cleanup_expired_blocks(self):
        """Remove expired temporary blocks"""
        expired = self.db.query(IPBlacklist)\
            .filter(IPBlacklist.expires_at < datetime.utcnow())\
            .filter(IPBlacklist.is_active == True)\
            .filter(IPBlacklist.auto_unblock == True)\
            .all()
        
        for block in expired:
            block.is_active = False
            logger.info(f"Auto-unblocked expired IP: {block.ip}")
        
        self.db.commit()
        return len(expired)
