"""
Threat Intelligence Service
"""
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger

from src.api.models.threat_intel import ThreatIntel


class ThreatIntelService:
    """Service for threat intelligence"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def check_threat(
        self,
        ip: str,
        path: str = None,
        query_params: Dict = None,
        body: str = None
    ) -> Dict:
        """
        Check if IP/path/query/body matches threat intelligence
        Returns: {
            'is_threat': bool,
            'threat_type': str,
            'severity': str,
            'source': str,
            'description': str
        }
        """
        # Check IP threats
        ip_threat = self.db.query(ThreatIntel)\
            .filter(ThreatIntel.threat_type == 'ip')\
            .filter(ThreatIntel.value == ip)\
            .filter(ThreatIntel.is_active == True)\
            .first()
        
        if ip_threat:
            # Check if expired
            if ip_threat.expires_at and ip_threat.expires_at < datetime.utcnow():
                ip_threat.is_active = False
                self.db.commit()
            else:
                return {
                    'is_threat': True,
                    'threat_type': 'ip',
                    'severity': ip_threat.severity,
                    'source': ip_threat.source,
                    'description': ip_threat.description or f"IP {ip} is in threat database"
                }
        
        # Check domain/path threats
        if path:
            domain_threats = self.db.query(ThreatIntel)\
                .filter(ThreatIntel.threat_type == 'domain')\
                .filter(ThreatIntel.is_active == True)\
                .all()
            
            for threat in domain_threats:
                if threat.value in path:
                    return {
                        'is_threat': True,
                        'threat_type': 'domain',
                        'severity': threat.severity,
                        'source': threat.source,
                        'description': threat.description or f"Path contains known malicious domain"
                    }
        
        # Check signature threats
        if query_params or body:
            combined_text = f"{str(query_params)} {body or ''}".lower()
            
            signature_threats = self.db.query(ThreatIntel)\
                .filter(ThreatIntel.threat_type == 'signature')\
                .filter(ThreatIntel.is_active == True)\
                .all()
            
            for threat in signature_threats:
                import re
                try:
                    if re.search(threat.value, combined_text, re.IGNORECASE):
                        return {
                            'is_threat': True,
                            'threat_type': 'signature',
                            'severity': threat.severity,
                            'source': threat.source,
                            'description': threat.description or "Malicious signature detected"
                        }
                except re.error:
                    logger.warning(f"Invalid regex in threat intel {threat.id}: {threat.value}")
                    continue
        
        return {
            'is_threat': False,
            'threat_type': None,
            'severity': None,
            'source': None,
            'description': None
        }
    
    def add_threat(
        self,
        threat_type: str,
        value: str,
        severity: str,
        category: str,
        source: str,
        description: str = None,
        expires_at: datetime = None
    ) -> ThreatIntel:
        """Add threat intelligence entry"""
        threat = ThreatIntel(
            threat_type=threat_type,
            value=value,
            severity=severity,
            category=category,
            source=source,
            description=description,
            expires_at=expires_at,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        self.db.add(threat)
        self.db.commit()
        self.db.refresh(threat)
        return threat
    
    def get_threats(
        self,
        threat_type: str = None,
        active_only: bool = True,
        limit: int = 100
    ) -> List[ThreatIntel]:
        """Get threat intelligence entries"""
        query = self.db.query(ThreatIntel)
        
        if threat_type:
            query = query.filter(ThreatIntel.threat_type == threat_type)
        if active_only:
            query = query.filter(ThreatIntel.is_active == True)
        
        return query.order_by(ThreatIntel.timestamp.desc()).limit(limit).all()
    
    def cleanup_expired(self):
        """Remove expired threat intelligence entries"""
        expired = self.db.query(ThreatIntel)\
            .filter(ThreatIntel.expires_at < datetime.utcnow())\
            .filter(ThreatIntel.is_active == True)\
            .all()
        
        for threat in expired:
            threat.is_active = False
        
        self.db.commit()
        return len(expired)
