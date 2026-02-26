"""
DNS zone and record models.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class DNSZone(Base):
    """DNS zone (domain) managed by the WAF."""

    __tablename__ = "dns_zones"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String(255), nullable=False, unique=True, index=True)
    status = Column(String(20), default="pending")  # pending, active, error
    provider = Column(String(50), default="manual")  # manual, cloudflare, route53, powerdns
    provider_zone_id = Column(String(255), nullable=True)
    nameservers = Column(Text, nullable=True)  # JSON array
    dnssec_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def to_dict(self):
        import json
        ns = None
        if self.nameservers:
            try:
                ns = json.loads(self.nameservers)
            except (json.JSONDecodeError, TypeError):
                ns = self.nameservers
        return {
            "id": self.id,
            "domain": self.domain,
            "status": self.status,
            "provider": self.provider,
            "provider_zone_id": self.provider_zone_id,
            "nameservers": ns,
            "dnssec_enabled": self.dnssec_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DNSRecord(Base):
    """DNS record within a zone."""

    __tablename__ = "dns_records"

    id = Column(Integer, primary_key=True, index=True)
    zone_id = Column(Integer, ForeignKey("dns_zones.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)  # subdomain or @ for apex
    record_type = Column(String(10), nullable=False)  # A, AAAA, CNAME, MX, TXT, SRV, CAA, NS
    content = Column(String(500), nullable=False)  # IP, hostname, text value
    original_content = Column(String(500), nullable=True)  # preserved when proxied=true swaps content
    ttl = Column(Integer, default=300)
    priority = Column(Integer, nullable=True)  # For MX, SRV
    proxied = Column(Boolean, default=False)  # orange cloud: traffic through WAF
    provider_record_id = Column(String(255), nullable=True)
    comment = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "zone_id": self.zone_id,
            "name": self.name,
            "record_type": self.record_type,
            "content": self.content,
            "original_content": self.original_content,
            "ttl": self.ttl,
            "priority": self.priority,
            "proxied": self.proxied,
            "provider_record_id": self.provider_record_id,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
