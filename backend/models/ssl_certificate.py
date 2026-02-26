"""
SSL certificate and TLS settings models.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class SSLCertificate(Base):
    """Stored SSL/TLS certificates."""

    __tablename__ = "ssl_certificates"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String(255), nullable=False, index=True)
    cert_type = Column(String(20), nullable=False, default="acme")  # acme, custom, self_signed
    status = Column(String(20), nullable=False, default="pending")  # pending, active, expired, revoked, error
    issuer = Column(String(255), nullable=True)
    not_before = Column(DateTime, nullable=True)
    not_after = Column(DateTime, nullable=True)
    serial_number = Column(String(100), nullable=True)
    fingerprint_sha256 = Column(String(100), nullable=True)
    cert_pem = Column(Text, nullable=True)
    key_pem_encrypted = Column(Text, nullable=True)
    acme_account_id = Column(String(255), nullable=True)
    auto_renew = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "domain": self.domain,
            "cert_type": self.cert_type,
            "status": self.status,
            "issuer": self.issuer,
            "not_before": self.not_before.isoformat() if self.not_before else None,
            "not_after": self.not_after.isoformat() if self.not_after else None,
            "serial_number": self.serial_number,
            "fingerprint_sha256": self.fingerprint_sha256,
            "auto_renew": self.auto_renew,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SSLSettings(Base):
    """Per-domain SSL/TLS settings."""

    __tablename__ = "ssl_settings"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String(255), nullable=False, unique=True, index=True)
    ssl_mode = Column(String(20), default="full")  # off, flexible, full, full_strict
    min_tls_version = Column(String(10), default="1.2")  # 1.0, 1.1, 1.2, 1.3
    hsts_enabled = Column(Boolean, default=False)
    hsts_max_age = Column(Integer, default=31536000)
    hsts_include_subdomains = Column(Boolean, default=False)
    hsts_preload = Column(Boolean, default=False)
    automatic_https_rewrites = Column(Boolean, default=True)
    tls_0rtt_enabled = Column(Boolean, default=False)
    mtls_enabled = Column(Boolean, default=False)
    mtls_ca_cert_pem = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "domain": self.domain,
            "ssl_mode": self.ssl_mode,
            "min_tls_version": self.min_tls_version,
            "hsts_enabled": self.hsts_enabled,
            "hsts_max_age": self.hsts_max_age,
            "hsts_include_subdomains": self.hsts_include_subdomains,
            "hsts_preload": self.hsts_preload,
            "automatic_https_rewrites": self.automatic_https_rewrites,
            "tls_0rtt_enabled": self.tls_0rtt_enabled,
            "mtls_enabled": self.mtls_enabled,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
