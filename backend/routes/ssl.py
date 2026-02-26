"""
Phase 1 — SSL/TLS Management API.
CRUD for certificates and per-domain SSL settings (modes, HSTS, min TLS, mTLS).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import SSLCertificate, SSLSettings

router = APIRouter()


# ─── Certificate schemas ─────────────────────────────────────────────────

class SSLCertificateCreate(BaseModel):
    domain: str
    cert_type: str = "custom"  # acme, custom, self_signed
    cert_pem: Optional[str] = None
    key_pem_encrypted: Optional[str] = None  # in production encrypt with CERT_ENCRYPTION_KEY
    auto_renew: bool = True


class SSLCertificateUpdate(BaseModel):
    status: Optional[str] = None
    cert_pem: Optional[str] = None
    key_pem_encrypted: Optional[str] = None
    auto_renew: Optional[bool] = None


# ─── Settings schemas ────────────────────────────────────────────────────

class SSLSettingsCreate(BaseModel):
    domain: str
    ssl_mode: str = "full"  # off, flexible, full, full_strict
    min_tls_version: str = "1.2"
    hsts_enabled: bool = False
    hsts_max_age: int = 31536000
    hsts_include_subdomains: bool = False
    hsts_preload: bool = False
    automatic_https_rewrites: bool = True
    tls_0rtt_enabled: bool = False
    mtls_enabled: bool = False
    mtls_ca_cert_pem: Optional[str] = None


class SSLSettingsUpdate(BaseModel):
    ssl_mode: Optional[str] = None
    min_tls_version: Optional[str] = None
    hsts_enabled: Optional[bool] = None
    hsts_max_age: Optional[int] = None
    hsts_include_subdomains: Optional[bool] = None
    hsts_preload: Optional[bool] = None
    automatic_https_rewrites: Optional[bool] = None
    tls_0rtt_enabled: Optional[bool] = None
    mtls_enabled: Optional[bool] = None
    mtls_ca_cert_pem: Optional[str] = None


# ─── Certificates ────────────────────────────────────────────────────────

@router.get("/certificates")
def list_certificates(
    db: Session = Depends(get_db),
    domain: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    q = db.query(SSLCertificate)
    if domain:
        q = q.filter(SSLCertificate.domain == domain)
    if status:
        q = q.filter(SSLCertificate.status == status)
    certs = q.order_by(SSLCertificate.domain).offset(skip).limit(limit).all()
    return [c.to_dict() for c in certs]


@router.post("/certificates", status_code=201)
def create_certificate(body: SSLCertificateCreate, db: Session = Depends(get_db)):
    cert = SSLCertificate(
        domain=body.domain,
        cert_type=body.cert_type,
        status="pending",
        cert_pem=body.cert_pem,
        key_pem_encrypted=body.key_pem_encrypted,
        auto_renew=body.auto_renew,
    )
    db.add(cert)
    db.commit()
    db.refresh(cert)
    return cert.to_dict()


@router.get("/certificates/{cert_id}")
def get_certificate(cert_id: int, db: Session = Depends(get_db)):
    cert = db.query(SSLCertificate).filter(SSLCertificate.id == cert_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return cert.to_dict()


@router.patch("/certificates/{cert_id}")
def update_certificate(cert_id: int, body: SSLCertificateUpdate, db: Session = Depends(get_db)):
    cert = db.query(SSLCertificate).filter(SSLCertificate.id == cert_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")
    if body.status is not None:
        cert.status = body.status
    if body.cert_pem is not None:
        cert.cert_pem = body.cert_pem
    if body.key_pem_encrypted is not None:
        cert.key_pem_encrypted = body.key_pem_encrypted
    if body.auto_renew is not None:
        cert.auto_renew = body.auto_renew
    db.commit()
    db.refresh(cert)
    return cert.to_dict()


@router.delete("/certificates/{cert_id}", status_code=204)
def delete_certificate(cert_id: int, db: Session = Depends(get_db)):
    cert = db.query(SSLCertificate).filter(SSLCertificate.id == cert_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")
    db.delete(cert)
    db.commit()
    return None


# ─── SSL Settings (per domain) ───────────────────────────────────────────

@router.get("/settings")
def list_settings(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
):
    settings = db.query(SSLSettings).order_by(SSLSettings.domain).offset(skip).limit(limit).all()
    return [s.to_dict() for s in settings]


@router.get("/settings/{domain}")
def get_settings(domain: str, db: Session = Depends(get_db)):
    s = db.query(SSLSettings).filter(SSLSettings.domain == domain).first()
    if not s:
        raise HTTPException(status_code=404, detail="SSL settings not found for domain")
    return s.to_dict()


@router.post("/settings", status_code=201)
def create_settings(body: SSLSettingsCreate, db: Session = Depends(get_db)):
    if db.query(SSLSettings).filter(SSLSettings.domain == body.domain).first():
        raise HTTPException(status_code=400, detail="Settings for this domain already exist; use PATCH to update")
    s = SSLSettings(
        domain=body.domain,
        ssl_mode=body.ssl_mode,
        min_tls_version=body.min_tls_version,
        hsts_enabled=body.hsts_enabled,
        hsts_max_age=body.hsts_max_age,
        hsts_include_subdomains=body.hsts_include_subdomains,
        hsts_preload=body.hsts_preload,
        automatic_https_rewrites=body.automatic_https_rewrites,
        tls_0rtt_enabled=body.tls_0rtt_enabled,
        mtls_enabled=body.mtls_enabled,
        mtls_ca_cert_pem=body.mtls_ca_cert_pem,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s.to_dict()


@router.patch("/settings/{domain}")
def update_settings(domain: str, body: SSLSettingsUpdate, db: Session = Depends(get_db)):
    s = db.query(SSLSettings).filter(SSLSettings.domain == domain).first()
    if not s:
        raise HTTPException(status_code=404, detail="SSL settings not found for domain")
    if body.ssl_mode is not None:
        s.ssl_mode = body.ssl_mode
    if body.min_tls_version is not None:
        s.min_tls_version = body.min_tls_version
    if body.hsts_enabled is not None:
        s.hsts_enabled = body.hsts_enabled
    if body.hsts_max_age is not None:
        s.hsts_max_age = body.hsts_max_age
    if body.hsts_include_subdomains is not None:
        s.hsts_include_subdomains = body.hsts_include_subdomains
    if body.hsts_preload is not None:
        s.hsts_preload = body.hsts_preload
    if body.automatic_https_rewrites is not None:
        s.automatic_https_rewrites = body.automatic_https_rewrites
    if body.tls_0rtt_enabled is not None:
        s.tls_0rtt_enabled = body.tls_0rtt_enabled
    if body.mtls_enabled is not None:
        s.mtls_enabled = body.mtls_enabled
    if body.mtls_ca_cert_pem is not None:
        s.mtls_ca_cert_pem = body.mtls_ca_cert_pem
    db.commit()
    db.refresh(s)
    return s.to_dict()


@router.delete("/settings/{domain}", status_code=204)
def delete_settings(domain: str, db: Session = Depends(get_db)):
    s = db.query(SSLSettings).filter(SSLSettings.domain == domain).first()
    if not s:
        raise HTTPException(status_code=404, detail="SSL settings not found for domain")
    db.delete(s)
    db.commit()
    return None
