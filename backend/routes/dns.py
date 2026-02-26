"""
Phase 1 — DNS-as-a-Service API.
CRUD for zones and records. Proxy (orange-cloud) toggle per record.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import DNSZone, DNSRecord

router = APIRouter()


# ─── Schemas ─────────────────────────────────────────────────────────────

class DNSZoneCreate(BaseModel):
    domain: str
    provider: str = "manual"
    dnssec_enabled: bool = False


class DNSZoneUpdate(BaseModel):
    status: Optional[str] = None
    provider: Optional[str] = None
    provider_zone_id: Optional[str] = None
    nameservers: Optional[str] = None  # JSON array string
    dnssec_enabled: Optional[bool] = None


class DNSRecordCreate(BaseModel):
    name: str  # subdomain or @ for apex
    record_type: str  # A, AAAA, CNAME, MX, TXT, SRV, CAA, NS
    content: str
    ttl: int = 300
    priority: Optional[int] = None
    proxied: bool = False
    comment: Optional[str] = None


class DNSRecordUpdate(BaseModel):
    content: Optional[str] = None
    original_content: Optional[str] = None
    ttl: Optional[int] = None
    priority: Optional[int] = None
    proxied: Optional[bool] = None
    comment: Optional[str] = None


# ─── Zones ───────────────────────────────────────────────────────────────

@router.get("/zones")
def list_zones(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    zones = db.query(DNSZone).order_by(DNSZone.domain).offset(skip).limit(limit).all()
    return [z.to_dict() for z in zones]


@router.post("/zones", status_code=201)
def create_zone(body: DNSZoneCreate, db: Session = Depends(get_db)):
    if db.query(DNSZone).filter(DNSZone.domain == body.domain).first():
        raise HTTPException(status_code=400, detail="Zone with this domain already exists")
    zone = DNSZone(
        domain=body.domain,
        provider=body.provider,
        dnssec_enabled=body.dnssec_enabled,
    )
    db.add(zone)
    db.commit()
    db.refresh(zone)
    return zone.to_dict()


@router.get("/zones/{zone_id}")
def get_zone(zone_id: int, db: Session = Depends(get_db)):
    zone = db.query(DNSZone).filter(DNSZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone.to_dict()


@router.patch("/zones/{zone_id}")
def update_zone(zone_id: int, body: DNSZoneUpdate, db: Session = Depends(get_db)):
    zone = db.query(DNSZone).filter(DNSZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    if body.status is not None:
        zone.status = body.status
    if body.provider is not None:
        zone.provider = body.provider
    if body.provider_zone_id is not None:
        zone.provider_zone_id = body.provider_zone_id
    if body.nameservers is not None:
        zone.nameservers = body.nameservers
    if body.dnssec_enabled is not None:
        zone.dnssec_enabled = body.dnssec_enabled
    db.commit()
    db.refresh(zone)
    return zone.to_dict()


@router.delete("/zones/{zone_id}", status_code=204)
def delete_zone(zone_id: int, db: Session = Depends(get_db)):
    zone = db.query(DNSZone).filter(DNSZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    db.query(DNSRecord).filter(DNSRecord.zone_id == zone_id).delete()
    db.delete(zone)
    db.commit()
    return None


# ─── Records (under a zone) ──────────────────────────────────────────────

@router.get("/zones/{zone_id}/records")
def list_records(
    zone_id: int,
    db: Session = Depends(get_db),
    record_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    zone = db.query(DNSZone).filter(DNSZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    q = db.query(DNSRecord).filter(DNSRecord.zone_id == zone_id)
    if record_type:
        q = q.filter(DNSRecord.record_type == record_type.upper())
    records = q.order_by(DNSRecord.name).offset(skip).limit(limit).all()
    return [r.to_dict() for r in records]


@router.post("/zones/{zone_id}/records", status_code=201)
def create_record(zone_id: int, body: DNSRecordCreate, db: Session = Depends(get_db)):
    zone = db.query(DNSZone).filter(DNSZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    rec = DNSRecord(
        zone_id=zone_id,
        name=body.name,
        record_type=body.record_type.upper(),
        content=body.content,
        ttl=body.ttl,
        priority=body.priority,
        proxied=body.proxied,
        comment=body.comment,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec.to_dict()


@router.get("/zones/{zone_id}/records/{record_id}")
def get_record(zone_id: int, record_id: int, db: Session = Depends(get_db)):
    rec = db.query(DNSRecord).filter(
        DNSRecord.zone_id == zone_id,
        DNSRecord.id == record_id,
    ).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found")
    return rec.to_dict()


@router.patch("/zones/{zone_id}/records/{record_id}")
def update_record(
    zone_id: int, record_id: int, body: DNSRecordUpdate, db: Session = Depends(get_db)
):
    rec = db.query(DNSRecord).filter(
        DNSRecord.zone_id == zone_id,
        DNSRecord.id == record_id,
    ).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found")
    if body.content is not None:
        rec.content = body.content
    if body.original_content is not None:
        rec.original_content = body.original_content
    if body.ttl is not None:
        rec.ttl = body.ttl
    if body.priority is not None:
        rec.priority = body.priority
    if body.proxied is not None:
        rec.proxied = body.proxied
    if body.comment is not None:
        rec.comment = body.comment
    db.commit()
    db.refresh(rec)
    return rec.to_dict()


@router.delete("/zones/{zone_id}/records/{record_id}", status_code=204)
def delete_record(zone_id: int, record_id: int, db: Session = Depends(get_db)):
    rec = db.query(DNSRecord).filter(
        DNSRecord.zone_id == zone_id,
        DNSRecord.id == record_id,
    ).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found")
    db.delete(rec)
    db.commit()
    return None
