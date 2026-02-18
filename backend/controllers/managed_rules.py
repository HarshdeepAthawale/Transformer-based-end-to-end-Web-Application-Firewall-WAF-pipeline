"""Managed rules (rule packs) controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.models.rule_packs import RulePack
from backend.models.security_rules import SecurityRule
from backend.services import managed_rules_sync
from backend.config import config


def get_managed_packs(db: Session, enabled_only: bool = True) -> dict:
    """List rule packs with optional filter by enabled. Returns packs with rule count."""
    query = db.query(RulePack)
    if enabled_only:
        query = query.filter(RulePack.enabled == True)
    packs = query.order_by(RulePack.name).all()
    result = []
    for p in packs:
        count = db.query(SecurityRule).filter(
            SecurityRule.rule_pack_id == p.id,
            SecurityRule.is_active == True,
        ).count()
        d = p.to_dict()
        d["rule_count"] = count
        result.append(d)
    return {
        "success": True,
        "data": result,
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_managed_rules_for_gateway(db: Session, enabled_only: bool = True) -> dict:
    """
    Response for GET /api/rules/managed?enabled_only=true.
    Shape: { "packs": [ { "pack_id", "version", "rules": [ { "id", "name", "pattern", "applies_to", "action" } ] } ] }
    """
    query = db.query(RulePack).filter(RulePack.enabled == True)
    if enabled_only:
        query = query.filter(RulePack.enabled == True)
    packs = query.all()
    packs_payload = []
    for p in packs:
        rules = (
            db.query(SecurityRule)
            .filter(SecurityRule.rule_pack_id == p.id, SecurityRule.is_active == True)
            .all()
        )
        packs_payload.append({
            "pack_id": p.pack_id,
            "name": p.name,
            "version": p.version or "",
            "enabled": p.enabled,
            "last_synced_at": p.last_synced_at.isoformat() if p.last_synced_at else None,
            "rules": [
                {
                    "id": r.id,
                    "name": r.name,
                    "pattern": r.pattern or "",
                    "applies_to": r.applies_to or "all",
                    "action": r.action.value if r.action else "block",
                }
                for r in rules
            ],
        })
    return {"packs": packs_payload}


def toggle_pack(db: Session, pack_id: str, enabled: bool) -> dict:
    """Enable or disable a rule pack by pack_id."""
    pack = db.query(RulePack).filter(RulePack.pack_id == pack_id).first()
    if not pack:
        return {"success": False, "message": "Pack not found", "timestamp": datetime.utcnow().isoformat()}
    pack.enabled = bool(enabled)
    pack.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(pack)
    return {
        "success": True,
        "data": pack.to_dict(),
        "timestamp": datetime.utcnow().isoformat(),
    }


def sync_managed_rules(db: Session, pack_id: str | None = None) -> dict:
    """
    Run sync for configured feed (or specific pack_id if provided).
    If pack_id is None, uses config MANAGED_RULES_* to sync default pack.
    """
    url = config.MANAGED_RULES_FEED_URL or ""
    if not url.strip() and not pack_id:
        return {
            "success": False,
            "message": "MANAGED_RULES_FEED_URL is not set",
            "timestamp": datetime.utcnow().isoformat(),
        }
    if pack_id:
        pack = db.query(RulePack).filter(RulePack.pack_id == pack_id).first()
        if not pack or not pack.source_url:
            return {
                "success": False,
                "message": f"Pack {pack_id} not found or has no source_url",
                "timestamp": datetime.utcnow().isoformat(),
            }
        name, source_url, feed_format = pack.name, pack.source_url, "json"
    else:
        pack_id = config.MANAGED_RULES_PACK_ID
        name = f"Managed rules ({pack_id})"
        source_url = url
        feed_format = config.MANAGED_RULES_FEED_FORMAT or "json"

    result = managed_rules_sync.sync_pack(
        db,
        pack_id=pack_id,
        name=name,
        source_url=source_url,
        feed_format=feed_format,
        auth_header=config.MANAGED_RULES_FEED_HEADER,
    )
    if result.get("error"):
        return {
            "success": False,
            "message": result["error"],
            "detail": result,
            "timestamp": datetime.utcnow().isoformat(),
        }
    return {
        "success": True,
        "data": result,
        "timestamp": datetime.utcnow().isoformat(),
    }
