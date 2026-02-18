"""
Managed rules sync: fetch rule feed from URL, parse (JSON or OWASP CRS), upsert into security_rules.
No hardcoded rule content or URLs; all from config/DB.
"""
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from sqlalchemy.orm import Session

from backend.config import config
from backend.models.rule_packs import RulePack
from backend.models.security_rules import SecurityRule, RuleAction


# JSON feed schema: { "rules": [ { "id": "...", "name": "...", "pattern": "...", "action": "block", "applies_to": "all" } ] }
def fetch_feed(
    url: str,
    feed_format: str,
    auth_header: Optional[str] = None,
    timeout: float = 30.0,
) -> bytes:
    """HTTP GET feed URL; return raw bytes."""
    if not url or not url.strip():
        raise ValueError("MANAGED_RULES_FEED_URL is empty")
    headers = {}
    if auth_header and auth_header.strip():
        headers["Authorization"] = auth_header.strip()
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=headers or None)
        resp.raise_for_status()
        return resp.content


def parse_json_rules(content: bytes) -> List[Dict[str, Any]]:
    """
    Parse project JSON format.
    Expected: { "rules": [ { "id": "...", "name": "...", "pattern": "...", "action": "block", "applies_to": "all" } ] }
    """
    data = json.loads(content.decode("utf-8", errors="replace"))
    rules = data.get("rules") if isinstance(data, dict) else []
    if not isinstance(rules, list):
        return []
    out = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        name = r.get("name") or r.get("id") or "Unnamed"
        pattern = r.get("pattern")
        if pattern is None:
            pattern = r.get("regex", "")
        action = (r.get("action") or "block").lower()
        applies_to = (r.get("applies_to") or "all").lower()
        external_id = r.get("id")
        if isinstance(external_id, (int, float)):
            external_id = str(external_id)
        out.append({
            "name": str(name),
            "pattern": str(pattern) if pattern else "",
            "action": action,
            "applies_to": applies_to,
            "external_id": external_id,
        })
    return out


def parse_owasp_crs(content: bytes) -> List[Dict[str, Any]]:
    """
    Parse ModSecurity-style OWASP CRS .conf content.
    SecRule ARGS|REQUEST_URI|REQUEST_HEADERS|REQUEST_BODY "@rx pattern" "id:123,phase:2,block"
    We extract id, and use a simple rx pattern when present; otherwise skip.
    """
    text = content.decode("utf-8", errors="replace")
    out = []
    # SecRule ... "@rx" "id:123,..." or SecRule ... "@rx pattern" "..."
    # Simplified: look for SecRule lines with @rx and id:
    secrule = re.compile(
        r"SecRule\s+(\S+)\s+[\"']@rx\s+([^\"']+)[\"']\s+[\"']([^\"']+)[\"']",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = secrule.search(line)
        if not m:
            continue
        variable, pattern, opts = m.group(1), m.group(2), m.group(3)
        id_match = re.search(r"\bid:(\d+)", opts, re.IGNORECASE)
        rule_id = id_match.group(1) if id_match else None
        action = "block" if "block" in opts.lower() else "log"
        # Map variable to applies_to
        var_lower = variable.upper()
        if "ARGS" in var_lower or "REQUEST_URI" in var_lower:
            applies_to = "all"
        elif "REQUEST_HEADERS" in var_lower:
            applies_to = "headers"
        elif "REQUEST_BODY" in var_lower:
            applies_to = "body"
        else:
            applies_to = "all"
        out.append({
            "name": f"CRS-{rule_id}" if rule_id else "CRS-rule",
            "pattern": pattern,
            "action": action,
            "applies_to": applies_to,
            "external_id": rule_id,
        })
    return out


def parse_feed(content: bytes, feed_format: str) -> List[Dict[str, Any]]:
    """Dispatch to JSON or OWASP CRS parser."""
    fmt = (feed_format or "json").lower().strip()
    if fmt == "owasp_crs":
        return parse_owasp_crs(content)
    return parse_json_rules(content)


def _action_to_enum(action: str) -> RuleAction:
    a = (action or "block").lower()
    if a == "log":
        return RuleAction.LOG
    if a == "alert":
        return RuleAction.ALERT
    if a == "redirect":
        return RuleAction.REDIRECT
    if a == "challenge":
        return RuleAction.CHALLENGE
    return RuleAction.BLOCK


def sync_pack(
    db: Session,
    pack_id: str,
    name: str,
    source_url: str,
    feed_format: str,
    auth_header: Optional[str] = None,
    version_from_feed: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch feed from source_url, parse, upsert rules into SecurityRule with rule_pack_id;
    create or update RulePack; set version and last_synced_at.
    Returns summary: rules_created, rules_updated, version, error.
    """
    pack = db.query(RulePack).filter(RulePack.pack_id == pack_id).first()
    if not pack:
        pack = RulePack(
            name=name or pack_id,
            pack_id=pack_id,
            source_url=source_url,
            version=version_from_feed or "0",
            enabled=True,
        )
        db.add(pack)
        db.flush()

    try:
        raw = fetch_feed(source_url, feed_format, auth_header)
    except Exception as e:
        logger.warning(f"Managed rules fetch failed for {pack_id}: {e}")
        return {"rules_created": 0, "rules_updated": 0, "version": pack.version, "error": str(e)}

    rules_data = parse_feed(raw, feed_format)
    version = version_from_feed or (datetime.utcnow().strftime("%Y%m%d%H%M") if not version_from_feed else pack.version)

    rules_created = 0
    rules_updated = 0
    existing_by_external = {
        r.external_id: r
        for r in db.query(SecurityRule)
        .filter(SecurityRule.rule_pack_id == pack.id)
        .all()
        if r.external_id
    }
    existing_by_pattern_name = {
        (r.pattern or "", r.name): r
        for r in db.query(SecurityRule).filter(SecurityRule.rule_pack_id == pack.id).all()
    }

    for r in rules_data:
        if not r.get("pattern"):
            continue
        external_id = r.get("external_id")
        rule = existing_by_external.get(external_id) if external_id else None
        if not rule:
            rule = existing_by_pattern_name.get((r["pattern"], r["name"]))
        if rule:
            rule.pattern = r["pattern"]
            rule.applies_to = r.get("applies_to") or "all"
            rule.action = _action_to_enum(r.get("action"))
            rule.rule_pack_version = version
            rule.is_active = True
            rules_updated += 1
        else:
            new_rule = SecurityRule(
                name=r["name"],
                rule_type="pattern",
                pattern=r["pattern"],
                applies_to=r.get("applies_to") or "all",
                action=_action_to_enum(r.get("action")),
                rule_pack_id=pack.id,
                rule_pack_version=version,
                external_id=str(external_id) if external_id else None,
                is_active=True,
                is_system_rule=True,
            )
            db.add(new_rule)
            rules_created += 1
            if new_rule.external_id:
                existing_by_external[new_rule.external_id] = new_rule
            existing_by_pattern_name[(new_rule.pattern, new_rule.name)] = new_rule

    pack.version = version
    pack.last_synced_at = datetime.utcnow()
    pack.source_url = source_url
    pack.updated_at = datetime.utcnow()
    db.commit()

    logger.info(f"Managed rules sync {pack_id}: created={rules_created}, updated={rules_updated}, version={version}")
    return {"rules_created": rules_created, "rules_updated": rules_updated, "version": version, "error": None}
