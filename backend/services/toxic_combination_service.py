"""
Toxic Combination Detection Service.

Implements Cloudflare's multi-signal correlation patterns that detect when
small signals converge into security incidents. Each pattern combines
bot signals, anomalies, vulnerabilities, and misconfigurations.
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session
from loguru import logger

from backend.models.security_event import SecurityEvent
from backend.models.toxic_combination import ToxicCombination
from backend.lib.datetime_utils import utc_now


# Sensitive admin/management paths (Pattern 1)
ADMIN_PATHS = [
    "/wp-admin/", "/admin/", "/administrator/", "/phpmyadmin/",
    "/manager/html/", "/actuator/", "/_search/", "/app/kibana/",
    "/server-status", "/server-info", "/.env", "/debug/",
]

# Debug parameters (Pattern 3)
DEBUG_PARAMS = ["debug=true", "debug=1", "test=true", "test=1", "trace=1", "dev=1"]

# Payment paths (Pattern 6)
PAYMENT_PATHS = ["/payment", "/checkout", "/cart", "/billing", "/subscribe", "/purchase"]

# IDOR parameter patterns (Pattern 2)
IDOR_PARAM_PATTERN = re.compile(r"(?:uid|user_id|id|user|account_id)=\d{3,10}", re.IGNORECASE)


class ToxicCombinationService:
    """Detects toxic combinations from security event data."""

    def evaluate_window(
        self,
        db: Session,
        org_id: int,
        window_minutes: int = 5,
    ) -> list[dict]:
        """
        Evaluate all toxic combination patterns over a recent time window.

        Args:
            db: Database session
            org_id: Organization ID
            window_minutes: How far back to look (default 5 min)

        Returns:
            List of detected toxic combinations
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=window_minutes)

        events = (
            db.query(SecurityEvent)
            .filter(
                SecurityEvent.timestamp >= window_start,
                SecurityEvent.org_id == org_id,
            )
            .all()
        )

        if not events:
            return []

        detections = []

        # Run each pattern detector
        detections.extend(self._detect_admin_probing(events, org_id))
        detections.extend(self._detect_debug_probing(events, org_id))
        detections.extend(self._detect_sqli_success(events, org_id))
        detections.extend(self._detect_coordinated_evasion(events, org_id))

        # Persist detections
        for detection in detections:
            self._persist_detection(db, detection)

        if detections:
            db.commit()

        return detections

    def _detect_admin_probing(self, events: list, org_id: int) -> list[dict]:
        """
        Pattern 1: Admin Endpoint Probing.
        Ingredients: bot_score < 30 + path matches admin patterns.
        """
        detections = []
        admin_events = defaultdict(list)

        for e in events:
            if (e.bot_score is not None and e.bot_score < 30 and e.path):
                for admin_path in ADMIN_PATHS:
                    if admin_path.lower() in (e.path or "").lower():
                        admin_events[e.ip].append(e)
                        break

        for ip, ip_events in admin_events.items():
            if len(ip_events) >= 3:  # At least 3 probing attempts
                paths = list({e.path for e in ip_events if e.path})
                detections.append({
                    "org_id": org_id,
                    "pattern_name": "admin_probing",
                    "severity": "high",
                    "description": f"Bot (score <30) probing {len(paths)} admin endpoints from {ip}",
                    "affected_path": ", ".join(paths[:5]),
                    "source_ips": [ip],
                    "signals": [
                        {"type": "bot", "detail": f"bot_score < 30 from {ip}"},
                        {"type": "anomaly", "detail": f"Repeated probing: {len(ip_events)} requests"},
                        {"type": "vulnerability", "detail": f"Admin paths accessed: {', '.join(paths[:3])}"},
                    ],
                    "event_count": len(ip_events),
                })

        return detections

    def _detect_debug_probing(self, events: list, org_id: int) -> list[dict]:
        """
        Pattern 3: Debug Parameter Probing.
        Ingredients: bot_score < 30 + debug/test params in path.
        """
        detections = []
        debug_events = defaultdict(list)

        for e in events:
            if e.bot_score is not None and e.bot_score < 30 and e.path:
                path_lower = (e.path or "").lower()
                for param in DEBUG_PARAMS:
                    if param in path_lower:
                        debug_events[e.ip].append(e)
                        break

        for ip, ip_events in debug_events.items():
            if len(ip_events) >= 2:
                paths = list({e.path for e in ip_events if e.path})
                detections.append({
                    "org_id": org_id,
                    "pattern_name": "debug_probing",
                    "severity": "medium",
                    "description": f"Bot probing debug endpoints from {ip} ({len(ip_events)} requests)",
                    "affected_path": ", ".join(paths[:5]),
                    "source_ips": [ip],
                    "signals": [
                        {"type": "bot", "detail": f"bot_score < 30 from {ip}"},
                        {"type": "misconfiguration", "detail": "Debug parameters active in production"},
                        {"type": "anomaly", "detail": f"Probing {len(paths)} unique endpoints"},
                    ],
                    "event_count": len(ip_events),
                })

        return detections

    def _detect_sqli_success(self, events: list, org_id: int) -> list[dict]:
        """
        Pattern 4: SQL Injection with Success Response.
        Ingredients: bot_score < 30 + waf_sqli_score < 30 + repeated mutations.
        """
        detections = []
        sqli_events = defaultdict(list)

        for e in events:
            if (
                e.bot_score is not None and e.bot_score < 30
                and e.waf_sqli_score is not None and e.waf_sqli_score < 30
            ):
                sqli_events[e.ip].append(e)

        for ip, ip_events in sqli_events.items():
            if len(ip_events) >= 2:
                paths = list({e.path for e in ip_events if e.path})
                detections.append({
                    "org_id": org_id,
                    "pattern_name": "sqli_success",
                    "severity": "critical",
                    "description": f"SQL injection attempts from {ip} with low WAF scores ({len(ip_events)} attempts)",
                    "affected_path": ", ".join(paths[:5]),
                    "source_ips": [ip],
                    "signals": [
                        {"type": "bot", "detail": f"bot_score < 30 from {ip}"},
                        {"type": "vulnerability", "detail": f"waf_sqli_score < 30 on {len(ip_events)} requests"},
                        {"type": "anomaly", "detail": f"Repeated mutations: {len(ip_events)} attempts"},
                    ],
                    "event_count": len(ip_events),
                })

        return detections

    def _detect_coordinated_evasion(self, events: list, org_id: int) -> list[dict]:
        """
        Pattern 5: Coordinated Rate Limit Evasion.
        Ingredients: 5+ IPs + same path + same time window.
        """
        detections = []
        path_ips = defaultdict(set)

        for e in events:
            if e.path and e.event_type == "rate_limit":
                path_ips[e.path].add(e.ip)

        for path, ips in path_ips.items():
            if len(ips) >= 5:
                detections.append({
                    "org_id": org_id,
                    "pattern_name": "coordinated_evasion",
                    "severity": "high",
                    "description": f"Coordinated rate limit evasion: {len(ips)} IPs targeting {path}",
                    "affected_path": path,
                    "source_ips": list(ips)[:20],
                    "signals": [
                        {"type": "anomaly", "detail": f"{len(ips)} distributed IPs hitting same endpoint"},
                        {"type": "vulnerability", "detail": f"Each IP just under rate limit threshold"},
                        {"type": "bot", "detail": "Coordinated automation pattern"},
                    ],
                    "event_count": len(ips),
                })

        return detections

    def _persist_detection(self, db: Session, detection: dict) -> None:
        """Create or update a ToxicCombination record."""
        # Check for existing active detection with same pattern + path
        existing = (
            db.query(ToxicCombination)
            .filter(
                ToxicCombination.org_id == detection["org_id"],
                ToxicCombination.pattern_name == detection["pattern_name"],
                ToxicCombination.affected_path == detection.get("affected_path"),
                ToxicCombination.status == "active",
            )
            .first()
        )

        if existing:
            # Update existing detection
            existing.last_seen = utc_now()
            existing.event_count = (existing.event_count or 0) + detection.get("event_count", 1)
            # Merge source IPs
            try:
                current_ips = json.loads(existing.source_ips or "[]")
            except (json.JSONDecodeError, TypeError):
                current_ips = []
            new_ips = list(set(current_ips + detection.get("source_ips", [])))[:50]
            existing.source_ips = json.dumps(new_ips)
        else:
            # Create new detection
            tc = ToxicCombination(
                org_id=detection["org_id"],
                pattern_name=detection["pattern_name"],
                severity=detection["severity"],
                description=detection.get("description"),
                affected_path=detection.get("affected_path"),
                source_ips=json.dumps(detection.get("source_ips", [])),
                signals=json.dumps(detection.get("signals", [])),
                event_count=detection.get("event_count", 1),
                first_seen=utc_now(),
                last_seen=utc_now(),
            )
            db.add(tc)

    def get_active(self, db: Session, org_id: int) -> list[dict]:
        """Get all active toxic combinations for an org."""
        results = (
            db.query(ToxicCombination)
            .filter(
                ToxicCombination.org_id == org_id,
                ToxicCombination.status.in_(["active", "investigating"]),
            )
            .order_by(ToxicCombination.last_seen.desc())
            .all()
        )
        return [r.to_dict() for r in results]

    def get_stats(self, db: Session, org_id: int) -> dict:
        """Get summary statistics for toxic combinations."""
        active_count = (
            db.query(func.count(ToxicCombination.id))
            .filter(ToxicCombination.org_id == org_id, ToxicCombination.status == "active")
            .scalar()
        ) or 0

        by_severity = {}
        for severity in ["critical", "high", "medium", "low"]:
            count = (
                db.query(func.count(ToxicCombination.id))
                .filter(
                    ToxicCombination.org_id == org_id,
                    ToxicCombination.status == "active",
                    ToxicCombination.severity == severity,
                )
                .scalar()
            ) or 0
            by_severity[severity] = count

        by_pattern = {}
        rows = (
            db.query(ToxicCombination.pattern_name, func.count(ToxicCombination.id))
            .filter(ToxicCombination.org_id == org_id, ToxicCombination.status == "active")
            .group_by(ToxicCombination.pattern_name)
            .all()
        )
        for name, count in rows:
            by_pattern[name] = count

        return {
            "active_count": active_count,
            "by_severity": by_severity,
            "by_pattern": by_pattern,
        }

    def update_status(
        self, db: Session, org_id: int, combination_id: int, status: str
    ) -> Optional[dict]:
        """Update the status of a toxic combination."""
        tc = (
            db.query(ToxicCombination)
            .filter(ToxicCombination.id == combination_id, ToxicCombination.org_id == org_id)
            .first()
        )
        if not tc:
            return None

        tc.status = status
        if status == "resolved":
            tc.resolved_at = utc_now()

        db.commit()
        db.refresh(tc)
        return tc.to_dict()
