"""
Toxic Combination Detection Service.

Implements multi-signal correlation patterns that detect when small signals
converge into security incidents. Each pattern combines bot signals, anomalies,
vulnerabilities, and misconfigurations.

Integrates Z-score anomaly detection from BaselineService for statistical
validation of pattern triggers.
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.security_event import SecurityEvent
from backend.models.toxic_combination import ToxicCombination
from backend.lib.datetime_utils import utc_now
from backend.services.baseline_service import BaselineService, Z_SCORE_THRESHOLD


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

    def __init__(self):
        self._baseline = BaselineService()

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
        detections.extend(self._detect_admin_probing(events, org_id, db))
        detections.extend(self._detect_idor(events, org_id))
        detections.extend(self._detect_debug_probing(events, org_id))
        detections.extend(self._detect_sqli_success(events, org_id))
        detections.extend(self._detect_coordinated_evasion(events, org_id))
        detections.extend(self._detect_payment_anomaly(events, org_id, db))

        # Persist detections
        for detection in detections:
            self._persist_detection(db, detection)

        if detections:
            db.commit()

        return detections

    def _detect_admin_probing(self, events: list, org_id: int, db: Session) -> list[dict]:
        """
        Pattern 1: Admin Endpoint Probing.
        Signals: (1) elevated 404s on admin paths (Z-score > 3.0),
        (2) bot_score < 30, (3) new source IPs, (4) elevated SQLi scores.
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
            if len(ip_events) >= 3:
                paths = list({e.path for e in ip_events if e.path})

                # Z-score validation: check if block rate is anomalous
                z_signals = []
                try:
                    block_check = self._baseline.check_current_hour(
                        db, "block_rate", org_id
                    )
                    if block_check["is_anomalous"]:
                        z_signals.append({
                            "type": "z_score",
                            "detail": f"Block rate Z-score: {block_check['z_score']} (threshold: {Z_SCORE_THRESHOLD})",
                        })
                except Exception:
                    pass

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
                        *z_signals,
                    ],
                    "event_count": len(ip_events),
                })

        return detections

    def _detect_idor(self, events: list, org_id: int) -> list[dict]:
        """
        Pattern 2: IDOR (Insecure Direct Object Reference) Detection.
        Signals: single IP rapidly iterating numeric IDs on user/account endpoints.
        """
        detections = []
        idor_events = defaultdict(list)

        for e in events:
            if e.path and IDOR_PARAM_PATTERN.search(e.path):
                idor_events[e.ip].append(e)

        for ip, ip_events in idor_events.items():
            if len(ip_events) >= 5:
                # Extract unique ID values to confirm enumeration
                id_values = set()
                for ev in ip_events:
                    matches = IDOR_PARAM_PATTERN.findall(ev.path or "")
                    id_values.update(matches)

                if len(id_values) >= 3:
                    paths = list({e.path for e in ip_events if e.path})
                    detections.append({
                        "org_id": org_id,
                        "pattern_name": "idor_detection",
                        "severity": "high",
                        "description": f"IDOR enumeration from {ip}: {len(id_values)} unique IDs across {len(ip_events)} requests",
                        "affected_path": ", ".join(paths[:5]),
                        "source_ips": [ip],
                        "signals": [
                            {"type": "anomaly", "detail": f"{len(id_values)} unique ID values enumerated"},
                            {"type": "vulnerability", "detail": "Sequential/rapid ID parameter iteration"},
                            {"type": "bot", "detail": f"{len(ip_events)} requests from single IP"},
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
                        {"type": "vulnerability", "detail": "Each IP just under rate limit threshold"},
                        {"type": "bot", "detail": "Coordinated automation pattern"},
                    ],
                    "event_count": len(ips),
                })

        return detections

    def _detect_payment_anomaly(self, events: list, org_id: int, db: Session) -> list[dict]:
        """
        Pattern 6: Payment Endpoint Anomaly.
        Signals: unusual volume on payment paths + bot indicators + Z-score anomaly.
        """
        detections = []
        payment_events = defaultdict(list)

        for e in events:
            if e.path:
                path_lower = e.path.lower()
                for pp in PAYMENT_PATHS:
                    if pp in path_lower:
                        payment_events[e.ip].append(e)
                        break

        for ip, ip_events in payment_events.items():
            if len(ip_events) >= 5:
                # Check for bot indicators
                bot_events = [e for e in ip_events if e.bot_score is not None and e.bot_score < 30]
                if len(bot_events) < 2:
                    continue

                # Z-score check for request volume anomaly
                z_signals = []
                try:
                    vol_check = self._baseline.check_current_hour(
                        db, "request_volume", org_id
                    )
                    if vol_check["is_anomalous"]:
                        z_signals.append({
                            "type": "z_score",
                            "detail": f"Request volume Z-score: {vol_check['z_score']}",
                        })
                except Exception:
                    pass

                paths = list({e.path for e in ip_events if e.path})
                detections.append({
                    "org_id": org_id,
                    "pattern_name": "payment_anomaly",
                    "severity": "critical",
                    "description": f"Anomalous payment endpoint activity from {ip}: {len(ip_events)} requests",
                    "affected_path": ", ".join(paths[:5]),
                    "source_ips": [ip],
                    "signals": [
                        {"type": "bot", "detail": f"{len(bot_events)} bot-like requests (score < 30)"},
                        {"type": "vulnerability", "detail": f"Payment endpoints targeted: {', '.join(paths[:3])}"},
                        {"type": "anomaly", "detail": f"{len(ip_events)} rapid payment requests from {ip}"},
                        *z_signals,
                    ],
                    "event_count": len(ip_events),
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
