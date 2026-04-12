"""
Emergency Rule Service: fast-deploy rules for zero-day threat response.

Emergency rules are checked BEFORE ML inference in the WAF middleware.
They use fast string matching and regex to block known exploit patterns
without waiting for the transformer model.

Inspired by our Emergency Rules that blocked Ivanti zero-day
exploits within 24 hours of PoC publication.
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from loguru import logger

from backend.models.emergency_rule import EmergencyRule


class EmergencyRuleService:
    """Manages and evaluates emergency security rules."""

    def __init__(self):
        self._rules_cache: list[dict] = []
        self._cache_loaded = False

    def load_rules(self, db: Session) -> None:
        """Load enabled emergency rules into memory cache."""
        now = datetime.now(timezone.utc)
        rules = (
            db.query(EmergencyRule)
            .filter(EmergencyRule.enabled.is_(True))
            .all()
        )
        self._rules_cache = []
        for rule in rules:
            # Skip expired rules
            if rule.expires_at and rule.expires_at < now:
                continue
            try:
                patterns = json.loads(rule.patterns) if rule.patterns else []
            except (json.JSONDecodeError, TypeError):
                patterns = []

            self._rules_cache.append({
                "id": rule.id,
                "name": rule.name,
                "patterns": patterns,
                "action": rule.action,
                "severity": rule.severity,
            })
        self._cache_loaded = True
        logger.info(f"Loaded {len(self._rules_cache)} emergency rules into cache")

    def check_request(
        self,
        method: str,
        path: str,
        query: str = "",
        body: str = "",
        headers: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Check a request against all active emergency rules.

        Returns matching rule info if blocked, None if allowed.
        This runs BEFORE ML inference for fast blocking.
        """
        if not self._rules_cache:
            return None

        request_data = {
            "method": method.upper(),
            "path": path or "",
            "query": query or "",
            "body": (body or "")[:4096],  # Limit body scanning
            "headers": headers or {},
        }

        for rule in self._rules_cache:
            if self._matches_rule(request_data, rule["patterns"]):
                return {
                    "blocked": True,
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "action": rule["action"],
                    "severity": rule["severity"],
                }

        return None

    def _matches_rule(self, request_data: dict, patterns: list[dict]) -> bool:
        """
        Check if a request matches ALL patterns in a rule.
        All patterns must match for the rule to trigger (AND logic).
        """
        if not patterns:
            return False

        for pattern in patterns:
            field = pattern.get("field", "")
            op = pattern.get("op", "")
            value = pattern.get("value", "")

            # Get the field value from request
            if field == "path":
                target = request_data["path"]
            elif field == "query":
                target = request_data["query"]
            elif field == "body":
                target = request_data["body"]
            elif field == "method":
                target = request_data["method"]
            elif field.startswith("header:"):
                header_name = field.split(":", 1)[1]
                target = request_data["headers"].get(header_name, "")
            else:
                continue

            # Apply operator
            if op == "contains":
                if value.lower() not in target.lower():
                    return False
            elif op == "regex":
                try:
                    if not re.search(value, target, re.IGNORECASE):
                        return False
                except re.error:
                    return False
            elif op == "equals":
                if target.lower() != value.lower():
                    return False
            elif op == "starts_with":
                if not target.lower().startswith(value.lower()):
                    return False
            else:
                return False

        return True

    def increment_hit_count(self, db: Session, rule_id: int) -> None:
        """Increment the hit counter for a matched rule."""
        rule = db.query(EmergencyRule).filter(EmergencyRule.id == rule_id).first()
        if rule:
            rule.hit_count = (rule.hit_count or 0) + 1
            db.commit()

    # CRUD operations

    def create_rule(self, db: Session, data: dict) -> dict:
        """Create a new emergency rule."""
        rule = EmergencyRule(
            name=data["name"],
            description=data.get("description"),
            cves=json.dumps(data.get("cves", [])),
            patterns=json.dumps(data.get("patterns", [])),
            action=data.get("action", "block"),
            severity=data.get("severity", "critical"),
            source=data.get("source", "manual"),
            enabled=data.get("enabled", True),
        )
        db.add(rule)
        db.commit()
        db.refresh(rule)

        # Reload cache
        self.load_rules(db)
        return rule.to_dict()

    def list_rules(self, db: Session) -> list[dict]:
        """List all emergency rules."""
        rules = db.query(EmergencyRule).order_by(EmergencyRule.created_at.desc()).all()
        return [r.to_dict() for r in rules]

    def toggle_rule(self, db: Session, rule_id: int, enabled: bool) -> Optional[dict]:
        """Enable or disable an emergency rule."""
        rule = db.query(EmergencyRule).filter(EmergencyRule.id == rule_id).first()
        if not rule:
            return None
        rule.enabled = enabled
        db.commit()
        db.refresh(rule)

        # Reload cache
        self.load_rules(db)
        return rule.to_dict()

    def delete_rule(self, db: Session, rule_id: int) -> bool:
        """Delete an emergency rule."""
        rule = db.query(EmergencyRule).filter(EmergencyRule.id == rule_id).first()
        if not rule:
            return False
        db.delete(rule)
        db.commit()

        # Reload cache
        self.load_rules(db)
        return True


# Pre-built zero-day patterns that can be rapidly activated
ZERO_DAY_PATTERNS = {
    "ivanti_auth_bypass": {
        "name": "Ivanti - Auth Bypass, Command Injection - CVE:CVE-2023-46805, CVE:CVE-2024-21887",
        "cves": ["CVE-2023-46805", "CVE-2024-21887"],
        "patterns": [
            {"field": "path", "op": "contains", "value": "/api/v1/totp/user-backup-code/../../"},
        ],
        "action": "block",
        "severity": "critical",
    },
    "log4shell": {
        "name": "Log4Shell - JNDI Injection - CVE:CVE-2021-44228",
        "cves": ["CVE-2021-44228"],
        "patterns": [
            {"field": "body", "op": "regex", "value": r"\$\{jndi:(ldap|rmi|dns|iiop)"},
        ],
        "action": "block",
        "severity": "critical",
    },
    "spring4shell": {
        "name": "Spring4Shell - RCE - CVE:CVE-2022-22965",
        "cves": ["CVE-2022-22965"],
        "patterns": [
            {"field": "query", "op": "contains", "value": "class.module.classLoader"},
        ],
        "action": "block",
        "severity": "critical",
    },
    "moveit_sqli": {
        "name": "MOVEit Transfer - SQL Injection - CVE:CVE-2023-34362",
        "cves": ["CVE-2023-34362"],
        "patterns": [
            {"field": "path", "op": "contains", "value": "/moveitisapi/moveitisapi.dll"},
        ],
        "action": "block",
        "severity": "critical",
    },
}
