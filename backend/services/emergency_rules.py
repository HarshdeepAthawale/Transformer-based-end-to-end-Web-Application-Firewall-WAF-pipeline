"""
Emergency Rule Service: fast-deploy rules for zero-day threat response.

Emergency rules are checked BEFORE ML inference in the WAF middleware.
They use Aho-Corasick multi-pattern string matching and regex to block
known exploit patterns without waiting for the transformer model.

Aho-Corasick automaton enables O(n) matching of all string patterns
simultaneously, vs O(p*n) for naive per-pattern substring search.
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

import ahocorasick
from sqlalchemy.orm import Session
from loguru import logger

from backend.models.emergency_rule import EmergencyRule


class EmergencyRuleService:
    """Manages and evaluates emergency security rules."""

    def __init__(self):
        self._rules_cache: list[dict] = []
        self._cache_loaded = False
        # Per-field Aho-Corasick automatons: field -> automaton
        # Each automaton value: (rule_idx, pattern_idx) for AND-logic tracking
        self._automatons: dict[str, ahocorasick.Automaton] = {}

    def load_rules(self, db: Session) -> None:
        """Load enabled emergency rules into memory cache and rebuild automatons."""
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
        self._rebuild_automatons()
        logger.info(f"Loaded {len(self._rules_cache)} emergency rules into cache")

    def _rebuild_automatons(self) -> None:
        """
        Build per-field Aho-Corasick automatons from all 'contains' patterns.

        Each entry in the automaton maps a lowercased string value to the set of
        (rule_idx, pattern_idx) pairs that require it, so a single O(n) scan
        over the field text resolves all string patterns for that field.
        """
        # Collect: field -> {lowercased_value -> set of (rule_idx, pat_idx)}
        field_patterns: dict[str, dict[str, set]] = {}

        for rule_idx, rule in enumerate(self._rules_cache):
            for pat_idx, pattern in enumerate(rule["patterns"]):
                if pattern.get("op") != "contains":
                    continue
                field = pattern.get("field", "")
                value = pattern.get("value", "").lower()
                if not value:
                    continue
                field_patterns.setdefault(field, {}).setdefault(value, set()).add(
                    (rule_idx, pat_idx)
                )

        self._automatons = {}
        for field, value_map in field_patterns.items():
            A = ahocorasick.Automaton()
            for keyword, pairs in value_map.items():
                A.add_word(keyword, (keyword, pairs))
            A.make_automaton()
            self._automatons[field] = A

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

        Uses Aho-Corasick for O(n) multi-pattern 'contains' matching,
        falling back to regex/equals/starts_with for non-string patterns.
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

        # Pre-compute which (rule_idx, pat_idx) pairs are satisfied by
        # Aho-Corasick string matching.  One scan per field that has an automaton.
        ac_matched: set[tuple[int, int]] = set()
        for field, automaton in self._automatons.items():
            target = self._get_field(request_data, field).lower()
            for _, (_keyword, pairs) in automaton.iter(target):
                ac_matched.update(pairs)

        for rule_idx, rule in enumerate(self._rules_cache):
            if self._matches_rule(request_data, rule["patterns"], rule_idx, ac_matched):
                return {
                    "blocked": True,
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "action": rule["action"],
                    "severity": rule["severity"],
                }

        return None

    def _get_field(self, request_data: dict, field: str) -> str:
        """Extract the string value for a named field from a request_data dict."""
        if field == "path":
            return request_data["path"]
        elif field == "query":
            return request_data["query"]
        elif field == "body":
            return request_data["body"]
        elif field == "method":
            return request_data["method"]
        elif field.startswith("header:"):
            header_name = field.split(":", 1)[1]
            return request_data["headers"].get(header_name, "")
        return ""

    def _matches_rule(
        self,
        request_data: dict,
        patterns: list[dict],
        rule_idx: int,
        ac_matched: set[tuple[int, int]],
    ) -> bool:
        """
        Check if a request matches ALL patterns in a rule (AND logic).

        'contains' patterns are resolved from the pre-computed ac_matched set
        (Aho-Corasick). Other operators (regex, equals, starts_with) are
        evaluated per-pattern.
        """
        if not patterns:
            return False

        for pat_idx, pattern in enumerate(patterns):
            field = pattern.get("field", "")
            op = pattern.get("op", "")
            value = pattern.get("value", "")

            if op == "contains":
                # Resolved via Aho-Corasick pre-scan
                if (rule_idx, pat_idx) not in ac_matched:
                    return False
            else:
                target = self._get_field(request_data, field)
                if not target and not field:
                    return False

                if op == "regex":
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
