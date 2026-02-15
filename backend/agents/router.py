"""Intent router — keyword-based classifier. No LLM call, fast and deterministic."""

from __future__ import annotations

import re
from enum import Enum


class AgentIntent(str, Enum):
    INVESTIGATE = "investigate"
    REMEDIATE = "remediate"
    ANALYZE = "analyze"
    EXPLAIN = "explain"
    FORENSICS = "forensics"


# Patterns checked in priority order — first match wins
_INTENT_PATTERNS: list[tuple[AgentIntent, re.Pattern]] = [
    (
        AgentIntent.REMEDIATE,
        re.compile(
            r"\b(block|unblock|whitelist|blacklist|ban|dismiss|acknowledge|create\s+rule|"
            r"add\s+rule|remove|remediat|mitigat|take\s+action|fix|stop\s+attack|"
            r"geo.?block|geo.?rule|security\s+rule)\b",
            re.IGNORECASE,
        ),
    ),
    (
        AgentIntent.FORENSICS,
        re.compile(
            r"\b(forensic|audit|timeline|trace|investig\w+\s+ip|track|"
            r"who\s+changed|who\s+did|history\s+of|log\s+of|"
            r"ip\s+reputation|reputation\s+of|what\s+did\s+ip)\b",
            re.IGNORECASE,
        ),
    ),
    (
        AgentIntent.INVESTIGATE,
        re.compile(
            r"\b(alert|threat|attack|incident|suspicious|breach|intrusion|"
            r"active\s+threat|what.?s\s+happening|current\s+threat|"
            r"show\s+me\s+threat|recent\s+threat|under\s+attack)\b",
            re.IGNORECASE,
        ),
    ),
    (
        AgentIntent.ANALYZE,
        re.compile(
            r"\b(analy\w*|trend|traffic\s+pattern|traffic\s+trend|traffic\s+analysis|"
            r"statistic|metric|chart|graph|overview|summary|dashboard|volume|"
            r"rate|performance|how\s+many|how\s+much|compare|percentage)\b",
            re.IGNORECASE,
        ),
    ),
    (
        AgentIntent.EXPLAIN,
        re.compile(
            r"\b(explain|what\s+is|what\s+are|how\s+does|why\s+does|teach|"
            r"describe|definition|meaning|owasp|best\s+practice|"
            r"difference\s+between|help\s+me\s+understand)\b",
            re.IGNORECASE,
        ),
    ),
]


def classify_intent(message: str) -> AgentIntent:
    """Classify user message into an agent intent using keyword patterns."""
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(message):
            return intent
    # Default to investigation
    return AgentIntent.INVESTIGATE
