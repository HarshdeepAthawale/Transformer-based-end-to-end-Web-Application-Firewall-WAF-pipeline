"""
Attack Score Rules Engine (industry-standard cf.waf.score rules).

Evaluates WAF attack scores against configurable rules to determine
blocking/challenge/log actions. Supports overall score and sub-category
scores (SQLi, XSS, RCE).

Example rules:
  {"field": "attack_score", "op": "le", "value": 20, "action": "block"}
  {"field": "waf_sqli_score", "op": "le", "value": 30, "action": "challenge"}
"""

from typing import Any, Optional


# Valid score fields that can be referenced in rules
VALID_SCORE_FIELDS = {
    "attack_score",      # Overall WAF attack score (1-99, lower = malicious)
    "waf_sqli_score",    # SQL injection sub-score
    "waf_xss_score",     # XSS sub-score
    "waf_rce_score",     # RCE/command injection sub-score
    "bot_score",         # Bot detection score (1-99, lower = bot)
}

# Valid operators
VALID_OPS = {"le", "lt", "ge", "gt", "eq", "ne"}

# Valid actions (ordered by severity)
VALID_ACTIONS = {"block", "challenge", "log", "allow"}

# Action priority (higher = more severe)
ACTION_PRIORITY = {"allow": 0, "log": 1, "challenge": 2, "block": 3}


def _compare(value: int, op: str, threshold: int) -> bool:
    """Compare a score value against a threshold using the given operator."""
    if op == "le":
        return value <= threshold
    elif op == "lt":
        return value < threshold
    elif op == "ge":
        return value >= threshold
    elif op == "gt":
        return value > threshold
    elif op == "eq":
        return value == threshold
    elif op == "ne":
        return value != threshold
    return False


def validate_rule(rule: dict) -> Optional[str]:
    """
    Validate an attack score rule definition.
    Returns error message if invalid, None if valid.
    """
    field = rule.get("field")
    if field not in VALID_SCORE_FIELDS:
        return f"Invalid field '{field}'. Must be one of: {VALID_SCORE_FIELDS}"

    op = rule.get("op")
    if op not in VALID_OPS:
        return f"Invalid operator '{op}'. Must be one of: {VALID_OPS}"

    value = rule.get("value")
    if not isinstance(value, (int, float)) or value < 1 or value > 99:
        return f"Invalid value '{value}'. Must be integer 1-99."

    action = rule.get("action")
    if action not in VALID_ACTIONS:
        return f"Invalid action '{action}'. Must be one of: {VALID_ACTIONS}"

    return None


def evaluate_attack_score_rules(
    scores: dict[str, Any],
    rules: list[dict],
) -> dict[str, Any]:
    """
    Evaluate a set of attack score rules against request scores.

    Args:
        scores: Dict with score fields (attack_score, waf_sqli_score, etc.)
        rules: List of rule dicts with {field, op, value, action, name?}

    Returns:
        Dict with:
          - action: the most severe matching action ("block", "challenge", "log", or "allow")
          - matched_rules: list of rule names/IDs that matched
          - details: human-readable explanation
    """
    matched_rules = []
    highest_action = "allow"

    for rule in rules:
        if not rule.get("enabled", True):
            continue

        field = rule.get("field", "")
        op = rule.get("op", "")
        threshold = rule.get("value", 0)
        action = rule.get("action", "log")

        score_value = scores.get(field)
        if score_value is None:
            continue

        if _compare(int(score_value), op, int(threshold)):
            rule_name = rule.get("name", f"{field} {op} {threshold}")
            matched_rules.append({
                "name": rule_name,
                "field": field,
                "op": op,
                "threshold": threshold,
                "actual_value": score_value,
                "action": action,
            })

            if ACTION_PRIORITY.get(action, 0) > ACTION_PRIORITY.get(highest_action, 0):
                highest_action = action

    return {
        "action": highest_action,
        "matched_rules": matched_rules,
        "rule_count": len(matched_rules),
    }


# Default attack score rules (can be overridden via DB)
DEFAULT_RULES = [
    {
        "name": "Block high-confidence attacks",
        "field": "attack_score",
        "op": "le",
        "value": 15,
        "action": "block",
        "enabled": True,
    },
    {
        "name": "Challenge suspicious requests",
        "field": "attack_score",
        "op": "le",
        "value": 40,
        "action": "challenge",
        "enabled": True,
    },
    {
        "name": "Log borderline requests",
        "field": "attack_score",
        "op": "le",
        "value": 60,
        "action": "log",
        "enabled": True,
    },
]
