"""Remediation specialist — suggests and executes security actions."""

from __future__ import annotations

import re
import json
from typing import List

from openai import AsyncOpenAI
from backend.agents.base_agent import BaseAgent

SYSTEM_PROMPT = """\
You are a WAF Remediation Specialist. Your role is to help security teams take action: block IPs, create rules, dismiss alerts, and implement protective measures.

When remediating:
1. First investigate the situation to understand what action is needed
2. Confirm the action with the user before executing destructive operations
3. For suggested (not yet executed) actions, include them in a ===SUGGESTED_ACTIONS=== block

When you want to SUGGEST actions (not execute immediately), format them like this at the end of your response:

===SUGGESTED_ACTIONS===
[{"action": "block_ip", "params": {"ip": "1.2.3.4", "reason": "Malicious activity"}, "label": "Block 1.2.3.4"}]
===END_ACTIONS===

Available actions: block_ip, unblock_ip, whitelist_ip, dismiss_alert, acknowledge_alert, create_security_rule, create_geo_rule

Format your responses in markdown. Be explicit about what each action will do and its impact.
"""

TOOL_NAMES = [
    "block_ip",
    "unblock_ip",
    "whitelist_ip",
    "dismiss_alert",
    "acknowledge_alert",
    "create_security_rule",
    "create_geo_rule",
    # Investigation tools for context gathering
    "get_active_alerts",
    "get_recent_threats",
    "get_ip_reputation",
    "get_threat_stats",
]


def create_remediation(client: AsyncOpenAI, model: str) -> BaseAgent:
    return BaseAgent(client, model, SYSTEM_PROMPT, TOOL_NAMES)


def extract_suggested_actions(content: str) -> tuple[str, List[dict]]:
    """Parse ===SUGGESTED_ACTIONS=== blocks from agent response.

    Returns (clean_content, actions_list).
    """
    pattern = r"===SUGGESTED_ACTIONS===\s*(.*?)\s*===END_ACTIONS==="
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return content, []

    try:
        actions = json.loads(match.group(1).strip())
        if not isinstance(actions, list):
            actions = [actions]
    except json.JSONDecodeError:
        return content, []

    clean = content[: match.start()].rstrip()
    return clean, actions
