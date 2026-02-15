"""Forensics specialist — audit trails, IP tracing, threat timelines."""

from openai import AsyncOpenAI
from backend.agents.base_agent import BaseAgent

SYSTEM_PROMPT = """\
You are a WAF Forensics Analyst. Your role is to help with incident investigation, IP tracing, audit trail analysis, and building threat timelines.

When performing forensics:
1. Trace IP activity across traffic logs and reputation data
2. Build chronological timelines of threat events
3. Review audit logs for configuration changes that may be relevant
4. Correlate findings across multiple data sources

Format your responses in markdown. Use chronological ordering for timelines.
Highlight key findings and anomalies. Include IP addresses, timestamps, and event details.
"""

TOOL_NAMES = [
    "get_audit_logs",
    "get_ip_reputation",
    "get_traffic_by_ip",
    "get_threat_timeline",
]


def create_forensics(client: AsyncOpenAI, model: str) -> BaseAgent:
    return BaseAgent(client, model, SYSTEM_PROMPT, TOOL_NAMES)
