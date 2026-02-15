"""Explainer specialist — security education and configuration explanation."""

from openai import AsyncOpenAI
from backend.agents.base_agent import BaseAgent

SYSTEM_PROMPT = """\
You are a WAF Security Educator. Your role is to explain security concepts, attack types, WAF configurations, and best practices in clear, accessible language.

When explaining:
1. Use the explain_threat_type tool for detailed attack information
2. Reference actual configured rules and signatures when relevant
3. Relate explanations to the user's WAF setup
4. Include OWASP references where applicable

Format your responses in markdown. Use headers for sections.
Provide examples when helpful. Tailor complexity to the question — brief for simple queries, detailed for complex ones.
"""

TOOL_NAMES = [
    "get_bot_signatures",
    "get_security_rules",
    "get_owasp_rules",
    "get_ip_reputation_explain",
    "explain_threat_type",
]


def create_explainer(client: AsyncOpenAI, model: str) -> BaseAgent:
    return BaseAgent(client, model, SYSTEM_PROMPT, TOOL_NAMES)
