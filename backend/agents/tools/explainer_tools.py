"""Explainer tools — wrap bot_detection, security_rules, ip_management controllers + static knowledge."""

from backend.agents.tools.registry import ToolDef, registry
from backend.agents.context import AgentContext

# Static threat-type knowledge base
THREAT_EXPLANATIONS = {
    "sql_injection": {
        "name": "SQL Injection",
        "description": "An attack where malicious SQL code is inserted into application queries via user input.",
        "risk": "Can lead to data exfiltration, authentication bypass, and database destruction.",
        "mitigation": "Use parameterized queries, input validation, WAF rules, and least-privilege DB accounts.",
        "owasp": "A03:2021 - Injection",
    },
    "xss": {
        "name": "Cross-Site Scripting (XSS)",
        "description": "An attack where malicious scripts are injected into web pages viewed by other users.",
        "risk": "Session hijacking, credential theft, defacement, malware distribution.",
        "mitigation": "Output encoding, Content Security Policy (CSP), input sanitization, HttpOnly cookies.",
        "owasp": "A03:2021 - Injection",
    },
    "ddos": {
        "name": "Distributed Denial of Service",
        "description": "An attack that overwhelms a server with traffic from multiple sources to make it unavailable.",
        "risk": "Service downtime, revenue loss, resource exhaustion.",
        "mitigation": "Rate limiting, geo-blocking, traffic analysis, CDN/WAF layer, auto-scaling.",
        "owasp": "N/A (Availability threat)",
    },
    "rfi": {
        "name": "Remote File Inclusion",
        "description": "An attack that tricks the application into including remote files with malicious code.",
        "risk": "Remote code execution, server compromise, data theft.",
        "mitigation": "Disable remote file inclusion, input validation, allowlists for includes.",
        "owasp": "A03:2021 - Injection",
    },
    "lfi": {
        "name": "Local File Inclusion",
        "description": "An attack that tricks the application into reading local files (e.g. /etc/passwd).",
        "risk": "Information disclosure, credential leakage, code execution via log poisoning.",
        "mitigation": "Input validation, chroot jails, disable directory traversal patterns.",
        "owasp": "A01:2021 - Broken Access Control",
    },
    "csrf": {
        "name": "Cross-Site Request Forgery",
        "description": "An attack that tricks authenticated users into performing unwanted actions.",
        "risk": "Unauthorized state changes, fund transfers, account modifications.",
        "mitigation": "CSRF tokens, SameSite cookies, origin header validation.",
        "owasp": "A01:2021 - Broken Access Control",
    },
    "path_traversal": {
        "name": "Path Traversal",
        "description": "An attack using ../ sequences to access files outside the intended directory.",
        "risk": "Information disclosure, access to sensitive system files.",
        "mitigation": "Input validation, canonicalization, chroot, restrict file access paths.",
        "owasp": "A01:2021 - Broken Access Control",
    },
    "command_injection": {
        "name": "Command Injection",
        "description": "An attack where OS commands are injected through application inputs.",
        "risk": "Full server compromise, data theft, lateral movement.",
        "mitigation": "Avoid system calls, use safe APIs, input validation, sandboxing.",
        "owasp": "A03:2021 - Injection",
    },
}


def _get_bot_signatures(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import bot_detection
    return bot_detection.get_signatures(ctx.db, active_only=kwargs.get("active_only", True))


def _get_security_rules(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import security_rules
    return security_rules.get_rules(ctx.db, active_only=kwargs.get("active_only", True))


def _get_owasp_rules(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import security_rules
    return security_rules.get_owasp_rules(ctx.db)


def _get_ip_reputation_explain(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import ip_management
    return ip_management.get_ip_reputation(ctx.db, ip=kwargs["ip"])


def _explain_threat_type(ctx: AgentContext, **kwargs) -> dict:
    threat_type = kwargs["threat_type"].lower().replace(" ", "_").replace("-", "_")
    info = THREAT_EXPLANATIONS.get(threat_type)
    if info:
        return {"success": True, "data": info}
    return {
        "success": True,
        "data": {
            "name": kwargs["threat_type"],
            "description": f"No detailed knowledge available for '{kwargs['threat_type']}'. The LLM can provide general information.",
            "risk": "Varies",
            "mitigation": "Consult OWASP guidelines and vendor documentation.",
            "owasp": "Unknown",
        },
    }


def register_explainer_tools() -> None:
    registry.register(ToolDef(
        name="get_bot_signatures",
        description="List all configured bot detection signatures/patterns.",
        parameters_json_schema={
            "type": "object",
            "properties": {"active_only": {"type": "boolean", "description": "Only return active signatures"}},
            "required": [],
        },
        handler=_get_bot_signatures,
    ))
    registry.register(ToolDef(
        name="get_security_rules",
        description="List all WAF security rules currently configured.",
        parameters_json_schema={
            "type": "object",
            "properties": {"active_only": {"type": "boolean", "description": "Only return active rules"}},
            "required": [],
        },
        handler=_get_security_rules,
    ))
    registry.register(ToolDef(
        name="get_owasp_rules",
        description="Get OWASP-categorized security rules.",
        parameters_json_schema={"type": "object", "properties": {}, "required": []},
        handler=_get_owasp_rules,
    ))
    registry.register(ToolDef(
        name="get_ip_reputation_explain",
        description="Look up IP reputation for explanation purposes.",
        parameters_json_schema={
            "type": "object",
            "properties": {"ip": {"type": "string", "description": "IP address to look up"}},
            "required": ["ip"],
        },
        handler=_get_ip_reputation_explain,
    ))
    registry.register(ToolDef(
        name="explain_threat_type",
        description="Get a detailed explanation of an attack type (SQL injection, XSS, DDoS, etc.).",
        parameters_json_schema={
            "type": "object",
            "properties": {"threat_type": {"type": "string", "description": "Attack type to explain"}},
            "required": ["threat_type"],
        },
        handler=_explain_threat_type,
    ))


register_explainer_tools()
