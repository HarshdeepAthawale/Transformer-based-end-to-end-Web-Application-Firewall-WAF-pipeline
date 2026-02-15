"""Forensics tools — wrap audit, ip_management, traffic, threats controllers."""

from backend.agents.tools.registry import ToolDef, registry
from backend.agents.context import AgentContext


def _get_audit_logs(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import audit
    return audit.get_logs(
        ctx.db,
        limit=kwargs.get("limit", 50),
        action=kwargs.get("action"),
        resource_type=kwargs.get("resource_type"),
        start_time=kwargs.get("start_time"),
    )


def _get_ip_reputation(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import ip_management
    return ip_management.get_ip_reputation(ctx.db, ip=kwargs["ip"])


def _get_traffic_by_ip(ctx: AgentContext, **kwargs) -> dict:
    """Get traffic from a specific IP by filtering recent traffic."""
    from backend.controllers import traffic
    result = traffic.get_recent(ctx.db, limit=200)
    if result.get("success") and result.get("data"):
        ip = kwargs["ip"]
        filtered = [t for t in result["data"] if t.get("ip") == ip]
        result["data"] = filtered[:kwargs.get("limit", 50)]
    return result


def _get_threat_timeline(ctx: AgentContext, **kwargs) -> dict:
    """Get threats over a range to build a timeline."""
    from backend.controllers import threats
    return threats.get_by_range(ctx.db, range_str=kwargs.get("range", "24h"))


def register_forensics_tools() -> None:
    registry.register(ToolDef(
        name="get_audit_logs",
        description="Get audit trail logs (user actions, config changes). Filterable by action and resource type.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max entries to return"},
                "action": {"type": "string", "description": "Filter by action type"},
                "resource_type": {"type": "string", "description": "Filter by resource type"},
                "start_time": {"type": "string", "description": "ISO timestamp to filter from"},
            },
            "required": [],
        },
        handler=_get_audit_logs,
    ))
    registry.register(ToolDef(
        name="get_ip_reputation",
        description="Look up the reputation/risk score for an IP address.",
        parameters_json_schema={
            "type": "object",
            "properties": {"ip": {"type": "string", "description": "IP address to look up"}},
            "required": ["ip"],
        },
        handler=_get_ip_reputation,
    ))
    registry.register(ToolDef(
        name="get_traffic_by_ip",
        description="Get recent traffic logs originating from a specific IP address.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "ip": {"type": "string", "description": "IP address to trace"},
                "limit": {"type": "integer", "description": "Max entries to return"},
            },
            "required": ["ip"],
        },
        handler=_get_traffic_by_ip,
    ))
    registry.register(ToolDef(
        name="get_threat_timeline",
        description="Get a chronological timeline of threats for a time range.",
        parameters_json_schema={
            "type": "object",
            "properties": {"range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"}},
            "required": [],
        },
        handler=_get_threat_timeline,
    ))


register_forensics_tools()
