"""Investigation tools — wrap alerts, threats, traffic, metrics controllers."""

from backend.agents.tools.registry import ToolDef, registry
from backend.agents.context import AgentContext


def _get_active_alerts(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import alerts
    return alerts.get_active(ctx.db)


def _get_recent_threats(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import threats
    return threats.get_recent(ctx.db, limit=kwargs.get("limit", 20))


def _get_threat_stats(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import threats
    return threats.get_stats(ctx.db, range_str=kwargs.get("range", "24h"))


def _get_recent_traffic(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import traffic
    return traffic.get_recent(ctx.db, limit=kwargs.get("limit", 20))


def _get_realtime_metrics(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import metrics
    return metrics.get_realtime(ctx.db)


def _get_threats_by_type(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import threats
    return threats.get_by_type(
        ctx.db,
        threat_type=kwargs["threat_type"],
        range_str=kwargs.get("range", "24h"),
    )


def _get_traffic_by_endpoint(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import traffic
    return traffic.get_by_endpoint(
        ctx.db,
        endpoint=kwargs["endpoint"],
        range_str=kwargs.get("range", "24h"),
    )


def register_investigation_tools() -> None:
    registry.register(ToolDef(
        name="get_active_alerts",
        description="Get all currently active (unacknowledged) alerts.",
        parameters_json_schema={"type": "object", "properties": {}, "required": []},
        handler=_get_active_alerts,
    ))
    registry.register(ToolDef(
        name="get_recent_threats",
        description="Get the most recent detected threats. Optional limit param (default 20).",
        parameters_json_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Number of threats to return"}},
            "required": [],
        },
        handler=_get_recent_threats,
    ))
    registry.register(ToolDef(
        name="get_threat_stats",
        description="Get threat statistics (counts by type/severity) for a time range.",
        parameters_json_schema={
            "type": "object",
            "properties": {"range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"}},
            "required": [],
        },
        handler=_get_threat_stats,
    ))
    registry.register(ToolDef(
        name="get_recent_traffic",
        description="Get recent HTTP traffic logs. Optional limit param (default 20).",
        parameters_json_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Number of traffic entries"}},
            "required": [],
        },
        handler=_get_recent_traffic,
    ))
    registry.register(ToolDef(
        name="get_realtime_metrics",
        description="Get real-time WAF metrics: requests/sec, blocked count, attack rate, uptime.",
        parameters_json_schema={"type": "object", "properties": {}, "required": []},
        handler=_get_realtime_metrics,
    ))
    registry.register(ToolDef(
        name="get_threats_by_type",
        description="Get threats filtered by attack type (sql_injection, xss, ddos, etc.).",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "threat_type": {"type": "string", "description": "Attack type to filter by"},
                "range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"},
            },
            "required": ["threat_type"],
        },
        handler=_get_threats_by_type,
    ))
    registry.register(ToolDef(
        name="get_traffic_by_endpoint",
        description="Get traffic logs for a specific URL endpoint.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "URL path to filter traffic"},
                "range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"},
            },
            "required": ["endpoint"],
        },
        handler=_get_traffic_by_endpoint,
    ))


register_investigation_tools()
