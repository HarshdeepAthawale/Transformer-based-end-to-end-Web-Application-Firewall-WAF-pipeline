"""Analytics tools — wrap analytics and charts controllers."""

from backend.agents.tools.registry import ToolDef, registry
from backend.agents.context import AgentContext


def _get_analytics_overview(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import analytics
    return analytics.get_overview(ctx.db, range_str=kwargs.get("range", "24h"))


def _get_analytics_trends(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import analytics
    return analytics.get_trends(
        ctx.db,
        metric=kwargs.get("metric", "requests"),
        range_str=kwargs.get("range", "24h"),
    )


def _get_analytics_summary(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import analytics
    return analytics.get_summary(ctx.db, range_str=kwargs.get("range", "24h"))


def _get_request_chart(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import charts
    return charts.get_requests(ctx.db, range_str=kwargs.get("range", "24h"))


def _get_threat_chart(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import charts
    return charts.get_threats(ctx.db, range_str=kwargs.get("range", "24h"))


def register_analytics_tools() -> None:
    registry.register(ToolDef(
        name="get_analytics_overview",
        description="Get analytics overview data (traffic volume, block rate, top endpoints) for a time range.",
        parameters_json_schema={
            "type": "object",
            "properties": {"range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"}},
            "required": [],
        },
        handler=_get_analytics_overview,
    ))
    registry.register(ToolDef(
        name="get_analytics_trends",
        description="Get trend data for a specific metric over time.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Metric: requests, blocked, threats, latency"},
                "range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"},
            },
            "required": [],
        },
        handler=_get_analytics_trends,
    ))
    registry.register(ToolDef(
        name="get_analytics_summary",
        description="Get a high-level summary of WAF analytics (total requests, block rate, top threats).",
        parameters_json_schema={
            "type": "object",
            "properties": {"range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"}},
            "required": [],
        },
        handler=_get_analytics_summary,
    ))
    registry.register(ToolDef(
        name="get_request_chart",
        description="Get request volume chart data over time.",
        parameters_json_schema={
            "type": "object",
            "properties": {"range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"}},
            "required": [],
        },
        handler=_get_request_chart,
    ))
    registry.register(ToolDef(
        name="get_threat_chart",
        description="Get threat distribution chart data over time.",
        parameters_json_schema={
            "type": "object",
            "properties": {"range": {"type": "string", "description": "Time range: 1h, 6h, 24h, 7d, 30d"}},
            "required": [],
        },
        handler=_get_threat_chart,
    ))


register_analytics_tools()
