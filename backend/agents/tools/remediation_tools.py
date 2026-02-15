"""Remediation tools — wrap ip_management, alerts, security_rules, geo_rules controllers."""

from backend.agents.tools.registry import ToolDef, registry
from backend.agents.context import AgentContext


def _block_ip(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import ip_management
    return ip_management.add_to_blacklist(
        ctx.db,
        ip=kwargs["ip"],
        reason=kwargs.get("reason"),
        source="agent",
        duration_hours=kwargs.get("duration_hours"),
    )


def _unblock_ip(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import ip_management
    return ip_management.remove_from_list(ctx.db, ip=kwargs["ip"], list_type="blacklist")


def _whitelist_ip(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import ip_management
    return ip_management.add_to_whitelist(
        ctx.db, ip=kwargs["ip"], reason=kwargs.get("reason")
    )


def _dismiss_alert(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import alerts
    return alerts.dismiss(ctx.db, alert_id=kwargs["alert_id"])


def _acknowledge_alert(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import alerts
    return alerts.acknowledge(ctx.db, alert_id=kwargs["alert_id"])


def _create_security_rule(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import security_rules
    return security_rules.create_rule(
        ctx.db,
        name=kwargs["name"],
        rule_type=kwargs["rule_type"],
        pattern=kwargs["pattern"],
        applies_to=kwargs.get("applies_to", "all"),
        action=kwargs.get("action", "block"),
        priority=kwargs.get("priority", "medium"),
        description=kwargs.get("description"),
        owasp_category=kwargs.get("owasp_category"),
    )


def _create_geo_rule(ctx: AgentContext, **kwargs) -> dict:
    from backend.controllers import geo_rules
    return geo_rules.create_rule(
        ctx.db,
        rule_type=kwargs["rule_type"],
        country_code=kwargs["country_code"],
        country_name=kwargs["country_name"],
        priority=kwargs.get("priority", 0),
        exception_ips=kwargs.get("exception_ips"),
        reason=kwargs.get("reason"),
    )


def register_remediation_tools() -> None:
    registry.register(ToolDef(
        name="block_ip",
        description="Block an IP address by adding it to the blacklist.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "ip": {"type": "string", "description": "IP address to block"},
                "reason": {"type": "string", "description": "Reason for blocking"},
                "duration_hours": {"type": "integer", "description": "Block duration in hours (null=permanent)"},
            },
            "required": ["ip"],
        },
        handler=_block_ip,
    ))
    registry.register(ToolDef(
        name="unblock_ip",
        description="Remove an IP from the blacklist.",
        parameters_json_schema={
            "type": "object",
            "properties": {"ip": {"type": "string", "description": "IP address to unblock"}},
            "required": ["ip"],
        },
        handler=_unblock_ip,
    ))
    registry.register(ToolDef(
        name="whitelist_ip",
        description="Add an IP to the whitelist so it bypasses WAF checks.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "ip": {"type": "string", "description": "IP address to whitelist"},
                "reason": {"type": "string", "description": "Reason for whitelisting"},
            },
            "required": ["ip"],
        },
        handler=_whitelist_ip,
    ))
    registry.register(ToolDef(
        name="dismiss_alert",
        description="Dismiss an active alert by its ID.",
        parameters_json_schema={
            "type": "object",
            "properties": {"alert_id": {"type": "integer", "description": "Alert ID to dismiss"}},
            "required": ["alert_id"],
        },
        handler=_dismiss_alert,
    ))
    registry.register(ToolDef(
        name="acknowledge_alert",
        description="Acknowledge an active alert by its ID.",
        parameters_json_schema={
            "type": "object",
            "properties": {"alert_id": {"type": "integer", "description": "Alert ID to acknowledge"}},
            "required": ["alert_id"],
        },
        handler=_acknowledge_alert,
    ))
    registry.register(ToolDef(
        name="create_security_rule",
        description="Create a new WAF security rule (e.g. block a pattern, rate limit).",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Rule name"},
                "rule_type": {"type": "string", "description": "Rule type: regex, rate_limit, header, etc."},
                "pattern": {"type": "string", "description": "Pattern to match"},
                "applies_to": {"type": "string", "description": "Where rule applies: all, path, header, body"},
                "action": {"type": "string", "description": "Action: block, alert, rate_limit"},
                "priority": {"type": "string", "description": "Priority: low, medium, high, critical"},
                "description": {"type": "string", "description": "Rule description"},
                "owasp_category": {"type": "string", "description": "OWASP category if applicable"},
            },
            "required": ["name", "rule_type", "pattern"],
        },
        handler=_create_security_rule,
    ))
    registry.register(ToolDef(
        name="create_geo_rule",
        description="Create a geographic blocking/allowing rule.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "rule_type": {"type": "string", "description": "Rule type: block or allow"},
                "country_code": {"type": "string", "description": "ISO country code (e.g. CN, RU)"},
                "country_name": {"type": "string", "description": "Country name"},
                "priority": {"type": "integer", "description": "Rule priority (higher = evaluated first)"},
                "reason": {"type": "string", "description": "Reason for the rule"},
            },
            "required": ["rule_type", "country_code", "country_name"],
        },
        handler=_create_geo_rule,
    ))


register_remediation_tools()
