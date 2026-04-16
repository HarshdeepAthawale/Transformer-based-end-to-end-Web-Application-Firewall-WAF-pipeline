"""Phase wrapper for managed rules evaluation (regex pattern matching)."""

from typing import Optional

from gateway.config import gateway_config
from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult


class ManagedRulesPhase(Phase):
    name = "managed_rules"
    order = 90
    requires_body = True

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if not gateway_config.MANAGED_RULES_ENABLED:
            return False, "managed_rules_disabled"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        from gateway.managed_rules import evaluate

        body_str = ""
        if ctx.body_bytes:
            try:
                body_str = ctx.body_bytes.decode("utf-8", errors="replace")
            except Exception:
                body_str = ""

        match = evaluate(
            method=ctx.method,
            path=ctx.path,
            headers=ctx.headers,
            query_string=ctx.query_string,
            body=body_str,
        )

        if match and match.get("action") == "block":
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=403,
                response_body={
                    "blocked": True,
                    "message": f"Blocked by rule: {match.get('rule_name', 'unknown')}",
                    "rule_id": match.get("rule_id"),
                },
                data=match,
            )

        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"match": match},
        )
