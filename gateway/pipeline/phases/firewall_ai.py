"""Phase wrapper for Firewall for AI (LLM endpoint protection)."""

from typing import Optional

from gateway.config import gateway_config
from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult


class FirewallAIPhase(Phase):
    name = "firewall_ai"
    order = 70
    requires_body = True

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if not gateway_config.FIREWALL_AI_ENABLED:
            return False, "firewall_ai_disabled"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        from gateway.firewall_ai import evaluate_request

        should_block, event_type, pattern = await evaluate_request(
            path=ctx.path,
            method=ctx.method,
            body=ctx.body_bytes or b"",
            headers=ctx.headers,
            client_ip=ctx.client_ip,
        )

        if should_block:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=403,
                response_body={
                    "blocked": True,
                    "message": "Request blocked by Firewall for AI",
                    "event_type": event_type,
                },
                data={"event_type": event_type, "pattern": pattern},
            )

        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"event_type": event_type, "pattern": pattern},
        )
