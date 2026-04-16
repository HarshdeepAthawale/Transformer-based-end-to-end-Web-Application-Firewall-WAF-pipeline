"""Phase wrapper for DDoS protection (size + block + burst checks)."""

from typing import Optional

from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult


class DDoSProtectionPhase(Phase):
    name = "ddos_protection"
    order = 40
    requires_body = False

    def __init__(self, ddos_protection=None):
        self._ddos = ddos_protection

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if self._ddos is None:
            return False, "no_ddos_protection"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        # Sub-check 1: request size
        content_length = None
        if "content-length" in ctx.headers:
            try:
                content_length = int(ctx.headers["content-length"])
            except (ValueError, TypeError):
                pass

        size_ok, size_reason = self._ddos.check_request_size(content_length)
        if not size_ok:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=413,
                response_body={"blocked": True, "message": size_reason},
                data={"sub_check": "size", "reason": size_reason},
            )

        # Sub-check 2: IP temporarily blocked
        is_blocked, ttl = await self._ddos.is_blocked(ctx.client_ip)
        if is_blocked:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=429,
                response_body={
                    "blocked": True,
                    "message": "Temporarily blocked (DDoS protection)",
                    "retry_after": ttl,
                },
                response_headers={"Retry-After": str(int(ttl))},
                data={"sub_check": "blocked", "ttl": ttl},
            )

        # Sub-check 3: burst detection
        allowed, was_burst = await self._ddos.record_request_and_check_burst(
            ctx.client_ip,
        )
        if not allowed:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=429,
                response_body={
                    "blocked": True,
                    "message": "DDoS burst detected",
                },
                data={"sub_check": "burst", "was_burst": was_burst},
            )

        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"ddos_allowed": True},
        )
