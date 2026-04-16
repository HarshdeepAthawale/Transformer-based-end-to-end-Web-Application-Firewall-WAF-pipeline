"""Phase wrapper for rate limiting (Redis sliding window)."""

from typing import Optional

from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult


class RateLimitPhase(Phase):
    name = "rate_limit"
    order = 30
    requires_body = False

    def __init__(self, rate_limiter=None):
        self._limiter = rate_limiter

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if self._limiter is None:
            return False, "no_rate_limiter"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        allowed, retry_after = await self._limiter.is_allowed(ctx.client_ip)
        if not allowed:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=429,
                response_body={
                    "blocked": True,
                    "message": "Rate limit exceeded",
                    "retry_after": retry_after,
                },
                response_headers={"Retry-After": str(int(retry_after))},
                data={"retry_after": retry_after},
            )
        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"rate_allowed": True},
        )
