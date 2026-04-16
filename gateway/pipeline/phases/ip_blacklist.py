"""Phase wrapper for IP blacklist checking (Redis O(1) lookup)."""

from typing import Optional

from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult


class IPBlacklistPhase(Phase):
    name = "ip_blacklist"
    order = 10
    requires_body = False

    def __init__(self, blacklist_checker=None):
        self._checker = blacklist_checker

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if self._checker is None:
            return False, "no_blacklist_checker"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        is_blocked, reason = await self._checker.is_blocked(ctx.client_ip)
        if is_blocked:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=403,
                response_body={
                    "blocked": True,
                    "message": reason or "IP is blacklisted",
                },
                data={"ip": ctx.client_ip, "reason": reason},
            )
        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"ip_allowed": True},
        )
