"""Phase wrapper for bot detection (UA signatures + behavioral analysis)."""

from typing import Optional

from gateway.config import gateway_config
from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult


class BotDetectionPhase(Phase):
    name = "bot_detection"
    order = 50
    requires_body = False

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if not gateway_config.BOT_ENABLED:
            return False, "bot_disabled"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        from gateway.bot_score import get_bot_score

        user_agent = ctx.headers.get("user-agent", "")
        result = await get_bot_score(user_agent, ctx.client_ip, ctx.headers)

        if result is None:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.CONTINUE,
                data={"bot_score_available": False},
            )

        action = result.get("action")
        bot_score = result.get("bot_score")

        if action == "block":
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=403,
                response_body={"blocked": True, "message": "Bot detected"},
                data={"bot_score": bot_score, "action": action},
            )

        if action == "challenge":
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="challenge",
                status_code=429,
                response_body={"blocked": True, "message": "Bot challenge required"},
                response_headers={
                    "Retry-After": str(gateway_config.BOT_CHALLENGE_RETRY_AFTER),
                },
                data={"bot_score": bot_score, "action": action},
            )

        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"bot_score": bot_score, "action": action or "allow"},
        )
