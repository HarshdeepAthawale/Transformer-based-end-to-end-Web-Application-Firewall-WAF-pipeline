"""Phase wrapper for credential leak detection (HIBP password check)."""

from typing import Optional

from gateway.config import gateway_config
from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult
from gateway.pipeline.filters import is_login_path


class CredentialLeakPhase(Phase):
    name = "credential_leak"
    order = 80
    requires_body = True

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if not gateway_config.CREDENTIAL_LEAK_ENABLED:
            return False, "credential_leak_disabled"
        if not is_login_path(ctx.path, ctx.method):
            return False, "not_login_path"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        from gateway.credential_leak import process_credential_leak

        should_block, event_type = await process_credential_leak(
            body_bytes=ctx.body_bytes or b"",
            path=ctx.path,
            client_ip=ctx.client_ip,
            method=ctx.method,
        )

        if should_block:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="block",
                status_code=403,
                response_body={
                    "blocked": True,
                    "message": "Credential leak detected",
                    "event_type": event_type,
                },
                data={"event_type": event_type},
            )

        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"event_type": event_type},
        )
