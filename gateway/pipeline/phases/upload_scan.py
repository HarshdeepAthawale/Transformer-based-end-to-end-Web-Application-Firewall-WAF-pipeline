"""Phase wrapper for malicious upload scanning."""

from typing import Optional

from gateway.config import gateway_config
from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult
from gateway.pipeline.filters import is_multipart


class UploadScanPhase(Phase):
    name = "upload_scan"
    order = 60
    requires_body = True

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if not gateway_config.UPLOAD_SCAN_ENABLED:
            return False, "upload_scan_disabled"
        if not is_multipart(ctx.content_type):
            return False, "not_multipart"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        from gateway.upload_scan import process_upload_scan

        should_block, scan_result, error = await process_upload_scan(
            body_bytes=ctx.body_bytes or b"",
            content_type=ctx.content_type,
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
                    "message": "Malicious file detected",
                    "scan_result": scan_result,
                },
                data={"scan_result": scan_result, "error": error},
            )

        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"scan_result": scan_result, "error": error},
        )
