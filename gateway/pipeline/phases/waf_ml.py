"""Phase wrapper for WAF ML inference (DistilBERT via ONNX Runtime)."""

from typing import Optional

from gateway.config import gateway_config
from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult
from gateway.pipeline.filters import is_benign_get, is_static_asset


class WAFMLPhase(Phase):
    name = "waf_ml"
    order = 100
    requires_body = True

    def __init__(self, waf_service=None):
        self._waf_service = waf_service

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if not gateway_config.WAF_ENABLED:
            return False, "waf_disabled"
        if self._waf_service is None:
            return False, "waf_service_unavailable"
        if is_static_asset(ctx.path):
            return False, "static_asset"
        if is_benign_get(ctx.method, ctx.path, ctx.query_string):
            return False, "benign_get"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        from gateway.waf_inspect import inspect_request

        # Build query_params dict from query string
        query_params = {}
        if ctx.query_string:
            for pair in ctx.query_string.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    query_params[k] = v

        should_block, waf_result = await inspect_request(
            waf_service=self._waf_service,
            method=ctx.method,
            path=ctx.path,
            query_params=query_params,
            headers=ctx.headers,
            body=ctx.body_bytes if ctx.body_bytes else None,
        )

        waf_data = {
            "anomaly_score": waf_result.get("anomaly_score"),
            "attack_score": waf_result.get("attack_score"),
            "waf_latency_ms": waf_result.get("waf_latency_ms"),
            "is_anomaly": waf_result.get("is_anomaly", False),
            "label": waf_result.get("label"),
        }

        if should_block:
            action = waf_result.get("action", "block")
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action=action,
                status_code=403,
                response_body={
                    "blocked": True,
                    "message": "Request blocked by WAF"
                    if action == "block"
                    else "Request challenged by WAF",
                    "action": action,
                    "attack_score": waf_result.get("attack_score"),
                    "anomaly_score": waf_result.get("anomaly_score", 0.0),
                    "threshold": waf_result.get(
                        "threshold", gateway_config.WAF_THRESHOLD
                    ),
                },
                data=waf_data,
            )

        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data=waf_data,
        )
