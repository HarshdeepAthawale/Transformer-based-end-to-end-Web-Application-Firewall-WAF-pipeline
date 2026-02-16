"""
WAF inspection logic for the gateway.
"""

import json
import time
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from gateway.config import gateway_config


async def inspect_request(
    waf_service,
    method: str,
    path: str,
    query_params: dict,
    headers: dict,
    body: Optional[bytes],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run WAF inspection on request components.

    Returns:
        (should_block, result_dict)
        In monitor mode, should_block is always False.
    """
    if waf_service is None:
        if gateway_config.WAF_FAIL_OPEN:
            logger.warning("WAF service unavailable, fail-open: allowing request")
            return False, {"skipped": True, "reason": "waf_unavailable"}
        else:
            logger.error("WAF service unavailable, fail-closed: blocking request")
            return True, {"skipped": True, "reason": "waf_unavailable_fail_closed"}

    try:
        # Parse body for inspection (truncate to BODY_MAX_BYTES)
        body_for_inspection = None
        if body:
            body_text = body[: gateway_config.BODY_MAX_BYTES].decode(
                "utf-8", errors="ignore"
            )
            try:
                body_for_inspection = json.loads(body_text)
            except (json.JSONDecodeError, ValueError):
                body_for_inspection = body_text

        start = time.perf_counter()
        result = await waf_service.check_request_async(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            body=body_for_inspection,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        is_anomaly = result.get("is_anomaly", False)
        anomaly_score = result.get("anomaly_score", 0.0)
        result["waf_latency_ms"] = round(elapsed_ms, 2)

        if is_anomaly:
            client_ip = headers.get("x-forwarded-for", "unknown")
            if gateway_config.WAF_MODE == "block":
                logger.warning(
                    f"WAF BLOCK: {method} {path} | "
                    f"score={anomaly_score:.4f} | ip={client_ip}"
                )
                return True, result
            else:
                logger.warning(
                    f"WAF MONITOR (would block): {method} {path} | "
                    f"score={anomaly_score:.4f} | ip={client_ip}"
                )
                return False, result

        return False, result

    except Exception as exc:
        logger.error(f"WAF inspection error: {exc}", exc_info=True)
        if gateway_config.WAF_FAIL_OPEN:
            return False, {"error": str(exc), "skipped": True}
        else:
            return True, {"error": str(exc), "skipped": True}
