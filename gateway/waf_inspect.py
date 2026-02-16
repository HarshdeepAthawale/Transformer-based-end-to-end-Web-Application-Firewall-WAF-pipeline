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
        # Parse body for inspection (truncate to WAF_BODY_INSPECT_MAX, not full proxy limit)
        body_for_inspection = None
        if body:
            body_text = body[: gateway_config.WAF_BODY_INSPECT_MAX].decode(
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

        anomaly_score = result.get("anomaly_score", 0.0)
        result["waf_latency_ms"] = round(elapsed_ms, 2)

        # Derive integer attack score (0-100)
        attack_score = result.get("attack_score")
        if attack_score is None:
            attack_score = max(0, min(100, int(round(anomaly_score * 100))))
        result["attack_score"] = attack_score

        block_threshold = gateway_config.WAF_ATTACK_SCORE_BLOCK_THRESHOLD
        challenge_threshold = gateway_config.WAF_ATTACK_SCORE_CHALLENGE_THRESHOLD
        client_ip = headers.get("x-forwarded-for", "unknown")

        if gateway_config.WAF_MODE == "block":
            if attack_score >= block_threshold:
                logger.warning(
                    f"WAF BLOCK: {method} {path} | "
                    f"attack_score={attack_score} (>={block_threshold}) | ip={client_ip}"
                )
                result["action"] = "block"
                return True, result

            if challenge_threshold > 0 and attack_score >= challenge_threshold:
                logger.warning(
                    f"WAF CHALLENGE: {method} {path} | "
                    f"attack_score={attack_score} (>={challenge_threshold}) | ip={client_ip}"
                )
                result["action"] = "challenge"
                return True, result
        else:
            # Monitor mode: never block, but log high scores
            if attack_score >= block_threshold:
                logger.warning(
                    f"WAF MONITOR (would block): {method} {path} | "
                    f"attack_score={attack_score} | ip={client_ip}"
                )

        return False, result

    except Exception as exc:
        logger.error(f"WAF inspection error: {exc}", exc_info=True)
        if gateway_config.WAF_FAIL_OPEN:
            return False, {"error": str(exc), "skipped": True}
        else:
            return True, {"error": str(exc), "skipped": True}
