"""
WAF Gateway - FastAPI reverse proxy with ML-based threat detection.

Usage:
    UPSTREAM_URL=http://your-app:8080 python -m gateway.main
    UPSTREAM_URL=http://your-app:8080 uvicorn gateway.main:app --host 0.0.0.0 --port 8080
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from gateway.config import gateway_config
from gateway.mongo import MongoEventStore
from gateway.proxy import forward_request
from gateway.waf_inspect import inspect_request
from gateway.rate_limit import create_rate_limiter
from gateway.ddos_protection import create_ddos_protection
from gateway.blacklist import create_blacklist_checker
from gateway.events import report_event, start_event_batcher, stop_event_batcher

# Configure logger
logger.add(
    "logs/gateway.log",
    rotation="10 MB",
    retention="7 days",
    level=gateway_config.LOG_LEVEL,
)


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request (X-Forwarded-For or client)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage gateway lifecycle: httpx client + WAF classifier + rate limit + DDoS + MongoDB."""
    logger.info(
        f"Starting WAF Gateway on "
        f"{gateway_config.GATEWAY_HOST}:{gateway_config.GATEWAY_PORT}"
    )
    logger.info(f"Upstream: {gateway_config.UPSTREAM_URL}")
    logger.info(
        f"WAF mode: {gateway_config.WAF_MODE} | "
        f"enabled: {gateway_config.WAF_ENABLED} | "
        f"threshold: {gateway_config.WAF_THRESHOLD}"
    )

    # Create shared httpx client
    app.state.http_client = httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(gateway_config.PROXY_TIMEOUT),
        limits=httpx.Limits(
            max_connections=gateway_config.PROXY_MAX_CONNECTIONS,
            max_keepalive_connections=20,
        ),
    )

    # MongoDB event store
    app.state.event_store = MongoEventStore()
    await app.state.event_store.connect()

    # Rate limiter (Redis-backed)
    app.state.rate_limiter = create_rate_limiter()
    if app.state.rate_limiter:
        logger.info("Rate limiting enabled")
    else:
        logger.info("Rate limiting disabled or unavailable")

    # DDoS protection
    app.state.ddos_protection = create_ddos_protection()
    if app.state.ddos_protection:
        logger.info("DDoS protection enabled")
    else:
        logger.info("DDoS protection disabled or unavailable")

    # IP blacklist (backend syncs to Redis)
    app.state.blacklist_checker = create_blacklist_checker()
    if app.state.blacklist_checker:
        logger.info("IP blacklist enforcement enabled")
    else:
        logger.info("IP blacklist disabled or unavailable")

    # Event batching: batch events before POST to backend
    start_event_batcher()

    # Initialize WAF classifier (reuse backend factory)
    app.state.waf_service = None
    if gateway_config.WAF_ENABLED:
        try:
            from backend.core.waf_factory import create_waf_service, is_model_available

            app.state.waf_service = create_waf_service()
            if app.state.waf_service:
                logger.info("WAF classifier loaded successfully")
            else:
                logger.warning(
                    "WAF classifier not available (model may be missing)"
                )
                if os.getenv("WAF_REQUIRE_MODEL", "false").lower() == "true":
                    if not is_model_available():
                        logger.critical(
                            "WAF_REQUIRE_MODEL=true but model is missing. "
                            "Ensure models/waf-distilbert exists. Exiting."
                        )
                        raise SystemExit(1)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error(f"Failed to initialize WAF classifier: {exc}")

    yield

    # Shutdown
    logger.info("Shutting down WAF Gateway...")
    await stop_event_batcher()
    await app.state.http_client.aclose()
    await app.state.event_store.close()
    if app.state.rate_limiter:
        await app.state.rate_limiter.close()
    if app.state.ddos_protection:
        await app.state.ddos_protection.close()
    if getattr(app.state, "blacklist_checker", None):
        await app.state.blacklist_checker.close()
    logger.info("Gateway shutdown complete")


app = FastAPI(
    title="WAF Gateway",
    description="Reverse proxy with transformer-based WAF inspection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/gateway/docs",
    openapi_url="/gateway/openapi.json",
)


@app.get("/gateway/health")
async def health(request: Request):
    """Gateway health check endpoint (liveness)."""
    model_loaded = False
    try:
        waf_svc = getattr(request.app.state, "waf_service", None)
        if waf_svc is not None and getattr(waf_svc, "is_loaded", False):
            model_loaded = True
    except Exception:
        pass
    return {
        "status": "healthy",
        "service": "waf-gateway",
        "waf_enabled": gateway_config.WAF_ENABLED,
        "model_loaded": model_loaded,
        "waf_mode": gateway_config.WAF_MODE,
        "upstream": gateway_config.UPSTREAM_URL,
        "rate_limit_enabled": gateway_config.RATE_LIMIT_ENABLED,
        "ddos_enabled": gateway_config.DDOS_ENABLED,
    }


@app.get("/gateway/ready")
async def ready(request: Request):
    """Readiness check — verifies MongoDB and upstream are reachable."""
    checks = {}

    # MongoDB check
    event_store: MongoEventStore = request.app.state.event_store
    checks["mongodb"] = await event_store.is_ready()

    # Upstream check
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(gateway_config.UPSTREAM_URL)
            checks["upstream"] = resp.status_code < 500
    except Exception:
        checks["upstream"] = False

    all_ready = all(checks.values())
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "ready": all_ready,
            "checks": checks,
        },
    )


def _log_gateway_event(
    request: Request,
    *,
    request_id: str,
    client_ip: str,
    body_bytes: bytes | None = None,
    decision: str,
    anomaly_score: float | None = None,
    attack_score: int | None = None,
    waf_latency_ms: float | None = None,
    upstream_status: int | None = None,
    total_latency_ms: float,
    blocked_by: str | None = None,
    retry_after_seconds: int | None = None,
    block_duration_seconds: int | None = None,
    content_length: int | None = None,
    max_bytes: int | None = None,
    block_ttl_seconds: int | None = None,
) -> None:
    """Helper to log an event to both MongoDB and the legacy event reporter."""
    event_store: MongoEventStore = request.app.state.event_store
    headers = dict(request.headers)
    header_content_length = headers.get("content-length")

    event_store.log_event(
        request_id=request_id,
        client_ip=client_ip,
        method=request.method,
        path=request.url.path,
        query_string=str(request.url.query),
        headers=headers,
        body=body_bytes,
        decision=decision,
        anomaly_score=anomaly_score,
        waf_latency_ms=waf_latency_ms,
        upstream_status=upstream_status,
        total_latency_ms=total_latency_ms,
        blocked_by=blocked_by,
        user_agent=headers.get("user-agent", ""),
        content_length=int(header_content_length) if header_content_length else None,
    )

    # Build payload for backend ingest (IngestEvent: retry_after, content_length, max_bytes, block_ttl_seconds, block_duration_seconds)
    event_payload = {
        "event_type": blocked_by or "allow",
        "request_id": request_id,
        "ip": client_ip,
        "method": request.method,
        "path": request.url.path,
        "decision": decision,
        "anomaly_score": anomaly_score,
    }
    if attack_score is not None:
        event_payload["attack_score"] = attack_score
    if retry_after_seconds is not None:
        event_payload["retry_after"] = retry_after_seconds
    if block_duration_seconds is not None:
        event_payload["block_duration_seconds"] = block_duration_seconds
    if content_length is not None:
        event_payload["content_length"] = content_length
    if max_bytes is not None:
        event_payload["max_bytes"] = max_bytes
    if block_ttl_seconds is not None:
        event_payload["block_ttl_seconds"] = block_ttl_seconds
    report_event(event_payload)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def gateway_proxy(request: Request, path: str):
    """
    Catch-all route: rate limit -> DDoS check -> WAF inspect -> forward to upstream.
    """
    # Skip WAF for gateway internal endpoints
    if request.url.path.startswith("/gateway/"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    client_ip = _get_client_ip(request)

    # --- IP blacklist check (first; backend syncs to Redis) ---
    blacklist = getattr(request.app.state, "blacklist_checker", None)
    if blacklist:
        is_blocked, reason = await blacklist.is_blocked(client_ip)
        if is_blocked:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="blacklist",
            )
            resp = JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "message": reason or "IP is blacklisted",
                },
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

    # --- Rate limit check (before body read) ---
    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter:
        allowed, retry_after = await rate_limiter.is_allowed(client_ip)
        if not allowed:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="rate_limit",
                retry_after_seconds=max(1, int(retry_after)),
            )
            resp = JSONResponse(
                status_code=429,
                content={
                    "blocked": True,
                    "message": "Too many requests",
                    "retry_after_seconds": max(1, int(retry_after)),
                },
                headers={"Retry-After": str(max(1, int(retry_after)))},
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

    # --- DDoS checks (before body read) ---
    ddos = getattr(request.app.state, "ddos_protection", None)
    if ddos:
        content_length = request.headers.get("content-length")
        try:
            cl = int(content_length) if content_length else None
        except (ValueError, TypeError):
            cl = None

        allowed_size, reason = ddos.check_request_size(cl)
        if not allowed_size:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="ddos_size",
                content_length=cl if cl is not None else 0,
                max_bytes=gateway_config.DDOS_MAX_BODY_BYTES,
            )
            resp = JSONResponse(
                status_code=413,
                content={
                    "blocked": True,
                    "message": "Request too large",
                    "reason": reason,
                },
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

        is_blocked, ttl = await ddos.is_blocked(client_ip)
        if is_blocked:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="ddos_blocked",
                retry_after_seconds=max(1, int(ttl)),
                block_ttl_seconds=max(1, int(ttl)),
            )
            resp = JSONResponse(
                status_code=429,
                content={
                    "blocked": True,
                    "message": "Temporarily blocked due to traffic spike",
                    "retry_after_seconds": max(1, int(ttl)),
                },
                headers={"Retry-After": str(max(1, int(ttl)))},
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

        allowed_burst, triggered = await ddos.record_request_and_check_burst(client_ip)
        if not allowed_burst:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="ddos_burst",
                block_duration_seconds=gateway_config.DDOS_BLOCK_DURATION_SECONDS,
            )
            resp = JSONResponse(
                status_code=429,
                content={
                    "blocked": True,
                    "message": "Traffic spike detected; temporarily blocked",
                    "retry_after_seconds": gateway_config.DDOS_BLOCK_DURATION_SECONDS,
                },
                headers={"Retry-After": str(gateway_config.DDOS_BLOCK_DURATION_SECONDS)},
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

    # Read body once (needed for both WAF inspection and forwarding)
    body_bytes = await request.body()

    # --- WAF Inspection ---
    should_block = False
    waf_result: dict = {}

    if gateway_config.WAF_ENABLED:
        should_block, waf_result = await inspect_request(
            waf_service=request.app.state.waf_service,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            headers=dict(request.headers),
            body=body_bytes if body_bytes else None,
        )

    if should_block:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        waf_action = waf_result.get("action", "block")
        waf_event_type = "waf_challenge" if waf_action == "challenge" else "waf_block"
        _log_gateway_event(
            request,
            request_id=request_id,
            client_ip=client_ip,
            body_bytes=body_bytes,
            decision="block",
            anomaly_score=waf_result.get("anomaly_score"),
            attack_score=waf_result.get("attack_score"),
            waf_latency_ms=waf_result.get("waf_latency_ms"),
            total_latency_ms=elapsed_ms,
            blocked_by=waf_event_type,
        )
        resp = JSONResponse(
            status_code=403,
            content={
                "blocked": True,
                "message": "Request blocked by WAF" if waf_action == "block" else "Request challenged by WAF",
                "action": waf_action,
                "attack_score": waf_result.get("attack_score"),
                "anomaly_score": waf_result.get("anomaly_score", 0.0),
                "threshold": waf_result.get(
                    "threshold", gateway_config.WAF_THRESHOLD
                ),
            },
        )
        resp.headers["X-Request-ID"] = request_id
        return resp

    # --- Forward to upstream ---
    response = await forward_request(
        client=request.app.state.http_client,
        request=request,
        body_bytes=body_bytes if body_bytes else None,
    )

    # Determine decision for event log
    is_anomaly = waf_result.get("is_anomaly", False)
    if is_anomaly and gateway_config.WAF_MODE == "monitor":
        decision = "monitor_would_block"
    else:
        decision = "allow"

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _log_gateway_event(
        request,
        request_id=request_id,
        client_ip=client_ip,
        body_bytes=body_bytes,
        decision=decision,
        anomaly_score=waf_result.get("anomaly_score"),
        attack_score=waf_result.get("attack_score"),
        waf_latency_ms=waf_result.get("waf_latency_ms"),
        upstream_status=response.status_code,
        total_latency_ms=elapsed_ms,
    )

    # Add gateway observability headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Gateway-Time-Ms"] = f"{elapsed_ms:.1f}"
    if waf_result and not waf_result.get("skipped"):
        response.headers["X-WAF-Score"] = str(
            waf_result.get("attack_score", "")
        )
        response.headers["X-WAF-Mode"] = gateway_config.WAF_MODE

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.main:app",
        host=gateway_config.GATEWAY_HOST,
        port=gateway_config.GATEWAY_PORT,
        workers=1,
        reload=True,
    )
