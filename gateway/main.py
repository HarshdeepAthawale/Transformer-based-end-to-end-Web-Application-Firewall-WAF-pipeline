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
from fastapi.responses import JSONResponse, Response
from loguru import logger

from gateway.config import gateway_config
from gateway.mongo import MongoEventStore
from gateway.proxy import forward_request
from gateway.waf_inspect import inspect_request
from gateway.rate_limit import create_rate_limiter
from gateway.rate_limit_config import get_rate_limit_configs, get_limit_for_path
from gateway.ddos_protection import create_ddos_protection
from gateway.blacklist import create_blacklist_checker
from gateway.events import report_event, start_event_batcher, stop_event_batcher
from gateway.bot_score import get_bot_score
from gateway import managed_rules as managed_rules_module
from gateway.upload_scan import is_upload_request, process_upload_scan
from gateway.firewall_ai import evaluate_request as firewall_ai_evaluate
from gateway.credential_leak import process_credential_leak
from gateway.edge_cache import create_edge_cache, create_purge_subscriber

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

    # Edge cache (Phase 1 — CDN layer)
    app.state.edge_cache = create_edge_cache()
    if app.state.edge_cache:
        await app.state.edge_cache.store.connect()
        purge_sub = create_purge_subscriber(app.state.edge_cache)
        await purge_sub.start()
        app.state.cache_purge_subscriber = purge_sub
        logger.info("Edge cache and cache purge subscriber enabled")
    else:
        app.state.cache_purge_subscriber = None

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

    # Phase pipeline (FL2-inspired modular architecture)
    if gateway_config.GATEWAY_USE_PIPELINE:
        try:
            from gateway.pipeline import PipelineOrchestrator, PipelineMetrics
            from gateway.pipeline.phases import (
                IPBlacklistPhase, EdgeCachePhase, RateLimitPhase,
                DDoSProtectionPhase, BotDetectionPhase, UploadScanPhase,
                FirewallAIPhase, CredentialLeakPhase, ManagedRulesPhase,
                WAFMLPhase,
            )

            pipeline_metrics = PipelineMetrics()
            phases = [
                IPBlacklistPhase(getattr(app.state, "blacklist_checker", None)),
                EdgeCachePhase(getattr(app.state, "edge_cache", None)),
                RateLimitPhase(getattr(app.state, "rate_limiter", None)),
                DDoSProtectionPhase(getattr(app.state, "ddos_protection", None)),
                BotDetectionPhase(),
                UploadScanPhase(),
                FirewallAIPhase(),
                CredentialLeakPhase(),
                ManagedRulesPhase(),
                WAFMLPhase(app.state.waf_service),
            ]
            app.state.pipeline = PipelineOrchestrator(phases, pipeline_metrics)
            app.state.pipeline_metrics = pipeline_metrics
            logger.info(
                f"Phase pipeline enabled with {len(phases)} phases"
            )
        except Exception as exc:
            logger.error(f"Failed to initialize phase pipeline: {exc}")
            app.state.pipeline = None
            app.state.pipeline_metrics = None
    else:
        app.state.pipeline = None
        app.state.pipeline_metrics = None

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
    if getattr(app.state, "cache_purge_subscriber", None):
        await app.state.cache_purge_subscriber.stop()
    if getattr(app.state, "edge_cache", None):
        await app.state.edge_cache.close()
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
        "bot_enabled": gateway_config.BOT_ENABLED,
        "managed_rules_enabled": gateway_config.MANAGED_RULES_ENABLED,
        "upload_scan_enabled": gateway_config.UPLOAD_SCAN_ENABLED,
        "firewall_ai_enabled": gateway_config.FIREWALL_AI_ENABLED,
        "credential_leak_enabled": gateway_config.CREDENTIAL_LEAK_ENABLED,
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
    bot_score: int | None = None,
    bot_action: str | None = None,
) -> None:
    """Helper to log an event to both MongoDB and the legacy event reporter."""
    event_store: MongoEventStore = request.app.state.event_store
    headers = dict(request.headers)
    # Redact sensitive headers to prevent credential leakage in logs
    _SENSITIVE_HEADERS = {"authorization", "cookie", "x-api-key", "proxy-authorization", "set-cookie"}
    headers = {k: ("[REDACTED]" if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()}
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

    # Build payload for backend ingest
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
    if bot_score is not None:
        event_payload["bot_score"] = bot_score
    if bot_action is not None:
        event_payload["bot_action"] = bot_action
    report_event(event_payload)


async def _pipeline_proxy(request: Request, pipeline) -> Response:
    """FL2-inspired modular phase pipeline path."""
    from gateway.pipeline.context import PhaseContext

    ctx = PhaseContext(
        client_ip=_get_client_ip(request),
        method=request.method,
        path=request.url.path,
        query_string=str(request.url.query) if request.url.query else "",
        headers=dict(request.headers),
        content_type=request.headers.get("content-type", ""),
        org_id=request.headers.get("x-waf-zone-id", "default"),
    )

    ctx = await pipeline.run(request, ctx)

    # Short circuit: return block/challenge response
    if ctx.final_verdict in ("block", "challenge"):
        elapsed_ms = (time.perf_counter() - ctx.start_time) * 1000
        waf_data = ctx.phase_results.get("waf_ml")
        _log_gateway_event(
            request,
            request_id=ctx.request_id,
            client_ip=ctx.client_ip,
            decision="block",
            anomaly_score=waf_data.data.get("anomaly_score") if waf_data else None,
            attack_score=waf_data.data.get("attack_score") if waf_data else None,
            total_latency_ms=elapsed_ms,
            blocked_by=ctx.blocking_phase,
        )
        return pipeline.build_block_response(ctx)

    # Cache hit
    if ctx.final_verdict == "cache_hit":
        cache_result = ctx.phase_results.get("edge_cache")
        if cache_result and cache_result.data.get("body") is not None:
            resp = Response(
                content=cache_result.data["body"],
                status_code=cache_result.data.get("status_code", 200),
                headers=cache_result.data.get("headers", {}),
            )
            resp.headers["X-Cache"] = cache_result.data.get("cache_status", "HIT")
            resp.headers["X-Request-ID"] = ctx.request_id
            return resp

    # Forward to upstream
    response = await forward_request(
        client=request.app.state.http_client,
        request=request,
        body_bytes=ctx.body_bytes if ctx.body_bytes else None,
    )

    # Log event
    elapsed_ms = (time.perf_counter() - ctx.start_time) * 1000
    waf_data = ctx.phase_results.get("waf_ml")
    _log_gateway_event(
        request,
        request_id=ctx.request_id,
        client_ip=ctx.client_ip,
        decision=ctx.final_verdict,
        anomaly_score=waf_data.data.get("anomaly_score") if waf_data else None,
        attack_score=waf_data.data.get("attack_score") if waf_data else None,
        waf_latency_ms=waf_data.data.get("waf_latency_ms") if waf_data else None,
        upstream_status=response.status_code,
        total_latency_ms=elapsed_ms,
    )

    # Add observability headers
    pipeline.add_observability_headers(response, ctx)
    if waf_data and waf_data.data.get("attack_score") is not None:
        response.headers["X-WAF-Mode"] = gateway_config.WAF_MODE

    # Store in edge cache on miss
    edge_cache = getattr(request.app.state, "edge_cache", None)
    if ctx.cache_ctx and edge_cache and request.method in ("GET", "HEAD"):
        try:
            resp_body = getattr(response, "body", None) or getattr(response, "_content", None) or b""
            if isinstance(resp_body, bytes) and edge_cache.is_cacheable_response(
                response.status_code, dict(response.headers)
            ):
                await edge_cache.store_response(
                    ctx.cache_ctx[0],
                    response.status_code,
                    resp_body,
                    dict(response.headers),
                    request.url.path,
                    request.method,
                    ctx.cache_ctx[1],
                )
                response.headers["X-Cache"] = "MISS"
        except Exception as e:
            logger.debug(f"Edge cache store failed: {e}")

    return response


@app.get("/gateway/phase-metrics")
async def phase_metrics(request: Request):
    """Per-phase pipeline metrics for dashboard consumption."""
    metrics = getattr(request.app.state, "pipeline_metrics", None)
    if metrics is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Pipeline not enabled. Set GATEWAY_USE_PIPELINE=true"},
        )
    return metrics.snapshot()


@app.get("/gateway/phase-metrics/prometheus")
async def phase_metrics_prometheus(request: Request):
    """Per-phase metrics in Prometheus text exposition format."""
    from fastapi.responses import PlainTextResponse

    metrics = getattr(request.app.state, "pipeline_metrics", None)
    if metrics is None:
        return PlainTextResponse(content="# Pipeline not enabled\n")
    return PlainTextResponse(
        content=metrics.prometheus_lines(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def gateway_proxy(request: Request, path: str):
    """
    Catch-all route: rate limit -> DDoS check -> WAF inspect -> forward to upstream.
    When GATEWAY_USE_PIPELINE=true, uses FL2-inspired modular phase pipeline.
    """
    # Skip WAF for gateway internal endpoints
    if request.url.path.startswith("/gateway/"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    # FL2-inspired pipeline path (feature-flagged)
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        return await _pipeline_proxy(request, pipeline)

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

    # --- Edge cache lookup (GET/HEAD only; skip body and heavy checks for cache hits) ---
    edge_cache = getattr(request.app.state, "edge_cache", None)
    cache_ctx = None  # (cache_key, full_url) to store after forward on miss
    if edge_cache and request.method in ("GET", "HEAD"):
        full_url = request.url.path + ("?" + request.url.query if request.url.query else "")
        if edge_cache.is_cacheable_request(request.method, dict(request.headers)):
            entry, cache_key, hit_status = await edge_cache.lookup(
                request.method, full_url, dict(request.headers)
            )
            if hit_status in ("HIT", "STALE") and entry:
                resp = Response(
                    content=entry.body,
                    status_code=entry.status_code,
                    headers=dict(entry.headers),
                )
                resp.headers["X-Cache"] = hit_status
                resp.headers["X-Request-ID"] = request_id
                return resp
            if hit_status == "REVALIDATED" and entry:
                resp = Response(status_code=304)
                resp.headers["X-Cache"] = "REVALIDATED"
                resp.headers["X-Request-ID"] = request_id
                return resp
            if hit_status == "MISS":
                cache_ctx = (cache_key, full_url)

    # --- Rate limit check (before body read) ---
    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter:
        # Look up per-org/per-path rate limit config from backend DB
        zone_id = request.headers.get("x-waf-zone-id", "default")
        rl_configs = await get_rate_limit_configs(zone_id)
        matched_config = get_limit_for_path(rl_configs, str(request.url.path))
        max_req_override = matched_config["requests_per_minute"] if matched_config else None
        allowed, retry_after = await rate_limiter.is_allowed(client_ip, max_requests_override=max_req_override)
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

    # --- Bot scoring (before body read) ---
    if gateway_config.BOT_ENABLED:
        user_agent = request.headers.get("user-agent", "")
        bot_result = await get_bot_score(user_agent, client_ip, dict(request.headers))
        if bot_result is None:
            if gateway_config.BOT_FAIL_OPEN:
                pass
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                _log_gateway_event(
                    request,
                    request_id=request_id,
                    client_ip=client_ip,
                    decision="block",
                    total_latency_ms=elapsed_ms,
                    blocked_by="bot_unavailable",
                )
                resp = JSONResponse(
                    status_code=503,
                    content={"blocked": True, "message": "Bot service unavailable"},
                )
                resp.headers["X-Request-ID"] = request_id
                return resp
        elif bot_result.get("action") == "block":
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="bot_block",
                bot_score=bot_result.get("bot_score"),
                bot_action="block",
            )
            resp = JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "message": "Request blocked by bot management",
                    "bot_score": bot_result.get("bot_score"),
                },
            )
            resp.headers["X-Request-ID"] = request_id
            return resp
        elif bot_result.get("action") == "challenge":
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            retry_after = gateway_config.BOT_CHALLENGE_RETRY_AFTER
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="bot_challenge",
                retry_after_seconds=retry_after,
                bot_score=bot_result.get("bot_score"),
                bot_action="challenge",
            )
            resp = JSONResponse(
                status_code=429,
                content={
                    "blocked": True,
                    "message": "Please complete the challenge",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

    # Read body once (needed for both WAF inspection and forwarding)
    body_bytes = await request.body()

    # --- Upload scan (multipart or path prefix; backend performs scan) ---
    content_type = request.headers.get("content-type") or ""
    if is_upload_request(content_type, request.url.path) and body_bytes:
        should_block, scan_result, err_msg = await process_upload_scan(
            body_bytes,
            content_type,
            request.url.path,
            client_ip,
            request.method,
        )
        if should_block:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="upload_scan_infected",
            )
            resp = JSONResponse(
                status_code=413 if err_msg == "File too large" else 403,
                content={
                    "blocked": True,
                    "message": err_msg or "Malicious file detected",
                },
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

    # --- Firewall for AI (LLM endpoint protection) ---
    if gateway_config.FIREWALL_AI_ENABLED:
        should_block_fa, event_type_fa, pattern_fa = await firewall_ai_evaluate(
            path=request.url.path,
            method=request.method,
            body=body_bytes,
            headers=dict(request.headers),
            client_ip=client_ip,
        )
        if should_block_fa and event_type_fa:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by=event_type_fa,
            )
            reason = "prompt_injection" if "prompt" in event_type_fa else "pii" if "pii" in event_type_fa else "abuse_rate"
            report_event({
                "event_type": event_type_fa,
                "ip": client_ip,
                "method": request.method,
                "path": request.url.path,
                "firewall_ai_reason": reason,
                "firewall_ai_pattern": pattern_fa,
                "firewall_ai_action": "block",
            })
            resp = JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "message": "Request blocked by Firewall for AI",
                    "reason": reason,
                },
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

    # --- Credential leak (login paths: check password against HIBP via backend) ---
    if gateway_config.CREDENTIAL_LEAK_ENABLED and body_bytes:
        should_block_cl, event_type_cl = await process_credential_leak(
            body_bytes,
            request.url.path,
            client_ip,
            request.method,
        )
        if should_block_cl:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _log_gateway_event(
                request,
                request_id=request_id,
                client_ip=client_ip,
                decision="block",
                total_latency_ms=elapsed_ms,
                blocked_by="credential_leak_block",
            )
            resp = JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "message": "Password has been found in a data breach; please choose a different password",
                },
            )
            resp.headers["X-Request-ID"] = request_id
            return resp

    # --- Managed rules (enabled packs from backend) ---
    if gateway_config.MANAGED_RULES_ENABLED:
        try:
            body_str = (body_bytes or b"").decode("utf-8", errors="replace")
            match = managed_rules_module.evaluate(
                method=request.method,
                path=request.url.path,
                headers=dict(request.headers),
                query_string=request.url.query or "",
                body=body_str,
            )
            if match:
                action = (match.get("action") or "block").lower()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                event_payload = {
                    "event_type": "managed_rule_block" if action == "block" else "managed_rule_log",
                    "request_id": request_id,
                    "ip": client_ip,
                    "method": request.method,
                    "path": request.url.path,
                    "decision": "block" if action == "block" else "allow",
                    "total_latency_ms": elapsed_ms,
                    "rule_id": match.get("rule_id"),
                    "pack_id": match.get("pack_id"),
                }
                report_event(event_payload)
                if action == "block":
                    resp = JSONResponse(
                        status_code=403,
                        content={
                            "blocked": True,
                            "message": "Request blocked by managed rule",
                            "rule_id": match.get("rule_id"),
                            "rule_name": match.get("rule_name"),
                            "pack_id": match.get("pack_id"),
                        },
                    )
                    resp.headers["X-Request-ID"] = request_id
                    return resp
        except Exception as e:
            if not gateway_config.MANAGED_RULES_FAIL_OPEN:
                logger.warning(f"Managed rules evaluation failed: {e}")
                resp = JSONResponse(
                    status_code=503,
                    content={"blocked": True, "message": "Managed rules check failed"},
                )
                resp.headers["X-Request-ID"] = request_id
                return resp
            logger.debug(f"Managed rules evaluation error (fail-open): {e}")

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

    # Store in edge cache on miss (GET/HEAD, cacheable response)
    if cache_ctx and edge_cache and request.method in ("GET", "HEAD"):
        try:
            body_bytes = getattr(response, "body", None) or getattr(response, "_content", None) or b""
            if isinstance(body_bytes, bytes) and edge_cache.is_cacheable_response(
                response.status_code, dict(response.headers)
            ):
                await edge_cache.store_response(
                    cache_ctx[0],
                    response.status_code,
                    body_bytes,
                    dict(response.headers),
                    request.url.path,
                    request.method,
                    cache_ctx[1],
                )
                response.headers["X-Cache"] = "MISS"
        except Exception as e:
            logger.debug(f"Edge cache store failed: {e}")

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
