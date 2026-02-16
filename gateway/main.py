"""
WAF Gateway - FastAPI reverse proxy with ML-based threat detection.

Usage:
    UPSTREAM_URL=http://your-app:8080 python -m gateway.main
    UPSTREAM_URL=http://your-app:8080 uvicorn gateway.main:app --host 0.0.0.0 --port 8080
"""

import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from loguru import logger

from gateway.config import gateway_config
from gateway.proxy import forward_request
from gateway.waf_inspect import inspect_request

# Configure logger
logger.add(
    "logs/gateway.log",
    rotation="10 MB",
    retention="7 days",
    level=gateway_config.LOG_LEVEL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage gateway lifecycle: httpx client + WAF classifier."""
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

    # Initialize WAF classifier (reuse backend factory)
    app.state.waf_service = None
    if gateway_config.WAF_ENABLED:
        try:
            from backend.core.waf_factory import create_waf_service

            app.state.waf_service = create_waf_service()
            if app.state.waf_service:
                logger.info("WAF classifier loaded successfully")
            else:
                logger.warning(
                    "WAF classifier not available (model may be missing)"
                )
        except Exception as exc:
            logger.error(f"Failed to initialize WAF classifier: {exc}")

    yield

    # Shutdown
    logger.info("Shutting down WAF Gateway...")
    await app.state.http_client.aclose()
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
async def health():
    """Gateway health check endpoint."""
    return {
        "status": "healthy",
        "service": "waf-gateway",
        "waf_enabled": gateway_config.WAF_ENABLED,
        "waf_mode": gateway_config.WAF_MODE,
        "upstream": gateway_config.UPSTREAM_URL,
    }


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def gateway_proxy(request: Request, path: str):
    """
    Catch-all route: inspect with WAF, then forward to upstream.
    """
    # Skip WAF for gateway internal endpoints
    if request.url.path.startswith("/gateway/"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    start_time = time.perf_counter()

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
        return JSONResponse(
            status_code=403,
            content={
                "blocked": True,
                "message": "Request blocked by WAF",
                "anomaly_score": waf_result.get("anomaly_score", 0.0),
                "threshold": waf_result.get(
                    "threshold", gateway_config.WAF_THRESHOLD
                ),
            },
        )

    # --- Forward to upstream ---
    response = await forward_request(
        client=request.app.state.http_client,
        request=request,
        body_bytes=body_bytes if body_bytes else None,
    )

    # Add gateway observability headers
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Gateway-Time-Ms"] = f"{elapsed_ms:.1f}"
    if waf_result and not waf_result.get("skipped"):
        response.headers["X-WAF-Score"] = str(
            waf_result.get("anomaly_score", "")
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
