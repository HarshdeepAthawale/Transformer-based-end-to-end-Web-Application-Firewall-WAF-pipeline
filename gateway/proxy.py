"""
Reverse proxy using httpx.AsyncClient.
"""

import time
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import Request
from fastapi.responses import Response
from loguru import logger

from gateway.config import gateway_config

# Headers that MUST NOT be forwarded between client and upstream.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
        "host",
    }
)


def _filter_headers(
    headers: dict, extra_strip: Optional[frozenset] = None
) -> dict:
    """Remove hop-by-hop and internal headers."""
    strip = HOP_BY_HOP_HEADERS | (extra_strip or frozenset())
    return {k: v for k, v in headers.items() if k.lower() not in strip}


async def forward_request(
    client: httpx.AsyncClient,
    request: Request,
    body_bytes: Optional[bytes] = None,
) -> Response:
    """
    Forward an HTTP request to the configured upstream and return the response.

    Args:
        client: Shared httpx.AsyncClient (managed by app lifespan).
        request: The incoming Starlette Request.
        body_bytes: Pre-read body bytes (already consumed for WAF inspection).

    Returns:
        A Response mirroring the upstream response.
    """
    upstream = gateway_config.UPSTREAM_URL.rstrip("/")
    path = request.url.path
    query = request.url.query
    target_url = f"{upstream}{path}"
    if query:
        target_url = f"{target_url}?{query}"

    # Build forwarded headers
    forwarded_headers = _filter_headers(dict(request.headers))

    # Set correct Host for upstream
    parsed = urlparse(upstream)
    forwarded_headers["host"] = parsed.netloc

    # Add X-Forwarded-* headers for upstream visibility
    client_host = request.client.host if request.client else "unknown"
    forwarded_headers.setdefault("x-forwarded-for", client_host)
    forwarded_headers.setdefault("x-forwarded-proto", request.url.scheme)
    forwarded_headers.setdefault(
        "x-forwarded-host", request.headers.get("host", "")
    )

    start = time.perf_counter()

    try:
        upstream_response = await client.request(
            method=request.method,
            url=target_url,
            headers=forwarded_headers,
            content=body_bytes,
            timeout=gateway_config.PROXY_TIMEOUT,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            f"Upstream {request.method} {path} -> "
            f"{upstream_response.status_code} ({elapsed_ms:.1f}ms)"
        )

        # Filter response headers — strip hop-by-hop and content-encoding/length
        # since we're returning the decoded body directly
        response_headers = _filter_headers(
            dict(upstream_response.headers),
            extra_strip=frozenset({"content-encoding", "content-length"}),
        )

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    except httpx.TimeoutException:
        logger.error(f"Upstream timeout: {request.method} {target_url}")
        return Response(content="Gateway Timeout", status_code=504)

    except httpx.ConnectError as exc:
        logger.error(f"Upstream connect error: {exc}")
        return Response(content="Bad Gateway", status_code=502)

    except Exception as exc:
        logger.error(f"Proxy error: {exc}", exc_info=True)
        return Response(content="Bad Gateway", status_code=502)
