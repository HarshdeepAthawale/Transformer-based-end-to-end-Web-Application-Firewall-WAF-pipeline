"""
HTTPS redirect and HSTS header helpers (Phase 1 — SSL/TLS).
Used when HTTPS_REWRITE_ENABLED is true to redirect HTTP→HTTPS and add HSTS.
"""

from fastapi import Request


def should_redirect_to_https(request: Request) -> bool:
    """Return True if request is HTTP and should be redirected to HTTPS."""
    return request.url.scheme.lower() == "http"


def build_hsts_header(max_age: int = 31536000, include_subdomains: bool = False, preload: bool = False) -> str:
    """Build Strict-Transport-Security header value."""
    parts = [f"max-age={max_age}"]
    if include_subdomains:
        parts.append("includeSubDomains")
    if preload:
        parts.append("preload")
    return "; ".join(parts)
