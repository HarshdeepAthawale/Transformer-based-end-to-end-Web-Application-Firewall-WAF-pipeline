"""
Request Fingerprinting for inference caching.

Creates cache-friendly fingerprints of HTTP requests to avoid redundant
ML inference on identical or near-identical requests.
"""

import hashlib
from typing import Optional
from urllib.parse import urlparse, parse_qs


def fingerprint_request(
    method: str,
    path: str,
    headers: Optional[dict] = None,
    body: Optional[str] = None,
) -> str:
    """
    Create a compact fingerprint of an HTTP request for caching.

    Normalization:
    - Uppercase method
    - Strip query param values (keep keys for structural fingerprint)
    - Sort and filter headers (skip volatile ones)
    - Truncate body to first 1024 bytes

    Args:
        method: HTTP method
        path: Request path (may include query string)
        headers: Request headers dict
        body: Request body string

    Returns:
        32-char hex digest (blake2b, 16 bytes)
    """
    parts = []

    # Method
    parts.append(method.upper())

    # Path: normalize and keep query param keys only
    parsed = urlparse(path)
    norm_path = parsed.path or "/"
    if parsed.query:
        param_keys = sorted(parse_qs(parsed.query, keep_blank_values=True).keys())
        norm_path += "?" + "&".join(param_keys)
    parts.append(norm_path)

    # Headers: skip volatile headers, sort remaining
    if headers:
        skip_headers = {
            "host", "content-length", "connection", "accept-encoding",
            "date", "x-request-id", "x-forwarded-for", "cookie",
            "authorization", "x-real-ip",
        }
        sorted_headers = sorted(
            (k.lower(), v)
            for k, v in headers.items()
            if k.lower() not in skip_headers
        )
        for k, v in sorted_headers:
            parts.append(f"{k}:{v}")

    # Body: truncate to 1024 bytes
    if body:
        parts.append(body[:1024])

    fingerprint_input = "|".join(parts).encode("utf-8", errors="replace")
    return hashlib.blake2b(fingerprint_input, digest_size=16).hexdigest()
