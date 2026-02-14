"""
Request Serializer

Converts an HTTPRequest into a compact string format for model input.
Output format matches what WAFClassifier._build_request_text() produces,
ensuring consistency between log-based and live-request paths.

Format:
    METHOD /path?param=val HTTP/1.1
    Header-Name: value
    Header-Name: value

    body content here
"""

import json
import logging
from typing import Optional

from backend.parsing.log_parser import HTTPRequest

logger = logging.getLogger(__name__)

# Headers to skip during serialization (internal/infrastructure headers)
_SKIP_HEADERS = frozenset({
    "host",
    "content-length",
    "connection",
    "accept-encoding",
    "transfer-encoding",
})


def serialize_request(
    request: HTTPRequest,
    include_headers: bool = True,
    include_body: bool = True,
    max_body_length: int = 2048,
) -> str:
    """Serialize an HTTPRequest to a compact string for model input.

    Args:
        request: The HTTPRequest to serialize.
        include_headers: Whether to include headers in output.
        include_body: Whether to include body in output.
        max_body_length: Maximum body length (truncated if exceeded).

    Returns:
        Serialized request string.
    """
    # Build request line
    full_path = request.path
    if request.query_params:
        query_str = "&".join(
            f"{k}={v}" for k, v in request.query_params.items()
        )
        if query_str:
            full_path = f"{request.path}?{query_str}"

    lines = [f"{request.method} {full_path} {request.protocol}"]

    # Add headers
    if include_headers and request.headers:
        for key, value in request.headers.items():
            if key.lower() not in _SKIP_HEADERS:
                lines.append(f"{key}: {value}")

    # Add body
    if include_body and request.body:
        lines.append("")  # blank line before body
        body = request.body
        if len(body) > max_body_length:
            body = body[:max_body_length]
        # Try to normalize JSON bodies
        body = _normalize_body(body)
        lines.append(body)

    return "\n".join(lines)


def _normalize_body(body: str) -> str:
    """Attempt to normalize body content (e.g., pretty JSON → compact)."""
    stripped = body.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, separators=(",", ":"))
        except (json.JSONDecodeError, ValueError):
            pass
    return body
