"""
Request Normalizer

Replaces dynamic values in HTTP requests with stable placeholders
so the model sees consistent patterns regardless of specific IDs,
timestamps, IPs, etc.
"""

import re
import logging
from typing import Optional

from backend.parsing.log_parser import HTTPRequest

logger = logging.getLogger(__name__)


# --- Regex patterns for dynamic values ---

# UUIDs: a1b2c3d4-e5f6-7890-abcd-ef1234567890
_UUID_RE = re.compile(
    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
)

# ISO 8601 timestamps: 2025-02-15T10:30:00Z, 2025-02-15T10:30:00+00:00
_ISO_TIMESTAMP_RE = re.compile(
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?'
)

# Unix timestamps (10+ digits, reasonable range)
_UNIX_TIMESTAMP_RE = re.compile(
    r'\b1[0-9]{9,12}\b'
)

# IPv4 addresses
_IPV4_RE = re.compile(
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
)

# IPv6 addresses (simplified)
_IPV6_RE = re.compile(
    r'\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b'
)

# JWT tokens: eyJ... (3 base64url segments separated by dots)
_JWT_RE = re.compile(
    r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'
)

# Session IDs: common patterns like sess=, session=, sid=, PHPSESSID=, JSESSIONID=
_SESSION_RE = re.compile(
    r'(?:sess(?:ion)?|sid|PHPSESSID|JSESSIONID|connect\.sid|_session_id)'
    r'[=:]\s*[A-Za-z0-9._-]{16,}',
    re.IGNORECASE,
)

# Hex tokens (32+ chars, likely session/API tokens)
_HEX_TOKEN_RE = re.compile(
    r'\b[0-9a-fA-F]{32,}\b'
)

# Numeric IDs in URL paths: /api/users/12345 → /api/users/{ID}
_PATH_NUMERIC_ID_RE = re.compile(
    r'(?<=/)\d{2,}(?=/|$|\?)'
)

# Numeric IDs in query values: id=12345
_QUERY_NUMERIC_ID_RE = re.compile(
    r'(?<=\=)\d{2,}(?=&|$)'
)

# Base64-encoded blobs (20+ chars, likely tokens)
_BASE64_BLOB_RE = re.compile(
    r'\b[A-Za-z0-9+/]{20,}={0,2}\b'
)


def normalize_request(request: HTTPRequest) -> HTTPRequest:
    """Normalize an HTTPRequest by replacing dynamic values with placeholders.

    Returns a new HTTPRequest with normalized fields. The original is not modified.
    """
    return HTTPRequest(
        method=request.method,
        path=_normalize_path(request.path),
        query_params=_normalize_query_params(request.query_params),
        headers=_normalize_headers(request.headers),
        body=_normalize_text(request.body) if request.body else request.body,
        remote_addr="{IP}" if request.remote_addr else None,
        timestamp="{TIMESTAMP}" if request.timestamp else None,
        user_agent=_normalize_text(request.user_agent) if request.user_agent else request.user_agent,
        referer=_normalize_text(request.referer) if request.referer else request.referer,
        status_code=request.status_code,
        response_size=request.response_size,
        protocol=request.protocol,
    )


def _normalize_path(path: str) -> str:
    """Normalize a URL path, replacing numeric IDs and UUIDs."""
    result = _UUID_RE.sub("{UUID}", path)
    result = _PATH_NUMERIC_ID_RE.sub("{ID}", result)
    return result


# Keys that indicate session/token values
_SESSION_PARAM_KEYS = frozenset({
    "phpsessid", "jsessionid", "sid", "session", "session_id",
    "sessionid", "connect.sid", "_session_id", "csrf_token",
    "csrftoken", "_csrf", "token", "auth_token", "api_key",
    "apikey", "access_token",
})


def _normalize_query_params(params: dict) -> dict:
    """Normalize query parameter values."""
    normalized = {}
    for key, value in params.items():
        if key.lower() in _SESSION_PARAM_KEYS:
            normalized[key] = "{SESSION_ID}"
        else:
            normalized[key] = _normalize_value(value)
    return normalized


def _normalize_headers(headers: dict) -> dict:
    """Normalize header values, preserving keys."""
    normalized = {}
    for key, value in headers.items():
        normalized[key] = _normalize_value(value)
    return normalized


def _normalize_value(value: str) -> str:
    """Normalize a single value string."""
    if not value or value == "-":
        return value
    return _apply_replacements(value)


def _normalize_text(text: Optional[str]) -> Optional[str]:
    """Normalize a free-text field (body, user-agent, etc.)."""
    if not text:
        return text
    return _apply_replacements(text)


def _apply_replacements(text: str) -> str:
    """Apply all normalization replacements to a text string.

    Order matters: more specific patterns first to avoid partial matches.
    """
    # JWT first (contains dots that could be confused with IPs)
    result = _JWT_RE.sub("{JWT}", text)

    # UUIDs before hex tokens (UUIDs are a subset of hex)
    result = _UUID_RE.sub("{UUID}", result)

    # Session IDs before generic hex tokens
    result = _SESSION_RE.sub("{SESSION_ID}", result)

    # ISO timestamps before generic numbers
    result = _ISO_TIMESTAMP_RE.sub("{TIMESTAMP}", result)

    # Unix timestamps
    result = _UNIX_TIMESTAMP_RE.sub("{TIMESTAMP}", result)

    # IPv4 and IPv6
    result = _IPV4_RE.sub("{IP}", result)
    result = _IPV6_RE.sub("{IP}", result)

    # Hex tokens (after UUIDs and session IDs)
    result = _HEX_TOKEN_RE.sub("{TOKEN}", result)

    return result
