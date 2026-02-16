"""
Log Parser

Parses Apache Common, Apache Combined, and Nginx Combined log lines
into structured HTTPRequest objects.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import parse_qs

from backend.ingestion.format_detector import LogFormat, detect_format

logger = logging.getLogger(__name__)


@dataclass
class HTTPRequest:
    """Structured representation of an HTTP request parsed from a log line."""
    method: str = "GET"
    path: str = "/"
    query_params: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    body: Optional[str] = None
    remote_addr: Optional[str] = None
    timestamp: Optional[str] = None
    user_agent: Optional[str] = None
    referer: Optional[str] = None
    status_code: Optional[int] = None
    response_size: Optional[int] = None
    protocol: str = "HTTP/1.1"


# Regex for the request line: "METHOD /path HTTP/1.1"
_REQUEST_LINE_RE = re.compile(
    r'^(\S+)\s+(\S+?)(?:\s+(HTTP/\S+))?$'
)

# Apache Common: host ident user [time] "request" status bytes
_COMMON_RE = re.compile(
    r'^(\S+)\s+'          # remote host
    r'(\S+)\s+'           # ident
    r'(\S+)\s+'           # user
    r'\[([^\]]+)\]\s+'    # timestamp
    r'"([^"]*?)"\s+'      # request line
    r'(\d{3})\s+'         # status code
    r'(\S+)'              # bytes
    r'\s*$'
)

# Combined (Apache/Nginx): host ident user [time] "request" status bytes "referer" "ua"
_COMBINED_RE = re.compile(
    r'^(\S+)\s+'          # remote host
    r'(\S+)\s+'           # ident
    r'(\S+)\s+'           # user
    r'\[([^\]]+)\]\s+'    # timestamp
    r'"([^"]*?)"\s+'      # request line
    r'(\d{3})\s+'         # status code
    r'(\S+)\s+'           # bytes
    r'"([^"]*?)"\s+'      # referer
    r'"([^"]*?)"'         # user agent
    r'\s*$'
)


def parse_log_line(
    log_line: str,
    log_format: Optional[LogFormat] = None,
) -> Optional[HTTPRequest]:
    """Parse a single log line into an HTTPRequest.

    Args:
        log_line: Raw log line string.
        log_format: Known format. If None, auto-detected.

    Returns:
        HTTPRequest if parsing succeeds, None otherwise.
    """
    line = log_line.strip()
    if not line:
        return None

    if log_format is None:
        log_format = detect_format(line)

    if log_format in (
        LogFormat.APACHE_COMBINED,
        LogFormat.NGINX_COMBINED,
    ):
        return _parse_combined(line)
    elif log_format == LogFormat.APACHE_COMMON:
        return _parse_common(line)
    else:
        # Try combined first, then common
        result = _parse_combined(line)
        if result:
            return result
        return _parse_common(line)


def _parse_combined(line: str) -> Optional[HTTPRequest]:
    """Parse a Combined format log line."""
    match = _COMBINED_RE.match(line)
    if not match:
        return None

    host, _ident, _user, timestamp, request_line, status, size, referer, ua = match.groups()

    request = _parse_request_line(request_line)
    if request is None:
        return None

    request.remote_addr = host
    request.timestamp = timestamp
    request.status_code = int(status)
    request.response_size = _parse_size(size)

    if referer and referer != "-":
        request.referer = referer
        request.headers["Referer"] = referer

    if ua and ua != "-":
        request.user_agent = ua
        request.headers["User-Agent"] = ua

    return request


def _parse_common(line: str) -> Optional[HTTPRequest]:
    """Parse an Apache Common format log line."""
    match = _COMMON_RE.match(line)
    if not match:
        return None

    host, _ident, _user, timestamp, request_line, status, size = match.groups()

    request = _parse_request_line(request_line)
    if request is None:
        return None

    request.remote_addr = host
    request.timestamp = timestamp
    request.status_code = int(status)
    request.response_size = _parse_size(size)

    return request


def _parse_request_line(request_line: str) -> Optional[HTTPRequest]:
    """Parse 'GET /path?q=1 HTTP/1.1' into an HTTPRequest."""
    if not request_line:
        return None

    match = _REQUEST_LINE_RE.match(request_line)
    if not match:
        return None

    method, uri, protocol = match.groups()

    # Split path and query string
    if "?" in uri:
        path, query_string = uri.split("?", 1)
        query_params = _parse_query_string(query_string)
    else:
        path = uri
        query_params = {}

    return HTTPRequest(
        method=method.upper(),
        path=path,
        query_params=query_params,
        protocol=protocol or "HTTP/1.1",
    )


def _parse_query_string(query_string: str) -> dict:
    """Parse a query string into a flat dict.

    For repeated keys, keeps the last value.
    """
    parsed = parse_qs(query_string, keep_blank_values=True)
    # Flatten: parse_qs returns lists, we take last value
    return {k: v[-1] for k, v in parsed.items()}


def _parse_size(size_str: str) -> Optional[int]:
    """Parse response size, handling '-' as None."""
    if size_str == "-":
        return None
    try:
        return int(size_str)
    except ValueError:
        return None


def parse_request_dict(request_data: dict) -> HTTPRequest:
    """Build an HTTPRequest from a dict (for live request processing).

    Args:
        request_data: Dict with keys: method, path, query_params, headers, body.

    Returns:
        HTTPRequest populated from the dict.
    """
    headers = request_data.get("headers", {})
    return HTTPRequest(
        method=request_data.get("method", "GET").upper(),
        path=request_data.get("path", "/"),
        query_params=request_data.get("query_params", {}),
        headers=dict(headers),
        body=request_data.get("body"),
        remote_addr=request_data.get("remote_addr"),
        user_agent=headers.get("User-Agent"),
        referer=headers.get("Referer"),
    )
