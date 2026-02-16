"""
Log Format Detection

Detects Apache Common, Apache Combined, and Nginx Combined log formats
from sample log lines using regex pattern matching.
"""

import re
from enum import Enum
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


class LogFormat(Enum):
    APACHE_COMMON = "apache_common"
    APACHE_COMBINED = "apache_combined"
    NGINX_COMBINED = "nginx_combined"
    UNKNOWN = "unknown"


# Patterns for each log format
# Apache Common: %h %l %u %t "%r" %>s %b
# Example: 127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326
_APACHE_COMMON_RE = re.compile(
    r'^(\S+)\s+'           # remote host
    r'(\S+)\s+'            # ident
    r'(\S+)\s+'            # user
    r'\[([^\]]+)\]\s+'     # timestamp
    r'"([^"]*?)"\s+'       # request line
    r'(\d{3})\s+'          # status code
    r'(\S+)'               # bytes
    r'\s*$'
)

# Apache Combined: Common + "%{Referer}i" "%{User-agent}i"
# Example: 127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326 "http://www.example.com/start.html" "Mozilla/4.08"
_APACHE_COMBINED_RE = re.compile(
    r'^(\S+)\s+'           # remote host
    r'(\S+)\s+'            # ident
    r'(\S+)\s+'            # user
    r'\[([^\]]+)\]\s+'     # timestamp
    r'"([^"]*?)"\s+'       # request line
    r'(\d{3})\s+'          # status code
    r'(\S+)\s+'            # bytes
    r'"([^"]*?)"\s+'       # referer
    r'"([^"]*?)"'          # user agent
    r'\s*$'
)

# Nginx Combined: same as Apache Combined but may include extra fields
# Nginx default: $remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent"
# Practically identical to Apache Combined, but we distinguish by checking
# for common Nginx patterns (e.g., request_time, upstream fields are absent in basic)
_NGINX_COMBINED_RE = re.compile(
    r'^(\S+)\s+'           # remote addr
    r'-\s+'                # Nginx always uses literal '-' for ident
    r'(\S+)\s+'            # remote user
    r'\[([^\]]+)\]\s+'     # time_local
    r'"([^"]*?)"\s+'       # request
    r'(\d{3})\s+'          # status
    r'(\d+)\s+'            # body_bytes_sent (Nginx uses numeric, not '-')
    r'"([^"]*?)"\s+'       # http_referer
    r'"([^"]*?)"'          # http_user_agent
    r'\s*$'
)


def detect_format(log_line: str) -> LogFormat:
    """Detect the log format of a single log line.

    Tests in order: Nginx Combined, Apache Combined, Apache Common.
    Returns LogFormat.UNKNOWN if no pattern matches.
    """
    line = log_line.strip()
    if not line:
        return LogFormat.UNKNOWN

    # Try Nginx Combined first (stricter: literal '-' for ident, numeric bytes)
    if _NGINX_COMBINED_RE.match(line):
        return LogFormat.NGINX_COMBINED

    # Try Apache Combined (has referer + user-agent)
    if _APACHE_COMBINED_RE.match(line):
        return LogFormat.APACHE_COMBINED

    # Try Apache Common (no referer/user-agent)
    if _APACHE_COMMON_RE.match(line):
        return LogFormat.APACHE_COMMON

    return LogFormat.UNKNOWN


def detect_from_file(
    log_path: str,
    sample_lines: int = 10,
    encoding: str = "utf-8",
) -> LogFormat:
    """Detect log format by sampling the first N non-empty lines from a file.

    Uses majority voting across sample lines. Supports plain text and .gz files.

    Args:
        log_path: Path to the log file.
        sample_lines: Number of lines to sample for detection.
        encoding: File encoding.

    Returns:
        The most common detected LogFormat across the sample.
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    lines = _read_sample_lines(path, sample_lines, encoding)
    if not lines:
        return LogFormat.UNKNOWN

    # Majority vote
    votes: dict[LogFormat, int] = {}
    for line in lines:
        fmt = detect_format(line)
        if fmt != LogFormat.UNKNOWN:
            votes[fmt] = votes.get(fmt, 0) + 1

    if not votes:
        return LogFormat.UNKNOWN

    winner = max(votes, key=lambda f: votes[f])
    logger.info(
        "Detected format %s from %d/%d lines in %s",
        winner.value, votes[winner], len(lines), log_path,
    )
    return winner


def _read_sample_lines(
    path: Path, n: int, encoding: str
) -> list[str]:
    """Read up to n non-empty lines from a file (supports .gz)."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    lines: list[str] = []
    try:
        with opener(path, "rt", encoding=encoding) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    lines.append(stripped)
                    if len(lines) >= n:
                        break
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
    return lines
