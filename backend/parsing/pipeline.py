"""
Parsing Pipeline

Chains parse → normalize → serialize into a single processing step.
Provides two entry points:
  - process(log_line)     — for log file lines
  - process_request(req)  — for live HTTPRequest objects
"""

import logging
from typing import Optional

from backend.ingestion.format_detector import LogFormat
from backend.parsing.log_parser import HTTPRequest, parse_log_line, parse_request_dict
from backend.parsing.normalizer import normalize_request
from backend.parsing.serializer import serialize_request

logger = logging.getLogger(__name__)


class ParsingPipeline:
    """Chains parsing, normalization, and serialization.

    Args:
        log_format: Known log format (auto-detected if None).
        include_headers: Include headers in serialized output.
        include_body: Include body in serialized output.
        max_body_length: Max body length before truncation.
    """

    def __init__(
        self,
        log_format: Optional[LogFormat] = None,
        include_headers: bool = True,
        include_body: bool = True,
        max_body_length: int = 2048,
    ):
        self._log_format = log_format
        self._include_headers = include_headers
        self._include_body = include_body
        self._max_body_length = max_body_length
        self._processed = 0
        self._failed = 0

    @property
    def processed(self) -> int:
        return self._processed

    @property
    def failed(self) -> int:
        return self._failed

    def process(self, log_line: str) -> Optional[str]:
        """Parse a log line → normalize → serialize.

        Args:
            log_line: Raw log line (Apache/Nginx format).

        Returns:
            Serialized normalized request string, or None if parsing fails.
        """
        try:
            request = parse_log_line(log_line, self._log_format)
            if request is None:
                self._failed += 1
                return None

            normalized = normalize_request(request)
            serialized = serialize_request(
                normalized,
                include_headers=self._include_headers,
                include_body=self._include_body,
                max_body_length=self._max_body_length,
            )
            self._processed += 1
            return serialized

        except Exception as e:
            logger.warning("Pipeline error for line: %s — %s", log_line[:80], e)
            self._failed += 1
            return None

    def process_request(self, request: HTTPRequest) -> str:
        """Normalize and serialize a live HTTPRequest.

        For live requests that are already parsed (not from log lines).

        Args:
            request: Parsed HTTPRequest object.

        Returns:
            Serialized normalized request string.
        """
        normalized = normalize_request(request)
        serialized = serialize_request(
            normalized,
            include_headers=self._include_headers,
            include_body=self._include_body,
            max_body_length=self._max_body_length,
        )
        self._processed += 1
        return serialized

    def process_dict(self, request_data: dict) -> str:
        """Normalize and serialize a request from a dict.

        For live requests coming as dicts (e.g., from WAF service API).

        Args:
            request_data: Dict with method, path, query_params, headers, body.

        Returns:
            Serialized normalized request string.
        """
        request = parse_request_dict(request_data)
        return self.process_request(request)

    def stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "processed": self._processed,
            "failed": self._failed,
            "log_format": self._log_format.value if self._log_format else "auto",
        }
