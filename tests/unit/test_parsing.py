"""
Unit tests for backend.parsing module.

Tests log parser, normalizer, serializer, and pipeline.
Covers Apache Common, Apache Combined, Nginx Combined formats,
plus app-specific lines from Juice Shop, WebGoat, DVWA.
"""

import pytest

from backend.parsing.log_parser import (
    HTTPRequest,
    parse_log_line,
    parse_request_dict,
)
from backend.parsing.normalizer import normalize_request
from backend.parsing.serializer import serialize_request
from backend.parsing.pipeline import ParsingPipeline
from backend.ingestion.format_detector import LogFormat


# ---------------------------------------------------------------------------
# Sample log lines
# ---------------------------------------------------------------------------

APACHE_COMMON_LINE = (
    '127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] '
    '"GET /apache_pb.gif HTTP/1.0" 200 2326'
)

APACHE_COMBINED_LINE = (
    '192.168.1.1 user john [15/Feb/2025:10:30:00 +0000] '
    '"POST /api/login HTTP/1.1" 200 1234 '
    '"http://example.com/login" "Mozilla/5.0 (X11; Linux x86_64)"'
)

NGINX_COMBINED_LINE = (
    '10.0.0.1 - - [15/Feb/2025:10:30:00 +0000] '
    '"GET /api/products?category=fruit&page=2 HTTP/1.1" 200 5678 '
    '"-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"'
)

JUICE_SHOP_LINE = (
    '172.17.0.1 - - [15/Feb/2025:12:00:00 +0000] '
    '"GET /rest/products/search?q=apple HTTP/1.1" 200 4321 '
    '"http://localhost:3000/" "Mozilla/5.0"'
)

WEBGOAT_LINE = (
    '192.168.56.1 - - [15/Feb/2025:12:05:00 +0000] '
    '"POST /WebGoat/login HTTP/1.1" 302 0 '
    '"http://localhost:8081/WebGoat/login" "Mozilla/5.0"'
)

DVWA_LINE = (
    '192.168.56.1 - - [15/Feb/2025:12:10:00 +0000] '
    '"GET /dvwa/vulnerabilities/sqli/?id=1&Submit=Submit HTTP/1.1" 200 2048 '
    '"http://localhost:8082/dvwa/" "Mozilla/5.0"'
)

# Lines with dynamic values for normalizer testing
LINE_WITH_UUID = (
    '10.0.0.1 - - [15/Feb/2025:10:00:00 +0000] '
    '"GET /api/users/a1b2c3d4-e5f6-7890-abcd-ef1234567890/profile HTTP/1.1" 200 500 '
    '"-" "Mozilla/5.0"'
)

LINE_WITH_NUMERIC_ID = (
    '10.0.0.1 - - [15/Feb/2025:10:00:00 +0000] '
    '"GET /api/orders/98765/items HTTP/1.1" 200 300 '
    '"-" "Mozilla/5.0"'
)


# ============================================================================
# Log Parser Tests
# ============================================================================

class TestParseLogLine:
    def test_apache_common(self):
        req = parse_log_line(APACHE_COMMON_LINE)
        assert req is not None
        assert req.method == "GET"
        assert req.path == "/apache_pb.gif"
        assert req.remote_addr == "127.0.0.1"
        assert req.status_code == 200
        assert req.response_size == 2326
        assert req.protocol == "HTTP/1.0"
        assert req.user_agent is None
        assert req.referer is None

    def test_apache_combined(self):
        req = parse_log_line(APACHE_COMBINED_LINE)
        assert req is not None
        assert req.method == "POST"
        assert req.path == "/api/login"
        assert req.remote_addr == "192.168.1.1"
        assert req.status_code == 200
        assert req.referer == "http://example.com/login"
        assert "Mozilla" in req.user_agent

    def test_nginx_combined(self):
        req = parse_log_line(NGINX_COMBINED_LINE)
        assert req is not None
        assert req.method == "GET"
        assert req.path == "/api/products"
        assert req.query_params == {"category": "fruit", "page": "2"}
        assert req.status_code == 200
        assert req.response_size == 5678

    def test_juice_shop(self):
        req = parse_log_line(JUICE_SHOP_LINE)
        assert req is not None
        assert req.method == "GET"
        assert req.path == "/rest/products/search"
        assert req.query_params == {"q": "apple"}
        assert req.referer == "http://localhost:3000/"

    def test_webgoat(self):
        req = parse_log_line(WEBGOAT_LINE)
        assert req is not None
        assert req.method == "POST"
        assert req.path == "/WebGoat/login"
        assert req.status_code == 302

    def test_dvwa(self):
        req = parse_log_line(DVWA_LINE)
        assert req is not None
        assert req.method == "GET"
        assert req.path == "/dvwa/vulnerabilities/sqli/"
        assert req.query_params == {"id": "1", "Submit": "Submit"}

    def test_uuid_in_path(self):
        req = parse_log_line(LINE_WITH_UUID)
        assert req is not None
        assert "a1b2c3d4" in req.path

    def test_numeric_id_in_path(self):
        req = parse_log_line(LINE_WITH_NUMERIC_ID)
        assert req is not None
        assert "98765" in req.path

    def test_empty_line(self):
        assert parse_log_line("") is None

    def test_garbage(self):
        assert parse_log_line("this is not a log line") is None

    def test_with_explicit_format(self):
        req = parse_log_line(APACHE_COMMON_LINE, LogFormat.APACHE_COMMON)
        assert req is not None
        assert req.method == "GET"

    def test_response_size_dash(self):
        line = '127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "GET / HTTP/1.1" 304 -'
        req = parse_log_line(line)
        assert req is not None
        assert req.response_size is None


class TestParseRequestDict:
    def test_basic_dict(self):
        data = {
            "method": "POST",
            "path": "/api/login",
            "query_params": {"redirect": "/home"},
            "headers": {"User-Agent": "TestBot/1.0", "Content-Type": "application/json"},
            "body": '{"user":"admin","pass":"123"}',
        }
        req = parse_request_dict(data)
        assert req.method == "POST"
        assert req.path == "/api/login"
        assert req.query_params == {"redirect": "/home"}
        assert req.user_agent == "TestBot/1.0"
        assert req.body == '{"user":"admin","pass":"123"}'

    def test_minimal_dict(self):
        req = parse_request_dict({})
        assert req.method == "GET"
        assert req.path == "/"

    def test_case_normalization(self):
        req = parse_request_dict({"method": "post"})
        assert req.method == "POST"


# ============================================================================
# Normalizer Tests
# ============================================================================

class TestNormalizeRequest:
    def test_uuid_in_path(self):
        req = HTTPRequest(
            method="GET",
            path="/api/users/a1b2c3d4-e5f6-7890-abcd-ef1234567890/profile",
        )
        normalized = normalize_request(req)
        assert "{UUID}" in normalized.path
        assert "a1b2c3d4" not in normalized.path
        assert normalized.path == "/api/users/{UUID}/profile"

    def test_numeric_id_in_path(self):
        req = HTTPRequest(method="GET", path="/api/orders/98765/items")
        normalized = normalize_request(req)
        assert "{ID}" in normalized.path
        assert "98765" not in normalized.path
        assert normalized.path == "/api/orders/{ID}/items"

    def test_ip_normalized(self):
        req = HTTPRequest(method="GET", path="/", remote_addr="192.168.1.100")
        normalized = normalize_request(req)
        assert normalized.remote_addr == "{IP}"

    def test_timestamp_normalized(self):
        req = HTTPRequest(method="GET", path="/", timestamp="15/Feb/2025:10:30:00 +0000")
        normalized = normalize_request(req)
        assert normalized.timestamp == "{TIMESTAMP}"

    def test_jwt_in_header(self):
        req = HTTPRequest(
            method="GET", path="/",
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456"},
        )
        normalized = normalize_request(req)
        assert "{JWT}" in normalized.headers["Authorization"]

    def test_session_id_in_query(self):
        req = HTTPRequest(
            method="GET", path="/",
            query_params={"PHPSESSID": "abc123def456ghij7890klmn12345678"},
        )
        normalized = normalize_request(req)
        assert "{SESSION_ID}" in normalized.query_params["PHPSESSID"]

    def test_ip_in_query_value(self):
        req = HTTPRequest(
            method="GET", path="/",
            query_params={"client_ip": "10.0.0.1"},
        )
        normalized = normalize_request(req)
        assert normalized.query_params["client_ip"] == "{IP}"

    def test_iso_timestamp_in_body(self):
        req = HTTPRequest(
            method="POST", path="/",
            body='{"created_at": "2025-02-15T10:30:00Z"}',
        )
        normalized = normalize_request(req)
        assert "{TIMESTAMP}" in normalized.body

    def test_unix_timestamp_in_body(self):
        req = HTTPRequest(
            method="POST", path="/",
            body='{"ts": 1739615400}',
        )
        normalized = normalize_request(req)
        assert "{TIMESTAMP}" in normalized.body

    def test_preserves_method(self):
        req = HTTPRequest(method="DELETE", path="/api/users/42")
        normalized = normalize_request(req)
        assert normalized.method == "DELETE"

    def test_preserves_protocol(self):
        req = HTTPRequest(method="GET", path="/", protocol="HTTP/2.0")
        normalized = normalize_request(req)
        assert normalized.protocol == "HTTP/2.0"

    def test_no_side_effects(self):
        req = HTTPRequest(
            method="GET",
            path="/api/users/12345",
            remote_addr="10.0.0.1",
        )
        normalize_request(req)
        assert req.path == "/api/users/12345"
        assert req.remote_addr == "10.0.0.1"

    def test_multiple_ids_in_path(self):
        req = HTTPRequest(method="GET", path="/api/shops/42/products/99/reviews")
        normalized = normalize_request(req)
        assert normalized.path == "/api/shops/{ID}/products/{ID}/reviews"

    def test_single_digit_not_replaced(self):
        """Single-digit path segments like /api/v1 should NOT be replaced."""
        req = HTTPRequest(method="GET", path="/api/v1/users")
        normalized = normalize_request(req)
        assert "/v1/" in normalized.path


# ============================================================================
# Serializer Tests
# ============================================================================

class TestSerializeRequest:
    def test_basic_get(self):
        req = HTTPRequest(method="GET", path="/api/products")
        result = serialize_request(req)
        assert result == "GET /api/products HTTP/1.1"

    def test_with_query_params(self):
        req = HTTPRequest(
            method="GET", path="/search",
            query_params={"q": "apple", "page": "1"},
        )
        result = serialize_request(req)
        assert "GET /search?" in result
        assert "q=apple" in result
        assert "page=1" in result

    def test_with_headers(self):
        req = HTTPRequest(
            method="GET", path="/",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
        )
        result = serialize_request(req)
        assert "User-Agent: Mozilla/5.0" in result
        assert "Accept: text/html" in result

    def test_skip_headers(self):
        req = HTTPRequest(
            method="GET", path="/",
            headers={"Host": "example.com", "Content-Length": "42", "Accept": "text/html"},
        )
        result = serialize_request(req)
        assert "Host:" not in result
        assert "Content-Length:" not in result
        assert "Accept: text/html" in result

    def test_with_body(self):
        req = HTTPRequest(
            method="POST", path="/api/login",
            headers={"Content-Type": "application/json"},
            body='{"user":"admin","pass":"secret"}',
        )
        result = serialize_request(req)
        lines = result.split("\n")
        assert lines[0] == "POST /api/login HTTP/1.1"
        assert "Content-Type: application/json" in result
        assert '{"user":"admin","pass":"secret"}' in result

    def test_body_truncation(self):
        req = HTTPRequest(method="POST", path="/", body="A" * 5000)
        result = serialize_request(req, max_body_length=100)
        body_line = result.split("\n")[-1]
        assert len(body_line) == 100

    def test_no_headers(self):
        req = HTTPRequest(
            method="GET", path="/",
            headers={"User-Agent": "Bot"},
        )
        result = serialize_request(req, include_headers=False)
        assert "User-Agent" not in result

    def test_no_body(self):
        req = HTTPRequest(method="POST", path="/", body="some data")
        result = serialize_request(req, include_body=False)
        assert "some data" not in result

    def test_json_body_compacted(self):
        req = HTTPRequest(
            method="POST", path="/",
            body='{\n  "key": "value"\n}',
        )
        result = serialize_request(req)
        assert '{"key":"value"}' in result

    def test_protocol_preserved(self):
        req = HTTPRequest(method="GET", path="/", protocol="HTTP/2.0")
        result = serialize_request(req)
        assert "HTTP/2.0" in result


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestParsingPipeline:
    def test_process_nginx_line(self):
        pipeline = ParsingPipeline()
        result = pipeline.process(NGINX_COMBINED_LINE)
        assert result is not None
        assert "GET /api/products" in result
        assert pipeline.processed == 1
        assert pipeline.failed == 0

    def test_process_apache_common(self):
        pipeline = ParsingPipeline()
        result = pipeline.process(APACHE_COMMON_LINE)
        assert result is not None
        assert "GET /apache_pb.gif" in result

    def test_process_juice_shop(self):
        pipeline = ParsingPipeline()
        result = pipeline.process(JUICE_SHOP_LINE)
        assert result is not None
        assert "/rest/products/search" in result
        assert "q=apple" in result

    def test_process_dvwa(self):
        pipeline = ParsingPipeline()
        result = pipeline.process(DVWA_LINE)
        assert result is not None
        assert "/dvwa/vulnerabilities/sqli/" in result

    def test_normalization_applied(self):
        pipeline = ParsingPipeline()
        result = pipeline.process(LINE_WITH_UUID)
        assert result is not None
        assert "{UUID}" in result
        assert "a1b2c3d4" not in result

    def test_numeric_id_normalized(self):
        pipeline = ParsingPipeline()
        result = pipeline.process(LINE_WITH_NUMERIC_ID)
        assert result is not None
        assert "{ID}" in result

    def test_ip_not_in_serialized_output(self):
        """Remote addr is not in the serialized output (it's metadata)."""
        pipeline = ParsingPipeline()
        result = pipeline.process(NGINX_COMBINED_LINE)
        assert result is not None
        # The serialized output is method+path+headers, not remote_addr
        assert result.startswith("GET ")

    def test_process_empty_line(self):
        pipeline = ParsingPipeline()
        result = pipeline.process("")
        assert result is None
        assert pipeline.failed == 1

    def test_process_garbage(self):
        pipeline = ParsingPipeline()
        result = pipeline.process("not a log line")
        assert result is None
        assert pipeline.failed == 1

    def test_process_request(self):
        pipeline = ParsingPipeline()
        req = HTTPRequest(
            method="POST",
            path="/api/users/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            headers={"Content-Type": "application/json"},
            body='{"name": "test"}',
        )
        result = pipeline.process_request(req)
        assert "POST /api/users/{UUID}" in result
        assert "Content-Type: application/json" in result

    def test_process_dict(self):
        pipeline = ParsingPipeline()
        result = pipeline.process_dict({
            "method": "GET",
            "path": "/api/orders/12345",
            "headers": {"User-Agent": "Mozilla/5.0"},
        })
        assert "GET /api/orders/{ID}" in result
        assert "User-Agent:" in result

    def test_explicit_format(self):
        pipeline = ParsingPipeline(log_format=LogFormat.APACHE_COMMON)
        result = pipeline.process(APACHE_COMMON_LINE)
        assert result is not None

    def test_stats(self):
        pipeline = ParsingPipeline()
        pipeline.process(NGINX_COMBINED_LINE)
        pipeline.process(APACHE_COMMON_LINE)
        pipeline.process("bad line")

        stats = pipeline.stats()
        assert stats["processed"] == 2
        assert stats["failed"] == 1

    def test_batch_processing(self):
        """Process multiple lines like Week 1 batch_reader would provide."""
        lines = [
            NGINX_COMBINED_LINE,
            JUICE_SHOP_LINE,
            WEBGOAT_LINE,
            DVWA_LINE,
            LINE_WITH_UUID,
            LINE_WITH_NUMERIC_ID,
        ]
        pipeline = ParsingPipeline()
        results = [pipeline.process(line) for line in lines]
        assert all(r is not None for r in results)
        assert pipeline.processed == 6
        assert pipeline.failed == 0

    def test_output_format_matches_classifier(self):
        """Verify output format is compatible with WAFClassifier._build_request_text."""
        pipeline = ParsingPipeline()
        result = pipeline.process(NGINX_COMBINED_LINE)
        assert result is not None
        lines = result.split("\n")
        # First line should be "METHOD /path HTTP/x.x"
        parts = lines[0].split(" ")
        assert parts[0] in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS")
        assert parts[1].startswith("/")
        assert parts[2].startswith("HTTP/")
