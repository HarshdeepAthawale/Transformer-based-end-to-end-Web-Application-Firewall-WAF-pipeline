"""
Unit tests for parsing system
"""
import pytest
from backend.ml.parsing.log_parser import LogParser, HTTPRequest
from backend.ml.parsing.normalizer import RequestNormalizer
from backend.ml.parsing.pipeline import ParsingPipeline


def test_nginx_parsing():
    """Test Nginx log parsing"""
    parser = LogParser()
    log_line = '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /test?id=123 HTTP/1.1" 404 153 "-" "curl/8.18.0"'
    
    request = parser.parse(log_line)
    assert request is not None
    assert request.method == "GET"
    assert request.path == "/test"
    assert "id" in request.query_params
    assert request.query_params["id"] == "123"


def test_apache_parsing():
    """Test Apache log parsing"""
    parser = LogParser()
    log_line = '192.168.1.1 - - [25/Dec/2023:10:00:00 +0000] "GET /api/users?page=1 HTTP/1.1" 200 1234 "-" "curl/8.0.0"'
    
    request = parser.parse(log_line)
    assert request is not None
    assert request.method == "GET"
    assert request.path == "/api/users"
    assert "page" in request.query_params


def test_normalization():
    """Test request normalization"""
    request = HTTPRequest(
        method="GET",
        path="/users/550e8400-e29b-41d4-a716-446655440000",
        query_params={"session": "abc123def456", "timestamp": "1703505600"},
        remote_addr="192.168.1.1"
    )
    
    normalizer = RequestNormalizer()
    normalized = normalizer.normalize(request)
    
    assert "<UUID>" in normalized.path or "<ID>" in normalized.path
    assert normalized.remote_addr == "<IP_ADDRESS>"
    assert "<SESSION_ID>" in str(normalized.query_params) or "<TIMESTAMP>" in str(normalized.query_params)


def test_pipeline():
    """Test complete pipeline"""
    pipeline = ParsingPipeline()
    log_line = '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /api/users/123?token=abc123 HTTP/1.1" 404 153 "-" "curl/8.18.0"'
    
    result = pipeline.process_log_line(log_line)
    assert result is not None
    assert "<NUMERIC_ID>" in result or "<ID>" in result


def test_query_params_parsing():
    """Test query parameter parsing"""
    parser = LogParser()
    log_line = '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /test?foo=bar&baz=qux HTTP/1.1" 200 100 "-" "curl/8.18.0"'
    
    request = parser.parse(log_line)
    assert request is not None
    assert "foo" in request.query_params
    assert "baz" in request.query_params
    assert request.query_params["foo"] == "bar"
    assert request.query_params["baz"] == "qux"


def test_path_normalization():
    """Test path normalization"""
    request = HTTPRequest(
        method="GET",
        path="/api/users/12345/posts/67890",
        remote_addr="192.168.1.1"
    )
    
    normalizer = RequestNormalizer()
    normalized = normalizer.normalize(request)
    
    assert "<ID>" in normalized.path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
