#!/usr/bin/env python3
"""
Test Parsing System

Tests the parsing and normalization system with real logs
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsing.pipeline import ParsingPipeline
from src.parsing.log_parser import LogParser
from src.parsing.normalizer import RequestNormalizer
from loguru import logger


def test_parser():
    """Test log parser"""
    logger.info("Testing log parser...")
    
    parser = LogParser()
    
    # Test Nginx format
    nginx_line = '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /api/data HTTP/1.1" 404 153 "-" "curl/8.18.0"'
    request = parser.parse(nginx_line)
    
    assert request is not None, "Failed to parse Nginx log"
    assert request.method == "GET", f"Expected GET, got {request.method}"
    assert request.path == "/api/data", f"Expected /api/data, got {request.path}"
    
    logger.info("✓ Parser test passed")


def test_normalizer():
    """Test request normalizer"""
    logger.info("Testing normalizer...")
    
    from src.parsing.log_parser import HTTPRequest
    
    request = HTTPRequest(
        method="GET",
        path="/users/550e8400-e29b-41d4-a716-446655440000",
        query_params={"session": "abc123def456", "timestamp": "1703505600"},
        remote_addr="192.168.1.1"
    )
    
    normalizer = RequestNormalizer()
    normalized = normalizer.normalize(request)
    
    assert "<UUID>" in normalized.path or "<ID>" in normalized.path, "UUID not normalized"
    assert normalized.remote_addr == "<IP_ADDRESS>", "IP not normalized"
    
    logger.info("✓ Normalizer test passed")


def test_pipeline():
    """Test complete pipeline"""
    logger.info("Testing pipeline...")
    
    pipeline = ParsingPipeline()
    log_line = '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /api/users/123?token=abc123 HTTP/1.1" 404 153 "-" "curl/8.18.0"'
    
    result = pipeline.process_log_line(log_line)
    
    assert result is not None, "Pipeline returned None"
    assert "<NUMERIC_ID>" in result or "<ID>" in result, "Numeric ID not normalized"
    
    logger.info("✓ Pipeline test passed")
    logger.info(f"  Result: {result[:100]}...")


def test_real_logs():
    """Test with real log files"""
    logger.info("Testing with real logs...")
    
    pipeline = ParsingPipeline()
    log_file = project_root / "data" / "raw" / "final_test.log"
    
    if not log_file.exists():
        logger.warning("Real log file not found, skipping")
        return
    
    count = 0
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            result = pipeline.process_log_line(line)
            if result:
                count += 1
            if count >= 5:
                break
    
    assert count > 0, "No logs processed"
    logger.info(f"✓ Processed {count} real log lines")


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Parsing System - Test Suite")
    logger.info("=" * 60)
    
    try:
        test_parser()
        test_normalizer()
        test_pipeline()
        test_real_logs()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
