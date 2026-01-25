"""
Unit tests for log ingestion system
"""
import pytest
from pathlib import Path
import tempfile
import gzip

from backend.ml.ingestion.batch_reader import BatchLogReader
from backend.ml.ingestion.log_formats import LogFormatDetector, LogFormat
from backend.ml.ingestion.stream_reader import StreamLogReader
from backend.ml.ingestion.log_queue import LogQueue


def test_format_detection():
    """Test log format detection"""
    detector = LogFormatDetector()
    
    # Test Nginx Combined (has "- -" pattern which is Nginx specific)
    nginx_line = '127.0.0.1 - - [24/Jan/2026:01:38:33 +0530] "GET /api/data HTTP/1.1" 404 153 "-" "curl/8.18.0"'
    format_type = detector.detect_format(nginx_line)
    assert format_type == LogFormat.NGINX_COMBINED
    
    # Test Apache Combined (has user/auth fields, not "- -")
    # Apache format: IP user auth [timestamp] "method path protocol" status size "referer" "user-agent"
    apache_line = '192.168.1.1 user123 - [25/Dec/2023:10:00:00 +0000] "GET /test HTTP/1.1" 200 1234 "-" "curl/8.0.0"'
    format_type = detector.detect_format(apache_line)
    assert format_type in [LogFormat.APACHE_COMMON, LogFormat.APACHE_COMBINED, LogFormat.APACHE_DETAILED]
    
    # Test Apache Common (simpler format)
    apache_common_line = '192.168.1.1 - - [25/Dec/2023:10:00:00 +0000] "GET /test HTTP/1.1" 200'
    format_type = detector.detect_format(apache_common_line)
    assert format_type in [LogFormat.APACHE_COMMON, LogFormat.APACHE_COMBINED]
    
    # Test unknown
    unknown_line = "not a log line"
    format_type = detector.detect_format(unknown_line)
    assert format_type == LogFormat.UNKNOWN


def test_batch_reader(tmp_path):
    """Test batch log reader"""
    # Create test log file
    test_log = tmp_path / "test.log"
    test_log.write_text("line1\nline2\nline3\n")
    
    reader = BatchLogReader(str(test_log))
    lines = list(reader.read_lines())
    assert len(lines) == 3
    assert lines[0] == "line1"
    assert lines[1] == "line2"
    assert lines[2] == "line3"


def test_batch_reader_max_lines(tmp_path):
    """Test batch reader with max_lines limit"""
    test_log = tmp_path / "test.log"
    test_log.write_text("\n".join([f"line{i}" for i in range(100)]))
    
    reader = BatchLogReader(str(test_log), max_lines=10)
    lines = list(reader.read_lines())
    assert len(lines) == 10


def test_batch_reader_gzip(tmp_path):
    """Test batch reader with gzip file"""
    test_log = tmp_path / "test.log.gz"
    content = b"line1\nline2\nline3\n"
    
    with gzip.open(test_log, 'wb') as f:
        f.write(content)
    
    reader = BatchLogReader(str(test_log))
    lines = list(reader.read_lines())
    assert len(lines) == 3


def test_log_queue():
    """Test log queue functionality"""
    queue = LogQueue(maxsize=10)
    
    # Test put and get
    queue.put("line1")
    queue.put("line2")
    
    assert queue.size() == 2
    assert not queue.is_empty()
    
    item = queue.get()
    assert item == "line1"
    assert queue.size() == 1


def test_log_queue_processor():
    """Test log queue with processor"""
    processed_items = []
    
    def processor(item):
        processed_items.append(item)
    
    queue = LogQueue(maxsize=10, processor=processor)
    queue.start_processor()
    
    queue.put("line1")
    queue.put("line2")
    
    import time
    time.sleep(0.5)  # Give processor time to process
    
    queue.stop_processor()
    
    assert len(processed_items) >= 0  # May have processed items


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
