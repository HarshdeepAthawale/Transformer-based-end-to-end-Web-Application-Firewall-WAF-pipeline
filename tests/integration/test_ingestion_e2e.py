"""
End-to-end integration test for the log ingestion pipeline.

Tests the full flow: log file → LogProcessor → format detection → batch reading → queue.
"""

import time

import pytest

from backend.ingestion.config import IngestionConfig, BatchConfig, StreamingConfig
from backend.ingestion.retry import RetryConfig
from backend.ingestion.queue import IngestionQueue
from backend.tasks.log_processor import LogProcessor


SAMPLE_NGINX_LINES = [
    '10.0.0.1 - - [15/Feb/2025:10:00:01 +0000] "GET /api/products HTTP/1.1" 200 1234 "-" "Mozilla/5.0"',
    '10.0.0.2 - - [15/Feb/2025:10:00:02 +0000] "POST /api/login HTTP/1.1" 200 567 "-" "Mozilla/5.0"',
    '10.0.0.3 - - [15/Feb/2025:10:00:03 +0000] "GET /rest/products/search?q=juice HTTP/1.1" 200 890 "-" "Mozilla/5.0"',
    '10.0.0.4 - - [15/Feb/2025:10:00:04 +0000] "GET /WebGoat/login HTTP/1.1" 302 0 "-" "Mozilla/5.0"',
    '10.0.0.5 - - [15/Feb/2025:10:00:05 +0000] "GET /dvwa/vulnerabilities/sqli/ HTTP/1.1" 200 2048 "-" "Mozilla/5.0"',
    '10.0.0.6 - - [15/Feb/2025:10:00:06 +0000] "PUT /api/users/42 HTTP/1.1" 200 128 "-" "Mozilla/5.0"',
    '10.0.0.7 - - [15/Feb/2025:10:00:07 +0000] "DELETE /api/sessions HTTP/1.1" 204 0 "-" "Mozilla/5.0"',
    '10.0.0.8 - - [15/Feb/2025:10:00:08 +0000] "GET /static/main.js HTTP/1.1" 200 45678 "-" "Mozilla/5.0"',
    '10.0.0.9 - - [15/Feb/2025:10:00:09 +0000] "GET /favicon.ico HTTP/1.1" 404 0 "-" "Mozilla/5.0"',
    '10.0.0.10 - - [15/Feb/2025:10:00:10 +0000] "POST /api/register HTTP/1.1" 201 256 "-" "Mozilla/5.0"',
]


class TestIngestionEndToEnd:
    """End-to-end test: log file → LogProcessor → queue."""

    def _make_config(self) -> IngestionConfig:
        return IngestionConfig(
            batch=BatchConfig(chunk_size=5),
            streaming=StreamingConfig(buffer_size=100),
            retry=RetryConfig(max_retries=1, initial_delay=0.1, max_delay=0.5),
        )

    def test_full_pipeline(self, tmp_path):
        """Write a log file, run LogProcessor, verify all lines are queued."""
        log_file = tmp_path / "access.log"
        log_file.write_text("\n".join(SAMPLE_NGINX_LINES) + "\n")

        config = self._make_config()
        queue = IngestionQueue(maxsize=100, name="test-e2e")
        processor = LogProcessor(
            log_path=str(log_file),
            interval_seconds=1,
            config=config,
            queue=queue,
        )

        processor.start()
        time.sleep(2)  # Let one processing cycle complete
        processor.stop()

        assert processor.detected_format is not None
        assert processor.detected_format.value in ("nginx_combined", "apache_combined")
        assert processor.lines_read == 10
        assert queue.produced == 10
        assert queue.size == 10

    def test_incremental_reading(self, tmp_path):
        """Append lines between cycles — processor reads only new lines."""
        log_file = tmp_path / "access.log"
        # Start with 5 lines
        log_file.write_text("\n".join(SAMPLE_NGINX_LINES[:5]) + "\n")

        config = self._make_config()
        queue = IngestionQueue(maxsize=100, name="test-incremental")
        processor = LogProcessor(
            log_path=str(log_file),
            interval_seconds=1,
            config=config,
            queue=queue,
        )

        processor.start()
        time.sleep(1.5)

        # Append 5 more lines
        with open(log_file, "a") as f:
            f.write("\n".join(SAMPLE_NGINX_LINES[5:]) + "\n")

        time.sleep(2)  # Wait for next cycle
        processor.stop()

        assert processor.lines_read == 10
        assert queue.produced == 10

    def test_missing_log_file(self, tmp_path):
        """Processor handles missing log file gracefully."""
        config = self._make_config()
        queue = IngestionQueue(maxsize=100, name="test-missing")
        processor = LogProcessor(
            log_path=str(tmp_path / "nonexistent.log"),
            interval_seconds=1,
            config=config,
            queue=queue,
        )

        processor.start()
        time.sleep(1.5)
        processor.stop()

        assert processor.lines_read == 0
        assert queue.size == 0

    def test_stats(self, tmp_path):
        """Processor stats reflect correct state after processing."""
        log_file = tmp_path / "access.log"
        log_file.write_text("\n".join(SAMPLE_NGINX_LINES[:3]) + "\n")

        config = self._make_config()
        processor = LogProcessor(
            log_path=str(log_file),
            interval_seconds=1,
            config=config,
        )

        processor.start()
        time.sleep(1.5)
        processor.stop()

        stats = processor.stats()
        assert stats["running"] is False
        assert stats["lines_read"] == 3
        assert stats["detected_format"] is not None
        assert stats["queue"]["produced"] == 3
