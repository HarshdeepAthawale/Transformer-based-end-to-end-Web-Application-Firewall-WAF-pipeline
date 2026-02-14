"""
Unit tests for backend.ingestion module.

Tests format detector, batch reader, stream tailer, and async queue.
"""

import asyncio
import gzip
import os
import tempfile
from pathlib import Path

import pytest

from backend.ingestion.format_detector import LogFormat, detect_format, detect_from_file
from backend.ingestion.batch_reader import read_chunks
from backend.ingestion.stream_tailer import tail_lines
from backend.ingestion.queue import IngestionQueue
from backend.ingestion.retry import RetryConfig, with_retry_sync, with_retry_async, retry
from backend.ingestion.config import (
    BatchConfig, StreamingConfig, IngestionConfig, load_ingestion_config,
)


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
    '"GET /api/products HTTP/1.1" 200 5678 '
    '"-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"'
)

# Juice Shop style
JUICE_SHOP_LINE = (
    '172.17.0.1 - - [15/Feb/2025:12:00:00 +0000] '
    '"GET /rest/products/search?q=apple HTTP/1.1" 200 4321 '
    '"http://localhost:3000/" "Mozilla/5.0"'
)

# WebGoat style
WEBGOAT_LINE = (
    '192.168.56.1 - - [15/Feb/2025:12:05:00 +0000] '
    '"POST /WebGoat/login HTTP/1.1" 302 0 '
    '"http://localhost:8081/WebGoat/login" "Mozilla/5.0"'
)

# DVWA style
DVWA_LINE = (
    '192.168.56.1 - - [15/Feb/2025:12:10:00 +0000] '
    '"GET /dvwa/vulnerabilities/sqli/?id=1&Submit=Submit HTTP/1.1" 200 2048 '
    '"http://localhost:8082/dvwa/" "Mozilla/5.0"'
)


# ============================================================================
# Format Detector Tests
# ============================================================================

class TestDetectFormat:
    def test_apache_common(self):
        assert detect_format(APACHE_COMMON_LINE) == LogFormat.APACHE_COMMON

    def test_apache_combined(self):
        assert detect_format(APACHE_COMBINED_LINE) == LogFormat.APACHE_COMBINED

    def test_nginx_combined(self):
        assert detect_format(NGINX_COMBINED_LINE) == LogFormat.NGINX_COMBINED

    def test_juice_shop_line(self):
        fmt = detect_format(JUICE_SHOP_LINE)
        assert fmt in (LogFormat.NGINX_COMBINED, LogFormat.APACHE_COMBINED)

    def test_webgoat_line(self):
        fmt = detect_format(WEBGOAT_LINE)
        assert fmt in (LogFormat.NGINX_COMBINED, LogFormat.APACHE_COMBINED)

    def test_dvwa_line(self):
        fmt = detect_format(DVWA_LINE)
        assert fmt in (LogFormat.NGINX_COMBINED, LogFormat.APACHE_COMBINED)

    def test_empty_line(self):
        assert detect_format("") == LogFormat.UNKNOWN

    def test_garbage(self):
        assert detect_format("this is not a log line") == LogFormat.UNKNOWN

    def test_whitespace_only(self):
        assert detect_format("   \t  ") == LogFormat.UNKNOWN

    def test_partial_line(self):
        assert detect_format('127.0.0.1 - - [date]') == LogFormat.UNKNOWN


class TestDetectFromFile:
    def test_detect_combined_file(self, tmp_path):
        log_file = tmp_path / "access.log"
        lines = "\n".join([NGINX_COMBINED_LINE] * 5)
        log_file.write_text(lines)
        assert detect_from_file(str(log_file)) == LogFormat.NGINX_COMBINED

    def test_detect_common_file(self, tmp_path):
        log_file = tmp_path / "access.log"
        lines = "\n".join([APACHE_COMMON_LINE] * 5)
        log_file.write_text(lines)
        assert detect_from_file(str(log_file)) == LogFormat.APACHE_COMMON

    def test_detect_from_gz(self, tmp_path):
        log_file = tmp_path / "access.log.gz"
        content = "\n".join([NGINX_COMBINED_LINE] * 5)
        with gzip.open(log_file, "wt") as f:
            f.write(content)
        assert detect_from_file(str(log_file)) == LogFormat.NGINX_COMBINED

    def test_empty_file(self, tmp_path):
        log_file = tmp_path / "empty.log"
        log_file.write_text("")
        assert detect_from_file(str(log_file)) == LogFormat.UNKNOWN

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            detect_from_file("/nonexistent/path/access.log")

    def test_majority_vote(self, tmp_path):
        log_file = tmp_path / "mixed.log"
        lines = [APACHE_COMMON_LINE] * 3 + [NGINX_COMBINED_LINE] * 7
        log_file.write_text("\n".join(lines))
        assert detect_from_file(str(log_file)) == LogFormat.NGINX_COMBINED


# ============================================================================
# Batch Reader Tests
# ============================================================================

class TestReadChunks:
    def _make_log(self, tmp_path, lines: list[str], gz: bool = False) -> str:
        if gz:
            path = tmp_path / "access.log.gz"
            with gzip.open(path, "wt") as f:
                f.write("\n".join(lines) + "\n")
        else:
            path = tmp_path / "access.log"
            path.write_text("\n".join(lines) + "\n")
        return str(path)

    def test_basic_chunking(self, tmp_path):
        lines = [f"line {i}" for i in range(10)]
        path = self._make_log(tmp_path, lines)
        chunks = list(read_chunks(path, chunk_size=3))
        assert len(chunks) == 4  # 3+3+3+1
        assert chunks[0] == ["line 0", "line 1", "line 2"]
        assert chunks[-1] == ["line 9"]

    def test_single_chunk(self, tmp_path):
        lines = [NGINX_COMBINED_LINE] * 5
        path = self._make_log(tmp_path, lines)
        chunks = list(read_chunks(path, chunk_size=100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_max_lines(self, tmp_path):
        lines = [f"line {i}" for i in range(100)]
        path = self._make_log(tmp_path, lines)
        all_lines = [l for chunk in read_chunks(path, chunk_size=10, max_lines=25) for l in chunk]
        assert len(all_lines) == 25

    def test_skip_lines(self, tmp_path):
        lines = [f"line {i}" for i in range(10)]
        path = self._make_log(tmp_path, lines)
        all_lines = [l for chunk in read_chunks(path, chunk_size=100, skip_lines=5) for l in chunk]
        assert len(all_lines) == 5
        assert all_lines[0] == "line 5"

    def test_skip_and_max(self, tmp_path):
        lines = [f"line {i}" for i in range(20)]
        path = self._make_log(tmp_path, lines)
        all_lines = [
            l for chunk in read_chunks(path, chunk_size=100, skip_lines=5, max_lines=3)
            for l in chunk
        ]
        assert len(all_lines) == 3
        assert all_lines[0] == "line 5"

    def test_gz_file(self, tmp_path):
        lines = [APACHE_COMBINED_LINE] * 5
        path = self._make_log(tmp_path, lines, gz=True)
        chunks = list(read_chunks(path, chunk_size=100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_empty_lines_skipped(self, tmp_path):
        lines = ["line 0", "", "  ", "line 1", "", "line 2"]
        path = self._make_log(tmp_path, lines)
        all_lines = [l for chunk in read_chunks(path, chunk_size=100) for l in chunk]
        assert all_lines == ["line 0", "line 1", "line 2"]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            list(read_chunks("/nonexistent.log"))

    def test_invalid_chunk_size(self, tmp_path):
        path = self._make_log(tmp_path, ["line"])
        with pytest.raises(ValueError):
            list(read_chunks(path, chunk_size=0))

    def test_real_log_lines(self, tmp_path):
        lines = [JUICE_SHOP_LINE, WEBGOAT_LINE, DVWA_LINE]
        path = self._make_log(tmp_path, lines)
        chunks = list(read_chunks(path, chunk_size=2))
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1


# ============================================================================
# Stream Tailer Tests
# ============================================================================

class TestTailLines:
    @pytest.mark.asyncio
    async def test_tail_existing_content(self, tmp_path):
        """Tail a file with existing content (start_from_end=False)."""
        log_file = tmp_path / "access.log"
        log_file.write_text("line 1\nline 2\nline 3\n")

        lines = []
        async for line in tail_lines(
            str(log_file), follow=False, start_from_end=False
        ):
            lines.append(line)

        assert lines == ["line 1", "line 2", "line 3"]

    @pytest.mark.asyncio
    async def test_tail_start_from_end(self, tmp_path):
        """Starting from end with follow=False should yield nothing."""
        log_file = tmp_path / "access.log"
        log_file.write_text("line 1\nline 2\n")

        lines = []
        async for line in tail_lines(
            str(log_file), follow=False, start_from_end=True
        ):
            lines.append(line)

        assert lines == []

    @pytest.mark.asyncio
    async def test_tail_follow_new_lines(self, tmp_path):
        """Test that new lines appended while tailing are picked up."""
        log_file = tmp_path / "access.log"
        log_file.write_text("")

        collected: list[str] = []

        async def writer():
            await asyncio.sleep(0.15)
            with open(log_file, "a") as f:
                f.write("new line 1\n")
                f.flush()
            await asyncio.sleep(0.15)
            with open(log_file, "a") as f:
                f.write("new line 2\n")
                f.flush()

        async def reader():
            async for line in tail_lines(
                str(log_file), follow=True, poll_interval=0.05,
                start_from_end=True,
            ):
                collected.append(line)
                if len(collected) >= 2:
                    break

        writer_task = asyncio.create_task(writer())
        try:
            await asyncio.wait_for(reader(), timeout=3.0)
        except asyncio.TimeoutError:
            pass
        await writer_task

        assert collected == ["new line 1", "new line 2"]

    @pytest.mark.asyncio
    async def test_tail_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            async for _ in tail_lines("/nonexistent.log"):
                pass

    @pytest.mark.asyncio
    async def test_tail_skips_empty_lines(self, tmp_path):
        log_file = tmp_path / "access.log"
        log_file.write_text("line 1\n\n  \nline 2\n")

        lines = []
        async for line in tail_lines(
            str(log_file), follow=False, start_from_end=False
        ):
            lines.append(line)

        assert lines == ["line 1", "line 2"]


# ============================================================================
# Async Queue Tests
# ============================================================================

class TestIngestionQueue:
    @pytest.mark.asyncio
    async def test_put_get(self):
        q = IngestionQueue(maxsize=10)
        await q.put("item1")
        await q.put("item2")
        assert q.size == 2
        assert q.produced == 2

        item = await q.get()
        assert item == "item1"
        assert q.consumed == 1

    @pytest.mark.asyncio
    async def test_consumer(self):
        q = IngestionQueue(maxsize=10)
        results = []

        async def handler(item):
            results.append(item)

        # Produce items then signal shutdown
        for i in range(5):
            await q.put(f"item{i}")
        await q.shutdown()

        await q.consume(handler)
        assert results == [f"item{i}" for i in range(5)]
        assert q.consumed == 5

    @pytest.mark.asyncio
    async def test_shutdown_stops_consumer(self):
        q = IngestionQueue(maxsize=10)

        async def handler(item):
            pass

        await q.shutdown()
        await asyncio.wait_for(q.consume(handler), timeout=1.0)
        assert not q.is_running

    @pytest.mark.asyncio
    async def test_stats(self):
        q = IngestionQueue(maxsize=100, name="test-queue")
        await q.put("a")
        await q.put("b")
        await q.get()

        stats = q.stats()
        assert stats["name"] == "test-queue"
        assert stats["produced"] == 2
        assert stats["consumed"] == 1
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_handler_error_does_not_stop_consumer(self):
        q = IngestionQueue(maxsize=10)
        results = []

        async def handler(item):
            if item == "bad":
                raise ValueError("test error")
            results.append(item)

        await q.put("good1")
        await q.put("bad")
        await q.put("good2")
        await q.shutdown()

        await q.consume(handler)
        assert results == ["good1", "good2"]

    @pytest.mark.asyncio
    async def test_get_timeout(self):
        q = IngestionQueue(maxsize=10)
        with pytest.raises(asyncio.TimeoutError):
            await q.get(timeout=0.1)


# ============================================================================
# Retry Tests
# ============================================================================

class TestRetry:
    def test_succeeds_first_try(self):
        call_count = 0
        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"
        result = with_retry_sync(fn, RetryConfig(max_retries=3))
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_oserror(self):
        call_count = 0
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("transient")
            return "recovered"
        config = RetryConfig(max_retries=3, initial_delay=0.01, max_delay=0.05)
        result = with_retry_sync(fn, config)
        assert result == "recovered"
        assert call_count == 3

    def test_exhausts_retries(self):
        def fn():
            raise IOError("persistent")
        config = RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.05)
        with pytest.raises(IOError, match="persistent"):
            with_retry_sync(fn, config)

    def test_does_not_retry_non_retryable(self):
        call_count = 0
        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        with pytest.raises(ValueError):
            with_retry_sync(fn, config)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry(self):
        call_count = 0
        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return "ok"
        config = RetryConfig(max_retries=3, initial_delay=0.01, max_delay=0.05)
        result = await with_retry_async(fn, config)
        assert result == "ok"
        assert call_count == 2

    def test_decorator_sync(self):
        call_count = 0
        config = RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.05)

        @retry(config)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("flaky")
            return "done"

        assert flaky() == "done"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_async(self):
        call_count = 0
        config = RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.05)

        @retry(config)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("flaky")
            return "done"

        assert await flaky() == "done"
        assert call_count == 2


# ============================================================================
# Config Loader Tests
# ============================================================================

class TestIngestionConfig:
    def test_load_from_project_config(self):
        """Load from the real config/config.yaml if it exists."""
        config = load_ingestion_config()
        assert config.batch.chunk_size == 1000
        assert config.streaming.poll_interval == 0.1
        assert config.retry.max_retries == 3

    def test_load_from_custom_path(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
ingestion:
  batch:
    chunk_size: 500
    max_lines: 5000
    skip_lines: 10
  streaming:
    poll_interval: 0.5
    follow: false
    buffer_size: 2000
  retry:
    max_retries: 5
    initial_delay: 2.0
    max_delay: 120.0
    exponential_base: 3.0
""")
        config = load_ingestion_config(str(config_file))
        assert config.batch.chunk_size == 500
        assert config.batch.max_lines == 5000
        assert config.batch.skip_lines == 10
        assert config.streaming.poll_interval == 0.5
        assert config.streaming.follow is False
        assert config.streaming.buffer_size == 2000
        assert config.retry.max_retries == 5
        assert config.retry.initial_delay == 2.0
        assert config.retry.max_delay == 120.0
        assert config.retry.exponential_base == 3.0

    def test_missing_file_returns_defaults(self, tmp_path):
        config = load_ingestion_config(str(tmp_path / "nonexistent.yaml"))
        assert config.batch.chunk_size == 1000
        assert config.streaming.follow is True
        assert config.retry.max_retries == 3

    def test_empty_ingestion_section(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ingestion: {}\n")
        config = load_ingestion_config(str(config_file))
        assert config.batch.chunk_size == 1000  # defaults

    def test_partial_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
ingestion:
  batch:
    chunk_size: 2000
""")
        config = load_ingestion_config(str(config_file))
        assert config.batch.chunk_size == 2000
        assert config.streaming.poll_interval == 0.1  # default
        assert config.retry.max_retries == 3  # default

    def test_defaults_factory(self):
        config = IngestionConfig.defaults()
        assert config.batch.chunk_size == 1000
        assert config.streaming.buffer_size == 10000
        assert config.retry.exponential_base == 2.0
