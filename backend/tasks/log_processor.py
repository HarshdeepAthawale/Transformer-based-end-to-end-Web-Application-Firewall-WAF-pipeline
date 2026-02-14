"""
Log Processor - Background task for ingesting and queuing WAF logs

Uses the ingestion pipeline (format detection, batch reading, queue)
to read from configured log files and buffer lines for downstream parsing.
"""

import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.ingestion.format_detector import LogFormat, detect_from_file
from backend.ingestion.batch_reader import read_chunks
from backend.ingestion.queue import IngestionQueue
from backend.ingestion.config import load_ingestion_config, IngestionConfig
from backend.ingestion.retry import RetryConfig, with_retry_sync


class LogProcessor:
    """
    Background worker that ingests log files into an async queue.

    Reads new lines from a log file on each processing cycle,
    tracks position to avoid re-reading, and pushes lines into
    an IngestionQueue for downstream consumers (parsing pipeline).
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        interval_seconds: int = 10,
        config: Optional[IngestionConfig] = None,
        queue: Optional[IngestionQueue] = None,
    ):
        """
        Initialize the log processor.

        Args:
            log_path: Path to the log file to ingest. If None, reads from
                      backend config LOG_PATH.
            interval_seconds: Seconds between processing cycles.
            config: Ingestion config. Loaded from YAML if None.
            queue: Queue to push lines into. Created with config defaults if None.
        """
        self._config = config or load_ingestion_config()
        self._log_path = log_path or self._resolve_log_path()
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lines_read = 0
        self._detected_format: Optional[LogFormat] = None

        self.queue = queue or IngestionQueue(
            maxsize=self._config.streaming.buffer_size,
            name="log-processor",
        )

    def _resolve_log_path(self) -> Optional[str]:
        """Resolve log path from backend config."""
        try:
            from backend.config import config as app_config
            return app_config.LOG_PATH
        except Exception:
            return None

    @property
    def detected_format(self) -> Optional[LogFormat]:
        return self._detected_format

    @property
    def lines_read(self) -> int:
        return self._lines_read

    def start(self):
        """Start the log processor background thread."""
        if self._running:
            logger.warning("Log processor is already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            "Log processor started (interval: {}s, log_path: {})",
            self.interval, self._log_path,
        )

    def stop(self):
        """Stop the log processor."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info(
            "Log processor stopped (total lines read: {})", self._lines_read
        )

    def _run(self):
        """Main processing loop."""
        while self._running:
            try:
                self._process_logs()
            except Exception as e:
                logger.error("Log processing error: {}", e)
            time.sleep(self.interval)

    def _process_logs(self):
        """Read new lines from the log file and push them into the queue."""
        if not self._log_path:
            return

        path = Path(self._log_path)
        if not path.exists():
            return

        # Detect format on first run
        if self._detected_format is None:
            try:
                self._detected_format = with_retry_sync(
                    detect_from_file,
                    RetryConfig(
                        max_retries=self._config.retry.max_retries,
                        initial_delay=self._config.retry.initial_delay,
                        max_delay=self._config.retry.max_delay,
                        exponential_base=self._config.retry.exponential_base,
                    ),
                    self._log_path,
                )
                logger.info(
                    "Detected log format: {} for {}",
                    self._detected_format.value, self._log_path,
                )
            except Exception as e:
                logger.error("Failed to detect log format: {}", e)
                return

        # Read new lines (skip already-read lines)
        batch_config = self._config.batch
        new_lines = 0

        try:
            for chunk in read_chunks(
                self._log_path,
                chunk_size=batch_config.chunk_size,
                skip_lines=self._lines_read,
                max_lines=batch_config.max_lines,
            ):
                for line in chunk:
                    # Use non-blocking put to avoid deadlock in sync thread
                    try:
                        self.queue._queue.put_nowait(line)
                        self.queue._produced += 1
                    except Exception:
                        logger.warning("Queue full, dropping line")
                        break
                    new_lines += 1

            self._lines_read += new_lines

            if new_lines > 0:
                logger.debug(
                    "Ingested {} new lines from {} (total: {})",
                    new_lines, self._log_path, self._lines_read,
                )
        except FileNotFoundError:
            logger.warning("Log file not found: {}", self._log_path)
        except Exception as e:
            logger.error("Error reading log file: {}", e)

    def stats(self) -> dict:
        """Return processor statistics."""
        return {
            "running": self._running,
            "log_path": self._log_path,
            "detected_format": self._detected_format.value if self._detected_format else None,
            "lines_read": self._lines_read,
            "queue": self.queue.stats(),
        }
