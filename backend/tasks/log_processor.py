"""
Log Processor - Background task for processing WAF logs
"""
import threading
import time
from typing import Optional
from loguru import logger


class LogProcessor:
    """
    Background worker that processes WAF logs asynchronously.
    Handles log aggregation, analysis, and storage.
    """

    def __init__(self, interval_seconds: int = 10):
        """
        Initialize the log processor.

        Args:
            interval_seconds: Processing interval in seconds
        """
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the log processor background thread."""
        if self._running:
            logger.warning("Log processor is already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Log processor started (interval: {self.interval}s)")

    def stop(self):
        """Stop the log processor."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Log processor stopped")

    def _run(self):
        """Main processing loop."""
        while self._running:
            try:
                self._process_logs()
            except Exception as e:
                logger.error(f"Log processing error: {e}")
            time.sleep(self.interval)

    def _process_logs(self):
        """Process pending logs."""
        # Log processing is handled inline by the WAF middleware
        # This is a placeholder for additional batch processing if needed
        pass
