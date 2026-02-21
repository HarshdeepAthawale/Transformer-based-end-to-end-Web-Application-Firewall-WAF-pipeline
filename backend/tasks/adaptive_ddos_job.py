"""
Periodic job: recompute adaptive DDoS threshold and write to Redis.
Runs every ADAPTIVE_DDOS_UPDATE_INTERVAL_MINUTES.
"""

import threading
import time
from loguru import logger

from backend.config import config
from backend.services.adaptive_ddos_service import compute_and_publish_threshold


class AdaptiveDDoSJob:
    """Run adaptive threshold computation on an interval."""

    def __init__(self):
        self.interval_minutes = max(1, getattr(config, "ADAPTIVE_DDOS_UPDATE_INTERVAL_MINUTES", 15))
        self.interval_seconds = self.interval_minutes * 60
        self.running = False
        self.thread = None

    def start(self):
        if not getattr(config, "ADAPTIVE_DDOS_ENABLED", False):
            logger.debug("Adaptive DDoS disabled; job not started")
            return
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Adaptive DDoS job started (interval={self.interval_minutes} min)")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Adaptive DDoS job stopped")

    def _run(self):
        while self.running:
            try:
                compute_and_publish_threshold()
            except Exception as e:
                logger.warning(f"Adaptive DDoS job error: {e}")
            for _ in range(self.interval_seconds):
                if not self.running:
                    return
                time.sleep(1)
