"""
Background task: flush Redis usage counters to PostgreSQL every 30 seconds.
"""
import threading
import time
from loguru import logger

from backend.database import SessionLocal
from backend.services.usage_service import UsageService


class UsageFlushTask:
    def __init__(self):
        self.interval = 30  # Flush every 30 seconds
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Usage flush task started (interval={self.interval}s)")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Usage flush task stopped")

    def _run(self):
        while self.running:
            try:
                db = SessionLocal()
                try:
                    UsageService.flush_redis_to_db(db)
                finally:
                    db.close()
            except Exception as e:
                logger.exception(f"Usage flush task error: {e}")
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
