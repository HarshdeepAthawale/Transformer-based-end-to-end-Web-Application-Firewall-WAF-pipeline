"""Background task: sync managed rules from feed on an interval."""
import threading
from loguru import logger

from backend.config import config
from backend.database import SessionLocal
from backend.services import managed_rules_sync


class ManagedRulesSyncTask:
    """Runs managed rules sync every MANAGED_RULES_UPDATE_INTERVAL_HOURS."""

    def __init__(self, interval_hours: int | None = None):
        self.interval_hours = interval_hours or config.MANAGED_RULES_UPDATE_INTERVAL_HOURS
        self.interval_seconds = max(3600, self.interval_hours * 3600)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run_once(self) -> None:
        url = (config.MANAGED_RULES_FEED_URL or "").strip()
        if not url:
            return
        db = SessionLocal()
        try:
            result = managed_rules_sync.sync_pack(
                db,
                pack_id=config.MANAGED_RULES_PACK_ID,
                name=f"Managed ({config.MANAGED_RULES_PACK_ID})",
                source_url=url,
                feed_format=config.MANAGED_RULES_FEED_FORMAT or "json",
                auth_header=config.MANAGED_RULES_FEED_HEADER,
            )
            if result.get("error"):
                logger.warning(f"Managed rules sync failed: {result['error']}")
            else:
                logger.info(f"Managed rules sync ok: {result}")
        except Exception as e:
            logger.exception(f"Managed rules sync error: {e}")
        finally:
            db.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._run_once()
            if self._stop.wait(timeout=self.interval_seconds):
                break

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"Managed rules sync task started (interval={self.interval_hours}h)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("Managed rules sync task stopped")
