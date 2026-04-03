"""
Background task: run alert rule evaluator every ALERT_EVALUATION_INTERVAL_SECONDS (Feature 10).
"""
import threading
import time
from loguru import logger

from backend.config import config
from backend.database import SessionLocal
from backend.services.alert_evaluator import run_evaluator_once


class AlertEvaluatorTask:
    def __init__(self):
        self.interval = getattr(config, "ALERT_EVALUATION_INTERVAL_SECONDS", 60)
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Alert evaluator task started (interval=%ss)", self.interval)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Alert evaluator task stopped")

    def _run(self):
        while self.running:
            try:
                db = SessionLocal()
                try:
                    from backend.models.organization import Organization
                    orgs = db.query(Organization).filter(Organization.is_active == True).all()
                    for org in orgs:
                        run_evaluator_once(db, org.id)
                finally:
                    db.close()
            except Exception as e:
                logger.exception("Alert evaluator task error: %s", e)
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
