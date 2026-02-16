"""Scheduler for periodic model updates (continuous learning)."""

import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.ml.learning.data_collector import IncrementalDataCollector
from backend.ml.learning.fine_tuning import IncrementalFineTuner
from backend.ml.learning.version_manager import ModelVersionManager
from backend.ml.learning.validator import ModelValidator
from backend.ml.learning.hot_swap import HotSwapManager


class LearningScheduler:
    """Run incremental learning pipeline on a schedule."""

    def __init__(
        self,
        log_path: str,
        model_path: str = "models/waf-distilbert",
        update_interval_hours: int = 24,
        min_samples: int = 500,
        incremental_dir: str = "data/incremental",
    ):
        self.log_path = log_path
        self.model_path = model_path
        self.update_interval_hours = update_interval_hours
        self.min_samples = min_samples
        self.incremental_dir = Path(incremental_dir)
        self.incremental_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the scheduler thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Learning scheduler started (interval: {self.update_interval_hours}h)"
        )

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Learning scheduler stopped")

    def _run_loop(self):
        """Main loop: wait, then run update."""
        interval_sec = self.update_interval_hours * 3600
        while self._running:
            time.sleep(min(60, interval_sec))
            if not self._running:
                break
            # Run update every N hours (simplified: run after first full interval)
            elapsed = 0
            while elapsed < interval_sec and self._running:
                time.sleep(60)
                elapsed += 60
            if self._running:
                self._run_update()

    def _run_update(self):
        """Execute one incremental learning cycle."""
        logger.info("Starting scheduled model update...")
        try:
            collector = IncrementalDataCollector(
                log_path=self.log_path,
                min_samples=self.min_samples,
            )
            data_path = self.incremental_dir / f"collected_{int(time.time())}.json"
            texts = collector.collect_new_data(
                output_path=str(data_path),
                max_samples=5000,
            )

            if len(texts) < self.min_samples:
                logger.warning(f"Not enough new data: {len(texts)} < {self.min_samples}")
                return

            tuner = IncrementalFineTuner(
                base_model_path=self.model_path,
                output_dir="models/waf-distilbert-incremental",
            )
            out_path = tuner.fine_tune(texts)
            if not out_path:
                return

            version_mgr = ModelVersionManager()
            version_id = version_mgr.create_version(
                str(out_path),
                metadata={"samples": len(texts), "type": "incremental"},
            )

            validator = ModelValidator(
                model_path=str(out_path),
                min_detection_rate=0.6,
            )
            result = validator.validate()
            if result.get("is_valid", False):
                hot_swap = HotSwapManager(version_mgr)
                if hot_swap.swap_model(version_id):
                    logger.info("Model update deployed successfully")
                else:
                    logger.warning("Hot swap failed, model not deployed")
            else:
                logger.warning(
                    f"Validation failed (rate={result.get('detection_rate', 0):.2f}), update cancelled"
                )
        except Exception as e:
            logger.error(f"Model update failed: {e}")
