"""Hot swap manager for deploying new model versions without downtime."""

from typing import Optional

from loguru import logger

from backend.core.waf_factory import reload_waf_service
from backend.ml.learning.version_manager import ModelVersionManager


class HotSwapManager:
    """Deploy new model version by reloading WAF classifier."""

    def __init__(self, version_manager: Optional[ModelVersionManager] = None):
        self.version_manager = version_manager or ModelVersionManager()

    def swap_model(self, version_id: str) -> bool:
        """Load new model version and reload WAF classifier. Returns True on success."""
        path = self.version_manager.get_version_path(version_id)
        if not path.exists():
            logger.error(f"Version {version_id} not found at {path}")
            return False

        try:
            classifier = reload_waf_service(str(path))
            if classifier and classifier.is_loaded:
                self.version_manager.set_current_version(version_id)
                logger.info(f"Hot swap complete: model {version_id} deployed")
                return True
            return False
        except Exception as e:
            logger.error(f"Hot swap failed: {e}")
            return False
