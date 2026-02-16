"""Model versioning for continuous learning."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class ModelVersionManager:
    """Manage model versions: create, list, rollback."""

    def __init__(
        self,
        versions_dir: str = "models/versions",
        metadata_file: str = "version_metadata.json",
    ):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.versions_dir / metadata_file
        self._metadata: Dict[str, Any] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {"versions": [], "current": None}

    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def create_version(
        self,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Copy model to versions dir with timestamp, return version id."""
        src = Path(model_path)
        if not src.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        version_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest = self.versions_dir / version_id
        dest.mkdir(parents=True, exist_ok=True)

        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)

        entry = {
            "id": version_id,
            "created_at": datetime.utcnow().isoformat(),
            "source": str(src),
            "metadata": metadata or {},
        }
        self._metadata["versions"].append(entry)
        self._metadata["current"] = version_id
        self._save_metadata()
        logger.info(f"Created version {version_id} at {dest}")
        return version_id

    def get_version_path(self, version_id: str) -> Path:
        """Get filesystem path for a version."""
        return self.versions_dir / version_id

    def get_current_version(self) -> Optional[str]:
        """Get current active version id."""
        return self._metadata.get("current")

    def set_current_version(self, version_id: str):
        """Set current active version."""
        vpath = self.get_version_path(version_id)
        if not vpath.exists():
            raise FileNotFoundError(f"Version {version_id} not found")
        self._metadata["current"] = version_id
        self._save_metadata()

    def list_versions(self) -> list:
        """List all versions (newest first)."""
        return list(reversed(self._metadata.get("versions", [])))

    def rollback(self) -> Optional[str]:
        """Rollback to previous version if exists."""
        versions = self._metadata.get("versions", [])
        if len(versions) < 2:
            return None
        prev = versions[-2]
        prev_id = prev["id"]
        self.set_current_version(prev_id)
        return prev_id
