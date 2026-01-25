"""
Model Version Manager

Manage model versions and metadata.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from loguru import logger


class ModelVersionManager:
    """Manage model versions"""
    
    def __init__(self, versions_dir: str = "models/versions"):
        """
        Initialize version manager
        
        Args:
            versions_dir: Directory to store version metadata
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.versions_dir / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict:
        """Load version metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self):
        """Save version metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def create_version(
        self,
        model_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new model version
        
        Args:
            model_path: Path to model checkpoint
            metadata: Additional metadata
        
        Returns:
            Version ID
        """
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_info = {
            'version_id': version_id,
            'model_path': model_path,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.versions[version_id] = version_info
        self._save_versions()
        
        logger.info(f"Created model version: {version_id}")
        return version_id
    
    def get_version(self, version_id: str) -> Optional[Dict]:
        """Get version information"""
        return self.versions.get(version_id)
    
    def get_version_path(self, version_id: str) -> Optional[str]:
        """Get model path for version"""
        version = self.get_version(version_id)
        if version:
            return version['model_path']
        return None
    
    def list_versions(self) -> List[Dict]:
        """List all versions"""
        return list(self.versions.values())
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest version ID"""
        if not self.versions:
            return None
        
        versions = sorted(
            self.versions.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )
        return versions[0][0]
    
    def set_current_version(self, version_id: str):
        """Set current active version"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # Update metadata
        for v_id in self.versions:
            self.versions[v_id]['metadata']['is_current'] = (v_id == version_id)
        
        self._save_versions()
        logger.info(f"Set current version to: {version_id}")
