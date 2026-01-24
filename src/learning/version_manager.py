"""
Model Version Manager Module

Manages model versions, deployments, and rollbacks
"""
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import json
import shutil
from loguru import logger


class ModelVersionManager:
    """Manage model versions and deployments"""
    
    def __init__(self, models_dir: str = "models/deployed"):
        """
        Initialize model version manager
        
        Args:
            models_dir: Directory to store versioned models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.models_dir / "versions.json"
        self.versions = self._load_versions()
        
        logger.info(f"ModelVersionManager initialized: models_dir={models_dir}")
    
    def _load_versions(self) -> Dict:
        """Load version information from file"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading versions file: {e}, starting fresh")
                return {}
        return {}
    
    def _save_versions(self):
        """Save version information to file"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def create_version(
        self,
        model_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create new model version
        
        Args:
            model_path: Path to model checkpoint
            metadata: Optional metadata about this version
        
        Returns:
            Version string (e.g., "v1.0.0")
        """
        # Generate version number
        version = self._generate_version()
        
        # Copy model to versioned location
        version_dir = self.models_dir / version
        version_dir.mkdir(exist_ok=True)
        
        model_file = version_dir / "model.pt"
        shutil.copy(model_path, model_file)
        
        # Save metadata
        version_info = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_path': str(model_file),
            'metadata': metadata or {},
            'status': 'pending',
            'is_active': False
        }
        
        self.versions[version] = version_info
        self._save_versions()
        
        logger.info(f"Created model version: {version}")
        return version
    
    def _generate_version(self) -> str:
        """Generate next version number"""
        if not self.versions:
            return "v1.0.0"
        
        # Get all version numbers
        version_numbers = []
        for v in self.versions.keys():
            try:
                # Extract version number (e.g., "v1.2.3" -> [1, 2, 3])
                parts = v[1:].split('.')
                if len(parts) == 3:
                    version_numbers.append([int(x) for x in parts])
            except:
                continue
        
        if not version_numbers:
            return "v1.0.0"
        
        # Find latest version
        latest = max(version_numbers, key=lambda x: (x[0], x[1], x[2]))
        
        # Increment patch version
        new_version = [latest[0], latest[1], latest[2] + 1]
        return f"v{'.'.join(map(str, new_version))}"
    
    def activate_version(self, version: str) -> bool:
        """
        Activate a model version
        
        Args:
            version: Version string to activate
        
        Returns:
            True if successful, False otherwise
        """
        if version not in self.versions:
            logger.error(f"Version {version} not found")
            return False
        
        # Deactivate current active version
        for v, info in self.versions.items():
            if info['is_active']:
                info['is_active'] = False
                info['status'] = 'inactive'
                info['deactivated_at'] = datetime.now().isoformat()
        
        # Activate new version
        self.versions[version]['is_active'] = True
        self.versions[version]['status'] = 'active'
        self.versions[version]['activated_at'] = datetime.now().isoformat()
        
        self._save_versions()
        logger.info(f"Activated model version: {version}")
        return True
    
    def get_active_version(self) -> Optional[str]:
        """Get currently active version"""
        for version, info in self.versions.items():
            if info.get('is_active', False):
                return version
        return None
    
    def get_version_path(self, version: str) -> Optional[str]:
        """Get model path for version"""
        if version in self.versions:
            return self.versions[version]['model_path']
        return None
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """Get information about a version"""
        return self.versions.get(version)
    
    def list_versions(self) -> List[Dict]:
        """List all versions with their status"""
        return [
            {
                'version': v,
                'status': info.get('status', 'unknown'),
                'is_active': info.get('is_active', False),
                'created_at': info.get('created_at'),
                'metadata': info.get('metadata', {})
            }
            for v, info in self.versions.items()
        ]
    
    def rollback(self) -> Optional[str]:
        """
        Rollback to previous active version
        
        Returns:
            Previous version string if successful, None otherwise
        """
        # Find current active version
        current_active = self.get_active_version()
        if not current_active:
            logger.warning("No active version to rollback from")
            return None
        
        # Find previous active versions (sorted by creation date)
        versions_by_date = sorted(
            [
                (info.get('created_at', ''), version, info)
                for version, info in self.versions.items()
                if version != current_active and info.get('status') in ['inactive', 'active']
            ],
            reverse=True
        )
        
        if not versions_by_date:
            logger.warning("No previous version to rollback to")
            return None
        
        # Get most recent previous version
        previous_version = versions_by_date[0][1]
        
        # Activate previous version
        if self.activate_version(previous_version):
            logger.info(f"Rolled back from {current_active} to {previous_version}")
            return previous_version
        
        return None
    
    def delete_version(self, version: str, force: bool = False) -> bool:
        """
        Delete a version (only if not active or force=True)
        
        Args:
            version: Version to delete
            force: Force deletion even if active
        
        Returns:
            True if successful, False otherwise
        """
        if version not in self.versions:
            logger.error(f"Version {version} not found")
            return False
        
        info = self.versions[version]
        
        if info.get('is_active', False) and not force:
            logger.error(f"Cannot delete active version {version}. Use force=True or rollback first.")
            return False
        
        # Delete model file
        model_path = Path(info['model_path'])
        if model_path.exists():
            model_path.unlink()
        
        # Delete version directory
        version_dir = model_path.parent
        if version_dir.exists() and version_dir.is_dir():
            try:
                version_dir.rmdir()
            except:
                pass  # Directory not empty, that's okay
        
        # Remove from versions
        del self.versions[version]
        self._save_versions()
        
        logger.info(f"Deleted version: {version}")
        return True
    
    def cleanup_old_versions(self, keep_versions: int = 5):
        """
        Clean up old versions, keeping only the most recent N
        
        Args:
            keep_versions: Number of versions to keep
        """
        # Get all versions sorted by creation date
        versions_by_date = sorted(
            [
                (info.get('created_at', ''), version, info)
                for version, info in self.versions.items()
            ],
            reverse=True
        )
        
        # Keep active version and most recent N-1
        active_version = self.get_active_version()
        versions_to_keep = set()
        
        if active_version:
            versions_to_keep.add(active_version)
        
        # Add most recent versions
        for _, version, _ in versions_by_date[:keep_versions]:
            versions_to_keep.add(version)
        
        # Delete old versions
        deleted_count = 0
        for _, version, _ in versions_by_date:
            if version not in versions_to_keep:
                if self.delete_version(version, force=True):
                    deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old versions")
