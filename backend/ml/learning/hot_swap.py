"""
Hot Swap Manager

Hot-swap models without service downtime.
"""
import threading
from typing import Optional
from loguru import logger

from backend.ml.model.anomaly_detector import AnomalyDetector
from backend.ml.model.scoring import AnomalyScorer
from backend.ml.tokenization.tokenizer import HTTPTokenizer
from backend.ml.tokenization.sequence_prep import SequencePreparator
from .version_manager import ModelVersionManager


class HotSwapManager:
    """Manage hot-swapping of models"""
    
    def __init__(
        self,
        waf_service,
        version_manager: ModelVersionManager
    ):
        """
        Initialize hot-swap manager
        
        Args:
            waf_service: WAFService instance to update
            version_manager: ModelVersionManager instance
        """
        self.waf_service = waf_service
        self.version_manager = version_manager
        self.swap_lock = threading.Lock()
    
    def swap_model(
        self,
        version_id: str,
        vocab_path: str
    ) -> bool:
        """
        Hot-swap model to new version
        
        Args:
            version_id: Version ID to swap to
            vocab_path: Path to vocabulary file
        
        Returns:
            True if swap successful
        """
        with self.swap_lock:
            try:
                logger.info(f"Hot-swapping to version {version_id}...")
                
                # Get model path
                model_path = self.version_manager.get_version_path(version_id)
                if not model_path:
                    logger.error(f"Model path not found for version {version_id}")
                    return False
                
                # Load new model
                device = self.waf_service.device
                new_model = AnomalyDetector.load_checkpoint(model_path, device=device)
                
                # Load tokenizer
                new_tokenizer = HTTPTokenizer()
                new_tokenizer.load_vocab(vocab_path)
                
                # Create new components
                new_preparator = SequencePreparator(new_tokenizer)
                new_scorer = AnomalyScorer(new_model, threshold=self.waf_service.threshold, device=device)
                
                # Swap components atomically
                old_model = self.waf_service.model
                self.waf_service.model = new_model
                self.waf_service.tokenizer = new_tokenizer
                self.waf_service.preparator = new_preparator
                self.waf_service.scorer = new_scorer
                self.waf_service.ml_enabled = True
                
                # Clean up old model (let GC handle it)
                del old_model
                
                # Update version manager
                self.version_manager.set_current_version(version_id)
                
                logger.info(f"Successfully hot-swapped to version {version_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to hot-swap model: {e}", exc_info=True)
                return False
    
    def rollback(self, vocab_path: str) -> bool:
        """
        Rollback to previous version
        
        Args:
            vocab_path: Path to vocabulary file
        
        Returns:
            True if rollback successful
        """
        versions = self.version_manager.list_versions()
        if len(versions) < 2:
            logger.warning("Not enough versions for rollback")
            return False
        
        # Get previous version (second latest)
        sorted_versions = sorted(
            versions,
            key=lambda x: x['created_at'],
            reverse=True
        )
        
        previous_version_id = sorted_versions[1]['version_id']
        logger.info(f"Rolling back to version {previous_version_id}")
        
        return self.swap_model(previous_version_id, vocab_path)
