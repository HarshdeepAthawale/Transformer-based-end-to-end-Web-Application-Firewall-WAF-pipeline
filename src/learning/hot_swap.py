"""
Hot-Swap Manager Module

Manages hot-swapping of models without service interruption
"""
from typing import Optional
from loguru import logger
import threading
import time

from src.inference.async_waf_service import AsyncWAFService
from src.learning.version_manager import ModelVersionManager


class HotSwapManager:
    """Manage hot-swapping of models without service interruption"""
    
    def __init__(
        self,
        waf_service: AsyncWAFService,
        version_manager: ModelVersionManager
    ):
        """
        Initialize hot-swap manager
        
        Args:
            waf_service: AsyncWAFService instance to swap
            version_manager: ModelVersionManager instance
        """
        self.waf_service = waf_service
        self.version_manager = version_manager
        self.swap_lock = threading.Lock()
        self.last_swap_time = None
        
        logger.info("HotSwapManager initialized")
    
    def swap_model(self, new_version: str) -> bool:
        """
        Hot-swap to new model version
        
        Args:
            new_version: Version string to swap to
        
        Returns:
            True if successful, False otherwise
        """
        with self.swap_lock:
            logger.info(f"Hot-swapping to version: {new_version}")
            
            try:
                # Get model path
                model_path = self.version_manager.get_version_path(new_version)
                if not model_path:
                    logger.error(f"Model path not found for version: {new_version}")
                    return False
                
                # Get vocab path from config (tokenizer doesn't store path)
                from pathlib import Path
                import yaml
                vocab_path = None
                config_path = Path("config/config.yaml")
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        vocab_path = config.get('waf_service', {}).get('vocab_path', 'models/vocabularies/http_vocab.json')
                        # Also check integration config
                        if not vocab_path or vocab_path == 'models/vocabularies/http_vocab.json':
                            vocab_path = config.get('integration', {}).get('waf_service', {}).get('vocab_path', vocab_path)
                else:
                    vocab_path = 'models/vocabularies/http_vocab.json'
                
                # Resolve relative path
                if not Path(vocab_path).is_absolute():
                    vocab_path = str(Path("config/config.yaml").parent.parent / vocab_path)
                
                # Get threshold from current service
                threshold = self.waf_service.threshold
                
                # Get device from current service
                device = self.waf_service.device
                
                # Get other config from current service
                max_workers = self.waf_service.executor._max_workers if hasattr(self.waf_service.executor, '_max_workers') else 4
                batch_size = self.waf_service.batch_size
                timeout = self.waf_service.timeout
                
                # Create new service instance with new model
                logger.info(f"Loading new model from {model_path}")
                new_service = AsyncWAFService(
                    model_path=model_path,
                    vocab_path=vocab_path,
                    threshold=threshold,
                    device=device,
                    max_workers=max_workers,
                    batch_size=batch_size,
                    timeout=timeout,
                    use_queue_manager=bool(self.waf_service.queue_manager),
                    queue_max_size=self.waf_service.queue_manager.queue.maxsize if self.waf_service.queue_manager else 1000,
                    queue_batch_timeout=self.waf_service.queue_manager.batch_timeout if self.waf_service.queue_manager else 0.1,
                    anomaly_log_file=getattr(self.waf_service, 'anomaly_log_file', None)
                )
                
                # Atomic swap
                logger.info("Performing atomic model swap...")
                old_service = self.waf_service
                self.waf_service = new_service
                
                # Update global service reference (if it matches)
                try:
                    import src.inference.async_waf_service as waf_module
                    if waf_module.waf_service is old_service:
                        waf_module.waf_service = new_service
                        logger.info("Updated global waf_service reference")
                except Exception as e:
                    logger.debug(f"Could not update global service reference: {e}")
                
                # Cleanup old service (shutdown executor)
                try:
                    old_service.shutdown()
                except:
                    pass
                
                # Update active version
                self.version_manager.activate_version(new_version)
                
                self.last_swap_time = time.time()
                
                logger.info(f"Successfully swapped to version: {new_version}")
                return True
                
            except Exception as e:
                logger.error(f"Error during hot-swap: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def get_service(self) -> AsyncWAFService:
        """Get current WAF service instance"""
        return self.waf_service
    
    def get_last_swap_time(self) -> Optional[float]:
        """Get timestamp of last swap"""
        return self.last_swap_time
    
    def can_swap(self) -> bool:
        """Check if swap is currently possible (not locked)"""
        return self.swap_lock.acquire(blocking=False)
