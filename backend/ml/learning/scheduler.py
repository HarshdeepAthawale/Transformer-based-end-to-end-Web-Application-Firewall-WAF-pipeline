"""
Update Scheduler

Schedule periodic model updates.
"""
import time
import threading
from datetime import datetime
from loguru import logger

from .data_collector import IncrementalDataCollector
from .fine_tuning import IncrementalFineTuner
from .version_manager import ModelVersionManager
from .validator import ModelValidator
from .hot_swap import HotSwapManager


class UpdateScheduler:
    """Schedule periodic model updates"""
    
    def __init__(
        self,
        log_path: str,
        base_model_path: str,
        vocab_path: str,
        validation_data: list,
        waf_service,
        update_interval_hours: int = 24
    ):
        """
        Initialize scheduler
        
        Args:
            log_path: Path to log file
            base_model_path: Path to base model
            vocab_path: Path to vocabulary
            validation_data: Validation data for model validation
            waf_service: WAFService instance
            update_interval_hours: Update interval in hours
        """
        self.log_path = log_path
        self.base_model_path = base_model_path
        self.vocab_path = vocab_path
        self.validation_data = validation_data
        self.waf_service = waf_service
        self.update_interval_hours = update_interval_hours
        
        self.collector = IncrementalDataCollector(log_path)
        self.version_manager = ModelVersionManager()
        self.hot_swap = HotSwapManager(waf_service, self.version_manager)
        
        self.running = False
        self.thread = None
        self.last_update = None
    
    def start(self):
        """Start scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        logger.info(f"Scheduler started (update interval: {self.update_interval_hours} hours)")
    
    def stop(self):
        """Stop scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Scheduler stopped")
    
    def _run_scheduler(self):
        """Run scheduler loop"""
        while self.running:
            # Check if it's time for update
            if self._should_update():
                self._update_model()
                self.last_update = datetime.now()
            
            # Sleep for 1 hour before checking again
            time.sleep(3600)
    
    def _should_update(self) -> bool:
        """Check if update should be performed"""
        if self.last_update is None:
            return True
        
        time_since_update = (datetime.now() - self.last_update).total_seconds() / 3600
        return time_since_update >= self.update_interval_hours
    
    def _update_model(self):
        """Perform model update"""
        logger.info("Starting scheduled model update...")
        
        try:
            # 1. Collect new data
            output_path = f"data/incremental/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            new_data = self.collector.collect_new_data(
                output_path=output_path,
                max_samples=5000
            )
            
            if len(new_data) < 100:  # Minimum samples
                logger.warning("Not enough new data for update")
                return
            
            # 2. Fine-tune model
            fine_tuner = IncrementalFineTuner(
                base_model_path=self.base_model_path,
                vocab_path=self.vocab_path
            )
            
            fine_tuned_path = "models/checkpoints/fine_tuned_latest.pt"
            fine_tuned_model = fine_tuner.fine_tune(
                new_data,
                validation_data=self.validation_data[:1000] if self.validation_data else None,
                output_path=fine_tuned_path
            )
            
            # 3. Save new version
            version = self.version_manager.create_version(
                model_path=fine_tuned_path,
                metadata={
                    'samples': len(new_data),
                    'update_type': 'incremental'
                }
            )
            
            # 4. Validate
            validator = ModelValidator(
                vocab_path=self.vocab_path,
                test_data=self.validation_data
            )
            
            validation_result = validator.validate(
                self.version_manager.get_version_path(version)
            )
            
            # 5. Deploy if valid
            if validation_result['is_valid']:
                self.hot_swap.swap_model(version, self.vocab_path)
                logger.info("Model update completed successfully")
            else:
                logger.warning("Model validation failed, update cancelled")
                
        except Exception as e:
            logger.error(f"Error during model update: {e}", exc_info=True)
