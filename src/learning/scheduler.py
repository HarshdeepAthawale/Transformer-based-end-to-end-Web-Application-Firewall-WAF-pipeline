"""
Update Scheduler Module

Schedules periodic model updates for continuous learning
"""
import schedule
import time
from threading import Thread
from loguru import logger
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.learning.data_collector import IncrementalDataCollector
from src.learning.fine_tuning import IncrementalFineTuner
from src.learning.version_manager import ModelVersionManager
from src.learning.validator import ModelValidator
from src.learning.hot_swap import HotSwapManager


class UpdateScheduler:
    """Schedule periodic model updates"""
    
    def __init__(
        self,
        log_path: str,
        base_model_path: str,
        vocab_path: str,
        validation_data: List[str],
        waf_service,
        update_interval_hours: int = 24,
        min_samples: int = 100,
        max_samples: Optional[int] = 5000,
        auto_deploy: bool = True,
        max_false_positive_rate: float = 0.05
    ):
        """
        Initialize update scheduler
        
        Args:
            log_path: Path to log files for data collection
            base_model_path: Path to base model for fine-tuning
            vocab_path: Path to vocabulary file
            validation_data: List of validation samples for testing
            waf_service: AsyncWAFService instance to update
            update_interval_hours: Hours between updates
            min_samples: Minimum samples required for update
            max_samples: Maximum samples to collect
            auto_deploy: Automatically deploy if validation passes
            max_false_positive_rate: Maximum allowed FPR for validation
        """
        self.log_path = log_path
        self.base_model_path = base_model_path
        self.vocab_path = vocab_path
        self.validation_data = validation_data
        self.waf_service = waf_service
        self.update_interval_hours = update_interval_hours
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.auto_deploy = auto_deploy
        self.max_false_positive_rate = max_false_positive_rate
        
        self.collector = IncrementalDataCollector(log_path, min_samples=min_samples)
        self.version_manager = ModelVersionManager()
        self.hot_swap = HotSwapManager(waf_service, self.version_manager)
        
        self.running = False
        self.thread = None
        self.update_count = 0
        
        logger.info(
            f"UpdateScheduler initialized: interval={update_interval_hours}h, "
            f"min_samples={min_samples}, auto_deploy={auto_deploy}"
        )
    
    def start(self):
        """Start scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        
        # Schedule updates
        schedule.every(self.update_interval_hours).hours.do(self._update_model)
        
        # Start scheduler thread
        self.thread = Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        logger.info(f"Scheduler started (update interval: {self.update_interval_hours} hours)")
    
    def stop(self):
        """Stop scheduler"""
        if not self.running:
            return
        
        self.running = False
        schedule.clear()
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        logger.info("Scheduler stopped")
    
    def _run_scheduler(self):
        """Run scheduler loop"""
        logger.info("Scheduler thread started")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
        
        logger.info("Scheduler thread stopped")
    
    def trigger_update(self) -> bool:
        """
        Manually trigger model update
        
        Returns:
            True if update was successful, False otherwise
        """
        logger.info("Manual update triggered")
        return self._update_model()
    
    def _update_model(self) -> bool:
        """Perform model update"""
        logger.info("=" * 60)
        logger.info("Starting scheduled model update...")
        logger.info("=" * 60)
        
        try:
            # 1. Collect new data
            logger.info("Step 1: Collecting new data...")
            incremental_data_dir = Path("data/incremental")
            incremental_data_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = incremental_data_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            new_data = self.collector.collect_new_data(
                output_path=str(output_path),
                max_samples=self.max_samples
            )
            
            if len(new_data) < self.min_samples:
                logger.warning(
                    f"Not enough new data for update: {len(new_data)} < {self.min_samples}. "
                    "Skipping update."
                )
                return False
            
            logger.info(f"Collected {len(new_data)} new samples")
            
            # 2. Fine-tune model
            logger.info("Step 2: Fine-tuning model...")
            fine_tuner = IncrementalFineTuner(
                base_model_path=self.base_model_path,
                vocab_path=self.vocab_path,
                device=self.waf_service.device
            )
            
            # Split data for validation if needed
            val_data = None
            if len(new_data) > 100:
                split_idx = int(len(new_data) * 0.9)
                train_data = new_data[:split_idx]
                val_data = new_data[split_idx:]
            else:
                train_data = new_data
            
            # Save fine-tuned model
            fine_tuned_path = Path("models/checkpoints/fine_tuned_latest.pt")
            fine_tuned_path.parent.mkdir(parents=True, exist_ok=True)
            
            fine_tuned_model = fine_tuner.fine_tune(
                new_data=train_data,
                validation_data=val_data,
                output_path=str(fine_tuned_path)
            )
            
            logger.info("Fine-tuning completed")
            
            # 3. Create new version
            logger.info("Step 3: Creating new model version...")
            version = self.version_manager.create_version(
                model_path=str(fine_tuned_path),
                metadata={
                    'samples': len(new_data),
                    'train_samples': len(train_data),
                    'val_samples': len(val_data) if val_data else 0,
                    'update_type': 'incremental',
                    'update_count': self.update_count + 1,
                    'base_model': self.base_model_path
                }
            )
            
            logger.info(f"Created version: {version}")
            
            # 4. Validate
            logger.info("Step 4: Validating model...")
            validator = ModelValidator(
                vocab_path=self.vocab_path,
                test_data=self.validation_data,
                threshold=self.waf_service.threshold,
                device=self.waf_service.device
            )
            
            model_path = self.version_manager.get_version_path(version)
            validation_result = validator.validate(
                model_path=model_path,
                max_false_positive_rate=self.max_false_positive_rate
            )
            
            # 5. Deploy if valid
            if validation_result['is_valid']:
                logger.info("Validation passed!")
                
                if self.auto_deploy:
                    logger.info("Step 5: Deploying new model (hot-swap)...")
                    success = self.hot_swap.swap_model(version)
                    
                    if success:
                        self.update_count += 1
                        logger.info("=" * 60)
                        logger.info(f"Model update completed successfully! Version: {version}")
                        logger.info("=" * 60)
                        return True
                    else:
                        logger.error("Hot-swap failed, but model is valid")
                        return False
                else:
                    logger.info("Auto-deploy disabled. Model validated but not deployed.")
                    logger.info(f"To deploy manually, use: hot_swap.swap_model('{version}')")
                    return True
            else:
                logger.warning(
                    f"Model validation failed: FPR={validation_result['false_positive_rate']:.4f} "
                    f"(max={self.max_false_positive_rate:.4f}). Update cancelled."
                )
                
                # Optional: Auto-rollback on validation failure
                # Uncomment to enable automatic rollback
                # logger.info("Attempting automatic rollback...")
                # rollback_version = self.version_manager.rollback()
                # if rollback_version:
                #     logger.info(f"Rolled back to version: {rollback_version}")
                #     self.hot_swap.swap_model(rollback_version)
                
                return False
                
        except Exception as e:
            logger.error(f"Error during model update: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_status(self) -> dict:
        """Get scheduler status"""
        return {
            'running': self.running,
            'update_interval_hours': self.update_interval_hours,
            'update_count': self.update_count,
            'next_update': schedule.next_run().isoformat() if schedule.jobs else None,
            'active_version': self.version_manager.get_active_version()
        }
