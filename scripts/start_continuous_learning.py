#!/usr/bin/env python3
"""
Start Continuous Learning System

Starts the update scheduler for automated incremental learning
"""
import sys
import signal
from pathlib import Path
import yaml
import json
from loguru import logger
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.learning.scheduler import UpdateScheduler
from src.learning.data_collector import IncrementalDataCollector
from src.inference.async_waf_service import initialize_service, waf_service


def load_config():
    """Load configuration from files"""
    config_path = project_root / "config" / "config.yaml"
    learning_config_path = project_root / "config" / "learning.yaml"
    
    # Load main config
    with open(config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Load learning config
    learning_config = {}
    if learning_config_path.exists():
        with open(learning_config_path, 'r') as f:
            learning_config = yaml.safe_load(f).get('learning', {})
    
    return main_config, learning_config


def load_validation_data(validation_path: str) -> list:
    """Load validation/test data"""
    validation_path = Path(validation_path)
    
    if not validation_path.exists():
        logger.warning(f"Validation data not found: {validation_path}")
        logger.info("Using empty validation set (will skip validation)")
        return []
    
    try:
        with open(validation_path, 'r') as f:
            if validation_path.suffix == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    logger.warning(f"Unexpected validation data format")
                    return []
            else:
                # Assume text file, one sample per line
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        return []


def main():
    """Main entry point"""
    logger.info("Starting Continuous Learning System...")
    
    # Load configuration
    main_config, learning_config = load_config()
    
    # Get WAF service config
    waf_config = main_config.get('waf_service', {})
    integration_config = main_config.get('integration', {}).get('waf_service', {})
    config = {**waf_config, **integration_config}
    
    # Get paths
    model_path = config.get('model_path', 'models/checkpoints/best_model.pt')
    vocab_path = config.get('vocab_path', 'models/vocabularies/http_vocab.json')
    
    # Resolve paths
    model_path = project_root / model_path
    vocab_path = project_root / vocab_path
    
    # Check if files exist
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        logger.error("Please train a model first")
        sys.exit(1)
    
    if not vocab_path.exists():
        logger.error(f"Vocabulary file not found: {vocab_path}")
        logger.error("Please generate vocabulary first")
        sys.exit(1)
    
    # Initialize WAF service (required for hot-swapping)
    logger.info("Initializing WAF service...")
    try:
        initialize_service(
            model_path=str(model_path),
            vocab_path=str(vocab_path),
            threshold=config.get('threshold', 0.5),
            max_workers=config.get('workers', 4),
            batch_size=32,
            timeout=config.get('timeout', 5.0)
        )
        logger.info("WAF service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize WAF service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get learning config
    incremental_config = learning_config.get('incremental', {})
    fine_tuning_config = learning_config.get('fine_tuning', {})
    validation_config = learning_config.get('validation', {})
    scheduling_config = learning_config.get('scheduling', {})
    
    # Load validation data
    validation_data_path = validation_config.get('test_data_path', 'data/validation/test_set.json')
    validation_data = load_validation_data(validation_data_path)
    
    if not validation_data:
        logger.warning("No validation data loaded. Validation will be skipped.")
    
    # Get log path
    log_path = scheduling_config.get('log_path', 'logs/waf_access.log')
    
    # Create scheduler
    scheduler = UpdateScheduler(
        log_path=log_path,
        base_model_path=str(model_path),
        vocab_path=str(vocab_path),
        validation_data=validation_data,
        waf_service=waf_service,
        update_interval_hours=scheduling_config.get('update_interval_hours', 24),
        min_samples=incremental_config.get('min_samples', 1000),
        max_samples=incremental_config.get('max_samples', 10000),
        auto_deploy=scheduling_config.get('auto_deploy', True),
        max_false_positive_rate=validation_config.get('max_false_positive_rate', 0.05)
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received...")
        scheduler.stop()
        if waf_service:
            waf_service.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start scheduler
    if scheduling_config.get('enabled', True):
        scheduler.start()
        logger.info("Continuous learning scheduler started")
        logger.info(f"Update interval: {scheduling_config.get('update_interval_hours', 24)} hours")
        logger.info(f"Auto-deploy: {scheduling_config.get('auto_deploy', True)}")
        
        # Print status
        status = scheduler.get_status()
        logger.info(f"Scheduler status: {status}")
        
        # Keep running
        try:
            while True:
                import time
                time.sleep(60)  # Sleep and check status periodically
                # Optionally print status every hour
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            scheduler.stop()
    else:
        logger.info("Scheduler is disabled in config. Use trigger_update() to run manually.")
        logger.info("Press Ctrl+C to exit")
        try:
            while True:
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
    
    if waf_service:
        waf_service.shutdown()


if __name__ == "__main__":
    main()
