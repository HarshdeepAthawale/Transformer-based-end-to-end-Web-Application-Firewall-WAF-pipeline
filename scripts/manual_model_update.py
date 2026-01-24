#!/usr/bin/env python3
"""
Manual Model Update Script

Manually trigger model update without scheduler
"""
import sys
from pathlib import Path
import yaml
import json
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.learning.scheduler import UpdateScheduler
from src.inference.async_waf_service import initialize_service, waf_service


def load_config():
    """Load configuration"""
    config_path = project_root / "config" / "config.yaml"
    learning_config_path = project_root / "config" / "learning.yaml"
    
    with open(config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    learning_config = {}
    if learning_config_path.exists():
        with open(learning_config_path, 'r') as f:
            learning_config = yaml.safe_load(f).get('learning', {})
    
    return main_config, learning_config


def load_validation_data(validation_path: str) -> list:
    """Load validation data"""
    validation_path = Path(validation_path)
    if not validation_path.exists():
        return []
    
    try:
        with open(validation_path, 'r') as f:
            if validation_path.suffix == '.json':
                data = json.load(f)
                return data if isinstance(data, list) else (data.get('data', []) if isinstance(data, dict) else [])
            else:
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        return []


def main():
    """Main entry point"""
    logger.info("Manual Model Update")
    logger.info("=" * 60)
    
    # Load config
    main_config, learning_config = load_config()
    
    # Get WAF config
    waf_config = main_config.get('waf_service', {})
    integration_config = main_config.get('integration', {}).get('waf_service', {})
    config = {**waf_config, **integration_config}
    
    # Get paths
    model_path = project_root / config.get('model_path', 'models/checkpoints/best_model.pt')
    vocab_path = project_root / config.get('vocab_path', 'models/vocabularies/http_vocab.json')
    
    # Check files
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    if not vocab_path.exists():
        logger.error(f"Vocab not found: {vocab_path}")
        sys.exit(1)
    
    # Initialize WAF service
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
    except Exception as e:
        logger.error(f"Failed to initialize WAF service: {e}")
        sys.exit(1)
    
    # Get learning config
    incremental_config = learning_config.get('incremental', {})
    validation_config = learning_config.get('validation', {})
    scheduling_config = learning_config.get('scheduling', {})
    
    # Load validation data
    validation_data_path = validation_config.get('test_data_path', 'data/validation/test_set.json')
    validation_data = load_validation_data(validation_data_path)
    
    # Create scheduler (but don't start it)
    scheduler = UpdateScheduler(
        log_path=scheduling_config.get('log_path', 'logs/waf_access.log'),
        base_model_path=str(model_path),
        vocab_path=str(vocab_path),
        validation_data=validation_data,
        waf_service=waf_service,
        update_interval_hours=24,  # Not used for manual update
        min_samples=incremental_config.get('min_samples', 1000),
        max_samples=incremental_config.get('max_samples', 10000),
        auto_deploy=scheduling_config.get('auto_deploy', True),
        max_false_positive_rate=validation_config.get('max_false_positive_rate', 0.05)
    )
    
    # Trigger update
    logger.info("Triggering manual update...")
    success = scheduler.trigger_update()
    
    if success:
        logger.info("=" * 60)
        logger.info("Manual update completed successfully!")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error("Manual update failed!")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
