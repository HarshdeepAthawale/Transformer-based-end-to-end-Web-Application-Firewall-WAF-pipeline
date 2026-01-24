#!/usr/bin/env python3
"""
Model Rollback Script

Rollback to previous model version with hot-swap
"""
import sys
from pathlib import Path
import yaml
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.learning.version_manager import ModelVersionManager
from src.learning.hot_swap import HotSwapManager
from src.inference.async_waf_service import initialize_service, waf_service


def main():
    """Main entry point"""
    logger.info("Model Rollback")
    logger.info("=" * 60)
    
    # Load config
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get WAF config
    waf_config = config.get('waf_service', {})
    integration_config = config.get('integration', {}).get('waf_service', {})
    config_merged = {**waf_config, **integration_config}
    
    # Get paths
    model_path = project_root / config_merged.get('model_path', 'models/checkpoints/best_model.pt')
    vocab_path = project_root / config_merged.get('vocab_path', 'models/vocabularies/http_vocab.json')
    
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
            threshold=config_merged.get('threshold', 0.5),
            max_workers=config_merged.get('workers', 4),
            batch_size=32,
            timeout=config_merged.get('timeout', 5.0)
        )
    except Exception as e:
        logger.error(f"Failed to initialize WAF service: {e}")
        sys.exit(1)
    
    # Initialize version manager and hot-swap
    version_manager = ModelVersionManager()
    hot_swap = HotSwapManager(waf_service, version_manager)
    
    # Get current active version
    current_version = version_manager.get_active_version()
    if not current_version:
        logger.warning("No active version found. Cannot rollback.")
        sys.exit(1)
    
    logger.info(f"Current active version: {current_version}")
    
    # List all versions
    versions = version_manager.list_versions()
    logger.info(f"Available versions: {[v['version'] for v in versions]}")
    
    # Perform rollback
    logger.info("Performing rollback...")
    previous_version = version_manager.rollback()
    
    if not previous_version:
        logger.error("Rollback failed: No previous version available")
        sys.exit(1)
    
    logger.info(f"Rolled back to version: {previous_version}")
    
    # Hot-swap to rolled back version
    logger.info("Hot-swapping to rolled back version...")
    success = hot_swap.swap_model(previous_version)
    
    if success:
        logger.info("=" * 60)
        logger.info(f"Rollback completed successfully!")
        logger.info(f"Active version: {previous_version}")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error("Rollback failed: Hot-swap unsuccessful")
        logger.error("Version manager updated, but service not swapped")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
