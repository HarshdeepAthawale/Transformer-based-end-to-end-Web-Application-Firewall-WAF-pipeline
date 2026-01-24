#!/usr/bin/env python3
"""
Example: Continuous Learning System

Demonstrates how to use the continuous learning components
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.learning.data_collector import IncrementalDataCollector
from src.learning.fine_tuning import IncrementalFineTuner
from src.learning.version_manager import ModelVersionManager
from src.learning.validator import ModelValidator
from src.learning.hot_swap import HotSwapManager
from src.learning.scheduler import UpdateScheduler
from src.inference.async_waf_service import AsyncWAFService
from loguru import logger


def example_data_collection():
    """Example: Collect incremental data"""
    logger.info("=== Example: Data Collection ===")
    
    collector = IncrementalDataCollector(
        log_path="logs/waf_access.log",
        min_samples=100
    )
    
    # Collect new data
    new_data = collector.collect_new_data(
        output_path="data/incremental/new_data.json",
        max_samples=1000
    )
    
    logger.info(f"Collected {len(new_data)} new samples")
    return new_data


def example_fine_tuning(new_data):
    """Example: Fine-tune model"""
    logger.info("=== Example: Fine-Tuning ===")
    
    if len(new_data) < 10:
        logger.warning("Not enough data for fine-tuning example")
        return None
    
    fine_tuner = IncrementalFineTuner(
        base_model_path="models/checkpoints/best_model.pt",
        vocab_path="models/vocabularies/http_vocab.json",
        learning_rate=1e-5,
        num_epochs=2,  # Fewer epochs for example
        batch_size=8
    )
    
    # Fine-tune
    fine_tuned_model = fine_tuner.fine_tune(
        new_data=new_data[:100],  # Use subset for example
        output_path="models/checkpoints/fine_tuned_example.pt"
    )
    
    logger.info("Fine-tuning completed")
    return "models/checkpoints/fine_tuned_example.pt"


def example_version_management(model_path):
    """Example: Version management"""
    logger.info("=== Example: Version Management ===")
    
    version_manager = ModelVersionManager(models_dir="models/deployed")
    
    # Create version
    version = version_manager.create_version(
        model_path=model_path,
        metadata={
            'example': True,
            'samples': 100
        }
    )
    
    logger.info(f"Created version: {version}")
    
    # List versions
    versions = version_manager.list_versions()
    logger.info(f"All versions: {versions}")
    
    # Activate version
    version_manager.activate_version(version)
    logger.info(f"Activated version: {version}")
    
    # Get active version
    active = version_manager.get_active_version()
    logger.info(f"Active version: {active}")
    
    return version


def example_validation():
    """Example: Model validation"""
    logger.info("=== Example: Model Validation ===")
    
    # Create test data (benign samples)
    test_data = [
        "GET /api/users HTTP/1.1",
        "POST /api/login HTTP/1.1",
        "GET /api/products?id=123 HTTP/1.1"
    ]
    
    validator = ModelValidator(
        vocab_path="models/vocabularies/http_vocab.json",
        test_data=test_data,
        threshold=0.5
    )
    
    # Validate model
    result = validator.validate(
        model_path="models/checkpoints/best_model.pt",
        max_false_positive_rate=0.05
    )
    
    logger.info(f"Validation result: {result}")
    return result


def example_hot_swap():
    """Example: Hot-swapping"""
    logger.info("=== Example: Hot-Swap ===")
    
    # Initialize WAF service
    waf_service = AsyncWAFService(
        model_path="models/checkpoints/best_model.pt",
        vocab_path="models/vocabularies/http_vocab.json",
        threshold=0.5
    )
    
    version_manager = ModelVersionManager()
    hot_swap = HotSwapManager(waf_service, version_manager)
    
    # Get active version
    active_version = version_manager.get_active_version()
    logger.info(f"Current active version: {active_version}")
    
    # List available versions
    versions = version_manager.list_versions()
    logger.info(f"Available versions: {[v['version'] for v in versions]}")
    
    # Example: Swap to a version (if available)
    if len(versions) > 1:
        # Get a different version
        other_version = [v['version'] for v in versions if v['version'] != active_version][0]
        logger.info(f"Swapping to version: {other_version}")
        success = hot_swap.swap_model(other_version)
        logger.info(f"Swap successful: {success}")
    
    # Cleanup
    waf_service.shutdown()


def example_scheduler():
    """Example: Update scheduler"""
    logger.info("=== Example: Update Scheduler ===")
    
    # Initialize WAF service
    waf_service = AsyncWAFService(
        model_path="models/checkpoints/best_model.pt",
        vocab_path="models/vocabularies/http_vocab.json",
        threshold=0.5
    )
    
    # Create scheduler
    scheduler = UpdateScheduler(
        log_path="logs/waf_access.log",
        base_model_path="models/checkpoints/best_model.pt",
        vocab_path="models/vocabularies/http_vocab.json",
        validation_data=[],  # Empty for example
        waf_service=waf_service,
        update_interval_hours=24,
        min_samples=100,
        max_samples=1000,
        auto_deploy=False  # Don't auto-deploy in example
    )
    
    # Get status
    status = scheduler.get_status()
    logger.info(f"Scheduler status: {status}")
    
    # Note: In real usage, call scheduler.start() to begin automated updates
    # For example, we'll just show the status
    
    # Cleanup
    waf_service.shutdown()


def main():
    """Run all examples"""
    logger.info("Continuous Learning System Examples")
    logger.info("=" * 60)
    
    # Check if model exists
    model_path = project_root / "models" / "checkpoints" / "best_model.pt"
    vocab_path = project_root / "models" / "vocabularies" / "http_vocab.json"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train a model first")
        return
    
    if not vocab_path.exists():
        logger.error(f"Vocabulary not found: {vocab_path}")
        logger.error("Please generate vocabulary first")
        return
    
    try:
        # Example 1: Data collection
        new_data = example_data_collection()
        
        # Example 2: Fine-tuning (if we have data)
        if new_data and len(new_data) >= 10:
            fine_tuned_path = example_fine_tuning(new_data)
            
            # Example 3: Version management
            if fine_tuned_path:
                version = example_version_management(fine_tuned_path)
        
        # Example 4: Validation
        example_validation()
        
        # Example 5: Hot-swap
        example_hot_swap()
        
        # Example 6: Scheduler
        example_scheduler()
        
        logger.info("=" * 60)
        logger.info("All examples completed!")
        
    except Exception as e:
        logger.error(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
