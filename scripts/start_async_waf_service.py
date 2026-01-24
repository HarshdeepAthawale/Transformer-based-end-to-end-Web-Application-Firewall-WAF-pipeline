#!/usr/bin/env python3
"""
Startup script for Async WAF Service

Initializes and runs the async WAF service with configuration from config files
"""
import sys
import os
from pathlib import Path
import yaml
import uvicorn
from loguru import logger
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.async_waf_service import initialize_service, initialize_rate_limiting, app


def load_config():
    """Load configuration from config files"""
    config_path = project_root / "config" / "config.yaml"
    inference_config_path = project_root / "config" / "inference.yaml"
    
    # Load main config
    with open(config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Load inference config
    inference_config = {}
    if inference_config_path.exists():
        with open(inference_config_path, 'r') as f:
            inference_config = yaml.safe_load(f).get('inference', {})
    
    # Merge configs
    waf_config = main_config.get('waf_service', {})
    integration_config = main_config.get('integration', {}).get('waf_service', {})
    
    # Use integration config if available, otherwise use waf_service config
    config = {**waf_config, **integration_config}
    
    # Override with inference config if present
    async_config = inference_config.get('async', {})
    if async_config:
        config.update({
            'max_workers': async_config.get('max_workers', config.get('workers', 4)),
            'batch_size': async_config.get('batch_size', 32),
            'timeout': async_config.get('timeout', config.get('timeout', 5.0))
        })
    
    return config, inference_config


def determine_device(inference_config: dict) -> str:
    """Determine device to use"""
    device_config = inference_config.get('device', {})
    
    if device_config.get('force_cpu', False):
        return "cpu"
    if device_config.get('force_cuda', False):
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
    
    # Auto-detect
    if device_config.get('auto_detect', True):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    return "cpu"


def main():
    """Main entry point"""
    logger.info("Starting Async WAF Service...")
    
    # Load configuration
    config, inference_config = load_config()
    
    # Get paths
    model_path = config.get('model_path', 'models/checkpoints/best_model.pt')
    vocab_path = config.get('vocab_path', 'models/vocabularies/http_vocab.json')
    threshold = config.get('threshold', 0.5)
    max_workers = config.get('max_workers', config.get('workers', 4))
    batch_size = config.get('batch_size', 32)
    timeout = config.get('timeout', 5.0)
    
    # Resolve paths relative to project root
    model_path = project_root / model_path
    vocab_path = project_root / vocab_path
    
    # Check if files exist
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        logger.error("Please train a model first or update the model_path in config.yaml")
        sys.exit(1)
    
    if not vocab_path.exists():
        logger.error(f"Vocabulary file not found: {vocab_path}")
        logger.error("Please generate vocabulary first or update the vocab_path in config.yaml")
        sys.exit(1)
    
    # Determine device
    device = determine_device(inference_config)
    
    # Get optimization config
    optimization_config = inference_config.get('optimization', {})
    optimization = None
    if optimization_config.get('quantization', False):
        optimization = "quantization"
    elif optimization_config.get('torchscript', False):
        optimization = "torchscript"
    
    # Get queue manager config
    queue_config = inference_config.get('queue', {})
    use_queue_manager = queue_config.get('enabled', False)
    queue_max_size = queue_config.get('max_size', 1000)
    queue_batch_timeout = queue_config.get('batch_timeout', 0.1)
    
    # Get anomaly log file
    integration_config = config.get('integration', {})
    logging_config = integration_config.get('logging', {})
    anomaly_log_file = None
    if logging_config.get('log_anomalies', False):
        anomaly_log_file = logging_config.get('log_file', 'logs/waf_detections.log')
        if not Path(anomaly_log_file).is_absolute():
            anomaly_log_file = str(project_root / anomaly_log_file)
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Vocab: {vocab_path}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Max Workers: {max_workers}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Timeout: {timeout}s")
    logger.info(f"  Optimization: {optimization or 'none'}")
    logger.info(f"  Queue Manager: {use_queue_manager}")
    logger.info(f"  Anomaly Log: {anomaly_log_file or 'disabled'}")
    
    # Initialize rate limiting
    rate_limit_config = inference_config.get('rate_limiting', {})
    if rate_limit_config.get('enabled', False):
        initialize_rate_limiting(
            enabled=True,
            max_requests_per_second=rate_limit_config.get('max_requests_per_second', 100),
            per_ip=rate_limit_config.get('per_ip', False),
            max_ips=rate_limit_config.get('max_ips', 10000)
        )
    
    # Initialize service
    try:
        initialize_service(
            model_path=str(model_path),
            vocab_path=str(vocab_path),
            threshold=threshold,
            max_workers=max_workers,
            batch_size=batch_size,
            timeout=timeout,
            device=device,
            optimization=optimization,
            use_queue_manager=use_queue_manager,
            queue_max_size=queue_max_size,
            queue_batch_timeout=queue_batch_timeout,
            anomaly_log_file=anomaly_log_file
        )
        logger.info("Async WAF Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get server config
    host = config.get('host', '0.0.0.0')
    port = config.get('port', 8000)
    workers = config.get('workers', 1)
    
    logger.info(f"Starting server on {host}:{port} with {workers} worker(s)")
    
    # Run server
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers if workers > 1 else None,  # uvicorn workers only for > 1
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
