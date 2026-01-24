#!/usr/bin/env python3
"""
WAF Service Startup Script

Starts the FastAPI WAF service with real model inference
"""
import argparse
import os
import sys
from pathlib import Path
import yaml
import uvicorn
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set TMPDIR to avoid PyTorch cache issues
os.environ['TMPDIR'] = '/tmp'

from integration.waf_service import app, initialize_waf_service


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Start WAF Service")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path",
        help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--vocab_path",
        help="Path to vocabulary file (overrides config)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Anomaly threshold (overrides config)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers"
    )
    parser.add_argument(
        "--log_level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    parser.add_argument(
        "--device",
        help="Device to use (cpu/cuda, auto-detect if not specified)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    waf_config = config.get('integration', {}).get('waf_service', {})
    training_config = config.get('training', {})

    # Determine paths
    model_path = args.model_path or waf_config.get('model_path', 'models/checkpoints/best_model.pt')
    vocab_path = args.vocab_path or waf_config.get('vocab_path', 'models/vocabularies/http_vocab.json')
    threshold = args.threshold or waf_config.get('threshold', training_config.get('anomaly', {}).get('threshold', 0.5))
    device = args.device or ("cuda" if os.environ.get('CUDA_AVAILABLE', '').lower() == 'true' else None)

    # Validate paths
    if not Path(model_path).exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        sys.exit(1)

    if not Path(vocab_path).exists():
        logger.error(f"Vocabulary file not found: {vocab_path}")
        sys.exit(1)

    # Initialize WAF service
    logger.info("Initializing WAF service with real model...")
    try:
        initialize_waf_service(
            model_path=model_path,
            vocab_path=vocab_path,
            threshold=threshold,
            device=device
        )
        logger.info("WAF service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize WAF service: {e}")
        sys.exit(1)

    # Start server
    logger.info(f"Starting WAF service on {args.host}:{args.port} with {args.workers} workers")

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("WAF service stopped by user")
    except Exception as e:
        logger.error(f"WAF service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()