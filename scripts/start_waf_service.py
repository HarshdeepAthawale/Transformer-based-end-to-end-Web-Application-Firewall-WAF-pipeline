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

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

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
        help="Path to HuggingFace model directory (overrides config)"
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Run in placeholder mode (no ML model, for testing)"
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

    threshold = args.threshold or waf_config.get('threshold', training_config.get('anomaly', {}).get('threshold', 0.5))
    device = args.device or ("cuda" if os.environ.get('CUDA_AVAILABLE', '').lower() == 'true' else None)

    # Resolve model path
    model_path = args.model_path or waf_config.get('model_path', 'models/waf-anomaly')
    model_dir = _project_root / model_path if not Path(model_path).is_absolute() else Path(model_path)

    # Check for valid HuggingFace model (config.json + tokenizer.json or model.safetensors)
    def _model_valid(p: Path) -> bool:
        if not p.is_dir():
            return False
        has_config = (p / "config.json").exists()
        has_tokenizer = (p / "tokenizer.json").exists()
        has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        return has_config and has_tokenizer and has_weights

    use_placeholder = args.placeholder
    if not use_placeholder and not _model_valid(model_dir):
        logger.warning(f"Model not found at {model_dir} (need config.json, tokenizer.json, model weights). Using placeholder mode.")
        use_placeholder = True

    # Initialize WAF service
    logger.info("Initializing WAF service..." + (" (placeholder mode)" if use_placeholder else ""))
    try:
        initialize_waf_service(
            model_path=None if use_placeholder else str(model_dir),
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