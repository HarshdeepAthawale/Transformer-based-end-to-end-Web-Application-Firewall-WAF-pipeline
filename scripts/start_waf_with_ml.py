#!/usr/bin/env python3
"""
Start WAF Service with ML Models

Start the WAF service with trained ML models loaded.
"""
import argparse
import uvicorn
from pathlib import Path
from loguru import logger

from backend.ml.waf_service import initialize_waf_service, app


def main():
    parser = argparse.ArgumentParser(description="Start WAF service with ML models")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, help="Path to vocabulary file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Try to find default paths if not provided
    project_root = Path(__file__).parent.parent
    
    if not args.model:
        default_model = project_root / "models" / "deployed" / "model.pt"
        if default_model.exists():
            args.model = str(default_model)
            logger.info(f"Using default model: {args.model}")
        else:
            logger.warning("No model path provided and default not found")
    
    if not args.vocab:
        default_vocab = project_root / "models" / "vocabularies" / "vocab.json"
        if default_vocab.exists():
            args.vocab = str(default_vocab)
            logger.info(f"Using default vocabulary: {args.vocab}")
        else:
            logger.warning("No vocabulary path provided and default not found")
    
    # Initialize WAF service
    logger.info("Initializing WAF service...")
    initialize_waf_service(
        model_path=args.model,
        vocab_path=args.vocab,
        threshold=args.threshold,
        device=args.device
    )
    
    # Start server
    logger.info(f"Starting WAF service on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
