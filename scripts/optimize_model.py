#!/usr/bin/env python3
"""
Model Optimization Script

Optimize model for faster inference using quantization or TorchScript
"""
import sys
from pathlib import Path
import torch
from loguru import logger
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.optimization import optimize_model, load_optimized_model, save_optimized_model
from src.model.anomaly_detector import AnomalyDetector


def optimize_and_save(
    model_path: str,
    output_path: str,
    method: str = "quantization",
    vocab_path: str = None
):
    """
    Optimize model and save
    
    Args:
        model_path: Path to original model
        output_path: Path to save optimized model
        method: Optimization method ("quantization" or "torchscript")
        vocab_path: Path to vocabulary (for model loading)
    """
    logger.info(f"Optimizing model: {model_path}")
    logger.info(f"Method: {method}")
    logger.info(f"Output: {output_path}")
    
    # Resolve paths
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Get vocab path if not provided
    if vocab_path is None:
        config_path = project_root / "config" / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                vocab_path = config.get('waf_service', {}).get('vocab_path', 'models/vocabularies/http_vocab.json')
        else:
            vocab_path = 'models/vocabularies/http_vocab.json'
    
    vocab_path = Path(vocab_path)
    if not vocab_path.is_absolute():
        vocab_path = project_root / vocab_path
    
    try:
        # Load and optimize
        logger.info("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        optimized_model = load_optimized_model(
            model_path=str(model_path),
            optimization=method if method != "none" else None,
            device=device
        )
        
        logger.info("Model loaded and optimized")
        
        # Save optimized model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if method == "torchscript":
            # Save TorchScript model
            torch.jit.save(optimized_model, str(output_path))
            logger.info(f"TorchScript model saved: {output_path}")
        else:
            # Save regular PyTorch model
            checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
            checkpoint['model_state_dict'] = optimized_model.state_dict()
            checkpoint['optimized'] = True
            checkpoint['optimization_method'] = method
            
            torch.save(checkpoint, str(output_path))
            logger.info(f"Optimized model saved: {output_path}")
        
        # Compare sizes
        original_size = model_path.stat().st_size / (1024 * 1024)  # MB
        optimized_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Original size: {original_size:.2f} MB")
        logger.info(f"Optimized size: {optimized_size:.2f} MB")
        logger.info(f"Size reduction: {(1 - optimized_size/original_size)*100:.1f}%")
        logger.info("=" * 60)
        
        # Test inference speed (optional)
        logger.info("\nTesting inference speed...")
        test_inference_speed(optimized_model, device)
        
    except Exception as e:
        logger.error(f"Error optimizing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_inference_speed(model, device: str, num_iterations: int = 100):
    """Test inference speed"""
    import time
    import torch
    
    # Create dummy input
    batch_size = 1
    seq_length = 128
    
    # Get vocab size from model if possible
    if hasattr(model, 'transformer'):
        vocab_size = model.transformer.embeddings.word_embeddings.num_embeddings
    else:
        vocab_size = 10000  # fallback
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            if isinstance(model, torch.jit.ScriptModule):
                model(input_ids, attention_mask)
            else:
                model(input_ids, attention_mask)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            if isinstance(model, torch.jit.ScriptModule):
                model(input_ids, attention_mask)
            else:
                model(input_ids, attention_mask)
    
    elapsed = time.time() - start_time
    avg_time = (elapsed / num_iterations) * 1000  # ms
    
    logger.info(f"Average inference time: {avg_time:.2f}ms ({num_iterations} iterations)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize WAF model")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", required=True, help="Path to save optimized model")
    parser.add_argument("--method", default="quantization", choices=["quantization", "torchscript", "none"],
                        help="Optimization method")
    parser.add_argument("--vocab_path", help="Path to vocabulary file")
    
    args = parser.parse_args()
    
    optimize_and_save(
        model_path=args.model_path,
        output_path=args.output_path,
        method=args.method,
        vocab_path=args.vocab_path
    )


if __name__ == "__main__":
    main()
