"""
Model Optimization Module

Optimize model for faster inference using quantization and TorchScript
"""
import torch
from torch.quantization import quantize_dynamic
from loguru import logger
from typing import Optional
from pathlib import Path

from src.model.anomaly_detector import AnomalyDetector


def optimize_model(model: AnomalyDetector, method: str = "quantization") -> AnomalyDetector:
    """
    Optimize model for inference
    
    Args:
        model: AnomalyDetector model to optimize
        method: Optimization method ("quantization", "torchscript", or None)
    
    Returns:
        Optimized model
    """
    model.eval()
    
    if method == "quantization":
        # Dynamic quantization for faster inference
        logger.info("Applying dynamic quantization...")
        try:
            quantized_model = quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Quantization complete")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, returning original model")
            return model
    
    elif method == "torchscript":
        # TorchScript compilation for faster inference
        logger.info("Compiling to TorchScript...")
        try:
            # Create example input for tracing
            vocab_size = model.transformer.embeddings.word_embeddings.num_embeddings
            max_length = model.transformer.embeddings.position_embeddings.num_embeddings
            
            example_input_ids = torch.randint(0, vocab_size, (1, min(128, max_length)))
            example_attention_mask = torch.ones_like(example_input_ids)
            
            # Trace the model
            traced_model = torch.jit.trace(
                model,
                (example_input_ids, example_attention_mask),
                strict=False
            )
            logger.info("TorchScript compilation complete")
            return traced_model
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}, returning original model")
            return model
    
    else:
        logger.warning(f"Unknown optimization method: {method}, returning original model")
        return model


def load_optimized_model(
    model_path: str,
    optimization: Optional[str] = None,
    device: str = "cpu"
) -> AnomalyDetector:
    """
    Load and optimize model
    
    Args:
        model_path: Path to model checkpoint
        optimization: Optimization method ("quantization", "torchscript", or None)
        device: Device to load model on
    
    Returns:
        Loaded and optionally optimized model
    """
    logger.info(f"Loading model from {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Extract model architecture from checkpoint
    state_dict = checkpoint['model_state_dict']
    vocab_size = checkpoint.get('vocab_size', 10000)
    
    # Infer architecture from state dict
    if 'transformer.embeddings.word_embeddings.weight' in state_dict:
        hidden_size = state_dict['transformer.embeddings.word_embeddings.weight'].shape[1]
    else:
        hidden_size = 768
    
    if 'transformer.embeddings.position_embeddings.weight' in state_dict:
        max_length = state_dict['transformer.embeddings.position_embeddings.weight'].shape[0]
    else:
        max_length = 512
    
    num_layers = 0
    while f'transformer.transformer.layer.{num_layers}.attention.q_lin.weight' in state_dict:
        num_layers += 1
    
    possible_heads = [8, 12, 16]
    num_heads = 12
    for heads in possible_heads:
        if hidden_size % heads == 0:
            num_heads = heads
            break
    
    logger.info(f"Model architecture: vocab_size={vocab_size}, hidden_size={hidden_size}, "
               f"num_layers={num_layers}, num_heads={num_heads}, max_length={max_length}")
    
    # Recreate model
    model = AnomalyDetector(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_length=max_length
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    # Apply optimization
    if optimization:
        model = optimize_model(model, optimization)
    
    return model


def save_optimized_model(model: AnomalyDetector, output_path: str, optimization: str):
    """
    Save optimized model
    
    Args:
        model: Optimized model
        output_path: Path to save model
        optimization: Optimization method used
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if optimization == "torchscript":
        # Save TorchScript model
        model.save(output_path)
        logger.info(f"TorchScript model saved to {output_path}")
    else:
        # Save regular PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimization': optimization
        }, output_path)
        logger.info(f"Optimized model saved to {output_path}")
