"""
Model Validator Module

Validates model before deployment to ensure quality
"""
from typing import Dict, List
import torch
from loguru import logger
from tqdm import tqdm

from src.model.anomaly_detector import AnomalyDetector
from src.model.scoring import AnomalyScorer
from src.tokenization.tokenizer import HTTPTokenizer
from src.tokenization.sequence_prep import SequencePreparator


class ModelValidator:
    """Validate model before deployment"""
    
    def __init__(
        self,
        vocab_path: str,
        test_data: List[str],
        threshold: float = 0.5,
        device: str = None
    ):
        """
        Initialize model validator
        
        Args:
            vocab_path: Path to vocabulary file
            test_data: List of benign test samples (should not be flagged as anomalies)
            threshold: Anomaly detection threshold
            device: Device to run validation on
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.vocab_path = vocab_path
        self.test_data = test_data
        self.threshold = threshold
        self.device = device
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {vocab_path}")
        self.tokenizer = HTTPTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        self.preparator = SequencePreparator(self.tokenizer)
        
        logger.info(f"ModelValidator initialized: test_samples={len(test_data)}, threshold={threshold}")
    
    def validate(
        self,
        model_path: str,
        max_false_positive_rate: float = 0.05,
        min_samples: int = 100
    ) -> Dict:
        """
        Validate model performance
        
        Args:
            model_path: Path to model checkpoint
            max_false_positive_rate: Maximum allowed false positive rate
            min_samples: Minimum number of test samples required
        
        Returns:
            Dictionary with validation results
        """
        if len(self.test_data) < min_samples:
            logger.warning(f"Insufficient test samples: {len(self.test_data)} < {min_samples}")
            return {
                'is_valid': False,
                'error': f'Insufficient test samples: {len(self.test_data)} < {min_samples}',
                'false_positive_rate': 1.0,
                'total_samples': len(self.test_data)
            }
        
        logger.info(f"Validating model: {model_path}")
        
        # Load model
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            vocab_size = checkpoint.get('vocab_size', len(self.tokenizer.word_to_id))
            
            # Infer architecture from checkpoint
            state_dict = checkpoint['model_state_dict']
            
            # Get hidden size from word embeddings
            if 'transformer.embeddings.word_embeddings.weight' in state_dict:
                hidden_size = state_dict['transformer.embeddings.word_embeddings.weight'].shape[1]
            else:
                hidden_size = 768  # fallback
            
            # Get max length from position embeddings
            if 'transformer.embeddings.position_embeddings.weight' in state_dict:
                max_length = state_dict['transformer.embeddings.position_embeddings.weight'].shape[0]
            else:
                max_length = 512  # fallback
            
            # Infer number of layers
            num_layers = 0
            while f'transformer.transformer.layer.{num_layers}.attention.q_lin.weight' in state_dict:
                num_layers += 1
            
            # Infer number of heads
            possible_heads = [8, 12, 16]
            num_heads = 12  # default
            for heads in possible_heads:
                if hidden_size % heads == 0:
                    num_heads = heads
                    break
            
            model = AnomalyDetector(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                max_length=max_length
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'false_positive_rate': 1.0,
                'total_samples': len(self.test_data)
            }
        
        scorer = AnomalyScorer(model, threshold=self.threshold, device=self.device)
        
        # Test on validation data
        false_positives = 0
        total = len(self.test_data)
        anomaly_scores = []
        
        logger.info(f"Testing on {total} benign samples...")
        
        with torch.no_grad():
            for text in tqdm(self.test_data, desc="Validating", leave=False):
                try:
                    token_ids, attention_mask = self.preparator.prepare_sequence(
                        text, max_length=max_length, padding=True, truncation=True
                    )
                    
                    input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
                    attn_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
                    
                    result = scorer.score(input_ids, attn_mask)
                    anomaly_scores.append(result['anomaly_score'])
                    
                    if result['is_anomaly']:
                        false_positives += 1
                
                except Exception as e:
                    logger.debug(f"Error processing sample: {e}")
                    # Count as false positive if we can't process
                    false_positives += 1
        
        false_positive_rate = false_positives / total if total > 0 else 0.0
        
        # Calculate statistics
        avg_score = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
        max_score = max(anomaly_scores) if anomaly_scores else 0.0
        
        # Validation result
        is_valid = false_positive_rate <= max_false_positive_rate
        
        result = {
            'is_valid': is_valid,
            'false_positive_rate': false_positive_rate,
            'max_allowed': max_false_positive_rate,
            'total_samples': total,
            'false_positives': false_positives,
            'true_negatives': total - false_positives,
            'avg_anomaly_score': avg_score,
            'max_anomaly_score': max_score,
            'threshold': self.threshold
        }
        
        if is_valid:
            logger.info(
                f"Model validation PASSED: FPR={false_positive_rate:.4f} "
                f"(max={max_false_positive_rate:.4f}), "
                f"avg_score={avg_score:.4f}"
            )
        else:
            logger.warning(
                f"Model validation FAILED: FPR={false_positive_rate:.4f} "
                f"(max={max_false_positive_rate:.4f}), "
                f"avg_score={avg_score:.4f}"
            )
        
        return result
