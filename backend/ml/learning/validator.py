"""
Model Validator

Validate models before deployment.
"""
from typing import Dict, List
from loguru import logger

from backend.ml.model.anomaly_detector import AnomalyDetector
from backend.ml.tokenization.tokenizer import HTTPTokenizer
from backend.ml.training.evaluator import ModelEvaluator
from backend.ml.tokenization.dataloader import create_dataloader


class ModelValidator:
    """Validate models before deployment"""
    
    def __init__(
        self,
        vocab_path: str,
        test_data: List[str],
        device: str = "cpu",
        min_accuracy: float = 0.95,
        max_fpr: float = 0.01
    ):
        """
        Initialize validator
        
        Args:
            vocab_path: Path to vocabulary file
            test_data: Test data for validation
            device: Device for validation
            min_accuracy: Minimum accuracy required
            max_fpr: Maximum false positive rate allowed
        """
        self.vocab_path = vocab_path
        self.test_data = test_data
        self.device = device
        self.min_accuracy = min_accuracy
        self.max_fpr = max_fpr
        
        # Load tokenizer
        self.tokenizer = HTTPTokenizer()
        self.tokenizer.load_vocab(vocab_path)
    
    def validate(
        self,
        model_path: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Validate model
        
        Args:
            model_path: Path to model checkpoint
            threshold: Detection threshold
        
        Returns:
            Validation result dictionary
        """
        logger.info(f"Validating model: {model_path}")
        
        try:
            # Load model
            model = AnomalyDetector.load_checkpoint(model_path, device=self.device)
            
            # Create evaluator
            evaluator = ModelEvaluator(model, self.tokenizer, device=self.device)
            
            # Create data loader
            test_loader = create_dataloader(
                self.test_data,
                self.tokenizer,
                batch_size=32,
                shuffle=False
            )
            
            # Evaluate
            metrics = evaluator.evaluate_on_loader(test_loader, threshold=threshold)
            
            # Check validation criteria
            is_valid = (
                metrics['accuracy'] >= self.min_accuracy and
                metrics['fpr'] <= self.max_fpr
            )
            
            result = {
                'is_valid': is_valid,
                'metrics': metrics,
                'passed_accuracy': metrics['accuracy'] >= self.min_accuracy,
                'passed_fpr': metrics['fpr'] <= self.max_fpr
            }
            
            if is_valid:
                logger.info("Model validation passed")
            else:
                logger.warning("Model validation failed")
                logger.warning(f"Accuracy: {metrics['accuracy']:.4f} (min: {self.min_accuracy:.4f})")
                logger.warning(f"FPR: {metrics['fpr']:.4f} (max: {self.max_fpr:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return {
                'is_valid': False,
                'error': str(e)
            }
