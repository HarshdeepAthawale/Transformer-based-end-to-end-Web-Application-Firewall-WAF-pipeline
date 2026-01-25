"""
Report Generator Module

Generate training and evaluation reports.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from loguru import logger


class ReportGenerator:
    """Generate training and evaluation reports"""
    
    @staticmethod
    def generate_training_report(
        metrics: Dict,
        output_path: str,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        training_config: Optional[Dict] = None
    ):
        """
        Generate training report
        
        Args:
            metrics: Training metrics dictionary
            output_path: Path to save report
            model_path: Path to trained model
            vocab_path: Path to vocabulary
            training_config: Training configuration
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'vocab_path': vocab_path,
            'training_config': training_config or {},
            'metrics': metrics
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {output_path}")
    
    @staticmethod
    def generate_evaluation_report(
        metrics: Dict,
        output_path: str,
        model_path: Optional[str] = None,
        dataset_info: Optional[Dict] = None
    ):
        """
        Generate evaluation report
        
        Args:
            metrics: Evaluation metrics dictionary
            output_path: Path to save report
            model_path: Path to evaluated model
            dataset_info: Dataset information
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'dataset_info': dataset_info or {},
            'metrics': metrics
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    @staticmethod
    def generate_threshold_optimization_report(
        optimization_result: Dict,
        output_path: str,
        target_fpr: float = 0.01
    ):
        """
        Generate threshold optimization report
        
        Args:
            optimization_result: Result from ThresholdOptimizer
            output_path: Path to save report
            target_fpr: Target FPR used
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_fpr': target_fpr,
            'optimal_threshold': optimization_result['optimal_threshold'],
            'metrics': optimization_result
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Threshold optimization report saved to {output_path}")
