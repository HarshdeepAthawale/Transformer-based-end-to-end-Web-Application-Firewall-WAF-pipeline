"""
Evaluation Report Generator Module

Generate comprehensive evaluation reports with metrics visualization
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class ReportGenerator:
    """Generate evaluation reports"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_training_report(
        self,
        training_results: Dict,
        metrics_history: List[Dict],
        optimal_threshold: Optional[float] = None,
        threshold_metrics: Optional[Dict] = None,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive training report
        
        Args:
            training_results: Training results dict
            metrics_history: List of metrics per epoch
            optimal_threshold: Optimal threshold found
            threshold_metrics: Metrics at optimal threshold
            output_filename: Output filename (default: auto-generated)
        
        Returns:
            Path to generated report
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"training_report_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        
        # Build report
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_summary': {
                'total_epochs': len(training_results.get('train_losses', [])),
                'best_val_loss': training_results.get('best_val_loss', 0.0),
                'final_train_loss': training_results.get('train_losses', [])[-1] if training_results.get('train_losses') else 0.0,
                'final_val_loss': training_results.get('val_losses', [])[-1] if training_results.get('val_losses') else 0.0,
            },
            'metrics_history': metrics_history,
            'optimal_threshold': optimal_threshold,
            'threshold_metrics': threshold_metrics,
            'best_metrics': self._get_best_metrics(metrics_history) if metrics_history else {},
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {output_path}")
        
        # Print summary
        self._print_summary(report)
        
        return str(output_path)
    
    def generate_evaluation_report(
        self,
        test_metrics: Dict,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate evaluation report for test set
        
        Args:
            test_metrics: Metrics on test set
            output_filename: Output filename
        
        Returns:
            Path to generated report
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"evaluation_report_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_metrics': test_metrics,
            'summary': {
                'accuracy': test_metrics.get('accuracy', 0.0),
                'tpr': test_metrics.get('tpr', 0.0),
                'fpr': test_metrics.get('fpr', 0.0),
                'precision': test_metrics.get('precision', 0.0),
                'recall': test_metrics.get('recall', 0.0),
                'f1_score': test_metrics.get('f1_score', 0.0),
                'roc_auc': test_metrics.get('roc_auc', 0.0),
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Print summary
        self._print_evaluation_summary(report)
        
        return str(output_path)
    
    def _get_best_metrics(self, metrics_history: List[Dict]) -> Dict:
        """Get best metrics from history"""
        if not metrics_history:
            return {}
        
        best_f1_idx = max(range(len(metrics_history)), key=lambda i: metrics_history[i].get('f1_score', 0))
        best_metrics = metrics_history[best_f1_idx]
        
        return {
            'epoch': best_f1_idx + 1,
            'f1_score': best_metrics.get('f1_score', 0.0),
            'tpr': best_metrics.get('tpr', 0.0),
            'fpr': best_metrics.get('fpr', 0.0),
            'precision': best_metrics.get('precision', 0.0),
            'recall': best_metrics.get('recall', 0.0),
        }
    
    def _print_summary(self, report: Dict):
        """Print training summary"""
        print("\n" + "="*60)
        print("TRAINING REPORT SUMMARY")
        print("="*60)
        
        summary = report.get('training_summary', {})
        print(f"Total Epochs: {summary.get('total_epochs', 0)}")
        print(f"Best Validation Loss: {summary.get('best_val_loss', 0.0):.4f}")
        
        best_metrics = report.get('best_metrics', {})
        if best_metrics:
            print(f"\nBest Metrics (Epoch {best_metrics.get('epoch', 0)}):")
            print(f"  F1 Score: {best_metrics.get('f1_score', 0.0):.4f}")
            print(f"  TPR (Recall): {best_metrics.get('tpr', 0.0):.4f}")
            print(f"  FPR: {best_metrics.get('fpr', 0.0):.4f}")
            print(f"  Precision: {best_metrics.get('precision', 0.0):.4f}")
        
        if report.get('optimal_threshold'):
            print(f"\nOptimal Threshold: {report['optimal_threshold']:.4f}")
            threshold_metrics = report.get('threshold_metrics', {})
            if threshold_metrics:
                metrics = threshold_metrics.get('metrics', {})
                print(f"  F1 Score: {metrics.get('f1_score', 0.0):.4f}")
                print(f"  TPR: {metrics.get('tpr', 0.0):.4f}")
                print(f"  FPR: {metrics.get('fpr', 0.0):.4f}")
        
        print("="*60)
    
    def _print_evaluation_summary(self, report: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        summary = report.get('summary', {})
        print(f"Accuracy: {summary.get('accuracy', 0.0):.4f}")
        print(f"TPR (Recall): {summary.get('tpr', 0.0):.4f}")
        print(f"FPR: {summary.get('fpr', 0.0):.4f}")
        print(f"Precision: {summary.get('precision', 0.0):.4f}")
        print(f"F1 Score: {summary.get('f1_score', 0.0):.4f}")
        if summary.get('roc_auc', 0.0) > 0:
            print(f"ROC-AUC: {summary.get('roc_auc', 0.0):.4f}")
        print("="*60)
