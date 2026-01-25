"""
Model Architecture Module

Transformer-based anomaly detection models for HTTP request analysis.
"""
from .anomaly_detector import AnomalyDetector
from .scoring import AnomalyScorer

__all__ = [
    'AnomalyDetector',
    'AnomalyScorer'
]
