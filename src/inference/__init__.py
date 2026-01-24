"""
Inference Module

Real-time non-blocking detection for WAF service
"""
from .async_waf_service import AsyncWAFService, initialize_service, initialize_rate_limiting, app, waf_service
from .queue_manager import RequestQueueManager
from .rate_limiter import RateLimiter, PerIPRateLimiter
from .optimization import optimize_model, load_optimized_model, save_optimized_model

__all__ = [
    'AsyncWAFService',
    'initialize_service',
    'initialize_rate_limiting',
    'app',
    'waf_service',
    'RequestQueueManager',
    'RateLimiter',
    'PerIPRateLimiter',
    'optimize_model',
    'load_optimized_model',
    'save_optimized_model'
]
