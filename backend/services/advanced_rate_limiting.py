"""
Advanced Rate Limiting Service
"""
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time
from threading import Lock
from loguru import logger

from backend.utils.rate_limiter import PerIPRateLimiter


class AdvancedRateLimiter:
    """Advanced rate limiting with per-endpoint and adaptive limits"""
    
    def __init__(self):
        self.per_ip_limiters: Dict[str, PerIPRateLimiter] = {}
        self.per_endpoint_limiters: Dict[str, PerIPRateLimiter] = {}
        self.adaptive_limits: Dict[str, Dict] = {}  # Track traffic patterns
        self.lock = Lock()
        
        # Default limits
        self.default_ip_limit = 100  # requests per minute
        self.default_endpoint_limit = 200  # requests per minute
        
        # DDoS detection
        self.request_counts: Dict[str, list] = defaultdict(list)  # Track request timestamps
        self.ddos_threshold = 1000  # requests per minute to trigger DDoS detection
    
    def check_rate_limit(self, ip: str, endpoint: str = None) -> Dict:
        """
        Check rate limit for IP and endpoint
        Returns: {
            'allowed': bool,
            'wait_time': float,
            'remaining': int,
            'limit': int,
            'is_ddos': bool
        }
        """
        # Check DDoS first
        is_ddos = self._detect_ddos(ip, endpoint)
        if is_ddos:
            return {
                'allowed': False,
                'wait_time': 60.0,
                'remaining': 0,
                'limit': 0,
                'is_ddos': True
            }
        
        # Get or create limiter for IP
        if ip not in self.per_ip_limiters:
            with self.lock:
                if ip not in self.per_ip_limiters:
                    limit = self._get_adaptive_limit(ip, 'ip')
                    self.per_ip_limiters[ip] = PerIPRateLimiter(
                        max_requests=limit,
                        window_seconds=60
                    )
        
        ip_limiter = self.per_ip_limiters[ip]
        
        # Check IP limit
        if not ip_limiter.is_allowed(ip):
            wait_time = ip_limiter.get_wait_time(ip)
            stats = ip_limiter.get_stats(ip)
            return {
                'allowed': False,
                'wait_time': wait_time,
                'remaining': stats.get('remaining_requests', 0),
                'limit': stats.get('max_requests', 0),
                'is_ddos': False
            }
        
        # Check endpoint limit if specified
        if endpoint:
            endpoint_key = f"{ip}:{endpoint}"
            if endpoint_key not in self.per_endpoint_limiters:
                with self.lock:
                    if endpoint_key not in self.per_endpoint_limiters:
                        limit = self._get_adaptive_limit(endpoint, 'endpoint')
                        self.per_endpoint_limiters[endpoint_key] = PerIPRateLimiter(
                            max_requests=limit,
                            window_seconds=60
                        )
            
            endpoint_limiter = self.per_endpoint_limiters[endpoint_key]
            if not endpoint_limiter.is_allowed(ip):
                wait_time = endpoint_limiter.get_wait_time(ip)
                stats = endpoint_limiter.get_stats(ip)
                return {
                    'allowed': False,
                    'wait_time': wait_time,
                    'remaining': stats.get('remaining_requests', 0),
                    'limit': stats.get('max_requests', 0),
                    'is_ddos': False
                }
        
        # Record request for adaptive limiting
        self._record_request(ip, endpoint)
        
        return {
            'allowed': True,
            'wait_time': 0.0,
            'remaining': ip_limiter.get_stats(ip).get('remaining_requests', 0),
            'limit': ip_limiter.get_stats(ip).get('max_requests', 0),
            'is_ddos': False
        }
    
    def _detect_ddos(self, ip: str, endpoint: str = None) -> bool:
        """Detect DDoS attack"""
        now = time.time()
        key = endpoint or ip
        
        # Clean old timestamps (older than 1 minute)
        self.request_counts[key] = [
            ts for ts in self.request_counts[key]
            if now - ts < 60
        ]
        
        # Add current request
        self.request_counts[key].append(now)
        
        # Check threshold
        if len(self.request_counts[key]) > self.ddos_threshold:
            logger.warning(f"DDoS detected: {key} has {len(self.request_counts[key])} requests in last minute")
            return True
        
        return False
    
    def _get_adaptive_limit(self, key: str, limit_type: str) -> int:
        """Get adaptive limit based on traffic patterns"""
        if key not in self.adaptive_limits:
            return self.default_ip_limit if limit_type == 'ip' else self.default_endpoint_limit
        
        pattern = self.adaptive_limits[key]
        avg_requests = pattern.get('avg_requests', 0)
        
        # Set limit to 2x average requests, with minimum
        limit = max(50, int(avg_requests * 2))
        return min(limit, 1000)  # Cap at 1000
    
    def _record_request(self, ip: str, endpoint: str = None):
        """Record request for adaptive limiting"""
        now = time.time()
        
        if endpoint:
            if endpoint not in self.adaptive_limits:
                self.adaptive_limits[endpoint] = {
                    'requests': [],
                    'avg_requests': 0
                }
            
            self.adaptive_limits[endpoint]['requests'].append(now)
            # Keep only last 5 minutes
            self.adaptive_limits[endpoint]['requests'] = [
                ts for ts in self.adaptive_limits[endpoint]['requests']
                if now - ts < 300
            ]
            
            # Calculate average
            if self.adaptive_limits[endpoint]['requests']:
                count = len(self.adaptive_limits[endpoint]['requests'])
                self.adaptive_limits[endpoint]['avg_requests'] = count / 5  # per minute
    
    def set_limit(self, ip: str, limit: int, window_seconds: int = 60):
        """Set custom limit for IP"""
        with self.lock:
            if ip in self.per_ip_limiters:
                del self.per_ip_limiters[ip]
            
            self.per_ip_limiters[ip] = PerIPRateLimiter(
                max_requests=limit,
                window_seconds=window_seconds
            )
