"""
Simple per-IP rate limiter utility
"""
import time
from collections import defaultdict
from threading import Lock
from typing import Dict, Optional
from loguru import logger


class PerIPRateLimiter:
    """Per-IP rate limiter using sliding window"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)  # IP -> list of timestamps
        self.lock = Lock()
    
    def is_allowed(self, ip: str) -> bool:
        """Check if IP is allowed to make a request"""
        now = time.time()
        with self.lock:
            # Clean old requests
            self.requests[ip] = [
                ts for ts in self.requests[ip]
                if now - ts < self.window_seconds
            ]
            
            # Check limit
            if len(self.requests[ip]) >= self.max_requests:
                return False
            
            # Record this request
            self.requests[ip].append(now)
            return True
    
    def get_wait_time(self, ip: str) -> float:
        """Get time to wait before next request is allowed"""
        now = time.time()
        with self.lock:
            if ip not in self.requests or not self.requests[ip]:
                return 0.0
            
            # Get oldest request in window
            oldest = min(self.requests[ip])
            wait = self.window_seconds - (now - oldest)
            return max(0.0, wait)
    
    def get_stats(self, ip: str) -> Dict:
        """Get rate limit stats for IP"""
        now = time.time()
        with self.lock:
            if ip not in self.requests:
                return {
                    'remaining_requests': self.max_requests,
                    'max_requests': self.max_requests,
                    'window_seconds': self.window_seconds
                }
            
            # Clean old requests
            self.requests[ip] = [
                ts for ts in self.requests[ip]
                if now - ts < self.window_seconds
            ]
            
            remaining = max(0, self.max_requests - len(self.requests[ip]))
            return {
                'remaining_requests': remaining,
                'max_requests': self.max_requests,
                'window_seconds': self.window_seconds,
                'current_requests': len(self.requests[ip])
            }
    
    def reset(self, ip: Optional[str] = None):
        """Reset rate limit for IP or all IPs"""
        with self.lock:
            if ip:
                if ip in self.requests:
                    del self.requests[ip]
            else:
                self.requests.clear()
