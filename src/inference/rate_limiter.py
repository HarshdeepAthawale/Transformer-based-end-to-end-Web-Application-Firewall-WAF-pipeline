"""
Rate Limiter Module

Rate limiting for request throttling
"""
from collections import deque
import time
from typing import Optional
from threading import Lock
from loguru import logger


class RateLimiter:
    """Rate limiter for request throttling"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 1):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = Lock()
        
        logger.info(f"RateLimiter initialized: max_requests={max_requests}, window_seconds={window_seconds}")
    
    def is_allowed(self) -> bool:
        """
        Check if request is allowed
        
        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            # Check limit
            if len(self.requests) >= self.max_requests:
                return False
            
            # Add current request
            self.requests.append(now)
            return True
    
    def get_wait_time(self) -> float:
        """
        Get time to wait before next request is allowed
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self.lock:
            if not self.requests:
                return 0.0
            
            now = time.time()
            
            # Remove old requests
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                return 0.0
            
            # Calculate wait time until oldest request expires
            oldest = self.requests[0]
            wait_time = self.window_seconds - (now - oldest)
            return max(0.0, wait_time)
    
    def reset(self):
        """Reset rate limiter"""
        with self.lock:
            self.requests.clear()
            logger.info("Rate limiter reset")
    
    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics
        
        Returns:
            Dictionary with current request count and window info
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            return {
                'current_requests': len(self.requests),
                'max_requests': self.max_requests,
                'window_seconds': self.window_seconds,
                'remaining_requests': max(0, self.max_requests - len(self.requests)),
                'is_limited': len(self.requests) >= self.max_requests
            }


class PerIPRateLimiter:
    """Rate limiter per IP address"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 1, max_ips: int = 10000):
        """
        Initialize per-IP rate limiter
        
        Args:
            max_requests: Maximum requests per IP in window
            window_seconds: Time window in seconds
            max_ips: Maximum number of IPs to track
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_ips = max_ips
        self.limiters: dict[str, RateLimiter] = {}
        self.lock = Lock()
        self.last_cleanup = time.time()
        
        logger.info(f"PerIPRateLimiter initialized: max_requests={max_requests}, window_seconds={window_seconds}, max_ips={max_ips}")
    
    def _get_limiter(self, ip: str) -> RateLimiter:
        """Get or create rate limiter for IP"""
        with self.lock:
            # Periodic cleanup of old limiters
            now = time.time()
            if now - self.last_cleanup > 60:  # Cleanup every minute
                self._cleanup()
                self.last_cleanup = now
            
            if ip not in self.limiters:
                # Limit number of tracked IPs
                if len(self.limiters) >= self.max_ips:
                    # Remove oldest entry (simple FIFO)
                    oldest_ip = next(iter(self.limiters))
                    del self.limiters[oldest_ip]
                
                self.limiters[ip] = RateLimiter(self.max_requests, self.window_seconds)
            
            return self.limiters[ip]
    
    def is_allowed(self, ip: str) -> bool:
        """Check if request from IP is allowed"""
        limiter = self._get_limiter(ip)
        return limiter.is_allowed()
    
    def get_wait_time(self, ip: str) -> float:
        """Get wait time for IP"""
        limiter = self._get_limiter(ip)
        return limiter.get_wait_time()
    
    def _cleanup(self):
        """Cleanup old limiters"""
        # In a production system, you might want more sophisticated cleanup
        # For now, we rely on the max_ips limit
        pass
    
    def get_stats(self, ip: Optional[str] = None) -> dict:
        """Get statistics for specific IP or all IPs"""
        if ip:
            if ip in self.limiters:
                return self.limiters[ip].get_stats()
            else:
                return {'current_requests': 0, 'max_requests': self.max_requests}
        else:
            return {
                'tracked_ips': len(self.limiters),
                'max_ips': self.max_ips,
                'max_requests_per_ip': self.max_requests,
                'window_seconds': self.window_seconds
            }
