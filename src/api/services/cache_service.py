"""
Caching Service using Redis or in-memory cache
"""
from typing import Optional, Any
import json
import time
from loguru import logger


class CacheService:
    """Caching service for frequently accessed data"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_ttl = {}
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis client if available"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized")
        except ImportError:
            logger.info("Redis not available, using in-memory cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory cache")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Fallback to memory cache
        if key in self.memory_cache:
            # Check TTL
            if key in self.cache_ttl:
                if time.time() > self.cache_ttl[key]:
                    del self.memory_cache[key]
                    del self.cache_ttl[key]
                    return None
            return self.memory_cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL (seconds)"""
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value)
                )
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Fallback to memory cache
        self.memory_cache[key] = value
        self.cache_ttl[key] = time.time() + ttl
    
    def delete(self, key: str):
        """Delete key from cache"""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        if key in self.memory_cache:
            del self.memory_cache[key]
        if key in self.cache_ttl:
            del self.cache_ttl[key]
    
    def clear(self):
        """Clear all cache"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        
        self.memory_cache.clear()
        self.cache_ttl.clear()


# Global cache instance
cache_service = CacheService()
