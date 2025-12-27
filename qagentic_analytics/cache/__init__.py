"""Cache management for the Analytics Engine."""

import logging
from typing import Any, Optional
import json
import asyncio

import aioredis
from aioredis.client import Redis

from qagentic_analytics.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class Cache:
    """Redis-based cache with fallback to in-memory cache."""

    def __init__(self):
        self._redis: Optional[Redis] = None
        self._local_cache: dict = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self._redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {str(e)}")
            logger.warning("Falling back to in-memory cache")
            self._redis = None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        try:
            if self._redis:
                value = await self._redis.get(key)
                if value:
                    return json.loads(value)
            else:
                async with self._lock:
                    return self._local_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Optional expiration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            serialized = json.dumps(value)
            
            if self._redis:
                if expire:
                    await self._redis.setex(key, expire, serialized)
                else:
                    await self._redis.set(key, serialized)
            else:
                async with self._lock:
                    self._local_cache[key] = value
                    if expire:
                        asyncio.create_task(
                            self._expire_local(key, expire)
                        )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self._redis:
                await self._redis.delete(key)
            else:
                async with self._lock:
                    self._local_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False

    async def clear(self) -> bool:
        """
        Clear all cached values.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self._redis:
                await self._redis.flushdb()
            else:
                async with self._lock:
                    self._local_cache.clear()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return False

    async def _expire_local(self, key: str, seconds: int) -> None:
        """Expire a key from local cache after delay."""
        await asyncio.sleep(seconds)
        async with self._lock:
            self._local_cache.pop(key, None)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

# Global cache instance
_cache: Optional[Cache] = None

async def initialize_cache() -> None:
    """Initialize the global cache instance."""
    global _cache
    if not _cache:
        _cache = Cache()
        await _cache.initialize()

def get_cache() -> Cache:
    """
    Get the global cache instance.
    
    Returns:
        Cache instance
        
    Raises:
        RuntimeError: If cache not initialized
    """
    if not _cache:
        raise RuntimeError("Cache not initialized")
    return _cache
