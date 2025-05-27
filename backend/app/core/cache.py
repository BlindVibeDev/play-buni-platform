"""
Cache Manager for Play Buni Platform

Redis-based caching system for performance optimization.
"""

import json
import asyncio
from typing import Optional, Any, Union
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Redis-based cache manager for the application
    """
    
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self._connection_pool = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            # Create connection pool
            self._connection_pool = redis.ConnectionPool.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True,
                max_connections=20
            )
            
            # Create Redis client
            self.redis_client = Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache manager: {e}")
            # Fallback to in-memory cache
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if not self.redis_client:
                return None
            
            value = await self.redis_client.get(key)
            return value
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: str, 
        expire: Optional[int] = None
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds
            
        Returns:
            True if successful
        """
        try:
            if not self.redis_client:
                return False
            
            if expire:
                await self.redis_client.setex(key, expire, value)
            else:
                await self.redis_client.set(key, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful
        """
        try:
            if not self.redis_client:
                return False
            
            await self.redis_client.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
        """
        try:
            if not self.redis_client:
                return False
            
            result = await self.redis_client.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a numeric value in cache
        
        Args:
            key: Cache key
            amount: Amount to increment by
            
        Returns:
            New value after increment
        """
        try:
            if not self.redis_client:
                return None
            
            result = await self.redis_client.incrby(key, amount)
            return result
            
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return None
    
    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration time for a key
        
        Args:
            key: Cache key
            seconds: Expiration time in seconds
            
        Returns:
            True if successful
        """
        try:
            if not self.redis_client:
                return False
            
            await self.redis_client.expire(key, seconds)
            return True
            
        except Exception as e:
            logger.error(f"Error setting expiration for cache key {key}: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """
        Get JSON value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Parsed JSON value or None
        """
        try:
            value = await self.get(key)
            if value:
                return json.loads(value)
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from cache key {key}: {e}")
            return None
    
    async def set_json(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[int] = None
    ) -> bool:
        """
        Set JSON value in cache
        
        Args:
            key: Cache key
            value: Value to serialize and cache
            expire: Expiration time in seconds
            
        Returns:
            True if successful
        """
        try:
            json_value = json.dumps(value, default=str)
            return await self.set(key, json_value, expire)
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing JSON for cache key {key}: {e}")
            return False
    
    async def get_many(self, keys: list) -> dict:
        """
        Get multiple values from cache
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        try:
            if not self.redis_client or not keys:
                return {}
            
            values = await self.redis_client.mget(keys)
            return dict(zip(keys, values))
            
        except Exception as e:
            logger.error(f"Error getting multiple cache keys: {e}")
            return {}
    
    async def set_many(
        self, 
        mapping: dict, 
        expire: Optional[int] = None
    ) -> bool:
        """
        Set multiple values in cache
        
        Args:
            mapping: Dictionary of key-value pairs
            expire: Expiration time in seconds
            
        Returns:
            True if successful
        """
        try:
            if not self.redis_client or not mapping:
                return False
            
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            
            for key, value in mapping.items():
                if expire:
                    pipe.setex(key, expire, value)
                else:
                    pipe.set(key, value)
            
            await pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Error setting multiple cache keys: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern
        
        Args:
            pattern: Redis pattern (e.g., "user:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            if not self.redis_client:
                return 0
            
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get time to live for a key
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        try:
            if not self.redis_client:
                return None
            
            ttl = await self.redis_client.ttl(key)
            return ttl
            
        except Exception as e:
            logger.error(f"Error getting TTL for cache key {key}: {e}")
            return None


# Global cache manager instance
cache_manager = CacheManager()


async def init_cache():
    """Initialize the cache manager"""
    await cache_manager.initialize()


async def close_cache():
    """Close the cache manager"""
    await cache_manager.close()


# Utility functions for common caching patterns
async def cache_with_ttl(
    key: str,
    fetch_func,
    ttl: int = 300,
    *args,
    **kwargs
):
    """
    Cache the result of a function with TTL
    
    Args:
        key: Cache key
        fetch_func: Function to call if cache miss
        ttl: Time to live in seconds
        *args: Arguments for fetch_func
        **kwargs: Keyword arguments for fetch_func
        
    Returns:
        Cached or fresh result
    """
    # Try to get from cache first
    cached_result = await cache_manager.get_json(key)
    if cached_result is not None:
        return cached_result
    
    # Cache miss, fetch fresh data
    try:
        if asyncio.iscoroutinefunction(fetch_func):
            result = await fetch_func(*args, **kwargs)
        else:
            result = fetch_func(*args, **kwargs)
        
        # Cache the result
        await cache_manager.set_json(key, result, expire=ttl)
        return result
        
    except Exception as e:
        logger.error(f"Error in cache_with_ttl for key {key}: {e}")
        return None


def cache_key(*parts) -> str:
    """
    Generate a cache key from parts
    
    Args:
        *parts: Key parts to join
        
    Returns:
        Cache key string
    """
    return ":".join(str(part) for part in parts if part is not None) 