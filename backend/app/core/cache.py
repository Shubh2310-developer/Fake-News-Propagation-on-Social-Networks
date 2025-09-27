"""
Redis cache connection and utilities.
"""

import redis.asyncio as redis
from typing import Optional, Any
import json
from app.core.config import settings

# Global Redis client
redis_client: Optional[redis.Redis] = None


async def init_redis():
    """Initialize Redis connection."""
    global redis_client

    try:
        redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

        # Test connection
        await redis_client.ping()
        print(" Redis connected successfully")

    except Exception as e:
        print(f"L Redis connection failed: {e}")
        redis_client = None


async def get_redis():
    """Dependency to get Redis client."""
    if redis_client is None:
        await init_redis()
    return redis_client


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        print("= Redis connection closed")


class CacheService:
    """Redis cache service with common operations."""

    @staticmethod
    async def get(key: str) -> Optional[Any]:
        """Get value from cache."""
        client = await get_redis()
        if client:
            try:
                value = await client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                print(f"Cache get error: {e}")
        return None

    @staticmethod
    async def set(key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache with expiration."""
        client = await get_redis()
        if client:
            try:
                await client.setex(key, expire, json.dumps(value))
                return True
            except Exception as e:
                print(f"Cache set error: {e}")
        return False

    @staticmethod
    async def delete(key: str) -> bool:
        """Delete key from cache."""
        client = await get_redis()
        if client:
            try:
                await client.delete(key)
                return True
            except Exception as e:
                print(f"Cache delete error: {e}")
        return False

    @staticmethod
    async def exists(key: str) -> bool:
        """Check if key exists in cache."""
        client = await get_redis()
        if client:
            try:
                return bool(await client.exists(key))
            except Exception as e:
                print(f"Cache exists error: {e}")
        return False


# Global cache service instance
cache = CacheService()