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
                await client.setex(key, expire, json.dumps(value, default=str))
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

    @staticmethod
    async def get_all_keys(pattern: str = "*") -> list:
        """Get all keys matching pattern."""
        client = await get_redis()
        if client:
            try:
                keys = await client.keys(pattern)
                return keys if keys else []
            except Exception as e:
                print(f"Cache keys error: {e}")
        return []

    @staticmethod
    async def get_ttl(key: str) -> int:
        """Get TTL of a key in seconds."""
        client = await get_redis()
        if client:
            try:
                return await client.ttl(key)
            except Exception as e:
                print(f"Cache TTL error: {e}")
        return -1


class SimulationCacheService:
    """Redis cache service specialized for simulation storage."""

    # TTL constants (in seconds)
    SIMULATION_TTL = 24 * 60 * 60  # 24 hours

    # Key prefixes
    SIMULATION_PREFIX = "simulation:"
    SIMULATION_INDEX = "simulation:index"

    @staticmethod
    def _get_simulation_key(simulation_id: str) -> str:
        """Generate Redis key for simulation data."""
        return f"{SimulationCacheService.SIMULATION_PREFIX}{simulation_id}"

    @staticmethod
    async def create_simulation(simulation_id: str, data: dict) -> bool:
        """
        Create a new simulation entry in Redis.

        Args:
            simulation_id: Unique simulation identifier
            data: Simulation data dictionary

        Returns:
            True if successful, False otherwise
        """
        client = await get_redis()
        if not client:
            return False

        try:
            key = SimulationCacheService._get_simulation_key(simulation_id)

            # Store simulation data
            serialized = json.dumps(data, default=str)
            await client.setex(key, SimulationCacheService.SIMULATION_TTL, serialized)

            # Add to index (sorted set by creation time)
            created_at = data.get('created_at', '')
            await client.zadd(
                SimulationCacheService.SIMULATION_INDEX,
                {simulation_id: created_at}
            )

            return True
        except Exception as e:
            print(f"Failed to create simulation {simulation_id}: {e}")
            return False

    @staticmethod
    async def get_simulation(simulation_id: str) -> Optional[dict]:
        """
        Retrieve simulation data from Redis.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            Simulation data dictionary or None if not found
        """
        client = await get_redis()
        if not client:
            return None

        try:
            key = SimulationCacheService._get_simulation_key(simulation_id)
            value = await client.get(key)

            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Failed to get simulation {simulation_id}: {e}")
            return None

    @staticmethod
    async def update_simulation(simulation_id: str, data: dict) -> bool:
        """
        Update existing simulation data in Redis.

        Args:
            simulation_id: Unique simulation identifier
            data: Updated simulation data dictionary

        Returns:
            True if successful, False otherwise
        """
        client = await get_redis()
        if not client:
            return False

        try:
            key = SimulationCacheService._get_simulation_key(simulation_id)

            # Check if simulation exists
            exists = await client.exists(key)
            if not exists:
                print(f"Simulation {simulation_id} does not exist")
                return False

            # Get current TTL to preserve it
            ttl = await client.ttl(key)
            if ttl < 0:
                ttl = SimulationCacheService.SIMULATION_TTL

            # Update with preserved TTL
            serialized = json.dumps(data, default=str)
            await client.setex(key, ttl, serialized)

            return True
        except Exception as e:
            print(f"Failed to update simulation {simulation_id}: {e}")
            return False

    @staticmethod
    async def delete_simulation(simulation_id: str) -> bool:
        """
        Delete simulation from Redis.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            True if successful, False otherwise
        """
        client = await get_redis()
        if not client:
            return False

        try:
            key = SimulationCacheService._get_simulation_key(simulation_id)

            # Delete simulation data
            await client.delete(key)

            # Remove from index
            await client.zrem(SimulationCacheService.SIMULATION_INDEX, simulation_id)

            return True
        except Exception as e:
            print(f"Failed to delete simulation {simulation_id}: {e}")
            return False

    @staticmethod
    async def exists(simulation_id: str) -> bool:
        """
        Check if simulation exists in Redis.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            True if exists, False otherwise
        """
        client = await get_redis()
        if not client:
            return False

        try:
            key = SimulationCacheService._get_simulation_key(simulation_id)
            return bool(await client.exists(key))
        except Exception as e:
            print(f"Failed to check simulation existence {simulation_id}: {e}")
            return False

    @staticmethod
    async def list_simulations(limit: int = 50, offset: int = 0) -> list:
        """
        List all simulation IDs ordered by creation time (newest first).

        Args:
            limit: Maximum number of simulations to return
            offset: Number of simulations to skip

        Returns:
            List of simulation IDs
        """
        client = await get_redis()
        if not client:
            return []

        try:
            # Get IDs from sorted set in reverse order (newest first)
            simulation_ids = await client.zrange(
                SimulationCacheService.SIMULATION_INDEX,
                offset,
                offset + limit - 1,
                desc=True
            )
            return simulation_ids if simulation_ids else []
        except Exception as e:
            print(f"Failed to list simulations: {e}")
            return []

    @staticmethod
    async def count_simulations() -> int:
        """
        Get total count of simulations in Redis.

        Returns:
            Total number of simulations
        """
        client = await get_redis()
        if not client:
            return 0

        try:
            count = await client.zcard(SimulationCacheService.SIMULATION_INDEX)
            return count if count else 0
        except Exception as e:
            print(f"Failed to count simulations: {e}")
            return 0

    @staticmethod
    async def get_all_simulations() -> list:
        """
        Get all simulation data (useful for filtering).

        Returns:
            List of simulation data dictionaries
        """
        client = await get_redis()
        if not client:
            return []

        try:
            # Get all simulation IDs
            simulation_ids = await client.zrange(
                SimulationCacheService.SIMULATION_INDEX,
                0,
                -1,
                desc=True
            )

            if not simulation_ids:
                return []

            # Fetch all simulation data
            simulations = []
            for sim_id in simulation_ids:
                sim_data = await SimulationCacheService.get_simulation(sim_id)
                if sim_data:
                    simulations.append(sim_data)

            return simulations
        except Exception as e:
            print(f"Failed to get all simulations: {e}")
            return []


# Global cache service instances
cache = CacheService()
simulation_cache = SimulationCacheService()