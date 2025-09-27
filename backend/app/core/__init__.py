"""
Core package for the Fake News Game Theory backend.

This module initializes core services such as:
- Database (PostgreSQL via SQLAlchemy/AsyncSession)
- Cache (Redis)
- Config & settings
- Logging
- Security
"""

from app.core.database import init_db, get_db
from app.core.cache import init_redis, get_redis
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.security import hash_password, verify_password, create_access_token, get_current_user

__all__ = [
    "init_db",
    "get_db",
    "init_redis",
    "get_redis",
    "settings",
    "configure_logging",
    "get_logger",
    "hash_password",
    "verify_password",
    "create_access_token",
    "get_current_user",
]