"""
Core package for the Fake News Game Theory backend.

This module initializes core services such as:
- Database (PostgreSQL via SQLAlchemy/AsyncSession)
- Cache (Redis)
- Config & settings
- Logging
- Security
"""

# Avoid circular imports - import modules directly when needed
__all__ = []