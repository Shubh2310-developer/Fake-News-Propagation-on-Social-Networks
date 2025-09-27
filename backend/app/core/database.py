"""
Database connection management for the Fake News Game Theory backend.
Uses SQLAlchemy (async) with PostgreSQL.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings


# -------------------------
# SQLAlchemy Base
# -------------------------
Base = declarative_base()

# -------------------------
# Engine & Session
# -------------------------
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    future=True,
    echo=settings.DEBUG,  # Log SQL queries in debug mode
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# -------------------------
# Dependency for FastAPI
# -------------------------
async def get_db():
    """
    FastAPI dependency that provides a DB session.
    """
    async with SessionLocal() as session:
        yield session


# -------------------------
# Database Init (for startup)
# -------------------------
async def init_db():
    """
    Initialize database connection and run migrations if needed.
    """
    async with engine.begin() as conn:
        # Create tables if they don't exist
        # (better practice: use Alembic for real migrations)
        await conn.run_sync(Base.metadata.create_all)


# -------------------------
# Database Cleanup (for shutdown)
# -------------------------
async def close_db():
    """Close database connections."""
    await engine.dispose()
    print("🔌 Database connections closed")