"""
Configuration module for the Fake News Game Theory backend.
Uses Pydantic BaseSettings to load environment variables.
"""

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # -------------------
    # App Settings
    # -------------------
    PROJECT_NAME: str = "Fake News Game Theory"
    ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "super-secret-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # -------------------
    # Database
    # -------------------
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/fakenews_db"

    # -------------------
    # Redis
    # -------------------
    REDIS_URL: str = "redis://localhost:6379/0"
    redis_url: str = "redis://localhost:6379/0"  # Alias for cache.py compatibility
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # -------------------
    # CORS
    # -------------------
    CORS_ORIGINS: List[AnyHttpUrl] = []

    # -------------------
    # ML Model Settings
    # -------------------
    MODEL_CACHE_SIZE: int = 100
    BERT_MODEL_NAME: str = "bert-base-uncased"
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 32

    # -------------------
    # Simulation Settings
    # -------------------
    DEFAULT_SIMULATION_STEPS: int = 100
    MAX_NETWORK_SIZE: int = 10000
    DEFAULT_LEARNING_RATE: float = 0.01

    # -------------------
    # Logging & Monitoring
    # -------------------
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"  # Ignore extra fields from .env that aren't defined in the model
    }


# Global settings instance
settings = Settings()