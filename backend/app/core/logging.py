"""
Logging configuration for the Fake News Game Theory backend.
Provides structured logging for development and production.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from app.core.config import settings


# -------------------------
# Log Formatters
# -------------------------
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    """
    Configure global logging based on environment.
    - Development: logs to console (colored if supported).
    - Production: logs to file + console with rotation.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    handlers = []

    # Console Handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handlers.append(console_handler)

    # File Handler (only in non-debug mode)
    if not settings.DEBUG:
        file_handler = RotatingFileHandler(
            "logs/app.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
    )

    logging.getLogger("uvicorn").handlers = []  # prevent duplicate logs
    logging.getLogger("uvicorn.access").handlers = []


# -------------------------
# Logger Utility
# -------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    Example:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return logging.getLogger(name)