# backend/ml_models/__init__.py

"""
ML Models package for the Fake News Game Theory backend.

This package provides a comprehensive framework for machine learning operations
including text classification, feature extraction, and model management.

The package is organized into:
- classifiers: Abstract base classes and concrete classifier implementations
- preprocessing: Text processing and feature extraction utilities

This modular design ensures extensibility and maintainability for all ML operations.
"""

from .classifiers import BaseClassifier
from .preprocessing import TextProcessor, FeatureExtractor, create_news_processor, create_social_media_processor

__all__ = [
    # Classifiers
    "BaseClassifier",

    # Preprocessing
    "TextProcessor",
    "FeatureExtractor",
    "create_news_processor",
    "create_social_media_processor",
]

__version__ = "1.0.0"