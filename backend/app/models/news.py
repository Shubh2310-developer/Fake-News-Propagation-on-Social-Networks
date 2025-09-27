"""
News article and classification models.
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class NewsArticle(Base):
    """News article model for storing news content."""

    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    source = Column(String(200), nullable=True)
    url = Column(String(1000), nullable=True)
    author = Column(String(200), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to classifications
    classifications = relationship("Classification", back_populates="article")


class Classification(Base):
    """Classification results for news articles."""

    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    is_fake = Column(Boolean, nullable=False)
    confidence_score = Column(Float, nullable=False)
    classification_details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to article
    article = relationship("NewsArticle", back_populates="classifications")