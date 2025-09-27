"""
Social network models for nodes and edges.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class SocialNode(Base):
    """Social network node representing users/entities."""

    __tablename__ = "social_nodes"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(100), unique=True, index=True, nullable=False)
    node_type = Column(String(50), nullable=False)  # user, bot, organization
    influence_score = Column(Float, default=0.0)
    trust_score = Column(Float, default=0.5)
    activity_level = Column(Float, default=0.0)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    outgoing_edges = relationship("SocialEdge", foreign_keys="SocialEdge.source_id", back_populates="source")
    incoming_edges = relationship("SocialEdge", foreign_keys="SocialEdge.target_id", back_populates="target")


class SocialEdge(Base):
    """Social network edge representing connections between nodes."""

    __tablename__ = "social_edges"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("social_nodes.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("social_nodes.id"), nullable=False)
    edge_type = Column(String(50), nullable=False)  # follow, friend, mention, retweet
    weight = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    source = relationship("SocialNode", foreign_keys=[source_id], back_populates="outgoing_edges")
    target = relationship("SocialNode", foreign_keys=[target_id], back_populates="incoming_edges")