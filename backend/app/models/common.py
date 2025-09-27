# backend/app/models/common.py

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Base configuration for all schemas
# ---------------------------------------------------------
class CommonBaseModel(BaseModel):
    """Base class for all Pydantic models with shared config."""
    model_config = {
        "from_attributes": True,
        "validate_by_name": True,
        "validate_assignment": True
    }

# ---------------------------------------------------------
# Standard response wrappers
# ---------------------------------------------------------
class ResponseMetadata(BaseModel):
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(CommonBaseModel):
    error: str
    detail: Optional[str] = None
    metadata: Optional[ResponseMetadata] = None

class SuccessResponse(CommonBaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[ResponseMetadata] = None

# ---------------------------------------------------------
# Pagination / query helpers
# ---------------------------------------------------------
class Pagination(BaseModel):
    page: int = Field(1, ge=1)
    size: int = Field(10, ge=1, le=100)
    total: Optional[int] = None

# ---------------------------------------------------------
# Generic identifiers
# ---------------------------------------------------------
class IDResponse(CommonBaseModel):
    id: str = Field(..., description="Unique identifier for resource")
    created_at: datetime = Field(default_factory=datetime.utcnow)