"""
Social network router - social graph APIs.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def social_root():
    """Social network API root endpoint."""
    return {"message": "Social Network API - endpoints coming soon"}