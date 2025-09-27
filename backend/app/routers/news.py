"""
News router - news submission & ML classification.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def news_root():
    """News API root endpoint."""
    return {"message": "News API - endpoints coming soon"}