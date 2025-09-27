"""
Authentication router - login, register, token management.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def auth_root():
    """Authentication router root endpoint."""
    return {"message": "Authentication API - endpoints coming soon"}