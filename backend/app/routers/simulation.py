"""
Simulation router - run & fetch game theory simulations.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def simulation_root():
    """Simulation API root endpoint."""
    return {"message": "Simulation API - endpoints coming soon"}