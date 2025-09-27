# backend/app/api/__init__.py

from fastapi import APIRouter
from app.api.v1 import classifier, simulation, equilibrium, analysis, data

# Create main API router
api_router = APIRouter()

# Include all v1 routers with appropriate prefixes and tags
api_router.include_router(
    classifier.router,
    prefix="/v1/classifier",
    tags=["classifier", "machine-learning"],
    responses={404: {"description": "Not found"}}
)

api_router.include_router(
    simulation.router,
    prefix="/v1/simulation",
    tags=["simulation", "game-theory"],
    responses={404: {"description": "Not found"}}
)

api_router.include_router(
    equilibrium.router,
    prefix="/v1/equilibrium",
    tags=["equilibrium", "game-theory", "analysis"],
    responses={404: {"description": "Not found"}}
)

api_router.include_router(
    analysis.router,
    prefix="/v1/analysis",
    tags=["analysis", "network", "visualization"],
    responses={404: {"description": "Not found"}}
)

api_router.include_router(
    data.router,
    prefix="/v1/data",
    tags=["data", "datasets", "preprocessing"],
    responses={404: {"description": "Not found"}}
)