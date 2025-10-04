# backend/app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os

# Local imports (routers, DB, cache, etc.)
from app.routers import auth, news, social, simulation
from app.api import api_router
from app.core.database import init_db, close_db
from app.core.cache import init_redis, close_redis
from app.core.logging import configure_logging, get_logger
from app.services.classifier_service import ClassifierService
from app.services.model_loader import SimpleEnsemble, CalibratedEnsemble  # Import for ensemble loading

# Load environment variables
PROJECT_NAME = os.getenv("PROJECT_NAME", "Fake News Game Theory")
ENV = os.getenv("ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")


# --------------------------
# Lifespan Context Manager
# --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Configure logging first
    configure_logging()
    logger = get_logger(__name__)

    # Startup
    logger.info("Starting application...")
    await init_db()
    await init_redis()

    # Load ML models
    logger.info("Loading ML models...")
    classifier_service = ClassifierService()
    loading_results = await classifier_service.load_models()
    logger.info(f"ML models loaded: {loading_results}")

    # Store classifier service in app state for access in routes
    app.state.classifier_service = classifier_service

    logger.info("✅ Application startup complete.")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await close_db()
    await close_redis()
    logger.info("🛑 Application shutdown complete.")


# --------------------------
# Application Factory
# --------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title=PROJECT_NAME,
        description="Fake News Detection & Game Theory Simulation API",
        version="1.0.0",
        debug=DEBUG,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS if CORS_ORIGINS != [""] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the main API router
    app.include_router(api_router, prefix="/api")

    # Legacy routers (can be kept for backward compatibility)
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(news.router, prefix="/news", tags=["News"])
    app.include_router(social.router, prefix="/social", tags=["Social Network"])
    app.include_router(simulation.router, prefix="/simulation", tags=["Simulation"])

    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        return JSONResponse(
            {"status": "ok", "environment": ENV, "project": PROJECT_NAME}
        )

    return app


# --------------------------
# Create app instance
# --------------------------
app = create_app()


# --------------------------
# Local run (dev only)
# --------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        workers=1,
    )