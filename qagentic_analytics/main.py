"""Main entry point for the QAagentic Analytics Engine."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qagentic_common.utils.db import init_db, shutdown_db

from qagentic_analytics.config import get_settings
from qagentic_analytics.routes import clusters, metrics, dashboard, visualization

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler for startup/shutdown events.
    
    Initialize database connection on startup and close it on shutdown.
    """
    # Startup
    logger.info("Initializing Analytics Engine database connections")
    await init_db(settings.DATABASE_URL, 
                  pool_size=settings.DB_POOL_SIZE,
                  max_overflow=settings.DB_MAX_OVERFLOW,
                  pool_recycle=settings.DB_POOL_RECYCLE)
    
    # Initialize any models or resources for analytics
    logger.info("Analytics Engine startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down Analytics Engine")
    await shutdown_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Analytics Engine for QAagentic Platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.ALLOW_ORIGIN_REGEX,
        allow_credentials=settings.ALLOW_CREDENTIALS,
        allow_methods=settings.ALLOW_METHODS,
        allow_headers=settings.ALLOW_HEADERS,
    )

    # Add API routes
    app.include_router(clusters.router, prefix=settings.API_V1_PREFIX)
    app.include_router(metrics.router, prefix=settings.API_V1_PREFIX)
    app.include_router(dashboard.router, prefix=settings.API_V1_PREFIX)
    app.include_router(visualization.router)

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "analytics-engine"}

    return app


app = create_app()
