"""Main entry point for the QAagentic Analytics Engine."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qagentic_common.utils.db import init_db, shutdown_db
from qagentic_common.cache import initialize_global_cache, close_global_cache
from qagentic_common.realtime import initialize_global_publisher, close_global_publisher

from qagentic_analytics.config import get_settings
from qagentic_analytics.routes import clusters, metrics, dashboard, visualization, custom_metrics

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler for startup/shutdown events.

    Initialize database connection, cache, and real-time publisher on startup.
    """
    # Startup
    logger.info("Initializing Analytics Engine")

    # Initialize database
    logger.info("Initializing database connections")
    await init_db(settings.DATABASE_URL,
                  pool_size=settings.DB_POOL_SIZE,
                  max_overflow=settings.DB_MAX_OVERFLOW,
                  pool_recycle=settings.DB_POOL_RECYCLE)

    # Initialize cache
    if hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
        try:
            logger.info("Initializing cache manager")
            await initialize_global_cache(settings.REDIS_URL, namespace="analytics")
            logger.info("Cache manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}")

    # Initialize real-time publisher
    if hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
        try:
            logger.info("Initializing real-time publisher")
            await initialize_global_publisher(settings.REDIS_URL)
            logger.info("Real-time publisher initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize real-time publisher: {e}")

    logger.info("Analytics Engine startup complete")
    yield

    # Shutdown
    logger.info("Shutting down Analytics Engine")
    await close_global_cache()
    await close_global_publisher()
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
        allow_origins=settings.ALLOW_ORIGINS,
        allow_origin_regex=settings.ALLOW_ORIGIN_REGEX,
        allow_credentials=settings.ALLOW_CREDENTIALS,
        allow_methods=settings.ALLOW_METHODS,
        allow_headers=settings.ALLOW_HEADERS,
    )

    # Add API routes
    app.include_router(clusters.router, prefix=settings.API_V1_PREFIX)
    app.include_router(metrics.router, prefix=settings.API_V1_PREFIX)
    app.include_router(custom_metrics.router, prefix=settings.API_V1_PREFIX)
    app.include_router(dashboard.router, prefix=settings.API_V1_PREFIX)
    app.include_router(visualization.router)

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "analytics-engine"}

    return app


app = create_app()
