"""FastAPI dependency injection for services."""

import logging
from functools import lru_cache

from fastapi import Depends

from qagentic_analytics.config import Settings, get_settings
from qagentic_analytics.services.clustering import ClusteringService
from qagentic_analytics.services.embedding import EmbeddingService
from qagentic_analytics.services.metrics import MetricsService
from qagentic_analytics.db import get_engine

logger = logging.getLogger(__name__)


@lru_cache
def get_embedding_service(settings: Settings = Depends(get_settings)) -> EmbeddingService:
    """Get the embedding service."""
    logger.debug("Creating embedding service instance")
    return EmbeddingService(model_name=settings.EMBEDDING_MODEL)


def get_clustering_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    settings: Settings = Depends(get_settings),
) -> ClusteringService:
    """Get the clustering service."""
    logger.debug("Creating clustering service instance")
    return ClusteringService(
        embedding_service=embedding_service,
        algorithm=settings.CLUSTERING_ALGORITHM,
        min_cluster_size=settings.CLUSTERING_MIN_CLUSTER_SIZE,
        min_samples=settings.CLUSTERING_MIN_SAMPLES,
        epsilon=settings.CLUSTERING_EPSILON,
    )


def get_metrics_service(settings: Settings = Depends(get_settings)) -> MetricsService:
    """Get the metrics service."""
    logger.debug("Creating metrics service instance")
    service = MetricsService(
        enable_flake_detection=settings.ENABLE_FLAKE_DETECTION,
        enable_trend_analysis=settings.ENABLE_TREND_ANALYSIS,
    )
    # Initialize with database engine
    service.db = get_engine()
    return service
