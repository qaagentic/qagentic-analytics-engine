"""API endpoints for accessing failure clusters."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import UUID4

from qagentic_analytics.services.clustering import ClusteringService
from qagentic_analytics.services.dependecies import get_clustering_service
from qagentic_analytics.schemas.cluster import (
    FailureCluster,
    FailureClusterDetails,
    FailureClusterSummary,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["clusters"], prefix="/clusters")


@router.get("/", response_model=List[FailureClusterSummary])
async def get_clusters(
    skip: int = Query(0, description="Number of items to skip"),
    limit: int = Query(100, description="Maximum number of items to return"),
    min_failures: int = Query(1, description="Minimum number of failures in cluster"),
    status: Optional[str] = Query(None, description="Filter by status (open, closed)"),
    service: Optional[str] = Query(None, description="Filter by service name"),
    branch: Optional[str] = Query(None, description="Filter by branch"),
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """
    Get all failure clusters, optionally filtered by parameters.
    """
    try:
        clusters = await clustering_service.get_clusters(
            skip=skip,
            limit=limit,
            min_failures=min_failures,
            status=status,
            service=service,
            branch=branch,
        )
        return clusters
    except Exception as e:
        logger.error(f"Error retrieving clusters: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving clusters")


@router.get("/{cluster_id}", response_model=FailureClusterDetails)
async def get_cluster(
    cluster_id: UUID4,
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """
    Get detailed information about a specific failure cluster.
    """
    try:
        cluster = await clustering_service.get_cluster_by_id(cluster_id)
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")
        return cluster
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving cluster {cluster_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving cluster")


@router.get("/runs/{run_id}", response_model=List[FailureClusterSummary])
async def get_clusters_for_run(
    run_id: UUID4,
    clustering_service: ClusteringService = Depends(get_clustering_service),
):
    """
    Get all failure clusters associated with a specific test run.
    """
    try:
        clusters = await clustering_service.get_clusters_by_run_id(run_id)
        return clusters
    except Exception as e:
        logger.error(f"Error retrieving clusters for run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving clusters")
