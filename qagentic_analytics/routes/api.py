"""API endpoints for the Analytics Engine."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, Query
from sqlalchemy.ext.asyncio import AsyncSession

from qagentic_analytics.db import get_db_session
from qagentic_analytics.services.clustering_service import ClusteringService
from qagentic_analytics.services.pattern_service import PatternService
from qagentic_analytics.services.trend_service import TrendService
from qagentic_analytics.services.metrics_service import MetricsService
from qagentic_analytics.schemas.requests import (
    ClusteringRequest,
    PatternAnalysisRequest,
    TrendAnalysisRequest
)
from qagentic_analytics.schemas.responses import (
    ClusteringResponse,
    PatternAnalysisResponse,
    TrendAnalysisResponse,
    MetricsResponse
)

router = APIRouter(prefix="/api/v1")

# Service instances
clustering_service = ClusteringService()
pattern_service = PatternService()
trend_service = TrendService()
metrics_service = MetricsService()

@router.post("/clusters", response_model=ClusteringResponse)
async def cluster_failures(
    request: ClusteringRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Cluster similar test failures.
    
    Args:
        request: Clustering parameters
        session: Database session
        
    Returns:
        Clustering results
    """
    try:
        clusters = await clustering_service.cluster_failures(
            time_window=request.time_window,
            min_failures=request.min_failures,
            similarity_threshold=request.similarity_threshold
        )
        
        return ClusteringResponse(
            clusters=clusters,
            total_clusters=len(clusters),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clustering failures: {str(e)}"
        )

@router.post("/patterns", response_model=PatternAnalysisResponse)
async def analyze_patterns(
    request: PatternAnalysisRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Detect patterns in test failures.
    
    Args:
        request: Pattern analysis parameters
        session: Database session
        
    Returns:
        Detected patterns
    """
    try:
        patterns = await pattern_service.detect_patterns(
            time_window=request.time_window,
            min_occurrences=request.min_occurrences,
            confidence_threshold=request.confidence_threshold
        )
        
        return PatternAnalysisResponse(
            patterns=patterns,
            total_patterns=len(patterns),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting patterns: {str(e)}"
        )

@router.post("/trends", response_model=TrendAnalysisResponse)
async def analyze_trends(
    request: TrendAnalysisRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Analyze test quality trends.
    
    Args:
        request: Trend analysis parameters
        session: Database session
        
    Returns:
        Trend analysis results
    """
    try:
        trends = await trend_service.analyze_trends(
            time_window=request.time_window,
            interval=request.interval,
            min_data_points=request.min_data_points
        )
        
        return TrendAnalysisResponse(
            trends=trends,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing trends: {str(e)}"
        )

@router.get("/metrics/{test_run_id}", response_model=MetricsResponse)
async def get_metrics(
    test_run_id: str,
    realtime: bool = Query(False, description="Whether to compute metrics in real-time"),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Get metrics for a test run.
    
    Args:
        test_run_id: ID of the test run
        realtime: Whether to compute metrics in real-time
        session: Database session
        
    Returns:
        Test run metrics
    """
    try:
        metrics = await metrics_service.compute_metrics(
            test_run_id=test_run_id,
            realtime=realtime
        )
        
        return MetricsResponse(
            test_run_id=test_run_id,
            metrics=metrics,
            timestamp=datetime.utcnow()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing metrics: {str(e)}"
        )

@router.get("/metrics/history")
async def get_metrics_history(
    window: int = Query(30, description="Time window in days"),
    interval: str = Query("1D", description="Aggregation interval"),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Get historical metrics with aggregations.
    
    Args:
        window: Time window in days
        interval: Aggregation interval
        session: Database session
        
    Returns:
        Historical metrics
    """
    try:
        metrics = await metrics_service.get_historical_metrics(
            time_window=timedelta(days=window),
            interval=interval
        )
        
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving historical metrics: {str(e)}"
        )

@router.websocket("/metrics/stream/{test_run_id}")
async def stream_metrics(
    websocket: WebSocket,
    test_run_id: Optional[str] = None
):
    """
    Stream real-time metrics updates.
    
    Args:
        websocket: WebSocket connection
        test_run_id: Optional test run ID to filter updates
    """
    await metrics_service.subscribe_to_metrics(
        websocket=websocket,
        test_run_id=test_run_id
    )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow()
    }
