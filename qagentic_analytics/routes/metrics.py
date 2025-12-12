"""API endpoints for accessing test metrics and analytics."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import UUID4

from qagentic_analytics.services.dependecies import get_metrics_service
from qagentic_analytics.services.metrics import MetricsService
from qagentic_analytics.schemas.metrics import (
    MetricSummary,
    MetricTrend,
    FlakyTestsReport,
    MetricTimeRange,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"], prefix="/metrics")


@router.get("/summary", response_model=MetricSummary)
async def get_metrics_summary(
    service: Optional[str] = Query(None, description="Filter by service name"),
    branch: Optional[str] = Query(None, description="Filter by branch"),
    from_date: Optional[datetime] = Query(None, description="Start date for metrics"),
    to_date: Optional[datetime] = Query(None, description="End date for metrics"),
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """
    Get overall quality metrics summary, optionally filtered by parameters.
    """
    try:
        time_range = MetricTimeRange(from_date=from_date, to_date=to_date) if from_date or to_date else None
        
        summary = await metrics_service.get_summary(
            service=service,
            branch=branch,
            time_range=time_range,
        )
        return summary
    except Exception as e:
        logger.error(f"Error retrieving metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving metrics summary")


@router.get("/trends", response_model=List[MetricTrend])
async def get_metrics_trends(
    metric: str = Query(..., description="Metric to retrieve (pass_rate, mttr, flake_rate, etc)"),
    service: Optional[str] = Query(None, description="Filter by service name"),
    branch: Optional[str] = Query(None, description="Filter by branch"),
    from_date: Optional[datetime] = Query(None, description="Start date for metrics"),
    to_date: Optional[datetime] = Query(None, description="End date for metrics"),
    interval: str = Query("day", description="Interval for trend data (hour, day, week, month)"),
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """
    Get trending metrics over time, optionally filtered by parameters.
    """
    try:
        time_range = MetricTimeRange(from_date=from_date, to_date=to_date) if from_date or to_date else None
        
        trends = await metrics_service.get_trends(
            metric=metric,
            service=service,
            branch=branch,
            time_range=time_range,
            interval=interval,
        )
        return trends
    except Exception as e:
        logger.error(f"Error retrieving metrics trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving metrics trends")


@router.get("/flaky", response_model=FlakyTestsReport)
async def get_flaky_tests(
    service: Optional[str] = Query(None, description="Filter by service name"),
    branch: Optional[str] = Query(None, description="Filter by branch"),
    min_flake_rate: float = Query(0.05, description="Minimum flake rate to consider (0-1)"),
    min_executions: int = Query(5, description="Minimum number of executions to analyze"),
    from_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    to_date: Optional[datetime] = Query(None, description="End date for analysis"),
    metrics_service: MetricsService = Depends(get_metrics_service),
):
    """
    Get flaky tests analysis, optionally filtered by parameters.
    """
    try:
        time_range = MetricTimeRange(from_date=from_date, to_date=to_date) if from_date or to_date else None
        
        flaky_report = await metrics_service.get_flaky_tests(
            service=service,
            branch=branch,
            min_flake_rate=min_flake_rate,
            min_executions=min_executions,
            time_range=time_range,
        )
        return flaky_report
    except Exception as e:
        logger.error(f"Error retrieving flaky tests report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving flaky tests report")
