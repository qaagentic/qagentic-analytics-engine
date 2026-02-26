"""
Custom Metrics API routes for QAagentic Analytics Engine.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from qagentic_common.models.custom_metrics import (
    MetricDefinitionCreate,
    MetricDefinitionUpdate,
    MetricDefinitionResponse,
    MetricEvaluationRequest,
    MetricEvaluationResponse,
    MetricTimeSeriesRequest,
    MetricTimeSeriesResponse,
)
from qagentic_analytics.services.custom_metrics_service import get_custom_metrics_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/custom-metrics", tags=["custom-metrics"])


@router.post("/", response_model=MetricDefinitionResponse, status_code=201)
async def create_metric(
    metric_data: MetricDefinitionCreate,
    # TODO: Get from auth token
    organization_id: Optional[UUID] = None,
    created_by: Optional[UUID] = None
):
    """
    Create a new custom metric definition.

    Args:
        metric_data: Metric definition

    Returns:
        Created metric definition

    Example:
        ```json
        {
          "name": "pass_rate",
          "display_name": "Test Pass Rate",
          "description": "Percentage of tests that passed",
          "category": "Quality",
          "metric_type": "ratio",
          "definition": {
            "numerator": "COUNT(*) FILTER (WHERE status = 'passed')",
            "denominator": "COUNT(*)",
            "filters": {}
          },
          "schedule": "hourly",
          "enabled": true
        }
        ```
    """
    try:
        service = get_custom_metrics_service()
        metric = await service.create_metric(metric_data, organization_id, created_by)
        return metric
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error creating metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to create metric")


@router.get("/", response_model=List[MetricDefinitionResponse])
async def list_metrics(
    organization_id: Optional[UUID] = None,
    enabled_only: bool = Query(True, description="Only return enabled metrics"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """
    List all custom metrics.

    Args:
        organization_id: Filter by organization
        enabled_only: Only return enabled metrics
        category: Filter by category

    Returns:
        List of metric definitions
    """
    try:
        service = get_custom_metrics_service()
        metrics = await service.list_metrics(organization_id, enabled_only, category)
        return metrics
    except Exception as e:
        logger.exception(f"Error listing metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to list metrics")


@router.get("/{metric_id}", response_model=MetricDefinitionResponse)
async def get_metric(metric_id: UUID):
    """
    Get metric definition by ID.

    Args:
        metric_id: Metric UUID

    Returns:
        Metric definition
    """
    try:
        service = get_custom_metrics_service()
        metric = await service.get_metric(metric_id)

        if not metric:
            raise HTTPException(status_code=404, detail=f"Metric not found: {metric_id}")

        return metric
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metric")


@router.put("/{metric_id}", response_model=MetricDefinitionResponse)
async def update_metric(metric_id: UUID, updates: MetricDefinitionUpdate):
    """
    Update metric definition.

    Args:
        metric_id: Metric UUID
        updates: Fields to update

    Returns:
        Updated metric definition
    """
    try:
        service = get_custom_metrics_service()
        metric = await service.update_metric(metric_id, updates)

        if not metric:
            raise HTTPException(status_code=404, detail=f"Metric not found: {metric_id}")

        return metric
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to update metric")


@router.delete("/{metric_id}", status_code=204)
async def delete_metric(metric_id: UUID):
    """
    Delete metric definition.

    Args:
        metric_id: Metric UUID
    """
    try:
        service = get_custom_metrics_service()
        deleted = await service.delete_metric(metric_id)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Metric not found: {metric_id}")

        return JSONResponse(status_code=204, content=None)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete metric")


@router.post("/evaluate", response_model=MetricEvaluationResponse)
async def evaluate_metric(request: MetricEvaluationRequest):
    """
    Evaluate a custom metric and return the result.

    Args:
        request: Metric evaluation request

    Returns:
        Evaluation result with calculated value

    Example:
        ```json
        {
          "metric_id": "550e8400-e29b-41d4-a716-446655440000",
          "start_time": "2024-01-01T00:00:00Z",
          "end_time": "2024-01-31T23:59:59Z",
          "dimensions": {
            "branch": "main",
            "environment": "production"
          }
        }
        ```
    """
    try:
        service = get_custom_metrics_service()
        result = await service.evaluate_metric(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error evaluating metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate metric")


@router.post("/timeseries", response_model=MetricTimeSeriesResponse)
async def get_metric_timeseries(request: MetricTimeSeriesRequest):
    """
    Get time-series data for a metric.

    Args:
        request: Time-series request

    Returns:
        Time-series data points

    Example:
        ```json
        {
          "metric_id": "550e8400-e29b-41d4-a716-446655440000",
          "start_time": "2024-01-01T00:00:00Z",
          "end_time": "2024-01-31T23:59:59Z",
          "interval": "1d",
          "dimensions": {"branch": "main"}
        }
        ```
    """
    try:
        service = get_custom_metrics_service()
        result = await service.get_metric_timeseries(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting time-series: {e}")
        raise HTTPException(status_code=500, detail="Failed to get time-series data")


@router.get("/categories", response_model=List[str])
async def list_categories(organization_id: Optional[UUID] = None):
    """
    Get list of available metric categories.

    Args:
        organization_id: Filter by organization

    Returns:
        List of category names
    """
    try:
        service = get_custom_metrics_service()
        metrics = await service.list_metrics(organization_id, enabled_only=False)

        categories = set(m.category for m in metrics if m.category)
        return sorted(list(categories))
    except Exception as e:
        logger.exception(f"Error listing categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to list categories")


@router.get("/examples", response_model=List[dict])
async def get_metric_examples():
    """
    Get example metric definitions for common use cases.

    Returns:
        List of example metric definitions
    """
    examples = [
        {
            "name": "pass_rate",
            "display_name": "Test Pass Rate",
            "description": "Percentage of tests that passed",
            "category": "Quality",
            "metric_type": "ratio",
            "definition": {
                "numerator": "COUNT(*) FILTER (WHERE status = 'passed')",
                "denominator": "COUNT(*)",
                "filters": {}
            },
            "schedule": "hourly"
        },
        {
            "name": "failure_count",
            "display_name": "Failed Tests Count",
            "description": "Number of failed tests",
            "category": "Quality",
            "metric_type": "count",
            "definition": {
                "expression": "COUNT(*) FILTER (WHERE status = 'failed')",
                "filters": {},
                "group_by": []
            },
            "schedule": "realtime"
        },
        {
            "name": "p95_duration",
            "display_name": "95th Percentile Duration",
            "description": "95th percentile of test execution time",
            "category": "Performance",
            "metric_type": "percentile",
            "definition": {
                "field": "duration",
                "percentile": 95,
                "filters": {}
            },
            "schedule": "hourly"
        },
        {
            "name": "avg_duration",
            "display_name": "Average Test Duration",
            "description": "Average time to run tests",
            "category": "Performance",
            "metric_type": "average",
            "definition": {
                "field": "duration",
                "filters": {}
            },
            "schedule": "hourly"
        },
        {
            "name": "flaky_rate",
            "display_name": "Flaky Test Rate",
            "description": "Percentage of tests marked as flaky",
            "category": "Reliability",
            "metric_type": "ratio",
            "definition": {
                "numerator": "COUNT(*) FILTER (WHERE is_flaky = true)",
                "denominator": "COUNT(*)",
                "filters": {}
            },
            "schedule": "daily"
        },
        {
            "name": "coverage",
            "display_name": "Code Coverage",
            "description": "Percentage of code covered by tests",
            "category": "Coverage",
            "metric_type": "average",
            "definition": {
                "field": "coverage_percentage",
                "filters": {}
            },
            "schedule": "daily"
        }
    ]

    return examples
