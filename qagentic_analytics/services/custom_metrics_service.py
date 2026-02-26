"""
Custom Metrics Service for QAagentic Analytics Engine.

Handles custom metric definition, evaluation, and storage.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from qagentic_common.models.custom_metrics import (
    CustomMetricDefinition,
    CustomMetricValue,
    MetricType,
    MetricDefinitionCreate,
    MetricDefinitionUpdate,
    MetricDefinitionResponse,
    MetricValueResponse,
    MetricEvaluationRequest,
    MetricEvaluationResponse,
    MetricTimeSeriesRequest,
    MetricTimeSeriesResponse,
    MetricTimeSeriesPoint,
    RatioMetricDefinition,
    CountMetricDefinition,
    PercentileMetricDefinition,
    CustomFormulaDefinition,
)
from qagentic_common.utils.db import get_db_session
from qagentic_common.cache import get_global_cache
from qagentic_common.realtime import get_global_publisher

logger = logging.getLogger(__name__)


class CustomMetricsService:
    """Service for managing and evaluating custom metrics."""

    def __init__(self):
        """Initialize custom metrics service."""
        self.cache = None
        self.publisher = None

        try:
            self.cache = get_global_cache()
        except RuntimeError:
            logger.warning("Cache not initialized, custom metrics will not be cached")

        try:
            self.publisher = get_global_publisher()
        except RuntimeError:
            logger.warning("Real-time publisher not initialized")

    async def create_metric(
        self,
        metric_data: MetricDefinitionCreate,
        organization_id: Optional[UUID] = None,
        created_by: Optional[UUID] = None
    ) -> MetricDefinitionResponse:
        """
        Create a new custom metric definition.

        Args:
            metric_data: Metric definition data
            organization_id: Organization ID for multi-tenancy
            created_by: User ID creating the metric

        Returns:
            Created metric definition
        """
        async with get_db_session() as session:
            # Check if metric with same name exists
            existing = await session.execute(
                select(CustomMetricDefinition).where(
                    and_(
                        CustomMetricDefinition.name == metric_data.name,
                        CustomMetricDefinition.organization_id == organization_id
                    )
                )
            )
            if existing.scalar_one_or_none():
                raise ValueError(f"Metric with name '{metric_data.name}' already exists")

            # Create metric
            metric = CustomMetricDefinition(
                **metric_data.model_dump(),
                organization_id=organization_id,
                created_by=created_by
            )

            session.add(metric)
            await session.commit()
            await session.refresh(metric)

            logger.info(f"Created custom metric: {metric.name} (ID: {metric.id})")

            return MetricDefinitionResponse.model_validate(metric)

    async def get_metric(self, metric_id: UUID) -> Optional[MetricDefinitionResponse]:
        """Get metric definition by ID."""
        async with get_db_session() as session:
            result = await session.execute(
                select(CustomMetricDefinition).where(CustomMetricDefinition.id == metric_id)
            )
            metric = result.scalar_one_or_none()

            if metric:
                return MetricDefinitionResponse.model_validate(metric)
            return None

    async def list_metrics(
        self,
        organization_id: Optional[UUID] = None,
        enabled_only: bool = True,
        category: Optional[str] = None
    ) -> List[MetricDefinitionResponse]:
        """
        List all custom metrics.

        Args:
            organization_id: Filter by organization
            enabled_only: Only return enabled metrics
            category: Filter by category

        Returns:
            List of metric definitions
        """
        async with get_db_session() as session:
            query = select(CustomMetricDefinition)

            filters = []
            if organization_id:
                filters.append(CustomMetricDefinition.organization_id == organization_id)
            if enabled_only:
                filters.append(CustomMetricDefinition.enabled == True)
            if category:
                filters.append(CustomMetricDefinition.category == category)

            if filters:
                query = query.where(and_(*filters))

            result = await session.execute(query.order_by(CustomMetricDefinition.name))
            metrics = result.scalars().all()

            return [MetricDefinitionResponse.model_validate(m) for m in metrics]

    async def update_metric(
        self,
        metric_id: UUID,
        updates: MetricDefinitionUpdate
    ) -> Optional[MetricDefinitionResponse]:
        """Update metric definition."""
        async with get_db_session() as session:
            result = await session.execute(
                select(CustomMetricDefinition).where(CustomMetricDefinition.id == metric_id)
            )
            metric = result.scalar_one_or_none()

            if not metric:
                return None

            # Update fields
            for field, value in updates.model_dump(exclude_unset=True).items():
                setattr(metric, field, value)

            metric.updated_at = datetime.utcnow()

            await session.commit()
            await session.refresh(metric)

            logger.info(f"Updated custom metric: {metric.name}")

            return MetricDefinitionResponse.model_validate(metric)

    async def delete_metric(self, metric_id: UUID) -> bool:
        """Delete metric definition."""
        async with get_db_session() as session:
            result = await session.execute(
                select(CustomMetricDefinition).where(CustomMetricDefinition.id == metric_id)
            )
            metric = result.scalar_one_or_none()

            if not metric:
                return False

            await session.delete(metric)
            await session.commit()

            logger.info(f"Deleted custom metric: {metric.name}")

            return True

    async def evaluate_metric(
        self,
        request: MetricEvaluationRequest
    ) -> MetricEvaluationResponse:
        """
        Evaluate a custom metric and return the result.

        Args:
            request: Metric evaluation request

        Returns:
            Evaluation result with calculated value
        """
        start_time = time.time()

        # Get metric definition
        metric = await self.get_metric(request.metric_id)
        if not metric:
            raise ValueError(f"Metric not found: {request.metric_id}")

        # Check cache first
        cache_key = f"metric:{metric.id}:{request.dimensions}"
        if self.cache:
            cached_value = await self.cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for metric {metric.name}")
                return MetricEvaluationResponse(**cached_value)

        # Evaluate based on metric type
        value = await self._evaluate_metric_by_type(metric, request)

        # Calculate execution time
        calculation_time_ms = (time.time() - start_time) * 1000

        response = MetricEvaluationResponse(
            metric_id=metric.id,
            metric_name=metric.name,
            value=value,
            dimensions=request.dimensions or {},
            timestamp=datetime.utcnow(),
            calculation_time_ms=calculation_time_ms
        )

        # Cache the result
        if self.cache:
            await self.cache.set(
                cache_key,
                response.model_dump(),
                ttl=timedelta(minutes=5)
            )

        # Store the value
        await self._store_metric_value(metric.id, value, request.dimensions or {})

        # Publish real-time update
        if self.publisher:
            await self.publisher.publish_metric_update(
                metric_name=metric.name,
                value=value,
                dimensions=request.dimensions or {}
            )

        logger.info(f"Evaluated metric {metric.name}: {value} ({calculation_time_ms:.2f}ms)")

        return response

    async def _evaluate_metric_by_type(
        self,
        metric: MetricDefinitionResponse,
        request: MetricEvaluationRequest
    ) -> float:
        """Evaluate metric based on its type."""
        metric_type = MetricType(metric.metric_type)

        if metric_type == MetricType.RATIO:
            return await self._evaluate_ratio_metric(metric, request)
        elif metric_type == MetricType.COUNT:
            return await self._evaluate_count_metric(metric, request)
        elif metric_type == MetricType.PERCENTILE:
            return await self._evaluate_percentile_metric(metric, request)
        elif metric_type == MetricType.AVERAGE:
            return await self._evaluate_average_metric(metric, request)
        elif metric_type == MetricType.SUM:
            return await self._evaluate_sum_metric(metric, request)
        elif metric_type == MetricType.CUSTOM:
            return await self._evaluate_custom_metric(metric, request)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")

    async def _evaluate_ratio_metric(
        self,
        metric: MetricDefinitionResponse,
        request: MetricEvaluationRequest
    ) -> float:
        """Evaluate ratio metric (e.g., pass_rate = passed / total)."""
        definition = RatioMetricDefinition(**metric.definition)

        # Build query
        query = f"""
        SELECT
            ({definition.numerator})::float AS numerator,
            ({definition.denominator})::float AS denominator
        FROM qagentic.test_runs
        WHERE 1=1
        """

        # Add time range filter
        if request.start_time:
            query += f" AND created_at >= '{request.start_time}'"
        if request.end_time:
            query += f" AND created_at <= '{request.end_time}'"

        # Add dimension filters
        dimensions = {**definition.filters, **request.dimensions}
        for key, value in dimensions.items():
            query += f" AND {key} = '{value}'"

        async with get_db_session() as session:
            result = await session.execute(text(query))
            row = result.first()

            if row and row.denominator and row.denominator != 0:
                return (row.numerator / row.denominator) * 100  # Return as percentage
            return 0.0

    async def _evaluate_count_metric(
        self,
        metric: MetricDefinitionResponse,
        request: MetricEvaluationRequest
    ) -> float:
        """Evaluate count metric."""
        definition = CountMetricDefinition(**metric.definition)

        query = f"""
        SELECT {definition.expression} AS count
        FROM qagentic.test_runs
        WHERE 1=1
        """

        if request.start_time:
            query += f" AND created_at >= '{request.start_time}'"
        if request.end_time:
            query += f" AND created_at <= '{request.end_time}'"

        dimensions = {**definition.filters, **request.dimensions}
        for key, value in dimensions.items():
            query += f" AND {key} = '{value}'"

        if definition.group_by:
            query += f" GROUP BY {', '.join(definition.group_by)}"

        async with get_db_session() as session:
            result = await session.execute(text(query))
            row = result.first()

            return float(row.count) if row else 0.0

    async def _evaluate_percentile_metric(
        self,
        metric: MetricDefinitionResponse,
        request: MetricEvaluationRequest
    ) -> float:
        """Evaluate percentile metric."""
        definition = PercentileMetricDefinition(**metric.definition)

        query = f"""
        SELECT PERCENTILE_CONT({definition.percentile / 100.0})
               WITHIN GROUP (ORDER BY {definition.field}) AS percentile
        FROM qagentic.test_runs
        WHERE 1=1
        """

        if request.start_time:
            query += f" AND created_at >= '{request.start_time}'"
        if request.end_time:
            query += f" AND created_at <= '{request.end_time}'"

        dimensions = {**definition.filters, **request.dimensions}
        for key, value in dimensions.items():
            query += f" AND {key} = '{value}'"

        async with get_db_session() as session:
            result = await session.execute(text(query))
            row = result.first()

            return float(row.percentile) if row and row.percentile else 0.0

    async def _evaluate_average_metric(
        self,
        metric: MetricDefinitionResponse,
        request: MetricEvaluationRequest
    ) -> float:
        """Evaluate average metric."""
        # Similar to count but with AVG
        definition = metric.definition
        field = definition.get('field', 'duration')

        query = f"""
        SELECT AVG({field}) AS average
        FROM qagentic.test_runs
        WHERE 1=1
        """

        if request.start_time:
            query += f" AND created_at >= '{request.start_time}'"
        if request.end_time:
            query += f" AND created_at <= '{request.end_time}'"

        filters = definition.get('filters', {})
        dimensions = {**filters, **request.dimensions}
        for key, value in dimensions.items():
            query += f" AND {key} = '{value}'"

        async with get_db_session() as session:
            result = await session.execute(text(query))
            row = result.first()

            return float(row.average) if row and row.average else 0.0

    async def _evaluate_sum_metric(
        self,
        metric: MetricDefinitionResponse,
        request: MetricEvaluationRequest
    ) -> float:
        """Evaluate sum metric."""
        definition = metric.definition
        field = definition.get('field', 'total_tests')

        query = f"""
        SELECT SUM({field}) AS total
        FROM qagentic.test_runs
        WHERE 1=1
        """

        if request.start_time:
            query += f" AND created_at >= '{request.start_time}'"
        if request.end_time:
            query += f" AND created_at <= '{request.end_time}'"

        filters = definition.get('filters', {})
        dimensions = {**filters, **request.dimensions}
        for key, value in dimensions.items():
            query += f" AND {key} = '{value}'"

        async with get_db_session() as session:
            result = await session.execute(text(query))
            row = result.first()

            return float(row.total) if row and row.total else 0.0

    async def _evaluate_custom_metric(
        self,
        metric: MetricDefinitionResponse,
        request: MetricEvaluationRequest
    ) -> float:
        """Evaluate custom formula metric."""
        definition = CustomFormulaDefinition(**metric.definition)

        # Build CTE for variables
        cte_parts = []
        for var_name, var_expr in definition.variables.items():
            cte_parts.append(f"({var_expr}) AS {var_name}")

        query = f"""
        SELECT {definition.formula} AS result
        FROM (
            SELECT {', '.join(cte_parts)}
            FROM qagentic.test_runs
            WHERE 1=1
        """

        if request.start_time:
            query += f" AND created_at >= '{request.start_time}'"
        if request.end_time:
            query += f" AND created_at <= '{request.end_time}'"

        dimensions = {**definition.filters, **request.dimensions}
        for key, value in dimensions.items():
            query += f" AND {key} = '{value}'"

        query += ") AS vars"

        async with get_db_session() as session:
            result = await session.execute(text(query))
            row = result.first()

            return float(row.result) if row and row.result else 0.0

    async def _store_metric_value(
        self,
        metric_id: UUID,
        value: float,
        dimensions: Dict[str, str]
    ):
        """Store metric value in database."""
        async with get_db_session() as session:
            metric_value = CustomMetricValue(
                metric_definition_id=metric_id,
                value=value,
                dimensions=dimensions,
                timestamp=datetime.utcnow()
            )

            session.add(metric_value)
            await session.commit()

    async def get_metric_timeseries(
        self,
        request: MetricTimeSeriesRequest
    ) -> MetricTimeSeriesResponse:
        """Get time-series data for a metric."""
        metric = await self.get_metric(request.metric_id)
        if not metric:
            raise ValueError(f"Metric not found: {request.metric_id}")

        async with get_db_session() as session:
            query = f"""
            SELECT
                time_bucket('{request.interval}', timestamp) AS bucket,
                AVG(value) AS avg_value,
                dimensions
            FROM qagentic.custom_metric_values
            WHERE metric_definition_id = :metric_id
              AND timestamp >= :start_time
              AND timestamp <= :end_time
            """

            # Add dimension filters
            if request.dimensions:
                query += " AND dimensions @> :dimensions::jsonb"

            query += " GROUP BY bucket, dimensions ORDER BY bucket"

            params = {
                "metric_id": request.metric_id,
                "start_time": request.start_time,
                "end_time": request.end_time
            }

            if request.dimensions:
                import json
                params["dimensions"] = json.dumps(request.dimensions)

            result = await session.execute(text(query), params)
            rows = result.all()

            data_points = [
                MetricTimeSeriesPoint(
                    timestamp=row.bucket,
                    value=float(row.avg_value),
                    dimensions=row.dimensions or {}
                )
                for row in rows
            ]

            return MetricTimeSeriesResponse(
                metric_id=request.metric_id,
                metric_name=metric.name,
                interval=request.interval,
                data_points=data_points,
                start_time=request.start_time,
                end_time=request.end_time
            )


# Singleton instance
_service = None


def get_custom_metrics_service() -> CustomMetricsService:
    """Get singleton custom metrics service instance."""
    global _service
    if _service is None:
        _service = CustomMetricsService()
    return _service
