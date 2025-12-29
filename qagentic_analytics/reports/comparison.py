"""Comparison logic for generating comparative reports."""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ComparisonType(str, Enum):
    """Types of comparisons."""
    TIME_PERIOD = "time_period"
    BRANCH = "branch"
    SERVICE = "service"
    BUILD = "build"
    CUSTOM = "custom"


class ComparisonMetric(str, Enum):
    """Metrics that can be compared."""
    PASS_RATE = "pass_rate"
    FAILURE_RATE = "failure_rate"
    FLAKY_RATE = "flaky_rate"
    MTTR = "mttr"
    TEST_COUNT = "test_count"
    FAILURE_COUNT = "failure_count"
    DURATION = "duration"


@dataclass
class MetricValue:
    """A metric value with metadata."""
    metric: ComparisonMetric
    value: float
    timestamp: datetime
    label: str


@dataclass
class ComparisonItem:
    """An item being compared."""
    item_id: str
    item_name: str
    item_type: str
    metrics: Dict[str, MetricValue]
    metadata: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Result of a comparison."""
    comparison_id: str
    comparison_type: ComparisonType
    items: List[ComparisonItem]
    metrics_compared: List[ComparisonMetric]
    created_at: datetime
    analysis: Dict[str, Any]


class ComparisonService:
    """Service for comparing metrics and generating comparative reports."""

    def __init__(self):
        """Initialize comparison service."""
        self.comparisons: Dict[str, ComparisonResult] = {}

    def compare_time_periods(
        self,
        metrics_data: Dict[str, Any],
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime,
        metrics: List[ComparisonMetric],
    ) -> ComparisonResult:
        """
        Compare metrics between two time periods.

        Args:
            metrics_data: Dictionary of metrics data
            period1_start: Start of first period
            period1_end: End of first period
            period2_start: Start of second period
            period2_end: End of second period
            metrics: List of metrics to compare

        Returns:
            ComparisonResult
        """
        comparison_id = f"comp-{datetime.utcnow().timestamp()}"

        # Create comparison items for each period
        period1_item = ComparisonItem(
            item_id="period1",
            item_name=f"{period1_start.date()} to {period1_end.date()}",
            item_type="time_period",
            metrics={},
            metadata={"start": period1_start, "end": period1_end},
        )

        period2_item = ComparisonItem(
            item_id="period2",
            item_name=f"{period2_start.date()} to {period2_end.date()}",
            item_type="time_period",
            metrics={},
            metadata={"start": period2_start, "end": period2_end},
        )

        # Populate metrics (in real implementation, would query database)
        for metric in metrics:
            period1_item.metrics[metric.value] = MetricValue(
                metric=metric,
                value=self._get_metric_value(metrics_data, metric, "period1"),
                timestamp=period1_end,
                label=f"Period 1: {metric.value}",
            )
            period2_item.metrics[metric.value] = MetricValue(
                metric=metric,
                value=self._get_metric_value(metrics_data, metric, "period2"),
                timestamp=period2_end,
                label=f"Period 2: {metric.value}",
            )

        # Perform analysis
        analysis = self._analyze_comparison([period1_item, period2_item], metrics)

        result = ComparisonResult(
            comparison_id=comparison_id,
            comparison_type=ComparisonType.TIME_PERIOD,
            items=[period1_item, period2_item],
            metrics_compared=metrics,
            created_at=datetime.utcnow(),
            analysis=analysis,
        )

        self.comparisons[comparison_id] = result
        logger.info(f"Created time period comparison: {comparison_id}")

        return result

    def compare_branches(
        self,
        metrics_data: Dict[str, Any],
        branches: List[str],
        metrics: List[ComparisonMetric],
    ) -> ComparisonResult:
        """
        Compare metrics across branches.

        Args:
            metrics_data: Dictionary of metrics data
            branches: List of branch names
            metrics: List of metrics to compare

        Returns:
            ComparisonResult
        """
        comparison_id = f"comp-{datetime.utcnow().timestamp()}"

        items = []
        for branch in branches:
            item = ComparisonItem(
                item_id=f"branch-{branch}",
                item_name=branch,
                item_type="branch",
                metrics={},
                metadata={"branch": branch},
            )

            for metric in metrics:
                item.metrics[metric.value] = MetricValue(
                    metric=metric,
                    value=self._get_metric_value(metrics_data, metric, branch),
                    timestamp=datetime.utcnow(),
                    label=f"{branch}: {metric.value}",
                )

            items.append(item)

        # Perform analysis
        analysis = self._analyze_comparison(items, metrics)

        result = ComparisonResult(
            comparison_id=comparison_id,
            comparison_type=ComparisonType.BRANCH,
            items=items,
            metrics_compared=metrics,
            created_at=datetime.utcnow(),
            analysis=analysis,
        )

        self.comparisons[comparison_id] = result
        logger.info(f"Created branch comparison: {comparison_id}")

        return result

    def compare_services(
        self,
        metrics_data: Dict[str, Any],
        services: List[str],
        metrics: List[ComparisonMetric],
    ) -> ComparisonResult:
        """
        Compare metrics across services.

        Args:
            metrics_data: Dictionary of metrics data
            services: List of service names
            metrics: List of metrics to compare

        Returns:
            ComparisonResult
        """
        comparison_id = f"comp-{datetime.utcnow().timestamp()}"

        items = []
        for service in services:
            item = ComparisonItem(
                item_id=f"service-{service}",
                item_name=service,
                item_type="service",
                metrics={},
                metadata={"service": service},
            )

            for metric in metrics:
                item.metrics[metric.value] = MetricValue(
                    metric=metric,
                    value=self._get_metric_value(metrics_data, metric, service),
                    timestamp=datetime.utcnow(),
                    label=f"{service}: {metric.value}",
                )

            items.append(item)

        # Perform analysis
        analysis = self._analyze_comparison(items, metrics)

        result = ComparisonResult(
            comparison_id=comparison_id,
            comparison_type=ComparisonType.SERVICE,
            items=items,
            metrics_compared=metrics,
            created_at=datetime.utcnow(),
            analysis=analysis,
        )

        self.comparisons[comparison_id] = result
        logger.info(f"Created service comparison: {comparison_id}")

        return result

    def get_comparison(self, comparison_id: str) -> Optional[ComparisonResult]:
        """
        Get a comparison result.

        Args:
            comparison_id: ID of the comparison

        Returns:
            ComparisonResult or None
        """
        return self.comparisons.get(comparison_id)

    def _analyze_comparison(
        self,
        items: List[ComparisonItem],
        metrics: List[ComparisonMetric],
    ) -> Dict[str, Any]:
        """
        Analyze comparison results.

        Args:
            items: Items being compared
            metrics: Metrics being compared

        Returns:
            Analysis dictionary
        """
        analysis = {
            "best_performers": {},
            "worst_performers": {},
            "improvements": {},
            "regressions": {},
            "summary": {},
        }

        if len(items) < 2:
            return analysis

        # Find best and worst performers for each metric
        for metric in metrics:
            metric_values = []
            for item in items:
                if metric.value in item.metrics:
                    metric_values.append(
                        (item.item_name, item.metrics[metric.value].value)
                    )

            if metric_values:
                # Sort by value (higher is better for pass_rate, lower is better for failure_rate)
                if metric in [ComparisonMetric.PASS_RATE]:
                    metric_values.sort(key=lambda x: x[1], reverse=True)
                else:
                    metric_values.sort(key=lambda x: x[1])

                if metric_values:
                    analysis["best_performers"][metric.value] = metric_values[0]
                    analysis["worst_performers"][metric.value] = metric_values[-1]

        # Calculate improvements and regressions (for time period comparisons)
        if len(items) == 2:
            item1, item2 = items[0], items[1]
            for metric in metrics:
                if metric.value in item1.metrics and metric.value in item2.metrics:
                    val1 = item1.metrics[metric.value].value
                    val2 = item2.metrics[metric.value].value
                    change = val2 - val1
                    change_percent = (change / val1 * 100) if val1 != 0 else 0

                    if change > 0:
                        if metric in [ComparisonMetric.PASS_RATE]:
                            analysis["improvements"][metric.value] = {
                                "change": change,
                                "change_percent": change_percent,
                            }
                        else:
                            analysis["regressions"][metric.value] = {
                                "change": change,
                                "change_percent": change_percent,
                            }
                    elif change < 0:
                        if metric in [ComparisonMetric.PASS_RATE]:
                            analysis["regressions"][metric.value] = {
                                "change": change,
                                "change_percent": change_percent,
                            }
                        else:
                            analysis["improvements"][metric.value] = {
                                "change": change,
                                "change_percent": change_percent,
                            }

        # Generate summary
        analysis["summary"] = {
            "items_compared": len(items),
            "metrics_compared": len(metrics),
            "total_improvements": len(analysis["improvements"]),
            "total_regressions": len(analysis["regressions"]),
        }

        return analysis

    def _get_metric_value(
        self,
        metrics_data: Dict[str, Any],
        metric: ComparisonMetric,
        context: str,
    ) -> float:
        """
        Get a metric value from metrics data.

        Args:
            metrics_data: Dictionary of metrics
            metric: Metric to retrieve
            context: Context (period, branch, service, etc.)

        Returns:
            Metric value
        """
        # In a real implementation, this would query the database
        # For now, return placeholder values
        placeholder_values = {
            ComparisonMetric.PASS_RATE: 0.92,
            ComparisonMetric.FAILURE_RATE: 0.08,
            ComparisonMetric.FLAKY_RATE: 0.016,
            ComparisonMetric.MTTR: 45.5,
            ComparisonMetric.TEST_COUNT: 1500,
            ComparisonMetric.FAILURE_COUNT: 120,
            ComparisonMetric.DURATION: 300,
        }

        return metrics_data.get(metric.value, placeholder_values.get(metric, 0.0))

    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Get summary of all comparisons.

        Returns:
            Dictionary with comparison statistics
        """
        if not self.comparisons:
            return {
                "total_comparisons": 0,
                "by_type": {},
            }

        comparisons = list(self.comparisons.values())

        by_type = {}
        for comp_type in ComparisonType:
            count = sum(1 for c in comparisons if c.comparison_type == comp_type)
            if count > 0:
                by_type[comp_type.value] = count

        return {
            "total_comparisons": len(comparisons),
            "by_type": by_type,
            "most_recent": comparisons[-1].created_at if comparisons else None,
        }
