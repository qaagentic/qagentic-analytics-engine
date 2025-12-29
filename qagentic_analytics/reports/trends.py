"""Trend analysis for reports."""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class TrendDirection(str, Enum):
    """Direction of a trend."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"


class TrendMetric(str, Enum):
    """Metrics for trend analysis."""
    PASS_RATE = "pass_rate"
    FAILURE_RATE = "failure_rate"
    FLAKY_RATE = "flaky_rate"
    MTTR = "mttr"
    TEST_COUNT = "test_count"
    FAILURE_COUNT = "failure_count"


@dataclass
class DataPoint:
    """A single data point in a trend."""
    timestamp: datetime
    value: float
    label: str


@dataclass
class TrendAnalysis:
    """Analysis of a metric trend."""
    metric: TrendMetric
    data_points: List[DataPoint]
    direction: TrendDirection
    slope: float
    average: float
    min_value: float
    max_value: float
    std_dev: float
    forecast: Optional[List[DataPoint]] = None
    insights: List[str] = None

    def __post_init__(self):
        if self.insights is None:
            self.insights = []


@dataclass
class TrendReport:
    """A trend analysis report."""
    report_id: str
    title: str
    time_range_start: datetime
    time_range_end: datetime
    metrics_analyzed: List[TrendAnalysis]
    created_at: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TrendAnalysisService:
    """Service for trend analysis."""

    def __init__(self):
        """Initialize trend analysis service."""
        self.trend_reports: Dict[str, TrendReport] = {}

    def analyze_metric_trend(
        self,
        metric: TrendMetric,
        data_points: List[Tuple[datetime, float]],
        forecast_periods: int = 7,
    ) -> TrendAnalysis:
        """
        Analyze trend for a metric.

        Args:
            metric: Metric to analyze
            data_points: List of (timestamp, value) tuples
            forecast_periods: Number of periods to forecast

        Returns:
            TrendAnalysis object
        """
        if not data_points:
            return TrendAnalysis(
                metric=metric,
                data_points=[],
                direction=TrendDirection.STABLE,
                slope=0.0,
                average=0.0,
                min_value=0.0,
                max_value=0.0,
                std_dev=0.0,
            )

        # Convert to DataPoint objects
        points = [
            DataPoint(timestamp=ts, value=val, label=ts.strftime("%Y-%m-%d"))
            for ts, val in data_points
        ]

        # Calculate statistics
        values = [p.value for p in points]
        average = statistics.mean(values)
        min_value = min(values)
        max_value = max(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

        # Calculate slope (linear regression)
        slope = self._calculate_slope(points)

        # Determine direction
        direction = self._determine_direction(metric, slope, std_dev, values)

        # Generate forecast
        forecast = self._generate_forecast(points, slope, forecast_periods)

        # Generate insights
        insights = self._generate_insights(metric, direction, slope, values, average)

        analysis = TrendAnalysis(
            metric=metric,
            data_points=points,
            direction=direction,
            slope=slope,
            average=average,
            min_value=min_value,
            max_value=max_value,
            std_dev=std_dev,
            forecast=forecast,
            insights=insights,
        )

        logger.info(
            f"Analyzed trend for {metric}: direction={direction}, "
            f"slope={slope:.4f}, avg={average:.2f}"
        )

        return analysis

    def generate_trend_report(
        self,
        title: str,
        time_range_start: datetime,
        time_range_end: datetime,
        metrics_data: Dict[TrendMetric, List[Tuple[datetime, float]]],
    ) -> TrendReport:
        """
        Generate a trend analysis report.

        Args:
            title: Report title
            time_range_start: Start of time range
            time_range_end: End of time range
            metrics_data: Dictionary of metric data

        Returns:
            TrendReport object
        """
        report_id = f"trend-report-{datetime.utcnow().timestamp()}"

        metrics_analyzed = []
        for metric, data_points in metrics_data.items():
            analysis = self.analyze_metric_trend(metric, data_points)
            metrics_analyzed.append(analysis)

        report = TrendReport(
            report_id=report_id,
            title=title,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            metrics_analyzed=metrics_analyzed,
            created_at=datetime.utcnow(),
        )

        self.trend_reports[report_id] = report
        logger.info(f"Generated trend report: {report_id}")

        return report

    def get_trend_report(self, report_id: str) -> Optional[TrendReport]:
        """
        Get a trend report.

        Args:
            report_id: ID of the report

        Returns:
            TrendReport or None
        """
        return self.trend_reports.get(report_id)

    def detect_anomalies(
        self,
        data_points: List[Tuple[datetime, float]],
        sensitivity: float = 2.0,
    ) -> List[Tuple[datetime, float, str]]:
        """
        Detect anomalies in data.

        Args:
            data_points: List of (timestamp, value) tuples
            sensitivity: Sensitivity factor (standard deviations)

        Returns:
            List of (timestamp, value, reason) tuples
        """
        if len(data_points) < 3:
            return []

        values = [v for _, v in data_points]
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)

        anomalies = []
        for ts, val in data_points:
            z_score = abs((val - mean) / std_dev) if std_dev > 0 else 0
            if z_score > sensitivity:
                reason = "High" if val > mean else "Low"
                anomalies.append((ts, val, f"{reason} anomaly (z-score: {z_score:.2f})"))

        return anomalies

    def get_trend_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trend reports.

        Returns:
            Dictionary with trend statistics
        """
        if not self.trend_reports:
            return {
                "total_reports": 0,
                "metrics_analyzed": 0,
            }

        reports = list(self.trend_reports.values())
        total_metrics = sum(len(r.metrics_analyzed) for r in reports)

        improving_count = sum(
            1 for r in reports
            for m in r.metrics_analyzed
            if m.direction == TrendDirection.IMPROVING
        )

        degrading_count = sum(
            1 for r in reports
            for m in r.metrics_analyzed
            if m.direction == TrendDirection.DEGRADING
        )

        return {
            "total_reports": len(reports),
            "metrics_analyzed": total_metrics,
            "improving_metrics": improving_count,
            "degrading_metrics": degrading_count,
            "most_recent_report": reports[-1].created_at if reports else None,
        }

    def _calculate_slope(self, points: List[DataPoint]) -> float:
        """Calculate slope using linear regression."""
        if len(points) < 2:
            return 0.0

        n = len(points)
        x_values = list(range(n))
        y_values = [p.value for p in points]

        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)

        numerator = sum(
            (x_values[i] - x_mean) * (y_values[i] - y_mean)
            for i in range(n)
        )
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _determine_direction(
        self,
        metric: TrendMetric,
        slope: float,
        std_dev: float,
        values: List[float],
    ) -> TrendDirection:
        """Determine trend direction."""
        # If standard deviation is high relative to mean, it's volatile
        if len(values) > 0 and values[0] != 0:
            cv = std_dev / statistics.mean(values)
            if cv > 0.3:
                return TrendDirection.VOLATILE

        # Determine if improving or degrading based on metric type
        if metric in [TrendMetric.PASS_RATE]:
            if slope > 0.01:
                return TrendDirection.IMPROVING
            elif slope < -0.01:
                return TrendDirection.DEGRADING
        else:
            if slope < -0.01:
                return TrendDirection.IMPROVING
            elif slope > 0.01:
                return TrendDirection.DEGRADING

        return TrendDirection.STABLE

    def _generate_forecast(
        self,
        points: List[DataPoint],
        slope: float,
        periods: int,
    ) -> List[DataPoint]:
        """Generate forecast for future periods."""
        if not points or periods <= 0:
            return []

        forecast = []
        last_point = points[-1]
        last_value = last_point.value

        # Simple linear forecast
        for i in range(1, periods + 1):
            future_value = last_value + (slope * i)
            future_time = last_point.timestamp + timedelta(days=i)
            forecast.append(
                DataPoint(
                    timestamp=future_time,
                    value=future_value,
                    label=future_time.strftime("%Y-%m-%d"),
                )
            )

        return forecast

    def _generate_insights(
        self,
        metric: TrendMetric,
        direction: TrendDirection,
        slope: float,
        values: List[float],
        average: float,
    ) -> List[str]:
        """Generate insights from trend analysis."""
        insights = []

        # Direction insight
        if direction == TrendDirection.IMPROVING:
            insights.append(f"{metric.value} is improving over time")
        elif direction == TrendDirection.DEGRADING:
            insights.append(f"{metric.value} is degrading over time")
        elif direction == TrendDirection.VOLATILE:
            insights.append(f"{metric.value} shows high volatility")
        else:
            insights.append(f"{metric.value} is stable")

        # Magnitude insight
        if abs(slope) > 0.05:
            insights.append(f"Significant trend detected (slope: {slope:.4f})")

        # Extreme values insight
        if values:
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val

            if range_val > average * 0.5:
                insights.append(f"Large variation observed (range: {range_val:.2f})")

            if min_val < average * 0.8:
                insights.append("Recent low values detected")

            if max_val > average * 1.2:
                insights.append("Recent high values detected")

        return insights
