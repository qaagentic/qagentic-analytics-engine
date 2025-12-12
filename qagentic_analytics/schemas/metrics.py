"""Data models for metrics and analytics."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field, UUID4


class MetricTimeRange(BaseModel):
    """Time range for metric queries."""

    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class TestMetrics(BaseModel):
    """Basic test metrics for a service or branch."""

    total_runs: int
    total_tests: int
    total_failures: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    flaky_tests: int
    pass_rate: float
    failure_rate: float
    flaky_rate: float
    mttr_minutes: Optional[float] = None  # Mean time to recovery


class ServiceMetrics(TestMetrics):
    """Test metrics for a specific service."""

    service_name: str


class BranchMetrics(TestMetrics):
    """Test metrics for a specific branch."""

    branch_name: str


class MetricSummary(BaseModel):
    """Overall quality metrics summary."""

    overall: TestMetrics
    by_service: List[ServiceMetrics]
    by_branch: List[BranchMetrics]
    time_range: Optional[MetricTimeRange] = None
    top_failing_tests: List[Dict[str, Any]] = Field(default_factory=list)
    top_flaky_tests: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricDataPoint(BaseModel):
    """Single data point for a metric trend."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricTrend(BaseModel):
    """Trending data for a specific metric."""

    metric_name: str
    service: Optional[str] = None
    branch: Optional[str] = None
    data_points: List[MetricDataPoint]
    time_range: Optional[MetricTimeRange] = None
    interval: str  # hour, day, week, month


class FlakyTest(BaseModel):
    """Information about a flaky test."""

    id: UUID4
    name: str
    flake_rate: float  # 0.0-1.0
    total_executions: int
    pass_count: int
    fail_count: int
    alternating_patterns: Optional[List[str]] = None  # e.g., ["PFPFPF", "PPFPPF"]
    last_flaky_run_id: Optional[UUID4] = None
    first_seen_flaky: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlakyTestsReport(BaseModel):
    """Report on flaky tests in the system."""

    flaky_tests: List[FlakyTest]
    total_flaky_tests: int
    overall_flaky_rate: float
    time_range: Optional[MetricTimeRange] = None
    by_service: Dict[str, int] = Field(default_factory=dict)  # Service -> flaky test count
    recommendations: List[str] = Field(default_factory=list)
