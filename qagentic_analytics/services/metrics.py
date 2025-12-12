"""Service for computing and providing metrics."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
from pydantic import UUID4

from qagentic_analytics.schemas.metrics import (
    MetricSummary,
    MetricTrend,
    FlakyTestsReport,
    MetricTimeRange,
    TestMetrics,
    ServiceMetrics,
    BranchMetrics,
    FlakyTest,
    MetricDataPoint,
)

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for computing and providing metrics."""

    def __init__(
        self,
        enable_flake_detection: bool = True,
        enable_trend_analysis: bool = True,
    ):
        """Initialize the metrics service."""
        self.enable_flake_detection = enable_flake_detection
        self.enable_trend_analysis = enable_trend_analysis
        logger.info(f"Initializing metrics service")
        
    async def get_summary(
        self,
        service: Optional[str] = None,
        branch: Optional[str] = None,
        time_range: Optional[MetricTimeRange] = None,
    ) -> MetricSummary:
        """
        Get overall quality metrics summary.
        
        Args:
            service: Optional service name to filter by
            branch: Optional branch name to filter by
            time_range: Optional time range to filter by
            
        Returns:
            A summary of quality metrics
        """
        # This is a placeholder implementation
        # In a real implementation, we would query the database
        
        # Generate overall metrics
        overall = TestMetrics(
            total_runs=120,
            total_tests=1500,
            total_failures=95,
            passed_tests=1380,
            failed_tests=85,
            skipped_tests=35,
            flaky_tests=24,
            pass_rate=0.92,
            failure_rate=0.057,
            flaky_rate=0.016,
            mttr_minutes=45.5,
        )
        
        # Generate service-specific metrics
        services = ["checkout", "payment", "user", "product"]
        by_service = []
        
        for svc in services:
            if service and svc != service:
                continue
                
            # Generate slightly different metrics for each service
            modifier = hash(svc) % 20 / 100.0  # Deterministic variation
            by_service.append(
                ServiceMetrics(
                    service_name=svc,
                    total_runs=30,
                    total_tests=overall.total_tests // len(services),
                    total_failures=int(overall.total_failures * (0.25 + modifier)),
                    passed_tests=int(overall.passed_tests * (0.25 - modifier * 0.5)),
                    failed_tests=int(overall.failed_tests * (0.25 + modifier)),
                    skipped_tests=int(overall.skipped_tests * 0.25),
                    flaky_tests=int(overall.flaky_tests * (0.25 + modifier * 0.5)),
                    pass_rate=max(0, min(1, overall.pass_rate + modifier * 0.1 - 0.05)),
                    failure_rate=max(0, min(1, overall.failure_rate - modifier * 0.01 + 0.005)),
                    flaky_rate=max(0, min(1, overall.flaky_rate + modifier * 0.005)),
                    mttr_minutes=overall.mttr_minutes * (0.8 + modifier * 0.4),
                )
            )
            
        # Generate branch-specific metrics
        branches = ["main", "develop", "feature/new-ui"]
        by_branch = []
        
        for br in branches:
            if branch and br != branch:
                continue
                
            # Generate slightly different metrics for each branch
            modifier = hash(br) % 20 / 100.0  # Deterministic variation
            by_branch.append(
                BranchMetrics(
                    branch_name=br,
                    total_runs=40,
                    total_tests=overall.total_tests // len(branches),
                    total_failures=int(overall.total_failures * (0.33 + modifier)),
                    passed_tests=int(overall.passed_tests * (0.33 - modifier * 0.5)),
                    failed_tests=int(overall.failed_tests * (0.33 + modifier)),
                    skipped_tests=int(overall.skipped_tests * 0.33),
                    flaky_tests=int(overall.flaky_tests * (0.33 + modifier * 0.5)),
                    pass_rate=max(0, min(1, overall.pass_rate + modifier * 0.1 - 0.05)),
                    failure_rate=max(0, min(1, overall.failure_rate - modifier * 0.01 + 0.005)),
                    flaky_rate=max(0, min(1, overall.flaky_rate + modifier * 0.005)),
                    mttr_minutes=overall.mttr_minutes * (0.8 + modifier * 0.4),
                )
            )
            
        # Generate top failing and flaky tests
        top_failing_tests = [
            {
                "id": str(uuid.uuid4()),
                "name": "test_checkout_with_promo_code",
                "failure_count": 12,
                "failure_rate": 0.4,
                "service": "checkout",
                "last_failure": str(datetime.now() - timedelta(hours=3)),
            },
            {
                "id": str(uuid.uuid4()),
                "name": "test_payment_with_expired_card",
                "failure_count": 10,
                "failure_rate": 0.33,
                "service": "payment",
                "last_failure": str(datetime.now() - timedelta(hours=5)),
            },
            {
                "id": str(uuid.uuid4()),
                "name": "test_user_registration_with_existing_email",
                "failure_count": 8,
                "failure_rate": 0.27,
                "service": "user",
                "last_failure": str(datetime.now() - timedelta(hours=12)),
            },
        ]
        
        top_flaky_tests = [
            {
                "id": str(uuid.uuid4()),
                "name": "test_product_image_upload",
                "flake_rate": 0.3,
                "service": "product",
                "last_flaky_run": str(datetime.now() - timedelta(hours=6)),
            },
            {
                "id": str(uuid.uuid4()),
                "name": "test_checkout_with_multiple_items",
                "flake_rate": 0.25,
                "service": "checkout",
                "last_flaky_run": str(datetime.now() - timedelta(hours=8)),
            },
            {
                "id": str(uuid.uuid4()),
                "name": "test_user_login_with_remember_me",
                "flake_rate": 0.2,
                "service": "user",
                "last_flaky_run": str(datetime.now() - timedelta(hours=24)),
            },
        ]
        
        return MetricSummary(
            overall=overall,
            by_service=by_service,
            by_branch=by_branch,
            time_range=time_range,
            top_failing_tests=top_failing_tests,
            top_flaky_tests=top_flaky_tests,
        )
        
    async def get_trends(
        self,
        metric: str,
        service: Optional[str] = None,
        branch: Optional[str] = None,
        time_range: Optional[MetricTimeRange] = None,
        interval: str = "day",
    ) -> List[MetricTrend]:
        """
        Get trending metrics over time.
        
        Args:
            metric: The metric to retrieve trends for
            service: Optional service name to filter by
            branch: Optional branch name to filter by
            time_range: Optional time range to filter by
            interval: Interval for trend data points (hour, day, week, month)
            
        Returns:
            List of metric trends
        """
        if not self.enable_trend_analysis:
            logger.warning("Trend analysis is disabled")
            return []
            
        # This is a placeholder implementation
        # In a real implementation, we would query the database for historical metrics
        
        # Determine the time range for the trend data
        now = datetime.now()
        if time_range:
            start_time = time_range.from_date or now - timedelta(days=30)
            end_time = time_range.to_date or now
        else:
            # Default to last 30 days
            start_time = now - timedelta(days=30)
            end_time = now
            
        # Determine the interval between data points
        if interval == "hour":
            delta = timedelta(hours=1)
            num_points = int((end_time - start_time).total_seconds() / 3600)
        elif interval == "week":
            delta = timedelta(weeks=1)
            num_points = int((end_time - start_time).days / 7)
        elif interval == "month":
            delta = timedelta(days=30)
            num_points = int((end_time - start_time).days / 30)
        else:  # Default to day
            delta = timedelta(days=1)
            num_points = (end_time - start_time).days
            
        # Cap the number of points to avoid excessive data
        num_points = min(num_points, 100)
        
        # Generate the trend data
        trends = []
        
        # If service is specified, generate trend for that service
        if service:
            data_points = []
            current_time = start_time
            
            # Choose base value and variation based on the metric
            if metric == "pass_rate":
                base_value = 0.92
                variation = 0.03
            elif metric == "failure_rate":
                base_value = 0.06
                variation = 0.02
            elif metric == "flaky_rate":
                base_value = 0.02
                variation = 0.01
            elif metric == "mttr_minutes":
                base_value = 45.0
                variation = 15.0
            else:
                base_value = 0.5
                variation = 0.1
            
            for i in range(num_points):
                # Generate a deterministic but varying data point value
                seed = hash(f"{service}:{metric}:{current_time.isoformat()}")
                np.random.seed(seed)
                
                # Create a slightly random walk
                if i > 0:
                    prev_value = data_points[-1].value
                    value = max(0, prev_value + np.random.normal(0, variation * 0.3))
                    # For rates, cap at 1.0
                    if metric.endswith("_rate"):
                        value = min(1.0, value)
                else:
                    value = base_value + np.random.normal(0, variation)
                    if metric.endswith("_rate"):
                        value = max(0, min(1.0, value))
                    else:
                        value = max(0, value)
                        
                data_points.append(
                    MetricDataPoint(
                        timestamp=current_time,
                        value=value,
                    )
                )
                current_time += delta
                
            trends.append(
                MetricTrend(
                    metric_name=metric,
                    service=service,
                    branch=branch,
                    data_points=data_points,
                    time_range=time_range,
                    interval=interval,
                )
            )
            
        # If branch is specified (and service isn't), generate trend for that branch
        elif branch:
            data_points = []
            current_time = start_time
            
            # Similar to above, but use branch-specific seed
            if metric == "pass_rate":
                base_value = 0.90
                variation = 0.04
            elif metric == "failure_rate":
                base_value = 0.08
                variation = 0.03
            elif metric == "flaky_rate":
                base_value = 0.03
                variation = 0.015
            elif metric == "mttr_minutes":
                base_value = 50.0
                variation = 20.0
            else:
                base_value = 0.5
                variation = 0.1
            
            for i in range(num_points):
                seed = hash(f"{branch}:{metric}:{current_time.isoformat()}")
                np.random.seed(seed)
                
                if i > 0:
                    prev_value = data_points[-1].value
                    value = max(0, prev_value + np.random.normal(0, variation * 0.3))
                    if metric.endswith("_rate"):
                        value = min(1.0, value)
                else:
                    value = base_value + np.random.normal(0, variation)
                    if metric.endswith("_rate"):
                        value = max(0, min(1.0, value))
                    else:
                        value = max(0, value)
                        
                data_points.append(
                    MetricDataPoint(
                        timestamp=current_time,
                        value=value,
                    )
                )
                current_time += delta
                
            trends.append(
                MetricTrend(
                    metric_name=metric,
                    service=service,
                    branch=branch,
                    data_points=data_points,
                    time_range=time_range,
                    interval=interval,
                )
            )
            
        # If neither service nor branch specified, generate overall trend
        else:
            data_points = []
            current_time = start_time
            
            if metric == "pass_rate":
                base_value = 0.94
                variation = 0.02
            elif metric == "failure_rate":
                base_value = 0.05
                variation = 0.015
            elif metric == "flaky_rate":
                base_value = 0.015
                variation = 0.005
            elif metric == "mttr_minutes":
                base_value = 40.0
                variation = 10.0
            else:
                base_value = 0.5
                variation = 0.1
            
            for i in range(num_points):
                seed = hash(f"overall:{metric}:{current_time.isoformat()}")
                np.random.seed(seed)
                
                if i > 0:
                    prev_value = data_points[-1].value
                    value = max(0, prev_value + np.random.normal(0, variation * 0.3))
                    if metric.endswith("_rate"):
                        value = min(1.0, value)
                else:
                    value = base_value + np.random.normal(0, variation)
                    if metric.endswith("_rate"):
                        value = max(0, min(1.0, value))
                    else:
                        value = max(0, value)
                        
                data_points.append(
                    MetricDataPoint(
                        timestamp=current_time,
                        value=value,
                    )
                )
                current_time += delta
                
            trends.append(
                MetricTrend(
                    metric_name=metric,
                    service=None,
                    branch=None,
                    data_points=data_points,
                    time_range=time_range,
                    interval=interval,
                )
            )
            
        return trends
    
    async def get_flaky_tests(
        self,
        service: Optional[str] = None,
        branch: Optional[str] = None,
        min_flake_rate: float = 0.05,
        min_executions: int = 5,
        time_range: Optional[MetricTimeRange] = None,
    ) -> FlakyTestsReport:
        """
        Get flaky tests analysis.
        
        Args:
            service: Optional service name to filter by
            branch: Optional branch name to filter by
            min_flake_rate: Minimum flake rate to consider (0-1)
            min_executions: Minimum number of executions to analyze
            time_range: Optional time range to filter by
            
        Returns:
            A report on flaky tests
        """
        if not self.enable_flake_detection:
            logger.warning("Flake detection is disabled")
            return FlakyTestsReport(
                flaky_tests=[],
                total_flaky_tests=0,
                overall_flaky_rate=0.0,
            )
            
        # This is a placeholder implementation
        # In a real implementation, we would query the database
        
        # Generate sample flaky tests
        flaky_tests = []
        services_flaky = {"checkout": 0, "payment": 0, "user": 0, "product": 0}
        
        # Add some checkout service tests
        if not service or service == "checkout":
            tests = [
                ("test_checkout_with_multiple_items", 0.25, 20, 15, 5),
                ("test_checkout_with_promo_code", 0.15, 15, 12, 3),
            ]
            for name, rate, execs, passes, fails in tests:
                if rate >= min_flake_rate and execs >= min_executions:
                    flaky_tests.append(
                        FlakyTest(
                            id=uuid.uuid4(),
                            name=name,
                            flake_rate=rate,
                            total_executions=execs,
                            pass_count=passes,
                            fail_count=fails,
                            alternating_patterns=["PFPFP", "PPFPP"],
                            last_flaky_run_id=uuid.uuid4(),
                            first_seen_flaky=datetime.now() - timedelta(days=7),
                            metadata={"service": "checkout"},
                        )
                    )
                    services_flaky["checkout"] += 1
        
        # Add some payment service tests
        if not service or service == "payment":
            tests = [
                ("test_payment_with_expired_card", 0.1, 10, 9, 1),
                ("test_payment_processing", 0.2, 30, 24, 6),
                ("test_payment_refund", 0.33, 12, 8, 4),
            ]
            for name, rate, execs, passes, fails in tests:
                if rate >= min_flake_rate and execs >= min_executions:
                    flaky_tests.append(
                        FlakyTest(
                            id=uuid.uuid4(),
                            name=name,
                            flake_rate=rate,
                            total_executions=execs,
                            pass_count=passes,
                            fail_count=fails,
                            alternating_patterns=["PFPFP", "PPFPP"],
                            last_flaky_run_id=uuid.uuid4(),
                            first_seen_flaky=datetime.now() - timedelta(days=5),
                            metadata={"service": "payment"},
                        )
                    )
                    services_flaky["payment"] += 1
                    
        # Add some user service tests
        if not service or service == "user":
            tests = [
                ("test_user_login_with_remember_me", 0.2, 25, 20, 5),
                ("test_user_password_reset", 0.08, 12, 11, 1),
            ]
            for name, rate, execs, passes, fails in tests:
                if rate >= min_flake_rate and execs >= min_executions:
                    flaky_tests.append(
                        FlakyTest(
                            id=uuid.uuid4(),
                            name=name,
                            flake_rate=rate,
                            total_executions=execs,
                            pass_count=passes,
                            fail_count=fails,
                            alternating_patterns=["PFPFP", "PPFPP"],
                            last_flaky_run_id=uuid.uuid4(),
                            first_seen_flaky=datetime.now() - timedelta(days=10),
                            metadata={"service": "user"},
                        )
                    )
                    services_flaky["user"] += 1
                    
        # Add some product service tests
        if not service or service == "product":
            tests = [
                ("test_product_image_upload", 0.3, 20, 14, 6),
                ("test_product_search", 0.15, 25, 21, 4),
            ]
            for name, rate, execs, passes, fails in tests:
                if rate >= min_flake_rate and execs >= min_executions:
                    flaky_tests.append(
                        FlakyTest(
                            id=uuid.uuid4(),
                            name=name,
                            flake_rate=rate,
                            total_executions=execs,
                            pass_count=passes,
                            fail_count=fails,
                            alternating_patterns=["PFPFP", "PPFPP"],
                            last_flaky_run_id=uuid.uuid4(),
                            first_seen_flaky=datetime.now() - timedelta(days=3),
                            metadata={"service": "product"},
                        )
                    )
                    services_flaky["product"] += 1
        
        # Calculate overall stats
        total_flaky = len(flaky_tests)
        overall_rate = sum(t.flake_rate for t in flaky_tests) / len(flaky_tests) if flaky_tests else 0
        
        # Filter by service if needed
        services_count = {k: v for k, v in services_flaky.items() if not service or k == service}
        
        # Generate some recommendations based on the flaky tests
        recommendations = [
            "Consider adding retries to tests with high flake rates",
            "Review the timing and synchronization in UI tests",
            "Check for race conditions in tests that access shared resources",
        ]
        
        return FlakyTestsReport(
            flaky_tests=flaky_tests,
            total_flaky_tests=total_flaky,
            overall_flaky_rate=overall_rate,
            time_range=time_range,
            by_service=services_count,
            recommendations=recommendations,
        )
