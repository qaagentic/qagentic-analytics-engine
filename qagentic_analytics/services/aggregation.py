"""Service for optimized data aggregation."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from qagentic_analytics.db import get_db_session
from qagentic_analytics.models.metrics import TestMetrics
from qagentic_analytics.models.test_run import TestRun
from qagentic_analytics.cache import get_cache
from qagentic_analytics.utils.time_series import resample_time_series

logger = logging.getLogger(__name__)

class AggregationService:
    """Service for efficient data aggregation and analysis."""

    def __init__(self):
        self.cache = get_cache()
        self._aggregation_lock = asyncio.Lock()

    async def get_aggregated_metrics(
        self,
        time_window: timedelta = timedelta(days=30),
        interval: str = "1D",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics with caching.
        
        Args:
            time_window: Time window to analyze
            interval: Aggregation interval (e.g., "1H", "1D", "1W")
            use_cache: Whether to use cached results
            
        Returns:
            Aggregated metrics dictionary
        """
        cache_key = f"metrics:{interval}:{time_window.days}"
        
        # Try cache first
        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        async with self._aggregation_lock:
            # Check cache again in case another request computed it
            if use_cache:
                cached = await self.cache.get(cache_key)
                if cached:
                    return cached

            async with get_db_session() as session:
                # Get raw data
                metrics = await self._get_raw_metrics(session, time_window)
                
                # Compute aggregations
                aggregated = await self._compute_aggregations(
                    metrics,
                    interval
                )
                
                # Cache results
                if use_cache:
                    await self.cache.set(
                        cache_key,
                        aggregated,
                        expire=300  # 5 minutes
                    )
                
                return aggregated

    async def _get_raw_metrics(
        self,
        session: AsyncSession,
        time_window: timedelta
    ) -> List[TestMetrics]:
        """Get raw metrics data efficiently."""
        cutoff_time = datetime.utcnow() - time_window
        
        # Use optimized query with indexes
        query = text("""
            SELECT m.*
            FROM test_metrics m
            WHERE m.timestamp >= :cutoff
            ORDER BY m.timestamp DESC
        """)
        
        result = await session.execute(
            query,
            {"cutoff": cutoff_time}
        )
        
        return [TestMetrics(**row) for row in result]

    async def _compute_aggregations(
        self,
        metrics: List[TestMetrics],
        interval: str
    ) -> Dict[str, Any]:
        """Compute efficient aggregations."""
        # Convert to time series
        ts_data = self._prepare_time_series(metrics)
        
        # Resample to requested interval
        resampled = resample_time_series(ts_data, interval)
        
        # Compute aggregations
        return {
            "summary": await self._compute_summary(resampled),
            "trends": await self._compute_trends(resampled),
            "distributions": await self._compute_distributions(resampled),
            "correlations": await self._compute_correlations(resampled)
        }

    def _prepare_time_series(
        self,
        metrics: List[TestMetrics]
    ) -> Dict[str, List[float]]:
        """Prepare time series data."""
        series = defaultdict(list)
        
        for metric in sorted(metrics, key=lambda m: m.timestamp):
            series["timestamps"].append(metric.timestamp)
            series["pass_rate"].append(
                metric.passed_tests / metric.total_tests
                if metric.total_tests > 0 else 0
            )
            series["duration"].append(metric.avg_duration)
            series["flaky_rate"].append(
                metric.flaky_tests / metric.total_tests
                if metric.total_tests > 0 else 0
            )
            
        return dict(series)

    async def _compute_summary(
        self,
        data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Compute summary statistics."""
        summaries = {}
        
        for metric, values in data.items():
            if metric == "timestamps":
                continue
                
            values = np.array(values)
            summaries[metric] = {
                "current": float(values[-1]),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "percentiles": {
                    "25": float(np.percentile(values, 25)),
                    "75": float(np.percentile(values, 75)),
                    "95": float(np.percentile(values, 95))
                }
            }
            
        return summaries

    async def _compute_trends(
        self,
        data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Compute trend analysis."""
        trends = {}
        
        for metric, values in data.items():
            if metric == "timestamps":
                continue
                
            values = np.array(values)
            
            # Simple linear regression
            x = np.arange(len(values))
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, values, rcond=None)[0]
            
            # Recent vs historical comparison
            split_idx = len(values) // 2
            recent = values[split_idx:]
            historical = values[:split_idx]
            
            trends[metric] = {
                "slope": float(slope),
                "direction": "up" if slope > 0 else "down",
                "recent_vs_historical": float(
                    np.mean(recent) / np.mean(historical)
                    if len(historical) > 0 and np.mean(historical) != 0
                    else 1.0
                )
            }
            
        return trends

    async def _compute_distributions(
        self,
        data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Compute distribution analysis."""
        distributions = {}
        
        for metric, values in data.items():
            if metric == "timestamps":
                continue
                
            values = np.array(values)
            
            # Compute histogram
            hist, bins = np.histogram(
                values,
                bins="auto",
                density=True
            )
            
            distributions[metric] = {
                "histogram": {
                    "counts": [float(x) for x in hist],
                    "bins": [float(x) for x in bins[:-1]]
                },
                "skewness": float(self._compute_skewness(values)),
                "kurtosis": float(self._compute_kurtosis(values))
            }
            
        return distributions

    async def _compute_correlations(
        self,
        data: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Compute correlation analysis."""
        correlations = {}
        metrics = [m for m in data.keys() if m != "timestamps"]
        
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                key = f"{metric1}_vs_{metric2}"
                correlations[key] = float(np.corrcoef(
                    data[metric1],
                    data[metric2]
                )[0, 1])
                
        return correlations

    def _compute_skewness(self, values: np.ndarray) -> float:
        """Compute distribution skewness."""
        if len(values) < 2:
            return 0.0
            
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
            
        return float(np.mean(((values - mean) / std) ** 3))

    def _compute_kurtosis(self, values: np.ndarray) -> float:
        """Compute distribution kurtosis."""
        if len(values) < 2:
            return 0.0
            
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
            
        return float(np.mean(((values - mean) / std) ** 4)) - 3

    async def get_failure_patterns(
        self,
        time_window: timedelta = timedelta(days=30)
    ) -> List[Dict[str, Any]]:
        """
        Analyze failure patterns across test runs.
        
        Args:
            time_window: Time window to analyze
            
        Returns:
            List of identified failure patterns
        """
        async with get_db_session() as session:
            # Get test runs
            runs = await self._get_test_runs(session, time_window)
            
            # Group failures by pattern
            patterns = self._group_failures(runs)
            
            # Analyze patterns
            return await self._analyze_patterns(patterns)

    async def _get_test_runs(
        self,
        session: AsyncSession,
        time_window: timedelta
    ) -> List[TestRun]:
        """Get test runs efficiently."""
        cutoff_time = datetime.utcnow() - time_window
        
        query = text("""
            SELECT r.*, array_agg(f.*) as failures
            FROM test_runs r
            LEFT JOIN test_failures f ON f.test_run_id = r.id
            WHERE r.created_at >= :cutoff
            GROUP BY r.id
            ORDER BY r.created_at DESC
        """)
        
        result = await session.execute(
            query,
            {"cutoff": cutoff_time}
        )
        
        return [TestRun(**row) for row in result]

    def _group_failures(
        self,
        runs: List[TestRun]
    ) -> List[Dict[str, Any]]:
        """Group failures by pattern."""
        patterns = defaultdict(list)
        
        for run in runs:
            for failure in run.failures:
                # Create failure signature
                signature = self._create_failure_signature(failure)
                patterns[signature].append(failure)
                
        return [
            {
                "signature": signature,
                "failures": failures,
                "count": len(failures),
                "first_seen": min(f.created_at for f in failures),
                "last_seen": max(f.created_at for f in failures)
            }
            for signature, failures in patterns.items()
        ]

    def _create_failure_signature(self, failure: Any) -> str:
        """Create unique signature for a failure."""
        # Normalize error message
        error_msg = (failure.error_message or "").lower()
        
        # Extract key parts (customize based on your error format)
        parts = [
            failure.test_name,
            error_msg.split("\n")[0] if error_msg else "",
            failure.error_type or ""
        ]
        
        return "||".join(parts)
