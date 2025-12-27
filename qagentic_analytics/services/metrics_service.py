"""Real-time metrics computation service."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import WebSocket
import numpy as np

from qagentic_analytics.db import get_db_session
from qagentic_analytics.models.metrics import TestMetrics
from qagentic_analytics.models.test_run import TestRun
from qagentic_analytics.cache import get_cache
from qagentic_analytics.utils.stats import calculate_percentiles
from qagentic_analytics.utils.streaming import MetricsStream

logger = logging.getLogger(__name__)

class MetricsService:
    """Service for real-time metrics computation and streaming."""

    def __init__(self):
        self.cache = get_cache()
        self.metrics_stream = MetricsStream()
        self._update_lock = asyncio.Lock()
        self._subscribers: List[WebSocket] = []

    async def compute_metrics(
        self,
        test_run_id: str,
        realtime: bool = True
    ) -> Dict[str, Any]:
        """
        Compute metrics for a test run.
        
        Args:
            test_run_id: ID of the test run
            realtime: Whether to compute metrics in real-time
            
        Returns:
            Dictionary containing computed metrics
        """
        # Try cache first
        cache_key = f"metrics:{test_run_id}"
        cached = await self.cache.get(cache_key)
        if cached and not realtime:
            return cached
            
        async with get_db_session() as session:
            # Get test run data
            test_run = await self._get_test_run(session, test_run_id)
            if not test_run:
                raise ValueError(f"Test run not found: {test_run_id}")
                
            # Compute metrics
            metrics = await self._compute_test_run_metrics(session, test_run)
            
            # Cache results
            if not realtime:
                await self.cache.set(cache_key, metrics, expire=3600)  # 1 hour
                
            # Notify subscribers if realtime
            if realtime:
                await self._notify_subscribers(test_run_id, metrics)
                
            return metrics

    async def get_historical_metrics(
        self,
        time_window: timedelta = timedelta(days=30),
        interval: str = "1D"
    ) -> Dict[str, Any]:
        """Get historical metrics with aggregations."""
        async with get_db_session() as session:
            metrics = await self._get_metrics_history(session, time_window)
            
            # Aggregate metrics
            aggregated = self._aggregate_metrics(metrics, interval)
            
            return {
                "interval": interval,
                "metrics": aggregated
            }

    async def subscribe_to_metrics(
        self,
        websocket: WebSocket,
        test_run_id: Optional[str] = None
    ) -> None:
        """Subscribe to real-time metrics updates."""
        await websocket.accept()
        self._subscribers.append(websocket)
        
        try:
            while True:
                # Send metrics updates
                if test_run_id:
                    metrics = await self.compute_metrics(test_run_id, realtime=True)
                    await websocket.send_json(metrics)
                    
                # Wait for next update
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in metrics subscription: {str(e)}")
            
        finally:
            self._subscribers.remove(websocket)
            await websocket.close()

    async def _get_test_run(
        self,
        session: AsyncSession,
        test_run_id: str
    ) -> Optional[TestRun]:
        """Get test run data from database."""
        return await session.query(TestRun).filter(
            TestRun.id == test_run_id
        ).first()

    async def _compute_test_run_metrics(
        self,
        session: AsyncSession,
        test_run: TestRun
    ) -> Dict[str, Any]:
        """Compute comprehensive metrics for a test run."""
        async with self._update_lock:
            # Basic metrics
            basic_metrics = await self._compute_basic_metrics(test_run)
            
            # Performance metrics
            perf_metrics = await self._compute_performance_metrics(test_run)
            
            # Quality metrics
            quality_metrics = await self._compute_quality_metrics(session, test_run)
            
            # Combine all metrics
            metrics = {
                "test_run_id": test_run.id,
                "timestamp": datetime.utcnow().isoformat(),
                "basic": basic_metrics,
                "performance": perf_metrics,
                "quality": quality_metrics,
                "metadata": {
                    "environment": test_run.environment,
                    "branch": test_run.branch,
                    "commit": test_run.commit_hash
                }
            }
            
            # Save metrics
            await self._save_metrics(session, metrics)
            
            return metrics

    async def _compute_basic_metrics(
        self,
        test_run: TestRun
    ) -> Dict[str, Any]:
        """Compute basic test metrics."""
        total_tests = len(test_run.test_cases)
        passed_tests = sum(1 for t in test_run.test_cases if t.status == "passed")
        failed_tests = sum(1 for t in test_run.test_cases if t.status == "failed")
        skipped_tests = sum(1 for t in test_run.test_cases if t.status == "skipped")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "fail_rate": failed_tests / total_tests if total_tests > 0 else 0
        }

    async def _compute_performance_metrics(
        self,
        test_run: TestRun
    ) -> Dict[str, Any]:
        """Compute performance-related metrics."""
        durations = [t.duration for t in test_run.test_cases if t.duration is not None]
        
        if not durations:
            return {
                "avg_duration": 0,
                "total_duration": 0,
                "percentiles": {}
            }
            
        return {
            "avg_duration": np.mean(durations),
            "total_duration": sum(durations),
            "percentiles": calculate_percentiles(durations, [50, 75, 90, 95, 99])
        }

    async def _compute_quality_metrics(
        self,
        session: AsyncSession,
        test_run: TestRun
    ) -> Dict[str, Any]:
        """Compute quality-related metrics."""
        # Get historical data for comparison
        historical = await self._get_historical_test_runs(
            session,
            test_run.branch,
            limit=10
        )
        
        # Identify flaky tests
        flaky_tests = await self._identify_flaky_tests(historical)
        
        # Calculate stability metrics
        stability = await self._calculate_stability_metrics(historical)
        
        return {
            "flaky_tests": len(flaky_tests),
            "flaky_rate": len(flaky_tests) / len(test_run.test_cases),
            "stability_score": stability["score"],
            "reliability_score": stability["reliability"],
            "trend": stability["trend"]
        }

    async def _get_historical_test_runs(
        self,
        session: AsyncSession,
        branch: str,
        limit: int = 10
    ) -> List[TestRun]:
        """Get historical test runs for comparison."""
        return await session.query(TestRun).filter(
            TestRun.branch == branch
        ).order_by(TestRun.created_at.desc()).limit(limit).all()

    async def _identify_flaky_tests(
        self,
        test_runs: List[TestRun]
    ) -> List[Dict[str, Any]]:
        """Identify flaky tests from historical runs."""
        test_results = {}
        
        # Collect results by test
        for run in test_runs:
            for test in run.test_cases:
                if test.name not in test_results:
                    test_results[test.name] = []
                test_results[test.name].append(test.status)
                
        # Identify flaky tests
        flaky_tests = []
        for test_name, results in test_results.items():
            if len(results) >= 2:  # Need at least 2 runs
                status_changes = sum(
                    1 for i in range(1, len(results))
                    if results[i] != results[i-1]
                )
                if status_changes > 0:
                    flaky_tests.append({
                        "name": test_name,
                        "flake_rate": status_changes / (len(results) - 1),
                        "last_status": results[-1]
                    })
                    
        return flaky_tests

    async def _calculate_stability_metrics(
        self,
        test_runs: List[TestRun]
    ) -> Dict[str, Any]:
        """Calculate test suite stability metrics."""
        if not test_runs:
            return {
                "score": 1.0,
                "reliability": 1.0,
                "trend": "stable"
            }
            
        # Calculate pass rates
        pass_rates = []
        for run in test_runs:
            total = len(run.test_cases)
            passed = sum(1 for t in run.test_cases if t.status == "passed")
            pass_rates.append(passed / total if total > 0 else 0)
            
        # Calculate stability score
        stability_score = np.mean(pass_rates)
        
        # Calculate reliability (inverse of variance)
        reliability = 1 - np.std(pass_rates) if len(pass_rates) > 1 else 1.0
        
        # Determine trend
        if len(pass_rates) >= 2:
            recent_avg = np.mean(pass_rates[-2:])
            older_avg = np.mean(pass_rates[:-2]) if len(pass_rates) > 2 else pass_rates[0]
            
            if recent_avg > older_avg + 0.05:
                trend = "improving"
            elif recent_avg < older_avg - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
            
        return {
            "score": stability_score,
            "reliability": reliability,
            "trend": trend
        }

    async def _get_metrics_history(
        self,
        session: AsyncSession,
        time_window: timedelta
    ) -> List[TestMetrics]:
        """Get historical metrics from database."""
        cutoff_time = datetime.utcnow() - time_window
        
        return await session.query(TestMetrics).filter(
            TestMetrics.timestamp >= cutoff_time
        ).order_by(TestMetrics.timestamp).all()

    def _aggregate_metrics(
        self,
        metrics: List[TestMetrics],
        interval: str
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by time interval."""
        if not metrics:
            return []
            
        # Group by interval
        intervals = {}
        for metric in metrics:
            key = self._get_interval_key(metric.timestamp, interval)
            if key not in intervals:
                intervals[key] = []
            intervals[key].append(metric)
            
        # Aggregate each interval
        aggregated = []
        for key in sorted(intervals.keys()):
            interval_metrics = intervals[key]
            aggregated.append({
                "interval": key,
                "metrics": {
                    "total_tests": np.mean([m.total_tests for m in interval_metrics]),
                    "pass_rate": np.mean([m.passed_tests / m.total_tests for m in interval_metrics]),
                    "avg_duration": np.mean([m.avg_duration for m in interval_metrics]),
                    "flaky_rate": np.mean([m.flaky_tests / m.total_tests for m in interval_metrics])
                }
            })
            
        return aggregated

    def _get_interval_key(self, timestamp: datetime, interval: str) -> str:
        """Get key for a time interval."""
        if interval == "1H":
            return timestamp.strftime("%Y-%m-%d %H:00")
        elif interval == "1D":
            return timestamp.strftime("%Y-%m-%d")
        elif interval == "1W":
            return timestamp.strftime("%Y-W%W")
        else:
            return timestamp.strftime("%Y-%m")

    async def _notify_subscribers(
        self,
        test_run_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Notify subscribers of metrics updates."""
        message = {
            "type": "metrics_update",
            "test_run_id": test_run_id,
            "data": metrics
        }
        
        for websocket in self._subscribers:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending metrics update: {str(e)}")

    async def _save_metrics(
        self,
        session: AsyncSession,
        metrics: Dict[str, Any]
    ) -> None:
        """Save metrics to database."""
        test_metrics = TestMetrics(
            test_run_id=metrics["test_run_id"],
            timestamp=datetime.fromisoformat(metrics["timestamp"]),
            total_tests=metrics["basic"]["total_tests"],
            passed_tests=metrics["basic"]["passed_tests"],
            failed_tests=metrics["basic"]["failed_tests"],
            skipped_tests=metrics["basic"]["skipped_tests"],
            avg_duration=metrics["performance"]["avg_duration"],
            flaky_tests=metrics["quality"]["flaky_tests"],
            metadata=metrics["metadata"]
        )
        
        session.add(test_metrics)
        await session.commit()
