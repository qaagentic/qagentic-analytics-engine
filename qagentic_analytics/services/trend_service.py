"""Trend analysis service for test metrics and patterns."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from qagentic_analytics.db import get_db_session
from qagentic_analytics.models.trend import TrendAnalysis
from qagentic_analytics.models.metrics import TestMetrics
from qagentic_analytics.utils.time_series import resample_time_series

logger = logging.getLogger(__name__)

class TrendService:
    """Service for analyzing test quality trends."""

    async def analyze_trends(
        self,
        time_window: timedelta = timedelta(days=90),
        interval: str = "1D",  # Daily by default
        min_data_points: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze trends in test metrics.
        
        Args:
            time_window: Time window to analyze
            interval: Resampling interval (e.g., "1H", "1D", "1W")
            min_data_points: Minimum points needed for trend analysis
            
        Returns:
            Dictionary containing trend analysis results
        """
        async with get_db_session() as session:
            # Get historical metrics
            metrics = await self._get_metrics(session, time_window)
            
            if len(metrics) < min_data_points:
                logger.warning(f"Insufficient data points for trend analysis: {len(metrics)}")
                return {}
            
            # Convert to time series
            ts_data = self._prepare_time_series(metrics, interval)
            
            # Analyze different aspects
            results = {
                "overall_health": await self._analyze_overall_health(ts_data),
                "failure_trends": await self._analyze_failure_trends(ts_data),
                "performance_trends": await self._analyze_performance_trends(ts_data),
                "flakiness_trends": await self._analyze_flakiness_trends(ts_data),
                "stability_score": await self._calculate_stability_score(ts_data)
            }
            
            # Generate insights
            insights = await self._generate_insights(results)
            results["insights"] = insights
            
            # Save analysis
            await self._save_analysis(session, results)
            
            return results

    async def _get_metrics(
        self,
        session: AsyncSession,
        time_window: timedelta
    ) -> List[TestMetrics]:
        """Get historical test metrics."""
        cutoff_time = datetime.utcnow() - time_window
        
        metrics = await session.query(TestMetrics).filter(
            TestMetrics.timestamp >= cutoff_time
        ).order_by(TestMetrics.timestamp).all()
        
        return metrics

    def _prepare_time_series(
        self,
        metrics: List[TestMetrics],
        interval: str
    ) -> pd.DataFrame:
        """Prepare time series data for analysis."""
        # Convert to DataFrame
        data = []
        for metric in metrics:
            data.append({
                "timestamp": metric.timestamp,
                "total_tests": metric.total_tests,
                "passed_tests": metric.passed_tests,
                "failed_tests": metric.failed_tests,
                "skipped_tests": metric.skipped_tests,
                "avg_duration": metric.avg_duration,
                "flaky_tests": metric.flaky_tests
            })
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        # Resample to regular intervals
        resampled = resample_time_series(df, interval)
        
        # Add derived metrics
        resampled["pass_rate"] = resampled["passed_tests"] / resampled["total_tests"]
        resampled["fail_rate"] = resampled["failed_tests"] / resampled["total_tests"]
        resampled["flaky_rate"] = resampled["flaky_tests"] / resampled["total_tests"]
        
        return resampled

    async def _analyze_overall_health(
        self,
        ts_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze overall test health trends."""
        # Calculate trend slopes
        pass_rate_slope = self._calculate_trend_slope(ts_data["pass_rate"])
        duration_slope = self._calculate_trend_slope(ts_data["avg_duration"])
        
        # Calculate stability metrics
        pass_rate_stability = self._calculate_stability(ts_data["pass_rate"])
        duration_stability = self._calculate_stability(ts_data["avg_duration"])
        
        # Determine health status
        health_score = self._calculate_health_score(
            pass_rate_slope,
            duration_slope,
            pass_rate_stability,
            duration_stability
        )
        
        return {
            "health_score": health_score,
            "trends": {
                "pass_rate": {
                    "slope": float(pass_rate_slope),
                    "stability": float(pass_rate_stability)
                },
                "duration": {
                    "slope": float(duration_slope),
                    "stability": float(duration_stability)
                }
            },
            "current_stats": {
                "pass_rate": float(ts_data["pass_rate"].iloc[-1]),
                "avg_duration": float(ts_data["avg_duration"].iloc[-1])
            }
        }

    async def _analyze_failure_trends(
        self,
        ts_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze failure patterns and trends."""
        # Analyze failure rate trends
        failure_slope = self._calculate_trend_slope(ts_data["fail_rate"])
        failure_stability = self._calculate_stability(ts_data["fail_rate"])
        
        # Detect failure patterns
        patterns = self._detect_failure_patterns(ts_data)
        
        # Predict future failures
        failure_forecast = self._forecast_failures(ts_data)
        
        return {
            "trend": {
                "slope": float(failure_slope),
                "stability": float(failure_stability)
            },
            "patterns": patterns,
            "forecast": failure_forecast
        }

    async def _analyze_performance_trends(
        self,
        ts_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze test execution performance trends."""
        # Analyze duration trends
        duration_trend = self._analyze_duration_trend(ts_data)
        
        # Detect performance anomalies
        anomalies = self._detect_performance_anomalies(ts_data)
        
        # Calculate performance metrics
        perf_metrics = self._calculate_performance_metrics(ts_data)
        
        return {
            "trend": duration_trend,
            "anomalies": anomalies,
            "metrics": perf_metrics
        }

    async def _analyze_flakiness_trends(
        self,
        ts_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze test flakiness trends."""
        # Analyze flaky test trends
        flaky_slope = self._calculate_trend_slope(ts_data["flaky_rate"])
        flaky_stability = self._calculate_stability(ts_data["flaky_rate"])
        
        # Identify consistently flaky tests
        consistent_flaky = self._identify_consistent_flaky(ts_data)
        
        return {
            "trend": {
                "slope": float(flaky_slope),
                "stability": float(flaky_stability)
            },
            "consistent_flaky": consistent_flaky,
            "current_rate": float(ts_data["flaky_rate"].iloc[-1])
        }

    def _calculate_trend_slope(self, series: pd.Series) -> float:
        """Calculate trend slope using linear regression."""
        if series.empty:
            return 0.0
            
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        
        # Handle missing values
        mask = ~np.isnan(y).reshape(-1)
        if not np.any(mask):
            return 0.0
            
        X = X[mask]
        y = y[mask]
        
        # Standardize for comparable slopes
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        return model.coef_[0][0]

    def _calculate_stability(self, series: pd.Series) -> float:
        """Calculate stability score (inverse of volatility)."""
        if series.empty:
            return 1.0
            
        # Remove missing values
        series = series.dropna()
        if series.empty:
            return 1.0
            
        # Calculate coefficient of variation
        cv = series.std() / series.mean() if series.mean() != 0 else 0
        
        # Convert to stability score (1 - normalized cv)
        stability = 1 - min(cv, 1)
        
        return stability

    def _calculate_health_score(
        self,
        pass_slope: float,
        duration_slope: float,
        pass_stability: float,
        duration_stability: float
    ) -> float:
        """Calculate overall health score."""
        # Weights for different components
        weights = {
            "pass_slope": 0.3,
            "duration_slope": 0.2,
            "pass_stability": 0.3,
            "duration_stability": 0.2
        }
        
        # Normalize slopes to [-1, 1]
        pass_slope_norm = np.tanh(pass_slope)
        duration_slope_norm = -np.tanh(duration_slope)  # Negative because lower duration is better
        
        # Calculate weighted score
        score = (
            weights["pass_slope"] * pass_slope_norm +
            weights["duration_slope"] * duration_slope_norm +
            weights["pass_stability"] * pass_stability +
            weights["duration_stability"] * duration_stability
        )
        
        # Convert to 0-1 scale
        return (score + 1) / 2

    def _detect_failure_patterns(
        self,
        ts_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detect patterns in test failures."""
        patterns = []
        
        # Detect periodic patterns
        periodic = self._detect_periodic_patterns(ts_data["fail_rate"])
        if periodic:
            patterns.extend(periodic)
            
        # Detect sudden spikes
        spikes = self._detect_spikes(ts_data["fail_rate"])
        if spikes:
            patterns.extend(spikes)
            
        return patterns

    def _forecast_failures(
        self,
        ts_data: pd.DataFrame,
        horizon: int = 7  # Days
    ) -> Dict[str, Any]:
        """Forecast future failure rates."""
        # Simple linear extrapolation for now
        # TODO: Implement more sophisticated forecasting
        
        current_rate = ts_data["fail_rate"].iloc[-1]
        slope = self._calculate_trend_slope(ts_data["fail_rate"])
        
        forecast = {
            "horizon": horizon,
            "values": [
                max(0, min(1, current_rate + slope * i))
                for i in range(1, horizon + 1)
            ],
            "confidence": 0.7  # Placeholder
        }
        
        return forecast

    async def _calculate_stability_score(
        self,
        ts_data: pd.DataFrame
    ) -> float:
        """Calculate overall test suite stability score."""
        weights = {
            "pass_rate": 0.4,
            "flaky_rate": 0.3,
            "duration": 0.3
        }
        
        pass_stability = self._calculate_stability(ts_data["pass_rate"])
        flaky_stability = 1 - self._calculate_stability(ts_data["flaky_rate"])
        duration_stability = self._calculate_stability(ts_data["avg_duration"])
        
        score = (
            weights["pass_rate"] * pass_stability +
            weights["flaky_rate"] * flaky_stability +
            weights["duration"] * duration_stability
        )
        
        return score

    async def _generate_insights(
        self,
        results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable insights from trend analysis."""
        insights = []
        
        # Health score insights
        health_score = results["overall_health"]["health_score"]
        if health_score < 0.6:
            insights.append({
                "type": "warning",
                "title": "Declining Test Health",
                "description": "Overall test health score is below threshold",
                "score": health_score,
                "recommendation": "Review recent changes and test infrastructure"
            })
            
        # Failure trend insights
        failure_slope = results["failure_trends"]["trend"]["slope"]
        if failure_slope > 0.1:
            insights.append({
                "type": "alert",
                "title": "Increasing Failure Rate",
                "description": "Test failure rate is trending upward",
                "metric": failure_slope,
                "recommendation": "Investigate recent failures for common patterns"
            })
            
        # Performance insights
        perf_metrics = results["performance_trends"]["metrics"]
        if perf_metrics.get("degradation_rate", 0) > 0.1:
            insights.append({
                "type": "warning",
                "title": "Performance Degradation",
                "description": "Test execution times are increasing",
                "metric": perf_metrics["degradation_rate"],
                "recommendation": "Review test implementation and infrastructure"
            })
            
        return insights

    async def _save_analysis(
        self,
        session: AsyncSession,
        results: Dict[str, Any]
    ) -> None:
        """Save trend analysis results."""
        analysis = TrendAnalysis(
            timestamp=datetime.utcnow(),
            health_score=results["overall_health"]["health_score"],
            results=results
        )
        
        session.add(analysis)
        await session.commit()
