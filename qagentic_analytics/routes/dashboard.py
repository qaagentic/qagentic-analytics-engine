"""Dashboard metrics endpoint for the UI."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from qagentic_analytics.services.metrics import MetricsService
from qagentic_analytics.db import get_engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"], prefix="/dashboard")


@router.get("/metrics")
async def get_dashboard_metrics():
    """
    Get dashboard metrics in the format expected by the UI.
    Returns real data from the database.
    """
    try:
        # Initialize metrics service with database engine
        engine = get_engine()
        metrics_service = MetricsService()
        metrics_service.db = engine
        
        # Get metrics summary from the service
        summary = await metrics_service.get_summary()
        
        # Get trend data for pass rate over last 7 days
        trends = await metrics_service.get_trends(
            metric="pass_rate",
            interval="day"
        )
        
        # Extract trend arrays for the chart
        pass_rate_trend = []
        failures_trend = []
        flaky_trend = []
        
        if trends and len(trends) > 0:
            trend = trends[0]
            pass_rate_trend = [dp.value * 100 for dp in trend.data_points[-7:]]  # Last 7 days
        
        # Pad arrays to 7 elements if needed
        while len(pass_rate_trend) < 7:
            pass_rate_trend.insert(0, 0)
        while len(failures_trend) < 7:
            failures_trend.insert(0, 0)
        while len(flaky_trend) < 7:
            flaky_trend.insert(0, 0)
        
        # Get failure trends
        failure_trends = await metrics_service.get_trends(
            metric="failure_rate",
            interval="day"
        )
        
        if failure_trends and len(failure_trends) > 0:
            trend = failure_trends[0]
            failures_trend = [dp.value for dp in trend.data_points[-7:]]
            
        # Get flaky trends
        flaky_trends = await metrics_service.get_trends(
            metric="flaky_rate",
            interval="day"
        )
        
        if flaky_trends and len(flaky_trends) > 0:
            trend = flaky_trends[0]
            flaky_trend = [dp.value for dp in trend.data_points[-7:]]
        
        # Calculate changes (simple comparison with previous period)
        pass_rate_change = 0
        failed_tests_change = 0
        duration_change = 0
        flaky_tests_change = 0
        
        if len(pass_rate_trend) >= 2 and pass_rate_trend[-2] > 0:
            pass_rate_change = ((pass_rate_trend[-1] - pass_rate_trend[-2]) / pass_rate_trend[-2]) * 100
            
        if len(failures_trend) >= 2 and failures_trend[-2] > 0:
            failed_tests_change = ((failures_trend[-1] - failures_trend[-2]) / failures_trend[-2]) * 100
        
        # Return data in the format the UI expects
        return {
            # Trend data for charts
            "pass_rate_trend": pass_rate_trend,
            "failures_trend": failures_trend,
            "flaky_trend": flaky_trend,
            
            # Summary metrics for cards
            "current_pass_rate": summary.overall.pass_rate * 100,
            "failed_tests_count": summary.overall.failed_tests,
            "average_duration": summary.overall.mttr_minutes * 60,  # Convert to seconds
            "flaky_tests_count": summary.overall.flaky_tests,
            
            # Change percentages
            "pass_rate_change": pass_rate_change,
            "failed_tests_change": failed_tests_change,
            "duration_change": duration_change,
            "flaky_tests_change": flaky_tests_change,
            
            # Additional summary data
            "total_runs": summary.overall.total_runs,
            "total_tests": summary.overall.total_tests,
            "passed_tests": summary.overall.passed_tests,
            "skipped_tests": summary.overall.skipped_tests,
        }
        
    except Exception as e:
        logger.exception(f"Error fetching dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard metrics: {str(e)}")
