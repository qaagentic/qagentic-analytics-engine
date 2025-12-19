"""Dashboard metrics endpoint for the UI."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from qagentic_analytics.db import get_metrics_from_db, get_test_runs_trend

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"], prefix="/dashboard")


@router.get("/metrics")
async def get_dashboard_metrics():
    """
    Get dashboard metrics in the format expected by the UI.
    Returns real data from the database.
    """
    try:
        # Get real metrics from database
        db_metrics = get_metrics_from_db()
        
        # Get trend data for the last 7 days
        trend_data = get_test_runs_trend(days=7)
        
        # Extract trend arrays for the chart
        pass_rate_trend = [d.get("pass_rate", 0) for d in trend_data] if trend_data else [0] * 7
        failures_trend = [d.get("failed_tests", 0) for d in trend_data] if trend_data else [0] * 7
        flaky_trend = [d.get("flaky_tests", 0) for d in trend_data] if trend_data else [0] * 7
        
        # Pad arrays to 7 elements if needed
        while len(pass_rate_trend) < 7:
            pass_rate_trend.insert(0, 0)
        while len(failures_trend) < 7:
            failures_trend.insert(0, 0)
        while len(flaky_trend) < 7:
            flaky_trend.insert(0, 0)
        
        # Calculate changes (simple comparison with previous period)
        pass_rate_change = 0
        failed_tests_change = 0
        duration_change = 0
        flaky_tests_change = 0
        
        if len(pass_rate_trend) >= 2:
            if pass_rate_trend[-2] > 0:
                pass_rate_change = ((pass_rate_trend[-1] - pass_rate_trend[-2]) / pass_rate_trend[-2]) * 100
        
        # Return data in the format the UI expects
        return {
            # Trend data for charts
            "pass_rate_trend": pass_rate_trend,
            "failures_trend": failures_trend,
            "flaky_trend": flaky_trend,
            
            # Summary metrics for cards
            "current_pass_rate": db_metrics.get("pass_rate", 0),
            "failed_tests_count": db_metrics.get("failed_tests", 0),
            "average_duration": int(db_metrics.get("avg_duration_ms", 0) / 1000),  # Convert to seconds
            "flaky_tests_count": db_metrics.get("flaky_tests", 0),
            
            # Change percentages
            "pass_rate_change": pass_rate_change,
            "failed_tests_change": failed_tests_change,
            "duration_change": duration_change,
            "flaky_tests_change": flaky_tests_change,
            
            # Additional summary data
            "total_runs": db_metrics.get("total_runs", 0),
            "total_tests": db_metrics.get("total_tests", 0),
            "passed_tests": db_metrics.get("passed_tests", 0),
            "skipped_tests": db_metrics.get("skipped_tests", 0),
        }
        
    except Exception as e:
        logger.exception(f"Error fetching dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard metrics: {str(e)}")
