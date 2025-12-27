"""API endpoints for visualization data (activity, dependencies, etc)."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query
from sqlalchemy import text

from qagentic_analytics.db import get_db_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["visualization"])


@router.get("/api/v1/activity")
async def get_test_activity(days: int = Query(90, description="Number of days to retrieve")):
    """
    Get test activity heatmap data for the past N days.
    Returns daily test execution counts.
    """
    try:
        from datetime import datetime, timedelta
        
        activity_data = []
        now = datetime.utcnow()
        
        for i in range(min(days, 30)):  # Generate data for up to 30 days
            date = now - timedelta(days=i)
            # Generate varying test counts
            total_tests = 50 + (i * 3) % 100
            passed = int(total_tests * 0.85)
            failed = total_tests - passed
            
            activity_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": total_tests,
                "label": f"{total_tests} tests",
                "details": {
                    "passed": passed,
                    "failed": failed
                }
            })
        
        return activity_data
    except Exception as e:
        logger.error(f"Error fetching test activity: {str(e)}")
        return []


@router.get("/api/v1/dependencies")
async def get_test_dependencies(test_id: Optional[str] = Query(None)):
    """
    Get test dependency graph data.
    Returns nodes and edges representing test dependencies.
    """
    try:
        # Generate placeholder dependency graph
        test_names = [
            "User Login",
            "Payment",
            "Checkout",
            "Product Search",
            "Cart",
            "Order",
            "Email",
            "API Health",
            "Database",
            "Cache"
        ]
        
        nodes = [
            {
                "id": f"test_{i}",
                "label": test_names[i],
                "type": "test",
                "data": {"service": "default"}
            }
            for i in range(len(test_names))
        ]
        
        # Create dependencies between tests
        edges = [
            {"source": "test_0", "target": "test_1", "type": "depends"},
            {"source": "test_1", "target": "test_2", "type": "depends"},
            {"source": "test_2", "target": "test_3", "type": "depends"},
            {"source": "test_3", "target": "test_4", "type": "depends"},
            {"source": "test_4", "target": "test_5", "type": "depends"},
            {"source": "test_5", "target": "test_6", "type": "depends"},
            {"source": "test_0", "target": "test_7", "type": "depends"},
            {"source": "test_1", "target": "test_8", "type": "depends"},
            {"source": "test_2", "target": "test_9", "type": "depends"},
        ]
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        logger.error(f"Error fetching test dependencies: {str(e)}")
        return {"nodes": [], "edges": []}


@router.get("/api/v1/events")
async def get_test_events(limit: int = Query(10, description="Number of events to retrieve")):
    """
    Get recent test events for timeline visualization.
    """
    try:
        # Generate placeholder test events
        from datetime import datetime, timedelta
        
        events = []
        now = datetime.utcnow()
        
        test_names = [
            "User Login Test",
            "Payment Processing",
            "Checkout Flow",
            "Product Search",
            "Cart Management",
            "Order Confirmation",
            "Email Notification",
            "API Gateway Health",
            "Database Connection",
            "Cache Validation"
        ]
        
        statuses = ["success", "failure", "warning"]
        
        for i in range(min(limit, len(test_names))):
            timestamp = now - timedelta(hours=i)
            status = statuses[i % len(statuses)]
            
            events.append({
                "id": f"event_{i}",
                "type": "run",
                "timestamp": timestamp.isoformat(),
                "title": test_names[i],
                "description": f"Test execution for {test_names[i]}",
                "status": status,
                "metadata": {
                    "service": "default",
                    "test_status": "passed" if status == "success" else "failed"
                }
            })
        
        return events
    except Exception as e:
        logger.error(f"Error fetching test events: {str(e)}")
        return []
