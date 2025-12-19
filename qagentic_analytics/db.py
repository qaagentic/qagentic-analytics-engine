"""Database utilities for the Analytics Engine using psycopg2."""

import logging
import os
import psycopg2
from typing import Dict, List

logger = logging.getLogger(__name__)


def get_connection():
    """Get a database connection using psycopg2."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        dbname=os.getenv("POSTGRES_DB", "qagentic"),
    )


def get_metrics_from_db() -> dict:
    """Query real metrics from the database."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Get test run counts and stats from public.test_runs
        cur.execute("""
            SELECT 
                COUNT(*) as total_runs,
                COALESCE(SUM(total_tests), 0) as total_tests,
                COALESCE(SUM(passed_tests), 0) as passed_tests,
                COALESCE(SUM(failed_tests), 0) as failed_tests,
                COALESCE(SUM(skipped_tests), 0) as skipped_tests,
                COALESCE(SUM(flaky_tests), 0) as flaky_tests,
                COALESCE(AVG(duration_ms), 0) as avg_duration_ms
            FROM public.test_runs
        """)
        row = cur.fetchone()
        
        total_runs = row[0] or 0
        total_tests = row[1] or 0
        passed_tests = row[2] or 0
        failed_tests = row[3] or 0
        skipped_tests = row[4] or 0
        flaky_tests = row[5] or 0
        avg_duration_ms = row[6] or 0
        
        # Calculate rates
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        failure_rate = (failed_tests / total_tests * 100) if total_tests > 0 else 0
        flaky_rate = (flaky_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Get test runs from gateway tables as well
        cur.execute("""
            SELECT COUNT(*) as total_runs
            FROM qagentic.gateway_test_runs
        """)
        gateway_row = cur.fetchone()
        gateway_runs = gateway_row[0] or 0
        
        # Combine counts
        combined_runs = total_runs + gateway_runs
        
        cur.close()
        
        return {
            "total_runs": combined_runs,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "flaky_tests": flaky_tests,
            "pass_rate": pass_rate,
            "failure_rate": failure_rate,
            "flaky_rate": flaky_rate,
            "avg_duration_ms": float(avg_duration_ms),
        }
    except Exception as e:
        logger.exception(f"Error getting metrics from database: {e}")
        return {
            "total_runs": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "flaky_tests": 0,
            "pass_rate": 0,
            "failure_rate": 0,
            "flaky_rate": 0,
            "avg_duration_ms": 0,
        }
    finally:
        if conn:
            conn.close()


def get_test_runs_trend(days: int = 7) -> list:
    """Get test runs trend for the last N days."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute(f"""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as run_count,
                COALESCE(SUM(total_tests), 0) as total_tests,
                COALESCE(SUM(passed_tests), 0) as passed_tests,
                COALESCE(SUM(failed_tests), 0) as failed_tests,
                COALESCE(SUM(flaky_tests), 0) as flaky_tests
            FROM public.test_runs
            WHERE created_at >= NOW() - INTERVAL '{days} days'
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at)
        """)
        
        trend_data = []
        for row in cur.fetchall():
            total = row[3] + row[4] if (row[3] + row[4]) > 0 else 1
            trend_data.append({
                "date": str(row[0]),
                "run_count": row[1],
                "total_tests": row[2],
                "passed_tests": row[3],
                "failed_tests": row[4],
                "flaky_tests": row[5],
                "pass_rate": (row[3] / total * 100) if total > 0 else 0,
            })
        
        cur.close()
        return trend_data
    except Exception as e:
        logger.exception(f"Error getting test runs trend: {e}")
        return []
    finally:
        if conn:
            conn.close()
