"""Database management and optimization."""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    AsyncEngine,
    async_sessionmaker
)
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy.orm import sessionmaker

from qagentic_analytics.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global engine instance
_engine: Optional[AsyncEngine] = None

def create_engine() -> AsyncEngine:
    """
    Create database engine with optimized settings.
    
    Returns:
        AsyncEngine instance
    """
    return create_async_engine(
        str(settings.DATABASE_URL),
        echo=settings.SQL_DEBUG,
        future=True,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_recycle=settings.DB_POOL_RECYCLE,
        pool_pre_ping=True,
        poolclass=AsyncAdaptedQueuePool
    )

def get_engine() -> AsyncEngine:
    """
    Get the global database engine instance.
    
    Returns:
        AsyncEngine instance
        
    Raises:
        RuntimeError: If engine not initialized
    """
    global _engine
    if not _engine:
        _engine = create_engine()
    return _engine

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session.
    
    Yields:
        AsyncSession instance
    """
    engine = get_engine()
    
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False
    )
    
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def optimize_queries() -> None:
    """Apply query optimizations."""
    async with get_db_session() as session:
        # Update table statistics
        await session.execute("ANALYZE")
        
        # Optimize common queries
        await session.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_metrics_timestamp
            ON test_metrics (timestamp DESC)
        """)
        
        await session.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_runs_created_at
            ON test_runs (created_at DESC)
        """)
        
        await session.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_failures_test_run_id
            ON test_failures (test_run_id)
        """)
        
        # Create materialized view for common aggregations
        await session.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS test_metrics_daily AS
            SELECT
                date_trunc('day', timestamp) as day,
                COUNT(*) as total_runs,
                AVG(pass_rate) as avg_pass_rate,
                AVG(duration) as avg_duration,
                SUM(total_tests) as total_tests,
                SUM(passed_tests) as passed_tests,
                SUM(failed_tests) as failed_tests
            FROM test_metrics
            GROUP BY date_trunc('day', timestamp)
        """)
        
        # Create function to refresh materialized view
        await session.execute("""
            CREATE OR REPLACE FUNCTION refresh_test_metrics_daily()
            RETURNS trigger AS $$
            BEGIN
                REFRESH MATERIALIZED VIEW CONCURRENTLY test_metrics_daily;
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        # Create trigger to refresh view
        await session.execute("""
            DROP TRIGGER IF EXISTS refresh_test_metrics_daily_trigger
            ON test_metrics;
            
            CREATE TRIGGER refresh_test_metrics_daily_trigger
            AFTER INSERT OR UPDATE OR DELETE
            ON test_metrics
            FOR EACH STATEMENT
            EXECUTE FUNCTION refresh_test_metrics_daily();
        """)
        
        # Optimize for pattern matching
        await session.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_failures_error_pattern
            ON test_failures USING gin(to_tsvector('english', error_message))
        """)
        
        # Create partial indexes for common filters
        await session.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_runs_failed
            ON test_runs (created_at)
            WHERE status = 'failed'
        """)
        
        await session.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_metrics_low_pass_rate
            ON test_metrics (timestamp)
            WHERE pass_rate < 0.9
        """)
        
        # Create composite indexes for common joins
        await session.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_failures_composite
            ON test_failures (test_run_id, error_type, created_at)
        """)
        
        # Enable parallel query execution
        await session.execute("""
            ALTER TABLE test_metrics SET (parallel_workers = 4)
        """)
        
        await session.execute("""
            ALTER TABLE test_runs SET (parallel_workers = 4)
        """)
        
        # Set storage parameters for large tables
        await session.execute("""
            ALTER TABLE test_metrics SET (
                autovacuum_vacuum_scale_factor = 0.01,
                autovacuum_analyze_scale_factor = 0.005,
                fillfactor = 90
            )
        """)
        
        await session.execute("""
            ALTER TABLE test_runs SET (
                autovacuum_vacuum_scale_factor = 0.01,
                autovacuum_analyze_scale_factor = 0.005,
                fillfactor = 90
            )
        """)
        
        # Create hypertable for time-series data
        try:
            await session.execute("""
                CREATE EXTENSION IF NOT EXISTS timescaledb;
                
                SELECT create_hypertable(
                    'test_metrics',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """)
        except Exception as e:
            logger.warning(f"TimescaleDB extension not available: {str(e)}")
            logger.warning("Falling back to standard PostgreSQL")

async def cleanup_database() -> None:
    """Perform database cleanup and maintenance."""
    async with get_db_session() as session:
        # Remove old data
        retention_days = settings.DATA_RETENTION_DAYS
        
        await session.execute(f"""
            DELETE FROM test_metrics
            WHERE timestamp < NOW() - INTERVAL '{retention_days} days'
        """)
        
        await session.execute(f"""
            DELETE FROM test_runs
            WHERE created_at < NOW() - INTERVAL '{retention_days} days'
        """)
        
        # Vacuum analyze tables
        await session.execute("VACUUM ANALYZE test_metrics")
        await session.execute("VACUUM ANALYZE test_runs")
        await session.execute("VACUUM ANALYZE test_failures")
        
        # Reindex tables
        await session.execute("REINDEX TABLE test_metrics")
        await session.execute("REINDEX TABLE test_runs")
        await session.execute("REINDEX TABLE test_failures")
        
        # Refresh materialized views
        await session.execute("""
            REFRESH MATERIALIZED VIEW CONCURRENTLY test_metrics_daily
        """)

async def initialize_database() -> None:
    """Initialize and optimize database."""
    # Create tables
    engine = get_engine()
    async with engine.begin() as conn:
        from qagentic_analytics.models.base import Base
        await conn.run_sync(Base.metadata.create_all)
    
    # Apply optimizations
    await optimize_queries()
    
    logger.info("Database initialized and optimized")
