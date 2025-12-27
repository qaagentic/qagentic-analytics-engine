"""Real-time data streaming service."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable
from datetime import datetime
import json

from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession
import aioredis

from qagentic_analytics.db import get_db_session
from qagentic_analytics.models.test_run import TestRun
from qagentic_analytics.models.metrics import TestMetrics
from qagentic_analytics.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class StreamingService:
    """Service for real-time data streaming."""

    def __init__(self):
        self.redis = aioredis.from_url(settings.REDIS_URL)
        self.subscribers: Dict[str, Set[WebSocket]] = {}
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}
        self._stream_tasks: Dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        """Start the streaming service."""
        # Initialize Redis pub/sub
        self.pubsub = self.redis.pubsub()
        
        # Register default handlers
        self.register_handler("test_run", self._handle_test_run)
        self.register_handler("metrics", self._handle_metrics)
        
        # Start background tasks
        self._stream_tasks["metrics"] = asyncio.create_task(
            self._stream_metrics()
        )
        
        logger.info("Streaming service started")

    async def stop(self) -> None:
        """Stop the streaming service."""
        # Cancel all background tasks
        for task in self._stream_tasks.values():
            task.cancel()
            
        try:
            await asyncio.gather(*self._stream_tasks.values())
        except asyncio.CancelledError:
            pass
            
        # Close Redis connection
        await self.redis.close()
        
        logger.info("Streaming service stopped")

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register a handler for an event type."""
        self.handlers[event_type] = handler
        logger.info(f"Registered handler for {event_type}")

    async def subscribe(
        self,
        websocket: WebSocket,
        event_types: List[str]
    ) -> None:
        """Subscribe a WebSocket to event types."""
        await websocket.accept()
        
        # Add to subscribers
        for event_type in event_types:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = set()
            self.subscribers[event_type].add(websocket)
            
        try:
            # Keep connection alive and handle messages
            while True:
                data = await websocket.receive_text()
                await self._handle_websocket_message(websocket, data)
                
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            
        finally:
            # Clean up on disconnect
            for event_type in event_types:
                if event_type in self.subscribers:
                    self.subscribers[event_type].remove(websocket)
                    if not self.subscribers[event_type]:
                        del self.subscribers[event_type]
            await websocket.close()

    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Publish an event to Redis."""
        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis.publish(
            f"qagentic:{event_type}",
            json.dumps(message)
        )

    async def _handle_websocket_message(
        self,
        websocket: WebSocket,
        data: str
    ) -> None:
        """Handle incoming WebSocket messages."""
        try:
            message = json.loads(data)
            message_type = message.get("type")
            
            if message_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid message format: {data}")
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")

    async def _stream_metrics(self) -> None:
        """Stream real-time metrics."""
        while True:
            try:
                async with get_db_session() as session:
                    # Get latest metrics
                    metrics = await self._get_latest_metrics(session)
                    
                    # Publish to subscribers
                    await self._broadcast_metrics(metrics)
                    
                # Wait before next update
                await asyncio.sleep(5)  # 5 second interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error streaming metrics: {str(e)}")
                await asyncio.sleep(5)  # Wait before retry

    async def _get_latest_metrics(
        self,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Get latest metrics efficiently."""
        # Get most recent test run
        latest_run = await session.query(TestRun).order_by(
            TestRun.created_at.desc()
        ).first()
        
        if not latest_run:
            return {}
            
        # Get recent metrics
        recent_metrics = await session.query(TestMetrics).filter(
            TestMetrics.test_run_id == latest_run.id
        ).all()
        
        # Compute real-time metrics
        return {
            "test_run_id": latest_run.id,
            "status": latest_run.status,
            "total_tests": latest_run.total_tests,
            "passed_tests": latest_run.passed_tests,
            "failed_tests": latest_run.failed_tests,
            "running_tests": latest_run.running_tests,
            "duration": latest_run.duration,
            "metrics": [metric.to_dict() for metric in recent_metrics]
        }

    async def _broadcast_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Broadcast metrics to subscribers."""
        if not metrics:
            return
            
        message = {
            "type": "metrics",
            "data": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if "metrics" in self.subscribers:
            dead_sockets = set()
            
            for websocket in self.subscribers["metrics"]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception:
                    dead_sockets.add(websocket)
                    
            # Clean up dead connections
            for websocket in dead_sockets:
                self.subscribers["metrics"].remove(websocket)

    async def _handle_test_run(
        self,
        data: Dict[str, Any]
    ) -> None:
        """Handle test run updates."""
        if "test_run" not in self.subscribers:
            return
            
        message = {
            "type": "test_run",
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        dead_sockets = set()
        
        for websocket in self.subscribers["test_run"]:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception:
                dead_sockets.add(websocket)
                
        # Clean up dead connections
        for websocket in dead_sockets:
            self.subscribers["test_run"].remove(websocket)

    async def _handle_metrics(
        self,
        data: Dict[str, Any]
    ) -> None:
        """Handle metrics updates."""
        await self._broadcast_metrics(data)
