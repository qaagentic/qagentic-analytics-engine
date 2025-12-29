#!/bin/sh
PORT="${PORT:-8083}"
exec uvicorn qagentic_analytics.main:app --host 0.0.0.0 --port "$PORT" --workers 1
