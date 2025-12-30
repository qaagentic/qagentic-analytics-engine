FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Create start script
RUN printf '%s\n' \
    '#!/bin/bash' \
    'exec uvicorn qagentic_analytics.main:app \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8083}" \
        --workers 1 \
        --proxy-headers \
        --forwarded-allow-ips="*"' \
    > start.sh && \
    chmod +x start.sh

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8083}/health || exit 1

# Run the service
CMD ["/app/start.sh"]
