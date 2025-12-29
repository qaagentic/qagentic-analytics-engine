FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip and dependencies
RUN pip install --no-cache-dir pip setuptools wheel

# Copy package files first
COPY pyproject.toml README.md ./

# Copy application code
COPY qagentic_analytics qagentic_analytics/

# Create required directories
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8083

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${API_PORT}/health || exit 1

# Install package and dependencies
RUN pip install . && \
    pip install "fastapi[all]" "uvicorn[standard]" && \
    pip install --index-url https://${GITHUB_TOKEN}@github.com/qaagentic/qagentic-common/raw/main/dist/ qagentic-common

# Run the service
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "${API_HOST}", "--port", "${API_PORT}", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
