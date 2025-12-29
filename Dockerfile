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

# Copy and install qagentic-common
COPY ../qagentic-common /tmp/qagentic-common/
RUN cd /tmp/qagentic-common && pip install .

# Copy package files
COPY pyproject.toml README.md ./

# Install package
RUN pip install .

# Copy application code
COPY qagentic_analytics qagentic_analytics/

# Create required directories
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8083}/health || exit 1

# Run the service
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8083}", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
