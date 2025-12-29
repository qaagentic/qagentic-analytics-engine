FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip and dependencies
RUN pip install --no-cache-dir pip setuptools wheel

# Copy requirements first
COPY pyproject.toml .
COPY README.md .

# Install direct dependencies
RUN pip install --no-cache-dir \
    python-dotenv>=1.0.0 \
    fastapi>=0.104.0 \
    sqlalchemy>=2.0.22 \
    numpy>=1.26.0 \
    hdbscan>=0.8.33 \
    pandas>=2.1.1 \
    uvicorn>=0.24.0 \
    psycopg2-binary>=2.9.9 \
    httpx>=0.25.0 \
    python-multipart>=0.0.6 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0

# Copy application code
COPY qagentic_analytics qagentic_analytics/

# Create required directories
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8083}/health || exit 1

# Run the service in production mode
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8083}", "--workers", "1"]
