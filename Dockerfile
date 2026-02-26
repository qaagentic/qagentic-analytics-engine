FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy service files
COPY . /app/

# Install qagentic-common first
RUN pip install --no-cache-dir -e /app/qagentic-common

# Install service dependencies
RUN pip install --no-cache-dir -e /app/qagentic-analytics-engine

# Run the service
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "0.0.0.0", "--port", "8083", "--reload"]
