FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all files first
COPY . .

# Install pip and dependencies
RUN pip install --no-cache-dir pip setuptools wheel
RUN pip install --no-cache-dir -e .

# Run the service in production mode
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8083}", "--workers", "4"]
