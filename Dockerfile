FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first to leverage Docker cache
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY qagentic_analytics qagentic_analytics

# Install pip and dependencies
RUN pip install --no-cache-dir pip setuptools wheel
RUN pip install --no-cache-dir .[dev]

# Copy rest of the code
COPY . .

# Run the service in production mode
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8083}", "--workers", "4"]
