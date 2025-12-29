FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY setup.py setup.py
COPY README.md README.md
COPY qagentic_analytics qagentic_analytics

# Install dependencies
RUN pip install --no-cache-dir -e .

# Copy rest of the code
COPY . .

# Run the service in production mode
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "0.0.0.0", "--port", "8083", "--workers", "4"]
