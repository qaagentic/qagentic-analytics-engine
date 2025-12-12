FROM python:3.10-slim

WORKDIR /app

# Copy all files first
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Run the service
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "0.0.0.0", "--port", "8083", "--reload"]
