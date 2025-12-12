FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Install qagentic-common from mounted volume at runtime
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "0.0.0.0", "--port", "8083", "--reload"]
