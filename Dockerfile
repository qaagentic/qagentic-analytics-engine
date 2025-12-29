FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip and dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Configure git to use token
RUN git config --global credential.helper store

# Install base dependencies first
RUN pip install "fastapi[all]" "uvicorn[standard]" \
    numpy==1.24.3 \
    pandas>=2.1.1 \
    scikit-learn>=1.3.2 \
    sqlalchemy>=2.0.22 \
    asyncpg>=0.28.0 \
    httpx>=0.24.1 \
    python-dotenv>=1.0.0 \
    hdbscan>=0.8.33 \
    sentence-transformers>=2.2.2 \
    nltk>=3.8.1 \
    spacy>=3.7.1

# Install qagentic-common from GitHub using token
ARG GITHUB_TOKEN
RUN echo "https://oauth2:${GITHUB_TOKEN}@github.com" > ~/.git-credentials && \
    pip install git+https://github.com/qaagentic/qagentic-common.git

# Copy package files
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

# Install the package itself
RUN pip install .

# Run the service
CMD ["uvicorn", "qagentic_analytics.main:app", "--host", "${API_HOST}", "--port", "${API_PORT}", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
