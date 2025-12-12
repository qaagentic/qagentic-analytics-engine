# QAagentic Analytics Engine

Core test analytics, metrics computation, and failure clustering service for the QAagentic platform.

## Features

- **Causal Failure Atlas**: Automatically clusters failures across test frameworks, branches, and services into causal stories
- **Root Cause Analysis**: Uses NLP and machine learning to identify patterns in test failures
- **Flake Surgeon**: Analyzes test run history to identify flaky tests and suggest optimizations
- **Metric Computation**: Calculates key test metrics (pass rate, MTTR, flake rate) across different dimensions
- **Trend Analysis**: Tracks quality metrics over time to identify degradations

## Architecture

The analytics engine is composed of several key components:

1. **Clustering Service**: Groups similar failures based on stack traces, error messages, and affected code paths
2. **Embedding Service**: Converts failure text and contexts into vector embeddings for similarity comparison
3. **Metrics Service**: Computes and exposes various test quality metrics
4. **Trend Service**: Analyzes quality metrics over time to identify degradation patterns
5. **REST API**: Provides endpoints for fetching analytics data

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database (shared with other QAagentic services)
- Access to test run data via qagentic-common

### Installation

```bash
# Clone the repository
git clone https://github.com/qaagentic/qagentic-analytics-engine.git
cd qagentic-analytics-engine

# Install dependencies
pip install -e .
```

### Configuration

Create a `.env` file in the root directory with the following variables:

```
DATABASE_URL=postgresql+asyncpg://user:password@localhost/qagentic
API_HOST=0.0.0.0
API_PORT=8082
LOG_LEVEL=INFO
ALLOW_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Running

```bash
uvicorn qagentic_analytics.main:app --host 0.0.0.0 --port 8082 --reload
```

## API Endpoints

### Failure Clustering

- `GET /api/v1/clusters`: Get all failure clusters
- `GET /api/v1/clusters/{id}`: Get details for a specific cluster
- `GET /api/v1/clusters/runs/{run_id}`: Get clusters for a specific test run

### Metrics

- `GET /api/v1/metrics/summary`: Get overall quality metrics
- `GET /api/v1/metrics/trends`: Get trending metrics over time
- `GET /api/v1/metrics/flaky`: Get flakiness analysis

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest

# Format code
black qagentic_analytics tests
isort qagentic_analytics tests
```
