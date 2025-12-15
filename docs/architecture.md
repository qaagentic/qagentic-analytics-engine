# QAagentic Analytics Engine Architecture

## Overview

The Analytics Engine is a core component of the QAagentic platform responsible for processing test results, computing metrics, identifying patterns, and providing insights about test failures. It serves as the analytical brain of the platform, enabling teams to understand test performance, identify problematic areas, and make data-driven decisions.

## Core Responsibilities

- Process and analyze test results from the Ingestion Service
- Compute test metrics and trends over time
- Cluster similar test failures for root cause analysis
- Identify flaky tests and recurring issues
- Generate insights and recommendations for test improvements
- Provide APIs for querying analytical data

## Architecture Components

### 1. API Layer

The Analytics Engine exposes a RESTful API built with FastAPI that provides endpoints for:

- **Metrics**:
  - `/api/v1/metrics/summary`: Get summary metrics across all tests
  - `/api/v1/metrics/trends`: Get metrics trends over time
  - `/api/v1/metrics/by-service`: Get metrics broken down by service
  - `/api/v1/metrics/by-branch`: Get metrics broken down by branch

- **Failure Analysis**:
  - `/api/v1/clusters/`: Get failure clusters
  - `/api/v1/clusters/{cluster_id}`: Get details for a specific failure cluster
  - `/api/v1/clusters/by-test/{test_id}`: Get clusters related to a specific test

- **Flake Detection**:
  - `/api/v1/flakes/`: Get flaky tests
  - `/api/v1/flakes/{test_id}`: Get flakiness details for a specific test

- **Health & Monitoring**:
  - `/health`: Service health status
  - `/`: Service information and status

### 2. Service Layer

The Service Layer contains the business logic for:

- **Metrics Service**: Computing and aggregating test metrics
  - Calculating pass/fail rates
  - Computing mean time to recovery (MTTR)
  - Tracking test execution trends

- **Clustering Service**: Grouping similar test failures
  - Text similarity analysis of error messages
  - Stack trace analysis
  - Temporal clustering of failures

- **Embedding Service**: Creating vector representations of failures
  - Text embedding generation
  - Dimensionality reduction
  - Similarity computation

- **Trend Service**: Analyzing trends over time
  - Time series analysis of test metrics
  - Anomaly detection
  - Regression analysis

### 3. Data Layer

The Data Layer handles database interactions using SQLAlchemy ORM with asynchronous database operations:

- **Database Models**: ORM models representing the database schema
- **Database Session Management**: Asynchronous database session handling
- **Data Access Patterns**: Repository pattern for database operations

### 4. Analysis Layer

The Analysis Layer contains the algorithms and models for advanced analytics:

- **Clustering Algorithms**: K-means, DBSCAN, hierarchical clustering
- **Text Processing**: TF-IDF, word embeddings, n-grams
- **Machine Learning Models**: Classification, regression, anomaly detection
- **Statistical Analysis**: Hypothesis testing, correlation analysis

## Database Schema

The Analytics Engine primarily reads data from the core database tables created by the Ingestion Service (test_runs, test_suites, test_cases, failures) but also maintains its own tables:

### failure_clusters

Groups of related failures.

| Column           | Type      | Description                                |
|------------------|-----------|--------------------------------------------|
| id               | UUID      | Primary key                                |
| name             | VARCHAR   | Cluster name                               |
| description      | TEXT      | Cluster description                        |
| pattern          | TEXT      | Error pattern that defines the cluster     |
| first_seen       | TIMESTAMP | First occurrence of this cluster           |
| last_seen        | TIMESTAMP | Most recent occurrence of this cluster     |
| occurrence_count | INTEGER   | Number of failures in this cluster         |
| status           | VARCHAR   | Cluster status (active, resolved)          |
| priority         | VARCHAR   | Cluster priority (high, medium, low)       |
| metadata         | JSON      | Additional metadata                        |
| created_at       | TIMESTAMP | Record creation time                       |
| updated_at       | TIMESTAMP | Record update time                         |

### failure_cluster_members

Links between failures and clusters.

| Column           | Type      | Description                                |
|------------------|-----------|--------------------------------------------|
| id               | UUID      | Primary key                                |
| cluster_id       | UUID      | Foreign key to failure_clusters            |
| failure_id       | UUID      | Foreign key to failures                    |
| similarity_score | FLOAT     | Similarity score (0-1)                     |
| created_at       | TIMESTAMP | Record creation time                       |

### flaky_tests

Tests identified as flaky.

| Column           | Type      | Description                                |
|------------------|-----------|--------------------------------------------|
| id               | UUID      | Primary key                                |
| test_case_id     | UUID      | Foreign key to test_cases                  |
| flake_rate       | FLOAT     | Flakiness rate (0-1)                       |
| first_detected   | TIMESTAMP | When flakiness was first detected          |
| last_detected    | TIMESTAMP | Most recent flaky occurrence               |
| flaky_runs_count | INTEGER   | Number of flaky runs                       |
| status           | VARCHAR   | Status (active, resolved)                  |
| metadata         | JSON      | Additional metadata                        |
| created_at       | TIMESTAMP | Record creation time                       |
| updated_at       | TIMESTAMP | Record update time                         |

### metrics_snapshots

Point-in-time snapshots of key metrics.

| Column           | Type      | Description                                |
|------------------|-----------|--------------------------------------------|
| id               | UUID      | Primary key                                |
| timestamp        | TIMESTAMP | Snapshot timestamp                         |
| period           | VARCHAR   | Period type (daily, weekly, monthly)       |
| total_runs       | INTEGER   | Total test runs                            |
| total_tests      | INTEGER   | Total test cases                           |
| passed_tests     | INTEGER   | Number of passed tests                     |
| failed_tests     | INTEGER   | Number of failed tests                     |
| skipped_tests    | INTEGER   | Number of skipped tests                    |
| flaky_tests      | INTEGER   | Number of flaky tests                      |
| pass_rate        | FLOAT     | Pass rate (0-1)                            |
| failure_rate     | FLOAT     | Failure rate (0-1)                         |
| flaky_rate       | FLOAT     | Flaky rate (0-1)                           |
| mttr_minutes     | FLOAT     | Mean time to recovery in minutes           |
| metadata         | JSON      | Additional metadata                        |
| created_at       | TIMESTAMP | Record creation time                       |

## Data Flow

1. **Metrics Computation**:
   - Test results are ingested by the Ingestion Service
   - Analytics Engine processes the results
   - Metrics are computed and stored
   - Metrics are exposed via APIs

2. **Failure Clustering**:
   - New test failures are detected
   - Failures are converted to vector representations
   - Clustering algorithms group similar failures
   - Clusters are stored and exposed via APIs

3. **Flake Detection**:
   - Test execution history is analyzed
   - Tests with inconsistent results are identified
   - Flaky tests are flagged and tracked
   - Flakiness information is exposed via APIs

4. **Trend Analysis**:
   - Metrics are tracked over time
   - Trends and patterns are identified
   - Anomalies are detected
   - Trend data is exposed via APIs

## Asynchronous Processing

The Analytics Engine uses asynchronous processing for:

- **Database Operations**: Using SQLAlchemy with asyncpg
- **Batch Processing**: Background processing of large datasets
- **Scheduled Analysis**: Periodic computation of metrics and trends

## Configuration

The service is configured via environment variables:

- **Database Settings**:
  - `QAGENTIC_DATABASE_URL`: Database connection string

- **API Settings**:
  - `api_prefix`: API prefix for all endpoints (default: "/api/v1")
  - `DEBUG`: Debug mode flag

- **Analysis Settings**:
  - `CLUSTERING_ALGORITHM`: Algorithm for failure clustering
  - `SIMILARITY_THRESHOLD`: Threshold for considering failures similar
  - `MAX_CLUSTER_SIZE`: Maximum number of failures in a cluster

- **Caching Settings**:
  - `REDIS_URL`: Redis connection string for caching
  - `CACHE_TTL_SECONDS`: Cache time-to-live in seconds

- **Logging**:
  - `LOG_LEVEL`: Logging level (default: "INFO")

## Deployment

The service is containerized using Docker and can be deployed:

- As a standalone service
- As part of the QAagentic platform using docker-compose
- In Kubernetes using the provided manifests

## Security Considerations

- **Authentication**: API endpoints require authentication
- **Authorization**: Access control based on user roles
- **Input Validation**: Validation of query parameters
- **Rate Limiting**: Protection against excessive API usage

## Error Handling

The service implements robust error handling:

- **API Errors**: Structured error responses with appropriate HTTP status codes
- **Analysis Errors**: Graceful handling of algorithm failures
- **Database Errors**: Transaction rollback on failure

## Monitoring and Logging

- **Health Endpoints**: `/health` for service health checks
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Performance Metrics**: Tracking of API response times and resource usage

## Integration with Other Services

- **Ingestion Service**: Consumes test results for analysis
- **LLM Service**: Provides failure data for root cause analysis
- **UI Portal**: Supplies data for dashboards and visualizations

## Future Enhancements

- **Advanced ML Models**: More sophisticated failure clustering
- **Predictive Analytics**: Predicting test failures before they occur
- **Recommendation Engine**: Suggesting test improvements
- **Custom Metrics**: User-defined metrics and KPIs
- **Real-time Analytics**: Stream processing of test results

## Known Issues and Limitations

- **Clustering Accuracy**: Clustering algorithm may group unrelated failures
- **Performance with Large Datasets**: Processing large volumes of test data may be slow
- **Incomplete Implementation**: Some endpoints like `/api/v1/clusters/` return internal server errors
- **Limited Historical Data**: Trend analysis requires sufficient historical data
