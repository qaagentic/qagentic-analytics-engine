"""Service for clustering similar failures."""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN
from pydantic import UUID4

from qagentic_analytics.schemas.cluster import (
    FailureCluster, 
    FailureClusterSummary, 
    FailureClusterDetails,
    RelatedFailure,
    FailurePattern,
    FailureLocation,
    ClusterStatus,
    ClusterSeverity
)
from qagentic_analytics.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class ClusteringService:
    """Service for clustering similar failures."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        algorithm: str = "hdbscan",
        min_cluster_size: int = 3,
        min_samples: int = 1,
        epsilon: float = 0.5,
    ):
        """
        Initialize the clustering service.
        
        Args:
            embedding_service: Service for creating embeddings
            algorithm: Clustering algorithm to use ('hdbscan' or 'dbscan')
            min_cluster_size: Minimum number of samples in a cluster
            min_samples: Minimum number of samples for core points
            epsilon: Maximum distance between samples in a cluster (for DBSCAN)
        """
        self.embedding_service = embedding_service
        self.algorithm = algorithm
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.epsilon = epsilon
        logger.info(f"Initializing clustering service with algorithm {algorithm}")
        
    async def cluster_failures(self, failure_embeddings: List[Tuple[Dict[str, Any], List[float]]]) -> List[List[int]]:
        """
        Cluster failure embeddings.
        
        Args:
            failure_embeddings: List of (failure_data, embedding_vector) tuples
            
        Returns:
            List of clusters, where each cluster is a list of indices into the original list
        """
        if not failure_embeddings:
            return []
            
        # Extract just the embedding vectors
        embeddings = np.array([e[1] for e in failure_embeddings])
        
        # Perform clustering
        if self.algorithm.lower() == "hdbscan":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="cosine",
            )
            clusterer.fit(embeddings)
            labels = clusterer.labels_
        else:  # Default to DBSCAN
            clusterer = DBSCAN(
                eps=self.epsilon,
                min_samples=self.min_samples,
                metric="cosine",
            )
            labels = clusterer.fit_predict(embeddings)
        
        # Group indices by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label >= 0:  # Ignore noise points (label -1)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
                
        return list(clusters.values())
    
    async def get_clusters(
        self,
        skip: int = 0,
        limit: int = 100,
        min_failures: int = 1,
        status: Optional[str] = None,
        service: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> List[FailureClusterSummary]:
        """
        Get all failure clusters, optionally filtered by parameters.
        
        This is a placeholder implementation that would be connected to a database
        in a real implementation.
        """
        # In a real implementation, we would query the database
        # For now, return some sample data
        clusters = [
            FailureClusterSummary(
                id=uuid.uuid4(),
                title=f"Database connection error in checkout service",
                description="Tests fail with 'Could not connect to database' error",
                status=ClusterStatus.OPEN,
                severity=ClusterSeverity.HIGH,
                first_seen=datetime.now().replace(day=datetime.now().day - 2),
                last_seen=datetime.now(),
                failure_count=15,
                affected_services=["checkout", "payment"],
                affected_test_count=5,
                flaky=False,
                root_cause="Database connection pool exhaustion",
            ),
            FailureClusterSummary(
                id=uuid.uuid4(),
                title=f"Authentication timeout in user service",
                description="Tests fail with 'Authentication timed out' error",
                status=ClusterStatus.INVESTIGATING,
                severity=ClusterSeverity.MEDIUM,
                first_seen=datetime.now().replace(day=datetime.now().day - 5),
                last_seen=datetime.now().replace(day=datetime.now().day - 1),
                failure_count=8,
                affected_services=["user", "auth"],
                affected_test_count=3,
                flaky=True,
                root_cause=None,
            ),
        ]
        
        # Apply filters
        if status:
            clusters = [c for c in clusters if c.status == status]
        if service:
            clusters = [c for c in clusters if service in c.affected_services]
        if branch:
            # This is just a placeholder, in a real implementation we would filter by branch
            pass
        if min_failures > 1:
            clusters = [c for c in clusters if c.failure_count >= min_failures]
            
        # Apply pagination
        return clusters[skip:skip + limit]
    
    async def get_cluster_by_id(self, cluster_id: UUID4) -> Optional[FailureClusterDetails]:
        """
        Get detailed information about a specific failure cluster.
        
        This is a placeholder implementation.
        """
        # In a real implementation, we would query the database by ID
        # For now, return a sample cluster if the ID matches a pattern
        
        # Create a deterministic sample based on the last digit of the UUID
        last_digit = int(str(cluster_id)[-1], 16) % 2
        
        if last_digit == 0:
            return FailureClusterDetails(
                id=cluster_id,
                title="Database connection error in checkout service",
                description="Tests fail with 'Could not connect to database' error",
                status=ClusterStatus.OPEN,
                severity=ClusterSeverity.HIGH,
                first_seen=datetime.now().replace(day=datetime.now().day - 2),
                last_seen=datetime.now(),
                failure_count=15,
                affected_services=["checkout", "payment"],
                affected_test_count=5,
                flaky=False,
                root_cause="Database connection pool exhaustion",
                patterns=[
                    FailurePattern(
                        pattern_type="error_message",
                        description="Could not connect to database",
                        confidence=0.95,
                        examples=[
                            "Error: Could not connect to database at checkout-db:5432",
                            "DatabaseError: Connection refused",
                        ],
                    ),
                    FailurePattern(
                        pattern_type="stack_trace",
                        description="Error occurs in DatabasePool.getConnection",
                        confidence=0.85,
                        examples=[
                            "at DatabasePool.getConnection (database.js:45)",
                            "at CheckoutRepository.save (checkout.js:78)",
                        ],
                    ),
                ],
                locations=[
                    FailureLocation(
                        file_path="src/database/pool.js",
                        line_number=45,
                        method_name="getConnection",
                        class_name="DatabasePool",
                    ),
                    FailureLocation(
                        file_path="src/checkout/repository.js",
                        line_number=78,
                        method_name="save",
                        class_name="CheckoutRepository",
                    ),
                ],
                related_failures=[
                    RelatedFailure(
                        id=uuid.uuid4(),
                        test_case_id=uuid.uuid4(),
                        test_case_name="test_checkout_with_valid_cart",
                        error_message="Error: Could not connect to database at checkout-db:5432",
                        error_type="DatabaseError",
                        stack_trace="at DatabasePool.getConnection (database.js:45)\nat CheckoutRepository.save (checkout.js:78)",
                        timestamp=datetime.now().replace(hour=datetime.now().hour - 1),
                        test_run_id=uuid.uuid4(),
                        branch="main",
                        service="checkout",
                        similarity_score=1.0,
                    ),
                    RelatedFailure(
                        id=uuid.uuid4(),
                        test_case_id=uuid.uuid4(),
                        test_case_name="test_payment_processing",
                        error_message="DatabaseError: Connection refused",
                        error_type="DatabaseError",
                        stack_trace="at DatabasePool.getConnection (database.js:45)\nat PaymentService.process (payment.js:32)",
                        timestamp=datetime.now().replace(hour=datetime.now().hour - 2),
                        test_run_id=uuid.uuid4(),
                        branch="main",
                        service="payment",
                        similarity_score=0.92,
                    ),
                ],
                metadata={
                    "affected_endpoints": ["/api/checkout", "/api/payment"],
                    "environment": "staging",
                    "database_service": "checkout-db",
                },
                created_at=datetime.now().replace(day=datetime.now().day - 2),
                updated_at=datetime.now(),
                similar_clusters=[],
                embedding_vector=None,  # We don't expose the embedding vector in the API
            )
        elif last_digit == 1:
            return FailureClusterDetails(
                id=cluster_id,
                title="Authentication timeout in user service",
                description="Tests fail with 'Authentication timed out' error",
                status=ClusterStatus.INVESTIGATING,
                severity=ClusterSeverity.MEDIUM,
                first_seen=datetime.now().replace(day=datetime.now().day - 5),
                last_seen=datetime.now().replace(day=datetime.now().day - 1),
                failure_count=8,
                affected_services=["user", "auth"],
                affected_test_count=3,
                flaky=True,
                root_cause=None,
                patterns=[
                    FailurePattern(
                        pattern_type="error_message",
                        description="Authentication timeout",
                        confidence=0.9,
                        examples=[
                            "Error: Authentication timed out after 5000ms",
                            "TimeoutError: Authentication service unavailable",
                        ],
                    ),
                ],
                locations=[
                    FailureLocation(
                        file_path="src/auth/client.js",
                        line_number=67,
                        method_name="authenticate",
                        class_name="AuthClient",
                    ),
                ],
                related_failures=[
                    RelatedFailure(
                        id=uuid.uuid4(),
                        test_case_id=uuid.uuid4(),
                        test_case_name="test_user_login",
                        error_message="Error: Authentication timed out after 5000ms",
                        error_type="TimeoutError",
                        stack_trace="at AuthClient.authenticate (auth/client.js:67)\nat UserService.login (user/service.js:42)",
                        timestamp=datetime.now().replace(day=datetime.now().day - 1),
                        test_run_id=uuid.uuid4(),
                        branch="main",
                        service="user",
                        similarity_score=1.0,
                    ),
                ],
                metadata={
                    "affected_endpoints": ["/api/login", "/api/user"],
                    "environment": "staging",
                    "auth_service": "auth-service",
                },
                created_at=datetime.now().replace(day=datetime.now().day - 5),
                updated_at=datetime.now().replace(day=datetime.now().day - 1),
                similar_clusters=[],
                embedding_vector=None,
            )
        return None
    
    async def get_clusters_by_run_id(self, run_id: UUID4) -> List[FailureClusterSummary]:
        """
        Get all failure clusters associated with a specific test run.
        
        This is a placeholder implementation.
        """
        # In a real implementation, we would query the database by run_id
        # For now, return sample data based on the last digit of the UUID
        last_digit = int(str(run_id)[-1], 16) % 3
        
        if last_digit == 0:
            # No clusters for this run
            return []
        elif last_digit == 1:
            # One cluster
            return [
                FailureClusterSummary(
                    id=uuid.uuid4(),
                    title="Database connection error in checkout service",
                    description="Tests fail with 'Could not connect to database' error",
                    status=ClusterStatus.OPEN,
                    severity=ClusterSeverity.HIGH,
                    first_seen=datetime.now().replace(day=datetime.now().day - 2),
                    last_seen=datetime.now(),
                    failure_count=15,
                    affected_services=["checkout", "payment"],
                    affected_test_count=5,
                    flaky=False,
                    root_cause="Database connection pool exhaustion",
                ),
            ]
        else:
            # Two clusters
            return [
                FailureClusterSummary(
                    id=uuid.uuid4(),
                    title="Database connection error in checkout service",
                    description="Tests fail with 'Could not connect to database' error",
                    status=ClusterStatus.OPEN,
                    severity=ClusterSeverity.HIGH,
                    first_seen=datetime.now().replace(day=datetime.now().day - 2),
                    last_seen=datetime.now(),
                    failure_count=15,
                    affected_services=["checkout", "payment"],
                    affected_test_count=5,
                    flaky=False,
                    root_cause="Database connection pool exhaustion",
                ),
                FailureClusterSummary(
                    id=uuid.uuid4(),
                    title="Authentication timeout in user service",
                    description="Tests fail with 'Authentication timed out' error",
                    status=ClusterStatus.INVESTIGATING,
                    severity=ClusterSeverity.MEDIUM,
                    first_seen=datetime.now().replace(day=datetime.now().day - 5),
                    last_seen=datetime.now().replace(day=datetime.now().day - 1),
                    failure_count=8,
                    affected_services=["user", "auth"],
                    affected_test_count=3,
                    flaky=True,
                    root_cause=None,
                ),
            ]
