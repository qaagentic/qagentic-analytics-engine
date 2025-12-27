"""Enhanced clustering service for test failures."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

from qagentic_analytics.db import get_db_session
from qagentic_analytics.models.failure import TestFailure
from qagentic_analytics.models.cluster import FailureCluster
from qagentic_analytics.utils.text import preprocess_text
from qagentic_analytics.utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)

class ClusteringService:
    """Service for clustering similar test failures."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )

    async def cluster_failures(
        self,
        time_window: timedelta = timedelta(days=7),
        min_failures: int = 2,
        similarity_threshold: float = 0.8
    ) -> List[FailureCluster]:
        """
        Cluster similar test failures using enhanced algorithms.
        
        Args:
            time_window: Time window to consider for clustering
            min_failures: Minimum failures to form a cluster
            similarity_threshold: Threshold for considering failures similar
            
        Returns:
            List of failure clusters
        """
        async with get_db_session() as session:
            # Get recent failures
            failures = await self._get_recent_failures(session, time_window)
            
            if not failures:
                return []

            # Extract features for clustering
            feature_matrix = await self._extract_features(failures)
            
            # Perform clustering
            clusters = await self._perform_clustering(
                feature_matrix,
                failures,
                min_failures,
                similarity_threshold
            )
            
            # Save clusters to database
            await self._save_clusters(session, clusters)
            
            return clusters

    async def _get_recent_failures(
        self,
        session: AsyncSession,
        time_window: timedelta
    ) -> List[TestFailure]:
        """Get test failures within the specified time window."""
        cutoff_time = datetime.utcnow() - time_window
        
        # Query failures with error messages and stack traces
        failures = await session.query(TestFailure).filter(
            TestFailure.created_at >= cutoff_time,
            TestFailure.error_message.isnot(None)
        ).all()
        
        return failures

    async def _extract_features(self, failures: List[TestFailure]) -> np.ndarray:
        """Extract features from failures for clustering."""
        # Combine error messages and stack traces
        texts = [
            f"{f.error_message} {f.stack_trace or ''}" 
            for f in failures
        ]
        
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Get text embeddings
        embeddings = await get_embeddings(processed_texts)
        
        # Combine with TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        
        # Concatenate embeddings and TF-IDF features
        combined_features = np.hstack([
            embeddings,
            tfidf_features.toarray()
        ])
        
        return combined_features

    async def _perform_clustering(
        self,
        features: np.ndarray,
        failures: List[TestFailure],
        min_failures: int,
        similarity_threshold: float
    ) -> List[FailureCluster]:
        """Perform clustering on the feature matrix."""
        # Use DBSCAN for clustering
        eps = 1 - similarity_threshold  # Convert similarity to distance
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_failures,
            metric='cosine'
        )
        
        cluster_labels = clusterer.fit_predict(features)
        
        # Group failures by cluster
        clusters: Dict[int, List[TestFailure]] = {}
        for label, failure in zip(cluster_labels, failures):
            if label != -1:  # Ignore noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(failure)
        
        # Create FailureCluster objects
        failure_clusters = []
        for label, cluster_failures in clusters.items():
            cluster = await self._create_cluster(cluster_failures)
            failure_clusters.append(cluster)
        
        return failure_clusters

    async def _create_cluster(
        self,
        failures: List[TestFailure]
    ) -> FailureCluster:
        """Create a FailureCluster from a group of failures."""
        # Extract common patterns
        error_messages = [f.error_message for f in failures]
        stack_traces = [f.stack_trace for f in failures if f.stack_trace]
        
        # Find common error pattern
        common_pattern = self._extract_common_pattern(error_messages)
        
        # Find common stack trace elements
        common_trace = self._extract_common_trace(stack_traces) if stack_traces else None
        
        # Create cluster
        return FailureCluster(
            title=common_pattern[:100],  # Truncate for title
            pattern=common_pattern,
            stack_trace_pattern=common_trace,
            failure_count=len(failures),
            first_seen=min(f.created_at for f in failures),
            last_seen=max(f.created_at for f in failures),
            failures=failures
        )

    def _extract_common_pattern(self, texts: List[str]) -> str:
        """Extract common pattern from error messages."""
        if not texts:
            return ""
            
        # Use longest common subsequence algorithm
        pattern = texts[0]
        for text in texts[1:]:
            pattern = self._get_lcs(pattern, text)
            
        return pattern.strip()

    def _extract_common_trace(self, traces: List[str]) -> Optional[str]:
        """Extract common elements from stack traces."""
        if not traces:
            return None
            
        # Split into lines and find common lines
        trace_lines = [t.split('\n') for t in traces]
        common_lines = set(trace_lines[0])
        
        for trace in trace_lines[1:]:
            common_lines &= set(trace)
            
        return '\n'.join(sorted(common_lines)) if common_lines else None

    def _get_lcs(self, str1: str, str2: str) -> str:
        """Get longest common subsequence of two strings."""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill dp table
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if str1[i] == str2[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        
        # Reconstruct LCS
        lcs = []
        i = j = 0
        while i < m and j < n:
            if str1[i] == str2[j]:
                lcs.append(str1[i])
                i += 1
                j += 1
            elif dp[i + 1][j] >= dp[i][j + 1]:
                i += 1
            else:
                j += 1
                
        return ''.join(lcs)

    async def _save_clusters(
        self,
        session: AsyncSession,
        clusters: List[FailureCluster]
    ) -> None:
        """Save clusters to database."""
        for cluster in clusters:
            session.add(cluster)
        await session.commit()
