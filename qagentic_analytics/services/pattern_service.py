"""Pattern detection service for identifying test failure patterns."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from scipy import stats

from qagentic_analytics.db import get_db_session
from qagentic_analytics.models.failure import TestFailure
from qagentic_analytics.models.pattern import FailurePattern
from qagentic_analytics.utils.time_series import create_time_series

logger = logging.getLogger(__name__)

class PatternService:
    """Service for detecting patterns in test failures."""

    async def detect_patterns(
        self,
        time_window: timedelta = timedelta(days=30),
        min_occurrences: int = 3,
        confidence_threshold: float = 0.95
    ) -> List[FailurePattern]:
        """
        Detect patterns in test failures.
        
        Args:
            time_window: Time window to analyze
            min_occurrences: Minimum occurrences to consider a pattern
            confidence_threshold: Statistical confidence threshold
            
        Returns:
            List of detected failure patterns
        """
        async with get_db_session() as session:
            # Get failure data
            failures = await self._get_failure_data(session, time_window)
            
            if not failures:
                return []
                
            # Detect temporal patterns
            temporal_patterns = await self._detect_temporal_patterns(
                failures,
                min_occurrences,
                confidence_threshold
            )
            
            # Detect environmental patterns
            env_patterns = await self._detect_environmental_patterns(
                failures,
                min_occurrences,
                confidence_threshold
            )
            
            # Detect code change patterns
            code_patterns = await self._detect_code_patterns(
                failures,
                min_occurrences,
                confidence_threshold
            )
            
            # Combine and rank patterns
            all_patterns = temporal_patterns + env_patterns + code_patterns
            ranked_patterns = self._rank_patterns(all_patterns)
            
            # Save patterns
            await self._save_patterns(session, ranked_patterns)
            
            return ranked_patterns

    async def _get_failure_data(
        self,
        session: AsyncSession,
        time_window: timedelta
    ) -> List[Dict[str, Any]]:
        """Get failure data with context."""
        cutoff_time = datetime.utcnow() - time_window
        
        # Query failures with related data
        failures = await session.query(TestFailure).filter(
            TestFailure.created_at >= cutoff_time
        ).all()
        
        # Enrich with context
        enriched_failures = []
        for failure in failures:
            failure_data = {
                "id": failure.id,
                "test_name": failure.test_name,
                "error_message": failure.error_message,
                "created_at": failure.created_at,
                "environment": failure.environment,
                "commit_hash": failure.commit_hash,
                "branch": failure.branch,
                "duration": failure.duration
            }
            enriched_failures.append(failure_data)
            
        return enriched_failures

    async def _detect_temporal_patterns(
        self,
        failures: List[Dict[str, Any]],
        min_occurrences: int,
        confidence_threshold: float
    ) -> List[FailurePattern]:
        """Detect temporal patterns in failures."""
        patterns = []
        
        # Create time series
        time_series = create_time_series(failures)
        
        # Check for periodicity
        for test_name, series in time_series.items():
            # Perform FFT analysis
            frequencies = np.fft.fft(series)
            magnitudes = np.abs(frequencies)
            
            # Find significant frequencies
            threshold = np.mean(magnitudes) + 2 * np.std(magnitudes)
            significant_freq = np.where(magnitudes > threshold)[0]
            
            if len(significant_freq) >= min_occurrences:
                # Calculate confidence
                confidence = stats.norm.cdf(
                    len(significant_freq),
                    loc=min_occurrences,
                    scale=1
                )
                
                if confidence >= confidence_threshold:
                    pattern = FailurePattern(
                        type="temporal",
                        test_name=test_name,
                        confidence=confidence,
                        description=f"Periodic failure pattern detected for {test_name}",
                        metadata={
                            "frequency": float(np.mean(significant_freq)),
                            "magnitude": float(np.mean(magnitudes[significant_freq]))
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns

    async def _detect_environmental_patterns(
        self,
        failures: List[Dict[str, Any]],
        min_occurrences: int,
        confidence_threshold: float
    ) -> List[FailurePattern]:
        """Detect patterns related to environment."""
        patterns = []
        
        # Group failures by environment
        env_failures: Dict[str, List[Dict[str, Any]]] = {}
        for failure in failures:
            env = failure["environment"]
            if env not in env_failures:
                env_failures[env] = []
            env_failures[env].append(failure)
            
        # Analyze each environment
        for env, env_fails in env_failures.items():
            if len(env_fails) >= min_occurrences:
                # Calculate failure rate
                total_tests = await self._get_total_tests(env)
                failure_rate = len(env_fails) / total_tests
                
                # Compare with baseline
                baseline_rate = await self._get_baseline_failure_rate()
                
                if failure_rate > baseline_rate * 1.5:  # 50% higher than baseline
                    confidence = min(
                        1.0,
                        (failure_rate - baseline_rate) / baseline_rate
                    )
                    
                    if confidence >= confidence_threshold:
                        pattern = FailurePattern(
                            type="environmental",
                            test_name="*",  # Affects all tests
                            confidence=confidence,
                            description=f"Higher failure rate in environment: {env}",
                            metadata={
                                "environment": env,
                                "failure_rate": failure_rate,
                                "baseline_rate": baseline_rate
                            }
                        )
                        patterns.append(pattern)
                        
        return patterns

    async def _detect_code_patterns(
        self,
        failures: List[Dict[str, Any]],
        min_occurrences: int,
        confidence_threshold: float
    ) -> List[FailurePattern]:
        """Detect patterns related to code changes."""
        patterns = []
        
        # Group failures by commit/branch
        commit_failures: Dict[str, List[Dict[str, Any]]] = {}
        for failure in failures:
            commit = failure["commit_hash"]
            if commit not in commit_failures:
                commit_failures[commit] = []
            commit_failures[commit].append(failure)
            
        # Analyze each commit
        for commit, commit_fails in commit_failures.items():
            if len(commit_fails) >= min_occurrences:
                # Get changed files for commit
                changed_files = await self._get_changed_files(commit)
                
                # Look for common patterns in changed files
                file_patterns = self._analyze_file_patterns(changed_files)
                
                for file_pattern in file_patterns:
                    confidence = file_pattern["confidence"]
                    
                    if confidence >= confidence_threshold:
                        pattern = FailurePattern(
                            type="code_change",
                            test_name=file_pattern["test_name"],
                            confidence=confidence,
                            description=file_pattern["description"],
                            metadata={
                                "commit_hash": commit,
                                "files": file_pattern["files"],
                                "change_type": file_pattern["change_type"]
                            }
                        )
                        patterns.append(pattern)
                        
        return patterns

    def _rank_patterns(
        self,
        patterns: List[FailurePattern]
    ) -> List[FailurePattern]:
        """Rank patterns by confidence and impact."""
        # Calculate impact score for each pattern
        for pattern in patterns:
            impact_score = self._calculate_impact(pattern)
            pattern.metadata["impact_score"] = impact_score
            
        # Sort by confidence and impact
        ranked = sorted(
            patterns,
            key=lambda p: (p.confidence, p.metadata["impact_score"]),
            reverse=True
        )
        
        return ranked

    def _calculate_impact(self, pattern: FailurePattern) -> float:
        """Calculate impact score for a pattern."""
        base_score = pattern.confidence
        
        # Adjust based on pattern type
        type_multiplier = {
            "temporal": 1.0,
            "environmental": 1.2,  # Environment issues often more severe
            "code_change": 1.1  # Code issues important but can be fixed
        }
        
        # Adjust based on scope
        scope_multiplier = 1.0
        if pattern.test_name == "*":  # Affects all tests
            scope_multiplier = 1.5
            
        return base_score * type_multiplier.get(pattern.type, 1.0) * scope_multiplier

    async def _get_total_tests(self, environment: str) -> int:
        """Get total number of tests run in an environment."""
        # TODO: Implement actual query
        return 1000  # Placeholder

    async def _get_baseline_failure_rate(self) -> float:
        """Get baseline failure rate across all environments."""
        # TODO: Implement actual calculation
        return 0.05  # Placeholder

    async def _get_changed_files(self, commit_hash: str) -> List[Dict[str, Any]]:
        """Get files changed in a commit."""
        # TODO: Implement VCS integration
        return []  # Placeholder

    def _analyze_file_patterns(
        self,
        changed_files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in changed files."""
        # TODO: Implement pattern analysis
        return []  # Placeholder

    async def _save_patterns(
        self,
        session: AsyncSession,
        patterns: List[FailurePattern]
    ) -> None:
        """Save patterns to database."""
        for pattern in patterns:
            session.add(pattern)
        await session.commit()
