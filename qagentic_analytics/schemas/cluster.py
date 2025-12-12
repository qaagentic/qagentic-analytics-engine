"""Data models for failure clusters."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, UUID4


class ClusterStatus(str, Enum):
    """Status of a failure cluster."""

    OPEN = "open"
    CLOSED = "closed"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"


class ClusterSeverity(str, Enum):
    """Severity level of a failure cluster."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"


class FailurePattern(BaseModel):
    """Common pattern identified in a failure cluster."""

    pattern_type: str
    description: str
    confidence: float
    examples: List[str]


class FailureClusterSummary(BaseModel):
    """Summary information about a failure cluster."""

    id: UUID4
    title: str
    description: str
    status: ClusterStatus
    severity: ClusterSeverity
    first_seen: datetime
    last_seen: datetime
    failure_count: int
    affected_services: List[str]
    affected_test_count: int
    flaky: bool = False
    root_cause: Optional[str] = None


class FailureLocation(BaseModel):
    """Location information for a failure."""

    file_path: Optional[str] = None
    line_number: Optional[int] = None
    method_name: Optional[str] = None
    class_name: Optional[str] = None


class RelatedFailure(BaseModel):
    """Information about a related failure."""

    id: UUID4
    test_case_id: UUID4
    test_case_name: str
    error_message: str
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: datetime
    test_run_id: UUID4
    branch: str
    service: str
    similarity_score: float


class FailureClusterDetails(FailureClusterSummary):
    """Detailed information about a failure cluster."""

    patterns: List[FailurePattern]
    locations: List[FailureLocation]
    related_failures: List[RelatedFailure]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    similar_clusters: List[UUID4] = Field(default_factory=list)
    embedding_vector: Optional[List[float]] = None


class FailureCluster(FailureClusterDetails):
    """Internal model for a failure cluster."""

    pass
