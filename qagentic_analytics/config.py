"""Configuration settings for the QAagentic Analytics Engine."""

import logging
import os
from functools import lru_cache
from typing import List, Optional, Union

from pydantic import AnyHttpUrl, Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8082
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "QAagentic Analytics Engine"
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")

    # CORS Settings
    ALLOW_ORIGINS: Union[str, List[str]] = Field(default="http://localhost:3000")
    ALLOW_ORIGIN_REGEX: Optional[str] = None
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    ALLOW_HEADERS: List[str] = ["*"]

    @field_validator("ALLOW_ORIGINS", mode="before")
    @classmethod
    def validate_allow_origins(cls, v) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        if isinstance(v, list):
            return [str(origin).strip() for origin in v]
        return ["http://localhost:3000"]  # fallback

    # Database Settings
    DATABASE_URL: PostgresDsn
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_STATEMENT_TIMEOUT: int = 30000
    DB_COMMAND_TIMEOUT: int = 30
    SQL_DEBUG: bool = False

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def convert_to_async_db_url(cls, v):
        """Convert PostgreSQL URL to async driver URL."""
        if isinstance(v, str):
            # Convert postgresql:// to postgresql+asyncpg://
            if v.startswith("postgresql://"):
                return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # Analytics Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CLUSTERING_ALGORITHM: str = "hdbscan"
    CLUSTERING_MIN_CLUSTER_SIZE: int = 3
    CLUSTERING_MIN_SAMPLES: int = 1
    CLUSTERING_EPSILON: float = 0.5
    
    # Feature toggles
    ENABLE_FLAKE_DETECTION: bool = True
    ENABLE_ROOT_CAUSE_ANALYSIS: bool = True
    ENABLE_TREND_ANALYSIS: bool = True

    @field_validator("LOG_LEVEL", mode="after")
    @classmethod
    def setup_logging(cls, v):
        """Set logging level based on configuration."""
        logging.getLogger().setLevel(getattr(logging, v))
        return v
        
    def get_dsn_parameters(self):
        """Extract database connection parameters from DATABASE_URL."""
        components = self.DATABASE_URL.components
        return {
            "dbname": components.path.lstrip("/"),
            "user": components.user,
            "password": components.password,
            "host": components.host,
            "port": components.port or 5432,
        }


@lru_cache
def get_settings():
    """Get application settings."""
    return Settings()
