"""Configuration management for the Digital Twin application."""

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    # Use PostgreSQL by default for production
    # Set DB__USE_SQLITE=true for local development without PostgreSQL
    use_sqlite: bool = Field(default=False)
    sqlite_path: str = Field(default="data/digital_twin.db")

    # PostgreSQL settings (used when use_sqlite=False)
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5433)
    postgres_db: str = Field(default="digital_twin")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="digitaltwin2024")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)

    @property
    def database_url(self) -> str:
        """Get the appropriate database URL (SQLite or PostgreSQL)."""
        if self.use_sqlite:
            return f"sqlite:///{self.sqlite_path}"
        return self.postgres_url

    @property
    def async_database_url(self) -> str:
        """Get the appropriate async database URL."""
        if self.use_sqlite:
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        return self.async_postgres_url

    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def async_postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


class ModelSettings(BaseSettings):
    """ML model settings."""

    model_type: str = Field(default="transformer")
    hidden_size: int = Field(default=128)
    num_layers: int = Field(default=2)
    num_heads: int = Field(default=8)
    dropout: float = Field(default=0.1)
    sequence_length: int = Field(default=24)
    prediction_horizons: list[int] = Field(default=[6, 12, 18, 24])
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=0.001)
    epochs: int = Field(default=100)


class LLMSettings(BaseSettings):
    """LLM configuration."""

    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3:8b")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


class Settings(BaseSettings):
    """Main application settings."""

    app_name: str = Field(default="Diabetes Digital Twin")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # Sub-settings
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Optional[Path] = Field(default=None)
    vector_dir: Optional[Path] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore"
    )

    def model_post_init(self, __context: Any) -> None:
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.vector_dir is None:
            self.vector_dir = self.data_dir / "vectors"


def load_yaml_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file with environment variable substitution."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    if not config_path.exists():
        return {}

    with open(config_path) as f:
        content = f.read()

    # Substitute environment variables: ${VAR:default}
    pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

    def replace_env(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    content = re.sub(pattern, replace_env, content)
    return yaml.safe_load(content)


# Global settings instance
settings = Settings()
