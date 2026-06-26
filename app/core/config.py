from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Net-Zero AI Server"
    environment: str = "local"
    redis_url: str | None = None
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "meta.llama3-2-1b-instruct-v1:0"
    vector_db_path: Path = Path(".cache/vector_cache.sqlite3")
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
