from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Net-Zero AI Server"
    environment: str = "local"
    redis_url: str | None = None
    data_source: str = "local"
    s3_bucket: str | None = None
    s3_prefix: str = ""
    s3_inventory_flow_key: str = "inventory_flow_5days.csv"
    s3_item_master_key: str = "item_master.csv"
    s3_order_policy_key: str = "order_policy.csv"
    store_id: str = "inha-store-001"
    llm_provider: str = "bedrock"
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "meta.llama3-2-1b-instruct-v1:0"
    elasticache_replication_group_id: str | None = None
    elasticache_cache_cluster_id: str | None = None
    aws_metrics_window_minutes: int = 5
    vector_db_path: Path = Path(".cache/vector_cache.sqlite3")
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
