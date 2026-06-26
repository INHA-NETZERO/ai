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
    s3_inventory_flow_key: str = "inventory_flow_5y.csv"
    s3_item_master_key: str = "item_master.csv"
    s3_order_policy_key: str = "order_policy.csv"
    store_id: str = "inha-store-001"
    llm_provider: str = "local"
    local_llm_backend: str = "ollama"
    local_llm_model: str = "llama3.2:1b"
    local_hf_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    local_gguf_model_path: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    aws_region: str = "us-east-1"
    aws_profile: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    vector_db_path: Path = Path(".cache/vector_cache.sqlite3")
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    rag_knowledge_dir: Path = Path("app/data/knowledge")
    rag_top_k: int = Field(default=4, ge=1, le=8)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
