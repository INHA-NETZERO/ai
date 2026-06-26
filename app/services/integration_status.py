import os
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.demand_model import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, FEATURE_NAMES


def build_integration_status(
    settings: Settings,
    exact_cache_backend: str,
    semantic_cache_backend: str,
) -> dict[str, Any]:
    aws_credentials = _aws_credentials_status(settings)
    model = _model_status()
    s3_configured = bool(settings.s3_bucket)
    cloudwatch_configured = bool(settings.elasticache_replication_group_id or settings.elasticache_cache_cluster_id)
    gaps = _integration_gaps(
        settings=settings,
        aws_credentials_configured=aws_credentials["configured"],
        exact_cache_backend=exact_cache_backend,
        s3_configured=s3_configured,
        cloudwatch_configured=cloudwatch_configured,
        model_loaded=model["available"],
    )
    return {
        "environment": settings.environment,
        "aws": aws_credentials,
        "llm": {
            "provider": settings.llm_provider,
            "bedrock_model_id": settings.bedrock_model_id,
            "credentials_configured": aws_credentials["configured"],
            "actual_bedrock_call_ready": settings.llm_provider == "bedrock" and aws_credentials["configured"],
            "readiness_note": "This is credential configuration readiness. Run scripts/check_bedrock.py to verify real Bedrock permission and model access.",
            "fallback_when_unavailable": True,
        },
        "data_source": {
            "active": settings.data_source,
            "v1_presigned_url_loader_implemented": True,
            "v1_presigned_urls_require_aws_credentials": False,
            "local_csv_active": settings.data_source == "local",
            "s3_active": settings.data_source == "s3",
            "s3_configured": s3_configured,
            "s3_bucket_configured": bool(settings.s3_bucket),
            "s3_prefix": settings.s3_prefix,
            "s3_keys": {
                "inventory_flow": settings.s3_inventory_flow_key,
                "item_master": settings.s3_item_master_key,
                "order_policy": settings.s3_order_policy_key,
            },
            "s3_loader_implemented": True,
            "s3_requires_aws_credentials": settings.data_source == "s3",
        },
        "cache": {
            "exact_cache_backend": exact_cache_backend,
            "semantic_cache_backend": semantic_cache_backend,
            "elasticache_redis_active": exact_cache_backend == "elasticache_redis",
            "cloudwatch_elasticache_metrics_configured": cloudwatch_configured,
            "cloudwatch_metrics_require_aws_credentials": cloudwatch_configured,
        },
        "model": model,
        "gaps": gaps,
    }


def _aws_credentials_status(settings: Settings) -> dict[str, Any]:
    settings_key_pair = bool(settings.aws_access_key_id and settings.aws_secret_access_key)
    settings_profile = bool(settings.aws_profile)
    env_key_pair = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
    profile = bool(os.getenv("AWS_PROFILE"))
    web_identity = bool(os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE") and os.getenv("AWS_ROLE_ARN"))
    container_role = bool(os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") or os.getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI"))
    shared_files = _aws_shared_files_exist()
    configured = settings_key_pair or settings_profile or env_key_pair or profile or web_identity or container_role or shared_files
    sources = [
        name
        for name, enabled in {
            "settings_env_file_key_pair": settings_key_pair,
            "settings_env_file_profile": settings_profile,
            "env_key_pair": env_key_pair,
            "aws_profile": profile,
            "web_identity": web_identity,
            "container_role": container_role,
            "shared_aws_files": shared_files,
        }.items()
        if enabled
    ]
    return {
        "configured": configured,
        "detected_sources": sources,
        "note": "This checks configured credential sources only, not actual IAM permission or Bedrock model access.",
    }


def _aws_shared_files_exist() -> bool:
    aws_dir = Path.home() / ".aws"
    try:
        return (aws_dir / "credentials").exists() or (aws_dir / "config").exists()
    except OSError:
        return False


def _model_status() -> dict[str, Any]:
    metadata: dict[str, Any] | None = None
    schema_matches = False
    if DEFAULT_METADATA_PATH.exists():
        try:
            import json

            metadata = json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
            schema_matches = metadata.get("feature_names") == FEATURE_NAMES
        except Exception:
            metadata = None
    return {
        "available": DEFAULT_MODEL_PATH.exists() and DEFAULT_METADATA_PATH.exists() and schema_matches,
        "model_path": str(DEFAULT_MODEL_PATH),
        "metadata_path": str(DEFAULT_METADATA_PATH),
        "feature_schema_matches": schema_matches,
        "evaluation": metadata.get("evaluation") if metadata else None,
    }


def _integration_gaps(
    settings: Settings,
    aws_credentials_configured: bool,
    exact_cache_backend: str,
    s3_configured: bool,
    cloudwatch_configured: bool,
    model_loaded: bool,
) -> list[str]:
    gaps = []
    if settings.llm_provider == "bedrock" and not aws_credentials_configured:
        gaps.append("AWS credentials are not configured, so Bedrock Llama calls will fall back to deterministic text.")
    if settings.data_source == "local":
        gaps.append("API endpoints currently read local app/data CSV files. Set DATA_SOURCE=s3 to read S3 CSV files.")
    if settings.data_source == "s3" and not s3_configured:
        gaps.append("DATA_SOURCE=s3 is active but S3_BUCKET is not configured.")
    if settings.data_source == "s3" and not aws_credentials_configured:
        gaps.append("DATA_SOURCE=s3 is active but AWS credentials are not configured.")
    if exact_cache_backend == "memory":
        gaps.append("Exact cache is using process memory; Redis/ElastiCache is not active.")
    if cloudwatch_configured and not aws_credentials_configured:
        gaps.append("ElastiCache CloudWatch metrics are configured but cannot be fetched without AWS credentials.")
    if not model_loaded:
        gaps.append("Saved LightGBM model is unavailable or metadata schema does not match; forecast falls back locally.")
    return gaps
