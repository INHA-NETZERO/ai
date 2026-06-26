from datetime import datetime, timedelta, timezone
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings
from app.services.aws_clients import create_aws_client


ELASTICACHE_METRICS = [
    "CacheHits",
    "CacheMisses",
    "CacheHitRate",
    "CurrConnections",
    "BytesUsedForCache",
    "EngineCPUUtilization",
]


class AwsMetricsClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = create_aws_client("cloudwatch", settings)

    def get_elasticache_metrics(self) -> dict[str, Any] | None:
        dimension = _elasticache_dimension(self.settings)
        if dimension is None:
            return None

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=self.settings.aws_metrics_window_minutes)
        metrics: dict[str, float | None] = {}
        try:
            for metric_name in ELASTICACHE_METRICS:
                response = self._client.get_metric_statistics(
                    Namespace="AWS/ElastiCache",
                    MetricName=metric_name,
                    Dimensions=[dimension],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=60,
                    Statistics=["Average", "Sum"],
                )
                metrics[metric_name] = _latest_metric_value(response.get("Datapoints", []), metric_name)
        except (BotoCoreError, ClientError):
            return {
                "enabled": True,
                "available": False,
                "dimension": dimension,
                "metrics": {},
            }

        return {
            "enabled": True,
            "available": True,
            "dimension": dimension,
            "metrics": metrics,
        }


def _elasticache_dimension(settings: Settings) -> dict[str, str] | None:
    if settings.elasticache_replication_group_id:
        return {
            "Name": "ReplicationGroupId",
            "Value": settings.elasticache_replication_group_id,
        }
    if settings.elasticache_cache_cluster_id:
        return {
            "Name": "CacheClusterId",
            "Value": settings.elasticache_cache_cluster_id,
        }
    return None


def _latest_metric_value(datapoints: list[dict[str, Any]], metric_name: str) -> float | None:
    if not datapoints:
        return None
    latest = max(datapoints, key=lambda point: point["Timestamp"])
    if metric_name in {"CacheHits", "CacheMisses"}:
        value = latest.get("Sum", latest.get("Average"))
    else:
        value = latest.get("Average", latest.get("Sum"))
    return round(float(value), 4) if value is not None else None
