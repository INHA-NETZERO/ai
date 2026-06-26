import hashlib
import json
import time
from collections.abc import Callable
from typing import Any

import redis


JsonValue = dict[str, Any]


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


class ExactCache:
    def __init__(self, redis_url: str | None, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._memory: dict[str, tuple[float, JsonValue]] = {}
        self._client: redis.Redis | None = None
        if redis_url:
            try:
                client = redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._client = client
            except redis.RedisError:
                self._client = None

    def get(self, key: str) -> JsonValue | None:
        if self._client:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None

        record = self._memory.get(key)
        if not record:
            return None
        expires_at, value = record
        if expires_at < time.time():
            self._memory.pop(key, None)
            return None
        return value

    def set(self, key: str, value: JsonValue) -> None:
        if self._client:
            self._client.setex(key, self.ttl_seconds, stable_json(value))
            return
        self._memory[key] = (time.time() + self.ttl_seconds, value)


def get_or_set(cache: ExactCache, key: str, factory: Callable[[], JsonValue]) -> tuple[JsonValue, bool]:
    cached = cache.get(key)
    if cached is not None:
        return cached, True
    value = factory()
    cache.set(key, value)
    return value, False
