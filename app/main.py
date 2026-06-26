import hashlib
from contextlib import asynccontextmanager
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import FastAPI
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.engines.deterministic import forecast_demand, recommend_orders
from app.schemas import (
    CacheInfo,
    ForecastRequest,
    ForecastResponse,
    OrderRecommendationRequest,
    OrderRecommendationResponse,
)
from app.services.cache import ExactCache, cache_key, stable_json
from app.services.embedding import HashEmbeddingService
from app.services.llm import BedrockLlamaClient
from app.services.vector_store import SemanticCacheRecord, SQLiteVectorStore

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.exact_cache = ExactCache(settings.redis_url)
        self.embedding_service = HashEmbeddingService()
        self.vector_store = SQLiteVectorStore(settings.vector_db_path, self.embedding_service.dimensions)
        self.llm_client = _build_bedrock_client(settings)


def _build_bedrock_client(settings: Settings) -> BedrockLlamaClient | None:
    try:
        return BedrockLlamaClient(settings.aws_region, settings.bedrock_model_id)
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = AppState(get_settings())
    yield


app = FastAPI(title=get_settings().app_name, version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "bedrock_model_id": settings.bedrock_model_id}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    return _cached_response("forecast", request, ForecastResponse, lambda: forecast_demand(request))


@app.post("/order-recommendation", response_model=OrderRecommendationResponse)
def order_recommendation(request: OrderRecommendationRequest) -> OrderRecommendationResponse:
    return _cached_response(
        "order-recommendation",
        request,
        OrderRecommendationResponse,
        lambda: recommend_orders(request),
    )


def _cached_response(
    endpoint: str,
    request: BaseModel,
    response_model: type[ResponseT],
    factory: Callable[[], ResponseT],
) -> ResponseT:
    state: AppState = app.state.runtime
    payload = request.model_dump()
    exact_key = cache_key(endpoint, payload)
    vector_endpoint = _vector_endpoint(endpoint, payload)

    exact = state.exact_cache.get(exact_key)
    if exact is not None:
        exact["cache"] = CacheInfo(exact_hit=True).model_dump()
        return response_model.model_validate(exact)

    embedding = state.embedding_service.embed(stable_json(payload))
    semantic = state.vector_store.search(
        endpoint=vector_endpoint,
        embedding=embedding,
        threshold=state.settings.semantic_cache_threshold,
    )
    if semantic is not None:
        response, metadata, score = semantic
        response["cache"] = CacheInfo(semantic_hit=True, semantic_score=score).model_dump()
        metadata["last_hit_type"] = "semantic"
        state.exact_cache.set(exact_key, response)
        return response_model.model_validate(response)

    response = factory().model_dump()
    response["cache"] = CacheInfo().model_dump()
    state.exact_cache.set(exact_key, response)
    state.vector_store.add(
        SemanticCacheRecord(
            endpoint=vector_endpoint,
            input_hash=_input_hash(payload),
            embedding=embedding,
            response=response,
            metadata={"source": "deterministic_engine", "api_endpoint": endpoint},
        )
    )
    return response_model.model_validate(response)


def _input_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _vector_endpoint(endpoint: str, payload: dict[str, Any]) -> str:
    if endpoint == "order-recommendation":
        return f"{endpoint}:{payload.get('policy', 'base_stock')}"
    return endpoint
