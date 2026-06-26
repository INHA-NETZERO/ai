from contextlib import asynccontextmanager
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import Body, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ValidationError

from app.api_contracts import (
    GenerateRequest,
    GenerateResponse,
    V1ForecastRequest,
    V1ForecastResponse,
    V1OrderRecommendationRequest,
    V1OrderRecommendationResponse,
)
from app.core.config import Settings, get_settings
from app.engines.deterministic import forecast_demand_from_closing_data, recommend_orders
from app.schemas import (
    CacheInfo,
    CacheStatusResponse,
    ForecastResponse,
    IntegrationStatusResponse,
    OrderRecommendationResponse,
)
from app.services.cache import ExactCache, cache_key
from app.services.demo_data import build_order_request, closing_cache_payload, load_closing_data
from app.services.llm import LocalLlamaClient
from app.services.metrics import CacheMetrics
from app.services.semantic_cache import ChatSemanticCache
from app.services.integration_status import build_integration_status
from app.services.v1_contract import (
    LLM_MODEL_VERSION,
    MODEL_VERSION,
    generate_grounded_answer,
    predict_order_quantiles,
    predict_single_day_quantiles,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.exact_cache = ExactCache(settings.redis_url)
        self.llm_client = _build_llm_client(settings)
        self.chat_semantic_cache: ChatSemanticCache | None = None
        self.cache_metrics = CacheMetrics()


def _build_llm_client(settings: Settings) -> LocalLlamaClient | None:
    try:
        return LocalLlamaClient(settings)
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = AppState(get_settings())
    yield


app = FastAPI(title=get_settings().app_name, version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    message = str(first_error.get("msg", "Invalid request"))
    return JSONResponse(
        status_code=400,
        content={"error": {"code": "BAD_REQUEST", "message": message}},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    runtime = getattr(app.state, "runtime", None)
    exact_backend = runtime.exact_cache.backend if runtime is not None else "unknown"
    semantic_backend = (
        runtime.chat_semantic_cache.backend
        if runtime is not None and runtime.chat_semantic_cache is not None
        else "sqlite_vec_or_sqlite"
    )
    status = build_integration_status(settings, exact_backend, semantic_backend)
    return {
        "status": "UP",
        "model": {
            "forecast": MODEL_VERSION,
            "llm": LLM_MODEL_VERSION if status["llm"]["configured"] else "fallback_v1",
        },
        "llm_provider": settings.llm_provider,
        "local_llm_backend": settings.local_llm_backend,
        "local_llm_model": settings.local_llm_model,
        "data_source": status["data_source"]["active"],
        "gaps": status["gaps"],
    }


@app.get("/cache-status", response_model=CacheStatusResponse)
def cache_status() -> CacheStatusResponse:
    state: AppState = app.state.runtime
    semantic_backend = (
        state.chat_semantic_cache.backend
        if state.chat_semantic_cache is not None
        else "sqlite_vec_or_sqlite"
    )
    return CacheStatusResponse(
        exact_cache_backend=state.exact_cache.backend,
        semantic_cache_backend=semantic_backend,
        **state.cache_metrics.model_dump(),
    )


@app.get("/integration-status", response_model=IntegrationStatusResponse)
def integration_status() -> IntegrationStatusResponse:
    state: AppState = app.state.runtime
    semantic_backend = (
        state.chat_semantic_cache.backend
        if state.chat_semantic_cache is not None
        else "sqlite_vec_or_sqlite"
    )
    return IntegrationStatusResponse.model_validate(
        build_integration_status(
            state.settings,
            state.exact_cache.backend,
            semantic_backend,
        )
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    state: AppState = app.state.runtime
    metrics_dump = state.cache_metrics.model_dump()
    return "\n".join(
        [
            f"ai_exact_cache_hits {metrics_dump['exact_hits']}",
            f"ai_exact_cache_misses {metrics_dump['exact_misses']}",
            f"ai_semantic_cache_hits {metrics_dump['semantic_hits']}",
            f"ai_semantic_cache_misses {metrics_dump['semantic_misses']}",
            f"ai_estimated_llm_calls_saved {metrics_dump['estimated_llm_calls_saved']}",
            "",
        ]
    )


@app.post("/v1/order-recommendation", response_model=V1OrderRecommendationResponse)
def v1_order_recommendation(
    payload: dict[str, Any] = Body(...),
) -> V1OrderRecommendationResponse:
    request = _validate_body(V1OrderRecommendationRequest, payload)
    return predict_order_quantiles(request)


@app.post("/v1/forecast", response_model=V1ForecastResponse)
def v1_forecast(payload: dict[str, Any] = Body(...)) -> V1ForecastResponse:
    request = _validate_body(V1ForecastRequest, payload)
    return predict_single_day_quantiles(request)


@app.post("/v1/generate", response_model=GenerateResponse)
def v1_generate(payload: dict[str, Any] = Body(...)) -> GenerateResponse:
    request = _validate_body(GenerateRequest, payload)
    state: AppState = app.state.runtime
    response = generate_grounded_answer(
        request,
        semantic_cache=_chat_semantic_cache(state),
        llm_client=state.llm_client,
    )
    if response.cache_hit:
        state.cache_metrics.semantic_hits += 1
    else:
        state.cache_metrics.semantic_misses += 1
    return response


def _validate_body(model: type[ResponseT], payload: dict[str, Any]) -> ResponseT:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


@app.post("/forecast", response_model=ForecastResponse)
def forecast() -> ForecastResponse:
    state: AppState = app.state.runtime
    data = load_closing_data(state.settings)
    payload = closing_cache_payload(data)
    return _cached_response("forecast", payload, ForecastResponse, lambda: forecast_demand_from_closing_data(data))


@app.post("/order-recommendation", response_model=OrderRecommendationResponse)
def order_recommendation() -> OrderRecommendationResponse:
    state: AppState = app.state.runtime
    data = load_closing_data(state.settings)
    forecast_response = forecast_demand_from_closing_data(data)
    request = build_order_request(data, forecast_response.forecasts)
    payload = closing_cache_payload(data) | {"policy": request.policy}
    return _cached_response(
        "order-recommendation",
        payload,
        OrderRecommendationResponse,
        lambda: recommend_orders(request),
    )


def _cached_response(
    endpoint: str,
    payload: dict[str, Any],
    response_model: type[ResponseT],
    factory: Callable[[], ResponseT],
) -> ResponseT:
    state: AppState = app.state.runtime
    exact_key = cache_key(endpoint, payload)

    exact = state.exact_cache.get(exact_key)
    if exact is not None:
        state.cache_metrics.exact_hits += 1
        exact["cache"] = CacheInfo(exact_hit=True).model_dump()
        return response_model.model_validate(exact)
    state.cache_metrics.exact_misses += 1

    response = factory().model_dump()
    response["cache"] = CacheInfo().model_dump()
    state.exact_cache.set(exact_key, response)
    return response_model.model_validate(response)


def _chat_semantic_cache(state: AppState) -> ChatSemanticCache:
    if state.chat_semantic_cache is None:
        state.chat_semantic_cache = ChatSemanticCache(state.settings)
    return state.chat_semantic_cache
