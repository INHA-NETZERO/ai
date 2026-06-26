from contextlib import asynccontextmanager
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import FastAPI
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.engines.deterministic import forecast_demand_from_closing_data, recommend_orders
from app.schemas import (
    CacheInfo,
    CacheStatusResponse,
    ChatRequest,
    ChatResponse,
    DailyCloseResponse,
    ForecastResponse,
    OrderRecommendationResponse,
)
from app.services.cache import ExactCache, cache_key
from app.services.aws_metrics import AwsMetricsClient
from app.services.demo_data import build_order_request, closing_cache_payload, load_demo_closing_data
from app.services.llm import BedrockLlamaClient
from app.services.metrics import CacheMetrics
from app.services.rag import build_order_rag_context
from app.services.semantic_cache import ChatSemanticCache

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.exact_cache = ExactCache(settings.redis_url)
        self.llm_client = _build_bedrock_client(settings)
        self.chat_semantic_cache: ChatSemanticCache | None = None
        self.cache_metrics = CacheMetrics()
        self.aws_metrics_client = _build_aws_metrics_client(settings)


def _build_bedrock_client(settings: Settings) -> BedrockLlamaClient | None:
    try:
        return BedrockLlamaClient(settings.aws_region, settings.bedrock_model_id)
    except Exception:
        return None


def _build_aws_metrics_client(settings: Settings) -> AwsMetricsClient | None:
    if not settings.elasticache_replication_group_id and not settings.elasticache_cache_cluster_id:
        return None
    try:
        return AwsMetricsClient(settings)
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
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "bedrock_model_id": settings.bedrock_model_id,
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
        aws_elasticache=(
            state.aws_metrics_client.get_elasticache_metrics()
            if state.aws_metrics_client is not None
            else None
        ),
        **state.cache_metrics.model_dump(),
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast() -> ForecastResponse:
    data = load_demo_closing_data()
    payload = closing_cache_payload(data)
    return _cached_response("forecast", payload, ForecastResponse, lambda: forecast_demand_from_closing_data(data))


@app.post("/order-recommendation", response_model=OrderRecommendationResponse)
def order_recommendation() -> OrderRecommendationResponse:
    data = load_demo_closing_data()
    forecast_response = forecast_demand_from_closing_data(data)
    request = build_order_request(data, forecast_response.forecasts)
    payload = closing_cache_payload(data) | {"policy": request.policy}
    return _cached_response(
        "order-recommendation",
        payload,
        OrderRecommendationResponse,
        lambda: recommend_orders(request),
    )


@app.post("/daily-close", response_model=DailyCloseResponse)
def daily_close() -> DailyCloseResponse:
    data = load_demo_closing_data()
    payload = closing_cache_payload(data) | {"output": "llm_summary"}

    def factory() -> DailyCloseResponse:
        forecast_response = forecast_demand_from_closing_data(data)
        order_response = recommend_orders(build_order_request(data, forecast_response.forecasts))
        llm_output = _build_llm_output(data, forecast_response, order_response)
        return DailyCloseResponse(
            store_id=data["store_id"],
            business_date=data["business_date"],
            data_version=data["data_version"],
            forecast=forecast_response,
            order_recommendation=order_response,
            llm_output=llm_output,
        )

    return _cached_response("daily-close", payload, DailyCloseResponse, factory)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state: AppState = app.state.runtime
    data = load_demo_closing_data()
    namespace = f"chat:{data['store_id']}:{data['business_date']}:{data['data_version']}"
    chat_semantic_cache = _chat_semantic_cache(state)
    cached = chat_semantic_cache.get(namespace, request.question)
    if cached is not None:
        response, score = cached
        if response.get("sources"):
            state.cache_metrics.semantic_hits += 1
            response["cache"] = CacheInfo(semantic_hit=True, semantic_score=score).model_dump()
            return ChatResponse.model_validate(response)
    state.cache_metrics.semantic_misses += 1

    forecast_response = forecast_demand_from_closing_data(data)
    order_response = recommend_orders(build_order_request(data, forecast_response.forecasts))
    answer, sources = _build_chat_answer(data, forecast_response, order_response, request.question)
    response = ChatResponse(answer=answer, sources=sources).model_dump()
    chat_semantic_cache.set(namespace, request.question, response)
    return ChatResponse.model_validate(response)


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


def _build_llm_output(
    data: dict[str, Any],
    forecast_response: ForecastResponse,
    order_response: OrderRecommendationResponse,
) -> str:
    prompt = _summary_prompt(data, forecast_response, order_response)
    state: AppState = app.state.runtime
    if state.llm_client is None:
        return _fallback_summary(data, order_response)
    try:
        return state.llm_client.generate_text(
            prompt=prompt,
            system_prompt=(
                "You explain POS closing order recommendations in Korean. "
                "Do not change quantities. Keep the answer short and practical."
            ),
            max_tokens=500,
            temperature=0,
        )
    except Exception:
        return _fallback_summary(data, order_response)


def _build_chat_answer(
    data: dict[str, Any],
    forecast_response: ForecastResponse,
    order_response: OrderRecommendationResponse,
    question: str,
) -> tuple[str, list[str]]:
    state: AppState = app.state.runtime
    rag_context, sources = build_order_rag_context(data, forecast_response, order_response, question)
    prompt = (
        f"점주 질문: {question}\n"
        f"매장: {data['store_id']}\n"
        f"마감일: {data['business_date']}\n"
        f"RAG 근거:\n{rag_context}\n"
        "위 근거 안에서만 답하고, 계산값은 바꾸지 말고 추천 근거를 짧게 설명해줘."
    )
    if state.llm_client is None:
        return _fallback_summary(data, order_response), sources
    try:
        return state.llm_client.generate_text(
            prompt=prompt,
            system_prompt=(
                "You are an AWS Bedrock Llama ordering explanation chatbot. "
                "Answer in Korean, cite only the provided POS closing and recommendation context, "
                "and do not change numeric recommendations."
            ),
            max_tokens=400,
            temperature=0,
        ), sources
    except Exception:
        return _fallback_summary(data, order_response), sources


def _summary_prompt(
    data: dict[str, Any],
    forecast_response: ForecastResponse,
    order_response: OrderRecommendationResponse,
) -> str:
    return (
        f"매장: {data['store_id']}\n"
        f"마감일: {data['business_date']}\n"
        f"예측 결과: {forecast_response.model_dump_json(ensure_ascii=False)}\n"
        f"발주 추천: {order_response.model_dump_json(ensure_ascii=False)}\n"
        "점주 화면에 띄울 오늘 마감 발주 추천 요약을 한국어로 작성해줘."
    )


def _fallback_summary(data: dict[str, Any], order_response: OrderRecommendationResponse) -> str:
    lines = [f"{data['business_date']} 마감 기준 발주 추천입니다."]
    for item in order_response.recommendations:
        lines.append(
            f"{item.sku}: {item.recommended_quantity:g}개 추천 "
            f"(기준재고 {item.base_stock_level:g}, 예상가용 {item.projected_position:g})."
        )
    return " ".join(lines)
