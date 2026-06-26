import csv
import hashlib
import io
import time
from datetime import date, timedelta
from typing import Any

import httpx

from app.api_contracts import (
    DailyQuantilePrediction,
    GenerateRequest,
    GenerateResponse,
    OrderPrediction,
    QuantilePrediction,
    SingleDayPrediction,
    V1ForecastRequest,
    V1ForecastResponse,
    V1OrderRecommendationRequest,
    V1OrderRecommendationResponse,
    WeatherDay,
)
from app.services.llm import BedrockLlamaClient
from app.services.semantic_cache import ChatSemanticCache


MODEL_VERSION = "baseline_v1"
LLM_MODEL_VERSION = "bedrock-llama3.2-1b"
SALES_CSV_COLUMNS_V1 = [
    "날짜",
    "요일",
    "날씨",
    "기온",
    "강수mm",
    "행사중여부",
    "공휴일여부",
    "신메뉴여부",
    "품목",
    "구분",
    "수요",
    "판매수량",
    "매진여부",
    "매진시각",
    "비고_시나리오",
]
LEGACY_SALES_CSV_COLUMNS_WITH_HOLIDAY = [
    "날짜",
    "요일",
    "날씨",
    "기온",
    "강수mm",
    "행사",
    "공휴일",
    "신메뉴",
    "품목",
    "구분",
    "판매수량",
    "비고_시나리오",
]


def predict_order_quantiles(request: V1OrderRecommendationRequest) -> V1OrderRecommendationResponse:
    history_rows = _load_sales_history(request.sales_history.presigned_urls)
    start = date.fromisoformat(request.target_date) + timedelta(days=1)
    weather_by_date = {day.forecast_date: day for day in request.weather}
    predictions = []

    for row in request.rows:
        daily = []
        for offset in range(request.coverage.coverage_days):
            target_day = start + timedelta(days=offset)
            weather = weather_by_date.get(target_day.isoformat())
            quantile = _baseline_quantile(row.features.ma7, row.features.trend, offset, weather)
            daily.append(
                DailyQuantilePrediction(
                    date=target_day.isoformat(),
                    p10=quantile.p10,
                    p50=quantile.p50,
                    p90=quantile.p90,
                )
            )
        predictions.append(OrderPrediction(itemId=row.item_id, daily=daily))

    return V1OrderRecommendationResponse(modelVersion=_model_version(history_rows), predictions=predictions)


def predict_single_day_quantiles(request: V1ForecastRequest) -> V1ForecastResponse:
    history_rows = _load_sales_history(request.sales_history.presigned_urls)
    predictions = []
    for row in request.rows:
        quantile = _baseline_quantile(row.features.ma7, row.features.trend, 0, request.weather)
        predictions.append(
            SingleDayPrediction(
                itemId=row.item_id,
                p10=quantile.p10,
                p50=quantile.p50,
                p90=quantile.p90,
            )
        )
    return V1ForecastResponse(
        modelVersion=_model_version(history_rows),
        targetDate=request.target_date,
        predictions=predictions,
    )


def generate_grounded_answer(
    request: GenerateRequest,
    semantic_cache: ChatSemanticCache,
    llm_client: BedrockLlamaClient | None,
) -> GenerateResponse:
    start = time.perf_counter()
    grounding_text = _stable_grounding_text(request.grounding)
    grounding_hash = hashlib.sha256(grounding_text.encode("utf-8")).hexdigest()
    namespace = f"v1:generate:{request.locale}:{grounding_hash}"
    cached = semantic_cache.get(namespace, request.question)
    if cached is not None:
        response, _score = cached
        return GenerateResponse(
            answer=str(response["answer"]),
            cacheHit=True,
            latencyMs=_elapsed_ms(start),
            tokens=int(response.get("tokens", 0)),
        )

    answer = _fallback_grounded_answer(request)
    if llm_client is not None:
        try:
            answer = llm_client.generate_text(
                prompt=_generate_prompt(request),
                system_prompt=(
                    "You are a Korean ordering explanation assistant. "
                    "Use only grounding values. Do not invent, change, or recalculate numbers. "
                    "Answer in two or three Korean sentences."
                ),
                max_tokens=220,
                temperature=0,
            )
        except Exception:
            answer = _fallback_grounded_answer(request)

    tokens = _estimate_tokens(request.question, request.grounding, answer)
    semantic_cache.set(namespace, request.question, {"answer": answer, "tokens": tokens})
    return GenerateResponse(answer=answer, cacheHit=False, latencyMs=_elapsed_ms(start), tokens=tokens)


def _baseline_quantile(ma7: float, trend: float, offset: int, weather: WeatherDay | None) -> QuantilePrediction:
    p50 = max(0.0, ma7 + trend * offset)
    if weather is not None:
        if weather.precipitation_prob >= 70 or weather.precipitation_mm >= 10:
            p50 *= 0.95
        if weather.sky_code <= 2:
            p50 *= 1.02
    p50 = round(p50, 3)
    return QuantilePrediction(
        p10=round(max(0.0, p50 * 0.8), 3),
        p50=p50,
        p90=round(max(p50, p50 * 1.3), 3),
    )


def _load_sales_history(urls: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for url in urls:
        if "example.invalid" in url:
            continue
        try:
            response = httpx.get(url, timeout=3.0)
            response.raise_for_status()
        except httpx.HTTPError:
            continue
        rows.extend(_parse_sales_csv(response.text))
    return rows


def _parse_sales_csv(content: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
    fieldnames = list(reader.fieldnames or [])
    if fieldnames not in [SALES_CSV_COLUMNS_V1, LEGACY_SALES_CSV_COLUMNS_WITH_HOLIDAY]:
        return []
    return list(reader)


def _model_version(history_rows: list[dict[str, str]]) -> str:
    return MODEL_VERSION


def _generate_prompt(request: GenerateRequest) -> str:
    return (
        f"질문: {request.question}\n"
        f"언어: {request.locale}\n"
        f"근거: {_stable_grounding_text(request.grounding)}\n"
        "근거에 있는 숫자만 그대로 인용해서 답하세요."
    )


def _fallback_grounded_answer(request: GenerateRequest) -> str:
    grounding = request.grounding
    item = grounding.get("item", {})
    recommendation = grounding.get("recommendation", {})
    forecast = grounding.get("forecast", {})
    carbon = grounding.get("carbon", {})
    item_name = item.get("itemName") or item.get("itemId") or "해당 품목"
    unit = item.get("unit", "")
    quantity = recommendation.get("recommendedQuantity")
    p50 = forecast.get("p50")
    potential = carbon.get("potentialSavingKg")

    parts = []
    if quantity is not None:
        parts.append(f"{item_name}는 {quantity}{unit} 발주를 권장합니다.")
    else:
        parts.append(f"{item_name}에 대한 근거 기준 설명입니다.")
    if p50 is not None:
        parts.append(f"제공된 수요 중앙값은 {p50}{unit}입니다.")
    if potential is not None:
        parts.append(f"제공된 잠재 탄소 절감량은 {potential}kgCO2e입니다.")
    return " ".join(parts)


def _stable_grounding_text(grounding: dict[str, Any]) -> str:
    import json

    return json.dumps(grounding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _estimate_tokens(question: str, grounding: dict[str, Any], answer: str) -> int:
    return max(1, int((len(question) + len(_stable_grounding_text(grounding)) + len(answer)) / 3))


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))
