import csv
import hashlib
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np

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
from app.services.demand_model import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    _korean_weekday,
    _training_row_features,
)
from app.services.llm import BedrockLlamaClient
from app.services.semantic_cache import ChatSemanticCache


MODEL_VERSION = "lgbm_global_v1"
BASELINE_MODEL_VERSION = "baseline_v1"
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
    model_bundle = _load_lgbm_bundle()
    if model_bundle is not None and _all_rows_resolvable(request.rows, model_bundle["metadata"]):
        return _predict_order_with_lgbm(request, history_rows, model_bundle)

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

    return V1OrderRecommendationResponse(modelVersion=BASELINE_MODEL_VERSION, predictions=predictions)


def predict_single_day_quantiles(request: V1ForecastRequest) -> V1ForecastResponse:
    history_rows = _load_sales_history(request.sales_history.presigned_urls)
    model_bundle = _load_lgbm_bundle()
    if model_bundle is not None and _all_rows_resolvable(request.rows, model_bundle["metadata"]):
        return _predict_single_day_with_lgbm(request, history_rows, model_bundle)

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
        modelVersion=BASELINE_MODEL_VERSION,
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


def _predict_order_with_lgbm(
    request: V1OrderRecommendationRequest,
    history_rows: list[dict[str, str]],
    model_bundle: dict[str, Any],
) -> V1OrderRecommendationResponse:
    start = date.fromisoformat(request.target_date) + timedelta(days=1)
    weather_by_date = {day.forecast_date: day for day in request.weather}
    latest_by_item = _latest_history_by_item(history_rows)
    predictions = []

    for row in request.rows:
        item_name = row.item_name or ""
        current = dict(latest_by_item.get(item_name) or _fallback_lag_row(row, model_bundle["metadata"]))
        daily = []
        for offset in range(request.coverage.coverage_days):
            target_day = start + timedelta(days=offset)
            weather = weather_by_date.get(target_day.isoformat())
            p50 = _predict_lgbm_quantity(row, target_day, weather, current, model_bundle)
            quantile = _quantiles_from_point(p50)
            daily.append(
                DailyQuantilePrediction(
                    date=target_day.isoformat(),
                    p10=quantile.p10,
                    p50=quantile.p50,
                    p90=quantile.p90,
                )
            )
            current = _next_lag_row(row, target_day, p50, weather, model_bundle["metadata"])

        predictions.append(OrderPrediction(itemId=row.item_id, daily=daily))

    return V1OrderRecommendationResponse(modelVersion=MODEL_VERSION, predictions=predictions)


def _predict_single_day_with_lgbm(
    request: V1ForecastRequest,
    history_rows: list[dict[str, str]],
    model_bundle: dict[str, Any],
) -> V1ForecastResponse:
    target_day = date.fromisoformat(request.target_date)
    latest_by_item = _latest_history_by_item(history_rows)
    predictions = []

    for row in request.rows:
        item_name = row.item_name or ""
        lag_row = latest_by_item.get(item_name) or _fallback_lag_row(row, model_bundle["metadata"])
        p50 = _predict_lgbm_quantity(row, target_day, request.weather, lag_row, model_bundle)
        quantile = _quantiles_from_point(p50)
        predictions.append(
            SingleDayPrediction(
                itemId=row.item_id,
                p10=quantile.p10,
                p50=quantile.p50,
                p90=quantile.p90,
            )
        )

    return V1ForecastResponse(
        modelVersion=MODEL_VERSION,
        targetDate=request.target_date,
        predictions=predictions,
    )


def _load_lgbm_bundle(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> dict[str, Any] | None:
    if not model_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != FEATURE_NAMES:
        return None

    import lightgbm as lgb

    return {
        "model": lgb.Booster(model_file=str(model_path)),
        "metadata": metadata,
    }


def _all_rows_resolvable(rows: list[Any], metadata: dict[str, Any]) -> bool:
    item_names = set(metadata.get("item_codes", {}))
    return all(bool(row.item_name) and row.item_name in item_names for row in rows)


def _predict_lgbm_quantity(
    row: Any,
    target_day: date,
    weather: WeatherDay | None,
    lag_row: dict[str, Any],
    model_bundle: dict[str, Any],
) -> float:
    metadata = model_bundle["metadata"]
    model = model_bundle["model"]
    model_row = _model_input_row(row, target_day, weather, metadata)
    feature = _training_row_features(model_row, metadata, lag_row)
    prediction = float(model.predict(np.array([feature], dtype=np.float64))[0])
    return round(max(0.0, prediction), 3)


def _model_input_row(row: Any, target_day: date, weather: WeatherDay | None, metadata: dict[str, Any]) -> dict[str, Any]:
    defaults = metadata.get("defaults", {})
    return {
        "날짜": target_day.isoformat(),
        "요일": _korean_weekday(target_day),
        "날씨": _weather_label(weather),
        "기온": weather.avg_temp if weather is not None else defaults.get("기온", 20.0),
        "강수mm": weather.precipitation_mm if weather is not None else defaults.get("강수mm", 0.0),
        "행사중여부": "False",
        "공휴일여부": row.features.is_holiday,
        "신메뉴여부": "False",
        "품목": row.item_name,
        "구분": _item_type(row, metadata),
        "수요": row.features.ma7,
        "판매수량": row.features.ma7,
        "매진여부": "False",
        "매진시각": "",
        "비고_시나리오": defaults.get("비고_시나리오", "normal"),
    }


def _fallback_lag_row(row: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    defaults = metadata.get("defaults", {})
    return {
        "날짜": "",
        "품목": row.item_name,
        "구분": _item_type(row, metadata),
        "수요": row.features.ma7,
        "판매수량": row.features.ma7,
        "매진여부": "False",
        "매진시각": defaults.get("매진시각", -1.0),
    }


def _next_lag_row(
    row: Any,
    target_day: date,
    prediction: float,
    weather: WeatherDay | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    next_row = _model_input_row(row, target_day, weather, metadata)
    next_row["수요"] = prediction
    next_row["판매수량"] = prediction
    return next_row


def _latest_history_by_item(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest_by_item: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda value: value.get("날짜", "")):
        latest_by_item[row.get("품목", "")] = row
    return latest_by_item


def _item_type(row: Any, metadata: dict[str, Any]) -> str:
    item_info = metadata.get("item_by_name", {}).get(row.item_name or "", {})
    return row.item_type or item_info.get("type") or "완제품"


def _weather_label(weather: WeatherDay | None) -> str:
    if weather is None:
        return "맑음"
    if weather.precipitation_mm > 0 or weather.precipitation_prob >= 60:
        return "비"
    if weather.sky_code >= 3:
        return "흐림"
    return "맑음"


def _quantiles_from_point(p50: float) -> QuantilePrediction:
    p50 = round(max(0.0, p50), 3)
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
