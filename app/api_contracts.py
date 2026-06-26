from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class SalesHistory(BaseModel):
    presigned_urls: list[str] = Field(..., alias="presignedUrls", min_length=1)
    format: Literal["sales_csv_v1"]


class Coverage(BaseModel):
    lead_time_days: int = Field(..., alias="leadTimeDays", ge=0)
    order_cycle_days: int = Field(..., alias="orderCycleDays", ge=0)
    coverage_days: int = Field(..., alias="coverageDays", ge=1)

    @model_validator(mode="after")
    def validate_coverage_days(self) -> "Coverage":
        expected = self.lead_time_days + self.order_cycle_days
        if self.coverage_days != expected:
            raise ValueError("coverageDays mismatch (leadTimeDays + orderCycleDays)")
        return self


class WeatherDay(BaseModel):
    forecast_date: str = Field(..., alias="forecastDate")
    avg_temp: float = Field(..., alias="avgTemp")
    precipitation_mm: float = Field(..., alias="precipitationMm", ge=0)
    precipitation_prob: int = Field(..., alias="precipitationProb", ge=0, le=100)
    sky_code: int = Field(..., alias="skyCode", ge=1, le=4)


class PredictionFeatures(BaseModel):
    day_of_week: int = Field(..., alias="dayOfWeek", ge=0, le=6)
    is_holiday: bool = Field(..., alias="isHoliday")
    ma7: float = Field(..., ge=0)
    trend: float = 0.0


class OrderForecastRow(BaseModel):
    item_id: int = Field(..., alias="itemId")
    item_name: str | None = Field(default=None, alias="itemName")
    item_type: str | None = Field(default=None, alias="itemType")
    order_cycle_days: int | None = Field(default=None, alias="orderCycleDays", ge=0)
    lead_time_days: int | None = Field(default=None, alias="leadTimeDays", ge=0)
    features: PredictionFeatures


class ForecastRow(BaseModel):
    item_id: int = Field(..., alias="itemId")
    item_name: str | None = Field(default=None, alias="itemName")
    item_type: str | None = Field(default=None, alias="itemType")
    features: PredictionFeatures


class V1OrderRecommendationRequest(BaseModel):
    store_id: int = Field(..., alias="storeId")
    target_date: str = Field(..., alias="targetDate")
    sales_history: SalesHistory = Field(..., alias="salesHistory")
    coverage: Coverage
    weather: list[WeatherDay]
    rows: list[OrderForecastRow] = Field(..., min_length=1)


class V1ForecastRequest(BaseModel):
    store_id: int = Field(..., alias="storeId")
    target_date: str = Field(..., alias="targetDate")
    sales_history: SalesHistory = Field(..., alias="salesHistory")
    weather: WeatherDay
    rows: list[ForecastRow] = Field(..., min_length=1)


class QuantilePrediction(BaseModel):
    p10: float = Field(..., ge=0)
    p50: float = Field(..., ge=0)
    p90: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "QuantilePrediction":
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError("quantiles must satisfy p10 <= p50 <= p90")
        return self


class DailyQuantilePrediction(QuantilePrediction):
    date: str


class OrderPrediction(BaseModel):
    item_id: int = Field(..., alias="itemId")
    daily: list[DailyQuantilePrediction] | None = None
    aggregate: QuantilePrediction | None = None


class V1OrderRecommendationResponse(BaseModel):
    model_version: str = Field(..., alias="modelVersion")
    predictions: list[OrderPrediction]


class SingleDayPrediction(QuantilePrediction):
    item_id: int = Field(..., alias="itemId")


class V1ForecastResponse(BaseModel):
    model_version: str = Field(..., alias="modelVersion")
    target_date: str = Field(..., alias="targetDate")
    predictions: list[SingleDayPrediction]


class GenerateRequest(BaseModel):
    question: str
    locale: str = "ko"
    grounding: dict[str, Any]


class GenerateResponse(BaseModel):
    answer: str
    cache_hit: bool = Field(..., alias="cacheHit")
    latency_ms: int = Field(..., alias="latencyMs")
    tokens: int


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    locale: str = "ko"
    grounding: dict[str, Any]
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    cache_hit: bool = Field(..., alias="cacheHit")
    latency_ms: int = Field(..., alias="latencyMs")
    tokens: int
