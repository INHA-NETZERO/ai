from typing import Any, Literal

from pydantic import BaseModel, Field


class CacheInfo(BaseModel):
    exact_hit: bool = False
    semantic_hit: bool = False
    semantic_score: float | None = None


class ForecastPoint(BaseModel):
    sku: str
    period: str
    quantity: float = Field(..., ge=0)


class ForecastRequest(BaseModel):
    history: list[ForecastPoint] = Field(..., min_length=1)
    horizon: int = Field(default=7, ge=1, le=365)


class ForecastResponse(BaseModel):
    forecasts: list[ForecastPoint]
    method: str
    cache: CacheInfo = Field(default_factory=CacheInfo)


class InventoryPosition(BaseModel):
    sku: str
    on_hand: float = Field(..., ge=0)
    on_order: float = Field(default=0, ge=0)
    backorder: float = Field(default=0, ge=0)
    lead_time_days: int = Field(default=1, ge=0)
    safety_stock: float = Field(default=0, ge=0)
    pack_size: float = Field(default=1, gt=0)


class OrderRecommendationRequest(BaseModel):
    inventory: list[InventoryPosition] = Field(..., min_length=1)
    forecast: list[ForecastPoint] = Field(..., min_length=1)
    policy: Literal["base_stock", "ortools"] = "base_stock"


class RecommendedOrder(BaseModel):
    sku: str
    recommended_quantity: float = Field(..., ge=0)
    base_stock_level: float = Field(..., ge=0)
    projected_position: float
    reason: str


class OrderRecommendationResponse(BaseModel):
    recommendations: list[RecommendedOrder]
    method: str
    cache: CacheInfo = Field(default_factory=CacheInfo)


class CacheStatusResponse(BaseModel):
    exact_cache_backend: str
    semantic_cache_backend: str
    exact_hits: int
    exact_misses: int
    exact_hit_rate: float
    semantic_hits: int
    semantic_misses: int
    semantic_hit_rate: float
    estimated_llm_calls_saved: int


class IntegrationStatusResponse(BaseModel):
    environment: str
    aws: dict[str, Any]
    llm: dict[str, Any]
    data_source: dict[str, Any]
    cache: dict[str, Any]
    model: dict[str, Any]
    gaps: list[str] = Field(default_factory=list)
