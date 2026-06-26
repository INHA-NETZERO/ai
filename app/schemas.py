from typing import Literal

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


class CarbonItem(BaseModel):
    sku: str | None = None
    name: str
    category: str
    quantity: float = Field(..., ge=0)
    weight_kg_per_unit: float = Field(default=1, ge=0)
    distance_km: float = Field(default=0, ge=0)
    transport_mode: Literal["truck", "rail", "ship", "air"] = "truck"


class CarbonEstimateRequest(BaseModel):
    items: list[CarbonItem] = Field(..., min_length=1)


class CarbonItemEstimate(BaseModel):
    name: str
    category: str
    emissions_kg_co2e: float
    material_kg_co2e: float
    transport_kg_co2e: float
    factor_source: str


class CarbonEstimateResponse(BaseModel):
    items: list[CarbonItemEstimate]
    total_kg_co2e: float
    cache: CacheInfo = Field(default_factory=CacheInfo)
