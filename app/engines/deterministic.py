from collections import defaultdict
from datetime import date, timedelta
from math import ceil

import numpy as np

from app.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    InventoryPosition,
    OrderRecommendationRequest,
    OrderRecommendationResponse,
    RecommendedOrder,
)


def forecast_demand(request: ForecastRequest) -> ForecastResponse:
    by_sku: dict[str, list[ForecastPoint]] = defaultdict(list)
    for point in request.history:
        by_sku[point.sku].append(point)

    forecasts: list[ForecastPoint] = []
    for sku, points in by_sku.items():
        ordered = sorted(points, key=lambda point: point.period)
        quantities = np.array([point.quantity for point in ordered], dtype=np.float64)
        next_values = _lightgbm_forecast(quantities, request.horizon)
        start = _next_period(ordered[-1].period)
        for offset, quantity in enumerate(next_values):
            forecasts.append(
                ForecastPoint(
                    sku=sku,
                    period=(start + timedelta(days=offset)).isoformat(),
                    quantity=round(max(0.0, float(quantity)), 3),
                )
            )

    method = "lightgbm" if _lightgbm_available() and len(request.history) >= 6 else "moving_average_trend"
    return ForecastResponse(forecasts=forecasts, method=method)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


def _lightgbm_forecast(values: np.ndarray, horizon: int) -> list[float]:
    if len(values) < 6 or not _lightgbm_available():
        return _moving_average_trend(values, horizon)

    import lightgbm as lgb

    window = min(7, len(values) - 1)
    features = []
    targets = []
    for idx in range(window, len(values)):
        features.append(values[idx - window : idx])
        targets.append(values[idx])

    model = lgb.LGBMRegressor(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=3,
        min_child_samples=1,
        verbosity=-1,
        random_state=42,
    )
    model.fit(np.array(features), np.array(targets))

    rolling = list(values[-window:])
    output = []
    for _ in range(horizon):
        prediction = float(model.predict(np.array([rolling[-window:]]))[0])
        output.append(prediction)
        rolling.append(max(0.0, prediction))
    return output


def _moving_average_trend(values: np.ndarray, horizon: int) -> list[float]:
    window = min(7, len(values))
    baseline = float(values[-window:].mean())
    if len(values) >= 2:
        trend = float((values[-1] - values[0]) / max(1, len(values) - 1))
    else:
        trend = 0.0
    return [baseline + trend * (idx + 1) for idx in range(horizon)]


def _next_period(period: str) -> date:
    try:
        return date.fromisoformat(period) + timedelta(days=1)
    except ValueError:
        return date.today() + timedelta(days=1)


def recommend_orders(request: OrderRecommendationRequest) -> OrderRecommendationResponse:
    forecast_by_sku: dict[str, list[ForecastPoint]] = defaultdict(list)
    for point in request.forecast:
        forecast_by_sku[point.sku].append(point)

    if request.policy == "ortools":
        recommendations = _ortools_base_stock_recommendations(request.inventory, forecast_by_sku)
        method = "ortools_base_stock"
    else:
        recommendations = [
            _base_stock_recommendation(item, forecast_by_sku.get(item.sku, []))
            for item in request.inventory
        ]
        method = "base_stock"
    return OrderRecommendationResponse(recommendations=recommendations, method=method)


def _base_stock_recommendation(item: InventoryPosition, forecast: list[ForecastPoint]) -> RecommendedOrder:
    daily_values = [point.quantity for point in sorted(forecast, key=lambda point: point.period)]
    lead_time_demand = sum(daily_values[: max(1, item.lead_time_days)])
    base_stock_level = lead_time_demand + item.safety_stock
    projected_position = item.on_hand + item.on_order - item.backorder
    raw_order = max(0.0, base_stock_level - projected_position)
    recommended = ceil(raw_order / item.pack_size) * item.pack_size if raw_order > 0 else 0.0
    return RecommendedOrder(
        sku=item.sku,
        recommended_quantity=round(recommended, 3),
        base_stock_level=round(base_stock_level, 3),
        projected_position=round(projected_position, 3),
        reason="Order up to lead-time demand plus safety stock, rounded to pack size.",
    )


def _ortools_base_stock_recommendations(
    inventory: list[InventoryPosition],
    forecast_by_sku: dict[str, list[ForecastPoint]],
) -> list[RecommendedOrder]:
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        return [_base_stock_recommendation(item, forecast_by_sku.get(item.sku, [])) for item in inventory]

    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return [_base_stock_recommendation(item, forecast_by_sku.get(item.sku, [])) for item in inventory]

    pack_vars = {}
    base_levels = {}
    projected_positions = {}
    for item in inventory:
        forecast = sorted(forecast_by_sku.get(item.sku, []), key=lambda point: point.period)
        lead_time_demand = sum(point.quantity for point in forecast[: max(1, item.lead_time_days)])
        base_stock_level = lead_time_demand + item.safety_stock
        projected_position = item.on_hand + item.on_order - item.backorder
        pack_var = solver.IntVar(0, solver.infinity(), f"packs_{item.sku}")
        solver.Add(projected_position + pack_var * item.pack_size >= base_stock_level)
        pack_vars[item.sku] = (item, pack_var)
        base_levels[item.sku] = base_stock_level
        projected_positions[item.sku] = projected_position

    solver.Minimize(sum(pack_var * item.pack_size for item, pack_var in pack_vars.values()))
    status = solver.Solve()
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        return [_base_stock_recommendation(item, forecast_by_sku.get(item.sku, [])) for item in inventory]

    recommendations = []
    for sku, (item, pack_var) in pack_vars.items():
        recommended = pack_var.solution_value() * item.pack_size
        recommendations.append(
            RecommendedOrder(
                sku=sku,
                recommended_quantity=round(recommended, 3),
                base_stock_level=round(base_levels[sku], 3),
                projected_position=round(projected_positions[sku], 3),
                reason="OR-Tools minimized replenishment while satisfying base-stock constraints.",
            )
        )
    return recommendations
