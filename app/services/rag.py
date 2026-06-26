from typing import Any

from app.schemas import ForecastResponse, OrderRecommendationResponse


def build_order_rag_context(
    data: dict[str, Any],
    forecast_response: ForecastResponse,
    order_response: OrderRecommendationResponse,
    question: str,
    max_items: int = 3,
) -> tuple[str, list[str]]:
    latest_rows = _latest_inventory_rows(data["inventory_flow"])
    policy_by_name = {row["품목명"]: row for row in data["order_policy"]}
    forecast_by_sku = _forecast_by_sku(forecast_response)
    selected_names = _select_items(question, order_response, latest_rows, max_items)

    blocks = []
    sources = []
    for name in selected_names:
        latest = latest_rows.get(name, {})
        policy = policy_by_name.get(name, {})
        recommendation = next((item for item in order_response.recommendations if item.sku == name), None)
        forecast_values = forecast_by_sku.get(name, [])[:5]
        blocks.append(
            "\n".join(
                [
                    f"품목: {name}",
                    f"최근 마감일: {latest.get('날짜', data['business_date'])}",
                    f"최근 수요: {latest.get('수요', 'N/A')}",
                    f"실판매: {latest.get('실판매', 'N/A')}",
                    f"결품: {latest.get('결품', 'N/A')}",
                    f"폐기: {latest.get('폐기', 'N/A')}",
                    f"기말재고: {latest.get('기말재고', 'N/A')}",
                    f"발주주기: {policy.get('발주주기_T(일)', 'N/A')}",
                    f"리드타임: {policy.get('리드타임_LT(일)', 'N/A')}",
                    f"예측 horizon: {policy.get('예측horizon_T+LT(일)', 'N/A')}",
                    f"MOQ: {policy.get('발주단위(MOQ)', 'N/A')}",
                    f"추천발주량: {recommendation.recommended_quantity if recommendation else 'N/A'}",
                    f"기준재고: {recommendation.base_stock_level if recommendation else 'N/A'}",
                    f"예상가용재고: {recommendation.projected_position if recommendation else 'N/A'}",
                    f"예측수요(앞 5개): {forecast_values}",
                ]
            )
        )
        sources.append(f"inventory_flow: {name} {latest.get('날짜', data['business_date'])} 마감 row")
        sources.append(f"order_policy: {name} 발주 정책")
        sources.append(f"forecast/order_recommendation: {name} 계산 결과")

    context = "\n\n---\n\n".join(blocks)
    return context, sources


def _latest_inventory_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest_date = max(row["날짜"] for row in rows)
    return {row["품목"]: row for row in rows if row["날짜"] == latest_date}


def _forecast_by_sku(response: ForecastResponse) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for point in response.forecasts:
        grouped.setdefault(point.sku, []).append(point.quantity)
    return grouped


def _select_items(
    question: str,
    order_response: OrderRecommendationResponse,
    latest_rows: dict[str, dict[str, str]],
    max_items: int,
) -> list[str]:
    normalized = question.replace(" ", "").lower()
    matched = [
        name
        for name in latest_rows
        if name.replace(" ", "").lower() in normalized
        or any(token and token in normalized for token in name.replace("/", " ").split())
    ]
    if matched:
        return matched[:max_items]
    ranked = sorted(
        order_response.recommendations,
        key=lambda item: item.recommended_quantity,
        reverse=True,
    )
    return [item.sku for item in ranked[:max_items]]
