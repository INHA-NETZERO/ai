#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.engines.deterministic import forecast_demand_from_closing_data, recommend_orders
from app.services.demo_data import ITEM_MASTER_PATH, ORDER_POLICY_PATH, _to_float, build_order_request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run saved LightGBM demand forecast and order recommendation from closing/inventory CSV."
    )
    parser.add_argument(
        "--inventory-flow",
        default="app/data/Test/inventory_demo_1y.csv",
        help="POS closing/inventory flow CSV.",
    )
    parser.add_argument("--item-master", default=str(ITEM_MASTER_PATH))
    parser.add_argument("--order-policy", default=str(ORDER_POLICY_PATH))
    parser.add_argument("--policy", choices=["base_stock", "ortools"], default="base_stock")
    parser.add_argument(
        "--item-type",
        action="append",
        default=[],
        help="Optional item type filter, for example --item-type 완제품. Repeatable.",
    )
    parser.add_argument(
        "--output",
        default="app/data/predictions/test_order_recommendations.csv",
        help="Output .csv or .json path.",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    data = {
        "store_id": "local-test",
        "business_date": "",
        "data_version": f"csv:{args.inventory_flow}",
        "inventory_flow": _read_csv(Path(args.inventory_flow)),
        "item_master": _read_csv(Path(args.item_master)),
        "order_policy": _read_csv(Path(args.order_policy)),
    }
    if args.item_type:
        allowed_types = set(args.item_type)
        data["inventory_flow"] = [row for row in data["inventory_flow"] if row.get("구분") in allowed_types]
        data["item_master"] = [row for row in data["item_master"] if row.get("구분") in allowed_types]
        data["order_policy"] = [row for row in data["order_policy"] if row.get("구분") in allowed_types]

    if not data["inventory_flow"]:
        raise SystemExit(f"{args.inventory_flow} has no rows.")

    data["business_date"] = max(row["날짜"] for row in data["inventory_flow"])
    forecast = forecast_demand_from_closing_data(data)
    order_request = build_order_request(data, forecast.forecasts)
    order_request.policy = args.policy
    response = recommend_orders(order_request)

    rows = _build_output_rows(data["inventory_flow"], forecast.forecasts, response.model_dump()["recommendations"])
    _write_output(Path(args.output), rows)

    summary = _summary(rows, forecast.method, response.method, data["business_date"], args.item_type)
    print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote {len(rows)} order recommendation rows: {args.output}")
    for row in rows[: max(0, args.limit)]:
        print(json.dumps(row, ensure_ascii=False))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _build_output_rows(
    inventory_flow: list[dict[str, str]],
    forecasts: list[Any],
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_rows = _latest_rows_by_item(inventory_flow)
    forecast_by_sku = _forecast_by_sku(forecasts)
    output = []

    for recommendation in sorted(recommendations, key=lambda row: row["sku"]):
        sku = recommendation["sku"]
        latest = latest_rows.get(sku, {})
        forecast_values = forecast_by_sku.get(sku, [])
        forecast_total = round(sum(forecast_values), 3)
        current_stock = _to_float(latest.get("기말재고", 0))
        stockout = _to_float(latest.get("결품", 0))
        historical_order = _to_float(latest.get("발주", 0))
        recommended_quantity = _to_float(recommendation["recommended_quantity"])
        output.append(
            {
                "business_date": latest.get("날짜", ""),
                "itemName": sku,
                "itemType": latest.get("구분", ""),
                "unit": latest.get("단위", ""),
                "latest_demand": _to_float(latest.get("수요", 0)),
                "latest_sales": _to_float(latest.get("실판매", 0)),
                "current_stock": current_stock,
                "stockout": stockout,
                "forecast_days": len(forecast_values),
                "forecast_total": forecast_total,
                "base_stock_level": recommendation["base_stock_level"],
                "projected_position": recommendation["projected_position"],
                "recommended_quantity": recommended_quantity,
                "historical_order_quantity": historical_order,
                "recommendation_minus_historical": round(recommended_quantity - historical_order, 3),
                "reason": recommendation["reason"],
            }
        )
    return output


def _latest_rows_by_item(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest_date = max(row["날짜"] for row in rows)
    return {row["품목"]: row for row in rows if row["날짜"] == latest_date}


def _forecast_by_sku(forecasts: list[Any]) -> dict[str, list[float]]:
    by_sku: dict[str, list[float]] = {}
    for point in forecasts:
        by_sku.setdefault(point.sku, []).append(float(point.quantity))
    return by_sku


def _summary(
    rows: list[dict[str, Any]],
    forecast_method: str,
    order_method: str,
    business_date: str,
    item_type_filter: list[str],
) -> dict[str, Any]:
    total_recommended = round(sum(_to_float(row["recommended_quantity"]) for row in rows), 3)
    total_historical = round(sum(_to_float(row["historical_order_quantity"]) for row in rows), 3)
    total_forecast = round(sum(_to_float(row["forecast_total"]) for row in rows), 3)
    ordered_items = sum(1 for row in rows if _to_float(row["recommended_quantity"]) > 0)
    return {
        "business_date": business_date,
        "item_type_filter": item_type_filter or "all",
        "forecast_method": forecast_method,
        "order_method": order_method,
        "items": len(rows),
        "ordered_items": ordered_items,
        "total_forecast_demand": total_forecast,
        "total_recommended_quantity": total_recommended,
        "total_historical_order_quantity": total_historical,
        "total_recommendation_minus_historical": round(total_recommended - total_historical, 3),
    }


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
