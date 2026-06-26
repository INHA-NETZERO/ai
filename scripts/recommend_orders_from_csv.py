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
    parser.add_argument(
        "--report",
        help="Optional Markdown report path. Defaults to <output stem>_report.md.",
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
    summary = _summary(rows, forecast.method, response.method, data["business_date"], args.item_type)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(f"{output_path.stem}_report.md")
    _write_output(output_path, rows)
    _write_report(report_path, summary, rows)

    print(_human_summary(summary))
    print(f"상세 CSV 저장: {output_path}")
    print(f"요약 리포트 저장: {report_path}")
    print(_human_table(rows[: max(0, args.limit)]))


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
        order_reduction = max(0.0, historical_order - recommended_quantity)
        existing_waste = _to_float(latest.get("폐기", 0))
        existing_waste_kg = _to_float(latest.get("폐기_kg", 0))
        existing_carbon = _to_float(latest.get("탄소_kgCO2e", 0))
        expected_waste_reduction = min(order_reduction, existing_waste)
        expected_waste_reduction_kg = (
            existing_waste_kg * expected_waste_reduction / existing_waste
            if existing_waste > 0
            else 0.0
        )
        expected_carbon_saving = (
            existing_carbon * expected_waste_reduction / existing_waste
            if existing_waste > 0
            else 0.0
        )
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
                "order_reduction_quantity": round(order_reduction, 3),
                "existing_waste_quantity": existing_waste,
                "expected_waste_reduction_quantity": round(expected_waste_reduction, 3),
                "expected_waste_reduction_kg": round(expected_waste_reduction_kg, 4),
                "existing_carbon_kgco2e": existing_carbon,
                "expected_carbon_saving_kgco2e": round(expected_carbon_saving, 4),
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
    total_order_reduction = round(sum(_to_float(row["order_reduction_quantity"]) for row in rows), 3)
    total_waste_reduction = round(sum(_to_float(row["expected_waste_reduction_quantity"]) for row in rows), 3)
    total_waste_reduction_kg = round(sum(_to_float(row["expected_waste_reduction_kg"]) for row in rows), 4)
    total_carbon_saving = round(sum(_to_float(row["expected_carbon_saving_kgco2e"]) for row in rows), 4)
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
        "total_order_reduction_quantity": total_order_reduction,
        "total_expected_waste_reduction_quantity": total_waste_reduction,
        "total_expected_waste_reduction_kg": total_waste_reduction_kg,
        "total_expected_carbon_saving_kgco2e": total_carbon_saving,
    }


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = [_korean_row(row) for row in rows]
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(output_rows[0]) if output_rows else [])
        writer.writeheader()
        writer.writerows(output_rows)


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# 발주 추천 결과",
            "",
            _human_summary(summary),
            "",
            "## 품목별 추천",
            "",
            _markdown_table(rows),
            "",
            "## 시연 흐름",
            "",
            "```text",
            "시연용 판매/마감 데이터 -> LightGBM 수요 예측 -> 추천 발주량",
            "                                   ↓",
            "                         기존 발주량과 비교",
            "                                   ↓",
            "                         폐기 감소 -> 탄소 절감",
            "```",
            "",
            "## 산식",
            "",
            "- 추천 발주량 = max(0, base-stock level - 현재 가용재고)을 발주단위에 맞춰 올림한 값",
            "- base-stock level = 예측 수요 + 안전재고",
            "- 기존 발주 대비 = 추천 발주량 - 더미 데이터의 기존 발주량",
            "- 예상 폐기 감소 = min(기존 발주량 - 추천 발주량, 기존 폐기량)",
            "- 예상 탄소 절감 = 기존 폐기 탄소량 × 예상 폐기 감소 / 기존 폐기량",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _human_summary(summary: dict[str, Any]) -> str:
    item_filter = summary["item_type_filter"]
    if isinstance(item_filter, list):
        item_filter_text = ", ".join(item_filter)
    else:
        item_filter_text = str(item_filter)
    return "\n".join(
        [
            f"마감일: {summary['business_date']}",
            f"대상: {item_filter_text}",
            f"예측 모델: {summary['forecast_method']}",
            f"발주 정책: {summary['order_method']}",
            f"발주 추천 품목: {summary['ordered_items']} / {summary['items']}개",
            f"예측 수요 합계: {summary['total_forecast_demand']}",
            f"추천 발주량 합계: {summary['total_recommended_quantity']}",
            f"기존 발주량 합계: {summary['total_historical_order_quantity']}",
            f"기존 대비 증감: {summary['total_recommendation_minus_historical']:+}",
            f"발주 감축량 합계: {summary['total_order_reduction_quantity']}",
            f"예상 폐기 감소: {summary['total_expected_waste_reduction_quantity']}개 / {summary['total_expected_waste_reduction_kg']}kg",
            f"예상 탄소 절감: {summary['total_expected_carbon_saving_kgco2e']}kgCO2e",
        ]
    )


def _human_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "품목별 추천 결과가 없습니다."
    lines = [
        "",
        "품목별 추천",
        _markdown_table(rows),
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "품목",
        "현재재고",
        "최근수요",
        "예측수요합계",
        "추천발주",
        "기존발주",
        "증감",
        "예상폐기감소",
        "예상탄소절감",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["itemName"],
            _format_number(row["current_stock"]),
            _format_number(row["latest_demand"]),
            _format_number(row["forecast_total"]),
            _format_number(row["recommended_quantity"]),
            _format_number(row["historical_order_quantity"]),
            _format_signed(row["recommendation_minus_historical"]),
            _format_number(row["expected_waste_reduction_quantity"]),
            f"{_format_number(row['expected_carbon_saving_kgco2e'])}kgCO2e",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _korean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "마감일": row["business_date"],
        "품목": row["itemName"],
        "구분": row["itemType"],
        "단위": row["unit"],
        "최근수요": row["latest_demand"],
        "최근판매": row["latest_sales"],
        "현재재고": row["current_stock"],
        "결품": row["stockout"],
        "예측일수": row["forecast_days"],
        "예측수요합계": row["forecast_total"],
        "기준재고수준": row["base_stock_level"],
        "가용재고": row["projected_position"],
        "추천발주량": row["recommended_quantity"],
        "기존발주량": row["historical_order_quantity"],
        "추천-기존": row["recommendation_minus_historical"],
        "발주감축량": row["order_reduction_quantity"],
        "기존폐기량": row["existing_waste_quantity"],
        "예상폐기감소량": row["expected_waste_reduction_quantity"],
        "예상폐기감소_kg": row["expected_waste_reduction_kg"],
        "기존폐기탄소_kgCO2e": row["existing_carbon_kgco2e"],
        "예상탄소절감_kgCO2e": row["expected_carbon_saving_kgco2e"],
        "추천근거": "예측 수요와 안전재고를 기준으로 현재 가용재고를 보충하도록 계산",
    }


def _format_number(value: Any) -> str:
    number = _to_float(value)
    return str(int(number)) if number.is_integer() else f"{number:.3f}".rstrip("0").rstrip(".")


def _format_signed(value: Any) -> str:
    number = _to_float(value)
    if number == 0:
        return "0"
    formatted = _format_number(abs(number))
    return f"+{formatted}" if number > 0 else f"-{formatted}"


if __name__ == "__main__":
    main()
