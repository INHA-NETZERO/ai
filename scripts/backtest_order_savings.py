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
from scripts.recommend_orders_from_csv import _build_output_rows, _human_table, _korean_row, _markdown_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest daily order recommendations and waste/carbon savings.")
    parser.add_argument("--inventory-flow", default="app/data/Test/inventory_demo_1y.csv")
    parser.add_argument("--item-master", default=str(ITEM_MASTER_PATH))
    parser.add_argument("--order-policy", default=str(ORDER_POLICY_PATH))
    parser.add_argument("--policy", choices=["base_stock", "ortools"], default="base_stock")
    parser.add_argument("--item-type", action="append", default=[], help="Optional item type filter. Omit to include all types.")
    parser.add_argument("--min-history-days", type=int, default=14)
    parser.add_argument("--output", default="app/data/predictions/backtest_order_savings.csv")
    parser.add_argument("--report", help="Optional Markdown report path. Defaults to <output stem>_report.md.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    item_types = _unique(args.item_type)
    inventory_flow = _read_csv(Path(args.inventory_flow))
    item_master = _read_csv(Path(args.item_master))
    order_policy = _read_csv(Path(args.order_policy))
    if item_types:
        allowed_types = set(item_types)
        inventory_flow = [row for row in inventory_flow if row.get("구분") in allowed_types]
        item_master = [row for row in item_master if row.get("구분") in allowed_types]
        order_policy = [row for row in order_policy if row.get("구분") in allowed_types]
    if not inventory_flow:
        raise SystemExit("No inventory rows to backtest.")

    rows = run_backtest(
        inventory_flow=inventory_flow,
        item_master=item_master,
        order_policy=order_policy,
        policy=args.policy,
        min_history_days=args.min_history_days,
    )
    if not rows:
        raise SystemExit("No backtest rows were produced. Lower --min-history-days or check data.")

    summary = _summary(rows, item_types, args.policy, args.min_history_days)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(f"{output_path.stem}_report.md")
    _write_output(output_path, rows)
    _write_report(report_path, summary, rows)

    print(_backtest_human_summary(summary))
    print(f"상세 CSV 저장: {output_path}")
    print(f"요약 리포트 저장: {report_path}")
    print(_human_table(rows[: max(0, args.limit)]))


def run_backtest(
    inventory_flow: list[dict[str, str]],
    item_master: list[dict[str, str]],
    order_policy: list[dict[str, str]],
    policy: str,
    min_history_days: int,
) -> list[dict[str, Any]]:
    dates = sorted({row["날짜"] for row in inventory_flow})
    output: list[dict[str, Any]] = []

    for index, business_date in enumerate(dates):
        if index + 1 < min_history_days:
            continue

        history_rows = [row for row in inventory_flow if row["날짜"] <= business_date]
        data = {
            "store_id": "local-backtest",
            "business_date": business_date,
            "data_version": f"backtest:{business_date}",
            "inventory_flow": history_rows,
            "item_master": item_master,
            "order_policy": order_policy,
        }
        try:
            forecast = forecast_demand_from_closing_data(data)
            request = build_order_request(data, forecast.forecasts)
            request.policy = policy
            response = recommend_orders(request)
        except Exception:
            continue

        output.extend(_build_output_rows(history_rows, forecast.forecasts, response.model_dump()["recommendations"]))

    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _summary(rows: list[dict[str, Any]], item_type_filter: list[str], order_method: str, min_history_days: int) -> dict[str, Any]:
    dates = sorted({row["business_date"] for row in rows})
    total_recommended = round(sum(_to_float(row["recommended_quantity"]) for row in rows), 3)
    total_historical = round(sum(_to_float(row["historical_order_quantity"]) for row in rows), 3)
    total_forecast = round(sum(_to_float(row["forecast_total"]) for row in rows), 3)
    total_order_reduction = round(sum(_to_float(row["order_reduction_quantity"]) for row in rows), 3)
    total_waste_reduction = round(sum(_to_float(row["expected_waste_reduction_quantity"]) for row in rows), 3)
    total_waste_reduction_kg = round(sum(_to_float(row["expected_waste_reduction_kg"]) for row in rows), 4)
    total_carbon_saving = round(sum(_to_float(row["expected_carbon_saving_kgco2e"]) for row in rows), 4)
    ordered_items = sum(1 for row in rows if _to_float(row["recommended_quantity"]) > 0)
    return {
        "business_date": f"{dates[0]} ~ {dates[-1]}",
        "item_type_filter": item_type_filter or "all",
        "forecast_method": "lightgbm_saved_model",
        "order_method": order_method,
        "items": len(rows),
        "distinct_items": len({row["itemName"] for row in rows}),
        "ordered_items": ordered_items,
        "backtest_days": len(dates),
        "min_history_days": min_history_days,
        "total_forecast_demand": total_forecast,
        "total_recommended_quantity": total_recommended,
        "total_historical_order_quantity": total_historical,
        "total_recommendation_minus_historical": round(total_recommended - total_historical, 3),
        "total_order_reduction_quantity": total_order_reduction,
        "total_expected_waste_reduction_quantity": total_waste_reduction,
        "total_expected_waste_reduction_kg": total_waste_reduction_kg,
        "total_expected_carbon_saving_kgco2e": total_carbon_saving,
    }


def _backtest_human_summary(summary: dict[str, Any]) -> str:
    item_filter = summary["item_type_filter"]
    item_filter_text = ", ".join(item_filter) if isinstance(item_filter, list) else str(item_filter)
    return "\n".join(
        [
            f"백테스트 기간: {summary['business_date']}",
            f"대상: {item_filter_text}",
            f"백테스트 일수: {summary['backtest_days']}일",
            f"초기 히스토리 제외: {summary['min_history_days']}일",
            f"대상 품목 수: {summary['distinct_items']}개",
            f"예측 모델: {summary['forecast_method']}",
            f"발주 정책: {summary['order_method']}",
            f"발주 추천 건수: {summary['ordered_items']} / {summary['items']}건",
            f"예측 수요 합계: {summary['total_forecast_demand']}",
            f"추천 발주량 합계: {summary['total_recommended_quantity']}",
            f"기존 발주량 합계: {summary['total_historical_order_quantity']}",
            f"기존 대비 증감: {summary['total_recommendation_minus_historical']:+}",
            f"발주 감축량 합계: {summary['total_order_reduction_quantity']}",
            f"예상 폐기 감소: {summary['total_expected_waste_reduction_quantity']}개 / {summary['total_expected_waste_reduction_kg']}kg",
            f"예상 탄소 절감: {summary['total_expected_carbon_saving_kgco2e']}kgCO2e",
        ]
    )


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
    top_rows = sorted(rows, key=lambda row: _to_float(row["expected_carbon_saving_kgco2e"]), reverse=True)[:10]
    content = "\n".join(
        [
            "# 1년치 발주 추천 백테스트 결과",
            "",
            _backtest_human_summary(summary),
            "",
            "## 탄소 절감 상위 품목/일자",
            "",
            _markdown_table(top_rows),
            "",
            "## 시연 흐름",
            "",
            "```text",
            "시연용 1년치 마감 데이터 -> 날짜별 LightGBM 수요 예측 -> 추천 발주량",
            "                                             ↓",
            "                                  기존 발주량과 비교",
            "                                             ↓",
            "                                  폐기 감소 -> 탄소 절감 누적",
            "```",
            "",
            "## 보수적 절감 산식",
            "",
            "- 예상 폐기 감소 = min(기존 발주량 - 추천 발주량, 기존 폐기량)",
            "- 예상 탄소 절감 = 기존 폐기 탄소량 × 예상 폐기 감소 / 기존 폐기량",
            "- 추천 발주량이 기존 발주량보다 큰 날은 절감량 0으로 계산",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


if __name__ == "__main__":
    main()
