import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import pstdev
from typing import Any

from app.schemas import ForecastPoint, ForecastRequest, InventoryPosition, OrderRecommendationRequest


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INVENTORY_FLOW_PATH = DATA_DIR / "inventory_flow_5days.csv"
ITEM_MASTER_PATH = DATA_DIR / "item_master.csv"
ORDER_POLICY_PATH = DATA_DIR / "order_policy.csv"
VALID_ITEM_ID_PATTERN = re.compile(r"^[PRDS]\d{2}$")
SAFETY_Z = {
    "낮춤": 0.5,
    "중상": 1.28,
    "중": 1.0,
    "높임": 1.65,
}
REQUIRED_INVENTORY_COLUMNS = {
    "날짜",
    "품목",
    "단위",
    "수요",
    "결품",
    "기말재고",
}
REQUIRED_MASTER_COLUMNS = {
    "품목ID",
    "품목명",
    "구분",
    "관리단위",
}
REQUIRED_POLICY_COLUMNS = {
    "품목ID",
    "품목명",
    "예측horizon_T+LT(일)",
    "안전재고_z방향",
    "발주단위(MOQ)",
}


def load_demo_closing_data() -> dict[str, Any]:
    inventory_flow = _read_csv(INVENTORY_FLOW_PATH)
    item_master = _read_csv(ITEM_MASTER_PATH)
    order_policy = [
        row for row in _read_csv(ORDER_POLICY_PATH)
        if VALID_ITEM_ID_PATTERN.match(row.get("품목ID", ""))
    ]
    _validate_columns("inventory_flow_5days.csv", inventory_flow, REQUIRED_INVENTORY_COLUMNS)
    _validate_columns("item_master.csv", item_master, REQUIRED_MASTER_COLUMNS)
    _validate_columns("order_policy.csv", order_policy, REQUIRED_POLICY_COLUMNS)
    _validate_item_links(inventory_flow, item_master, order_policy)
    latest_date = max(row["날짜"] for row in inventory_flow)
    return {
        "store_id": "inha-store-001",
        "business_date": latest_date,
        "data_version": "csv-demo-v1",
        "inventory_flow": inventory_flow,
        "item_master": item_master,
        "order_policy": order_policy,
    }


def build_forecast_request(data: dict[str, Any]) -> ForecastRequest:
    policy_by_name = {row["품목명"]: row for row in data["order_policy"]}
    horizon = max(
        int(float(policy_by_name[row["품목"]]["예측horizon_T+LT(일)"]))
        for row in data["inventory_flow"]
        if row["품목"] in policy_by_name
    )
    points = [
        ForecastPoint(
            sku=row["품목"],
            period=row["날짜"],
            quantity=_to_float(row["수요"]),
        )
        for row in data["inventory_flow"]
    ]
    return ForecastRequest(history=points, horizon=horizon)


def build_order_request(data: dict[str, Any], forecast: list[ForecastPoint]) -> OrderRecommendationRequest:
    latest_rows = _latest_inventory_rows(data["inventory_flow"])
    policy_by_name = {row["품목명"]: row for row in data["order_policy"]}
    demand_by_name = _historical_demand_by_name(data["inventory_flow"])
    inventory = []

    for name, row in latest_rows.items():
        policy = policy_by_name[name]
        horizon_days = int(float(policy["예측horizon_T+LT(일)"]))
        inventory.append(
            InventoryPosition(
                sku=name,
                on_hand=_to_float(row["기말재고"]),
                on_order=0,
                backorder=_to_float(row["결품"]),
                lead_time_days=horizon_days,
                safety_stock=_safety_stock(policy["안전재고_z방향"], demand_by_name[name], horizon_days),
                pack_size=_pack_size(policy["발주단위(MOQ)"], row["단위"]),
            )
        )

    return OrderRecommendationRequest(
        inventory=inventory,
        forecast=forecast,
        policy="base_stock",
    )


def closing_cache_payload(data: dict[str, Any]) -> dict[str, str]:
    return {
        "store_id": data["store_id"],
        "business_date": data["business_date"],
        "data_version": data["data_version"],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _validate_columns(file_name: str, rows: list[dict[str, str]], required: set[str]) -> None:
    if not rows:
        raise ValueError(f"{file_name} has no rows")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{file_name} is missing columns: {', '.join(sorted(missing))}")


def _validate_item_links(
    inventory_flow: list[dict[str, str]],
    item_master: list[dict[str, str]],
    order_policy: list[dict[str, str]],
) -> None:
    inventory_names = {row["품목"] for row in inventory_flow}
    master_names = {row["품목명"] for row in item_master}
    policy_names = {row["품목명"] for row in order_policy}
    missing_master = inventory_names - master_names
    missing_policy = inventory_names - policy_names
    if missing_master or missing_policy:
        details = []
        if missing_master:
            details.append(f"missing item_master rows: {', '.join(sorted(missing_master))}")
        if missing_policy:
            details.append(f"missing order_policy rows: {', '.join(sorted(missing_policy))}")
        raise ValueError("; ".join(details))


def _latest_inventory_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest_date = max(row["날짜"] for row in rows)
    return {
        row["품목"]: row
        for row in rows
        if row["날짜"] == latest_date
    }


def _historical_demand_by_name(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    demand_by_name: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        demand_by_name[row["품목"]].append(_to_float(row["수요"]))
    return demand_by_name


def _safety_stock(direction: str, demand: list[float], horizon_days: int) -> float:
    z = next((value for keyword, value in SAFETY_Z.items() if keyword in direction), 1.0)
    sigma = pstdev(demand) if len(demand) > 1 else 0.0
    return round(z * sigma * (horizon_days ** 0.5), 3)


def _pack_size(moq: str, unit: str) -> float:
    if "1L" in moq and unit == "mL":
        return 1000
    if "1kg" in moq and unit == "g":
        return 1000
    match = re.search(r"\d+(?:\.\d+)?", moq)
    if match:
        return _to_float(match.group())
    return 1


def _to_float(value: str | int | float) -> float:
    if value == "":
        return 0.0
    return float(value)
