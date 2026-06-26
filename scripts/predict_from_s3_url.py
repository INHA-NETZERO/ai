#!/usr/bin/env python
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.api_contracts import WeatherDay
from app.services.demo_data import _to_float
from app.services.v1_contract import (
    BASELINE_MODEL_VERSION,
    MODEL_VERSION,
    SALES_CSV_COLUMNS_V1,
    _fallback_lag_row,
    _latest_history_by_item,
    _load_lgbm_bundle,
    _next_lag_row,
    _predict_lgbm_quantity,
    _quantiles_from_point,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sales CSV from S3 presigned URL and run saved LightGBM forecasts."
    )
    parser.add_argument("--url", action="append", default=[], help="S3 presigned sales CSV URL. Repeatable.")
    parser.add_argument("--input", action="append", default=[], help="Local sales CSV path for dry-run/testing. Repeatable.")
    parser.add_argument("--target-date", help="First forecast date. Defaults to latest CSV date + 1 day.")
    parser.add_argument("--days", type=int, default=1, help="Number of forecast days to produce.")
    parser.add_argument("--avg-temp", type=float, default=20.0)
    parser.add_argument("--precipitation-mm", type=float, default=0.0)
    parser.add_argument("--precipitation-prob", type=int, default=0)
    parser.add_argument("--sky-code", type=int, default=1)
    parser.add_argument("--holiday", action="store_true", help="Mark forecast target day(s) as holiday.")
    parser.add_argument(
        "--output",
        default="app/data/predictions/s3_url_forecast.json",
        help="Output .json or .csv path.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Preview rows to print.")
    args = parser.parse_args()

    if not args.url and not args.input:
        raise SystemExit("Pass at least one --url '<presigned-url>' or --input local_sales.csv")
    if args.days < 1:
        raise SystemExit("--days must be >= 1")

    rows = _load_sales_rows(args.url, args.input)
    if not rows:
        raise SystemExit("No sales rows were loaded from the given input.")

    result_rows = forecast_rows(
        rows,
        target_date=args.target_date,
        days=args.days,
        avg_temp=args.avg_temp,
        precipitation_mm=args.precipitation_mm,
        precipitation_prob=args.precipitation_prob,
        sky_code=args.sky_code,
        holiday=args.holiday,
    )
    _write_output(Path(args.output), result_rows)

    print(f"loaded_rows={len(rows)}")
    print(f"wrote {len(result_rows)} forecast rows: {args.output}")
    for row in result_rows[: max(0, args.limit)]:
        print(json.dumps(row, ensure_ascii=False))


def forecast_rows(
    rows: list[dict[str, str]],
    target_date: str | None,
    days: int,
    avg_temp: float,
    precipitation_mm: float,
    precipitation_prob: int,
    sky_code: int,
    holiday: bool,
) -> list[dict[str, Any]]:
    model_bundle = _load_lgbm_bundle()
    if model_bundle is None:
        raise SystemExit("Saved LightGBM model is missing or schema-mismatched. Run scripts/train_lightgbm.py first.")

    latest_date = max(date.fromisoformat(row["날짜"]) for row in rows)
    start = date.fromisoformat(target_date) if target_date else latest_date + timedelta(days=1)
    latest_by_item = _latest_history_by_item(rows)
    stats_by_item = _item_stats(rows)
    output: list[dict[str, Any]] = []

    for item_name in sorted(latest_by_item):
        latest = latest_by_item[item_name]
        item_type = latest.get("구분") or _metadata_item_type(item_name, model_bundle["metadata"])
        row_context = _row_context(
            item_name=item_name,
            item_type=item_type,
            item_id=len(output) + 1,
            stats=stats_by_item[item_name],
            holiday=holiday,
        )
        current = dict(latest or _fallback_lag_row(row_context, model_bundle["metadata"]))

        for offset in range(days):
            forecast_day = start + timedelta(days=offset)
            weather = _weather(
                forecast_day=forecast_day,
                avg_temp=avg_temp,
                precipitation_mm=precipitation_mm,
                precipitation_prob=precipitation_prob,
                sky_code=sky_code,
            )
            p50 = _predict_lgbm_quantity(row_context, forecast_day, weather, current, model_bundle)
            quantile = _quantiles_from_point(p50)
            output.append(
                {
                    "modelVersion": MODEL_VERSION,
                    "date": forecast_day.isoformat(),
                    "itemName": item_name,
                    "itemType": item_type,
                    "p10": quantile.p10,
                    "p50": quantile.p50,
                    "p90": quantile.p90,
                }
            )
            current = _next_lag_row(row_context, forecast_day, p50, weather, model_bundle["metadata"])

    return output


def _load_sales_rows(urls: list[str], input_paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in input_paths:
        rows.extend(_read_sales_csv(Path(path).read_text(encoding="utf-8-sig")))
    for url in urls:
        try:
            response = httpx.get(url, timeout=20.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SystemExit(
                f"S3 presigned URL reached S3 but failed with HTTP {exc.response.status_code}. "
                "Check whether the URL expired."
            ) from exc
        except httpx.HTTPError as exc:
            raise SystemExit("Could not download S3 presigned URL. Check network and URL.") from exc
        rows.extend(_read_sales_csv(response.text))
    return rows


def _read_sales_csv(content: str) -> list[dict[str, str]]:
    reader = csv.DictReader(content.lstrip("\ufeff").splitlines())
    fieldnames = list(reader.fieldnames or [])
    if fieldnames != SALES_CSV_COLUMNS_V1:
        raise SystemExit(
            "Sales CSV must use columns in this exact order: "
            + ",".join(SALES_CSV_COLUMNS_V1)
        )
    return list(reader)


def _item_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda value: (value["품목"], value["날짜"])):
        grouped[row["품목"]].append(row)

    stats = {}
    for item_name, item_rows in grouped.items():
        demands = [_to_float(row["수요"]) for row in item_rows]
        recent = demands[-7:]
        previous = demands[-14:-7]
        ma7 = sum(recent) / len(recent) if recent else 0.0
        prev_ma7 = sum(previous) / len(previous) if previous else ma7
        stats[item_name] = {
            "ma7": round(ma7, 3),
            "trend": round((ma7 - prev_ma7) / 7.0, 3),
        }
    return stats


def _row_context(
    item_name: str,
    item_type: str,
    item_id: int,
    stats: dict[str, float],
    holiday: bool,
) -> SimpleNamespace:
    features = SimpleNamespace(
        day_of_week=0,
        is_holiday=holiday,
        ma7=stats["ma7"],
        trend=stats["trend"],
    )
    return SimpleNamespace(
        item_id=item_id,
        item_name=item_name,
        item_type=item_type,
        features=features,
    )


def _metadata_item_type(item_name: str, metadata: dict[str, Any]) -> str:
    return metadata.get("item_by_name", {}).get(item_name, {}).get("type", "완제품")


def _weather(
    forecast_day: date,
    avg_temp: float,
    precipitation_mm: float,
    precipitation_prob: int,
    sky_code: int,
) -> WeatherDay:
    return WeatherDay.model_validate(
        {
            "forecastDate": forecast_day.isoformat(),
            "avgTemp": avg_temp,
            "precipitationMm": precipitation_mm,
            "precipitationProb": precipitation_prob,
            "skyCode": sky_code,
        }
    )


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        return
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
