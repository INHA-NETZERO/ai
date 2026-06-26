#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.demand_model import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    TARGET_COLUMN,
    TRAINING_COLUMNS,
    _evaluate_model,
    _training_row_features,
    forecast_with_saved_model,
)
from app.services.demo_data import _to_float


CLOSING_REQUIRED_COLUMNS = {"날짜", "품목", "구분"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local predictions with the saved LightGBM demand model.")
    parser.add_argument("--input", required=True, help="New CSV file to predict or evaluate.")
    parser.add_argument(
        "--mode",
        choices=["auto", "sales", "closing"],
        default="auto",
        help="sales: labeled sales CSV with 수요 target, closing: POS closing/inventory CSV, auto: infer from header.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--output", help="Optional CSV or JSON output path for full predictions.")
    parser.add_argument("--limit", type=int, default=20, help="Preview rows to print when --output is omitted.")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    metadata_path = Path(args.metadata)
    _require_file(input_path, "input CSV")
    _require_file(model_path, "saved LightGBM model")
    _require_file(metadata_path, "saved model metadata")

    fieldnames, rows = _read_csv(input_path)
    if not rows:
        raise SystemExit(f"{input_path} has no rows.")

    mode = _resolve_mode(args.mode, fieldnames)
    if mode == "sales" and fieldnames != TRAINING_COLUMNS:
        raise SystemExit(
            "sales evaluation CSV must use columns in this exact order: "
            + ",".join(TRAINING_COLUMNS)
        )
    if mode == "sales":
        result = _evaluate_sales_rows(rows, model_path, metadata_path)
    else:
        result = _forecast_closing_rows(rows, model_path, metadata_path)

    if args.output:
        _write_output(Path(args.output), result["rows"])
        print(f"wrote {len(result['rows'])} prediction rows: {args.output}")

    print(result["summary"])
    for row in result["rows"][: max(0, args.limit)]:
        print(json.dumps(row, ensure_ascii=False))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames or []), list(reader)


def _resolve_mode(mode: str, fieldnames: list[str]) -> str:
    if mode != "auto":
        return mode
    if fieldnames == TRAINING_COLUMNS:
        return "sales"
    if CLOSING_REQUIRED_COLUMNS <= set(fieldnames):
        return "closing"
    raise SystemExit(
        "Cannot infer CSV mode. Use --mode sales for labeled training-schema CSV "
        "or --mode closing for POS closing/inventory CSV."
    )


def _evaluate_sales_rows(rows: list[dict[str, str]], model_path: Path, metadata_path: Path) -> dict[str, Any]:
    import lightgbm as lgb

    metadata = _load_metadata(metadata_path)
    model = lgb.Booster(model_file=str(model_path))
    sorted_rows = sorted(rows, key=lambda row: (row["날짜"], row.get("_source_file", ""), row["품목"]))
    features_list = []
    previous_by_item: dict[str, dict[str, str]] = {}
    for row in sorted_rows:
        features_list.append(_training_row_features(row, metadata, previous_by_item.get(row["품목"])))
        previous_by_item[row["품목"]] = row
    features = np.array(features_list, dtype=np.float64)
    actual = np.array([_to_float(row[TARGET_COLUMN]) for row in sorted_rows], dtype=np.float64)
    predictions = np.maximum(0.0, model.predict(features))
    metrics = _evaluate_model(model, features, actual)

    output_rows = []
    for row, predicted, actual_value in zip(sorted_rows, predictions, actual, strict=True):
        error = float(predicted - actual_value)
        ape = abs(error / actual_value) * 100 if actual_value else 0.0
        output_rows.append(
            {
                "날짜": row["날짜"],
                "품목": row["품목"],
                "구분": row["구분"],
                "actual_demand": round(float(actual_value), 3),
                "predicted_demand": round(float(predicted), 3),
                "error": round(error, 3),
                "abs_error": round(abs(error), 3),
                "ape": round(ape, 3),
            }
        )

    return {
        "summary": (
            "sales evaluation: "
            f"rows={len(rows)}, MAE={metrics['mae']}, RMSE={metrics['rmse']}, MAPE={metrics['mape']}%"
        ),
        "rows": output_rows,
    }


def _forecast_closing_rows(rows: list[dict[str, str]], model_path: Path, metadata_path: Path) -> dict[str, Any]:
    missing = CLOSING_REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise SystemExit(f"closing CSV is missing columns: {', '.join(sorted(missing))}")

    try:
        response = forecast_with_saved_model(
            {"inventory_flow": rows},
            model_path=model_path,
            metadata_path=metadata_path,
        )
    except ValueError as exc:
        raise SystemExit(f"Could not forecast closing CSV: {exc}") from exc
    if response is None:
        raise SystemExit("Saved model could not be loaded. Train a model first with scripts/train_lightgbm.py.")

    output_rows = [
        {
            "sku": point.sku,
            "period": point.period,
            "predicted_quantity": point.quantity,
            "method": response.method,
        }
        for point in response.forecasts
    ]
    return {
        "summary": f"closing forecast: rows={len(output_rows)}, method={response.method}",
        "rows": output_rows,
    }


def _load_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise SystemExit("Saved model metadata feature schema does not match current code. Retrain the model first.")
    return metadata


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


if __name__ == "__main__":
    main()
