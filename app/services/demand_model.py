import csv
import glob
import json
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from app.schemas import ForecastPoint, ForecastResponse
from app.services.demo_data import (
    VALID_ITEM_ID_PATTERN,
    _read_csv,
    _to_float,
)


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "demand_lgbm.txt"
DEFAULT_METADATA_PATH = MODEL_DIR / "demand_lgbm_metadata.json"
FEATURE_NAMES = [
    "day_index",
    "month",
    "day_of_month",
    "weekday_code",
    "is_weekend",
    "weather_code",
    "temperature",
    "rain_mm",
    "event_flag",
    "holiday_flag",
    "new_menu_flag",
    "item_code",
    "type_code",
    "scenario_code",
    "shelf_life_days",
    "lead_time_days",
    "review_period_days",
    "horizon_days",
    "safety_z",
    "unit_ef",
]
SAFETY_Z = {
    "낮춤": 0.5,
    "중상": 1.28,
    "중": 1.0,
    "높임": 1.65,
}
TRAINING_COLUMNS = [
    "날짜",
    "요일",
    "날씨",
    "기온",
    "강수mm",
    "행사",
    "공휴일",
    "신메뉴",
    "품목",
    "구분",
    "판매수량",
    "비고_시나리오",
]


def train_lightgbm_model(
    training_paths: list[Path],
    item_master_path: Path,
    order_policy_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    item_master = _read_csv(item_master_path)
    order_policy = [
        row for row in _read_csv(order_policy_path)
        if VALID_ITEM_ID_PATTERN.match(row.get("품목ID", ""))
    ]
    metadata = _build_metadata(item_master, order_policy)
    metadata = _augment_metadata_from_training_paths(training_paths, metadata)
    features, targets = _build_training_matrix_from_paths(training_paths, metadata)
    if len(targets) < 10:
        raise ValueError("Need at least 10 training examples to train the saved demand model")

    import lightgbm as lgb

    train_features, train_targets, valid_features, valid_targets = _train_validation_split(features, targets)
    dataset = lgb.Dataset(train_features, label=train_targets, feature_name=FEATURE_NAMES, free_raw_data=False)
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 3,
            "feature_pre_filter": False,
            "verbosity": -1,
            "seed": 42,
        },
        dataset,
        num_boost_round=120,
    )
    evaluation = _evaluate_model(model, valid_features, valid_targets)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    metadata["feature_names"] = FEATURE_NAMES
    metadata["training_examples"] = int(len(targets))
    metadata["train_examples"] = int(len(train_targets))
    metadata["validation_examples"] = int(len(valid_targets))
    metadata["evaluation"] = evaluation
    metadata["training_columns"] = TRAINING_COLUMNS
    metadata["training_files"] = [str(path) for path in training_paths]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "training_examples": int(len(targets)),
        "evaluation": evaluation,
        "features": len(FEATURE_NAMES),
    }


def forecast_with_saved_model(
    data: dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> ForecastResponse | None:
    if not model_path.exists() or not metadata_path.exists():
        return None

    import lightgbm as lgb

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != FEATURE_NAMES:
        return None
    model = lgb.Booster(model_file=str(model_path))
    policy_by_name = metadata["policy_by_name"]
    latest_by_item = _latest_rows_by_item(data["inventory_flow"])
    horizon = max(
        int(float(policy_by_name[item]["horizon_days"]))
        for item in latest_by_item
        if item in policy_by_name
    )

    forecasts: list[ForecastPoint] = []
    for item, latest in latest_by_item.items():
        if item not in policy_by_name:
            continue
        current = dict(latest)
        start = date.fromisoformat(latest["날짜"]) + timedelta(days=1)

        for offset in range(horizon):
            feature = _inference_features(current, start + timedelta(days=offset), metadata)
            prediction = max(0.0, float(model.predict(np.array([feature]))[0]))
            forecasts.append(
                ForecastPoint(
                    sku=item,
                    period=(start + timedelta(days=offset)).isoformat(),
                    quantity=round(prediction, 3),
                )
            )
            current = _synthetic_next_row(current, prediction, start + timedelta(days=offset))

    return ForecastResponse(forecasts=forecasts, method="lightgbm_saved_model")


def inventory_paths_from_glob(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    return sorted({path for path in paths if path.exists()})


def _load_inventory_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            _validate_training_columns(path, reader.fieldnames)
            for row in reader:
                row["_source_file"] = path.name
                rows.append(row)
    return rows


def _build_training_matrix_from_paths(paths: list[Path], metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    feature_chunks = []
    target_chunks = []
    for path in paths:
        features, targets = _build_training_matrix(_load_inventory_rows([path]), metadata)
        if len(targets):
            feature_chunks.append(features)
            target_chunks.append(targets)
    if not target_chunks:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,))
    return np.vstack(feature_chunks), np.concatenate(target_chunks)


def _augment_metadata_from_training_paths(paths: list[Path], metadata: dict[str, Any]) -> dict[str, Any]:
    weekdays = set()
    weather_values = set()
    scenario_values = set()
    item_values = set(metadata["item_codes"])
    type_values = set(metadata["type_codes"])
    temperatures = []
    rain_values = []

    for path in paths:
        for row in _load_inventory_rows([path]):
            weekdays.add(row["요일"])
            weather_values.add(row["날씨"])
            scenario_values.add(row["비고_시나리오"])
            item_values.add(row["품목"])
            type_values.add(row["구분"])
            temperatures.append(_to_float(row["기온"]))
            rain_values.append(_to_float(row["강수mm"]))

    metadata["weekday_codes"] = _category_codes(weekdays)
    metadata["weather_codes"] = _category_codes(weather_values)
    metadata["scenario_codes"] = _category_codes(scenario_values)
    metadata["item_codes"] = _category_codes(item_values)
    metadata["type_codes"] = _category_codes(type_values)
    metadata["defaults"] = {
        "요일": "월",
        "날씨": _mode(weather_values, "맑음"),
        "기온": round(float(mean(temperatures)), 3) if temperatures else 20.0,
        "강수mm": round(float(mean(rain_values)), 3) if rain_values else 0.0,
        "행사": "N",
        "공휴일": "N",
        "신메뉴": "N",
        "비고_시나리오": _mode(scenario_values, "normal"),
    }
    return metadata


def _build_metadata(item_master: list[dict[str, str]], order_policy: list[dict[str, str]]) -> dict[str, Any]:
    item_by_name = {row["품목명"]: row for row in item_master}
    policy_by_name = {
        row["품목명"]: {
            "review_period_days": _to_float_or_default(row["발주주기_T(일)"], 1),
            "lead_time_days": _to_float_or_default(row["리드타임_LT(일)"], 1),
            "horizon_days": _to_float_or_default(row["예측horizon_T+LT(일)"], 1),
            "safety_z": _safety_z(row["안전재고_z방향"]),
        }
        for row in order_policy
    }
    names = sorted(set(item_by_name) | set(policy_by_name))
    return {
        "item_codes": {name: idx for idx, name in enumerate(names)},
        "type_codes": _category_codes(row["구분"] for row in item_master),
        "unit_codes": _category_codes(row["관리단위"] for row in item_master),
        "item_by_name": {
            name: {
                "type": row["구분"],
                "unit": row["관리단위"],
                "shelf_life_days": _to_float_or_default(row["유통기한_일"], 0),
                "unit_ef": _to_float_or_default(row["단위당_EF(kgCO2e)"], 0),
            }
            for name, row in item_by_name.items()
        },
        "policy_by_name": policy_by_name,
    }


def _build_training_matrix(rows: list[dict[str, str]], metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for row in rows:
        features.append(_training_row_features(row, metadata))
        targets.append(_to_float(row["판매수량"]))
    return np.array(features, dtype=np.float64), np.array(targets, dtype=np.float64)


def _train_validation_split(
    features: np.ndarray,
    targets: np.ndarray,
    validation_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validation_size = max(1, int(len(targets) * validation_ratio))
    if len(targets) - validation_size < 5:
        validation_size = max(1, len(targets) - 5)
    split_at = len(targets) - validation_size
    return (
        features[:split_at],
        targets[:split_at],
        features[split_at:],
        targets[split_at:],
    )


def _evaluate_model(model: Any, features: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    if len(targets) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    predictions = np.maximum(0.0, model.predict(features))
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    nonzero = targets != 0
    mape = float(np.mean(np.abs(errors[nonzero] / targets[nonzero])) * 100) if np.any(nonzero) else 0.0
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
    }


def _training_row_features(row: dict[str, Any], metadata: dict[str, Any]) -> list[float]:
    item = row["품목"]
    item_info = metadata["item_by_name"].get(item, {})
    policy = metadata["policy_by_name"].get(item, {})
    day = date.fromisoformat(row["날짜"])
    return [
        float(day.toordinal()),
        float(day.month),
        float(day.day),
        _code(metadata["weekday_codes"], row["요일"]),
        1.0 if day.weekday() >= 5 else 0.0,
        _code(metadata["weather_codes"], row["날씨"]),
        _to_float(row["기온"]),
        _to_float(row["강수mm"]),
        _binary_flag(row["행사"]),
        _binary_flag(row["공휴일"]),
        _binary_flag(row["신메뉴"]),
        _code(metadata["item_codes"], item),
        _code(metadata["type_codes"], item_info.get("type") or row["구분"]),
        _code(metadata["scenario_codes"], row["비고_시나리오"]),
        _to_float(item_info.get("shelf_life_days", 0)),
        _to_float(policy.get("lead_time_days", 1)),
        _to_float(policy.get("review_period_days", 1)),
        _to_float(policy.get("horizon_days", 1)),
        _to_float(policy.get("safety_z", 1)),
        _to_float(item_info.get("unit_ef", 0)),
    ]


def _inference_features(row: dict[str, Any], target_date: date, metadata: dict[str, Any]) -> list[float]:
    defaults = metadata.get("defaults", {})
    training_like_row = {
        "날짜": target_date.isoformat(),
        "요일": _korean_weekday(target_date),
        "날씨": row.get("날씨", defaults.get("날씨", "맑음")),
        "기온": row.get("기온", defaults.get("기온", 20.0)),
        "강수mm": row.get("강수mm", defaults.get("강수mm", 0.0)),
        "행사": row.get("행사", defaults.get("행사", "N")),
        "공휴일": row.get("공휴일", defaults.get("공휴일", "N")),
        "신메뉴": row.get("신메뉴", defaults.get("신메뉴", "N")),
        "품목": row["품목"],
        "구분": row["구분"],
        "비고_시나리오": row.get("비고_시나리오", defaults.get("비고_시나리오", "normal")),
    }
    return _training_row_features(training_like_row, metadata)


def _latest_rows_by_item(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest_date = max(row["날짜"] for row in rows)
    return {row["품목"]: row for row in rows if row["날짜"] == latest_date}


def _synthetic_next_row(row: dict[str, Any], prediction: float, next_date: date) -> dict[str, Any]:
    next_row = dict(row)
    next_row["날짜"] = next_date.isoformat()
    next_row["수요"] = str(prediction)
    next_row["실판매"] = str(prediction)
    next_row["결품"] = "0"
    next_row["폐기"] = "0"
    return next_row


def _validate_training_columns(path: Path, fieldnames: list[str] | None) -> None:
    if fieldnames != TRAINING_COLUMNS:
        raise ValueError(
            f"{path} must use training columns in this exact order: "
            + ",".join(TRAINING_COLUMNS)
        )


def _category_codes(values: Any) -> dict[str, int]:
    return {value: idx for idx, value in enumerate(sorted(set(values)))}


def _safety_z(direction: str) -> float:
    return next((value for keyword, value in SAFETY_Z.items() if keyword in direction), 1.0)


def _to_float_or_default(value: str | int | float, default: float) -> float:
    try:
        return _to_float(value)
    except ValueError:
        return default


def _code(mapping: dict[str, int], value: str) -> float:
    return float(mapping.get(value, -1))


def _binary_flag(value: str | int | float | bool) -> float:
    return 1.0 if str(value).strip().upper() in {"Y", "1", "TRUE", "T"} else 0.0


def _korean_weekday(day: date) -> str:
    return ["월", "화", "수", "목", "금", "토", "일"][day.weekday()]


def _mode(values: set[str], default: str) -> str:
    return sorted(values)[0] if values else default
