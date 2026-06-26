from fastapi.testclient import TestClient
from uuid import uuid4

from app.main import app
from app.services.demo_data import load_demo_closing_data


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"
    assert response.json()["model"]["forecast"] in {"lgbm_global_v1", "baseline_v1"}


def test_forecast() -> None:
    with TestClient(app) as client:
        response = client.post("/forecast")

    assert response.status_code == 200
    body = response.json()
    assert body["method"] in {"lightgbm", "lightgbm_saved_model"}
    assert len(body["forecasts"]) == 144


def test_forecast_uses_exact_cache_without_payload() -> None:
    with TestClient(app) as client:
        first = client.post("/forecast")
        second = client.post("/forecast")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cache"]["exact_hit"] is True
    assert second.json()["cache"]["semantic_hit"] is False


def test_cache_status_tracks_exact_cache() -> None:
    with TestClient(app) as client:
        client.post("/forecast")
        client.post("/forecast")
        response = client.get("/cache-status")

    assert response.status_code == 200
    body = response.json()
    assert body["exact_cache_backend"] in {"memory", "redis", "elasticache_redis"}
    assert body["elasticache_compatible"] is True
    assert body["exact_hits"] >= 1


def test_integration_status_reports_runtime_gaps() -> None:
    with TestClient(app) as client:
        response = client.get("/integration-status")

    assert response.status_code == 200
    body = response.json()
    assert "aws" in body
    assert "actual_bedrock_call_ready" in body["llm"]
    assert body["data_source"]["active"] in {"local", "s3"}
    assert isinstance(body["gaps"], list)


def test_v1_order_recommendation_returns_daily_quantiles() -> None:
    payload = {
        "storeId": 1,
        "targetDate": "2026-06-27",
        "salesHistory": {
            "presignedUrls": ["https://example.invalid/sales.csv"],
            "format": "sales_csv_v1",
        },
        "coverage": {
            "leadTimeDays": 1,
            "orderCycleDays": 2,
            "coverageDays": 3,
        },
        "weather": [
            {
                "forecastDate": "2026-06-28",
                "avgTemp": 21.2,
                "precipitationMm": 12.0,
                "precipitationProb": 80,
                "skyCode": 4,
            }
        ],
        "rows": [
            {
                "itemId": 101,
                "orderCycleDays": 2,
                "leadTimeDays": 1,
                "features": {"dayOfWeek": 6, "isHoliday": False, "ma7": 9.4, "trend": -0.3},
            }
        ],
    }
    with TestClient(app) as client:
        response = client.post("/v1/order-recommendation", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "baseline_v1"
    daily = body["predictions"][0]["daily"]
    assert len(daily) == 3
    assert daily[0]["p10"] <= daily[0]["p50"] <= daily[0]["p90"]


def test_v1_forecast_returns_single_day_quantiles() -> None:
    payload = {
        "storeId": 1,
        "targetDate": "2026-06-28",
        "salesHistory": {
            "presignedUrls": ["https://example.invalid/sales.csv"],
            "format": "sales_csv_v1",
        },
        "weather": {
            "forecastDate": "2026-06-28",
            "avgTemp": 21.2,
            "precipitationMm": 12.0,
            "precipitationProb": 80,
            "skyCode": 4,
        },
        "rows": [
            {"itemId": 101, "features": {"dayOfWeek": 6, "isHoliday": False, "ma7": 9.4, "trend": -0.3}}
        ],
    }
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["targetDate"] == "2026-06-28"
    prediction = body["predictions"][0]
    assert prediction["itemId"] == 101
    assert prediction["p10"] <= prediction["p50"] <= prediction["p90"]


def test_v1_generate_returns_required_metrics_and_cache_hit() -> None:
    question = f"내일 우유 얼마나 시켜요? {uuid4()}"
    payload = {
        "question": question,
        "locale": "ko",
        "grounding": {
            "item": {"itemId": 101, "itemName": "우유", "unit": "L"},
            "forecast": {"p10": 60, "p50": 80, "p90": 108},
            "recommendation": {"recommendedQuantity": 66},
            "carbon": {"potentialSavingKg": 39.4},
        },
    }
    with TestClient(app) as client:
        first = client.post("/v1/generate", json=payload)
        second = client.post("/v1/generate", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cacheHit"] is False
    assert second.json()["cacheHit"] is True
    assert second.json()["latencyMs"] >= 0
    assert second.json()["tokens"] >= 1


def test_v1_validation_errors_use_contract_shape() -> None:
    payload = {
        "storeId": 1,
        "targetDate": "2026-06-27",
        "salesHistory": {
            "presignedUrls": ["https://example.invalid/sales.csv"],
            "format": "sales_csv_v1",
        },
        "coverage": {
            "leadTimeDays": 1,
            "orderCycleDays": 2,
            "coverageDays": 99,
        },
        "weather": [],
        "rows": [
            {
                "itemId": 101,
                "features": {"dayOfWeek": 6, "isHoliday": False, "ma7": 9.4, "trend": -0.3},
            }
        ],
    }
    with TestClient(app) as client:
        response = client.post("/v1/order-recommendation", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_order_recommendation() -> None:
    with TestClient(app) as client:
        response = client.post("/order-recommendation")

    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 9
    assert {item["sku"] for item in body["recommendations"]} >= {"치킨 토마토 치즈 샌드위치", "우유"}


def test_daily_close_returns_llm_output_without_payload() -> None:
    with TestClient(app) as client:
        response = client.post("/daily-close")

    assert response.status_code == 200
    body = response.json()
    assert body["store_id"] == "inha-store-001"
    assert body["llm_output"]
    assert body["business_date"] == "2025-06-25"
    assert len(body["order_recommendation"]["recommendations"]) == 9


def test_chat_is_the_only_semantic_cache_user() -> None:
    payload = {"question": "샌드위치 발주가 왜 필요한지 알려줘"}
    with TestClient(app) as client:
        first = client.post("/chat", json=payload)
        second = client.post("/chat", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cache"]["semantic_hit"] is True
    assert second.json()["sources"]


def test_csv_demo_data_columns_and_links_are_valid() -> None:
    data = load_demo_closing_data()

    assert len(data["inventory_flow"]) == 45
    assert len(data["item_master"]) == 28
    assert len(data["order_policy"]) == 28
