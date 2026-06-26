from fastapi.testclient import TestClient

from app.main import app
from app.services.demo_data import load_demo_closing_data


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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
