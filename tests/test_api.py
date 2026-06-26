from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast() -> None:
    payload = {
        "history": [
            {"sku": "apple", "period": "2026-06-20", "quantity": 10},
            {"sku": "apple", "period": "2026-06-21", "quantity": 11},
            {"sku": "apple", "period": "2026-06-22", "quantity": 12},
        ],
        "horizon": 2,
    }
    with TestClient(app) as client:
        response = client.post("/forecast", json=payload)

    assert response.status_code == 200
    assert len(response.json()["forecasts"]) == 2


def test_forecast_uses_cache_on_repeated_payload() -> None:
    payload = {
        "history": [
            {"sku": "banana", "period": "2026-06-20", "quantity": 8},
            {"sku": "banana", "period": "2026-06-21", "quantity": 9},
        ],
        "horizon": 1,
    }
    with TestClient(app) as client:
        first = client.post("/forecast", json=payload)
        second = client.post("/forecast", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cache"]["exact_hit"] is True


def test_order_recommendation() -> None:
    payload = {
        "inventory": [{"sku": "apple", "on_hand": 5, "lead_time_days": 2, "safety_stock": 3}],
        "forecast": [
            {"sku": "apple", "period": "2026-06-27", "quantity": 4},
            {"sku": "apple", "period": "2026-06-28", "quantity": 4},
        ],
    }
    with TestClient(app) as client:
        response = client.post("/order-recommendation", json=payload)

    assert response.status_code == 200
    assert response.json()["recommendations"][0]["recommended_quantity"] == 6


def test_order_recommendation_ortools_policy() -> None:
    payload = {
        "inventory": [{"sku": "apple", "on_hand": 5, "lead_time_days": 2, "safety_stock": 3, "pack_size": 4}],
        "forecast": [
            {"sku": "apple", "period": "2026-06-27", "quantity": 4},
            {"sku": "apple", "period": "2026-06-28", "quantity": 4},
        ],
        "policy": "ortools",
    }
    with TestClient(app) as client:
        response = client.post("/order-recommendation", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "ortools_base_stock"
    assert body["recommendations"][0]["recommended_quantity"] == 8

