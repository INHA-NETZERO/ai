from fastapi.testclient import TestClient
from io import BytesIO
from uuid import uuid4

from app.core.config import Settings
from app.main import app
from app.services.aws_clients import create_aws_session
from app.services.demand_model import DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH
from app.services.demo_data import (
    INVENTORY_FLOW_PATH,
    ITEM_MASTER_PATH,
    ORDER_POLICY_PATH,
    load_closing_data,
    load_demo_closing_data,
)


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


def test_v1_forecast_uses_lightgbm_when_item_name_is_available() -> None:
    payload = {
        "storeId": 1,
        "targetDate": "2026-01-01",
        "salesHistory": {
            "presignedUrls": ["https://example.invalid/sales.csv"],
            "format": "sales_csv_v1",
        },
        "weather": {
            "forecastDate": "2026-01-01",
            "avgTemp": 1.2,
            "precipitationMm": 0.0,
            "precipitationProb": 10,
            "skyCode": 1,
        },
        "rows": [
            {
                "itemId": 101,
                "itemName": "아메리카노",
                "itemType": "판매음료",
                "features": {"dayOfWeek": 3, "isHoliday": False, "ma7": 40.0, "trend": 0.0},
            }
        ],
    }
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    expected_model = "lgbm_global_v1" if DEFAULT_MODEL_PATH.exists() and DEFAULT_METADATA_PATH.exists() else "baseline_v1"
    assert body["modelVersion"] == expected_model
    prediction = body["predictions"][0]
    assert prediction["p10"] <= prediction["p50"] <= prediction["p90"]


def test_v1_forecast_downloads_backend_presigned_url(monkeypatch) -> None:
    csv_text = (
        "날짜,요일,날씨,기온,강수mm,행사중여부,공휴일여부,신메뉴여부,품목,구분,수요,판매수량,매진여부,매진시각,비고_시나리오\n"
        "2026-06-01,월,맑음,22.0,0,False,False,False,우유,완제품,10,10,False,,normal\n"
    )
    calls = []

    class FakeHttpResponse:
        text = csv_text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: float):
        calls.append((url, timeout))
        return FakeHttpResponse()

    monkeypatch.setattr("app.services.v1_contract.httpx.get", fake_get)
    payload = {
        "storeId": 1,
        "targetDate": "2026-06-28",
        "salesHistory": {
            "presignedUrls": ["https://bucket.s3.ap-northeast-2.amazonaws.com/sales.csv?X-Amz-Signature=test"],
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
    assert calls == [("https://bucket.s3.ap-northeast-2.amazonaws.com/sales.csv?X-Amz-Signature=test", 3.0)]


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
    assert body["business_date"] == "2025-12-31"
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

    assert len(data["inventory_flow"]) == 16434
    assert len(data["item_master"]) == 28
    assert len(data["order_policy"]) == 28


def test_s3_closing_data_loader_uses_configured_keys(monkeypatch) -> None:
    objects = {
        "daily/closing/inventory_flow_5y.csv": INVENTORY_FLOW_PATH.read_bytes(),
        "daily/closing/item_master.csv": ITEM_MASTER_PATH.read_bytes(),
        "daily/closing/order_policy.csv": ORDER_POLICY_PATH.read_bytes(),
    }
    calls = []

    class FakeS3Client:
        def get_object(self, Bucket: str, Key: str):
            calls.append((Bucket, Key))
            return {"Body": BytesIO(objects[Key])}

    monkeypatch.setattr("app.services.demo_data.create_aws_client", lambda *args, **kwargs: FakeS3Client())
    settings = Settings(
        data_source="s3",
        s3_bucket="zero-wave-demo",
        s3_prefix="daily/closing",
        s3_inventory_flow_key="inventory_flow_5y.csv",
        aws_region="ap-northeast-2",
    )

    data = load_closing_data(settings)

    assert data["store_id"] == "inha-store-001"
    assert data["data_version"].startswith("s3:zero-wave-demo:daily/closing/inventory_flow_5y.csv")
    assert len(data["inventory_flow"]) == 16434
    assert calls == [
        ("zero-wave-demo", "daily/closing/inventory_flow_5y.csv"),
        ("zero-wave-demo", "daily/closing/item_master.csv"),
        ("zero-wave-demo", "daily/closing/order_policy.csv"),
    ]


def test_aws_session_uses_env_file_credentials(monkeypatch) -> None:
    captured = {}

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.services.aws_clients.boto3.Session", FakeSession)
    settings = Settings(
        aws_region="us-east-1",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
        aws_session_token="token",
    )

    create_aws_session(settings)

    assert captured == {
        "region_name": "us-east-1",
        "aws_access_key_id": "access",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
    }


def test_bedrock_client_uses_bedrock_bearer_token(monkeypatch) -> None:
    from app.services.llm import BedrockLlamaClient

    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("app.services.llm.httpx.post", fake_post)
    client = BedrockLlamaClient(
        Settings(
            _env_file=None,
            aws_region="us-east-1",
            bedrock_model_id="meta.llama3-2-1b-instruct-v1:0",
            aws_bearer_token_bedrock="secret-bearer-token",
        )
    )

    assert client.generate_text("hello", max_tokens=10) == "ok"
    assert captured["url"] == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
        "meta.llama3-2-1b-instruct-v1%3A0/converse"
    )
    assert captured["headers"]["Authorization"] == "Bearer secret-bearer-token"
    assert captured["json"]["inferenceConfig"]["maxTokens"] == 10


def test_bedrock_client_keeps_legacy_api_key_fallback(monkeypatch) -> None:
    from app.services.llm import BedrockLlamaClient

    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

    def fake_post(url, headers, json, timeout):
        captured.update({"headers": headers})
        return FakeResponse()

    monkeypatch.setattr("app.services.llm.httpx.post", fake_post)
    client = BedrockLlamaClient(
        Settings(
            _env_file=None,
            bedrock_api_key="legacy-secret",
        )
    )

    assert client.generate_text("hello", max_tokens=10) == "ok"
    assert captured["headers"]["Authorization"] == "Bearer legacy-secret"
