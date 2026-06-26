# Net-Zero AI Server

FastAPI server for POS closing-based demand forecasting, order recommendation, and explanation output.

## Stack

- FastAPI: `/forecast`, `/order-recommendation`, `/daily-close`, `/chat`
- Redis: exact response cache for deterministic closing calculations when `REDIS_URL` is available
- sqlite-vec semantic cache reserved for the Bedrock Llama chatbot only
- AWS Bedrock: Meta Llama 3.2 1B Instruct API for recommendation explanations and chatbot answers
- Deterministic engine: LightGBM forecast, OR-Tools/base-stock order policy

## Bedrock model

Default model ID:

```text
meta.llama3-2-1b-instruct-v1:0
```

The AWS Bedrock model card also lists geo inference IDs such as:

```text
us.meta.llama3-2-1b-instruct-v1:0
eu.meta.llama3-2-1b-instruct-v1:0
```

Set `BEDROCK_MODEL_ID` if your region or inference profile needs a different ID.

Required runtime environment for Bedrock calls:

```text
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=meta.llama3-2-1b-instruct-v1:0
```

AWS credentials are resolved through the normal boto3 chain, such as environment variables, AWS CLI profile, or an attached IAM role. The application uses `bedrock-runtime` with Converse API first and falls back to `invoke_model`.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Train Local Demand Model

Put generated inventory-flow CSV files under `app/data/training/`, then train and save the initial LightGBM model:

```bash
.venv/bin/python scripts/train_lightgbm.py --inventory "app/data/training/*.csv"
```

If `app/data/training/` is empty, the script falls back to `app/data/inventory_flow_5days.csv` for a small smoke-test model.

Saved artifacts:

```text
app/models/demand_lgbm.txt
app/models/demand_lgbm_metadata.json
```

The FastAPI server automatically loads these files when they exist. If they are missing, it falls back to the lightweight in-process forecast path.

API docs:

```text
http://127.0.0.1:8000/docs
```

## Example

The deterministic APIs do not require request payloads. They read the demo POS closing CSV files from `app/data/`.

- `inventory_flow_5days.csv`: daily inventory flow, demand, sales, stockout, waste, and ending inventory
- `item_master.csv`: item IDs, item names, units, shelf life, and ESG-related item metadata
- `order_policy.csv`: review period, lead time, forecast horizon, safety-stock direction, and MOQ text

```bash
curl -X POST http://127.0.0.1:8000/daily-close
```

Chatbot explanation endpoint:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"샌드위치 발주가 왜 필요한지 알려줘"}'
```

## API Specification

### `GET /health`

Purpose: service and Bedrock Llama configuration check.

Response:

```json
{
  "status": "ok",
  "llm_provider": "bedrock",
  "bedrock_model_id": "meta.llama3-2-1b-instruct-v1:0"
}
```

### `POST /forecast`

Purpose: run demand forecasting from the local POS closing CSV files. No request body is required.

Data source:

```text
app/data/inventory_flow_5days.csv
app/data/item_master.csv
app/data/order_policy.csv
```

Response fields:

```json
{
  "forecasts": [
    {
      "sku": "치킨 토마토 치즈 샌드위치",
      "period": "2025-06-26",
      "quantity": 15.5
    }
  ],
  "method": "lightgbm_saved_model",
  "cache": {
    "exact_hit": false,
    "semantic_hit": false,
    "semantic_score": null
  }
}
```

Notes:

- Loads `app/models/demand_lgbm.txt` when a trained model exists.
- Falls back to the lightweight in-process forecast path when the saved model is missing.
- Uses exact cache only.

### `POST /order-recommendation`

Purpose: calculate order quantities from forecast output, ending inventory, stockout, safety stock, lead time, and MOQ policy. No request body is required.

Response fields:

```json
{
  "recommendations": [
    {
      "sku": "우유",
      "recommended_quantity": 6000,
      "base_stock_level": 5934.166,
      "projected_position": -80,
      "reason": "Order up to lead-time demand plus safety stock, rounded to pack size."
    }
  ],
  "method": "base_stock",
  "cache": {
    "exact_hit": false,
    "semantic_hit": false,
    "semantic_score": null
  }
}
```

Notes:

- LLM is not used for numeric order calculation.
- Uses exact cache only.

### `POST /daily-close`

Purpose: run the full POS closing flow and generate a store-owner explanation through AWS Bedrock Llama API. No request body is required.

Flow:

```text
CSV closing data
-> saved LightGBM demand forecast
-> base-stock order recommendation
-> Bedrock Llama explanation output
```

Response fields:

```json
{
  "store_id": "inha-store-001",
  "business_date": "2025-06-25",
  "data_version": "csv-demo-v1",
  "forecast": {},
  "order_recommendation": {},
  "llm_output": "2025-06-25 마감 기준 발주 추천입니다...",
  "cache": {
    "exact_hit": false,
    "semantic_hit": false,
    "semantic_score": null
  }
}
```

Notes:

- Uses Bedrock Llama only for explanation text.
- If Bedrock credentials are unavailable, returns a deterministic fallback summary.
- Uses exact cache only.

### `POST /chat`

Purpose: answer store-owner questions about the recommendation using AWS Bedrock Llama API. This endpoint is the only semantic-cache user.

Request:

```json
{
  "question": "샌드위치 발주가 왜 필요한지 알려줘"
}
```

Response:

```json
{
  "answer": "샌드위치는 최근 수요와 결품을 고려했을 때...",
  "cache": {
    "exact_hit": false,
    "semantic_hit": true,
    "semantic_score": 1.0
  }
}
```

Notes:

- The chatbot explains recommendations; it does not change forecast or order quantities.
- Semantic cache stores similar chatbot questions and Bedrock Llama answers to reduce repeated AWS calls.
- Cache namespace includes store ID, business date, and data version.

## Notes

- Redis is optional. If unavailable, the app uses an in-memory exact cache.
- `/forecast`, `/order-recommendation`, and `/daily-close` use exact cache only. Semantic cache is intentionally not used for numeric order calculations.
- `/chat` is the only endpoint that uses semantic cache. It is meant for the AWS Bedrock Llama explanation chatbot, not for changing forecast or order quantities.
- Bedrock is optional for local development. `app.services.llm.BedrockLlamaClient` calls the configured Llama 3.2 1B model through Bedrock Converse API, with an `invoke_model` fallback.
- The vector cache uses sqlite-vec when available and keeps chatbot question embeddings, answer responses, and metadata. It can be swapped for pgvector without changing the chat cache interface.
