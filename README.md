# Net-Zero AI Server

FastAPI server for POS closing-based demand forecasting, order recommendation, and explanation output.

## Stack

- FastAPI: `/forecast`, `/order-recommendation`, `/daily-close`, `/chat`
- Redis: exact response cache for deterministic closing calculations when `REDIS_URL` is available
- sqlite-vec semantic cache reserved for the low-power chatbot only
- Amazon Bedrock: Meta Llama 3.2 1B Instruct client for recommendation explanations
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

## Notes

- Redis is optional. If unavailable, the app uses an in-memory exact cache.
- `/forecast`, `/order-recommendation`, and `/daily-close` use exact cache only. Semantic cache is intentionally not used for numeric order calculations.
- `/chat` is the only endpoint that uses semantic cache. It is meant for a small low-power explanation chatbot, not for changing forecast or order quantities.
- Bedrock is optional for local development. `app.services.llm.BedrockLlamaClient` calls the configured Llama 3.2 1B model through Bedrock Converse API, with an `invoke_model` fallback.
- The vector cache uses sqlite-vec when available and keeps chatbot question embeddings, answer responses, and metadata. It can be swapped for pgvector without changing the chat cache interface.
