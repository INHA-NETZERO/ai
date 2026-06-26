# Net-Zero AI Server

FastAPI server for demand forecasting, order recommendation, and carbon estimation.

## Stack

- FastAPI: `/forecast`, `/order-recommendation`, `/carbon-estimate`
- Redis: exact response cache when `REDIS_URL` is available
- sqlite-vec vector cache: cached input embeddings and semantic cache metadata, with plain SQLite fallback
- Amazon Bedrock: Meta Llama 3.2 1B Instruct client for LLM calls
- Deterministic engine: LightGBM forecast, OR-Tools/base-stock order policy, carbon lookup table

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

API docs:

```text
http://127.0.0.1:8000/docs
```

## Example

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"history":[{"sku":"apple","period":"2026-06-20","quantity":10},{"sku":"apple","period":"2026-06-21","quantity":12}],"horizon":3}'
```

## Notes

- Redis is optional. If unavailable, the app uses an in-memory exact cache.
- Bedrock is optional for local development. `app.services.llm.BedrockLlamaClient` calls the configured Llama 3.2 1B model through Bedrock Converse API, with an `invoke_model` fallback.
- The vector cache uses sqlite-vec when available and keeps embeddings, input hashes, responses, and metadata. It can be swapped for pgvector without changing endpoint contracts.
