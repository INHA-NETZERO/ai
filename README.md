# Net-Zero AI Server

POS 하루 마감 데이터를 기반으로 수요를 예측하고, 재고·결품·발주정책을 반영해 발주량을 추천하는 FastAPI 서버입니다. 점주가 추천 결과를 이해할 수 있도록 AWS Bedrock Llama API를 이용한 설명 출력과 챗봇 기능도 제공합니다.

## 기술 스택

- FastAPI: `/forecast`, `/order-recommendation`, `/daily-close`, `/chat`
- Redis: deterministic 계산 결과용 exact cache. `REDIS_URL`이 없거나 Redis가 꺼져 있으면 인메모리 캐시 사용
- sqlite-vec: Bedrock Llama 챗봇 전용 semantic cache
- AWS Bedrock: Meta Llama 3.2 1B Instruct API로 발주 추천 설명과 챗봇 답변 생성
- 예측/발주 엔진: LightGBM 수요 예측, base-stock/OR-Tools 발주 정책

## Bedrock 모델

기본 모델 ID:

```text
meta.llama3-2-1b-instruct-v1:0
```

AWS Bedrock 모델 카드에는 지역별 inference profile ID도 있습니다.

```text
us.meta.llama3-2-1b-instruct-v1:0
eu.meta.llama3-2-1b-instruct-v1:0
```

리전이나 inference profile에 맞춰야 하면 `BEDROCK_MODEL_ID`를 변경하세요.

Bedrock 호출에 필요한 런타임 설정:

```text
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=meta.llama3-2-1b-instruct-v1:0
```

AWS 인증은 boto3 기본 인증 체인을 따릅니다. 환경변수, AWS CLI profile, IAM Role 등을 사용할 수 있습니다. 서버는 `bedrock-runtime`의 Converse API를 먼저 사용하고, 실패하면 `invoke_model`로 fallback합니다.

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

API 문서:

```text
http://127.0.0.1:8000/docs
```

## 로컬 LightGBM 모델 학습

생성한 inventory-flow CSV 파일들을 아래 폴더에 넣습니다.

```text
app/data/training/
```

그다음 학습 스크립트를 실행합니다.

```bash
.venv/bin/python scripts/train_lightgbm.py --inventory "app/data/training/*.csv"
```

`app/data/training/`이 비어 있으면 작은 smoke test용으로 `app/data/inventory_flow_5days.csv`를 사용합니다.

학습 결과물:

```text
app/models/demand_lgbm.txt
app/models/demand_lgbm_metadata.json
```

FastAPI 서버는 위 모델 파일이 있으면 자동으로 로드합니다. 모델 파일이 없으면 서버 내부의 가벼운 fallback 예측 경로를 사용합니다.

## POS 마감 데이터 흐름

운영 환경에서는 POS 하루 마감 CSV를 S3에 적재하고, 서버가 해당 CSV를 내려받아 학습된 LightGBM 모델로 수요를 예측하는 흐름을 사용합니다. API 요청에서 예측용 payload를 직접 넘기지 않습니다.

현재 레포의 `app/data/` 파일은 로컬 개발과 테스트를 위한 샘플 CSV입니다. 운영 S3 CSV도 아래와 같은 스키마를 맞추면 같은 전처리/예측 파이프라인을 사용할 수 있습니다.

- `inventory_flow_5days.csv`: 일자별 재고 흐름, 수요, 실판매, 결품, 폐기, 기말재고
- `item_master.csv`: 품목 ID, 품목명, 단위, 유통기한, ESG 관련 품목 메타데이터
- `order_policy.csv`: 발주주기, 리드타임, 예측 horizon, 안전재고 방향, MOQ 텍스트

## 빠른 호출 예시

하루 마감 전체 플로우:

```bash
curl -X POST http://127.0.0.1:8000/daily-close
```

챗봇 설명:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"샌드위치 발주가 왜 필요한지 알려줘"}'
```

## API 명세

### `GET /health`

서버 상태와 Bedrock Llama 설정을 확인합니다.

응답 예시:

```json
{
  "status": "ok",
  "llm_provider": "bedrock",
  "bedrock_model_id": "meta.llama3-2-1b-instruct-v1:0"
}
```

### `POST /forecast`

S3에 적재된 POS 마감 CSV를 기반으로 수요를 예측하는 엔드포인트입니다. 요청 body는 필요 없습니다.

운영 데이터 흐름:

```text
S3 POS 마감 CSV
-> 서버 전처리
-> 저장된 LightGBM 모델 로드
-> 품목별 수요 예측
```

로컬 개발 환경에서는 같은 스키마의 샘플 CSV를 `app/data/`에서 읽습니다.

응답 예시:

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

동작 방식:

- `app/models/demand_lgbm.txt`가 있으면 저장된 LightGBM 모델을 로드합니다.
- 저장된 모델이 없으면 서버 내부 fallback 예측 경로를 사용합니다.
- 운영 환경에서는 S3에서 가져온 POS 마감 CSV를 같은 스키마로 전처리해 예측합니다.
- 수치 예측 API이므로 exact cache만 사용합니다.
- LLM은 수요 예측에 사용하지 않습니다.

### `POST /order-recommendation`

예측 수요, 최신 기말재고, 결품, 안전재고, 리드타임, MOQ 정책을 기반으로 품목별 발주량을 계산합니다. 요청 body는 필요 없습니다.

응답 예시:

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

동작 방식:

- 기본 방식은 base-stock 정책입니다.
- LLM은 발주량 계산에 사용하지 않습니다.
- 수치 계산 API이므로 exact cache만 사용합니다.

### `POST /daily-close`

하루 마감 전체 플로우를 실행하고, AWS Bedrock Llama API로 점주용 설명 문장을 생성합니다. 요청 body는 필요 없습니다.

처리 흐름:

```text
S3 POS 마감 CSV
-> 서버 전처리
-> 저장된 LightGBM 수요 예측
-> base-stock 발주 추천
-> Bedrock Llama 설명 생성
```

응답 예시:

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

동작 방식:

- Bedrock Llama는 설명 문장 생성에만 사용합니다.
- Bedrock 인증이 없거나 호출에 실패하면 deterministic fallback 요약을 반환합니다.
- 하루 마감 계산 결과는 exact cache만 사용합니다.

### `POST /chat`

점주가 발주 추천 결과에 대해 질문하면 AWS Bedrock Llama API로 답변합니다. 이 엔드포인트만 semantic cache를 사용합니다.

요청 예시:

```json
{
  "question": "샌드위치 발주가 왜 필요한지 알려줘"
}
```

응답 예시:

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

동작 방식:

- 챗봇은 발주 추천 근거를 설명합니다.
- 챗봇은 예측값이나 발주량을 변경하지 않습니다.
- 유사 질문은 semantic cache로 재사용해 Bedrock API 반복 호출을 줄입니다.
- semantic cache namespace에는 store ID, business date, data version이 포함됩니다.

## 캐시 정책

- `/forecast`, `/order-recommendation`, `/daily-close`: exact cache만 사용
- `/chat`: semantic cache 사용

수요 예측과 발주량 계산은 숫자 정확도가 중요하므로 semantic cache를 사용하지 않습니다. semantic cache는 점주 챗봇의 유사 질문 답변 재사용에만 사용합니다.

Redis가 없으면 exact cache는 인메모리로 동작합니다. semantic cache는 sqlite-vec를 우선 사용하고, 환경에 따라 일반 SQLite fallback 경로를 사용합니다.

## 테스트

```bash
.venv/bin/python -m pytest
```
