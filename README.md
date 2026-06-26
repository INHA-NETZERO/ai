# Net-Zero AI Server

백엔드가 넘겨준 POS 판매 이력과 grounding 데이터를 기반으로 수요 예측, 자연어 코멘트, 후속 챗봇 설명을 제공하는 FastAPI 서버입니다. 수치 예측은 LightGBM이 담당하고, 같은 로컬 Llama 3.2 1B가 짧은 코멘트와 사용자 질문 답변을 생성합니다.

## 기술 스택

- FastAPI: `/v1/forecast`, `/v1/order-recommendation`, `/v1/generate`, `/v1/chat`
- Redis: deterministic 계산 결과용 exact cache. `REDIS_URL`이 없거나 Redis가 꺼져 있으면 인메모리 캐시 사용
- sqlite-vec: `/v1/generate`, `/v1/chat` sLLM 설명용 semantic cache
- Local Llama: Ollama의 `llama3.2:1b` 또는 Hugging Face `meta-llama/Llama-3.2-1B-Instruct`로 grounding 기반 코멘트와 챗봇 답변 생성
- 예측 엔진: LightGBM 수요 분위 예측, 모델 미준비 시 `baseline_v1` 폴백

## 현재 구현 기준

이 레포는 로컬 실행과 데모 검증이 가능한 상태입니다. LLM은 기본적으로 로컬 Ollama의 Llama 3.2 1B를 호출합니다.

- Local Llama: `ollama pull llama3.2:1b` 후 Ollama 서버가 켜져 있으면 실제 LLM 답변을 반환합니다. 연결 실패 시 deterministic fallback 문장을 반환합니다.
- Redis: `REDIS_URL`이 설정되면 exact cache 저장소로 사용됩니다. 없으면 인메모리 cache입니다.
- Spring `/v1` S3 입력: 백엔드가 `salesHistory.presignedUrls`를 넘기면 AI 서버는 HTTP GET으로 CSV만 다운로드합니다. 이 경로는 S3 AWS key가 필요 없습니다.
- 기존 데모 API S3 로더: `DATA_SOURCE=s3`일 때 `inventory_flow`, `item_master`, `order_policy` CSV를 S3에서 직접 읽습니다. 이 경로는 S3 권한이 필요합니다.
- LightGBM: 로컬 학습/저장/로드/새 CSV 예측 테스트가 구현되어 있습니다.

현재 상태는 아래에서 확인할 수 있습니다.

```bash
curl -s http://127.0.0.1:8000/integration-status | python -m json.tool
```

## 로컬 Llama 설정

`/v1/generate`와 `/v1/chat`은 모두 `app/services/llm.py`의 `LocalLlamaClient`를 사용합니다.
AI 서버는 grounding의 숫자를 바꾸거나 새 수치를 만들지 않고, 받은 근거를 한국어 설명으로만 바꿉니다.

기본 방식은 Ollama입니다.

```bash
ollama pull llama3.2:1b
ollama serve
```

Ollama는 Llama 모델 서버입니다. FastAPI 서버와 별도로 켜져 있어야 합니다.

터미널 1에서 Llama 서버:

```bash
ollama serve
```

터미널 2에서 FastAPI 서버:

```bash
.venv/bin/uvicorn app.main:app --reload
```

`.env` 설정:

```text
LLM_PROVIDER=local
LOCAL_LLM_BACKEND=ollama
LOCAL_LLM_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
```

로컬 Llama 연결 확인:

```bash
.venv/bin/python scripts/check_local_llama.py
```

Hugging Face Transformers로 직접 로드하려면 `LOCAL_LLM_BACKEND=transformers`,
`LOCAL_HF_MODEL=meta-llama/Llama-3.2-1B-Instruct`를 사용합니다. 이 경우 별도 모델 접근 권한과 로컬 추론 의존성이 필요합니다.

GGUF 모델 파일을 직접 로드하려면 `llama-cpp-python` 설치 후 아래처럼 설정합니다.

```text
LOCAL_LLM_BACKEND=llama_cpp
LOCAL_GGUF_MODEL_PATH=/absolute/path/to/llama-3.2-1b-instruct.gguf
```

## Docker Compose 실행

Docker로 한 번에 띄우면 `ai-server`, `ollama`, `redis`가 같은 네트워크에서 실행됩니다. AI 서버는 컨테이너 내부에서 `http://ollama:11434`로 Llama를 호출합니다.

처음 실행:

```bash
docker compose up --build
```

첫 실행 시 `ollama-pull` 컨테이너가 `llama3.2:1b` 모델을 내려받습니다. 모델은 `ollama-models` Docker volume에 저장되므로 다음 실행부터는 다시 받지 않습니다.

백그라운드 실행:

```bash
docker compose up -d --build
```

상태 확인:

```bash
docker compose ps
curl -s http://127.0.0.1:8000/health
```

기능 전체 간단 테스트:

```bash
scripts/docker_smoke_test.sh
```

로그 확인:

```bash
docker compose logs -f ai-server
docker compose logs -f ollama
```

종료:

```bash
docker compose down
```

모델과 Redis 데이터 volume까지 지우려면:

```bash
docker compose down -v
```

## Redis 설정

exact cache를 Redis로 사용하려면 `.env`의 `REDIS_URL`을 Redis endpoint로 설정합니다.

```text
REDIS_URL=redis://<redis-host>:6379/0
```

Redis가 없거나 연결에 실패하면 exact cache는 자동으로 인메모리 캐시로 동작합니다.

## S3 입력 방식

Spring 백엔드 연동용 `/v1/forecast`, `/v1/order-recommendation`은 요청 body의 `salesHistory.presignedUrls`를 HTTP GET으로 다운로드합니다.

```json
{
  "salesHistory": {
    "presignedUrls": [
      "https://bucket.s3.ap-northeast-2.amazonaws.com/sales.csv?X-Amz-Algorithm=..."
    ],
    "format": "sales_csv_v1"
  }
}
```

이 방식은 URL 자체에 임시 읽기 권한이 들어 있으므로 AI 서버에 S3 Access Key가 없어도 됩니다. URL 유효시간이 끝났거나 다운로드가 실패하면 해당 이력 없이 baseline 경로로 폴백합니다.

presigned URL 다운로드만 확인하려면 아래처럼 실행합니다.

```bash
.venv/bin/python scripts/check_s3.py --url 'https://bucket.s3.ap-northeast-2.amazonaws.com/sales.csv?...'
```

presigned URL을 받자마자 저장된 LightGBM 모델로 예측까지 실행하려면 아래 스크립트를 사용합니다.

```bash
.venv/bin/python scripts/predict_from_s3_url.py \
  --url 'https://bucket.s3.ap-northeast-2.amazonaws.com/sales.csv?...' \
  --days 2 \
  --output app/data/predictions/s3_url_forecast.json
```

URL 없이 로컬 CSV로 같은 흐름을 미리 테스트하려면 `--input`을 사용합니다.

```bash
.venv/bin/python scripts/predict_from_s3_url.py \
  --input app/data/Test/sales_demo_1y.csv \
  --days 2
```

Test 마감/재고흐름 CSV에서 수요예측과 발주 추천량까지 한 번에 뽑으려면 아래 스크립트를 사용합니다.

```bash
.venv/bin/python scripts/recommend_orders_from_csv.py \
  --inventory-flow app/data/Test/inventory_demo_1y.csv \
  --output app/data/predictions/test_order_recommendations_all.csv
```

출력에는 품목별 현재재고, 결품, 예측수요 합계, base-stock level, 추천 발주량, 실제 더미 발주량 대비 차이가 포함됩니다.
특정 구분만 보고 싶을 때만 `--item-type 완제품`처럼 필터를 추가합니다.

1년치 Test 데이터 전체를 날짜별로 백테스트해서 누적 폐기 감소와 탄소 절감량을 계산하려면 아래 스크립트를 사용합니다.

```bash
.venv/bin/python scripts/backtest_order_savings.py \
  --inventory-flow app/data/Test/inventory_demo_1y.csv \
  --output app/data/predictions/backtest_order_savings_all.csv
```

기본 설정은 앞 14일을 초기 히스토리로 사용하고, 이후 날짜별로 추천 발주량과 기존 발주량을 비교합니다.
`--item-type`을 생략하면 완제품과 원재료를 포함한 전체 품목을 대상으로 계산합니다.

기존 데모 API(`/forecast`, `/order-recommendation`)가 로컬 CSV 대신 S3 CSV를 직접 읽게 하려면 `.env`를 아래처럼 설정합니다.

```text
DATA_SOURCE=s3
AWS_REGION=ap-northeast-2
S3_BUCKET=<bucket-name>
S3_PREFIX=<optional/prefix>
S3_INVENTORY_FLOW_KEY=inventory_flow_5y.csv
S3_ITEM_MASTER_KEY=item_master.csv
S3_ORDER_POLICY_KEY=order_policy.csv
STORE_ID=inha-store-001
```

최종 S3 key는 `S3_PREFIX`와 각 key를 합쳐서 만듭니다. 예를 들어 `S3_PREFIX=stores/1/closing/2026-06-27`이면 아래 객체를 읽습니다.

```text
s3://<bucket-name>/stores/1/closing/2026-06-27/inventory_flow_5y.csv
s3://<bucket-name>/stores/1/closing/2026-06-27/item_master.csv
s3://<bucket-name>/stores/1/closing/2026-06-27/order_policy.csv
```

직접 S3 읽기 방식에 필요한 IAM 권한:

```text
s3:GetObject
```

대상은 위 세 CSV 객체입니다. AWS 인증은 환경변수, `AWS_PROFILE`, EC2/ECS IAM Role 등 boto3 기본 인증 체인을 따릅니다. Spring 백엔드의 presigned URL 방식만 쓸 때는 이 설정이 필요 없습니다.

직접 S3 설정이 맞는지 확인하려면 `.env` 설정 후 아래 명령을 실행합니다.

```bash
.venv/bin/python scripts/check_s3.py
```

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

생성한 판매 학습 CSV 파일들을 아래 폴더에 넣습니다.

```text
app/data/training/
```

학습 CSV 컬럼은 아래 순서와 이름을 정확히 맞춰야 합니다.

```text
날짜,요일,날씨,기온,강수mm,행사중여부,공휴일여부,신메뉴여부,품목,구분,수요,판매수량,매진여부,매진시각,비고_시나리오
```

각 컬럼의 역할:

- `날짜`: 판매일, `YYYY-MM-DD`
- `요일`: 월/화/수/목/금/토/일
- `날씨`: 맑음, 흐림, 비 등
- `기온`: 숫자형 기온
- `강수mm`: 숫자형 강수량
- `행사중여부`: 행사 여부, `True`/`False` 또는 `Y`/`N`
- `공휴일여부`: 공휴일 여부, `True`/`False` 또는 `Y`/`N`
- `신메뉴여부`: 신메뉴 여부, `True`/`False` 또는 `Y`/`N`
- `품목`: 품목명
- `구분`: 완제품/원재료 등 품목 구분
- `수요`: LightGBM 학습 target
- `판매수량`: POS에서 실제 판매된 수량. 모델에는 같은 품목의 직전 마감 lag 피처로 들어갑니다.
- `매진여부`: 해당 일자 매진 여부. 모델에는 직전 마감 lag 피처로 들어갑니다.
- `매진시각`: 매진 시각. `14:30`, `14.5`, 빈 값 형식을 지원합니다.
- `비고_시나리오`: 더미 생성 시나리오 설명

그다음 학습 스크립트를 실행합니다.

```bash
.venv/bin/python scripts/train_lightgbm.py --training "app/data/training/sales_*.csv"
```

`--inventory` 옵션명도 이전 호환을 위해 남겨두었지만, 실제로는 위 판매 학습 CSV 스키마를 읽습니다. 재고흐름 CSV처럼 다른 헤더를 가진 파일을 넣으면 학습 전에 컬럼 검증에서 실패합니다.

학습 결과물:

```text
app/models/demand_lgbm.txt
app/models/demand_lgbm_metadata.json
```

FastAPI 서버는 위 모델 파일이 있으면 자동으로 로드합니다. 모델 파일이 없으면 서버 내부의 가벼운 fallback 예측 경로를 사용합니다.

학습 스크립트는 날짜 기준으로 오래된 데이터부터 `train 70%`, `validation 15%`, `test 15%`로 나눕니다. 모델은 train set으로 학습하고, validation set은 학습 중 성능 확인용, test set은 최종 성능 확인용으로 사용합니다.

출력과 metadata에는 각 set별 평가 지표와 train 대비 test 오차 증가량인 `overfit_gap`이 저장됩니다. 발표나 문서에는 test set 지표를 최종 모델 성능으로 쓰면 됩니다.

```text
MAE
RMSE
MAPE
overfit_gap
```

대량 CSV 학습 시에는 파일들을 glob으로 한 번에 넘길 수 있습니다. 학습 전 모든 row를 날짜순으로 정렬한 뒤 시간순 holdout을 만들기 때문에, 미래 데이터를 train set에 섞는 방식보다 오버피팅과 데이터 누수를 더 조심스럽게 확인할 수 있습니다.

## 로컬 새 데이터셋 예측/평가

학습된 모델을 다시 학습하지 않고, 새 CSV에 바로 적용하려면 아래 스크립트를 사용합니다.

```bash
.venv/bin/python scripts/predict_lightgbm.py --input app/data/inference/new_sales.csv
```

입력 CSV가 학습 스키마와 같은 판매 데이터이고 `수요`가 있으면 `sales` 모드로 동작합니다. 실제 수요와 예측 수요를 비교해서 MAE, RMSE, MAPE를 출력합니다.

```text
날짜,요일,날씨,기온,강수mm,행사중여부,공휴일여부,신메뉴여부,품목,구분,수요,판매수량,매진여부,매진시각,비고_시나리오
```

POS 마감/재고흐름 CSV처럼 `판매수량` 정답이 없는 파일은 `closing` 모드로 예측만 수행합니다.

```bash
.venv/bin/python scripts/predict_lightgbm.py \
  --mode closing \
  --input app/data/inference/new_inventory_flow.csv
```

예측 결과 전체를 파일로 저장하려면 `--output`을 사용합니다.

```bash
.venv/bin/python scripts/predict_lightgbm.py \
  --input app/data/inference/new_sales.csv \
  --output app/data/predictions/new_sales_predictions.csv
```

`app/data/inference/`, `app/data/predictions/`의 CSV/JSON 파일은 로컬 테스트용으로 `.gitignore` 처리되어 있습니다.

## POS 마감 데이터 흐름

운영 흐름은 POS 하루 마감 CSV를 S3에 적재하고, 서버가 해당 CSV를 내려받아 학습된 LightGBM 모델로 수요를 예측하는 방식입니다.

`DATA_SOURCE=local`이면 `app/data/` 파일을 로컬 개발과 테스트용 샘플 CSV로 읽고, `DATA_SOURCE=s3`이면 S3의 같은 스키마 CSV를 읽습니다.

- `inventory_flow_5y.csv`: 일자별 재고 흐름, 수요, 실판매, 결품, 폐기, 기말재고
- `item_master.csv`: 품목 ID, 품목명, 단위, 유통기한, ESG 관련 품목 메타데이터
- `order_policy.csv`: 발주주기, 리드타임, 예측 horizon, 안전재고 방향, MOQ 텍스트

## 빠른 호출 예시

백엔드 grounding 기반 LLM 설명:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"우유를 왜 66L 발주하나요?",
    "locale":"ko",
    "grounding":{
      "item":{"itemId":101,"itemName":"우유","unit":"L"},
      "forecast":{"p10":60,"p50":80,"p90":108},
      "recommendation":{"recommendedQuantity":66}
    }
  }'
```

사용자 후속 질문:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"왜 이 수량인지 더 자세히 알려줘",
    "locale":"ko",
    "grounding":{
      "item":{"itemId":101,"itemName":"우유","unit":"L"},
      "forecast":{"p10":60,"p50":80,"p90":108},
      "recommendation":{"recommendedQuantity":66},
      "carbon":{"potentialSavingKg":39.4}
    },
    "history":[{"role":"assistant","content":"우유는 66L 발주를 권장합니다."}]
  }'
```

## API 명세

### `GET /health`

서버 상태와 로컬 Llama 설정을 확인합니다.

응답 예시:

```json
{
  "status": "UP",
  "model": {
    "forecast": "baseline_v1",
    "llm": "local-llama3.2-1b"
  },
  "llm_provider": "local",
  "local_llm_backend": "ollama",
  "local_llm_model": "llama3.2:1b"
}
```

### Spring 백엔드 연동용 `/v1` 계약

Spring 백엔드가 붙는 API는 아래 네 개입니다. 기존 `/forecast`, `/order-recommendation`은 로컬 개발용 데모 엔드포인트입니다.

| Method | Path | 역할 |
| --- | --- | --- |
| `POST` | `/v1/order-recommendation` | 발주 커버기간 일별 p10/p50/p90 수요예측 |
| `POST` | `/v1/forecast` | 다음날 단일일 p10/p50/p90 수요예측 |
| `POST` | `/v1/generate` | LightGBM/백엔드 grounding 기반 짧은 자연어 코멘트 |
| `POST` | `/v1/chat` | 같은 grounding과 history 기반 후속 챗봇 설명 |

`/v1` API는 상태 없는 서비스로 동작합니다. DB를 직접 읽지 않고, 요청 body와 `salesHistory.presignedUrls`로 받은 CSV만 사용합니다.

저장된 LightGBM 모델은 학습 데이터의 `품목` 이름 기준으로 학습되어 있습니다. 따라서 백엔드가 `rows[].itemName`을 함께 넘기고, 그 값이 학습 metadata의 품목명과 일치하면 `lgbm_global_v1`로 예측합니다. `itemName`이 없거나 모델 파일이 없거나 품목명이 맞지 않으면 기존 `ma7/trend` 기반 `baseline_v1`로 안전하게 폴백합니다.

#### `POST /v1/order-recommendation`

요청 예시:

```json
{
  "storeId": 1,
  "targetDate": "2026-06-27",
  "salesHistory": {
    "presignedUrls": ["https://example.com/sales.csv?..."],
    "format": "sales_csv_v1"
  },
  "coverage": {
    "leadTimeDays": 1,
    "orderCycleDays": 7,
    "coverageDays": 8
  },
  "weather": [
    {
      "forecastDate": "2026-06-28",
      "avgTemp": 21.2,
      "precipitationMm": 12.0,
      "precipitationProb": 80,
      "skyCode": 4
    }
  ],
  "rows": [
    {
      "itemId": 101,
      "itemName": "아메리카노",
      "itemType": "판매음료",
      "orderCycleDays": 7,
      "leadTimeDays": 1,
      "features": {
        "dayOfWeek": 6,
        "isHoliday": false,
        "ma7": 9.4,
        "trend": -0.3
      }
    }
  ]
}
```

응답 예시:

```json
{
  "modelVersion": "lgbm_global_v1",
  "predictions": [
    {
      "itemId": 101,
      "daily": [
        { "date": "2026-06-28", "p10": 7.144, "p50": 8.93, "p90": 11.609 }
      ]
    }
  ]
}
```

#### `POST /v1/forecast`

다음날 단일일 수요 분위만 반환합니다.

```json
{
  "modelVersion": "lgbm_global_v1",
  "targetDate": "2026-06-28",
  "predictions": [
    { "itemId": 101, "p10": 7.144, "p50": 8.93, "p90": 11.609 }
  ]
}
```

#### `POST /v1/generate`

백엔드가 만든 grounding 숫자를 2~3문장 한국어 설명으로 바꿉니다. `cacheHit`, `latencyMs`, `tokens`는 항상 포함됩니다. 유사 질문은 semantic cache로 재사용합니다. 로컬 Llama 연결에 실패하면 fallback 문장을 반환합니다.

요청 예시:

```json
{
  "question": "우유를 왜 66L 발주하나요?",
  "locale": "ko",
  "grounding": {
    "item": { "itemId": 101, "itemName": "우유", "unit": "L" },
    "forecast": { "p10": 60, "p50": 80, "p90": 108 },
    "recommendation": { "recommendedQuantity": 66 },
    "carbon": { "potentialSavingKg": 39.4 }
  }
}
```

응답 예시:

```json
{
  "answer": "우유는 66L 발주를 권장합니다. 제공된 수요 중앙값은 80L입니다.",
  "cacheHit": false,
  "latencyMs": 12,
  "tokens": 142
}
```

#### `POST /v1/chat`

`/v1/generate`로 짧은 코멘트를 보여준 뒤, 사용자가 추가 질문을 입력하면 같은 grounding과 대화 history를 받아 더 자세히 설명합니다. 수치는 grounding에 있는 값만 인용하고 새로 계산하지 않습니다.

요청 예시:

```json
{
  "question": "왜 66L가 적당한지 더 자세히 알려줘",
  "locale": "ko",
  "grounding": {
    "item": { "itemId": 101, "itemName": "우유", "unit": "L" },
    "forecast": { "p10": 60, "p50": 80, "p90": 108 },
    "recommendation": { "recommendedQuantity": 66 },
    "carbon": { "potentialSavingKg": 39.4 }
  },
  "history": [
    { "role": "assistant", "content": "우유는 66L 발주를 권장합니다." }
  ]
}
```

응답 예시:

```json
{
  "answer": "제공된 근거 기준으로 우유의 추천 발주량은 66L입니다. 수요 중앙값은 80L이고 잠재 탄소 절감량은 39.4kgCO2e로 제공되어, 과발주를 줄이는 방향의 추천으로 볼 수 있습니다.",
  "cacheHit": false,
  "latencyMs": 18,
  "tokens": 168
}
```

요청 검증 실패는 명세대로 아래 형태의 `400` 응답을 반환합니다.

```json
{
  "error": {
    "code": "BAD_REQUEST",
    "message": "coverageDays mismatch (leadTimeDays + orderCycleDays)"
  }
}
```

### `GET /cache-status`

exact cache와 semantic cache의 적중률을 확인합니다. `REDIS_URL`을 설정하면 Redis를 exact cache 저장소로 사용하고, 없거나 연결 실패 시 인메모리 캐시를 사용합니다.

응답 예시:

```json
{
  "exact_cache_backend": "redis",
  "semantic_cache_backend": "sqlite_vec_or_sqlite",
  "exact_hits": 12,
  "exact_misses": 3,
  "exact_hit_rate": 0.8,
  "semantic_hits": 5,
  "semantic_misses": 7,
  "semantic_hit_rate": 0.4167,
  "estimated_llm_calls_saved": 5
}
```

참고:

- `REDIS_URL=redis://<redis-host>:6379/0` 형식으로 Redis를 사용할 수 있습니다.
- exact cache는 로컬 데모용 `/forecast`, `/order-recommendation` 계산 결과에 사용됩니다.
- semantic cache는 `/v1/generate`에서 Llama 반복 호출을 줄이는 데 사용됩니다.

### `GET /integration-status`

AWS 키, 로컬 Llama 설정, S3/Redis/모델 상태, 아직 남은 gap을 확인합니다.

응답 예시:

```json
{
  "environment": "local",
  "aws": {
    "configured": false,
    "detected_sources": [],
    "note": "AWS credentials are only needed for direct S3 loading or other AWS resources, not local Llama."
  },
  "llm": {
    "provider": "local",
    "backend": "ollama",
    "model": "llama3.2:1b",
    "ollama_base_url": "http://localhost:11434",
    "configured": true,
    "fallback_when_unavailable": true
  },
  "data_source": {
    "active": "local",
    "v1_presigned_url_loader_implemented": true,
    "v1_presigned_urls_require_aws_credentials": false,
    "local_csv_active": true,
    "s3_active": false,
    "s3_configured": false,
    "s3_loader_implemented": true
  },
  "gaps": [
    "API endpoints currently read local app/data CSV files. Set DATA_SOURCE=s3 to read S3 CSV files."
  ]
}
```

### `POST /forecast`

로컬 또는 S3의 POS 마감 CSV를 기반으로 수요를 예측하는 엔드포인트입니다. 요청 body는 필요 없습니다.

운영 데이터 흐름:

```text
S3 POS 마감 CSV
-> 서버 전처리
-> 저장된 LightGBM 모델 로드
-> 품목별 수요 예측
```

`DATA_SOURCE=local`이면 같은 스키마의 샘플 CSV를 `app/data/`에서 읽고, `DATA_SOURCE=s3`이면 S3에서 읽습니다.

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
- `DATA_SOURCE` 설정에 따라 로컬 CSV 또는 S3 CSV를 읽습니다.
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

## 캐시 정책

- `/forecast`, `/order-recommendation`: exact cache 사용
- `/v1/generate`, `/v1/chat`: semantic cache 사용

수요 예측은 숫자 정확도가 중요하므로 semantic cache를 사용하지 않습니다. semantic cache는 sLLM 자연어 설명과 후속 챗봇 답변 재사용에만 사용합니다.

Redis가 없으면 exact cache는 인메모리로 동작합니다. semantic cache는 sqlite-vec를 우선 사용하고, 환경에 따라 일반 SQLite fallback 경로를 사용합니다.

## 테스트

```bash
.venv/bin/python -m pytest
```
