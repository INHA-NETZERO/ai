#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "== health =="
curl -fsS "${BASE_URL}/health"
echo

echo "== generate comment =="
curl -fsS -X POST "${BASE_URL}/v1/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"내일 우유 얼마나 시켜요?",
    "locale":"ko",
    "grounding":{
      "item":{"itemId":101,"itemName":"우유","unit":"L"},
      "forecast":{"p10":60,"p50":80,"p90":108},
      "recommendation":{"recommendedQuantity":66},
      "carbon":{"potentialSavingKg":39.4}
    }
  }'
echo

echo "== follow-up chat =="
curl -fsS -X POST "${BASE_URL}/v1/chat" \
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
echo

echo "== single-day forecast =="
curl -fsS -X POST "${BASE_URL}/v1/forecast" \
  -H 'Content-Type: application/json' \
  -d '{
    "storeId":1,
    "targetDate":"2026-06-28",
    "salesHistory":{
      "presignedUrls":["https://example.invalid/sales.csv"],
      "format":"sales_csv_v1"
    },
    "weather":{
      "forecastDate":"2026-06-28",
      "avgTemp":21.2,
      "precipitationMm":12.0,
      "precipitationProb":80,
      "skyCode":4
    },
    "rows":[
      {
        "itemId":101,
        "features":{"dayOfWeek":6,"isHoliday":false,"ma7":9.4,"trend":-0.3}
      }
    ]
  }'
echo
