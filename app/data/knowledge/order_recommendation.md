# 발주 추천 근거

## 기본 발주 원칙
발주 추천은 LLM이 직접 계산하지 않는다. LightGBM 수요 예측과 deterministic engine이 계산한 추천 발주량을 기준으로 설명한다.
추천 발주량은 리드타임과 발주 주기 동안 필요한 수요, 현재 재고, 결품 위험, 포장 단위, 안전 재고를 고려한 결과다.
LLM은 제공된 grounding의 recommendedQuantity 값을 그대로 인용해야 하며, 숫자를 새로 만들거나 다시 계산하면 안 된다.

## 점주용 설명 방식
점주에게는 먼저 품목명과 추천 발주량을 말한다. 그 다음 수요 예측값, 현재 재고 또는 결품 위험, 탄소 절감 근거를 짧게 연결한다.
예시는 다음 흐름을 따른다. "크루아상은 44개 발주를 권장합니다. 예측 수요와 현재 재고를 반영하면 결품 위험을 줄이면서 과발주를 피하는 수량입니다. 제공된 탄소 절감 추정치는 12.7kgCO2e입니다."

## 기존 발주량 비교
grounding에 historicalOrderQuantity, historical_order_quantity, recommendationMinusHistorical, recommendation_minus_historical 값이 있으면 기존 발주량과 비교해 설명한다.
추천량이 기존 발주량보다 적으면 과발주와 폐기 감소 가능성을 말한다. 추천량이 기존 발주량보다 많으면 결품 방지 목적을 말한다.
