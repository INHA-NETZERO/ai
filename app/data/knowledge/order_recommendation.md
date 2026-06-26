# 발주 추천 근거

## 기본 발주 원칙
발주 추천은 LLM이 직접 계산하지 않는다. LightGBM 수요 예측과 deterministic engine이 계산한 추천 발주량을 기준으로 설명한다.
추천 발주량은 리드타임과 발주 주기 동안 필요한 수요, 현재 재고, 결품 위험, 포장 단위, 안전 재고를 고려한 결과다.
LLM은 제공된 grounding의 recommendedQuantity 값을 그대로 인용해야 하며, 숫자를 새로 만들거나 다시 계산하면 안 된다.

## 발주량 계산 흐름
발주 추천은 보통 다음 순서로 해석한다.
1. LightGBM이 대상 기간의 수요를 예측한다.
2. deterministic engine이 리드타임, 발주 주기, 현재 재고, 입고 예정, 포장 단위, 안전 재고를 반영한다.
3. 기준 재고 수준(baseStockLevel)에서 현재 보유 수량(projectedPosition)을 뺀 뒤, 포장 단위에 맞춰 추천 발주량(recommendedQuantity)을 정한다.
4. 추천량이 기존 발주량보다 적으면 폐기 감소 쪽의 의미가 크고, 추천량이 기존 발주량보다 많으면 결품 방지 쪽의 의미가 크다.
답변에서는 계산식을 새로 계산하지 말고, grounding에 들어온 forecastTotal, baseStockLevel, projectedPosition, recommendedQuantity, packSize 값을 근거로 설명한다.

## 기준 재고와 안전 재고
baseStockLevel은 발주 후 커버해야 하는 기간의 예상 수요와 안전 재고를 합친 목표 재고 수준이다.
safetyStock이 제공되면 날씨, 행사, 휴일, 매진 가능성처럼 수요가 흔들릴 때를 대비한 여유분이라고 설명한다.
projectedPosition 또는 currentStock이 제공되면 현재 남아 있는 재고를 먼저 소진하고 부족한 만큼만 발주하기 때문에 추천량이 예측 수요보다 작을 수 있다고 설명한다.

## 점주용 설명 방식
점주에게는 먼저 품목명과 추천 방향을 말한다. 그 다음 수요 예측 흐름, 현재 재고 또는 결품 위험, 폐기 감소 근거를 짧게 연결한다.
예시는 다음 흐름을 따른다. "크루아상은 결품을 막으면서 과발주를 피하는 쪽에 맞춘 추천입니다. 최근 수요 흐름과 현재 재고를 함께 반영해 필요한 만큼만 채우는 방향입니다."

## 점주용 설명 템플릿
점주 화면용 짧은 답변에서는 recommendedQuantity, p50, historicalOrderQuantity, recommendationMinusHistorical, potentialSavingKg 같은 숫자를 그대로 말하지 않는다.
추천량이 기존보다 줄어든 경우에는 "수요를 감당하면서 남는 재고 부담을 줄이는 방향"이라고 설명한다.
추천량이 기존보다 늘어난 경우에는 "결품 가능성을 줄이는 방향"이라고 설명한다.
currentStock 또는 projectedPosition이 있으면 "현재 재고를 먼저 반영해 부족분만 채우는 방식"이라고 설명한다.
baseStockLevel이 있으면 "목표 재고 수준까지 맞추는 방식"이라고 설명한다.

## 기존 발주량 비교
grounding에 historicalOrderQuantity, historical_order_quantity, recommendationMinusHistorical, recommendation_minus_historical 값이 있으면 기존 발주량과 비교해 설명한다.
추천량이 기존 발주량보다 적으면 과발주와 폐기 감소 가능성을 말한다. 추천량이 기존 발주량보다 많으면 결품 방지 목적을 말한다.

## 추천량이 예측 수요보다 작은 경우
추천 발주량이 p50 또는 forecastTotal보다 작다고 해서 오류는 아니다.
현재 재고, 입고 예정, 포장 단위, 기준 재고, 안전 재고가 함께 반영되면 예측 수요보다 적게 발주할 수 있다.
점주가 "예측 수요는 높은데 왜 적게 시켜요?"라고 물으면 "이미 남아 있는 재고와 목표 재고 수준을 반영해 부족분만 채우는 방식"이라고 설명한다.

## 추천량이 기존 발주량보다 큰 경우
추천 발주량이 기존 발주량보다 크면 폐기 감소보다 결품 방지 목적이 더 크다.
이때는 "최근 수요와 매진 가능성을 반영하면 기존 발주량으로는 부족할 수 있어 추천량이 증가했다"라고 설명한다.
탄소 절감량이 없거나 음수이면 억지로 탄소 절감을 말하지 않는다.
