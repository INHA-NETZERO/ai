# 기준 재고 발주 정책

## base-stock policy 개념
base-stock policy는 목표 재고 수준까지 부족한 만큼을 채우는 발주 방식이다.
목표 재고 수준은 리드타임과 발주 주기 동안의 예상 수요에 안전 재고를 더해 정한다.
현재 재고가 충분하면 예측 수요가 있어도 발주량은 작아질 수 있다.
현재 재고가 부족하거나 결품 위험이 높으면 기존보다 더 많이 발주할 수 있다.

## grounding 필드 해석
forecastDays는 발주가 커버하는 일수다.
forecastTotal은 forecastDays 동안의 예측 수요 합계다.
baseStockLevel은 목표 재고 수준이다.
projectedPosition은 현재 재고와 입고 예정 등을 반영한 예상 재고 위치다.
currentStock은 현재 남은 재고다.
safetyStock은 수요 변동에 대비한 여유 재고다.
packSize는 발주 단위이며, 추천 발주량은 packSize에 맞춰 반올림될 수 있다.
recommendedQuantity는 최종 추천 발주량이다.

## 설명 로직
forecastTotal이 recommendedQuantity보다 큰 경우에는 "현재 재고나 기준 재고 정책을 반영해 부족분만 발주한다"고 설명한다.
recommendedQuantity가 baseStockLevel과 비슷하면 "목표 재고 수준에 맞추는 발주"라고 설명한다.
projectedPosition이 0이면 "남은 재고가 거의 없어 결품 방지를 위해 발주가 필요하다"고 설명한다.
currentStock이 충분하면 "이미 보유한 재고를 먼저 반영해 과발주를 줄였다"고 설명한다.
packSize가 있으면 "포장 단위에 맞춰 최종 수량이 조정되었다"고 설명한다.

## 점주 질문 대응
"왜 예측 수요보다 적게 시켜요?"라는 질문에는 "예측 수요 전체를 새로 사는 것이 아니라, 현재 남은 재고를 뺀 부족분을 채우는 방식"이라고 답한다.
"왜 기존보다 많이 시켜요?"라는 질문에는 "최근 수요와 결품 위험을 반영하면 기존 수량으로 부족할 수 있기 때문"이라고 답한다.
"왜 딱 이 숫자예요?"라는 질문에는 "예측 수요, 현재 재고, 안전 재고, 발주 단위를 함께 반영한 최종 추천량"이라고 답한다.
