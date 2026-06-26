# 폐기 감소와 탄소 절감 근거

## 탄소 절감 설명
탄소 절감량은 LLM이 계산하지 않는다. 백엔드나 deterministic engine이 제공한 potentialSavingKg, carbonSavingKg, wasteReductionKg 같은 grounding 값을 그대로 사용한다.
추천 발주량이 기존 발주량보다 적고 수요를 충족할 수 있으면 폐기량 감소와 탄소 절감 가능성을 함께 설명한다.

## ESG 관점
이 서비스의 ESG 취지는 LLM이 모든 판단을 대신하는 것이 아니라, 경량 수요 예측 모델과 규칙 기반 발주 엔진으로 과발주와 폐기를 줄이는 것이다.
작은 Llama 모델은 점주 설명과 후속 질문 답변에만 사용하고, 수치 계산은 LightGBM과 deterministic engine이 담당한다.
