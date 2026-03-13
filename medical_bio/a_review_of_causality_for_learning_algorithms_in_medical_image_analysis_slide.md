---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# A Review of Causality for Learning Algorithms in Medical Image Analysis

- Athanasios Vlontzos, Daniel Rueckert, Bernhard Kainz
- Machine Learning for Biomedical Imaging 2022
- Review of causal thinking for robust medical imaging algorithms

---

## 문제 배경

- 의료영상 AI는 실험실 성능은 높아도 임상 번역 성공률은 낮다.
- 저자들은 원인 중 하나를 다음에서 찾는다.
  - 모델이 상관관계와 인과관계를 구분하지 못한다.
- 의료영상 데이터는 병원, 장비, 환자군, 라벨링 절차에 따라 크게 달라진다.
- 따라서 단순 empirical risk minimization만으로는
  임상 환경에서 안정적이지 않을 수 있다.

---

## 핵심 질문

- 모델은 진짜 질병 관련 원인을 학습하는가?
- 아니면 병원 식별자, 스캐너 특성, dataset bias 같은 shortcut을 학습하는가?
- domain shift가 생겼을 때 어떤 feature가 무너지고 무엇이 유지되는가?
- causality가 임상 번역 전 단계의 강건성을 높일 수 있는가?

---

## 논문의 핵심 메시지

- 이 논문은 새 모델을 제안하지 않는다.
- 대신 의료영상 AI의 병목을 **model size 부족**이 아니라
  **causal robustness 부족**으로 재해석한다.
- 즉, 중요한 질문은 "정확도가 높은가"보다
  "**왜 그 예측을 하는가, 다른 환경에서도 유지되는가**"이다.

---

## Structural Causal Models

- 논문은 Pearl 계열 인과 추론을 먼저 소개한다.
- 핵심 개념:
  - Structural Causal Model
  - Directed Acyclic Graph
  - intervention
  - counterfactual
- 의료영상에서 의미하는 바:
  - 관찰된 패턴이 아니라 **개입했을 때 결과가 어떻게 달라질지**를 묻는다.

---

## Potential Outcomes

- 다른 축으로 Rubin의 potential outcomes framework를 설명한다.
- 이 프레임은 treatment effect estimation과 자연스럽게 연결된다.
- 주요 개념:
  - average treatment effect
  - individual treatment effect
  - propensity score
- 발표 포인트:
  - SCM은 구조적 원인 모델에 가깝고
  - potential outcomes는 처치 효과 평가와 의사결정에 가깝다.

---

## 논문이 제시하는 분류 체계

- 저자들은 의료영상 causality 연구를 다음으로 정리한다.
  - causal discovery
  - causal reasoning and counterfactuals
  - causal representation learning
  - causal fairness
  - reinforcement learning and decision making
- 이 분류의 의미는 명확하다.
  - causality는 설명용 이론이 아니라 **robust learning 전반의 프레임**이다.

---

## Causal Discovery

- 데이터로부터 원인 구조를 추정하려는 접근이다.
- 매력은 크지만 의료영상에서는 어렵다.
  - 고차원 입력
  - 작은 표본
  - 관측되지 않은 confounder
  - identifiability 문제
- 따라서 full graph 복원보다 representation 수준의 제한적 causal discovery가 더 현실적이다.

---

## Counterfactual Reasoning

- 의료영상에서 counterfactual 질문은 매우 실용적이다.
  - 병변이 없었다면 예측이 어떻게 달라질까
  - acquisition setting이 달랐다면 segmentation은 어떻게 변할까
- 장점:
  - spurious feature와 disease-relevant feature 분리
  - causally plausible sample generation
  - explanation과 intervention planning 연결

---

## Causal Representation Learning

- 저자들은 좋은 representation이 단순 predictive feature가 아니라
  **domain shift에도 유지되는 invariant causal factor**를 포착해야 한다고 본다.
- 왜 중요한가:
  - 병원이나 장비가 바뀌어도 질병 기전은 크게 안 바뀔 수 있다.
  - shortcut feature는 외부 검증에서 쉽게 붕괴한다.
- 이 논문의 가장 현대적인 메시지도 여기에 있다.

---

## Fairness와 Decision Making

- causality는 fairness 문제에도 중요하다.
  - confounder와 mediator를 구분
  - 허용할 경로와 차단할 경로를 구조적으로 논의
- reinforcement learning과도 연결된다.
  - 진단 이후 추적 검사
  - 치료 선택
  - 개입 시점 결정
- 즉, causality는 예측을 넘어 의료 의사결정과 연결된다.

---

## 이 논문이 주는 실질적 결론

- causality는 domain shift와 spurious correlation 문제를 줄일 잠재력이 있다.
- 하지만 의료영상 분야에서 실제 채택은 아직 제한적이다.
- 특히 deployment 수준 검증은 매우 적다.
- 따라서 이 논문은 낙관론만이 아니라
  **중요하지만 아직 uptake는 낮다**는 냉정한 평가를 한다.

---

## 강점

- 의료영상 임상 번역 실패를 구조적으로 설명한다.
- SCM, intervention, counterfactual, potential outcomes를
  의료영상 맥락으로 연결하는 브리지 문헌 역할을 한다.
- 기술 자체보다 연구 질문을 바꾸게 만든다.
  - 무엇을 예측할 것인가
  - 무엇을 원인으로 볼 것인가

---

## 한계

- 정량적 메타분석과 실제 성능 비교는 부족하다.
- 인과 그래프 설계와 confounder 관측의 실무 난이도를 충분히 체감시키지는 못한다.
- 의료영상-specific 사례 수가 당시에는 아직 적다.
- 2022년 이후 커진 foundation model과 causality의 접점은 반영되지 않는다.

---

## 발표용 핵심 메시지

- 의료영상 AI의 진짜 병목은 더 큰 모델이 아니라
  **더 causally robust한 모델**일 수 있다.
- validation 성능만으로 모델을 신뢰하면 안 된다.
- 병원, 장비, 인구집단이 바뀌었을 때 유지되는 feature가 중요하다.
- causality는 explanation이 아니라
  **deployment robustness를 위한 도구**로 봐야 한다.

---

## 결론

- 이 논문은 의료영상 AI가 왜 임상에서 자주 무너지는지를 인과 관점에서 해석하는 리뷰다.
- 핵심은 causal discovery보다도
  spurious correlation, domain shift, fairness, counterfactual robustness를 다시 보게 만드는 데 있다.
- 구현 레시피 문헌은 아니지만,
  임상 번역과 강건성을 진지하게 다루려면 중요한 기준 문헌이다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/a_review_of_causality_for_learning_algorithms_in_medical_image_analysis_slide.md>
