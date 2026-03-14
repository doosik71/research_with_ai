---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Application of belief functions to medical image segmentation: A review

- Ling Huang, Su Ruan, Thierry Denoeux
- Information Fusion 2023
- Review of belief function theory for uncertainty-aware medical segmentation

---

## 문제 배경

- 의료영상 분할은 정확도만의 문제가 아니다.
- 실제 환경에는 다음이 함께 존재한다.
  - 애매한 경계
  - noisy annotation
  - 전문가 간 불일치
  - 다중 modality 간 충돌
  - 불완전한 입력 정보
- 논문의 출발점:
  - segmentation mask뿐 아니라 **결과를 얼마나 신뢰할 수 있는가**도 중요하다.

---

## 왜 Belief Function Theory인가

- 저자들은 belief function theory(BFT)를
  확률의 대체재라기보다 **확장된 불확실성 표현 도구**로 본다.
- BFT는 다음을 명시적으로 표현할 수 있다.
  - belief, plausibility, ignorance, conflict
- 핵심 차이:
  - probability는 불확실성을 클래스 확률로 나눠야 하지만
  - BFT는 "모른다"는 무지 자체를 질량으로 줄 수 있다.

---

## BFT의 기본 구성

- segmentation 파이프라인을 다음처럼 본다.
  - feature extraction, mass assignment, evidence fusion, decision-making
- 주요 개념:
  - **Mass function**, **Belief / Plausibility**, **Dempster’s rule**, **Discounting**
- 발표 포인트:
  - 이 논문은 architecture survey가 아니라 **uncertainty representation survey**에 가깝다.

---

## BBA 생성: Supervised 방식

- BFT segmentation의 시작점은 BBA, 즉 mass function 생성이다.
- supervised BBA는 라벨 정보를 이용한다.
- 대표 계열:
  - likelihood-based methods
  - distance-based methods
- distance-based evidential classifier는
  deep feature와 결합하기 쉬워 현대 딥러닝과 접점이 크다.

---

## BBA 생성: Unsupervised 방식

- 라벨이 부족한 의료영상에서는 unsupervised BBA도 중요하다.
- 논문이 정리하는 대표 계열:
  - FCM 기반 모델
  - Evidential C-means (ECM)
  - Gaussian distribution 기반 모델
  - BFOD 기반 변환
- 핵심 장점:
  - 애매한 경계의 샘플을 단일 클러스터가 아니라 **클러스터 집합에 대한 질량**으로 표현할 수 있다.

---

## 이 논문의 가장 좋은 분류

- segmentation 방법을 **fusion이 어디에서 일어나는가**로 분류한다.
- 크게 두 축이다.
  - single classifier or clusterer
  - several classifiers or clusterers
- 그리고 각 축에서 다시 나눈다.
  - single-modal evidence fusion
  - multimodal evidence fusion
- 즉, BFT의 핵심은 "무엇을 결합하느냐"보다 **어느 단계에서 결합하느냐**다.

---

## Single Classifier / Clusterer

- 가장 단순한 구조는 하나의 모델 안에서 evidence를 결합하는 방식이다.
- typical pipeline:
  - feature extraction
  - mass calculation
  - feature-level fusion
  - final decision
- single-modal에서도 uncertainty-aware segmentation이 가능하다.
- multimodal로 가면 PET/CT, multi-sequence MRI처럼 서로 다른 modality의 상보성과 충돌을 함께 다룰 수 있다.

---

## Several Classifiers / Clusterers

- 여러 모델을 조합하면 BFT의 장점이 더 분명해진다.
- 가능한 fusion 수준:
  - feature-level fusion
  - classifier-level fusion
  - modality-level fusion
- 가장 복잡한 경우는 여러 modality + 여러 classifier를 모두 evidentially 통합하는 구조다.
- 의미:
  - 실제 임상에서 여러 전문가 의견을 종합하는 방식과 닮아 있다.

---

## 어디서 특히 설득력 있는가

- cardiac MR segmentation
- lung / spinal canal CT segmentation
- color biomedical image segmentation
- PET/CT tumor segmentation
- multi-sequence MRI segmentation
- 공통점:
  - 입력 정보가 불완전하거나
  - 여러 정보원이 상충하거나
  - uncertainty 자체가 중요한 과제다.

---

## 이 논문의 핵심 메시지

- BFT의 가치는 "항상 Dice를 더 높인다"가 아니다.
- 진짜 강점은 다음에 있다.
  - ignorance를 분리해 표현
  - 상충하는 증거를 구조적으로 결합
  - multimodal 정보를 해석 가능하게 통합
  - segmentation 결과의 신뢰성까지 다룸
- 따라서 이 문헌은 정확도 경쟁보다
  **trustworthy segmentation** 관점에서 읽어야 한다.

---

## 강점

- BFT 기초부터 응용 taxonomy까지 일관된 구조로 정리한다.
- supervised / unsupervised BBA 방법을 함께 다룬다.
- multimodal, multi-classifier 의료영상 문제에 특히 잘 맞는 프레임을 제시한다.
- segmentation을 정확도 중심에서 신뢰성 중심으로 확장해 생각하게 만든다.

---

## 한계

- 많은 사례가 hand-crafted feature나 전통적 classifier에 머문다.
- 최신 deep segmentation backbone과의 결합은 아직 초기 단계다.
- calibration, OOD robustness, reliability 비교 체계가 부족하다.
- foundation model, promptable segmentation 같은 최근 흐름과는 거리가 있다.

---

## 현재 시점에서의 해석

- 오늘 관점에서 BFT는 Bayesian uncertainty의 경쟁자라기보다
  **보완적 불확실성 표현 체계**로 읽는 편이 맞다.
- 특히 PET/CT, multi-sequence MRI 같은 multimodal 환경에서 설득력이 크다.
- 향후 연결 방향은 명확하다.
  - strong backbone + evidential output layer
  - ensemble + belief fusion
  - human-in-the-loop correction with uncertainty maps

---

## 발표용 핵심 메시지

- 이 논문은 segmentation backbone보다 **uncertainty modeling**이 핵심이다.
- BFT는 확률이 표현하기 어려운 무지와 충돌을 직접 다룬다.
- multimodal medical imaging에서 evidential fusion은 특히 강력한 관점이다.
- 의료영상 AI가 임상에 들어가려면 정확도뿐 아니라
  **신뢰성과 불확실성 표현**이 함께 필요하다.

---

## 결론

- 이 논문은 의료영상 분할을 정확도 경쟁이 아니라
  **불확실성과 신뢰성의 문제**로 다시 보게 만드는 리뷰다.
- 핵심은 BBA 생성과 fusion 단계 taxonomy, 그리고 multimodal evidence integration이다.
- 최신 backbone survey는 아니지만,
  trustworthy medical segmentation을 이해하는 기준 문헌으로는 매우 유용하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/application_of_belief_functions_to_medical_image_segmentation_a_review_slide.md>
