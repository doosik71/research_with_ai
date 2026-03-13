---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Data Efficient Deep Learning for Medical Image Analysis: A Survey

- Suruchi Kumari, Pravendra Singh
- arXiv 2023
- Survey of label-efficient learning under multiple supervision shortages

---

## 문제 배경

- 의료영상 딥러닝의 가장 큰 병목은 여전히 데이터다.
- 그러나 의료에서 데이터 부족은 단순히 sample 수 부족이 아니다.
- 실제로는 다음 문제가 함께 존재한다.
  - 라벨 없음
  - 거친 라벨
  - 일부 라벨만 존재
  - noisy label
  - 매우 적은 수의 고품질 라벨
- 이 논문은 이 차이를 구조적으로 정리한다.

---

## 이 논문의 핵심 기여

- data-efficient learning을 supervision 수준에 따라 다섯 범주로 나눈다.
  - **no supervision**
  - **inexact supervision**
  - **incomplete supervision**
  - **inaccurate supervision**
  - **only limited supervision**
- 이 taxonomy 덕분에 SSL, weak supervision, semi-supervised, active learning, noisy label learning, few-shot, transfer learning을 하나의 좌표계에서 볼 수 있다.

---

## 전체 Taxonomy

- `No supervision`: self-supervised learning
- `Inexact supervision`: MIL, point / scribble / box supervision
- `Incomplete supervision`: semi-supervised, active learning, domain adaptation
- `Inaccurate supervision`: robust loss, re-weighting, label correction
- `Only limited supervision`: augmentation, few-shot learning, transfer learning

---

## No Supervision: Self-Supervised Learning

- 라벨이 전혀 없을 때 representation을 먼저 학습한다.
- 논문은 SSL을 세 부류로 본다.
  - predictive SSL
  - generative SSL
  - contrastive SSL
- 그리고 이들을 결합한 multi-self supervision도 정리한다.
- 발표 포인트:
  - 의료영상에서는 natural image pretext를 그대로 쓰기보다
    **해부학 구조와 3D 문맥을 반영한 pretext**가 더 중요하다.

---

## Inexact Supervision

- 정밀한 픽셀 라벨 대신 거친 annotation을 사용하는 경우다.
- 대표 형태:
  - image-level label, point annotation, scribble annotation, bounding box, bag label (MIL)
- 핵심 분야는 pathology와 segmentation이다.
- 약한 라벨도 structured prior와 pseudo-label expansion을 붙이면 꽤 강력하다.

---

## MIL의 의미

- MIL은 특히 whole-slide pathology에서 핵심이다.
  - instance-based
  - bag-based
  - attention-based
  - graph-based
  
---

## Weak Annotation의 의미

- weak segmentation에서는 다음 아이디어가 많이 쓰인다.
  - CRF refinement
  - seed growing
  - box-tightness constraint
  - pseudo-mask expansion
- annotation cost가 높은 의료에서 가장 현실적인 축 중 하나다.

---

## Incomplete Supervision

- 일부 샘플만 라벨이 있고 나머지는 unlabeled인 경우다.
- 논문은 세 흐름으로 정리한다.
  - semi-supervised learning
  - active learning
  - domain-adaptive learning
- 실제 의료 데이터셋은 이 범주가 가장 흔하다.

---

## Semi-Supervised

- consistency regularization
- pseudo-labeling
- generative methods
- hybrid methods

---

## Active learning

- uncertainty
- representativeness
- disagreement
- suggestive annotation

---

## Domain-adaptive learning

- discrepancy minimization
- adversarial adaptation
- image translation
- pseudo-label UDA
- 핵심은 **annotation budget을 최대한 아끼면서 일반화하는 것**이다.

---

## Inaccurate Supervision

- 라벨은 있지만 noisy하거나 잘못되었을 수 있다.
- 논문은 세 전략으로 정리한다.
  - robust loss design, data re-weighting, training procedure redesign
- 예:
  - generalized cross entropy, sample reweighting, mutual teaching, prototype refinement, label correction
- crowd label, pseudo label, weak label이 섞일수록 중요해진다.

---

## Only Limited Supervision

- 라벨 수 자체가 매우 적지만 품질은 상대적으로 좋은 경우다.
- 대표 전략:
  - data augmentation
  - few-shot learning
  - transfer learning
- 특히 의료영상에서는 augmentation도 task-aware해야 한다.
  - 병변 구조와 해부학 구조를 보존해야 한다.

---

## Few-Shot과 Transfer Learning

- few-shot:
  - metric learning
  - meta-learning
  - prototype-based methods
  - self-supervised few-shot segmentation
- transfer learning:
  - ImageNet pretrained CNN fine-tuning
  - 2D→3D transfer
  - task-specific transfer

---

## 논문이 주는 메시지 (1/2)

- Transfer learning은 여전히 강한 baseline이지만
    volumetric segmentation에서는 한계도 분명하다.

- 의료영상의 데이터 효율성 문제는
  **sample scarcity**보다 **supervision imperfection** 문제에 가깝다.
- 가장 유망한 방향은 단일 기법이 아니라 조합이다.
  - SSL + weak supervision
  - pseudo-labeling + active learning
  - transfer learning + domain adaptation

---

## 논문이 주는 메시지 (2/2)

- task-specific design이 필수다.
  - anatomy
  - 3D context
  - class imbalance
  - annotation cost

---

## 강점

- taxonomy가 넓고 실무 친화적이다.
- 250편 이상을 다뤄 coverage가 넓다.
- dataset 정리가 유용하다.
- 서로 다른 연구 흐름을 하나의 supervision 좌표계 안에 넣어 비교하게 해 준다.

---

## 한계와 이후 흐름

- 범위가 넓은 대신 개별 방법의 정량 비교는 깊지 않다.
- 현실에서는 supervision 범주가 섞여 있는데 taxonomy는 다소 깔끔하게 단순화한다.
- 2023년 이후 급증한 흐름은 제한적으로만 반영된다.
  - medical foundation models
  - multimodal text supervision
  - large-scale VLM pretraining
- 그럼에도 문제 정의 틀로는 여전히 강하다.

---

## 발표용 핵심 메시지

- 의료영상에서 데이터 효율성은 단순 sample 부족이 아니라
  **supervision quality와 availability의 문제**다.
- 가장 중요한 것은 데이터 상황을 먼저 진단하는 것이다.
  - 라벨이 없는가
  - 거친가
  - 일부만 있는가
  - noisy한가
  - 너무 적은가
- 알고리즘 선택은 그 다음 문제다.

---

## 결론

- 이 논문은 의료영상에서 적은 데이터로 학습하는 방법을 가장 넓게 정리한 문헌 중 하나다.
- 핵심 가치는 supervision 수준 중심 taxonomy에 있다.
- 의료영상 연구자가 SSL, weak supervision, semi-supervised, active learning, noisy label learning, few-shot, transfer learning 중 무엇을 쓸지 판단할 때 좋은 설계 지도 역할을 한다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/data_efficient_deep_learning_for_medical_image_analysis_a_survey_slide.md>
