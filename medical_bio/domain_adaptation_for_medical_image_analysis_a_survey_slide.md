---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Domain Adaptation for Medical Image Analysis: A Survey

- Hao Guan, Mingxia Liu
- arXiv 2021
- Survey of medical domain shift and adaptation strategies

---

## 문제 배경

- 의료영상 모델은 학습 도메인과 테스트 도메인이 같다고 가정하기 쉽다.
- 실제 임상에서는 이 가정이 거의 깨진다.
  - 병원 차이
  - 스캐너 제조사 차이
  - 촬영 프로토콜 차이
  - 환자군 차이
  - 해상도와 노이즈 차이
- 논문은 이를 `domain shift` 문제로 정리한다.

---

## 왜 의료영상에서 더 어려운가

- target 도메인 라벨을 새로 얻는 비용이 매우 높다.
- modality 자체가 다양하다.
  - CT
  - MRI
  - PET
  - pathology
- 2D뿐 아니라 3D/4D 구조까지 포함한다.
- 따라서 단순 fine-tuning만으로는 일반화가 충분하지 않다.

---

## 이 논문의 핵심 기여

- DA를 `shallow`와 `deep`으로 나누고,
  각각을 supervised / semi-supervised / unsupervised로 다시 구분한다.
- 문제 설정 축을 네 가지로 정리한다.
  - label availability
  - modality difference
  - number of sources
  - adaptation steps
- 의료영상 DA benchmark dataset도 분야별로 정리한다.

---

## 문제 설정 축 (1/2)

- `Label availability`
  - supervised DA
  - semi-supervised DA
  - unsupervised DA
- `Modality difference`
  - single-modality DA
  - cross-modality DA

---

## 문제 설정 축 (2/2)

- `Number of sources`
  - single-source
  - multi-source
- `Adaptation steps`
  - one-step
  - multi-step

---

## 전체 Taxonomy

- **Shallow DA**
  - instance weighting
  - feature transformation
- **Deep DA**
  - supervised deep DA
  - semi-supervised deep DA
  - unsupervised deep DA
- 논문의 중심 메시지: 초기 연구는 shallow가 많았지만
  - 의료영상 DA의 중심은 이미 **deep DA**로 이동했다.

---

## Shallow DA

- handcrafted feature나 전통적 ML 위에서 분포 차이를 줄인다.
- 대표 방식:
  - source sample weighting
  - common latent space projection
  - subspace alignment
  - low-rank regularization
- 장점: 데이터가 적을 때 안정적일 수 있다.
- 한계: 복잡한 3D 의료영상 표현을 다루기 어렵다.

---

## Supervised / Semi-Supervised Deep DA

- supervised deep DA:
  - source 학습 후 target 소량 라벨로 fine-tuning
  - 일부 층만 조정
  - 3D backbone adaptation
- semi-supervised deep DA:
  - labeled source + 일부 labeled target + unlabeled target
  - Y-Net류
  - reconstruction branch
  - semi-supervised GAN

---

## Unsupervised Deep DA가 왜 핵심인가

- 의료영상에서 target 라벨 확보가 가장 어렵기 때문이다.
- 따라서 survey의 중심축도 unsupervised deep DA다.
- 이 흐름의 대표 전략:
  - adversarial feature alignment
  - image translation
  - image + feature alignment 결합
  - disentangled representation
  - self-ensembling / consistency learning

---

## Adversarial Feature Alignment

- 대표 구조는 DANN이다.
- 목표: feature extractor가 domain classifier를 속이게 만들어
    source와 target이 구분되지 않는 representation을 학습
- 의료영상 확장:
  - segmentation + discriminator 공동 학습
  - ROI / edge-aware alignment
  - low-level adaptation for cross-modality
- 장점: target label 없이 정렬 가능
- 위험: task-specific 정보까지 희석될 수 있다.

---

## Image Translation과 Cross-Modality

- CycleGAN 기반 translation은 의료영상 DA에서 매우 중요하다.
- 대표 응용:
  - MRI ↔ CT adaptation
  - pathology style transfer
  - synthetic tumor to realistic image translation
- 장점: style gap을 직관적으로 줄인다.
- 한계: 진단 신호 자체를 왜곡하지 않는지 항상 검증이 필요하다.

---

## 더 강한 방향: 결합과 분해

- `Image + feature alignment`
  - translation 이후 feature-level adversarial alignment를 함께 수행
- `Disentangled representation`
  - content와 style을 분리
  - anatomy는 content
  - modality appearance는 style
- 논문은 특히 cross-modality DA에서 disentanglement가 유망하다고 본다.

---

## Self-Ensembling과 Lifelong Adaptation

- teacher-student consistency learning은
  unlabeled target에 대해 안정적인 적응을 돕는다.
- domain별 batch norm 분리 같은 접근은
  multi-target / lifelong adaptation으로 이어진다.
- 실제 병원 운영 관점에서 중요한 설정:
  - 새로운 병원이 계속 추가되는 상황에서
    기존 성능을 잃지 않고 순차 적응해야 한다.

---

## 이 논문의 핵심 난제

- 3D/4D volumetric representation
- limited training data
- inter-modality heterogeneity
- 특히 논문의 메시지는 명확하다.
  - 의료영상 DA는 2D natural image adaptation보다 훨씬 어렵고
  - cross-modality와 multi-source가 실제 핵심 문제다.

---

## 강점

- 문제 설정과 방법론 분류가 매우 명확하다.
- supervised / semi-supervised / unsupervised를 균형 있게 다룬다.
- cross-modality와 multi-source를 별도 축으로 드러낸다.
- benchmark dataset 정리가 연구 설계에 직접 도움이 된다.

---

## 한계와 이후 흐름

- 2021년 이후의 다음 흐름은 반영되지 않는다.
  - self-supervised pretraining
  - foundation model adaptation
  - test-time adaptation
  - diffusion 기반 adaptation
- survey 시점 특성상 adversarial DA 비중이 높다.
- 그럼에도 기본 분류 축은 지금도 유효하다.

---

## 발표용 핵심 메시지

- 의료영상 DA는 단순 fine-tuning 문제가 아니라
  **구조적 distribution shift 문제**다.
- 가장 중요한 연구 축은 deep unsupervised DA다.
- 특히 cross-modality adaptation과 3D/4D adaptation이 핵심 난제다.
- 실제 임상 배포를 생각하면
  multi-source, multi-target, lifelong adaptation까지 함께 봐야 한다.

---

## 결론

- 이 논문은 의료영상 DA의 초창기 표준 정리 문헌에 가깝다.
- 핵심 가치는 문제 설정 축과 shallow/deep, supervised/semi-supervised/unsupervised 구분을 명확히 정리한 데 있다.
- 이후 foundation model 기반 적응을 읽을 때도
  이 논문의 좌표계는 여전히 좋은 기준점으로 작동한다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/domain_adaptation_for_medical_image_analysis_a_survey_slide.md>
