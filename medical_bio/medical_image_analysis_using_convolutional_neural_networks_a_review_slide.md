---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Medical Image Analysis using Convolutional Neural Networks: A Review

- Syed Muhammad Anwar et al.
- Journal of Medical Systems 2018 / arXiv 2017
- Early review of CNNs as a unifying tool for medical image analysis

---

## 문제 배경

- 의료영상 분석은 segmentation, detection, classification, CAD, retrieval 등 다양한 과제를 포함한다.
- 전통적 파이프라인은 feature extractor와 classifier가 분리되어 있다.
- 의료영상은 다음 특성 때문에 더 어렵다.
  - modality 다양성
  - 정상과 비정상의 미묘한 차이
  - 높은 라벨링 비용
- 이 논문은 왜 CNN이 이런 문제에 적합한지 정리한다.

---

## 이 논문의 핵심 기여

- 의료영상 분석 문제를 네 가지 축으로 정리한다.
  - segmentation
  - abnormality detection and classification
  - computer-aided diagnosis
  - image retrieval
- CNN의 작동 원리와 의료영상 적합성을 교육적으로 설명한다.
- 다양한 modality에서 CNN이 잘 작동한다는 점을 표로 정리한다.
- 데이터 부족, 3D 처리, black-box 문제 같은 한계를 함께 논의한다.

---

## 왜 CNN이 중요한가

- 기존 방식:
  - hand-crafted feature
  - separate classifier
- CNN 방식:
  - feature learning과 decision learning을 end-to-end로 결합
- 핵심 이점:
  - local receptive field
  - shared weights
  - hierarchical representation learning
  - end-to-end optimization
- 의료영상처럼 feature engineering 비용이 큰 영역에서 특히 유리하다.

---

## CNN 기본 원리

- convolution + nonlinearity
- pooling
- hierarchical abstraction
- regularization
  - dropout
  - batch normalization
- data augmentation
- 논문의 메시지:
  - 초기 층은 edge와 texture를 보고
  - 상위 층은 장기/병변 수준의 추상 표현을 학습한다.

---

## 응용 1: Segmentation

- 대표 과제:
  - brain tumor segmentation
  - lesion segmentation
  - prostate segmentation
- 반복되는 설계:
  - patch-based CNN
  - cascaded architecture
  - 3D CNN
  - CRF 후처리
- 핵심 문제:
  - class imbalance
  - sparse annotation
  - 3D context 활용

---

## 응용 2: Detection / Classification / CAD

- 대표 예시:
  - lung texture classification
  - thyroid nodule diagnosis
  - breast cancer diagnosis
  - diabetic retinopathy
  - Alzheimer disease classification
- 이 논문이 강조하는 점:
  - CNN은 특정 질환용 트릭이 아니라
    여러 modality에서 공통적으로 작동하는 표현학습 도구다.

---

## 응용 3: Retrieval

- CNN feature는 retrieval에서도 유용하다.
- 의미:
  - CNN은 분류기뿐 아니라 representation learner다.
- multimodal medical image retrieval, radiographic image retrieval 같은 작업에서
  좋은 결과가 보고되었다.
- 이 지점은 이후 metric learning과 foundation model 흐름의 전조처럼 볼 수 있다.

---

## 이 논문이 보여 주는 정량 메시지

- 논문은 다양한 task에서 CNN이 경쟁력 있음을 요약한다.
- 대표 예:
  - body part recognition on CT: 92.23%
  - thyroid nodule diagnosis: 83%
  - breast cancer diagnosis: 82.43%
  - Alzheimer multi-class classification: 98.88%
  - radiographic image retrieval: 97.79%
- 핵심은 숫자 자체보다
  **CT, MRI, ultrasound, pathology, fundus 등 여러 modality에 공통 적용 가능**하다는 점이다.

---

## 전통적 기법과의 대비

- ILD classification에서 CNN은 HOG + SVM 계열보다 우수했다.
- body organ recognition에서도 CNN이 RF, kNN, SVM 계열보다 우수했다.
- 따라서 이 논문은
  CNN이 단지 "deep learning hype"가 아니라
  실제로 hand-crafted feature를 넘어서는 전환점이라고 본다.

---

## 한계

- labeled medical data 부족
- 3D CNN의 높은 계산 비용
- black-box 문제
- modality-specific noise와 artifact
- volumetric data 처리의 어려움
- 흥미로운 점:
  - 2018년 논문이지만 이 문제들 대부분은 지금도 남아 있다.

---

## 이후 연구를 예고한 포인트

- transfer learning
- medical-domain pretraining
- data augmentation
- GAN 기반 synthetic data
- deeper network
- 2D/3D hybrid architecture
- 이 제안들은 이후 self-supervised learning, foundation model, 3D transformer 흐름으로 이어졌다.

---

## 강점

- 의료영상 분석에서 CNN이 왜 표준 후보가 되었는지 잘 설명한다.
- segmentation, diagnosis, retrieval을 하나의 표현학습 프레임으로 묶어 준다.
- 연구 입문자에게 역사적 출발점을 제공한다.

---

## 한계와 현재 관점

- 오늘 기준으로는 오래된 리뷰다.
- 다루지 못하는 것:
  - self-supervised learning
  - foundation models
  - vision-language models
  - fairness / privacy / calibration
- 따라서 최신 SOTA 리뷰가 아니라
  **CNN 기반 의료영상 딥러닝의 초기 정리 문서**로 읽어야 한다.

---

## 발표용 핵심 메시지

- 이 논문은 의료영상 분석에서 CNN이 "가능성"이 아니라
  **표준 후보**가 되던 시점을 보여 준다.
- end-to-end representation learning이 hand-crafted feature를 대체한 이유를 설명한다.
- 모델 계열은 바뀌었지만
  데이터 부족, 3D 비용, domain shift, black-box 문제는 지금도 유효하다.

---

## 결론

- `Medical Image Analysis using Convolutional Neural Networks: A Review`는
  의료영상 딥러닝의 출발점과 문제의식을 정리한 전환기 리뷰다.
- 최신 방법론 문헌은 아니지만,
  왜 이후 연구가 transfer learning, 3D 모델, 대규모 pretraining 방향으로 전개됐는지 이해하는 데 여전히 가치가 있다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/medical_image_analysis_using_convolutional_neural_networks_a_review_slide.md>
