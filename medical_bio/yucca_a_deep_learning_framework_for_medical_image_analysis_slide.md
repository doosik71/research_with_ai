---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Yucca: A Deep Learning Framework For Medical Image Analysis

- Sebastian Nørgaard Llambias et al.
- arXiv 2024
- 의료영상 연구를 더 재현 가능하고 실험 친화적으로 만들기 위한 프레임워크 논문

---

## 문제 배경

- 의료영상 딥러닝 연구에서는 강한 baseline과 빠른 재현이 모두 중요하다.
- `nnU-Net`은 out-of-the-box 성능과 자동화가 강하지만 구조가 단단히 결합돼 있다.
- `MONAI`는 유연하지만 초보자나 빠른 재현이 필요한 연구에는 설정 부담이 크다.
- 실제 병목은 모델 하나보다도
  데이터 정리, 전처리, split, training, inference, evaluation을 일관되게 관리하는 일이다.
- Yucca는 이 간극을 메우려는 프레임워크다.

---

## 이 논문의 핵심

- 새로운 SOTA 모델 제안이 아니라 `프레임워크 설계`가 핵심이다.
- 목표는 nnU-Net의 강한 기본값과 MONAI의 유연성을 절충하는 것이다.
- 이를 위해 `Functional`, `Modules`, `Pipeline`의 세 계층 구조를 제안한다.
- 즉, 좋은 모델보다 `좋은 연구 작업 환경`을 제공하려는 논문이다.

---

## 핵심 기여

- 의료영상 워크플로를 위한 3계층 구조를 제안한다.
- task conversion부터 preprocessing, training, inference, evaluation까지 end-to-end 파이프라인을 제공한다.
- PyTorch와 PyTorch Lightning 기반으로 실험 관리와 재현성을 체계화한다.
- segmentation, detection, classification이 섞인 다양한 의료영상 과제에 적용 가능함을 보인다.
- 핵심 가치는 모델 혁신보다 `실험 생산성`이다.

---

## 3계층 구조 개요

- `Functional tier`: 상태가 없는 순수 함수 집합
- `Modules tier`: 함수와 PyTorch/PyTorch Lightning 로직을 묶는 중간 계층
- `Pipeline tier`: end-to-end workflow를 실행하는 상위 계층
- 이 구조는 사용자가 개입할 수준을 선택할 수 있게 한다.
- 초보자는 Pipeline을 쓰고,
  숙련자는 Functional/Modules 수준에서 세밀하게 수정할 수 있다.

---

## 1. Functional Tier

- 가장 아래 계층이다.
- 파일 입출력, 경로 조작, 배열 처리, filtering, bounding box 계산 같은 stateless building block을 제공한다.
- 장점은 단순하다.
- 테스트가 쉽다.
- 재사용이 쉽다.
- 특정 파이프라인 밖에서도 독립적으로 쓸 수 있다.
- 즉, 프레임워크의 기초 공구함 역할이다.

---

## 2. Modules Tier

- Functional tier를 감싸는 객체 지향 계층이다.
- transform, loss/metric wrapper, callback, network, DataModule, LightningModule 등이 여기에 속한다.
- 역할은 함수 수준 로직과 학습 시스템을 연결하는 것이다.
- 복잡한 학습 스크립트를 모듈 단위로 분리해
  실험 조립과 교체를 더 쉽게 만든다.

---

## 3. Pipeline Tier

- Yucca의 end-to-end workflow 계층이다.
- 사용 경험은 nnU-Net처럼 간단하게 유지하면서도
  내부 구성은 더 교체 가능하게 설계한다.
- 주요 단계는 네 개다.
- task conversion
- preprocessing
- model training
- inference and evaluation
- 이 계층이 실제 연구 생산성 향상과 가장 직접적으로 연결된다.

---

## Task Conversion

- raw dataset을 Yucca 형식으로 정리하는 단계다.
- 이미지-라벨 매칭, split, case 검사, 메타데이터 정리 등을 수행한다.
- 중요한 점은 test loader와 evaluation 단계까지 고려해
  데이터 누수나 split 오류를 줄이도록 설계했다는 것이다.
- 재현성 문제를 프레임워크 수준에서 줄이려는 의도가 분명하다.

---

## Preprocessing

- `Planner`와 `Preprocessor`가 중심이다.
- Planner는 데이터셋 통계를 보고 전처리 계획을 만든다.
- Preprocessor는 그 계획을 실행한다.
- 예를 들어 spacing, resampling, normalization 같은 설정을 자동화한다.
- 논문이 강조하는 포인트는
  전처리를 deterministic하게 유지하고 augmentation은 학습 단계에서 따로 적용한다는 점이다.

---

## Training

- training 단계에서는 `manager` 개념이 중요하다.
- 사용자가 일일이 다 지정하지 않아도
  데이터 통계와 설정에 맞춰 실험 구성을 자동으로 채운다.
- 예:
- random seed
- checkpoint / logging
- split 관리
- augmentation pipeline
- early stopping, scheduler 등
- 즉, 실수하기 쉬운 반복 작업을 시스템으로 흡수한다.

---

## Inference and Evaluation

- inference에서도 학습 시와 동일한 전처리 흐름을 유지한다.
- patch-based model은 sliding-window inference를 지원한다.
- 이후 inverse transform을 거쳐 원래 좌표계로 복원하고 평가를 수행한다.
- 이 단계까지 프레임워크가 관리하면
  후처리 불일치나 평가 스크립트 오류를 줄일 수 있다.

---

## 적용 사례

- 논문은 하나의 benchmark 승부보다
  프레임워크가 여러 과제에 쓰였다는 점을 강조한다.
- 예시:
- cerebral microbleeds segmentation / detection
- white matter hyperintensity segmentation / detection
- hippocampus segmentation
- stroke, multiple sclerosis 관련 lesion segmentation, detection, classification
- 메시지는 `한 프레임워크로 여러 2D/3D 의료 과제를 안정적으로 지원한다`는 것이다.

---

## 강점

- 자동화와 유연성의 절충점이 명확하다.
- 데이터부터 평가까지 전체 의료영상 workflow를 다룬다.
- 실험 재현성과 버전 관리 문제를 프레임워크 차원에서 줄인다.
- 초보자와 숙련자 모두 다른 계층에서 활용할 수 있다.
- 모델보다 연구 생산성 향상에 직접 기여한다.

---

## 한계

- 프레임워크 논문이라 대규모 정량 비교와 ablation은 제한적이다.
- 기여가 알고리즘보다 공학 구조에 치우쳐 있어 평가가 갈릴 수 있다.
- nnU-Net 대비 성능-유연성 trade-off를 더 정량적으로 보여줬으면 좋았을 수 있다.
- 진짜 가치는 시간이 지나며 커뮤니티가 얼마나 채택하느냐에 달려 있다.

---

## 현재 시점에서의 의미

- 의료영상 AI에서는 좋은 모델만큼 좋은 실험 인프라도 중요하다.
- Yucca는 이 점을 정면으로 다룬다.
- 특히 foundation model이나 복잡한 실험이 늘어날수록
  재현 가능한 파이프라인의 중요성은 더 커진다.
- 그래서 이 논문은 새로운 architecture 문헌보다
  `연구 생산성 프레임워크` 문헌으로 읽는 것이 맞다.

---

## 발표용 핵심 메시지

- Yucca의 핵심은 모델이 아니라 workflow engineering이다.
- nnU-Net의 자동화와 MONAI의 유연성 사이 절충점을 노린다.
- `Functional -> Modules -> Pipeline` 구조가 사용성과 확장성을 동시에 확보한다.
- 의료영상 연구에서는 성능만큼 reproducibility와 experiment hygiene가 중요하다는 점을 보여준다.

---

## 결론

- `Yucca: A Deep Learning Framework For Medical Image Analysis`는
  의료영상 딥러닝 연구를 더 재현 가능하고 실험 친화적으로 만들기 위한 프레임워크 논문이다.
- 핵심 공헌은 3계층 구조와 end-to-end workflow 관리에 있다.
- 알고리즘 혁신보다 `연구 인프라 혁신`에 가까우며,
  의료영상 연구 생산성을 높이는 실용적 가치가 크다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/yucca_a_deep_learning_framework_for_medical_image_analysis_slide.md>
