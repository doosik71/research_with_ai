---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# MedMAE: A Self-Supervised Backbone for Medical Imaging Tasks

- Anubhav Gupta, Islam Osman, Mohamed S. Shehata, John W. Braun
- arXiv 2024
- Medical-domain MAE pretraining as a reusable backbone

---

## 문제 배경

- 의료영상 모델은 여전히 ImageNet 사전학습에 많이 의존한다.
- 하지만 의료영상은 자연영상과 크게 다르다.
  - 촬영 장비
  - 해부학 구조
  - 질감
  - grayscale 중심 분포
- 논문의 출발점:
  - 자연영상 전이는 구조적 한계가 있으므로
    **의료영상 자체로 사전학습한 백본**이 필요하다.

---

## 이 논문의 핵심 기여

- 대규모 비라벨 의료영상 데이터셋 `MID`를 구축한다.
- ViT-B 기반 MAE를 의료영상으로 1000 epoch 사전학습해 `MedMAE`를 만든다.
- 네 가지 다운스트림 과제에서 일반 MAE와 ImageNet 사전학습 모델보다 나은 성능을 보고한다.
- 핵심 메시지:
  - 새 objective보다 **도메인 적합한 사전학습 데이터**가 중요하다.

---

## MID 데이터셋

- 다양한 공개 저장소에서 비라벨 의료영상을 수집한다.
- 포함 범위:
  - CT
  - MRI
  - X-ray
  - 여러 해부학 부위
- 설계 의도:
  - 의료 도메인 시각 특성을 충분히 반영할 규모
  - 모달리티와 해부학 다양성 확보
  - 인위적 증강보다 원천 데이터 다양성 중시

---

## MedMAE 아키텍처

- 기본 구조는 표준 MAE와 유사하다.
- 설정:
  - backbone: ViT-B
  - 75% patch masking
  - visible patch만 인코더 입력
  - 디코더가 전체 이미지 복원
- 차별점은 구조보다
  **무엇으로 사전학습했는가**에 있다.

---

## 학습 설정의 의미

- 사전학습 기간: 1000 epochs
- grayscale medical images 중심
- 고정 입력 크기
- 즉, 자연영상용 color statistics가 아니라
  의료영상 분포에 맞춘 표현을 직접 학습한다.
- 발표 포인트:
  - MedMAE는 "새로운 SSL 원리"보다
    **medical-domain MAE pipeline**에 가깝다.

---

## 다운스트림 과제

- CT / MRI scanner quality control
- breast cancer prediction
- pneumonia detection
- polyp segmentation
- 이 조합이 의미하는 바:
  - 분류만이 아니라
  - 품질관리, 탐지, 분할까지
    하나의 백본이 얼마나 범용적인지 보려는 시도다.

---

## 주요 결과

- CT 품질관리: `MedMAE 0.902` vs `MAE 0.785`
- MRI 품질관리: `MedMAE 0.856` vs `MAE 0.743`
- 유방암 예측: `MedMAE 0.932` vs `MAE 0.843`
- 폐렴 탐지: `MedMAE 0.701` vs `MAE 0.679`
- 폴립 분할: `MedMAE 0.714` vs `MAE 0.646`
- 네 과제 모두에서 일관되게 우세했다.

---

## 결과 해석

- 자연영상 기반 특징보다
  의료영상 기반 특징이 장비 품질, 질환 패턴, 병변 구조를 더 직접적으로 포착한다.
- 하나의 의료영상 전용 백본이
  분류와 분할을 모두 지원할 수 있음을 보였다.
- 아키텍처 변화보다
  **사전학습 데이터의 도메인 일치**가 중요하다는 점을 실험적으로 보여 준다.

---

## 강점

- 문제 설정이 매우 명확하다.
- MAE라는 단순한 SSL 틀을 그대로 쓰므로 확장성과 재현성이 높다.
- 서로 다른 성격의 다운스트림 과제를 포함해 범용성을 어느 정도 보여 준다.
- 의료영상 전용 사전학습의 필요성을 직접적으로 설득한다.

---

## 한계

- 평가 과제는 넓지만 깊이는 제한적이다.
- 일부 핵심 과제가 private dataset 기반이라 재현성이 약하다.
- 기여의 대부분은 새 학습 원리보다 데이터셋 구축에서 나온다.
- CT/MRI를 포함하지만 학습 단위는 사실상 2D에 가깝다.
- 논문과 저장소에서 `MID` / `LUMID` 명칭이 혼용되는 점도 아쉽다.

---

## 현재 시점에서의 해석

- MedMAE는 초거대 의료 foundation model의 경쟁자라기보다
  **의료영상 전용 backbone 사전학습의 초기형**에 가깝다.
- 중요한 교훈은 다음이다.
  - 자연영상 전이보다 의료 도메인 자체의 비라벨 데이터 수집이 우선일 수 있다.
  - 단순한 SSL 구조도 도메인 맞춤 데이터와 결합하면 강한 출발점이 된다.

---

## 발표용 핵심 메시지

- 이 논문의 핵심 기여는 architecture novelty보다
  **medical-domain pretraining 데이터**다.
- MedMAE는 "의료영상용 MAE 백본"이 실제로 유효하다는 것을 보여 준다.
- 향후 확장은 2D를 넘어
  3D, multimodal, report-image, clinical metadata 결합 방향으로 이어져야 한다.

---

## 결론

- `MedMAE`는 의료영상 자체로 사전학습한 self-supervised backbone이
  여러 다운스트림 과제에서 일관된 이점을 줄 수 있음을 보인 논문이다.
- 완전히 새로운 학습 원리보다는
  의료영상 전용 사전학습 파이프라인의 실용적 출발점으로 읽는 편이 정확하다.
- 데이터 중심 의료 foundation 연구의 근거 문헌으로 의미가 있다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/medmae_a_self_supervised_backbone_for_medical_imaging_tasks_slide.md>
