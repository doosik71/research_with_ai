---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision

- Mohammad Reza Hosseinzadeh Taher et al.
- DART @ MICCAI 2023 / 2024
- anatomy-aware self-supervision으로 의료영상 foundation model을 다시 정의한 작업

---

## 문제 배경

- 의료영상 분야에서는 NLP나 자연영상처럼 foundation model이 빠르게 정착하지 못했다.
- 이 논문은 원인을 단순히 데이터 규모 부족으로만 보지 않는다.
- 핵심 질문은 `무엇을 학습하느냐`다.
- 기존 SSL은 patch 복원이나 instance discrimination에 머무는 경우가 많다.
- 그러나 의료영상의 본질은 해부학적 구조이며,
  그 구조는 `locality`와 `compositionality`를 가진다.

---

## 이 논문의 핵심 주장

- 의료영상 foundation model은 anatomy를 이해하도록 학습되어야 한다.
- 표현 공간은 서로 다른 해부학 구조를 구분할 수 있어야 한다.
- 동시에 작은 구조와 큰 구조의 part-whole 관계도 반영해야 한다.
- 논문은 이 두 속성을 각각 `locality`, `compositionality`로 정식화한다.
- 즉, foundation model의 핵심을 `규모`보다 `anatomy-centered objective`로 재정의한다.

---

## 핵심 기여

- anatomy의 계층 구조를 반영한 self-supervised training strategy를 제안한다.
- embedding space가 실제로 locality와 compositionality를 가지는지 검증하는 해석 프레임을 제시한다.
- 제안 모델 `Adam`과 embedding `Eve`를 통해
  여러 downstream task와 few-shot setting에서 strong transfer를 보인다.
- 성능보다 더 중요한 공헌은 `의료 foundation model이 배워야 할 것`을 명시했다는 점이다.

---

## Adam과 Eve

- `Adam`: autodidactic dense anatomical model
- `Eve`: Adam이 생성하는 dense semantic embedding
- 이름 자체가 논문의 철학을 드러낸다.
- Adam은 라벨 없이 해부학 구조를 학습한다.
- Eve는 그 결과로 형성된 anatomy-aware embedding space다.
- 발표에서는 `Adam은 학습 프레임워크, Eve는 표현 공간`으로 설명하면 충분하다.

---

## 핵심 개념 1: Locality

- 서로 다른 해부학 구조는 embedding space에서 분리되어야 한다.
- 같은 구조는 환자가 달라도 가까워야 한다.
- 예를 들어 심장과 폐, 쇄골과 주변 배경은 representation 수준에서 구분돼야 한다.
- locality는 anatomy-aware representation의 최소 조건이다.
- 논문은 이를 landmark 기반 시각화와 patch clustering으로 검증한다.

---

## 핵심 개념 2: Compositionality

- 큰 해부학 구조는 작은 부분 구조의 조합으로 이해될 수 있어야 한다.
- 즉, part-whole relation이 embedding space에 반영돼야 한다.
- patch를 sub-patch로 분해했을 때
  전체 representation과 부분 representation 사이의 관계가 정렬돼야 한다.
- 이 논문은 medical SSL이 이 compositional structure를 배워야 한다고 본다.

---

## Anatomy Decomposer

- 첫 번째 핵심 모듈은 `Anatomy Decomposer (AD)`다.
- 입력 이미지를 coarse-to-fine 방식으로 점진적으로 분해한다.
- granularity level이 커질수록 더 세밀한 anatomical unit을 만든다.
- 목적은 모델이 해부학 구조를 계층적으로 보도록 만드는 것이다.
- 단순 patch sampling이 아니라 anatomy hierarchy를 모방한 curriculum 역할을 한다.

---

## Purposive Pruner

- 두 번째 핵심 모듈은 `Purposive Pruner (PP)`다.
- 일반 contrastive learning에서는 사실상 비슷한 anatomical structure가 negative로 들어가
  semantic collision이 일어날 수 있다.
- PP는 memory bank에서 anchor와 지나치게 유사한 sample을 제거한다.
- 그 뒤 pruned memory bank로 InfoNCE loss를 계산한다.
- 즉, anatomy적으로 가까운 구조를 잘못된 negative로 밀어내지 않게 하는 장치다.

---

## 학습 철학

- anatomy는 계층적이므로 학습도 coarse-to-fine이어야 한다.
- 같은 anatomical structure는 환자가 달라도 가까워야 한다.
- part와 whole 사이의 관계도 embedding에 반영돼야 한다.
- 이 논문은 의료 SSL의 목표를
  `이미지 복원`이나 `전역 분별`이 아니라 `해부학 이해`로 이동시킨다.

---

## 실험 설정

- pretraining 데이터는 `ChestX-ray14`, `EyePACS`를 사용한다.
- backbone은 `ResNet-50`이다.
- 입력 크기는 `224 x 224`, granularity level은 최대 4다.
- 비교 대상은 `MoCo-v2`, `DenseCL`, `VICRegL`, `PCRL`, `DiRA`, `Medical-MAE`, `SimMIM` 등이다.
- downstream 평가는 분류, 분할, few-shot task를 포함한 9개 과제로 구성된다.

---

## 주요 결과

- Adam은 여러 downstream task에서 기존 SSL baseline보다 우수하거나 경쟁력 있는 성능을 보인다.
- 특히 dense SSL이나 medical SSL baseline 대비 일관된 전이 성능 향상이 핵심이다.
- 논문 메시지는 단순하다.
- anatomy-aware objective가 generic SSL objective보다
  의료영상 전이에 더 잘 맞을 수 있다.

---

## Few-shot에서의 강점

- 이 논문의 가장 설득력 있는 결과는 few-shot segmentation이다.
- `SCR-Heart`와 `SCR-Clavicle`에서 3, 6, 12, 24-shot 전 구간에서 우세하다.
- 예시:
- SCR-Heart 3-shot Dice `84.35`
- SCR-Heart 24-shot Dice `90.45`
- SCR-Clavicle 3-shot Dice `66.69`
- SCR-Clavicle 24-shot Dice `84.76`
- 적은 라벨에서도 anatomy-aware pretraining이 annotation efficiency를 높인다는 해석이 가능하다.

---

## Locality 검증

- 논문은 ChestX-ray14 이미지에 landmark를 지정하고
  해당 patch embedding을 t-SNE로 시각화한다.
- Adam은 서로 다른 anatomical landmark를 더 분리된 cluster로 배치한다.
- 즉, 같은 구조는 모으고 다른 구조는 분리하는 locality 특성이 더 강하다.
- 이 부분은 단순 accuracy 비교를 넘어
  representation이 무엇을 배웠는지 보여주는 장점이다.

---

## Compositionality 검증

- patch를 2개, 3개, 4개 sub-patch로 나눈 뒤
  전체 patch embedding과 부분 embedding 집계 결과를 비교한다.
- Adam은 baseline보다 cosine similarity가 더 높다.
- 이는 전체 구조와 부분 구조의 표현이 더 일관되게 연결된다는 뜻이다.
- 즉, anatomy의 part-whole 관계가 embedding space에 반영됐다는 주장을 뒷받침한다.

---

## 강점

- 의료 foundation model의 설계 원리를 제시한다.
- anatomy-aware objective를 명시적으로 정의한다.
- 성능뿐 아니라 locality와 compositionality라는 해석 축을 제안한다.
- few-shot efficiency가 강하다.
- 향후 anatomy-driven medical FM 연구의 출발점으로 읽을 수 있다.

---

## 한계

- 제목과 달리 오늘날의 대규모 foundation model과는 거리가 있다.
- backbone이 `ResNet-50`에 머문다.
- Anatomy Decomposer는 실제 anatomy atlas가 아니라 규칙 기반 분해에 가깝다.
- 검증은 chest X-ray 중심이며 CT, MRI, 3D volumetric 일반화는 제한적이다.
- 따라서 이는 완성된 FM이라기보다 `FM을 향한 설계 원형`으로 보는 편이 정확하다.

---

## 현재 시점에서의 의미

- 이 논문은 `의료 foundation model은 anatomy를 배워야 한다`는 명제를 정교하게 제시했다.
- 오늘날 vision-language medical FM이나 3D medical FM이 등장한 뒤에도
  이 메시지는 여전히 유효하다.
- 규모를 키우는 것만으로는 부족하고,
  의료 도메인 고유 구조를 objective에 넣어야 한다는 뜻이다.
- 발표에서는 이 논문을 `anatomy-aware medical FM의 철학적 출발점`으로 잡으면 된다.

---

## 발표용 핵심 메시지

- foundation model의 본질은 크기보다 학습 목표에 있다.
- 의료영상에서는 anatomy locality와 compositionality를 반영한 SSL이 중요하다.
- Adam/Eve는 거대 범용 모델이 아니라 anatomy-centered representation learning 설계 원형이다.
- few-shot 전이 성능이 이 철학의 실용적 가치를 보여준다.

---

## 결론

- `Towards Foundation Models Learned from Anatomy in Medical Imaging via Self-Supervision`은
  의료 foundation model을 anatomy 중심 objective의 문제로 재정의한 논문이다.
- `Anatomy Decomposer`와 `Purposive Pruner`를 통해
  locality와 compositionality를 embedding space에 반영하려 한다.
- 완성형 FM보다는 방향 제시에 가깝지만,
  medical SSL이 무엇을 배워야 하는지 명확히 보여준다는 점에서 가치가 크다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/towards_foundation_models_learned_from_anatomy_in_medical_imaging_via_self_supervision_slide.md>
