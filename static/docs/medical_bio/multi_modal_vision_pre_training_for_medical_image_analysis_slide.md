---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Multi-modal Vision Pre-training for Medical Image Analysis

- Shaohao Rui et al.
- CVPR 2025
- BrainMVP: Multi-modal Vision Pre-training for Brain Image Analysis using Multi-parametric MRI

---

## 문제 배경

- 기존 의료영상 self-supervised pre-training은 대체로 uni-modal 설정에 머문다.
- 그러나 실제 brain MRI는 multi-parametric MRI(mpMRI)처럼 여러 modality가 함께 존재한다.
- 각 modality는 같은 해부학 구조를 다른 신호 공간에서 관찰한 결과다.
- 따라서 핵심 질문은 두 가지다.
- cross-modal correlation을 pre-training 단계에서 어떻게 학습할 것인가?
- missing modality가 있는 현실적 환경을 어떻게 견딜 것인가?

---

## 핵심 기여

- 8개 MRI modality를 포함한 대규모 brain mpMRI pre-training 데이터를 구축했다.
- 규모는 3,755 cases, 16,022 scans, 240만 장 이상 이미지다.
- `BrainMVP`를 제안했다.
- 구성은 `cross-modal reconstruction`, `modality-wise data distillation`, `modality-aware contrastive learning`이다.
- 10개 downstream task에서 일반화와 label efficiency를 함께 보였다.

---

## 이 논문의 실제 초점

- 제목은 넓지만 실제 범위는 일반 의료영상 전반이 아니다.
- 핵심은 `brain mpMRI 전용 multi-modal pre-training`이다.
- 즉, 이 논문은 범용 multimodal medical foundation model보다
  `동일 환자의 여러 MRI modality 사이 관계를 어떻게 학습할 것인가`에 집중한다.
- 발표에서는 이 점을 분명히 짚어야 결과 해석이 정확해진다.

---

## BrainMVP 개요

- 목표는 modality 간 상보성을 pretext task에 직접 넣는 것이다.
- backbone은 `UniFormer`를 사용한다.
- 입력은 고정 multi-channel 결합이 아니라 `single-channel modality image input` 중심이다.
- 이는 test-time missing modality 문제를 의식한 설계다.
- 전체 프레임은 세 개의 proxy task가 결합된 구조다.

---

## 1. Cross-Modal Reconstruction

- 한 modality를 다른 modality 정보로 복원하도록 학습한다.
- 일반적인 masked image modeling보다 한 단계 더 강한 cross-modal 복원 문제다.
- 이 과정에서 다음 능력을 함께 학습한다.
- modality-invariant structure 파악
- modality-specific difference 파악
- cross-modal fusion capability 확보
- 핵심은 복원 자체보다 `modality 관계를 학습시키는 장치`라는 점이다.

---

## 2. Modality-wise Data Distillation

- 각 modality마다 학습 가능한 `modality template`를 둔다.
- 이 template는 modality 수준의 공통 구조를 압축한 representation 역할을 한다.
- 역할은 두 가지다.
- pre-training 단계에서 modality별 구조 통계를 응축한다.
- downstream 단계에서 upstream representation과 downstream adaptation을 이어준다.
- 이 모듈이 BrainMVP를 단순 reconstruction 조합 이상으로 만든다.

---

## 3. Modality-aware Contrastive Learning

- 같은 case에서 나온 다양한 view는 가깝게 유지한다.
- 다른 case는 멀어지게 만든다.
- 동시에 modality-aware 제약을 넣어 semantic consistency를 유지한다.
- 결과적으로 acquisition protocol이나 dataset 차이가 있어도
  더 안정적인 representation을 얻도록 설계됐다.
- 논문 해석상 MCL은 최종 generalization 성능을 밀어 올리는 마지막 축이다.

---

## 데이터와 학습 설정

- pre-training 데이터는 5개 공개 brain mpMRI dataset으로 구성된다.
- 포함 데이터:
- `BraTS2021`
- `BraTS2023-SSA`
- `BraTS2023-MEN`
- `BrainAtlas`
- `UCSF-PDGM`
- 구현은 PyTorch, MONAI 기반이며 optimizer는 AdamW를 사용한다.
- 논문 기준 학습 자원은 `8 x RTX 4090`이다.

---

## Downstream 평가 범위

- 총 10개 공개 benchmark로 평가한다.
- Segmentation 6개:
- `BraTS-PED`
- `BraTS2023-MET`
- `ISLES22`
- `MRBrainS13`
- `UPENN-GBM`
- `VSseg`
- Classification 4개:
- `BraTS2018`
- `ADNI`
- `ADHD-200`
- `ABIDE-I`

---

## 주요 결과

- segmentation Dice 향상 폭은 `0.28% ~ 14.47%`다.
- classification accuracy 향상 폭은 `0.65% ~ 18.07%`다.
- 대표 결과:
- BraTS-PED Dice `72.52 -> 76.80`
- UPENN-GBM Dice `90.01`
- BraTS2018 AUC `0.9452`
- ADNI ACC `0.6765`
- 여러 medical SSL baseline 대비 전반적으로 우세한 결과를 보인다.

---

## Label Efficiency

- 이 논문의 실용적 강점은 평균 성능뿐 아니라 label efficiency다.
- 적은 annotation만으로도 강한 downstream 성능을 유지한다.
- 예시:
- BraTS-PED 20% label Dice `66.41`
- VSseg 20% label Dice `70.39`
- UPENN-GBM 20% label Dice `86.82`
- 의료영상처럼 라벨 비용이 큰 환경에서는 매우 중요한 장점이다.

---

## Ablation이 말해주는 것

- BraTS-PED Dice 기준:
- baseline `72.52`
- `+ CMR` -> `75.16`
- `+ CMR + MD` -> `75.87`
- `+ CMR + MD + MCL` -> `76.80`
- 즉, 세 모듈이 누적적으로 기여한다.
- 특히 CMR이 큰 폭의 초기 상승을 만들고,
  MD와 MCL이 representation 품질과 일반화를 더 밀어 올린다.

---

## 강점

- 문제 설정이 실제 mpMRI 환경과 잘 맞는다.
- missing modality를 고려한 입력 설계가 현실적이다.
- segmentation과 classification을 모두 포함해 검증 범위가 넓다.
- `modality-wise distillation`은 upstream-downstream 연결 관점에서 흥미롭다.
- label efficiency가 좋아 임상 annotation 비용 문제와 직접 연결된다.

---

## 한계

- 제목과 달리 실제 적용 범위는 거의 brain MRI에 한정된다.
- multimodal이라 해도 image-text가 아니라 image-image 범주다.
- modality template의 해석 가능성은 아직 제한적이다.
- UniFormer backbone 효과와 objective 효과가 완전히 분리되지는 않는다.
- 8개의 RTX 4090이 필요한 pre-training은 계산 비용 장벽이 높다.

---

## 현재 시점에서의 해석

- 이 논문은 의료 AGI나 범용 multimodal foundation model 논문은 아니다.
- 대신 `의료영상 pre-training은 modality를 따로 학습하는 것만으로는 부족하다`는 점을 강하게 보여준다.
- 특히 같은 환자에서 얻은 서로 다른 modality 사이 관계를
  pretext task 수준에서 직접 강제하는 접근이 효과적이라는 증거다.
- 이후 확장 방향은 자연스럽다.
- `CT-PET`, `multi-phase CT`, `MRI-ultrasound`, `image-text/report` 조합으로 이어질 수 있다.

---

## 발표용 핵심 메시지

- BrainMVP의 본질은 `cross-modal relation learning`이다.
- 핵심 성과는 단순 성능 향상보다 `generalization + label efficiency`에 있다.
- multi-modal medical pre-training은
  modality를 병렬로 모으는 것보다 modality 관계를 구조적으로 학습시키는 쪽이 더 중요하다.
- 이 논문은 그 출발점을 brain mpMRI에서 설득력 있게 보여준다.

---

## 결론

- `Multi-modal Vision Pre-training for Medical Image Analysis`는
  실제로는 brain mpMRI용 `BrainMVP`를 제안한 논문이다.
- `CMR + MD + MCL` 조합으로 modality 상보성과 missing modality 현실을 함께 다룬다.
- 범용성은 아직 제한적이지만,
  medical multi-modal pre-training의 방향성을 잘 제시한 작업으로 볼 수 있다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/multi_modal_vision_pre_training_for_medical_image_analysis_slide.md>
