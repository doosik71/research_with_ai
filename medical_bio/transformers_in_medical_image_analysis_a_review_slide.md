---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Transformers in Medical Image Analysis: A Review

- Kelei He et al.
- 2022 arXiv / 2023 Intelligent Medicine
- 의료영상 Transformer 연구의 초기 구조를 정리한 review

---

## 문제 배경

- Transformer는 NLP에서 long-range dependency modeling으로 성공한 뒤 vision으로 확장됐다.
- 의료영상 분석은 classification, segmentation, detection, registration, reconstruction 등 과제가 매우 넓다.
- CNN은 의료영상에서 강력하지만 local receptive field 중심의 한계가 있다.
- Transformer는 global context modeling에 강하지만
  데이터 요구량, 계산 비용, 구조 설계 문제가 있다.
- 이 review는 Transformer가 의료영상에서 어떻게 쓰이기 시작했는지 지형도를 정리한다.

---

## 이 review의 핵심 기여

- 의료영상 Transformer 응용을 `classification`, `segmentation`, `image-to-image translation`, `detection`, `registration`, `video-based application`으로 정리한다.
- self-attention, ViT, DeiT, Swin Transformer 등의 배경을 의료영상 관점에서 다시 설명한다.
- `pure Transformer`와 `hybrid Transformer`를 구분해 비교한다.
- `weak supervision`, `multi-task learning`, `multi-modal learning`과의 연결도 다룬다.
- 데이터 부족, 계산 효율, 해석 가능성 같은 향후 과제를 정리한다.

---

## Transformer 기본 개념

- `Self-attention`: 토큰 간 관계를 직접 계산해 long-range dependency를 모델링한다.
- `Multi-head attention`: 여러 표현 subspace에서 관계를 병렬로 본다.
- `ViT`: 이미지를 patch sequence로 바꿔 Transformer encoder에 넣는다.
- `DeiT`: data-efficient training과 distillation을 통해 ViT 학습을 개선한다.
- `Swin Transformer`: window attention과 shifted window로 비용을 줄이고 계층 구조를 만든다.

---

## 의료영상에서 중요한 구조 구도

- 이 review의 중요한 분류는 task만이 아니다.
- 구조적으로 다음 구도를 본다.
- `pure Transformer`
- `hybrid Transformer`
- `Transformer + CNN`
- `Transformer + graph`
- 발표에서 핵심 메시지는 분명하다.
- 실제 의료영상에서는 pure ViT보다 hybrid 설계가 훨씬 많고 실용적이다.

---

## 1. Classification

- CT, X-ray, MRI, ultrasound, OCT, histopathology, fundus, graph-based brain analysis까지 폭넓게 다룬다.
- classification에서는 세 흐름이 보인다.
- ViT를 직접 적용하는 pure Transformer
- CNN과 ViT를 결합해 local + global을 함께 쓰는 hybrid 구조
- graph representation과 Transformer를 결합하는 방식
- 결론적으로 Transformer는 강한 backbone이 될 수 있지만,
  초기 의료영상에서는 hybrid 구조가 더 안정적이었다.

---

## 2. Segmentation

- segmentation은 이 review에서 가장 큰 비중을 차지하는 영역이다.
- abdominal multi-organ, cardiac, brain tumor, polyp, skin lesion, prostate 등 다양한 사례를 포괄한다.
- 구조는 크게 두 축으로 정리된다.
- `Hybrid Transformers`
- `Pure Transformers`
- 핵심 해석은 간단하다.
- 의료 segmentation은 local boundary와 global context가 모두 중요해서
  CNN encoder-decoder와 Transformer를 결합한 U-shape hybrid가 자연스럽다.

---

## 3. Image-to-Image Translation

- 이 범주에는 synthesis, reconstruction, super-resolution, denoising이 포함된다.
- Transformer는 modality 간 관계 모델링과 전역 구조 보존에 강점을 보인다.
- 특히 MRI synthesis나 multi-modal translation에서는 attention이 유용하다.
- 다만 paired data 부족과 inter-subject variability는 여전히 큰 문제다.
- 그래서 unsupervised 또는 data-efficient transformer translation이 중요한 방향으로 제시된다.

---

## 4. Detection과 Registration

- detection에서는 DETR 계열 아이디어가 의료영상으로 확장되기 시작한 시기다.
- 하지만 segmentation, classification만큼 성숙한 분야는 아니었다.
- registration에서는 moving-fixed image 사이 correspondence modeling에 self-attention이 유망하다.
- 특히 deformable registration에서 long-range relation이 장점으로 평가된다.
- 다만 review 시점 기준으로는 여전히 초기 탐색 단계에 가깝다.

---

## 5. Video-based Applications

- 이 review는 surgical phase recognition, tool detection, ultrasound video analysis 같은 비디오 기반 의료 응용도 다룬다.
- 이는 Transformer가 공간 관계뿐 아니라 시간적 관계 modeling에도 자연스럽게 확장된다는 점을 보여준다.
- 즉, 의료영상 Transformer는 정적 이미지 분석에만 머무르지 않는다.

---

## Learning Paradigm 관점

- 이 review의 강점은 task taxonomy에만 머무르지 않는 점이다.
- `Multi-task learning`: 여러 의료 과제를 함께 다루는 구조
- `Multi-modal learning`: imaging + clinical variable, OCT + VF 같은 결합
- `Weakly-supervised learning`: limited annotation 환경에서의 transformer 활용
- 결국 Transformer의 가치는 architecture novelty보다
  복잡한 의료 AI 학습 시나리오와 잘 맞는다는 데 있다.

---

## 핵심 해석 1: Hybrid가 기본값이었다

- review 전체를 보면 pure ViT가 화제였어도
  실제 의료영상에서는 hybrid 구조가 기본값처럼 쓰였다.
- 이유는 분명하다.
- 의료영상은 fine-scale boundary, local texture, small lesion에 민감하다.
- 그래서 convolution의 inductive bias를 버리기보다
  Transformer로 보완하는 방식이 훨씬 자연스러웠다.

---

## 핵심 해석 2: Segmentation과 Classification이 중심

- review 시점 기준 Transformer 확산은 segmentation과 classification이 주도했다.
- detection, registration, video, multi-task는 가능성을 보였지만 아직 초기였다.
- 즉, 이 문서는 Transformer가 의료영상 전반으로 퍼지기 시작한 시기의 지도에 가깝다.

---

## 남은 과제

- `데이터 부족`: Transformer는 여전히 큰 데이터와 pretraining에 의존한다.
- `계산 비용`: 특히 고해상도 3D 의료영상에서는 attention 비용이 크다.
- `해석 가능성`: clinical safety와 연결되는 설명 가능성이 부족하다.
- `고급 학습 시나리오`: weak supervision, multi-modal fusion, multi-task에서 더 많은 검증이 필요하다.
- 이 과제들은 지금도 상당 부분 유효하다.

---

## 현재 시점에서의 해석

- 이 review는 2022년 기준의 초기 Transformer 의료영상 지형도다.
- 따라서 `SAM`, `CLIP`, `medical VLM`, `diffusion-based foundation model` 이후 흐름은 반영하지 못한다.
- 그럼에도 가치가 있다.
- 왜 hybrid가 많았는지,
  어떤 task가 먼저 Transformer를 받아들였는지,
  어떤 병목이 처음부터 지적됐는지를 한 번에 볼 수 있기 때문이다.

---

## 발표용 핵심 메시지

- Transformer는 의료영상에서 단번에 CNN을 대체하지 않았다.
- 실제 흐름은 `hybridization`이었다.
- segmentation과 classification이 adoption을 주도했다.
- 데이터 효율, 계산 비용, 해석 가능성 문제는 초기에 이미 제기됐고 지금도 이어진다.
- 이 논문은 최신 방법 총정리보다 `초기 구조 이해용 review`로 읽는 것이 맞다.

---

## 결론

- `Transformers in Medical Image Analysis: A Review`는
  Transformer가 의료영상으로 확산되던 초기 시기의 구조를 체계적으로 정리한 문헌이다.
- 핵심 메시지는 pure ViT의 승리가 아니라
  `의료영상에서는 hybrid Transformer가 실질적 기본값이었다`는 점이다.
- 오늘 시점에서는 다소 초기 문헌이지만,
  의료영상 Transformer 연구의 출발 구조를 이해하는 데 여전히 유효하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/transformers_in_medical_image_analysis_a_review_slide.md>
