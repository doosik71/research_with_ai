---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Attention Mechanisms in Medical Image Segmentation: A Survey

- Yutong Xie, Bing Yang, Qingbiao Guan, Jianpeng Zhang, Qi Wu, Yong Xia
- arXiv 2023
- Survey of non-Transformer and Transformer attention for medical segmentation

---

## 문제 배경

- 의료영상 분할은 작은 병변, 흐린 경계, 잡음, 클래스 불균형을 동시에 다룬다.
- CNN은 local receptive field에 강하지만 전역 문맥 포착에는 한계가 있다.
- Transformer attention은 장거리 관계를 잘 보지만 계산량과 데이터 요구가 크다.
- 그래서 핵심 질문은 단순하다.
  - attention을 어디에, 어떤 방식으로 써야 실제 분할 성능에 도움이 되는가?

---

## 논문의 가장 큰 기여

- 이 논문은 attention 연구를 크게 두 갈래로 나눈다.
  - **Non-Transformer attention**
  - **Transformer attention**
- 그리고 두 갈래 모두를 같은 세 질문으로 읽게 만든다.
  - 무엇을 강조하는가
  - 어떻게 구현하는가
  - 어디에 적용하는가
- 이 점이 단순 모델 나열형 survey와 가장 다르다.

---

## Non-Transformer Attention

- CNN 기반 segmentation에 부착되는 attention block 계열이다.
- 주로 다음 요소를 선택적으로 강조한다.
  - channel attention
  - spatial attention
  - scale attention
  - edge / boundary attention
  - dual or mixed attention
- 핵심은 전역 reasoning보다 **지역적 중요도 재가중치와 경계 복원 강화**다.

---

## Non-Transformer Attention은 어디에 넣는가

- encoder 내부에 삽입
- skip connection 위에서 feature selection 수행
- decoder에서 coarse-to-fine refinement 수행
- multi-scale fusion 단계에서 weighting 수행
- boundary branch와 결합
- 발표 포인트:
  - attention의 효과는 종류만이 아니라
    **네트워크 어느 위치에 넣는가**에 크게 좌우된다.

---

## Transformer Attention

- 전역 문맥과 장거리 의존성을 모델링하며, 다음 상황에서 강점이 있다.
  - ROI localization
  - long-range dependency modeling
  - multi-organ segmentation
  - 3D volumetric segmentation
- 하지만 비용도 크다.
  - 메모리 사용량 증가
  - 작은 의료 데이터셋에서 과적합 위험
  - tokenization 과정의 fine boundary 손실 가능성

---

## Transformer 구현 패턴

- 논문은 의료 segmentation 구조를 다음처럼 분류한다.
  - hybrid encoder + CNN decoder
  - pure Transformer encoder + CNN decoder
  - CNN encoder + Transformer decoder
  - Transformer encoder + Transformer decoder
- 핵심 해석:
  - 당시 실전에서는 pure Transformer보다
    **CNN과 결합한 hybrid 구조가 더 현실적**이었다.

---

## 이 논문이 주는 설계 프레임

- `What to use`
  - channel, spatial, scale, boundary, self-attention 중 무엇이 필요한가
- `How to use`
  - encoder, skip path, decoder, fusion, boundary refinement 중 어디에 넣을 것인가
- `Where to use`
  - brain MRI, abdominal CT, retinal vessel, polyp, skin lesion 등 과제별로 다르게 설계할 것인가
- 이 프레임은 실제 모델 설계에 바로 연결된다.

---

## 과제별 해석

- Brain MRI
  - 다중 스케일 구조와 3D 문맥 때문에 다양한 attention이 활발히 쓰인다.
- Abdominal / multi-organ CT
  - 장기 간 위치 관계와 전역 문맥이 중요해 spatial + Transformer 계열이 유리하다.
- Retinal vessel / thin structure
  - fine structure 보존이 중요해 edge-aware, spatial attention이 특히 유용하다.
- Polyp / lesion / skin
  - foreground가 작고 배경이 복잡해 ROI localization과 boundary refinement가 핵심이다.

---

## 핵심 결론

- attention은 대부분의 의료영상 segmentation 과제에서 성능 향상에 기여한다.
- 하지만 향상 폭은 attention이란 이름 자체보다
  **task와 attention 유형의 정합성**에 더 크게 좌우된다.
- non-Transformer attention은 경량성과 local detail 복원에 강하다.
- Transformer attention은 전역 문맥과 장거리 관계 모델링에 강하다.
- 실제로는 둘을 섞은 hybrid 전략이 많이 채택된다.

---

## 강점

- attention을 "모듈"이 아니라 **설계 공간**으로 재정의한다.
- non-Transformer와 Transformer의 역할을 명확히 분리해 읽게 만든다.
- `what / how / where` 프레임이 매우 실용적이다.
- 약 300편 이상의 문헌을 구조적으로 정리해 연구 지형 파악에 유리하다.

---

## 한계

- 자체 benchmark가 없어 정량 비교 기준은 제한적이다.
- 문헌 간 backbone, 증강, 손실함수, 후처리가 달라 공정 비교가 어렵다.
- 2023년 survey라 이후 등장한 다음 흐름은 충분히 반영하지 못한다.
  - SAM 기반 promptable segmentation
  - medical foundation model adaptation
  - Mamba / SSM 계열 구조
  - universal segmentation

---

## 발표용 핵심 메시지

- 의료영상 segmentation에서 attention은 하나의 기법이 아니라 **문제 맞춤형 설계 선택지**다.
- 작은 병변과 경계 복원이 중요하면 channel / spatial / boundary attention이 유리할 수 있다.
- 전역 장기 배치와 장거리 관계가 중요하면 Transformer attention이 유리할 수 있다.
- 중요한 것은 최신 구조를 붙이는 것이 아니라
  **과제 특성에 맞는 attention을 고르는 것**이다.

---

## 결론

- 이 논문은 의료영상 분할 attention 연구를 가장 구조적으로 정리한 survey 중 하나다.
- 핵심 가치는 `non-Transformer` vs `Transformer` 구분과
  `what` / `how` / `where` 프레임에 있다.
- 최신 foundation model 흐름은 부족하지만,
  attention 기반 segmentation 설계 원리를 이해하는 기준 문서로는 여전히 유용하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/attention_mechanisms_in_medical_image_segmentation_a_survey_slide.md>
