---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# From CNN to Transformer: A Review of Medical Image Segmentation Models

- Wenjian Yao, Zhaohui Jin, Yong Xia, Yanning Zhang
- arXiv 2023
- Review of CNN, Transformer, and hybrid segmentation models in medical imaging

---

## 문제 배경

- 의료영상 분할은 두 요구를 동시에 가진다.
  - 경계를 정밀하게 복원해야 한다.
  - 전역 문맥과 장거리 관계도 이해해야 한다.
- CNN과 Transformer는 이 문제를 서로 다른 방식으로 푼다.
- 논문의 핵심 질문:
  - 의료영상 분할에서 CNN, Transformer, hybrid는 각각 어디서 강한가?

---

## 논문의 핵심 기여

- 의료영상 분할 모델을 세 흐름으로 정리한다.
  - **CNN-based**
  - **Transformer-based**
  - **CNN-Transformer hybrid**
- 성능만이 아니라 다음도 함께 본다.
  - 계산 비용, 데이터 요구량, 3D 적용성, 일반화와 강건성
- 즉, backbone 유행사가 아니라
  **설계 trade-off 지도**를 제공한다.

---

## CNN 기반 분할

- 의료영상 분할의 출발점은 CNN 계열이다.
- 핵심 설계 요소:
  - encoder-decoder
  - skip connection
  - residual / dense connection
  - multi-scale fusion
  - dilated convolution
  - attention-enhanced CNN block
- 대표 기준점은 여전히 U-Net 계열이다.

---

## CNN의 강점과 한계

- 강점:
  - local texture와 경계 복원에 강하다
  - inductive bias가 강해 적은 데이터에서도 안정적이다
  - 2D/3D 확장 경험이 풍부하다
- 한계:
  - 장거리 문맥을 직접 모델링하기 어렵다
  - 다중 장기나 복잡한 해부학 배치에서 global reasoning이 약하다
- 메시지:
  - CNN은 기본기가 강하지만 전역 문맥에서 한계가 있다.

---

## Transformer 기반 분할

- Transformer는 self-attention으로 전역 관계를 직접 본다.
- 장점:
  - long-range dependency modeling
  - global contextual representation
  - complex shape variation 대응
- 의료영상에서의 구조 예:
  - ViT-style patch segmentation
  - U-shaped Transformer
  - window-based Transformer
  - 3D volumetric Transformer

---

## Transformer의 한계

- 데이터 요구량이 크다.
- 계산량과 메모리 비용이 크다.
- fine boundary 복원이 약할 수 있다.
- 3D 데이터에 적용하면 비용이 급격히 증가한다.
- 따라서 이 논문은
  "Transformer가 바로 표준 해법"이라고 보지 않는다.

---

## Hybrid가 왜 중요한가

- CNN은 local detail과 boundary preservation에 강하다.
- Transformer는 global context와 long-range dependency에 강하다.
- 의료영상 분할은 이 두 성질을 모두 요구한다.
- 그래서 실제로는 다음 혼합 전략이 많다.
  - CNN encoder + Transformer bottleneck
  - Transformer encoder + CNN decoder
  - stage-wise mixed block

---

## 이 논문의 실질적 결론

- 핵심 문제는 "CNN vs Transformer"가 아니다.
- 진짜 핵심은
  **local-global tradeoff를 어떻게 설계하느냐**다.
- 이 관점에서 hybrid는 단순 절충안이 아니라
  의료영상 분할의 가장 현실적인 해법으로 제시된다.

---

## 과제별 해석

- Brain MRI
  - 3D 문맥과 구조 관계가 중요해 Transformer 도입이 활발하다.
- Abdominal multi-organ
  - 장기 간 위치 관계 때문에 global context가 중요하다.
- Cardiac segmentation
  - 경계 정밀도와 전역 문맥이 모두 중요해 hybrid가 설득력 있다.
- Retinal / skin / polyp / lesion
  - 작은 구조와 미세 경계가 중요해 CNN 기반 강점도 여전히 크다.

---

## 결과를 읽는 방식

- Transformer가 CNN을 능가하는 사례는 많다.
- 하지만 해석은 단순하지 않다.
  - dataset size
  - 2D vs 3D
  - pretraining 여부
  - memory budget
  - efficiency
- 논문의 메시지:
  - 숫자만 보지 말고 data regime과 computational regime을 함께 봐야 한다.

---

## 강점

- CNN, Transformer, hybrid를 경쟁 모델이 아니라
  서로 다른 inductive bias의 조합으로 설명한다.
- task별 요구와 backbone 선택을 연결해 준다.
- 2023년 전환기 의료영상 분할 지형을 잘 정리한다.

---

## 한계와 이후 흐름

- 2023년 이후 부상한 흐름은 반영되지 않는다.
  - SAM adaptation
  - promptable / universal segmentation
  - medical foundation model pretraining
  - Mamba / diffusion 계열 구조
- 따라서 현재 연구 설계에는 최신 survey가 추가로 필요하다.

---

## 발표용 핵심 메시지

- 의료영상 분할의 발전은 backbone 교체의 역사처럼 보이지만,
  실제로는 **local-global tradeoff 설계의 역사**다.
- Transformer는 CNN을 대체하기보다 CNN의 global-context 약점을 보완하기 위해 도입됐다.
- pure Transformer가 종착점이라기보다
  hybrid와 efficient architecture가 더 현실적이다.
- 다음 단계는 architecture novelty보다
  pretraining, robustness, efficiency, foundation adaptation 쪽으로 이동한다.

---

## 결론

- 이 논문은 의료영상 분할의 구조적 진화를 이해하기 좋은 survey다.
- 가장 큰 메시지는 CNN, Transformer, hybrid를
  서로 다른 inductive bias와 적용 조건의 관점에서 읽어야 한다는 점이다.
- foundation model 이전 전환기를 정리한 기준 문헌으로 여전히 유용하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/from_cnn_to_transformer_a_review_of_medical_image_segmentation_models_slide.md>
