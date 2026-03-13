---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model

- Yizhe Zhang, Tao Zhou, Shuo Wang, Peixian Liang, Danny Z. Chen
- arXiv 2023 / MICCAI Workshop LNCS chapter
- Using SAM as a prior generator rather than a direct segmenter

---

## 문제 배경

- SAM은 자연영상에서 강한 segmentation foundation model이다.
- 하지만 의료영상에서는 zero-shot 성능이 충분하지 않다.
- 그렇다고 SAM을 항상 직접 fine-tuning하는 것이 최선은 아니다.
- 논문의 핵심 질문:
  - SAM을 직접 출력기로 쓰지 않고도
    의료 분할 성능을 높일 수 있는가?

---

## 이 논문의 핵심 아이디어

- SAM을 segmentation answer generator로 보지 않는다.
- 대신 SAM이 생성하는 다음 정보를 재활용한다.
  - segmentation mask
  - stability score
- 이를 이용해 prior map을 만들고, 원본 입력에 붙여 downstream segmentation model의 입력으로 사용한다.
- 즉, adaptation을 **모델 적응**이 아니라 **입력 적응** 문제로 바꾼다.

---

## 논문의 핵심 기여

- `SAMAug`라는 입력 증강 방식을 제안한다.
- SAM의 출력을 segmentation prior와 boundary prior로 재구성한다.
- parameter-free fusion으로 구현한다.
- SAM은 고정하고 downstream model만 학습한다.
- CNN과 Transformer 기반 모델 모두에서 개선을 보고한다.

---

## 방법 개요

- 1단계: 입력 이미지에 SAM을 실행한다.
- 2단계: SAM 출력과 stability score로 prior map을 만든다.
- 3단계: 원본 영상과 prior map을 채널 단위로 결합한다.
- 4단계: 이 augmented input을 기존 segmentation model에 넣는다.
- 핵심: backbone을 거의 바꾸지 않는다.

---

## Prior Map은 무엇인가

- 논문은 두 종류의 prior를 만든다.
  - **segmentation prior map**
  - **boundary prior map**
- 이 구분이 중요한 이유:
  - 많은 의료영상 분할 문제는
    ROI 자체와 ROI 경계가 모두 중요하기 때문이다.
- 특히 polyp, gland, nucleus segmentation에서
  boundary cue는 성능에 직접 영향을 준다.

---

## Fusion 방식

- fusion은 deliberately simple하다.
- gray-scale 의료영상 기준 입력 채널 예:
  - 원본 image
  - segmentation prior
  - boundary prior
- 복잡한 adapter나 attention block을 추가하지 않는다.
- 발표 포인트: foundation model 활용이 반드시 heavy fine-tuning일 필요는 없다.

---

## 왜 이 방식이 흥미로운가

- SAM의 direct output은 의료영상에서 noisy할 수 있다.
- 하지만 noisy한 출력도 prior로 쓰면 유용할 수 있다.
- 즉, foundation model의 가치는
  완성된 답을 내는 능력보다
  **학습에 도움이 되는 구조적 힌트를 주는 능력**일 수 있다.

---

## 실험 설정

- 논문은 세 가지 2D biomedical segmentation 과제를 사용한다.
  - polyp segmentation
  - gland segmentation
  - nucleus segmentation
- 명시적으로 언급되는 데이터셋:
  - GlaS
  - MoNuSeg
- CNN과 Transformer 기반 downstream model 모두를 비교한다.

---

## 주요 결과 해석

- SAM은 direct segmenter보다 prior generator로 더 유용했다.
- SAMAug는 여러 backbone에 공통적으로 도움이 되었다.
- boundary prior를 별도로 주는 것이 의미 있었다.
- 특히 이 논문의 설득력은
  **방법이 매우 단순한데도 개선이 일관적**이라는 데 있다.

---

## 이 논문의 실질적 메시지

- foundation model은 출력기보다 정보원으로 볼 수 있다.
- 의료영상에서는 입력 증강이 강력한 적응 전략이 될 수 있다.
- simple baseline이 의외로 강할 수 있다.
- 즉, SAM 활용의 한 축은 fine-tuning이 아니라
  **prior injection**이다.

---

## 강점

- 구현 난도가 낮다.
- backbone 비의존적이다.
- parameter-free fusion이라 부담이 적다.
- 기존 segmentation pipeline에 쉽게 붙일 수 있다.
- foundation model 활용에 대한 좋은 사고실험이 된다.

---

## 한계

- 모든 이미지에 대해 SAM을 먼저 돌려야 하므로 전처리 비용이 든다.
- prior 품질이 SAM 출력 품질에 의존한다.
- 실험은 주로 2D biomedical segmentation에 집중되어 있다.
- 3D CT/MRI multi-organ segmentation으로의 일반화는 별도 검증이 필요하다.
- 단순 fusion이라 성능 상한이 제한될 수도 있다.

---

## 발표용 핵심 메시지

- 이 논문은 "SAM fine-tuning" 논문이 아니다.
- 핵심은 SAM을 **prior map generator**로 쓰는 것이다.
- foundation model은 end-to-end 모델로만 쓸 필요가 없다.
- 의료영상에서는 입력을 더 똑똑하게 만드는 것만으로도
  성능 향상이 가능할 수 있다.

---

## 결론

- 이 논문은 foundation model을 의료영상 segmentation에 가볍게 연결하는 브리지 방법이다.
- 가장 큰 장점은 단순성, backbone 비의존성, 낮은 구현 비용이다.
- 2D 과제 중심이라는 한계는 있지만,
  SAM을 prior source로 활용하는 관점은 이후 연구에도 의미 있는 출발점이다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/input_augmentation_with_sam_boosting_medical_image_segmentation_with_segmentation_foundation_model_slide.md>
