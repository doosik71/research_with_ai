---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# U-Net and its variants for Medical Image Segmentation : A short review

- Vinay Ummadi
- arXiv 2022
- U-Net 계열 대표 변형을 빠르게 훑는 짧은 입문 리뷰

---

## 문제 배경

- 의료영상 segmentation은 진단과 치료 planning의 핵심 단계다.
- 그러나 수동 분할은 시간과 전문성이 많이 든다.
- 의료영상은 modality가 다양하고 boundary가 복잡해 전통적 segmentation 기법으로 한계가 있었다.
- U-Net은 적은 데이터에서도 잘 작동하는 구조로 biomedical segmentation의 표준이 됐다.
- 이 리뷰는 그 이후 U-Net 변형들이 무엇을 보완하려 했는지를 짧게 정리한다.

---

## 이 리뷰의 성격

- 폭넓고 정교한 대형 survey는 아니다.
- 대신 다음 질문에 빠르게 답하는 입문 문서에 가깝다.
- U-Net이 왜 강했는가?
- U-Net++는 무엇을 바꿨는가?
- R2U-Net은 왜 residual/recurrent를 넣었는가?
- Attention U-Net은 무엇에 집중하게 만들었는가?
- TransUNet은 왜 Transformer를 결합했는가?

---

## 먼저 전통적 방법의 한계

- thresholding, clustering, mean shift, graph cut 같은 전통 방법은
  라벨 데이터 없이도 쓸 수 있다는 장점이 있다.
- 하지만 의료영상의 intensity variation, 작은 ROI, 복잡한 anatomical boundary를 다루기 어렵다.
- 이 한계가 deep learning 기반 분할, 특히 U-Net의 배경이 된다.

---

## U-Net 2015

- U-Net의 핵심은 `encoder-decoder + skip connection`이다.
- encoder는 downsampling으로 의미 정보를 압축한다.
- decoder는 upsampling으로 segmentation map을 복원한다.
- skip connection은 위치 정보와 fine detail을 보존한다.
- 이 구조는 의료영상처럼 데이터가 적고 정확한 localization이 중요한 환경과 잘 맞았다.

---

## 왜 U-Net이 표준이 되었는가

- 적은 데이터에서도 학습이 잘 된다.
- end-to-end segmentation이 가능하다.
- 위치 정보 복원에 강하다.
- 다양한 biomedical segmentation task에 쉽게 적용된다.
- 발표에서는 `U-Net은 구조적으로 의료영상 문제에 맞춘 답`이라고 정리하면 된다.

---

## U-Net++ 2019

- U-Net++는 `nested skip connection`과 `deep supervision`을 도입한다.
- 목표는 encoder와 decoder 사이의 semantic gap을 줄이는 것이다.
- skip pathway에 추가 convolution을 넣고 dense connection을 구성한다.
- 결과적으로 gradient flow와 feature fusion을 더 부드럽게 만든다.
- 즉, U-Net++는 skip connection 자체를 더 정교하게 만든 변형이다.

---

## R2U-Net 2018

- R2U-Net은 `residual connection + recurrent connection`을 결합한다.
- recurrent-residual block으로 local context를 더 반복적으로 정제한다.
- residual path는 gradient propagation을 돕는다.
- 메시지는 단순하다.
- U-Net의 큰 뼈대는 유지하되,
  더 깊고 안정적으로 feature를 다루고 싶었던 변형이다.

---

## Attention U-Net 2018

- Attention U-Net은 skip pathway에 `attention gate`를 넣는다.
- decoder로 모든 feature를 그대로 보내지 않고 중요한 ROI 중심으로 선택한다.
- 특히 organ size variation이나 background clutter가 큰 문제에서 유용하다.
- 즉, localization을 더 선택적으로 만들고
  덜 중요한 feature를 억제하는 방향의 개선이다.

---

## TransUNet 2021

- TransUNet은 CNN 기반 U-Net과 Transformer를 결합한 hybrid 구조다.
- CNN은 local representation을 추출한다.
- Transformer는 patch-wise global dependency를 학습한다.
- decoder와 skip connection은 다시 precise localization을 복원한다.
- 이 논문에서 TransUNet은
  U-Net 계열이 global context 부족을 보완하려는 대표적 확장으로 소개된다.

---

## 변형들을 한 줄로 요약하면

- `U-Net`: localization이 강한 기본 골격
- `U-Net++`: skip connection 정교화
- `R2U-Net`: residual/recurrent로 feature refinement 강화
- `Attention U-Net`: ROI 선택성 강화
- `TransUNet`: global context modeling 추가
- 즉, 변형들의 역사는 U-Net을 버리는 과정이 아니라
  U-Net의 약점을 한 가지씩 보완하는 과정으로 볼 수 있다.

---

## 이 리뷰의 핵심 메시지

- 2016년 이후 많은 변형은 구조적으로 점진적 개선에 가깝다.
- 큰 구조 변화는 Transformer 결합에서 나타난다.
- 리뷰 관점에서 가장 주목할 전환점은 `TransUNet`이다.
- 이유는 U-Net의 local bias 한계를 직접 겨냥했기 때문이다.

---

## 강점

- U-Net 계열 진화 흐름을 빠르게 이해하기 좋다.
- 구조별 목적이 명확하게 정리돼 있다.
- residual, recurrent, attention, transformer가
  U-Net 안에 어떻게 흡수됐는지 한눈에 볼 수 있다.
- 입문용 발표 자료로 쓰기 좋다.

---

## 한계

- `short review`라 coverage가 좁다.
- `nnU-Net`, `UNet 3+`, `UNETR`, `Swin UNETR`, `MedT` 등 더 넓은 계열은 충분히 다루지 않는다.
- 정량 비교는 서로 다른 데이터셋과 조건에서 가져온 값이라 직접 leaderboard 비교에 적합하지 않다.
- 2022년 이후의 foundation model, promptable segmentation 흐름은 반영하지 못한다.

---

## 현재 시점에서의 해석

- 지금 보면 이 리뷰는 최신 survey라기보다
  `U-Net 계열 구조 진화의 짧은 안내서`에 가깝다.
- 특히 의료 segmentation을 처음 공부할 때
  구조 아이디어를 순서대로 이해하는 데는 유용하다.
- 반면 최신 비교나 전체 지형도를 보려면 더 큰 survey가 필요하다.

---

## 발표용 핵심 메시지

- U-Net은 의료 segmentation의 기본 골격을 만들었다.
- 이후 변형들은 skip, recurrence, attention, transformer를 붙이며 약점을 보완했다.
- 대부분은 점진적 개선이고,
  가장 큰 방향 전환은 global context를 넣은 Transformer 결합이다.
- 이 리뷰는 broad survey보다 `구조 진화 요약본`으로 읽는 것이 정확하다.

---

## 결론

- `U-Net and its variants for Medical Image Segmentation : A short review`는
  U-Net 계열 대표 변형을 짧고 직관적으로 정리한 입문 문서다.
- 핵심 가치는 최신 SOTA 정리가 아니라
  각 변형이 `무엇을 고치려 했는지`를 명확하게 보여준다는 데 있다.
- 의료영상 segmentation 구조 진화를 빠르게 설명할 때 유용한 요약 자료다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/u_net_and_its_variants_for_medical_image_segmentation_a_short_review_slide.md>
