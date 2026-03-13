---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# ProMISe: Promptable Medical Image Segmentation using SAM

- Jinfeng Wang et al.
- arXiv 2024
- SAM을 의료 semantic segmentation에 맞게 재설계한 prompt learning 접근

---

## 문제 배경

- SAM은 자연영상에서 강한 zero-shot interactive segmentation 성능을 보였다.
- 하지만 의료영상에서는 point/box prompt 기반 사용이 그대로 맞지 않는다.
- 실제 의료 분할은 interactive prompt보다 자동 semantic segmentation이 더 중요하다.
- 3D volumetric context와 multi-class 구조도 함께 다뤄야 한다.
- 따라서 질문은 분명하다.
- `SAM의 promptable 구조를 유지하면서 의료 semantic segmentation에 맞는 prompt를 학습할 수 있는가?`

---

## 핵심 아이디어

- ProMISe는 point/box prompt를 직접 넣는 대신 `learnable visual prompt`를 도입한다.
- 즉, 사람이 주는 prompt를 없애고 모델이 의료 분할에 맞는 prompt representation을 학습한다.
- 이를 통해 `promptable segmentation`을 `automatic medical segmentation`으로 재해석한다.
- 논문의 본질은 SAM encoder 재사용보다 `prompt definition의 재설계`에 있다.

---

## 핵심 기여

- `learnable visual prompts`를 통해 SAM을 의료 semantic segmentation에 맞게 적응시켰다.
- 2D와 3D segmentation을 모두 다루는 parameter-efficient adaptation 구조를 제안했다.
- SAM image encoder를 대부분 고정한 상태에서 적은 추가 파라미터만 학습한다.
- 4개 데이터셋, 5개 과제에서 강한 성능을 보고한다.
- multi-organ, multi-class, volumetric segmentation까지 promptable formulation을 확장했다.

---

## 왜 point/box prompt가 부족한가

- 의료영상 semantic segmentation은 test-time interactive prompt를 기대하기 어렵다.
- 매 환자, 매 slice마다 점이나 박스를 수동 입력하는 방식은 비효율적이다.
- 병변과 장기 구조는 class-aware prior가 중요하다.
- 따라서 의료영상에서는 prompt를 사람이 주기보다
  `모델이 구조적 prior로 학습`하는 편이 더 자연스럽다.

---

## ProMISe 구조

- 기본 backbone은 SAM이다.
- 그러나 SAM 전체를 full fine-tuning하지 않는다.
- image encoder는 largely frozen 상태로 유지한다.
- 대신 학습 가능한 visual prompt와 소규모 적응 모듈을 추가한다.
- 결과적으로 foundation model의 일반 표현력을 최대한 유지하면서
  의료 도메인 적응 비용을 줄인다.

---

## Learnable Visual Prompt의 의미

- prompt를 외부 입력이 아니라 `학습 가능한 내부 표현`으로 바꾼다.
- 이 prompt는 의료 segmentation에 필요한 구조적 prior를 담는 역할을 한다.
- point/box prompt dependence를 줄이면서도
  SAM의 promptable design 철학은 유지한다.
- 발표에서는 이를 `prompt engineering에서 prompt learning으로의 이동`으로 설명하면 된다.

---

## Parameter-Efficient Adaptation

- ProMISe의 중요한 메시지는 최고 성능 자체보다 `효율적 적응`이다.
- SAM 전체를 다시 학습하지 않아도 된다.
- 추가 파라미터가 적어 메모리와 학습 비용이 낮다.
- 데이터가 적은 의료영상 환경과 잘 맞는다.
- 여러 기관과 여러 과제로 확장할 때도 실용성이 높다.

---

## 2D와 3D를 함께 다루는 점

- 많은 SAM adaptation 연구는 2D medical image에 머무른다.
- ProMISe는 3D volumetric segmentation까지 평가 범위를 넓힌다.
- 이는 SAM adaptation이 toy setting이 아니라
  실제 multi-organ CT segmentation에도 적용 가능하다는 주장으로 이어진다.
- 논문의 실용성을 높이는 부분이다.

---

## 평가 설정

- 논문은 4개 데이터셋, 5개 과제에서 ProMISe를 평가한다.
- 평가 범주는 다음과 같다.
- 2D single-organ segmentation
- 2D multi-class segmentation
- multi-organ abdominal CT segmentation
- volumetric 3D segmentation
- 비교 대상은 CNN 계열과 Transformer 계열 segmentation baseline들이다.

---

## 주요 결과

- 2D 의료영상 분할에서 기존 SAM adaptation과 일반 segmentation baseline 대비 강한 성능을 보인다.
- 3D multi-organ segmentation에서도 경쟁력 있는 결과를 보고한다.
- 논문이 강조하는 대표 결과 중 하나는
  BTCV 같은 multi-organ CT에서의 높은 Dice 성능이다.
- 핵심은 `SAM 기반 적응이 실제 의료 분할 benchmark에서도 통한다`는 점이다.

---

## 논문이 말하는 핵심 해석

- SAM의 강점은 image encoder만이 아니다.
- `promptable design 자체를 의료 과제에 맞게 바꾸면` 성능 잠재력이 커진다.
- 즉, 의료영상에서는 prompt mismatch가 중요한 병목이다.
- ProMISe는 이 mismatch를 줄여
  interactive segmentation과 automatic segmentation의 경계를 흐린다.

---

## 강점

- SAM을 단순 backbone으로 쓰지 않고 promptability까지 재해석했다.
- parameter-efficient adaptation이라 실용성이 높다.
- 2D와 3D를 모두 포괄해 평가 범위가 넓다.
- multi-organ segmentation까지 연결해 toy setting에 머물지 않는다.
- foundation model adaptation의 한 가지 설득력 있는 방향을 제시한다.

---

## 한계

- 성능의 상당 부분은 SAM 사전학습 표현력에 기대고 있다.
- 2D SAM 기반이므로 3D-native volumetric foundation model은 아니다.
- learned prompt가 unseen anatomy나 새로운 기관에서 얼마나 일반화되는지는 더 검증이 필요하다.
- visual prompt가 실제로 어떤 구조 prior를 담는지 해석 가능성은 제한적이다.
- 보편적 표준이라기보다 유망한 방향 제시에 가깝다.

---

## 현재 시점에서의 의미

- 이 논문은 `SAM을 의료 segmentation backbone으로 어떻게 바꿀 것인가`에 대한 좋은 출발점이다.
- 특히 의료영상에서 prompt를 다시 설계해야 한다는 점을 분명히 보여준다.
- 이후 흐름은 자연스럽다.
- `MedSAM`, `SAM2`, `3D-native medical foundation model`, `multimodal promptable segmentation`으로 확장될 수 있다.
- 즉, ProMISe는 promptable medical segmentation의 초기 설계 원칙을 제시한 논문으로 볼 수 있다.

---

## 발표용 핵심 메시지

- ProMISe의 핵심은 SAM fine-tuning이 아니라 `prompt learning`이다.
- 의료영상에서는 사람이 주는 prompt보다 학습 가능한 구조적 prompt가 더 현실적이다.
- foundation model adaptation은 full fine-tuning보다
  parameter-efficient하게 설계하는 편이 의료 환경과 잘 맞는다.
- 이 논문은 SAM을 interactive tool에서 medical segmentation model로 옮기려는 시도다.

---

## 결론

- `ProMISe: Promptable Medical Image Segmentation using SAM`은
  learnable visual prompt를 통해 SAM을 의료 semantic segmentation에 맞게 재설계한 연구다.
- point/box prompt 의존성을 줄이고 자동 분할로 연결한 점이 핵심이다.
- 2D와 3D 모두에서 가능성을 보였지만,
  진정한 3D-native foundation model로 가기 전 단계로 보는 것이 타당하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/promise_promptable_medical_image_segmentation_using_sam_slide.md>
