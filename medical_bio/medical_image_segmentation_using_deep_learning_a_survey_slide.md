---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Medical Image Segmentation Using Deep Learning: A Survey

- Risheng Wang, Yu Meng, Zongyuan Ge, Jingke Ma, Jianfeng Zheng, Mingyang Ren, Yefeng Zheng
- IET Image Processing 2021
- Survey of supervised, weakly supervised, and emerging segmentation methods

---

## 문제 배경

- 의료영상 분할은 장기, 조직, 병변 경계를 픽셀/복셀 단위로 예측하는 문제다.
- 임상적으로는 다음의 기반이 된다.
  - 진단 보조
  - 수술 계획
  - 방사선 치료 계획
  - 추적 관찰 정량화
- 하지만 실제 환경은 어렵다.
  - modality 다양성
  - 작은 병변
  - 모호한 경계
  - 3D 계산 비용
  - 높은 annotation 비용

---

## 이 논문의 핵심 기여

- 의료영상 분할 딥러닝을 네 축으로 구조화한다.
  - supervised learning
  - weakly supervised learning
  - advanced methods
  - datasets / challenges / future directions
- 특히 supervised segmentation을
  **network block / loss / training strategy / post-processing**으로 분해한 점이 실용적이다.

---

## Supervised Segmentation의 기본 틀

- encoder-decoder 구조가 중심이다.
- 성능 차이를 만드는 설계 요소는 다음 네 부류다.
  - network blocks
  - loss functions
  - training strategies
  - post-processing
- 발표 포인트:
  - 좋은 segmentation 모델은 backbone 이름 하나로 설명되지 않는다.

---

## Network Blocks

- 대표 구성:
  - FCN / U-Net / V-Net 계열 backbone
  - skip connection
  - residual / dense block
  - attention block
  - multi-scale context block
  - 2D / 3D convolution
- 이 논문의 해석:
  - 경계 복원, 다중 스케일 문맥, 3D context 활용의 조합이 핵심이다.

---

## Loss Function

- 의료영상 분할에서 loss는 부속 요소가 아니다.
- 대표 loss:
  - cross-entropy
  - Dice loss
  - weighted loss
  - hybrid loss
- 왜 중요한가:
  - foreground가 매우 작을 수 있다.
  - class imbalance가 심하다.
  - boundary sensitivity가 중요하다.

---

## Training Strategy

- 대표 전략:
  - patch-based training
  - hard sample mining
  - cascaded / coarse-to-fine training
  - multi-stage training
  - deep supervision
- 의미:
  - 의료영상 분할은 단순 optimizer 선택보다
    **데이터 제약을 어떻게 우회할지**가 중요하다.

---

## Post-processing

- 분할 결과를 더 해부학적으로 타당하게 보정한다.
- 대표 방식:
  - CRF refinement
  - connected component filtering
  - morphology operation
  - shape prior / anatomical constraint
- 핵심 메시지:
  - 의료영상 분할 성능은 네트워크만으로 끝나지 않는다.

---

## Weakly Supervised Learning

- 이 논문은 weak supervision을 본론에서 크게 다룬다.
- 대표 축:
  - data augmentation
  - transfer learning
  - semi-supervised learning
- 왜 중요한가:
  - 의료영상에서는 완전한 mask annotation이 매우 비싸기 때문이다.

---

## Weak Supervision의 실질적 의미

- augmentation은 일반화 개선을 넘어 사실상 필수 요소다.
- transfer learning은 도움이 되지만,
  segmentation에서는 자연영상 전이 효과가 제한적일 수 있다.
- semi-supervised learning은 옵션이 아니라
  **실용적 표준 후보**로 제시된다.
- 즉, 적은 fully labeled sample + 많은 unlabeled sample 조합이 현실적 설정이다.

---

## Advanced Methods

- 논문 당시의 신흥 흐름을 네 가지로 묶는다.
  - neural architecture search
  - graph convolutional networks
  - multi-modality fusion
  - medical transformer
- 지금 보면 이 섹션은
  Transformer 시대 직전의 "예고편"에 가깝다.

---

## 이 논문이 주는 정성적 결론

- encoder-decoder는 여전히 기본 골격이다.
- Dice 기반 loss와 skip connection은 사실상 표준 요소다.
- 3D 문맥 활용은 중요하지만 계산 비용이 높다.
- weak supervision과 transfer learning은 데이터 부족에 실질적으로 기여한다.
- multi-modality fusion과 transformer는 다음 세대 확장 방향이다.

---

## 강점

- supervised segmentation을 구현 관점으로 분해해 설명한다.
- weak supervision을 부록이 아니라 핵심 장으로 다룬다.
- 2020년 전후 segmentation 연구 지형을 구조적으로 정리한다.

---

## 한계와 현재 관점

- foundation model 이전 시대의 survey다.
- 포함되지 않는 흐름:
  - SAM / promptable segmentation
  - large-scale medical pretraining
  - diffusion-based segmentation
  - universal segmentation
  - domain generalization benchmark
- 따라서 최신 지도라기보다
  **기반 survey**로 읽는 편이 정확하다.

---

## 발표용 핵심 메시지

- 의료영상 분할의 핵심은 좋은 backbone 하나를 고르는 것이 아니다.
- network block, loss, training strategy, post-processing이 함께 설계돼야 한다.
- 데이터 부족 때문에 weak supervision은 예외가 아니라 기본 문제다.
- 이 논문은 U-Net 중심 시대에서 Transformer 이전 시대로 넘어가는 중간 지점을 잘 정리한다.

---

## 결론

- `Medical Image Segmentation Using Deep Learning: A Survey`는
  의료영상 분할 딥러닝을 구조적으로 이해하기 좋은 기준 문헌이다.
- supervised, weakly supervised, advanced methods를 한 문서 안에서 연결한 점이 강점이다.
- 최신 foundation model 흐름은 빠져 있지만,
  데이터 부족, 클래스 불균형, 3D 비용, 도메인 이동 같은 핵심 문제는 지금도 유효하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/medical_image_segmentation_using_deep_learning_a_survey_slide.md>
