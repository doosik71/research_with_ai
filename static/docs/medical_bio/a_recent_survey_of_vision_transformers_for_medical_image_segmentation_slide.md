---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# A Recent Survey of Vision Transformers for Medical Image Segmentation

- Asifullah Khan et al.
- arXiv 2023 / IEEE Access 2025
- Survey of pure ViT and hybrid vision transformer methods for medical segmentation

---

## 문제 배경

- 의료영상 분할은 경계 복원과 전역 문맥 이해를 동시에 요구한다.
- CNN은 local pattern에 강하지만 long-range dependency 모델링은 제한적이다.
- ViT는 self-attention으로 global context를 잘 본다.
- 하지만 의료영상에서는 다음 제약이 크다.
  - 라벨 부족, 클래스 불균형, 노이즈와 artifact, 모호한 경계, 3D 계산 비용

---

## 논문의 핵심 질문

- ViT는 의료영상 분할에서 어디에 배치되는가?
- pure ViT와 hybrid vision transformer 중 무엇이 더 실용적인가?
- CT, MRI, ultrasound 등 modality마다 유리한 설계가 다른가?
- 정확도뿐 아니라 파라미터와 추론 비용까지 고려하면 어떤 결론이 나오는가?

---

## 가장 중요한 분류

- 논문은 의료영상 분할 모델을 크게 두 부류로 나눈다.
  - **Pure ViT-based methods**
  - **Hybrid Vision Transformer (HVT)-based methods**
- 발표 포인트:
  - 이 survey의 실제 결론은
    "Transformer가 무조건 낫다"가 아니라
    **의료영상에서는 hybrid가 더 실용적이다**에 가깝다.

---

## Pure ViT 분류

- pure ViT는 Transformer를 어디에 두느냐로 다시 나뉜다.
  - encoder에 배치
  - decoder에 배치
  - encoder-decoder interface에 배치
  - encoder와 decoder 양쪽에 배치
- 직관:
  - encoder는 global representation 학습
  - decoder는 segmentation mask 복원
  - interface는 병목 구간의 의미 연결

---

## HVT 분류

- HVT는 CNN의 local bias와 ViT의 global reasoning을 결합한다.
- 세 가지 통합 방식이 핵심이다.
  - encoder-based integration
  - decoder-based integration
  - encoder-decoder interface integration
- 논문은 특히 **encoder-based HVT**를
  여러 modality에서 가장 실용적인 형태로 해석한다.

---

## 왜 Hybrid가 강한가

- CNN은 local detail과 translation invariance를 제공한다.
- Transformer는 장거리 상호작용과 전역 문맥을 제공한다.
- 의료영상 분할은 두 능력이 모두 필요하다.
- 따라서 HVT는 다음 절충점을 만든다.
  - 성능 유지
  - 파라미터 절감
  - 추론 비용 통제
  - 경계 복원 보강

---

## CT에서의 결론

- CT에서는 encoder-based HVT가 성능과 효율성 균형이 좋다고 본다.
- 논문이 인용하는 대표 경향:
  - `TransUNet`은 BTCV에서 강한 Dice와 실용적 속도를 보임
  - pure ViT encoder 계열은 더 무겁고 느릴 수 있음
  - 특정 장기 특화 모델은 높게 나오지만 범용성은 제한될 수 있음
- 발표 포인트:
  - multi-organ CT에서는 **hybrid encoder 설계가 현실적**이다.

---

## MRI와 Ultrasound에서의 결론

- MRI에서도 parameter-efficiency 측면에서 HVT 우세 해석이 반복된다.
- 유사 성능을 더 적은 파라미터로 내는 구조가 주목된다.
- ultrasound는 실시간성과 잡음 문제가 커서 더욱 신중해야 한다.
- 저자들의 요지는 일관된다.
  - 모든 층에 무거운 ViT를 넣는다고 항상 이득은 아니다.
  - **필요한 위치에만 Transformer를 넣는 설계가 더 낫다.**

---

## Modality별 핵심 해석

- **CT**: multi-organ complexity와 효율성을 함께 봐야 한다.
- **MRI**: 3D 문맥과 계산 비용 때문에 효율적 hybrid가 유리하다.
- **Ultrasound**: 잡음과 실시간성 때문에 lightweight integration이 중요하다.
- **X-ray**: 최대 성능과 배포 복잡도 사이 tradeoff가 크다.
- **Histopathology / Microscopy**: patch-based context modeling 장점이 크지만 강건성 이슈가 남는다.

---

## 성능은 Backbone만으로 결정되지 않는다

- 논문은 구조 taxonomy 외에도 보조 전략을 강조한다.
  - pre-processing / post-processing
  - boundary-aware module
  - attention refinement
  - skip connection 보강
  - federated learning 결합
  - synthetic data generation
- 즉, segmentation 성능은 ViT냐 CNN이냐만의 문제가 아니다.

---

## 강점

- pure ViT와 HVT를 체계적으로 구분해 설계 공간을 정리한다.
- modality별 사례를 함께 보여 줘 실무 감각이 있다.
- 정확도뿐 아니라 파라미터 수와 추론 시간까지 보려 한다.
- 가장 구체적인 결론인 encoder-based HVT의 실용성을 선명하게 제시한다.

---

## 한계

- 모델별 실험 설정과 데이터셋이 달라 완전한 공정 비교는 어렵다.
- 표는 leaderboard보다 경향 해석으로 읽어야 한다.
- 2025년 survey지만 SAM 이후 의료 adaptation, universal segmentation, multimodal foundation model 흐름은 깊지 않다.
- 따라서 foundation model 시대의 최신 정리는 추가 문헌이 필요하다.

---

## 발표용 핵심 메시지

- 이 논문의 실질적 결론은 **pure ViT보다 HVT가 더 실용적**이라는 점이다.
- 특히 encoder-based HVT가 CT, MRI, ultrasound에서 반복적으로 좋은 균형을 보인다.
- 의료영상 분할에서는 global context만큼 local detail 보존이 중요하다.
- 앞으로의 확장은 단순 ViT 경쟁보다
  **foundation model adaptation, robustness, efficiency, deployment** 쪽이다.

---

## 결론

- 이 논문은 의료영상 분할의 ViT 설계 공간을 구조와 modality 양쪽에서 정리한 survey다.
- 가장 설득력 있는 메시지는 hybrid 설계, 특히 encoder-based HVT의 강세다.
- 다만 비교 엄밀성과 최신 foundation model 흐름 반영은 제한적이다.
- 그래서 이 문헌은 **ViT 의료영상 분할의 정리본이자 pre-foundation-model 단계의 종합 보고서**로 읽는 것이 적절하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/a_recent_survey_of_vision_transformers_for_medical_image_segmentation_slide.md>
