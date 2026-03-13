---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Vision Foundation Models in Medical Image Analysis: Advances and Challenges

- Pengchen Liang et al.
- arXiv 2025
- 의료영상에서 VFM을 어떻게 적응, 경량화, 배포할 것인가를 다룬 짧은 survey

---

## 문제 배경

- 최근 의료영상 분석에서 Vision Foundation Models(VFMs), 특히 ViT와 SAM 계열이 빠르게 확산되고 있다.
- 그러나 자연영상 기반 대규모 사전학습 모델을 의료영상에 그대로 적용하기는 어렵다.
- 의료영상은 modality 차이, 3D 구조, 데이터 부족, privacy, edge deployment 제약이 크다.
- 따라서 핵심 질문은 `foundation model을 그대로 쓸 수 있는가`보다
  `어떻게 의료 도메인에 맞게 적응시킬 것인가`다.

---

## 이 survey의 핵심 성격

- 완성된 대형 VFM 지형도라기보다
  `SAM 이후 의료영상 적응 연구를 빠르게 구조화한 обзор`에 가깝다.
- 특히 다음 흐름을 하나로 묶어 설명한다.
- domain adaptation
- model compression
- knowledge distillation
- federated learning
- 즉, 이 문서의 강점은 foundation model 자체보다 `실용적 adaptation`에 있다.

---

## 핵심 기여

- ViT 계열이 CNN의 한계를 어떻게 보완하는지 개괄한다.
- SAM 기반 의료영상 adaptation을 adapter, PEFT, 3D 확장, few-shot 관점에서 정리한다.
- compression과 distillation을 edge deployment 문제와 연결한다.
- federated learning과 foundation model 결합 가능성을 별도 축으로 논의한다.
- 의료영상 VFM 연구가 `성능`에서 `적응-배포-협업` 문제로 이동하고 있음을 보여준다.

---

## 1. ViT의 의료영상 진입

- ViT는 CNN보다 global spatial relationship modeling에 강하다.
- 의료영상에서도 organ-organ relation, lesion-context relation 같은 장점이 기대된다.
- 대표 예시는 `TransUNet`, `Swin-UNet` 같은 hybrid 구조다.
- 그러나 ViT는 데이터 요구량이 크고
  domain-specific pretraining이 부족하면 이점을 살리기 어렵다.
- 즉, ViT의 도입은 출발점이지 해결책 자체는 아니다.

---

## 2. Paradigm Shift: Task-specific Model -> Large Model Adaptation

- 이 survey는 의료영상 분석이 task-specific architecture 중심에서
  large model adaptation 중심으로 이동하고 있다고 본다.
- 그 전환의 상징이 `SAM`이다.
- SAM은 자연영상에서 강한 zero-shot segmentation capability를 보였지만
  의료영상에서는 prompt mismatch, 3D 구조, fine-grained boundary 문제를 드러냈다.
- 그래서 진짜 연구 주제는 `SAM 자체`보다 `SAM adaptation`이 된다.

---

## 3. SAM Adaptation Taxonomy

- 이 논문이 가장 집중하는 부분이다.
- 주요 방향은 다음과 같다.
- adapter-based improvement
- 3D medical adaptation
- parameter-efficient fine-tuning
- low-rank adaptation
- few-shot adaptation
- poor prompt 완화
- 발표에서는 `foundation model 자체보다 adaptation design이 경쟁력의 핵심`이라고 요약하면 된다.

---

## 대표 사례

- `Med-SA`: adapter와 hyper-prompting으로 의료 도메인 적응
- `3DMedSAM`: 2D SAM을 3D medical volume으로 확장
- `Trans-SAM`: PEFT 기반 adapter 설계
- `LoRASAM`: low-rank adaptation으로 학습 파라미터 대폭 절감
- `DeSAM`: poor prompt 영향을 줄이기 위한 decoupling
- 이 사례들은 공통적으로 zero-shot 순정 SAM보다
  domain-aware adaptation이 더 중요하다는 점을 보여준다.

---

## 4. Compression과 Distillation

- 의료 현장에서는 foundation model의 정확도만으로 충분하지 않다.
- 메모리, latency, edge deployment 가능성이 중요하다.
- 따라서 compression과 distillation이 별도 연구 축이 된다.
- teacher-student distillation, self-distillation, cross-distillation,
  lightweight ViT distillation이 주요 방향이다.
- `MobileSAM`, `TinySAM`, `EfficientViT-SAM` 같은 경량 모델도 이 흐름에 속한다.

---

## 왜 경량화가 중요한가

- 실제 임상 환경은 항상 대형 GPU 서버 위에서 돌아가지 않는다.
- portable scanner, point-of-care device, 병원 내 제한된 인프라를 고려해야 한다.
- 그래서 VFM 연구의 가치는 최고 성능뿐 아니라
  `배포 가능성`까지 포함해야 한다.
- 이 survey는 그 실용 조건을 비교적 분명히 반영한다.

---

## 5. Federated Learning과 VFM

- 의료 데이터는 기관 간 raw data 공유가 어렵다.
- 따라서 federated learning은 의료 VFM 확장의 중요한 축이다.
- 하지만 foundation model은 파라미터가 크기 때문에 communication cost가 높다.
- 그래서 `PEFT`, `adapter sharing`, `prompt-based personalization`, `sparse update`가 중요해진다.
- 이 survey는 FL을 별도 확장 방향으로 묶어 설명한다는 점이 특징이다.

---

## 핵심 해석

- 의료영상 VFM 연구의 중심은 더 이상 backbone 교체만이 아니다.
- 지금 중요한 것은 다음 네 가지다.
- 어떻게 적응할 것인가
- 어떻게 3D와 modality gap을 처리할 것인가
- 어떻게 경량화하고 배포할 것인가
- 어떻게 privacy-preserving setting에서 협업 학습할 것인가
- 즉, 연구 초점이 `모델 자체`에서 `시스템 수준 활용`으로 이동한다.

---

## 강점

- 기술 축이 비교적 명확하다.
- SAM 이후 adaptation 연구를 빠르게 훑기 좋다.
- compression, distillation, federated learning을 한 흐름으로 묶는다.
- 의료영상의 실제 제약인 privacy, edge deployment, data scarcity를 분명히 반영한다.

---

## 한계

- 제목에 비해 실제 범위는 다소 좁고 segmentation 적응 쪽에 편중된다.
- 분류, 보고서 생성, pathology FM, vision-language FM 등은 상대적으로 얕다.
- 17페이지 수준의 짧은 review라 깊은 benchmark 분석은 부족하다.
- foundation model의 정의 자체는 다소 느슨하다.
- domain-native medical foundation model과의 비교도 충분하지 않다.

---

## 현재 시점에서의 의미

- 이 문서는 `의료영상 foundation model이 무엇인가`를 엄밀히 정의하는 문헌은 아니다.
- 대신 `현장에서 어떻게 맞춰 쓸 것인가`를 중심으로 한 실용 안내서에 가깝다.
- 특히 SAM 이후 adaptation, PEFT, LoRA, FL, lightweight deployment 흐름을
  한 번에 잡는 데 유용하다.
- 따라서 입문 survey이면서도 시스템 관점의 통찰이 있다.

---

## 발표용 핵심 메시지

- 의료영상에서 foundation model의 핵심 문제는 성능보다 적응이다.
- SAM 이후 연구의 중심은 zero-shot보다 adapter, PEFT, 3D 확장, prompt 개선으로 이동했다.
- 경량화와 distillation은 deployment 문제와 직접 연결된다.
- federated learning과 personalization은 의료 데이터 현실 때문에 필수 축이다.

---

## 결론

- `Vision Foundation Models in Medical Image Analysis: Advances and Challenges`는
  의료 VFM 연구를 대형 panorama로 정리한 문헌이라기보다
  `적응, 경량화, 연합학습` 중심으로 재구성한 짧은 실용 survey다.
- 가장 중요한 메시지는 명확하다.
- 의료영상에서 foundation model은 `그대로 가져다 쓰는 대상`이 아니라
  `도메인 적응과 시스템 최적화가 필요한 출발점`이다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/vision_foundation_models_in_medical_image_analysis_advances_and_challenges_slide.md>
