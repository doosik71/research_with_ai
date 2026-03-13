---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# CLIP in Medical Imaging: A Survey

- Zihao Zhao et al.
- arXiv preprint, surveyed through 2025-03-26
- Medical vision-language learning with CLIP-style image-text alignment

---

## 문제 배경

- 기존 의료영상 AI는 image-only 학습에 강하게 의존해 왔다.
- 하지만 실제 의료 현장에는 영상과 함께 다음 텍스트가 존재한다.
  - 판독문
  - 임상 기술문
  - 병리 설명
  - 보고서 문장 구조
- 논문의 출발점:
  - 의료영상에서 중요한 의미는 이미지에만 있지 않고
    **전문가 언어와의 정렬**에도 있다.

---

## CLIP이 의료영상에서 왜 중요한가

- CLIP은 이미지와 텍스트를 공통 임베딩 공간에 정렬한다.
- 의료 도메인에서의 장점:
  - 텍스트 감독을 통한 진단 개념 정렬
  - zero-shot / prompt-based 추론 가능
  - explainability 개선 가능성
  - classification을 넘어 segmentation, retrieval, report generation까지 확장
- 즉, CLIP은 단순 backbone이 아니라
  **의료 영상과 전문가 언어를 연결하는 기반 모델**이다.

---

## 이 논문의 핵심 기여

- 의료 영상 CLIP 연구 224편을 정리한 대규모 survey다.
- taxonomy를 두 축으로 명확히 나눈다.
  - **Refined CLIP pre-training**
  - **CLIP-driven applications**
- 핵심 난제를 세 가지로 정리한다.
  - multi-scale features
  - data scarcity
  - specialized knowledge demands

---

## 의료용 CLIP의 핵심 난제

- **Multi-scale feature**
  - 작은 병변과 문장 단위 소견 정렬이 중요하다.
- **Data scarcity**
  - 웹 규모 image-text 쌍이 부족하다.
- **Specialized knowledge**
  - 의료 텍스트는 전문 용어와 구조적 개념을 포함한다.
- 발표 포인트:
  - 의료용 CLIP은 자연 이미지용 CLIP의 단순 이식이 아니다.

---

## Refined Pre-training 1: Multi-scale Contrast

- global image-text contrast만으로는 의료 데이터에 부족하다.
- 대표 방향:
  - word-level / sentence-level alignment
  - local region alignment
  - anatomy-aware alignment
  - prototype / memory / reconstruction 보강
- 대표 예시:
  - GLoRIA,  LoVT, PRIOR
- 핵심은 보고서의 문장 구조와 영상의 해부학 구조를 함께 반영하는 것이다.

---

## Refined Pre-training 2: Data-efficient Contrast

- 의료 데이터는 적고, in-batch negative 가정도 잘 맞지 않는다.
- 따라서 다음 전략이 등장한다.
  - semantic correlation 기반 soft target
  - positive / neutral / negative 세분화
  - disease-level cluster 활용
  - Findings / Impression / uncertainty 정보 활용
- 대표 예시:
  - MedCLIP, SAT, MGCA, BioViL / BioViL-T

---

## Refined Pre-training 3: Knowledge Enhancement

- 단순 contrastive loss만으로는 의료 개념 구조를 충분히 반영하기 어렵다.
- 그래서 외부 지식 주입이 중요해진다.
  - entity-relation graph, UMLS, RadGraph, ontology / prompt knowledge
- 의미:
  - CLIP을 co-occurrence learner에서 **clinically structured model**로 확장하려는 시도다.

---

## CLIP-driven Applications

- 논문은 downstream 활용을 세 부류로 정리한다.
  - classification
  - dense prediction
  - cross-modal tasks
- 이 구분의 장점:
  - CLIP이 단지 zero-shot 분류기인지,
    아니면 의료 AI 전반의 공통 모듈인지가 드러난다.

---

## Classification

- zero-shot disease classification
- prompt engineering
- context optimization
- observation-first reasoning

---

## Dense Prediction

- detection
- anomaly detection
- 2D segmentation
- 3D segmentation
- keypoint localization

특히 segmentation에서는 text embedding이
  task semantics를 제공하는 조건 벡터 역할도 한다.

---

## Cross-modal Tasks

- CLIP은 다음 과제로도 확장된다.
  - report generation
  - image-text retrieval
  - MedVQA
  - image synthesis
- 저자들의 해석:
  - retrieval과 report generation은
    CLIP이 실제 진단 보조 인터페이스로 이어질 가능성을 보여 준다.

---

## 이 논문의 핵심 해석

- image-only 모델은 높은 성능을 내도 설명성과 정렬성이 약할 수 있다.
- CLIP 기반 모델은 텍스트 개념을 매개로 사용해
  zero-shot, explainability, retrieval augmentation에서 구조적 장점이 있다.
- 반면 모든 modality에서 바로 강한 것은 아니다.
  - 3D 영상
  - 세밀한 병변 경계
  - 적은 데이터
  에서는 추가 설계가 필요하다.

---

## 강점

- taxonomy가 분명하고 연구 지형을 빠르게 파악할 수 있다.
- pre-training과 downstream을 하나의 프레임으로 연결한다.
- 의료 도메인의 특수 문제를 자연 이미지 CLIP과 구분해 설명한다.
- multi-scale alignment, data-efficient contrast, knowledge enhancement라는
  세 축이 매우 실용적이다.

---

## 한계와 향후 과제

- 현재 연구는 흉부 X-ray 편중이 강하다.
- 3D CT/MRI/PET에서는 volume-level alignment와 메모리 문제가 크다.
- 데이터와 언어 편향, multilingual bias, fairness 문제가 남는다.
- 평가도 downstream accuracy에 치우쳐 있다.
- 앞으로는 다음이 중요하다.
  - clinically grounded evaluation
  - multimodal medical AI
  - richer knowledge integration

---

## 발표용 핵심 메시지

- 의료용 CLIP의 성패는 모델 크기보다
  **보고서 구조와 해부학 구조를 얼마나 잘 반영하느냐**에 달려 있다.
- 핵심 과제는 multi-scale alignment, data scarcity, domain knowledge다.
- CLIP은 분류기보다
  **vision-language medical foundation model의 출발점**에 가깝다.
- 다음 단계는 3D, multilingual, multimodal, clinically grounded evaluation이다.

---

## 결론

- 이 논문은 의료 영상 CLIP 연구를 가장 체계적으로 정리한 문헌 중 하나다.
- 핵심 가치는 refined pre-training과 application taxonomy를 명확히 연결한 데 있다.
- CLIP을 자연 이미지 모델의 이식이 아니라
  **의료 텍스트와 영상의 구조를 반영해 재설계해야 하는 분야**로 정의한다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/clip_in_medical_imaging_a_survey_slide.md>
