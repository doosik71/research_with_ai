---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Explainable Artificial Intelligence (XAI) in Deep Learning-Based Medical Image Analysis

- Bas H.M. van der Velden, Hugo J. Kuijf, Kenneth G.A. Gilhuijs, Max A. Viergever
- arXiv 2021
- Survey of explanation methods and evaluation in medical imaging AI

---

## 문제 배경

- 딥러닝은 의료영상 분류, 탐지, 분할에서 높은 성능을 보인다.
- 하지만 임상에서는 정확도만으로 충분하지 않다.
- 중요한 질문은 다음이다.
  - 왜 이런 예측을 했는가?
  - 어떤 근거를 봤는가?
  - 그 설명을 신뢰할 수 있는가?
- 이 논문은 XAI를 단순 시각화 도구가 아니라
  **임상 신뢰 형성 문제**로 다룬다.

---

## 이 논문의 핵심 기여

- XAI를 세 축으로 분류한다.
  - model-based vs post hoc
  - model-specific vs model-agnostic
  - global vs local
- 실제 설명 형태를 세 가지로 정리한다.
  - visual explanation
  - textual explanation
  - example-based explanation
- 설명 평가도 별도 프레임으로 정리한다.

---

## XAI를 보는 세 가지 기준

- `Model-based vs Post hoc`
  - 모델 구조 자체가 해석 가능한가, 아니면 사후적으로 설명하는가
- `Model-specific vs Model-agnostic`
  - gradient / feature map에 의존하는가, 아니면 입력-출력 관계만 보는가
- `Global vs Local`
  - 데이터셋 수준 규칙을 설명하는가, 특정 환자 예측을 설명하는가
- 의료영상의 주류는
  **post hoc + model-specific + local explanation**이다.

---

## Visual Explanation

- 의료영상 XAI에서 가장 널리 쓰이는 형태다.
- 대표 기법:
  - CAM
  - Grad-CAM
  - guided backpropagation
  - Deep SHAP
  - attention visualization
- 목적: 모델이 실제로 병변이나 해부학적으로 타당한 영역을 보는지 확인
- 특히 Grad-CAM이 가장 자주 등장한다.

---

## Perturbation-based Visual Explanation

- gradient 대신 입력을 가리거나 변형해 중요도를 측정한다.
- 대표 기법:
  - occlusion sensitivity
  - LIME
  - meaningful perturbation
  - prediction difference analysis
- 장점: 모델 내부 gradient에 덜 의존한다.
- 의료영상에서는 pixel보다 **region / supervoxel 단위 설명**이 더 자연스러울 수 있다.

---

## Textual Explanation

- 설명을 heatmap이 아니라 문장, 개념, 보고서 형태로 제공한다.
- 대표 유형:
  - image captioning
  - caption + visual grounding
  - concept-based explanation (TCAV)
  - radiologist-style descriptive attributes
- 장점: 임상 문맥과 더 직접적으로 연결된다.
- 한계: 문장이 그럴듯해 보여도 실제 reasoning fidelity는 별도 검증이 필요하다.

---

## Example-based Explanation

- 현재 사례와 유사하거나 대조적인 사례를 보여 준다.
- 대표 방식:
  - triplet network retrieval
  - influence functions
  - prototype-based explanation
  - disentangled latent examples
- 의료진의 직관과 잘 맞는 설명 형식이다.
- 특히 prototype은
  "this looks like that" 형태로 비교적 직관적 설명을 제공한다.

---

## 이 논문이 보여 주는 실제 경향

- XAI 연구는 chest와 brain에 편중되어 있다.
- modality는 X-ray와 MRI 비중이 높다.
- visual explanation, 특히 Grad-CAM 계열이 압도적이다.
- 해석: 공개 데이터가 많고 구현이 쉬운 영역에서 먼저 확산된 것이다.

---

## 설명을 어떻게 평가할 것인가

- 이 논문의 중요한 공헌은 평가 프레임을 분리해 설명한 점이다.
- 세 가지 평가 방식:
  - **application-grounded**
  - **human-grounded**
  - **functionally-grounded**
- 핵심 메시지:
  - 시각적으로 그럴듯한 설명과
    실제로 충실한 설명은 다를 수 있다.

---

## 왜 비판이 중요한가

- 저자들은 XAI를 낙관적으로만 보지 않는다.
- 핵심 비판:
  - post hoc explanation은 모델의 실제 계산과 다를 수 있다.
  - saliency map은 파라미터가 바뀌어도 비슷하게 보일 수 있다.
  - explanation fidelity와 clinical usefulness는 별개다.
  - 단순 edge 강조가 의학적 reasoning처럼 보일 수 있다.
- 즉, explanation은 반드시 **faithfulness**를 검증해야 한다.

---

## 향후 방향

- visual + textual + example-based 설명의 결합
- biological explanation
  - 영상 특징과 분자/유전 정보 연결
- causality와 XAI의 결합
- 의료진의 추론 방식과 더 닮은 설명 인터페이스
  - 근거 영역
  - 언어적 서술
  - 유사 사례
  - 생물학적 맥락

---

## 강점

- visual explanation에만 머물지 않고 textual, example-based 방법까지 포함한다.
- 설명 평가 문제를 별도 섹션으로 다룬다.
- XAI 비판을 survey 내부에 포함해 균형이 좋다.
- 의료영상 XAI를 기술 목록이 아니라
  **검증 문제를 가진 연구 분야**로 보게 만든다.

---

## 한계와 이후 흐름

- 2021년 이전 연구 중심이라 이후의 흐름은 반영되지 않는다.
  - foundation model XAI
  - multimodal medical LLM explanation
  - prompt-based grounding
- 방법 taxonomy는 강하지만 과업별 정량 비교는 얕다.
- 그래도 핵심 질문은 여전히 유효하다.
  - 설명이 진짜 모델 reasoning을 반영하는가?

---

## 발표용 핵심 메시지

- 의료영상 XAI의 주류는 여전히 **post hoc, local, model-specific** 설명이다.
- saliency map은 출발점일 뿐 종착점이 아니다.
- 임상 신뢰를 위해서는
  - fidelity
  - robustness
  - evaluation protocol
  이 더 중요해질 것이다.
- 결국 유망한 방향은
  **근거 영역 + 언어적 설명 + 유사 사례**를 함께 주는 형태다.

---

## 결론

- 이 논문은 의료영상 XAI의 초기 기준점 역할을 하는 survey다.
- 핵심 가치는 설명 기법 분류, 적용 현황, 평가 프레임, 비판적 논의를 한 구조로 묶은 데 있다.
- 이후 foundation model 시대에도
  local/global, post hoc/model-based, fidelity/evaluation 문제는 여전히 중요한 기준으로 남는다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/explainable_artificial_intelligence_xai_in_deep_learning_based_medical_image_analysis_slide.md>
