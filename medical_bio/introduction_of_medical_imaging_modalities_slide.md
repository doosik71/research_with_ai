---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Introduction of Medical Imaging Modalities

- S. K. M Shadekul Islam et al.
- arXiv 2023 / referenced version 2024
- Introductory review of major medical imaging modalities

---

## 문제 배경

- 의료영상 AI를 하려면 먼저 modality 차이를 이해해야 한다.
- 같은 "의료영상"이라도 다음은 본질적으로 다르다.
  - 물리 원리
  - 해상도
  - 안전성
  - 비용
  - 임상 목적
- 이 논문은 주요 modality를 넓게 개관하는 입문 문헌이다.

---

## 이 논문의 핵심 기여

- 대표 의료영상 modality의 원리와 임상 활용을 한 문서에서 정리한다.
- 각 modality의 장점과 한계를 비교 가능하게 제시한다.
- diagnostic imaging과 therapeutic imaging 구분을 설명한다.
- EIT, cardiovascular imaging, data mining/search 같은 확장 주제도 포함한다.

---

## 전체 구조

- 기본 modality 소개
  - X-ray, CT, MRI, Ultrasound, Nuclear Imaging, EIT
- 확장 주제
  - contrast-enhanced MRI
  - osteoarthritis MR approaches
  - cardiovascular imaging
  - medical imaging data mining and search

---

## X-ray와 CT

- **X-ray**
  - 가장 오래되고 널리 쓰이는 modality
  - 골격계 평가에 강함
  - 장점: 접근성, 속도
  - 한계: soft tissue 구분 약함, ionizing radiation 사용
- **CT**
  - X-ray 기반 단층 영상
  - 뇌, 흉부, 복부, 골조직 평가에 강함
  - 장점: 빠르고 정밀함 / 한계: 방사선 노출

---

## MRI

- 강한 자기장과 radio wave를 사용한다.
- 연부조직 시각화에 매우 강하다.
- 대표 활용:
  - 뇌, 척추, 근골격계
- 장점:
  - ionizing radiation 없음, soft tissue contrast 우수
- 한계:
  - 비용과 검사 시간, 장비 접근성, 환자 적합성 제약

---

## Ultrasound

- 비침습적이고 안전하며 실시간성이 강하다.
- 대표 활용:
  - 산과, 복부, 혈관, 심장
- 장점:
  - 휴대성, 낮은 비용, 반복 사용 가능
- 한계:
  - operator dependence, 시야와 해상도 제한

---

## Nuclear Imaging

- PET 등 핵의학 영상은 구조보다 기능과 대사를 본다.
- tracer uptake를 통해 생리적 변화를 측정한다.
- 대표 활용:
  - 종양, 심장, 갑상선 질환
- 장점:
  - 기능 평가에 강함
- 한계:
  - 방사성 추적자 사용

---

## EIT와 확장 기술

- **EIT**
  - 전기적 특성 변화를 이용해 내부 구조를 추정
  - 장점: 저비용, 비침습 모니터링 잠재력
  - 한계: 해상도 약함
- 논문은 EIT를 기존 modality 대체재보다
  **신흥 보완 기술**로 소개한다.

---

## Cardiovascular Imaging과 Search

- 심혈관 영상은 단일 modality가 아니라 조합이다.
  - CTA, MRA, CMR, echocardiography
- 또 의료영상 활용은 촬영에서 끝나지 않는다.
  - CBIR, radiomics, NLP, data mining / search
- 의미:
  - 의료영상은 acquisition 이후 분석과 검색까지 이어지는 시스템이다.

---

## 이 논문의 실질적 메시지

- modality 차이는 곧 데이터 분포 차이다.
- 같은 모델이나 같은 annotation 전략을
  모든 modality에 그대로 적용하면 안 된다.
- X-ray, MRI, PET는 서로 다른 정보를 담는다.
- 따라서 모델 선택, 평가 지표, 임상 해석도 modality별로 달라져야 한다.

---

## 강점

- modality 전반을 빠르게 훑기 좋은 입문 문헌이다.
- 원리, 대표 활용, 장점, 한계를 한 번에 정리한다.
- data mining과 search까지 연결해 의료영상 활용 범위를 넓게 보여 준다.

---

## 한계

- 범위가 넓은 대신 modality별 심화는 제한적이다.
- 체계적 review보다는 개론형 chapter에 가깝다.
- AI 기반 분석 기법과의 직접 연결은 얕다.
- radiation dose optimization, multimodal fusion workflow, foundation model 활용 같은 최신 이슈는 부족하다.

---

## 발표용 핵심 메시지

- 의료영상 modality 차이는 단순 입력 형식 차이가 아니다.
- 물리 원리, 해상도, 안전성, 임상 목적이 모두 다르다.
- 의료영상 AI의 첫 단계는 모델 선택보다
  **어떤 modality가 어떤 정보를 담는지 이해하는 것**이다.
- 이 논문은 그 지형을 빠르게 잡게 해 주는 입문서다.

---

## 결론

- 이 논문은 의료영상 modality 전반을 이해하기 위한 개론 문헌이다.
- 최신 알고리즘 survey는 아니지만,
  X-ray, CT, MRI, ultrasound, nuclear imaging, EIT의 차이를 구조적으로 정리해 준다.
- 의료영상 AI를 시작하기 전 배경지식을 잡는 데 유용하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/introduction_of_medical_imaging_modalities_slide.md>
