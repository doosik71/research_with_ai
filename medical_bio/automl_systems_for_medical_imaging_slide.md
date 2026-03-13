---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# AutoML Systems For Medical Imaging

- Tasmia Tahmida Jidney, Angona Biswas, MD Abdullah Al Nasim, Ismail Hossain, Md Jahangir Alam, Sajedul Talukder, Mofazzal Hossain, Md Azim Ullah
- arXiv 2023 / version referenced in summary: 2024
- Medical imaging pipeline automation from feature design to NAS

---

## 문제 배경

- 의료영상 분석은 MRI, CT, X-ray 등 다양한 modality를 다룬다.
- 실제 모델 개발에는 다음 수작업이 많이 들어간다.
  - feature engineering
  - algorithm selection
  - hyperparameter tuning
  - architecture design
- 논문의 출발점은 명확하다.
  - 의료 현장에는 데이터는 늘지만 ML 전문성은 부족하다.
- 따라서 AutoML은 성능 향상 도구이면서 진입 장벽 완화 도구다.

---

## 논문의 핵심 질문

- 의료영상에서 AutoML은 무엇을 자동화하는가?
- 어떤 구성요소가 실제 시스템의 중심인가?
- classification 외의 imaging workflow까지 확장 가능한가?
- 임상 적용 시 어떤 제약이 남는가?

---

## 핵심 구성요소

- 저자들은 AutoML을 세 축으로 정리한다.
  - **Automated Feature Engineering**
  - **Automated Hyperparameter Optimization**
  - **Neural Architecture Search**
- 이 세 요소를 묶어 보면 AutoML은
  - 단일 알고리즘이 아니라
  - **모델 개발 자동화 시스템**이다.

---

## AutoML 파이프라인 관점

- 입력 데이터에서 표현 설계를 자동화한다.
- 후보 모델과 탐색 공간을 정의한다.
- search 또는 optimization으로 최적 조합을 찾는다.
- 최종적으로 성능이 좋은 pipeline을 선택한다.
- 발표 포인트:
  - AutoML의 본질은 "좋은 모델 하나"보다
    **좋은 탐색 절차 설계**에 가깝다.

---

## 의료영상에서 왜 특히 중요한가

- modality가 다양해 수작업 최적화 비용이 크다.
- annotation 부족과 데이터 이질성이 흔하다.
- 병원이나 연구실마다 ML 엔지니어링 역량 차이가 크다.
- 이런 조건에서 AutoML은
  - 반복 실험 비용을 줄이고
  - 비전문가 접근성을 높이는 수단으로 해석된다.

---

## 응용 영역

- 논문이 다루는 대표 응용은 다음과 같다.
  - diagnosis support
  - decision-making support
  - personalized medicine
  - medical image segmentation
  - medical image registration
  - medical image synthesis
  - medical image augmentation
- 즉, AutoML을 단순 classifier search에 한정하지 않는다.

---

## 실험 결과로 봐야 할 부분

- 이 논문은 benchmark 중심 비교 논문이 아니다.
- 직접적인 대규모 정량 비교보다
  - 문헌 사례
  - 응용 영역 정리
  - 시스템 구성 설명
  에 초점을 둔다.
- 예시로 MRI 기반 fluid intelligence prediction에서
  2600개 이상의 ML pipeline을 평가한 사례를 소개한다.
- 따라서 이 논문은 성능표보다 **문제 지도**가 핵심이다.

---

## 강점

- 의료영상 AutoML의 필요성과 구성요소를 빠르게 파악하게 해 준다.
- feature engineering, HPO, NAS를 일관된 틀로 묶는다.
- segmentation, synthesis, augmentation까지 응용 범위를 넓게 본다.
- 임상 적용 시 privacy, interpretability, transparency 이슈를 함께 언급한다.

---

## 한계

- Google AutoML, H2O, AutoKeras 같은 시스템 간 직접 비교가 약하다.
- 비용, 재현성, 배포 난이도 같은 실무 판단 기준이 부족하다.
- 의료영상 특화 이슈를 깊게 다루지는 않는다.
  - class imbalance
  - domain shift
  - annotation scarcity
  - regulatory validation
- foundation model 이후 흐름과는 거리감이 있다.

---

## 현재 시점에서의 해석

- 오늘날 AutoML은 NAS 자체보다 다음에 더 가깝다.
  - workflow orchestration
  - foundation model adaptation
  - parameter-efficient transfer tuning
  - multimodal pipeline composition
- 따라서 이 논문은 최신 솔루션 가이드라기보다
  - **의료영상 AutoML의 출발 문제를 정리한 개론 문헌**으로 읽는 편이 적절하다.

---

## 발표용 핵심 메시지

- AutoML의 가치는 최고 성능만이 아니라 **개발 자동화와 접근성 향상**에 있다.
- 의료영상에서는 모델 자동화보다 데이터 거버넌스가 더 중요해질 수 있다.
- 임상 도입의 병목은 accuracy만이 아니라
  - transparency
  - fairness
  - workflow integration
  이다.

---

## 결론

- 이 논문은 의료영상에서 AutoML이 왜 필요한지 정리하는 입문형 survey다.
- 핵심은 feature engineering, HPO, NAS를 통한 pipeline automation이다.
- 응용 범위는 넓지만 정량 비교는 제한적이다.
- 따라서 이 논문은 **어떤 도구를 바로 고를지**보다
  **의료영상 AutoML을 어떤 문제로 볼지**를 정리해 준다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/automl_systems_for_medical_imaging_slide.md>
