---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Introduction to Medical Imaging Informatics

- Md. Zihad Bin Jahangir et al.
- arXiv 2023
- Introductory overview of medical imaging as an informatics pipeline

---

## 문제 배경

- 의료영상은 단순 이미지가 아니다.
- 실제 진료에서는 다음 전체 체인이 함께 움직인다.
  - 영상 획득, 저장, 검색, 전송, 해석, 예측 모델링
- 이 논문은 의료영상 분석을
  **정보학 시스템 전체 안에서 이해해야 한다**고 설명한다.

---

## 이 논문의 핵심 기여

- medical imaging informatics의 정의와 범위를 기초 수준에서 정리한다.
- image processing, feature engineering, machine learning, deep learning을 하나의 흐름으로 연결한다.
- 저장·검색·해석·예측 모델링을 포괄하는 실무 구성요소를 소개한다.
- precision medicine 시대의 영상-임상데이터 통합 필요성을 강조한다.

---

## 의료영상 Informatics란 무엇인가

- 논문은 SIIM 정의를 바탕으로
  imaging informatics를 다음 전 과정을 다루는 분야로 본다.
  - image creation
  - acquisition
  - interpretation
  - reporting
  - communication
- 핵심 포인트:
  - PACS 운영만이 아니라 **의료영상 데이터 흐름 전체를 설계하는 영역**이다.

---

## 이 논문의 큰 구조

- 분야 정의
- 의료영상 실무 기능
- 기초 기술 스택
  - image processing
  - feature engineering
  - machine learning
  - deep learning
- 최근 computer vision 흐름과의 연결
- precision medicine과 데이터 통합 전망

---

## 의료영상 실무 기능

- 논문은 의료영상 informatics를 다음 기능으로 설명한다.
  - image coding, image processing, image distribution, acquisition device 연결, 환자 진료 지원 데이터 관리
- AI 활용 사례는 다음으로 묶는다.
  - image analysis and interpretation
  - computer-aided diagnosis
  - image-guided surgery
  - predictive analytics

---

## 기술 스택 1: Image Processing

- 가장 기초 계층은 image processing이다.
- 대표 구성:
  - enhancement
  - restoration
  - segmentation
  - compression
  - wavelet
  - morphology
- 메시지: 의료영상 AI도 결국 전처리와 품질 관리 위에 올라간다.

---

## 기술 스택 2: Feature Engineering에서 ML/DL까지

- 그 다음 단계는 raw data를 feature로 바꾸는 과정이다.
- 이어서 machine learning과 deep learning으로 연결된다.
- 논문이 설명하는 흐름:
  - feature engineering
  - supervised / unsupervised / reinforcement learning
  - CNN, RNN, LSTM, autoencoder, GAN, transfer learning
- 즉, informatics는 단일 모델보다 **계층형 기술 스택**이다.

---

## 이 논문의 가장 중요한 관점

- 의료영상 AI는 모델 문제가 아니라 파이프라인 문제다.
- 저장, 검색, 전송, 해석, 예측까지 연결해야
  실제 의료정보학 문제가 된다.
- 따라서 "좋은 segmentation model"만으로는 부족하다.
- 실제 가치가 생기려면
  **데이터 흐름과 임상 workflow 통합**이 필요하다.

---

## 영상은 곧 데이터 자산이다

- 논문은 radiomics와 precision medicine 맥락을 강조한다.
- 의료영상은 사람이 보는 그림이면서 동시에
  정량 특성을 추출할 수 있는 데이터다.
- 그래서 영상은 다음과 결합될수록 가치가 커진다.
  - EMR / EHR
  - 생리 데이터
  - 예후 정보
  - 의사결정 지원 시스템

---

## 최근 Computer Vision 흐름과의 연결

- 논문은 deep learning, transfer learning, GAN, robotics, AR, video analysis를
  의료영상 informatics와 연결한다.
- 이 부분은 최신 survey처럼 깊지는 않다.
- 하지만 의료영상 정보학이
  **컴퓨터비전 확장선 위에 있는 분야**라는 큰 그림을 준다.

---

## 강점

- 초심자에게 의료영상 informatics의 전체 지형을 빠르게 보여 준다.
- image processing부터 ML/DL, precision medicine까지 연결 구조가 명확하다.
- 의료영상 AI를 시스템 관점에서 보게 만든다.

---

## 한계

- 범위가 넓어 각 주제의 깊이는 제한적이다.
- 의료영상 informatics 고유 이론보다 일반 AI 개론에 가까운 부분도 있다.
- PACS, DICOM, RIS, workflow orchestration 같은 운영 수준 실무는 깊지 않다.
- foundation model, medical VLM, report-image alignment 같은 최신 흐름은 반영되지 않는다.

---

## 발표용 핵심 메시지

- 의료영상 AI는 모델보다 파이프라인 문제다.
- 영상 획득, 저장, 검색, 해석, 예측은 하나의 정보 흐름이다.
- medical imaging informatics는 이 전체 체인을 다루는 분야다.
- precision medicine 시대에는 영상 단독보다
  **영상과 임상 데이터의 통합 활용**이 더 중요해진다.

---

## 결론

- 이 논문은 의료영상 정보학을 처음 접하는 사람에게 적합한 입문 문헌이다.
- 최신 알고리즘 비교 논문은 아니지만,
  의료영상 분석을 데이터 관리, 저장·검색, ML/DL, precision medicine까지 연결된 체계로 보게 만든다.
- 연구 지형을 잡는 안내 문서로는 충분히 유용하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/introduction_to_medical_imaging_informatics_slide.md>
