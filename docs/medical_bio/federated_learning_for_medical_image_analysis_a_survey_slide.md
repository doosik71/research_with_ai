---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Federated Learning for Medical Image Analysis: A Survey

- Hao Guan, Pew-Thian Yap, Andrea Bozoki, Mingxia Liu
- arXiv survey, version referenced: 2024-07-07
- Privacy-preserving collaborative learning for medical imaging

---

## 문제 배경

- 한 기관 데이터만으로는 환자 다양성과 장비 차이를 충분히 반영하기 어렵다.
- 가장 직관적인 해법은 다기관 데이터 통합이다.
- 하지만 실제로는 다음 제약이 크다.
  - HIPAA / GDPR
  - 병원 정책
  - 원본 영상 공유 제한
- FL은 데이터를 모으지 않고
  **모델 업데이트만 교환**해 협업 학습하는 대안이다.

---

## FL이 의료영상에서 왜 중요한가

- raw image를 중앙에 모으지 않아도 다기관 학습이 가능하다.
- 기관별 데이터 다양성을 간접적으로 활용할 수 있다.
- 하지만 개인정보 문제가 자동으로 해결되지는 않는다.
  - gradient leakage
  - malicious update
  - communication bottleneck
- 논문은 FL을 만능 해법이 아니라
  **privacy와 협업 학습 사이의 절충안**으로 본다.

---

## 이 논문의 핵심 기여

- 2017년부터 2023년 10월까지 의료영상 FL 연구를 체계적으로 정리한다.
- taxonomy를 세 부분으로 나눈다.
  - **client-end**
  - **server-end**
  - **client-server communication**
- benchmark dataset, software platform, 대표 FL baseline 비교까지 포함한다.

---

## FL 기본 절차

- 서버가 global model을 초기화한다.
- 선택된 client가 로컬 데이터로 모델을 학습한다.
- client는 raw data 대신 model update만 서버에 보낸다.
- 서버는 업데이트를 aggregation해 새 global model을 만든다.
- global model을 다시 client에 배포한다.
- 이 단순한 루프 안에 의료영상 FL의 핵심 문제가 모두 들어 있다.

---

## 기본 유형

- `Horizontal FL`
  - feature space가 유사한 여러 병원이 협업
  - 가장 일반적인 의료영상 FL 형태
- `Vertical FL`
  - 같은 환자 집합에 대해 서로 다른 feature를 기관별로 보유
  - 영상 + EHR + 유전체 같은 정밀의료 시나리오에서 중요
- 발표 포인트:
  - 현재 의료영상 FL의 주류는 horizontal FL이다.

---

## 이 논문의 핵심 Taxonomy

- **Client-end learning**
  - client shift, limited labels, heterogeneous resources
- **Server-end learning**
  - aggregation, fairness, corrupted client handling
- **Communication**
  - privacy leakage, differential privacy, communication efficiency
- 이 구조는 FL를 시스템 설계 문제로 보게 만든다.

---

## Client-End: Client Shift

- 의료영상 FL의 핵심 난제는 병원 간 분포 차이다.
  - 스캐너 차이, 프로토콜 차이, 환자군 차이, 라벨 정책 차이
- 대표 대응:
  - personalized FL
  - shared encoder + local decoder
  - local discriminator / local module
  - federated domain adaptation
  - harmonization / style alignment
- 메시지: 모든 client에 완전히 같은 모델을 강제하면 잘 깨진다.

---

## Client-End: Limited Labels

- 각 client 내부 데이터도 작고 라벨은 더 적다.
- 그래서 다음 전략이 나온다.
  - self-supervised pretraining
  - representation learning
  - teacher-student distillation
  - semi-supervised / weakly supervised FL
  - synthetic sample generation
---

## Client-End: Heterogeneous Resources

- 또 병원마다 연산 자원과 네트워크 속도도 다르다.
- 즉, FL는 분산 최적화이면서 동시에 **운영 시스템 문제**다.

---

## Server-End: Aggregation과 Fairness

- 서버는 단순 평균기만이 아니다.
- aggregation이 성능과 공정성을 좌우한다.
- 대표 방향:
  - FedAvg / FedProx
  - client loss 기반 가중 aggregation
  - Fourier-based aggregation
  - underperforming client 가중 반영
- 핵심 질문:
  - 어떤 병원의 업데이트를 얼마나 반영할 것인가?

---

## Server-End: Corrupted Clients

- 모든 client가 항상 정상적이라고 가정할 수 없다.
- 실제 문제:
  - noisy label, poor image quality, malicious update, poisoning
- 논문은 outlier score 기반 suppression처럼
  비정상 client를 감지하고 가중치를 낮추는 전략을 소개한다.
- 의료영상에서는 이것이 security 문제이면서
  data quality management 문제이기도 하다.

---

## Communication: Privacy와 Efficiency

- raw data를 안 보낸다고 privacy가 끝나는 것은 아니다.
- gradient나 BN 통계로부터도 정보가 새어 나올 수 있다.
- 대응:
  - partial weight sharing, differential privacy, attack / defense 실험
- 동시에 통신 비용도 크다.
  - dynamic client selection
  - timeout handling
  - selective participation

---

## 실험이 주는 메시지

- 논문은 ADNI 기반 비교에서 다음 경향을 보인다.
  - centralized pooling(`Mix`)이 가장 좋다
  - cross-site direct transfer(`Cross`)는 가장 나쁘다
  - FL은 privacy 제약하에서 `Cross`와 `Single`보다 낫다
  - weight aggregation 계열이 gradient aggregation보다 유리했다
- 결론: FL은 centralized learning을 완전히 대체하진 않지만
    **현실적 타협점**이 될 수 있다.

---

## 강점

- client / server / communication 관점의 taxonomy가 명확하다.
- privacy, heterogeneity, efficiency를 한 프레임에서 본다.
- software platform과 benchmark dataset 정리가 실용적이다.
- survey인데도 baseline 비교 실험을 포함해 감각이 좋다.

---

## 한계와 이후 흐름

- 2024년 이전 연구 중심이라 이후 흐름은 부족하다.
  - federated foundation model
  - parameter-efficient FL
  - multimodal medical LLM 연계
- personalization 간 공정 비교는 제한적이다.
- 그래도 시스템 관점 분류는 지금도 유효하다.

---

## 발표용 핵심 메시지

- 의료영상 FL의 본질은 단순 privacy-preserving optimization이 아니다.
- 진짜 문제는
  **heterogeneous, privacy-sensitive, low-label 환경에서의 협업 학습**이다.
- client shift를 해결하지 못하면 FL은 단순 분산 평균에 머문다.
- 앞으로는 federation 내부 성능보다
  **unseen client generalization과 long-term operability**가 더 중요해질 수 있다.

---

## 결론

- 이 논문은 의료영상 FL를 가장 체계적으로 정리한 문헌 중 하나다.
- 핵심 가치는 알고리즘 이름이 아니라
  **시스템 구성요소 기준으로 문제를 분해한 것**에 있다.
- 의료기관 연합 환경에서 FL를 설계하려면
  client heterogeneity, aggregation fairness, privacy leakage, infrastructure constraints를 함께 봐야 한다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/federated_learning_for_medical_image_analysis_a_survey_slide.md>
