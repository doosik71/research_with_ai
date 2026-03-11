# Continual Learning in Practice

**저자**: Tom Diethe, Tom Borchert, Eno Thereska, Borja Balle, Neil Lawrence

**연도**: 2019

**arXiv ID**: <https://arxiv.org/abs/1903.05202v2>

**발표 정보**: NeurIPS 2018 Continual Learning Workshop 공개본

---

## 1. 논문 메타데이터와 한 줄 요약

이 논문은 일반적인 Continual Learning 알고리즘을 제안하는 논문이라기보다,
운영 중인 ML 시스템이 데이터 분포 변화, 이상치, 지연 라벨, 재학습 비용,
모델 롤백 같은 현실 제약 속에서도 스스로 유지보수되도록 만드는
**참조 아키텍처(reference architecture)** 를 제안하는 시스템 관점의 논문이다.

저자들은 이를 `Auto-Adaptive ML` 또는 `zero-touch ML`의 중간 단계로 보고,
모델 학습 자체뿐 아니라 배포 이후의 감시, 재학습, 정책 결정, 저장소 관리,
품질 검증까지 포함한 전체 수명주기를 continual learning의 실제 문제로 재정의한다.

---

## 2. 연구 배경 및 문제 정의

대부분의 ML 시스템은 정적인 학습-배포 패러다임을 전제로 한다.
즉, 훈련 데이터로 모델을 학습한 뒤 이를 프로덕션에 배포하고,
배포 시점의 테스트를 통과하면 충분하다고 보는 접근이다.

하지만 실제 운영 환경에서는 다음 문제가 반복적으로 발생한다.

- 입력 데이터 분포가 시간에 따라 변한다.
- 라벨이 지연되어 들어오거나 다른 시스템에서 뒤늦게 합류한다.
- 하류 시스템이 기존 모델 출력에 맞춰 조정되어 있어 모델 교체가 위험하다.
- 재학습 자체에도 시간, 인프라, 운영 비용이 든다.
- 단순 정확도 외에 비즈니스 비용과 리스크가 의사결정에 반영되어야 한다.

논문의 핵심 문제의식은 다음과 같다.

- 배포 시점의 검증만으로는 시간이 흐른 뒤의 품질을 보장할 수 없다.
- 따라서 ML 시스템은 정적 소프트웨어 컴포넌트가 아니라,
  **계속 진단되고 필요 시 수정되는 적응형 시스템** 이어야 한다.
- Continual Learning은 단지 "망각 방지 알고리즘"이 아니라,
  **운영 중인 ML 시스템 전체를 어떻게 검증하고 갱신할 것인가**의 문제와 연결되어야 한다.

저자들은 이를 기존 회귀 테스트(regression testing)에 대응되는
**progression testing** 문제로 설명한다.
즉, 시간이 지남에 따라 세계가 변하는 상황에서 시스템이 여전히 기대 성능을 만족하는지
계속 검증해야 한다는 주장이다.

---

## 3. 핵심 기여

이 논문의 주요 기여는 다음 네 가지로 정리할 수 있다.

1. **실무형 Continual Learning 문제 재정의**

   Continual learning을 단순한 순차적 태스크 학습이 아니라,
   데이터 스트림, 지연 피드백, 모니터링, 재학습, 롤백, 로그 관리가 얽힌
   운영 환경의 문제로 확장한다.

2. **Auto-Adaptive ML 참조 아키텍처 제안**

   스트리밍 입력, 데이터 모니터링, 예측 모니터링, 정책 엔진, 학습기,
   HPO, 모델 저장소, 검증 데이터 저장소를 포함하는 모듈형 시스템 구성을 제안한다.

3. **재학습 정책을 의사결정 문제로 정식화**

   언제 재학습할지, 기존 모델을 유지할지, 롤백할지, transfer learning을 할지 등을
   단순 규칙이 아니라 상태-행동-보상 구조를 갖는 정책 문제로 보고,
   장기적으로는 강화학습 기반 policy engine이 적합하다고 본다.

4. **스케치 기반 요약과 데이터셋 시프트 감시의 결합**

   대용량 스트림을 전량 저장하지 않고도 운영 가능한 시스템을 위해
   sketcher/compressor와 drift/anomaly/change-point detection을 결합하는
   현실적인 운영 전략을 제시한다.

---

## 4. 제안 아키텍처 상세

논문 Figure 1의 참조 아키텍처는 크게 다음 컴포넌트들로 구성된다.

### 4.1 Streams

전체 시스템은 배치보다 **스트리밍 우선(stream-first)** 관점을 채택한다.
저자들은 현대 시스템이 실제로는 연속적인 데이터 유입 위에서 동작하므로,
batch는 stream의 특수한 경우로 다룰 수 있지만 그 반대는 어렵다고 본다.

이 선택은 다음 문제를 자연스럽게 포함한다.

- 데이터가 매우 빠르게 유입될 수 있음
- 데이터가 늦게 도착할 수 있음
- 데이터 순서가 뒤바뀔 수 있음
- 일부 데이터가 누락될 수 있음

### 4.2 Sketcher / Compressor

입력량이 너무 커서 모든 데이터를 그대로 하류 학습 시스템에 넘길 수 없을 때,
이 컴포넌트가 입력을 요약한다.

가장 단순한 방식은 uniform down-sampling이지만,
논문은 부록에서 다음과 같은 sketching 계열을 검토한다.

- Bloom filter
- Count-min sketch
- HyperLogLog
- Stream-summary
- t-digest
- Random projection

핵심 아이디어는 다음과 같다.

- 원본 데이터를 전부 유지하지 않아도 운영상 필요한 통계와 이상 신호를 유지할 수 있다.
- 어떤 질의가 필요한지 사전에 확정하기 어렵다면,
  여러 종류의 스케치를 동시에 유지하는 방법도 가능하다.
- 이 구성은 메모리 사용량과 지연 시간을 줄이는 현실적인 절충안이다.

### 4.3 Joiner

실무에서는 라벨이 입력과 동시에 주어지지 않는 경우가 많다.
예측 결과의 정답은 며칠 뒤 도착하거나,
다른 서비스/DB에서 합류될 수도 있다.

Joiner의 역할은 다음과 같다.

- 입력 스트림과 지연 라벨 또는 피드백을 결합한다.
- Trainer와 Predictor가 요구하는 형식으로 데이터를 정렬한다.
- weak feedback, delayed supervision 같은 운영 조건을 흡수한다.

즉, 이 논문은 continual learning을 "새 데이터가 오면 바로 학습"이라는
단순 그림이 아니라, **라벨 결합 파이프라인 문제**까지 포함해 다룬다.

### 4.4 Shared Infrastructure

공유 인프라는 단순 저장소가 아니라 지능형 캐시와 유사한 역할을 한다.

논문에서 제시하는 주요 저장 대상은 다음과 같다.

- `Model DB`: 학습된 모델과 모델 provenance 저장
- `Training DB`: 학습 에피소드 이력 저장
- `Validation Data Reservoir`: 검증용 데이터 유지
- `System State DB`: 정책 엔진의 상태 판단 근거 저장
- `Diagnostic Logs`: 디버깅과 사후 감사용 로그 저장

저자들은 특히 **provenance**를 강조한다.
어떤 시점에 왜 그 모델이 사용되었는지, 왜 재학습 또는 롤백이 발생했는지
추적 가능해야 한다는 뜻이다.

또한 검증 데이터는 무한정 쌓는 것이 아니라,
공간 제약 아래 최근 데이터를 더 중시하는 reservoir 방식으로 유지할 수 있다고 본다.

### 4.5 Data Monitoring

이 컴포넌트는 입력 데이터 수준에서 drift, anomaly, change-point를 감시한다.

논문이 요구하는 최소 출력은 다음 세 가지다.

- shift 유형
- shift 크기
- 불확실성 추정

가능하다면 어떤 feature가 변화를 유발했는지 설명도 제공해야 한다고 본다.

부록에서는 데이터셋 시프트를 다음처럼 분류한다.

- covariate shift
- prior probability shift
- sample selection bias
- domain shift
- source component shift
- anomaly detection

그리고 탐지 방법은 크게 다음으로 구분한다.

- supervised monitoring
- unsupervised statistical distance methods
- anomaly detection
- change-point detection

논문은 실제 운영 환경에서는 라벨을 즉시 확보하기 어렵기 때문에,
완전한 supervised monitoring보다 **unsupervised shift detection**의 비중이 크다고 본다.

### 4.6 Prediction Monitoring

Prediction monitoring은 입력이 아니라 **출력과 시스템 상태**를 감시한다.

예시 상태 변수는 다음과 같다.

- 최근 재학습 이후 경과 시간
- 현재 시점의 중요도
- 예측 실패 비용
- 재학습 비용
- 상태별 최적 정책 추정값

즉, 이 계층은 정확도 숫자만 보는 것이 아니라,
예측 가치와 운영 비용을 함께 추적하는 비즈니스-연동형 모니터링 계층이다.

### 4.7 Trainer, HPO, Predictor

Trainer는 기존 조직의 학습 파이프라인을 감싸는 wrapper 역할을 한다.
논문은 새 학습기를 처음부터 만드는 것보다,
기존 학습 파이프라인에서 **재학습 단계와 메타데이터 기록을 체계화하는 것**을 더 현실적인 접근으로 본다.

Trainer가 기록해야 할 메타데이터 예시는 다음과 같다.

- validation metric
- 학습 시작 시각과 소요 시간
- hyperparameter 설정
- 훈련된 모델 정보

HPO는 warm-start 기반 재학습 효율화를 지원한다.
Predictor는 새 모델을 받아 실제 배포로 이어지는 구간을 담당한다.

### 4.8 Model Policy Engine

이 논문에서 가장 실무적으로 중요한 모듈이다.
정책 엔진은 다음 질문에 답해야 한다.

- 지금 재학습할 것인가
- 기존 모델을 유지할 것인가
- 이전 모델로 롤백할 것인가
- transfer learning을 적용할 것인가

저자들이 제시한 핵심 판단 축은 다음 네 가지다.

1. **Horizon**
   최신 데이터가 얼마나 빨리 모델에 반영되어야 하는가.
   또 어떤 시점의 데이터가 더 이상 중요하지 않게 되는가.

2. **Cadence**
   재학습을 언제 수행해야 하는가.
   drift 신호, 상태 추정, 가능한 행동 집합을 바탕으로 결정해야 한다.

3. **Provenance**
   어떤 결정이 왜 내려졌는지 추적할 수 있어야 한다.

4. **Costs**
   재학습 비용과 미재학습 비용을 함께 고려해야 한다.

저자들은 단순 규칙 기반 정책이 베이스라인으로는 가능하지만,
장기적으로는 policy engine 자체를 **강화학습 문제**로 다루는 것이 바람직하다고 주장한다.
이때 상태는 데이터 모니터링, 학습기, 예측 모니터링의 출력으로 구성되고,
행동은 재학습/유지/롤백/전이학습 등이 되며,
보상은 비즈니스 메트릭을 기반으로 구성된다.

---

## 5. 방법론적 성격과 논문의 범위

이 논문은 CL 벤치마크에서 특정 알고리즘의 평균 정확도나 forgetting score를
경쟁하는 형태의 실험 논문이 아니다.

대신 다음 특징을 가진다.

- 문제 정의가 알고리즘보다 시스템 설계에 가깝다.
- 핵심 산출물이 수식 기반 학습 목표가 아니라 아키텍처와 컴포넌트 분해다.
- "어떤 CL 알고리즘이 더 좋으냐"보다
  "운영 중인 ML 시스템을 어떻게 self-maintaining하게 만들 것이냐"를 다룬다.

따라서 이 논문은 continual learning 문헌 중에서도
**MLOps, AutoML, streaming systems, monitoring, policy learning**의 교차점에 위치한다.

---

## 6. 실험 및 결과

이 논문에는 일반적인 continual learning 논문에서 기대하는 형태의
정량 실험 벤치마크가 사실상 없다.

즉, 다음 항목은 제공되지 않는다.

- Split-MNIST, CIFAR, ImageNet 계열 정확도 비교표
- forgetting measure 비교
- SOTA 알고리즘과의 성능 경쟁

대신 논문은 다음을 제공한다.

- 시스템 아키텍처 설계안
- 각 컴포넌트의 역할 정의
- sketching 기법에 대한 부록 수준의 실무 참고
- dataset shift detection 기법에 대한 부록 수준의 실무 참고

이 점은 약점이면서도 동시에 논문의 의도와 맞닿아 있다.
저자들의 목표는 "새 CL 알고리즘 제안"이 아니라
"현실적인 CL 시스템의 운영 구조 제안"이기 때문이다.

---

## 7. 논문의 강점

### 7.1 Continual Learning을 운영 문제로 끌어내림

많은 CL 논문은 task sequence와 성능 지표에 집중하지만,
실무에서는 데이터 도착 지연, 라벨 부족, 재학습 비용, 배포 리스크가 더 큰 문제다.
이 논문은 그 간극을 정확히 짚는다.

### 7.2 모듈형 구조

아키텍처가 완전히 모듈식으로 설계되어 있어,
팀이 전체를 한 번에 도입하지 않고 부분적으로 채택할 수 있다.
예를 들어 데이터 모니터링만 먼저 붙이거나,
정책 엔진 없이 규칙 기반 재학습만 먼저 도입할 수 있다.

### 7.3 CL과 MLOps의 연결

이 논문은 오늘날의 MLOps 관점에서 다시 읽어도 유효하다.
모델 registry, validation reservoir, health monitoring, rollback, provenance 같은 개념은
현재의 production ML 시스템에서도 핵심 설계 원리다.

### 7.4 정책 최적화 관점

재학습 주기를 hand-tuned scheduler로 두지 않고,
상태-행동-보상 문제로 보는 관점은 이후 자동 재학습, adaptive pipeline,
RL for operations 같은 주제로 자연스럽게 이어진다.

---

## 8. 한계 및 비판적 해석

### 8.1 실증 검증 부족

참조 아키텍처 자체의 유효성을 정량적으로 입증하는 대규모 실험이 없다.
따라서 어떤 구성요소가 실제로 어느 정도의 비용 대비 효과를 내는지는
이 논문만으로는 판단하기 어렵다.

### 8.2 아키텍처 수준의 추상성

각 컴포넌트의 역할은 명확하지만,
컴포넌트 간 인터페이스 계약, 장애 처리, 상태 동기화 방식,
데이터 품질 SLA 같은 구현 세부는 제시되지 않는다.

### 8.3 Policy Engine의 난이도

강화학습 기반 정책 엔진은 개념적으로 매력적이지만,
실제 운영에서는 탐험 비용과 안정성 문제가 크다.
논문도 이를 해결했다고 주장하지 않으며,
오히려 활발한 연구 주제로 남겨둔다.

### 8.4 Continual Learning 알고리즘 자체와는 거리

망각 억제 메커니즘, rehearsal, regularization, parameter isolation 같은
전통적인 CL 알고리즘 비교 관점에서는 직접적인 기여가 제한적이다.
즉, CL 알고리즘 논문이라기보다 CL 시스템 설계 논문으로 읽어야 정확하다.

---

## 9. 후속 연구와의 연결점

이 논문은 이후 다음 주제들과 강하게 연결된다.

- online continual learning
- task-free continual learning
- production monitoring for ML
- model retraining policy optimization
- drift detection and data quality observability
- adaptive AutoML / zero-touch ML

특히 오늘날의 관점에서 보면,
이 논문은 **continual learning을 실제 운영 파이프라인으로 끌고 가기 위한 설계 청사진**
역할을 한다.
최근의 MLOps 플랫폼, feature store, observability stack,
자동 재학습 워크플로와도 자연스럽게 맞물린다.

---

## 10. 실무자를 위한 핵심 인사이트

### 10.1 정확도만 보고 재학습하면 안 된다

재학습 여부는 단순 성능 하락만으로 결정할 수 없다.
재학습 비용, 서비스 중요도, 실패 비용, 롤백 가능성까지 함께 봐야 한다.

### 10.2 데이터 모니터링과 예측 모니터링을 분리해야 한다

입력 분포가 바뀌었는지와 모델 출력이 비정상적인지는 서로 다른 신호다.
둘을 분리해야 원인을 더 잘 해석할 수 있다.

### 10.3 라벨 지연을 기본 가정으로 둬야 한다

실서비스에서는 즉시 라벨이 들어오는 경우가 드물다.
따라서 joiner, delayed feedback, validation reservoir 같은 구성은
부가 기능이 아니라 핵심 기능이다.

### 10.4 전체 자동화보다 단계적 도입이 현실적이다

논문도 모듈형 도입을 전제로 한다.
현실적으로는 다음 순서가 적절하다.

1. 입력/예측 모니터링 구축
2. 모델 및 데이터 provenance 확보
3. 규칙 기반 재학습 정책 도입
4. HPO 및 warm-start 자동화
5. 필요 시 학습형 policy engine 도입

---

## 11. 종합 평가

`Continual Learning in Practice`는 continual learning을
순차 학습 벤치마크의 문제에서 꺼내어,
운영 중인 ML 시스템의 수명주기 관리 문제로 확장한 점에서 의미가 크다.

이 논문은 새로운 망각 방지 알고리즘을 제시하지는 않지만,
실무에서 continual learning이 실제로 성립하려면 무엇이 필요한지에 대해
매우 구체적인 체크리스트를 제공한다.

정리하면 이 논문의 가치는 다음과 같다.

- CL을 시스템 공학 문제로 재해석했다.
- 모니터링, 재학습, 롤백, provenance를 하나의 아키텍처로 묶었다.
- 향후 zero-touch ML과 adaptive MLOps로 이어질 설계 방향을 제시했다.

반면, 정량 검증과 구현 세부가 부족하므로
실제 적용에는 별도의 엔지니어링 설계와 실험이 반드시 필요하다.

---

## 12. 참고 링크

- arXiv abs: <https://arxiv.org/abs/1903.05202v2>
- arXiv PDF: <https://arxiv.org/pdf/1903.05202v2.pdf>
- Amazon Science publication page:
  <https://www.amazon.science/publications/continual-learning-in-practice>

---

*이 요약은 원문 PDF를 기준으로 작성했으며, 논문의 성격상 알고리즘 성능 비교보다
시스템 아키텍처와 운영 관점의 의미를 중심으로 정리했다.*
