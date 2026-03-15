# A Survey of Multi-Agent Deep Reinforcement Learning with Communication

## 1. 논문 메타데이터

- 제목: A Survey of Multi-Agent Deep Reinforcement Learning with Communication
- 저자: Changxi Zhu, Mehdi Dastani, Shihan Wang
- 발표: 2022 (arXiv), 2024 (Autonomous Agents and Multi-Agent Systems, Volume 38, Article 4, Published 06 Jan 2024)
- arXiv ID: 2203.08975
- DOI: 10.1007/s10458-023-09633-6
- URL: https://arxiv.org/abs/2203.08975

## 2. 연구 배경 및 문제 정의

멀티에이전트 강화학습(MARL) 환경에서는 부분 관측성과 비정상성으로 인해 개별 에이전트가 제한된 정보만으로 의사결정을 내려야 한다. 이런 상황에서 통신은 관측 정보, 의도, 경험 등을 공유해 환경 인식을 확장하고 협업 성능을 개선하는 핵심 메커니즘으로 작동한다. 최근 딥러닝 기반 MARL의 발전과 함께 통신을 학습하는 Comm-MADRL 연구가 급격히 증가했지만, 기존 설문 연구는 통신 프로토콜을 체계적으로 분류하거나 다양한 설계 축을 종합적으로 비교하는 데 한계가 있었다. 이 논문은 이러한 공백을 메우기 위해 Comm-MADRL을 구조적으로 분류하고 비교할 수 있는 분석 프레임워크를 제안한다.

## 3. 핵심 기여

1. Comm-MADRL 문헌을 체계적으로 정리하기 위한 9개 분석 차원(communication dimensions)을 제안한다.
2. 제안된 차원을 이용해 기존 연구들을 다차원 공간에 투영하여 공통점과 차이점을 명확히 비교할 수 있는 분류 체계를 제공한다.
3. 분류 결과를 바탕으로 현재 연구 경향을 도출하고, 다차원 조합을 통해 도달 가능한 새로운 연구 방향을 제시한다.

## 4. 방법론 요약

본 논문은 Comm-MADRL 시스템을 설계할 때 반드시 고려해야 하는 핵심 요소들을 9개 차원으로 구조화한다. 이 차원들은 크게 문제 설정, 통신 프로세스, 학습 프로세스라는 세 가지 구성요소와 연동된다.

### 4.1 문제 설정 중심 차원

- **Controlled Goals**: 에이전트 목표가 협력(cooperative), 경쟁(non-cooperative), 혼합(mixed) 중 어떤 형태인지 규정한다. 목표 구조는 통신이 왜 필요한지와 어떤 협력 패턴이 적절한지에 직접적인 영향을 준다.
- **Communication Constraints**: 비용, 대역폭, 잡음, 지연 등의 현실적 제약을 고려하는지 여부를 정의한다. 제약이 없는 설정은 학습을 단순화하지만 현실 적용 가능성을 낮출 수 있다.

### 4.2 통신 프로세스 중심 차원

- **Communicatee Type**: 통신 대상이 모든 에이전트인지, 특정 그룹인지, 혹은 중재자(proxy)를 포함하는지 등 통신 대상 구조를 정의한다.
- **Communication Policy**: 언제, 누구에게 통신할지를 결정하는 정책 구조를 의미한다. 사전 정의(full/partial)된 정책과 학습 기반 정책(개별 제어/글로벌 제어)으로 구분된다.
- **Communicated Messages**: 에이전트가 공유하는 메시지의 내용 유형을 정의한다. 관측/상태 기반의 기존 지식뿐 아니라 의도나 미래 예측 정보(모델 기반)까지 확장될 수 있다.
- **Message Combination**: 다수 에이전트가 보낸 메시지를 어떻게 집계하거나 선택하는지(합산, 어텐션, 선택적 필터링 등)를 규정한다.
- **Inner Integration**: 수신된 메시지를 정책 네트워크에 통합하는 방식(예: concatenate, gating, attention, memory 구조)을 의미한다.

### 4.3 학습 프로세스 중심 차원

- **Learning Methods**: 통신이 학습되는 방식(미분 가능 통신, 강화학습 기반 통신, 내재 보상/규칙 기반 보조 등)을 구분한다.
- **Training Schemes**: 경험 수집과 학습을 어떤 구조로 수행하는지 정의한다. 완전 분산 학습, 중앙집중 학습, CTDE(centralized training & decentralized execution) 등이 대표적이다.

위 9개 차원은 Comm-MADRL 접근법을 해부하는 공통 언어로 작동하며, 서로 다른 논문들이 어느 설계 축에서 차별화되는지 정량적으로 비교할 수 있게 한다.

## 5. 실험 설정과 결과

이 논문은 새로운 알고리즘 실험을 제시하기보다는, 기존 Comm-MADRL 연구들을 체계적으로 분류하고 경향을 분석하는 설문 연구이다. 9개 차원을 기준으로 대표 모델들을 표와 비교 분석으로 정리하고, 통신 방식과 학습 구조가 성능에 어떤 영향을 주는지 질적으로 논의한다. 또한 평가 지표를 정리하고, 통신 메커니즘의 효과를 측정하는 방식에 대한 논의를 포함한다.

## 6. 한계 및 향후 연구 가능성

저자들은 Comm-MADRL이 빠르게 성장하고 있음에도 여전히 다양한 한계가 존재한다고 지적한다. 특히 통신 제약을 고려한 현실적 설정, 더 다양한 목표 구조와 에이전트 구성, 통신의 해석 가능성과 안정성, 그리고 통신 효과를 정량적으로 비교할 수 있는 평가 체계가 향후 주요 연구 과제로 제시된다.

## 7. 실무적 또는 연구적 인사이트

1. **설계 공간을 명시적으로 정의하라**: Comm-MADRL 시스템을 설계할 때 9개 차원을 체크리스트처럼 활용하면, 기존 연구와의 차별점과 미탐색 영역을 명확히 파악할 수 있다.
2. **현실 제약을 포함한 통신 설계가 필요**: 비용, 지연, 잡음 등의 제약을 내재화하지 않으면 실제 적용에서는 성능이 급격히 저하될 수 있다.
3. **통신 평가 지표 확장**: 단순 보상 이상의 지표(communication efficiency, emergence metrics)를 함께 기록하면 통신 프로토콜의 역할을 해석하기 쉬워진다.
4. **멀티모달 및 구조적 통신**: 텍스트/음성 기반 통신과 구조적 통신(그래프 기반 메시지 전달)은 차세대 Comm-MADRL 응용을 위한 유망한 방향이다.
5. **보안·신뢰성 고려**: 중앙 프록시를 활용하는 경우 악성 메시지나 노이즈로부터 통신을 보호하는 설계가 필요하다.

## 참고

- Changxi Zhu, Mehdi Dastani, Shihan Wang. *A Survey of Multi-Agent Deep Reinforcement Learning with Communication*. Autonomous Agents and Multi-Agent Systems, 2024. (arXiv:2203.08975)
