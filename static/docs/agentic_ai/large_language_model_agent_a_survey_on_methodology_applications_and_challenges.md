# Large Language Model Agent: A Survey on Methodology, Applications and Challenges

## 1. Paper Overview

이 논문은 LLM agent 연구를 하나의 통합된 관점에서 정리하려는 **대규모 survey**다. 핵심 목표는 개별 논문들을 단순 나열하는 것이 아니라, LLM agent를 이해하는 데 필요한 중심 축을 **구성(construction)**, **협업(collaboration)**, **진화(evolution)**라는 세 가지 방법론 차원으로 재구성하는 데 있다. 저자들은 여기에 더해 **평가와 도구**, **보안·프라이버시·사회적 영향**, **응용 분야**까지 함께 묶어, 현재의 agent 생태계를 전주기적으로 해석할 수 있는 프레임워크를 제시한다. 즉, “LLM agent를 어떻게 만들고, 어떻게 여러 agent가 함께 일하며, 시간이 지나면서 어떻게 개선되는가?”라는 질문에 답하는 구조화된 지도라고 볼 수 있다.

이 문제가 중요한 이유는, LLM agent가 더 이상 단순 질의응답 시스템이 아니라 **환경을 인식하고, 목표를 세우고, 계획하고, 행동하는** 방향으로 진화하고 있기 때문이다. 논문은 이러한 변화를 기존 AI agent와 구별되는 세 가지 동력으로 설명한다. 첫째는 LLM의 추론 능력 향상, 둘째는 도구 사용 및 환경 상호작용 능력, 셋째는 장기 경험 축적을 가능하게 하는 memory 구조다. 저자들은 이 조합이 assistant를 collaborator로 바꾸고 있으며, AGI로 가는 중요한 경로 중 하나가 될 수 있다고 본다.

## 2. Core Idea

이 논문의 중심 아이디어는 **LLM agent 연구를 기능 모듈별로 쪼개는 대신, 방법론적 연결 관계까지 포함하는 하나의 taxonomy로 재구성**하는 것이다. 기존 survey들이 특정 응용(게임, 멀티모달, 보안 등)에 치우치거나, 반대로 너무 넓게 다뤄서 설계 원리의 내부 구조를 충분히 드러내지 못했다면, 이 논문은 **Build–Collaborate–Evolve**라는 축을 통해 agent 시스템의 설계와 동작을 연속선상에서 본다.

이 관점의 장점은 다음과 같다.

첫째, single-agent 연구와 multi-agent 연구를 분리하지 않고 연결한다. 예를 들어 agent 내부의 profile, memory, planning, action 모듈은 협업 구조의 기반이 되고, 협업 과정에서 축적된 경험과 피드백은 다시 evolution으로 이어진다. 둘째, agent를 정적인 아키텍처가 아니라 **순환적 최적화 시스템**으로 본다. 셋째, 실제 배포 시 중요한 security, privacy, ethics까지 함께 다루므로 연구자뿐 아니라 실무자와 정책 측면에서도 유용하다.

논문이 내세우는 참신성은 “완전히 새로운 알고리즘 제안”이 아니라, **산발적으로 존재하던 agent 연구를 설계 원리 중심으로 정렬하고 상호 연결성을 드러낸 것**에 있다. 특히 저자들은 role definition, memory mechanism, planning capability, action execution 같은 agent 구성 요소를 독립 모듈로만 보지 않고, 실제 시스템에서는 상호의존적 루프를 형성한다고 본다.

## 3. Detailed Method Explanation

### 3.1 전체 구조

이 논문은 survey이므로 새로운 모델 파라미터화나 학습 objective를 제안하지 않는다. 대신 Figure 1을 중심으로 LLM agent 생태계를 네 개의 큰 층위로 정리한다.

1. **Agent Methodology**

   * Construction
   * Collaboration
   * Evolution

2. **Evaluation and Tools**

3. **Real-World Issues**

   * Security
   * Privacy
   * Social Impact

4. **Applications**

즉, 이 논문의 “방법”은 어떤 loss를 최소화하는 학습 알고리즘이 아니라, 기존 연구를 이해하기 위한 **분류 체계와 해석 프레임워크**의 설계라고 보는 것이 정확하다.

### 3.2 Agent Construction

논문에 따르면 agent construction은 goal-directed behavior를 가능하게 하는 핵심 설계 단계이며, 네 개의 상호의존적 pillar로 구성된다.

#### 3.2.1 Profile Definition

Profile definition은 agent의 역할, 정체성, 목표, 행동 스타일을 정의하는 층이다. 논문은 이를 크게 두 방향으로 본다.

* **Human-curated static profiles**
  사람이 역할과 규칙을 명시적으로 부여하는 방식
* **Batch-generated dynamic profiles**
  데이터나 상황에 따라 role/persona를 동적으로 생성하는 방식

이 구분은 agent가 단순한 프롬프트 래퍼가 아니라, 특정 역할을 가진 행위자로 동작하게 만드는 출발점이다. 예를 들어 software team 시뮬레이션, 사회적 agent, 추천 agent 등에서는 profile이 전체 행동의 초기 조건이 된다.

#### 3.2.2 Memory Mechanism

논문은 memory를 agent의 장기적 일관성과 적응성을 좌우하는 핵심 요소로 본다. 여기서 중요한 분류는 다음과 같다.

* **Short-Term Memory**
  현재 작업 맥락과 최근 상호작용을 유지
* **Long-Term Memory**
  과거 경험, 반성, 기술, 지식 등을 저장
* **Knowledge Retrieval as Memory**
  외부 지식베이스 검색을 memory처럼 활용

이 분류는 agent의 context window 한계를 넘어서기 위한 실용적 방향을 제시한다. 특히 장기 memory는 reflection, self-improvement, skill reuse와 연결되고, retrieval 기반 memory는 RAG/GraphRAG 계열과 연결된다. 논문은 메모리가 planning에 정보를 제공하고, 실행 결과가 다시 메모리를 갱신하는 **재귀 루프**를 이룬다고 설명한다.

#### 3.2.3 Planning Capability

Planning은 LLM agent를 단순 응답기에서 task solver로 바꾸는 핵심이다. 논문은 planning을 크게 두 흐름으로 정리한다.

* **Task Decomposition Strategies**
  복잡한 문제를 하위 문제로 쪼개는 전략
* **Feedback-Driven Iteration**
  실행 중 피드백을 받아 계획을 반복 개선하는 전략

여기서 중요한 관찰은 planning이 단발성 reasoning이 아니라는 점이다. ReAct, Tree-based search, discussion-based planning, behavior tree 방식 등 다양한 연구가 소개되며, agent는 목표를 바로 해결하지 않고 중간 단계와 분기 탐색을 통해 더 안정적인 행동을 만들어 낸다.

#### 3.2.4 Action Execution

논문은 action execution을 단순 텍스트 출력이 아니라, 외부 세계와 연결되는 실제 행위 계층으로 본다.

* **Tool Utilization**
  API, 코드 실행기, 검색기, 특수 도구 호출
* **Physical Interaction**
  로봇, 자율주행, embodied agent 같은 물리적 환경 상호작용

이 부분은 “agent”를 chatbot과 구분하는 결정적 요소다. LLM이 사고를 담당하고, 실제 작업은 도구 호출이나 환경 제어로 이어질 때 비로소 goal-driven autonomy가 형성된다.

### 3.3 Agent Collaboration

논문은 multi-agent를 단순히 agent 수를 늘린 구조가 아니라, **통제와 의사소통 방식**에 따라 다른 설계 원리를 갖는다고 본다.

#### 3.3.1 Centralized Control

중앙 coordinator 또는 manager가 전체 흐름을 통제하고, 각 하위 agent는 지정된 역할을 수행한다. 장점은 전체 최적화와 일관성 유지가 쉽다는 점이고, 단점은 bottleneck과 single point of failure 가능성이다.

#### 3.3.2 Decentralized Collaboration

각 agent가 보다 자율적으로 상호작용하며 문제를 해결하는 구조다. debate, peer review, distributed reasoning 같은 형태가 여기에 포함된다. 장점은 다양성과 견고성이고, 단점은 조정 비용과 불안정성이다.

#### 3.3.3 Hybrid Architecture

실제 시스템은 중앙집중형과 분산형을 혼합하는 경우가 많다. 고수준 task allocation은 central controller가 맡고, 세부 추론이나 역할 기반 토론은 분산형으로 진행하는 식이다. 논문은 이 hybrid가 실무적으로 가장 자연스러운 방향일 수 있음을 시사한다.

### 3.4 Agent Evolution

이 논문에서 흥미로운 부분은 agent를 고정 아키텍처가 아니라 **시간에 따라 발전하는 시스템**으로 본 점이다.

#### 3.4.1 Autonomous Optimization and Self-Learning

self-refine, self-verification, reward-guided refinement 등 agent가 스스로 산출물을 검토하고 개선하는 흐름이다. 이는 단순 inference-time trick이 아니라 agent competence를 점진적으로 높이는 메커니즘으로 해석된다.

#### 3.4.2 Multi-Agent Co-Evolution

여러 agent가 경쟁·토론·협력을 반복하면서 함께 성능이 향상되는 구조다. 한 agent의 오류가 다른 agent의 비판 기제가 되고, 논쟁 자체가 더 좋은 해답 탐색 과정이 된다.

#### 3.4.3 Evolution via External Resources

외부 도구, 검색, 지식베이스, 피드백 데이터, 실행 결과를 받아 agent를 개선하는 방향이다. 이는 closed-world LLM의 한계를 보완하는 현실적인 경로이며, 논문의 전반적 메시지와도 잘 맞는다.

### 3.5 Evaluation, Tools, Real-World Issues, Applications

이 논문은 방법론만 다루지 않고, agent 연구를 실전 단계로 옮길 때 필요한 주변 축도 함께 포함한다.

* **Evaluation and Tools**: benchmark, framework, protocol, 개발 생태계 정리
* **Real-World Issues**: security, privacy, social impact
* **Applications**: 과학, 의학, 소프트웨어, embodied setting 등

특히 Real-World Issues 파트는 LLM agent가 실제 사회 시스템과 상호작용할 때 발생하는 위험을 구조적으로 정리한다. 보안 위협은 agent-centric와 data-centric로 나뉘고, privacy는 memorization과 intellectual property 문제까지 포함하며, social impact는 효용과 윤리적 위험을 함께 본다.

## 4. Experiments and Findings

이 논문은 일반적인 method paper처럼 하나의 시스템을 구현해 benchmark에서 수치 비교를 하는 형태가 아니다. 따라서 “실험”은 좁은 의미의 controlled experiment라기보다, **광범위한 문헌 조사와 사례 비교를 통해 도출한 정리 결과**라고 보는 것이 맞다. 이 점은 읽을 때 반드시 구분해야 한다.

그럼에도 논문이 제시하는 중요한 발견은 분명하다.

첫째, LLM agent 연구는 이제 단순 prompting 단계를 넘어, **memory–planning–action–collaboration**이 결합된 시스템 공학 문제로 이동하고 있다. 둘째, multi-agent 구조는 단순 병렬화가 아니라 역할 분담, 조정, 토론, 검증 메커니즘 설계가 핵심이다. 셋째, 실사용 단계에서는 성능 향상만큼이나 **보안, 프라이버시, 신뢰성, 과학적 엄밀성**이 중요한 평가 축이 된다.

또한 논문은 현재의 연구 흐름이 세 방향으로 수렴한다고 해석할 수 있게 한다.

* agent 내부 구조의 정교화
* agent 간 협업 구조의 조직화
* self-improvement와 external feedback 기반 진화

즉, 앞으로의 agent 연구는 더 큰 base model만으로 해결되지 않고, **시스템 설계 원리**가 경쟁력을 좌우할 가능성이 크다는 메시지를 준다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **정리 방식의 명료함**이다. LLM agent 분야는 너무 빠르게 확장되고 있어, 개별 논문을 많이 읽어도 전체 그림을 잡기 어렵다. 그런데 이 논문은 construction, collaboration, evolution이라는 세 축을 중심으로 연구를 재배열해, 서로 다른 논문들이 어떤 자리에 놓이는지 비교 가능하게 만든다.

두 번째 강점은 **실전성을 놓치지 않았다는 점**이다. 단순 taxonomy에 머물지 않고, evaluation, tools, security, privacy, social impact, applications를 포함해 실제 agent deployment에서 중요한 문제를 함께 다룬다. 이 때문에 이 논문은 입문자용 survey를 넘어서, 연구 기획이나 시스템 설계의 출발점으로도 유용하다.

세 번째 강점은 **single-agent와 multi-agent, 정적 설계와 동적 진화**를 하나의 프레임 안에 넣었다는 점이다. 이 연결은 앞으로 agent 연구가 어디로 갈지 생각하는 데 도움을 준다.

### Limitations

한편 한계도 있다.

첫째, survey 논문이므로 개별 방법의 수학적 세부나 구현 디테일을 깊게 파고들지는 않는다. 따라서 특정 프레임워크를 실제로 재현하거나 구현하려는 독자에게는 원 논문들을 추가로 봐야 한다.

둘째, taxonomy는 매우 유용하지만, 분야가 워낙 빠르게 변하기 때문에 분류 체계 자체가 곧 다시 확장될 수 있다. 예를 들어 MCP 같은 프로토콜, agent OS, browser-use 계열, real-world orchestration 플랫폼이 급속히 발전하면 현재 분류보다 더 세밀한 운영 계층 taxonomy가 필요해질 수 있다.

셋째, 논문은 평가의 중요성을 강조하지만, 실제로 agent evaluation은 여전히 불안정하다. 특히 장기 상호작용, tool failure, 안전성, 사회적 영향은 표준화가 부족하다. 논문 후반부도 reliability와 scientific rigor 문제를 지적하며, hallucination과 stochastic instability가 high-stakes 환경에서 치명적일 수 있음을 강조한다.

### Interpretation

비판적으로 보면, 이 논문은 “LLM agent가 무엇인가”를 아주 넓게 정의한다. 그 결과 survey로서는 포괄적이지만, 한편으로는 agent와 advanced workflow, tool-augmented reasoning, orchestration framework 사이의 경계가 다소 넓게 잡혀 있다. 하지만 현재 연구 흐름 자체가 그 경계를 흐리고 있기 때문에, 오히려 이것이 현실을 잘 반영한 선택이라고도 볼 수 있다. 내 해석으로는, 이 논문의 진짜 가치는 개별 기술의 우열 비교보다 **agent를 시스템적·진화적 객체로 보게 만든 사고 틀**에 있다.

## 6. Conclusion

이 논문은 LLM agent 분야를 이해하기 위한 매우 강력한 survey다. 핵심 기여는 agent 연구를 **construction–collaboration–evolution**이라는 세 축으로 조직하고, 여기에 **평가, 도구, 실제 이슈, 응용**을 결합해 하나의 통합된 지형도를 제시한 데 있다. 논문은 LLM agent를 단순 prompt engineering의 연장이 아니라, memory, planning, action, coordination, self-improvement를 포함한 **복합 시스템 설계 문제**로 재정의한다.

실무적으로는 agent framework 설계, multi-agent orchestration, tool-use 시스템, 안전한 deployment 전략을 고민하는 사람들에게 방향성을 제공한다. 연구적으로는 앞으로 중요한 질문이 “더 큰 LLM인가?”보다 “어떤 구조의 agent 시스템이 더 신뢰 가능하고 확장 가능하며 안전한가?”로 이동하고 있음을 잘 보여준다. survey 논문이지만, 분야의 현재와 다음 단계를 동시에 읽게 해주는 좋은 기준점이다.
