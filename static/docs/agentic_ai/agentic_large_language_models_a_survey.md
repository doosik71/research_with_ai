# Agentic Large Language Models, a survey

## 1. Paper Overview

이 논문은 최근 급격히 확장되고 있는 **agentic LLM** 연구를 정리한 survey다. 저자들은 agentic LLM을 단순히 텍스트를 생성하는 모델이 아니라, **(1) reason, (2) act, (3) interact** 하는 시스템으로 정의한다. 즉, 외부 정보를 찾고, 스스로 반성하며, 도구나 로봇을 통해 실제 세계에 작용하고, 다른 에이전트와 사회적 상호작용까지 수행하는 LLM 계열을 하나의 통합된 틀로 보려는 시도다. 논문의 가장 큰 목적은 방대한 최근 문헌을 정리하고, 이들을 공통 taxonomy 아래 배치하며, 앞으로 무엇이 중요한 연구 과제가 될지를 제안하는 데 있다.  

이 논문이 중요한 이유는 agentic LLM이 단순한 프롬프트 기반 질의응답을 넘어, 의료 진단 보조, 물류, 금융 분석, 과학 연구 자동화 같은 고영향 응용으로 빠르게 확장되고 있기 때문이다. 저자들은 동시에, 이런 에이전트형 시스템이 단지 응용을 위한 기술일 뿐 아니라 **새로운 training data를 생성하는 메커니즘**이 될 수 있다고 본다. 즉, LLM이 세계와 상호작용해 만든 새로운 state와 feedback이, 향후 pretraining이나 finetuning의 입력이 될 수 있다는 주장이다. 이는 “LLM이 학습 데이터 고갈에 직면하고 있다”는 문제의식과도 연결된다.

## 2. Core Idea

논문의 중심 아이디어는 agentic LLM 연구를 세 갈래로 분류하는 것이다.

* **Reasoning**: 더 잘 생각하고, 계획하고, 반성하고, 검색하는 능력
* **Acting**: 외부 세계의 도구, API, 브라우저, 로봇 등을 사용해 실제 작업을 수행하는 능력
* **Interacting**: 다른 에이전트나 사람과 역할 기반 협력, 전략적 상호작용, 사회적 시뮬레이션을 수행하는 능력

이 taxonomy는 단순한 목차 정리가 아니다. 저자들은 세 범주가 서로 독립이 아니라 **상호보완적**이라고 강하게 주장한다. 예를 들어 retrieval은 tool use를 가능하게 하고, self-reflection은 multi-agent 협업을 개선하며, reasoning은 전 범주를 떠받친다. 또 acting과 interacting 과정에서 생성된 데이터는 다시 reasoning 계열 성능 향상에 사용될 수 있어, 전체가 하나의 **virtuous cycle**을 이룬다고 설명한다.

논문의 novelty는 특정 새 알고리즘 제안이 아니라, 서로 분산돼 있던 recent literature를 “reasoning–acting–interacting”이라는 강한 개념 틀로 재구성했다는 데 있다. 특히 이 틀은 agentic LLM을 단순히 “tool-using LLM”으로 축소하지 않고, reasoning enhancement와 social multi-agent dynamics까지 포괄한다는 점에서 더 넓고 설명력이 크다.  

## 3. Detailed Method Explanation

### 3.1 Agentic LLM의 정의

논문은 “models predict, agents reason, act, and interact”라는 식으로 agent와 model을 구분한다. 일반 모델이 입력에 대한 출력을 반환하는 수동적 시스템이라면, agent는 정보 탐색, 의사결정, 환경 변화 감지, 행동 수행, 커뮤니케이션을 수행하는 보다 능동적 시스템이다. 이 정의는 자연어처리뿐 아니라 robotics, reinforcement learning, multi-agent systems, symbolic AI 등의 전통과도 연결된다. 즉, agentic LLM은 완전히 새로운 발명이 아니라, 기존 AI의 여러 전통 위에 LLM이 얹힌 최신 형태로 이해된다.

### 3.2 왜 agentic LLM이 필요한가

저자들은 네 가지 배경 문제를 제시한다.

첫째, **prompt engineering** 문제다. 기본 LLM은 prompt wording에 민감하고, 사용자가 매번 더 나은 프롬프트를 수동으로 작성해야 한다.
둘째, **hallucination** 문제다. 그럴듯하지만 사실과 다른 출력을 내는 문제가 여전히 크며, grounding과 self-reflection이 필요하다.
셋째, **reasoning** 문제다. 특히 수학적/다단계 문제 해결에서 기존 LLM은 취약했고, step-by-step reasoning이나 search가 이를 개선하는 축으로 등장했다.
넷째, **training data** 문제다. 기존처럼 더 큰 정적 코퍼스에만 의존해서는 한계가 있고, inference-time retrieval이나 interaction으로 새로운 데이터를 만들어야 한다는 것이다.

저자들의 논리는 명확하다. 이러한 문제들이 retrieval, self-verification, prompt-improvement, tool use, multi-agent interaction 같은 inference-time augmentation을 낳았고, 바로 이 흐름이 agentic LLM 연구로 이어졌다는 것이다. 따라서 agentic LLM은 단순한 응용 트렌드가 아니라, 기본 LLM이 드러낸 구조적 한계를 보완하기 위해 등장한 연구 패러다임으로 볼 수 있다.

### 3.3 Reasoning 범주

Reasoning 범주는 세 하위축으로 정리된다.

* **Multi-step reasoning**
* **Self-reflection**
* **Retrieval augmentation**

논문은 Chain-of-Thought, Zero-shot CoT, Tree of Thoughts, Reflexion, Self-Refine, ReAct, retrieval augmentation 등을 이 축 아래 배치한다. 설명에 따르면 reasoning 범주의 목적은 기본 LLM의 의사결정 품질을 높이는 것이다. 특히 planning/search를 이용한 step-by-step reasoning, 자동 prompt improvement, 그리고 inference-time retrieval이 핵심 기술적 요소다. 저자들은 이 범주가 이후 acting과 interacting 범주의 기반 기술 역할을 한다고 본다.

### 3.4 Acting 범주

Acting 범주는 agent가 실제 세계에 **행동을 가하는 인터페이스**를 가지는 경우를 다룬다. 세 하위축은 다음과 같다.

* **World Models / Vision-Language-Action**
* **Robots and Tools**
* **Assistants**

여기서 world model과 VLA는 로봇이나 embodied system이 어떤 행동을 취해야 하는지를 학습하는 계열을 포함한다. robots/tools는 API, planner, browser, computer-use 같은 외부 도구 연결을 포함한다. assistants는 이런 도구 사용이 실제 사용자 가치를 만드는 응용 계층으로, 의료 보조, 거래 분석, 과학 연구 보조 등이 대표적 사례다. 저자들은 이 범주의 핵심을 “usefulness for users”로 표현하며, acting 과정이 다시 새로운 interactive training data를 생산한다는 점도 강조한다.  

### 3.5 Interacting 범주

Interacting은 agentic LLM을 사회적 맥락에 놓는 범주다. 하위축은 다음과 같다.

* **Social capabilities of LLMs**
* **Role-based interaction**
* **Simulating open-ended societies**

이 범주에서는 개별 agent의 지능 향상보다, 여러 agent의 상호작용에서 나타나는 협력, 전략, 사회 규범, 역할 분담, emergent behavior가 핵심 주제가 된다. 저자들은 특히 role-based teamwork와 open-ended societies가 social science 연구에도 중요한 실험 도구가 될 수 있다고 본다. 즉, LLM multi-agent simulation은 단순 benchmark가 아니라 사회적 현상 연구의 새로운 방법론으로 제시된다.

### 3.6 Virtuous cycle과 training pipeline 재해석

논문이 흥미로운 점은 agentic LLM을 단지 inference engineering으로 보지 않는다는 것이다. 저자들은 reasoning–acting–interacting이 연결된 **virtuous cycle**을 통해, acting과 interacting이 만든 경험 데이터가 다시 pretraining, finetuning, inference augmentation에 쓰일 수 있다고 설명한다. 이는 기존 LLM training pipeline의 “Acquire → Pretrain → Finetune → Align → Infer” 구조에, **interaction-generated data**라는 새 흐름을 덧붙이는 시각이다. 특히 training data plateau 문제를 해결할 가능성으로 agentic behavior를 본다는 점이 이 survey의 중요한 철학적 메시지다.  

## 4. Experiments and Findings

이 논문은 survey이므로 단일 실험 세트를 제시하는 empirical paper는 아니다. 대신 각 하위 분야의 대표 방법과 사용 사례를 분류하고, 어떤 방향의 성과가 축적되고 있는지 메타 수준에서 요약한다. 따라서 “실험 결과”는 특정 benchmark score보다 **연구 흐름과 사용 사례의 정리**로 이해하는 것이 적절하다.

### 4.1 Reasoning 쪽에서 보여주는 것

Reasoning 계열은 CoT, Tree of Thoughts, Reflexion, retrieval augmentation, interpreter/debugger 류 방법들이 기본 LLM의 한계였던 다단계 추론, 자기 수정, 최신 정보 접근을 개선하는 축으로 정리된다. 다만 저자들은 이 범주의 많은 use case가 아직은 math word problem, QA, algorithm generation, benchmark 최적화 중심이라고도 지적한다. 즉, reasoning research는 agentic behavior의 기반을 제공하지만, 실용성 측면에서는 acting 범주로 넘어가야 더 직접적인 사용자 가치가 발생한다는 뉘앙스가 있다.  

### 4.2 Acting 쪽의 응용 성과

Acting 범주에서는 robots, tools, assistants가 실제 응용으로 이어지고 있음을 보여준다. 논문 초록과 결론 수준에서 강조하는 대표 응용은 **medical diagnosis, logistics, financial market analysis**다. 또한 science assistants 영역에서는 AI Scientist 같은 시스템이 아이디어 생성부터 실험, 논문 작성, 리뷰까지 연구 workflow를 자동화하려는 시도로 제시된다. 저자들은 이 분야의 결과를 유망하게 보지만, 동시에 구현 오류, 제한된 실험, 시각적 오류, hallucination 가능성 등 현재 한계도 함께 언급한다. 즉, acting 범주는 가장 실용적이지만, 동시에 grounding과 reliability가 가장 중요한 영역이다.

### 4.3 Interacting 쪽의 함의

Interacting 범주에서는 multi-agent role play, teamwork, strategic behavior, emergent norms, open-ended societies가 중요한 주제로 정리된다. 저자들은 이 축이 사회과학 연구의 새로운 도구가 될 수 있다고 본다. 즉, 인간과 유사한 자연어 상호작용 능력을 가진 agent society를 통해 협력, 규범, 갈등, 전략적 상호작용 등을 시뮬레이션할 수 있다는 것이다. 이는 이전 세대 agent-based models보다 더 현실적인 사회 실험을 가능하게 할 잠재력으로 제시된다.  

### 4.4 논문이 실제로 보여주는 핵심 finding

이 survey가 전달하는 가장 중요한 finding은 다음이다.

* reasoning, acting, interacting은 개별 기술 스택이 아니라 서로 강화하는 구조다.
* retrieval은 tool use를 가능하게 하고, reflection은 collaboration을 개선한다.
* assistants와 multi-agent interaction은 단지 응용이 아니라 새로운 training data 원천이 될 수 있다.
* 실용적 가치와 과학적 가치가 동시에 존재한다. 사회적으로는 의료/금융/물류/일상 비서가 중요하고, 학문적으로는 social simulation과 automated science가 중요하다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **taxonomy의 설명력**이다. agentic LLM을 reasoning–acting–interacting 세 축으로 정리함으로써, 복잡하게 흩어진 최근 문헌을 이해하기 쉬운 구조로 재배치했다. 특히 reasoning만 다루는 연구, tool use만 다루는 연구, multi-agent simulation만 다루는 연구를 하나의 공통 프레임 안에 넣은 점이 강하다.  

두 번째 강점은 **연구 agenda 제시**다. 이 논문은 단순 요약으로 끝나지 않고, training data, reflection, safety, automated scientific discovery, human/agent behavior modeling, wider assistant applications 같은 후속 과제를 명시적으로 제안한다. 이는 survey로서 좋은 구조를 가진다.  

세 번째 강점은 **응용과 기초 연구를 동시에 묶는다**는 점이다. 많은 survey가 benchmark나 architecture에만 집중하는 반면, 이 논문은 의료·금융·로보틱스·과학 자동화와 사회 시뮬레이션까지 하나의 큰 흐름으로 본다.  

### 한계

가장 큰 한계는 survey 특유의 문제로, **폭은 넓지만 개별 방법에 대한 깊이는 제한적**이라는 점이다. 이 논문은 매우 많은 recent work를 다루기 때문에, 각 방법의 알고리즘 세부나 실험 재현성, 공정 비교 조건을 깊게 검토하는 논문은 아니다. 따라서 특정 접근의 우월성을 정밀하게 평가하기보다는 연구 지도(map)를 제공하는 데 더 가깝다. 이는 논문의 의도이기도 하다.

둘째, 포함된 문헌 중 상당수가 2024~2025의 최근 작업과 preprint를 포함하므로, field의 빠른 변화 속도에 비해 survey가 금방 일부 outdated될 수 있다. 저자들도 자신들이 “current status”를 다루고 있을 뿐이라고 밝힌다.

셋째, taxonomy가 강력한 장점인 동시에, 때로는 실제 시스템들이 여러 범주를 동시에 가질 때 경계를 흐릴 수 있다. 저자들도 일부 접근이 세 요소를 모두 가진다고 인정한다. 즉, reasoning/acting/interacting은 깔끔한 분류 기준이지만, 실제 agentic system은 점점 hybrid해지고 있어 향후 taxonomy 확장이 필요할 수 있다. 이 부분은 논문 구조를 바탕으로 한 해석이다.

### 해석

비판적으로 보면, 이 논문의 진짜 공헌은 “agentic LLM이 무엇인가”를 기능 목록이 아니라 **연결 구조**로 설명했다는 데 있다. 즉, reasoning은 더 좋은 decision making을, acting은 사용자 가치와 데이터 생성을, interacting은 사회적 지능과 emergent behavior를 담당하며, 이 셋이 다시 학습 사이클을 구성한다는 시각이다. 이 때문에 이 논문은 survey이면서 동시에 하나의 **research program 선언문**처럼 읽힌다.  

## 6. Conclusion

이 논문은 agentic LLM을 **reasoning, acting, interacting**의 세 범주로 정리하고, 각 범주가 서로 강화되는 구조를 제시한다. reasoning은 reflection과 retrieval을 통해 의사결정을 개선하고, acting은 tools/robots/assistants를 통해 실제 세계에 작용하며, interacting은 multi-agent 협업과 사회적 시뮬레이션을 가능하게 한다. 저자들은 이 전체 구조가 단지 현재 응용을 위한 기술 묶음이 아니라, 앞으로 LLM이 더 학습할 수 있게 만드는 새로운 데이터 생성 메커니즘이 될 수 있다고 본다.

실무적으로는 의료, 물류, 금융, 과학 연구 자동화가 중요한 응용 분야로 제시된다. 연구적으로는 self-reflection, role play, emergent norms, safety, training data generation이 핵심 미래 과제로 제안된다. 종합하면 이 survey는 agentic LLM을 “LLM + tools” 수준이 아니라, **추론·행동·상호작용을 결합한 차세대 학습 시스템**으로 이해해야 한다는 강한 메시지를 준다.  
