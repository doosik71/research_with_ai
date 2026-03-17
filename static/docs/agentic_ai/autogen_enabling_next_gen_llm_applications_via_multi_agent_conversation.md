# AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation

이 논문은 단순히 “여러 LLM agent를 같이 쓰면 좋다”는 수준을 넘어서, **멀티에이전트 LLM 애플리케이션을 체계적으로 만들기 위한 프레임워크**를 제안한다. 저자들은 이를 위해 **conversable agents**와 **conversation programming**이라는 두 핵심 개념을 제시하고, 수학 문제 풀이, RAG 기반 QA, 텍스트 환경 의사결정, 안전한 코드 생성, dynamic group chat, conversational chess 등 다양한 응용에서 이 추상화가 실제로 유용하다는 점을 보인다. AutoGen은 오픈소스 프레임워크이며, LLM·human·tool을 하나의 agent 추상 안에 넣고, 이들 간 메시지 교환을 통해 복잡한 워크플로를 표현한다는 점이 핵심이다.

## 1. Paper Overview

이 논문이 풀고자 하는 핵심 문제는 다음과 같다. LLM이 점점 더 강력해지면서 reasoning, tool usage, feedback incorporation이 가능한 “agent”를 만드는 일이 늘어나고 있지만, 실제 애플리케이션이 복잡해질수록 **하나의 agent만으로는 역할 분리, 검증, 도구 실행, 인간 개입, 안전성 관리**를 모두 다루기 어려워진다. 저자들은 자연스럽게 떠오르는 해법이 **multiple agents that cooperate**라고 본다. 실제로 prior work도 멀티에이전트가 divergent thinking, reasoning, factuality, validation에 도움이 될 가능성을 보였다고 소개한다. 하지만 문제는 “멀티에이전트를 어떻게 일반적으로 설계하고 프로그래밍할 것인가”다.

저자들은 이 문제를 두 개의 구체적 질문으로 쪼갠다. 첫째, **재사용 가능하고 커스터마이즈 가능하며 협업에 적합한 개별 agent를 어떻게 설계할 것인가**. 둘째, **다양한 conversation pattern을 수용하는 통일된 인터페이스를 어떻게 만들 것인가**다. 이 논문은 이 두 질문에 대한 시스템적 답을 제시하는 프레임워크 논문이다. 즉, 어떤 특정 task용 알고리즘을 제안한다기보다, **next-gen LLM applications를 만들기 위한 소프트웨어/상호작용 추상화**를 제안하는 논문이라고 보는 것이 정확하다.

왜 이 문제가 중요한가도 분명하다. 실제 LLM 애플리케이션은 단순 Q&A를 넘어, 코드 작성 후 실행, 에러 수정, 안전성 검사, 정보 검색, 사용자 피드백 수집, 온라인 환경 상호작용 등을 포함한다. 이런 복잡한 워크플로를 prompt 한 장으로 때우기보다, 여러 역할을 가진 agent들의 대화로 나누면 더 **모듈화되고, 유지보수 가능하며, 사람 개입도 자연스럽게 삽입**할 수 있다는 것이 논문의 문제의식이다.

## 2. Core Idea

이 논문의 핵심 아이디어는 크게 두 축이다.

첫째는 **conversable agents**다. AutoGen에서 agent는 단순 함수나 prompt template이 아니라, **메시지를 보내고 받고, 내부 컨텍스트를 유지하며, 특정 capability를 가진 대화 가능한 개체**다. 이 capability는 LLM, human input, tool execution, 혹은 그 조합일 수 있다. 즉, agent를 “누가 답하는가”가 아니라 “어떤 능력을 가지고 어떤 역할로 대화에 참여하는가”로 정의한다. 예를 들어 assistant agent는 LLM 중심, user proxy agent는 human input 및 code execution 중심으로 설정할 수 있다.

둘째는 **conversation programming**이다. 저자들은 복잡한 LLM 워크플로를 step-by-step imperative pipeline으로 보는 대신, **agent 간 conversation으로 모델링**한다. 여기서 중요한 개념이 두 개 나오는데, 하나는 각 agent가 reply를 만들기 위해 수행하는 **computation**, 다른 하나는 어떤 순서와 조건에서 이런 computation이 일어나는지를 정하는 **control flow**다. AutoGen은 이 둘 모두를 conversation 중심으로 설계한다. 즉, 대화 메시지가 다음 계산을 유도하고, 그 계산 결과가 다시 다음 대화를 유도하는 방식이다.

이 구조의 novelty는 단순 멀티에이전트 사용이 아니라, **natural language와 programming language를 함께 써서 conversation flow를 통제**할 수 있게 한 데 있다. 어떤 제어는 system prompt 안에서 자연어로 줄 수 있고, 어떤 제어는 Python 코드나 custom reply function으로 구현할 수 있으며, 둘 사이를 유연하게 넘나들 수 있다. 저자들은 이를 통해 멀티에이전트 애플리케이션을 보다 범용적으로 만들 수 있다고 주장한다.

## 3. Detailed Method Explanation

### 3.1 Conversable Agent의 정의

AutoGen에서 conversable agent는 특정 역할을 가진 개체이며, 다른 agent와 메시지를 주고받으면서 task를 진행한다. agent는 받은 메시지와 보낸 메시지에 기반해 내부 context를 유지하고, LLM·도구·사람 입력 등의 capability를 가질 수 있다. 이 정의가 중요한 이유는, AutoGen이 단순히 “LLM 호출을 여러 번 한다”는 수준이 아니라, **각 구성요소를 message-driven agent abstraction으로 통일**하기 때문이다.

### 3.2 Agent capability: LLM, human, tool

논문은 agent capability를 세 종류로 정리한다.

* **LLM-backed agent**: role playing, implicit state inference, feedback incorporation, coding 같은 고급 능력을 수행한다.
* **Human-backed agent**: 특정 시점에 사람에게 입력을 요청할 수 있다.
* **Tool-backed agent**: code execution 또는 function execution을 수행할 수 있다.

핵심은 이들이 배타적이지 않다는 점이다. 하나의 agent가 이 capability를 조합해 가질 수 있다. 예를 들어 기본 `UserProxyAgent`는 사람 입력을 대신 받아오거나, LLM이 제안한 코드를 실행하는 역할을 할 수 있다. 이는 인간 개입과 자동화를 분리된 시스템이 아니라 **하나의 대화 프레임 안에서 조합**할 수 있게 해 준다.

### 3.3 Built-in agent abstraction

논문은 `ConversableAgent`를 가장 상위의 일반 추상으로 두고, 그 위에 `AssistantAgent`, `UserProxyAgent` 같은 사전 구성된 subclass를 얹는다. `AssistantAgent`는 보통 LLM-backed assistant 역할을, `UserProxyAgent`는 인간 프록시 또는 도구 실행 역할을 맡는다. 이 설계는 중요하다. 사용자는 framework를 처음 쓸 때는 built-in agent로 빠르게 시작하고, 필요하면 더 specialized한 custom agent를 만들 수 있다. 즉, **재사용성과 확장성을 동시에 겨냥**한 구조다.

### 3.4 Conversation Programming

논문이 가장 강하게 밀고 있는 개념이다. conversation programming은 멀티에이전트 시스템 개발을 다음 두 단계로 바꾼다.

1. **역할과 capability가 다른 conversable agent 집합을 정의**
2. **그들 사이의 interaction behavior를 conversation-centric computation과 control로 프로그래밍**

여기서 “conversation-centric”라는 말이 중요하다. 각 agent의 행위는 현재 대화 맥락에 의존하며, 메시지를 생성하거나 전달하는 방식으로 다음 계산을 유도한다. 저자들은 복잡한 워크플로를 agent action과 message passing으로 바라보면 훨씬 직관적으로 설계할 수 있다고 본다.

### 3.5 Auto-reply mechanism

AutoGen의 중요한 메커니즘 중 하나는 **agent auto-reply**다. 어떤 agent가 메시지를 받으면, termination condition이 만족되지 않는 한 자동으로 `generate_reply`를 호출해 응답하고 다시 상대에게 보내는 방식이다. 이 점이 흥미로운 이유는, 별도의 중앙 제어기(control plane)를 두지 않고도 **대화 흐름이 자연스럽게 유도**된다는 점이다. 저자들은 이를 decentralized, modular, unified workflow definition이라고 설명한다.

이 메커니즘 덕분에 개발자는 모든 step을 외부에서 orchestration하지 않아도 된다. 초기 conversation만 시작하면, 이후에는 각 agent의 reply function이 이어지며 전체 task가 진행된다. 이 구조는 later agentic framework들에서 흔해졌지만, 당시에는 꽤 중요한 abstraction이었다고 볼 수 있다.

### 3.6 제어 방식: 자연어 + 코드

논문은 control flow를 세 가지 방식으로 설명한다.

* **자연어 기반 제어**: system message나 prompt로 agent 행동을 유도
* **프로그래밍 언어 기반 제어**: Python으로 termination 조건, human input mode, tool execution logic, custom auto-reply 작성
* **자연어와 코드 사이의 전이**: LLM inference 안에서 control logic을 넣거나, LLM이 function call을 제안해 코드 제어로 넘어감

예를 들어 assistant agent는 “오류가 있으면 코드를 고쳐 다시 생성하라”, “작업이 끝나면 TERMINATE라고 답하라” 같은 자연어 제어를 받는다. 동시에 개발자는 Python으로 max auto replies, 종료 조건, 특정 custom reply behavior를 설정할 수 있다. 이 혼합 제어가 AutoGen의 실질적 강점이다.

### 3.7 Dynamic conversation과 GroupChatManager

정적인 2-agent chat만 지원하는 것이 아니라, AutoGen은 **dynamic conversation**도 지원한다. 논문은 두 가지 일반적 방법을 든다.

1. custom `generate_reply` 안에서 다른 agent와의 대화를 호출
2. function call을 통해 필요한 시점에 다른 agent를 부름

그리고 더 복잡한 경우를 위해 **`GroupChatManager`**를 제공한다. 이 manager는 다음 speaker를 동적으로 고르고, 그 응답을 다른 agent들에게 broadcast한다. 즉, round-robin이 아니라 **현재 대화 상태를 보고 다음 말을 할 agent를 선택**하는 방식이다. 이는 후반부 dynamic group chat 응용의 핵심 기반이다.

## 4. Experiments and Findings

논문은 framework paper답게 하나의 benchmark만 깊게 파는 대신, **여섯 개 응용 사례**를 통해 범용성을 보여준다. 저자들이 말하고 싶은 것은 “특정 task에서 최고 정확도”보다도, **같은 agent abstraction과 conversation programming paradigm이 서로 다른 문제군에서 작동한다**는 점이다.

### 4.1 A1: Math Problem Solving

수학 문제 풀이에서는 built-in agent 두 개만 재사용해 autonomous math solving system을 만들고, MATH benchmark에서 GPT-4 기반 여러 대안과 비교한다. 저자들에 따르면 AutoGen의 built-in agents만으로도 vanilla GPT-4, Multi-Agent Debate, LangChain ReAct, 상용 툴 조합 대비 매우 경쟁력 있거나 더 나은 성능을 보였다. 또한 `human_input_mode='ALWAYS'`만 설정하면 human-in-the-loop 모드로 쉽게 전환되며, 나아가 multiple human users가 대화에 참여하는 시나리오도 지원한다. 이 사례는 AutoGen이 단순 automation뿐 아니라 **mixed-initiative interaction**에도 적합하다는 점을 보여준다.

### 4.2 A2: Retrieval-Augmented QA / Code Generation

RAG 응용에서는 Retrieval-augmented User Proxy와 Retrieval-augmented Assistant 두 agent를 구성한다. 여기서 흥미로운 포인트는 **interactive retrieval**이다. 문맥에 답이 없으면 assistant가 그냥 “I don’t know”로 멈추는 게 아니라, “UPDATE CONTEXT”를 통해 더 많은 retrieval을 유도한다. 논문은 이 interactive retrieval이 실제로 성능에 의미 있는 차이를 만든다고 보고한다. 즉, RAG를 한 번의 retrieve-and-answer로 끝내지 않고, **conversation loop 속의 iterative retrieval**로 바꾸었다는 점이 중요하다. 또한 private or post-training codebase에 대한 code generation에도 활용 가능함을 보인다.

### 4.3 A3: ALFWorld

ALFWorld에서는 assistant agent가 계획을 제안하고 executor agent가 환경에서 실행하는 2-agent system을 만든다. 이것만으로도 ReAct와 유사 성능을 내지만, 저자들은 여기에 **grounding agent**를 추가해 commonsense knowledge를 적절한 순간에 제공한다. 예를 들어 “object를 examine하려면 먼저 찾아서 take해야 한다” 같은 규칙을 반복 오류가 감지될 때 삽입한다. 이로 인해 error loop를 줄이고, 논문은 평균적으로 **약 15% 성능 향상**이 있었다고 보고한다. 이 사례는 멀티에이전트의 이점이 단순 토론이 아니라, **특정 실패 모드를 보정하는 전용 역할의 삽입**에서 나온다는 점을 잘 보여준다.

### 4.4 A4: Multi-Agent Coding with OptiGuide

OptiGuide 기반 coding system은 Commander, Writer, Safeguard의 3-agent 구조다. Writer가 코드를 작성하고, Safeguard가 안전성을 검사하고, Commander가 전체 흐름과 메모리를 관리하며 실제 실행과 결과 해석을 조정한다. 논문은 이 구조로 **핵심 workflow code가 430줄 이상에서 100줄로 줄었다**고 보고하며, 사용자 시간은 약 3배 절감되고 interaction 수는 평균 3–5배 줄었다고 주장한다. 또한 safe/unsafe coding task 100개에 대한 ablation에서 multi-agent design이 unsafe code 식별 F1을 **GPT-4 기준 8%, GPT-3.5-turbo 기준 35%** 높였다고 한다. 이 사례는 AutoGen이 “잘 답하는 것”보다도 **안전한 orchestration과 개발 생산성**을 높이는 방향에서 강점이 있음을 보여준다.

### 4.5 A5: Dynamic Group Chat

Dynamic group chat에서는 `GroupChatManager`가 speaker를 동적으로 고르고 broadcast하는 구조를 쓴다. 흥미로운 점은 speaker selection prompt 설계다. 저자들은 단순 task-based prompt보다 **role-play style prompt**가 conversation context와 역할 정합성을 더 잘 반영해, 12개의 복잡한 수작업 태스크에서 더 높은 success rate와 더 적은 LLM call을 보였다고 말한다. 이 실험은 AutoGen이 고정된 back-and-forth 말고 **상황에 따라 누가 말할지 달라지는 협업 패턴**을 지원한다는 점을 보여준다.

### 4.6 A6: Conversational Chess

Conversational Chess는 human/AI player agent와 board agent를 둔 자연어 체스 시스템이다. 여기서 board agent는 move legality를 검증한다. 저자들은 board agent를 제거하고 “서로 합법적인 수를 두라”는 prompt만 남기는 ablation도 했는데, 그 경우 **불법 수가 게임을 깨뜨렸다**고 보고한다. 이 사례는 grounding과 external validator의 중요성을 보여준다. 즉, AutoGen의 장점은 LLM끼리 대화를 시키는 데만 있지 않고, **규칙 기반 외부 에이전트를 자연스럽게 끼워 넣어 시스템 무결성을 지키는 것**에도 있다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **좋은 추상화**다. 멀티에이전트 LLM 시스템을 ad hoc하게 짜는 대신, conversable agents와 conversation programming이라는 통일된 틀을 제시했다. 그 결과 human, LLM, tool을 동일한 인터페이스 아래 놓고, 복잡한 워크플로를 대화 흐름으로 표현할 수 있게 했다. 이는 이후 수많은 agent framework들이 채택한 설계 철학과도 닿아 있다.

두 번째 강점은 **범용성 시연**이다. 수학, RAG, embodied text world, coding, dynamic group chat, 게임까지 폭넓은 응용을 한 논문 안에서 보여 준다. 이는 “프레임워크가 정말 일반적인가”라는 질문에 꽤 설득력 있는 답을 준다. 특히 각 응용이 단순 데모가 아니라, interactive retrieval, grounding agent, safeguard agent처럼 **역할 분리를 통한 개선 포인트**를 실제로 보여 준다는 점이 좋다.

세 번째 강점은 **개발 생산성과 안전성**이다. OptiGuide 사례에서 보듯이, 멀티에이전트는 단순 성능 향상뿐 아니라 코드 구조 단순화, 사용자 상호작용 감소, unsafe code 검출 개선에 기여한다. 즉, 이 프레임워크는 모델 성능 논문이라기보다 **시스템 설계와 개발 효율성 논문**으로 읽을 가치가 크다.

### 한계

하지만 한계도 분명하다. 첫째, 논문은 framework paper라서 **엄격히 통제된 하나의 공정 비교**보다는 여러 사례 연구를 제시하는 형태다. 따라서 “AutoGen이 특정 알고리즘보다 언제나 낫다”를 엄밀하게 보였다고 하기는 어렵다. 응용마다 비교 대상, 환경, metric이 제각각이라 통합적인 결론을 내리기는 조심스럽다. 이 점은 프레임워크 논문이 흔히 가지는 한계다.

둘째, multi-agent 확장이 항상 이득을 보장하지는 않는다. agent 수가 늘어나면 coordination cost, token cost, conversation length, termination failure, safety risk도 같이 커진다. 논문도 discussion에서 **agent topology, conversation pattern, automation–human control balance, safety challenge**가 앞으로의 연구 과제라고 말한다. 즉, 멀티에이전트는 강력한 수단이지만, 공짜 성능 향상 장치가 아니라는 점을 저자들도 인정한다.

셋째, 실제 quality는 결국 underlying LLM의 품질에 의존한다. AutoGen은 강한 abstraction이지만, reasoning quality나 coding quality 자체를 새로 만드는 것은 아니다. 오히려 **적절한 역할 분리와 상호작용 구조를 통해 base model의 능력을 더 잘 끌어내는 orchestration layer**에 가깝다. 따라서 base model이 약하면 framework만으로 근본적 한계를 넘기 어렵다. 이 해석은 논문의 built-in agent 설명과 각 응용이 GPT-3.5/GPT-4 같은 기반 모델에 의존하는 구성에 근거한다.

### 비판적 해석

개인적으로 이 논문의 진짜 의미는 “multi-agent가 좋다”가 아니라, **LLM application engineering을 workflow programming에서 conversation programming으로 이동시켰다**는 데 있다. 즉, 이 논문은 이후의 많은 agent 시스템들이 당연하게 쓰게 된 개념들—assistant/user proxy, tool agent, manager, critic, safeguard, group chat, function-calling control—을 비교적 이른 시기에 하나의 일관된 프레임으로 묶었다. 논문의 실험이 모두 최종적이거나 완벽하다고 보긴 어렵지만, **소프트웨어 추상과 설계 패턴을 제안한 영향력**은 꽤 크다고 평가할 수 있다.

## 6. Conclusion

이 논문은 AutoGen을 통해 **멀티에이전트 LLM 애플리케이션 개발의 범용 프레임워크**를 제안한다. 핵심 기여는 두 가지다. 첫째, LLM·human·tool을 조합한 **conversable agent abstraction**. 둘째, agent 간 대화를 중심으로 computation과 control flow를 설계하는 **conversation programming paradigm**이다. 이를 바탕으로 math solving, retrieval-augmented QA/code generation, ALFWorld, OptiGuide, dynamic group chat, conversational chess 등 다양한 응용에서 성능, 생산성, 유연성, 안전성 측면의 이점을 보였다고 주장한다.

실무적으로는 복잡한 LLM 시스템을 더 **모듈화하고 재사용 가능하게 만들고 싶은 개발자**에게 의미가 크다. 연구적으로는 이후 멀티에이전트 agent framework 흐름의 중요한 출발점 중 하나로 볼 수 있다. 다만 앞으로는 어떤 agent topology와 conversation pattern이 최적인지, 사람 개입을 어디에 둘지, 비용과 안전성을 어떻게 균형 잡을지가 더 중요해질 것이다. AutoGen은 그 질문들을 여는 프레임워크 논문이라고 평가할 수 있다.
