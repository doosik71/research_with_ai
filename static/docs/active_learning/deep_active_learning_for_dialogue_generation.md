# Deep Active Learning for Dialogue Generation

첨부된 ar5iv HTML 원문을 바탕으로 정리한 상세 분석 보고서다. 이 논문은 open-domain dialogue generation에서 자주 발생하는 **짧고 dull한 응답**, **일관성 부족**, **사람 취향을 반영하지 못하는 문제**를 해결하기 위해, 전통적인 hand-crafted reward 기반 reinforcement learning 대신 **online deep active learning**을 도입한다. 핵심은 Seq2Seq 대화 모델이 먼저 오프라인 supervised learning으로 기본 언어 능력을 익힌 뒤, 실제 사용자와의 상호작용 중에 여러 후보 응답을 제시하고, 사용자가 그중 최선의 답을 고르거나 직접 수정한 답을 주면 이를 즉시 학습에 반영하는 구조다. 저자들은 이 방식이 사람이 정의한 보상함수보다 더 자연스럽게 coherence, relevance, interestingness를 끌어낼 수 있다고 주장한다.  

## 1. Paper Overview

이 논문이 풀고자 하는 문제는 당시 neural conversational agent의 대표적 한계였다. Seq2Seq 기반 dialogue model은 문법적으로는 그럴듯하지만, 실제 대화에서는 “I don’t know”, “No”, “Ok” 같은 짧고 무난한 답을 자주 생성한다. 즉, 문장 생성 모델은 돌아가지만 **대화로서의 재미와 상호작용성**이 부족하다. 저자들은 이 문제를 단순 supervised learning만으로는 해결하기 어렵다고 보고, 또 open-domain dialogue에서는 task-oriented system처럼 명확한 reward를 정의하기도 어렵다고 본다. 그래서 사람이 직접 품질을 판단하는 **human-in-the-loop active learning**을 설계한다.  

왜 이 문제가 중요한가도 논문 안에서 분명하다. task-oriented dialogue는 성공/실패 기준이 비교적 명확하지만, open-domain dialogue는 “좋은 답변”이 무엇인지가 훨씬 미묘하다. coherence, relevance, engagement, interestingness를 모두 만족하는 handcrafted reward를 만드는 것은 어렵다. 저자들은 이런 복잡한 속성을 따로따로 설계하기보다, **사람의 선택 그 자체를 보상 신호처럼 사용**하는 편이 더 자연스럽다고 본다.

## 2. Core Idea

이 논문의 핵심 아이디어는 크게 세 가지다.

첫째, 모델 자체는 복잡한 새 구조가 아니라 **Seq2Seq LSTM encoder-decoder**를 backbone으로 사용한다. 즉, 새로운 점은 거대한 구조 혁신보다 **학습 절차와 상호작용 방식**에 있다. 모델은 먼저 오프라인으로 generic dialogue data와 short-text conversation data를 순차 학습해 기본기를 쌓는다.

둘째, 온라인 단계에서 모델은 한 번에 하나의 답만 내지 않고, **hamming-diverse beam search**로 여러 개의 후보 응답을 만든다. 사용자는 그중 가장 좋은 응답을 번호 하나로 고르거나, 더 나은 답을 직접 입력할 수 있다. 이때 사용자의 선택은 사실상 **암묵적 reward + 정답 레이블** 역할을 한다. 즉, 저자들은 reinforcement learning처럼 보이지만, 실제 업데이트는 선택된 응답을 target으로 하는 supervised update에 가깝게 설계한다.

셋째, 이 업데이트를 **one-shot learning** 수준으로 빠르게 반영할 수 있도록 learning rate를 조절한다. 적절히 높은 learning rate를 쓰면, 사용자가 방금 선택해 준 응답이 같은 prompt에 대해 즉시 가장 가능성 높은 출력으로 올라오게 된다. 이 덕분에 모델은 사람과 대화하면서 점진적으로 personality, mood, style까지 흡수할 수 있다.  

## 3. Detailed Method Explanation

### 3.1 전체 구조

논문의 backbone은 **one encoder-decoder layer with 300 LSTM units each**인 Seq2Seq 모델이다. 학습은 두 단계로 나뉜다.

* **Offline supervised learning**
* **Online active learning**

즉, 처음부터 사람과 직접 상호작용하며 배우는 게 아니라, 먼저 대규모 대화 데이터로 기본 언어 능력을 익힌 뒤, 그 위에 사용자 피드백을 얹는 방식이다.

### 3.2 Offline Two-Phase Supervised Learning

오프라인 학습은 두 단계로 구성된다.

**Phase 1**에서는 Cornell Movie Dialogs Corpus를 사용한다. 이 데이터는 약 **300K message-response pair**로 구성되며, 각 쌍을 input-target sequence로 보고 **joint cross-entropy loss**로 학습한다. 이 단계의 목적은 모델이 언어의 syntax와 semantics를 폭넓게 익히게 하는 것이다.

**Phase 2**에서는 JabberWacky의 chatlog에서 추출한 약 **8K short-text conversation pair**로 fine-tuning한다. 저자들의 설명대로, movie dialogue만으로는 짧은 인터넷식 대화와 스타일 차이가 커서 open-domain chat에 바로 잘 대응하지 못한다. 그래서 소규모지만 짧은 대화에 특화된 데이터를 추가로 학습시켜 더 현실적인 baseline을 만든다. 이 논문은 바로 이 two-phase supervised learning이 active learning 이전에도 성능을 꽤 개선한다고 보여준다.

### 3.3 Online Active Learning 절차

온라인 active learning의 절차는 논문 Algorithm 1에 비교적 명확하게 제시된다.

1. 사용자가 현재 turn에서 메시지 $u_i$를 보낸다.
2. 모델은 입력에 대해 $K$개의 후보 응답 $c_{i,1}, \dots, c_{i,K}$를 생성한다.
3. 사용자는 이 중 가장 좋은 것을 선택하거나, 아예 새로운 답변을 제안한다.
4. 선택된 pair $(u_i, c^*_{i,j})$를 이용해 모델을 즉시 업데이트한다.
5. 다음 turn으로 넘어간다.

중요한 점은, 여기서 active learning이 보통의 “가장 uncertainty 높은 unlabeled sample을 고르기”와는 다르게 쓰인다는 점이다. 이 논문에서 active learning은 **모델이 여러 candidate response를 내고, 사람의 feedback을 통해 정답에 가까운 반응을 적극적으로 획득**하는 interactive learning 메커니즘이다. 즉, query 대상이 unlabeled input sample이 아니라 **candidate response set**이라고 이해하는 편이 더 정확하다.  

### 3.4 Hamming-Diverse Beam Search

후보 응답 생성에서 논문의 핵심 기술은 **Diverse Beam Search(DBS)**, 그중에서도 **hamming diversity metric**을 쓰는 부분이다. 일반 beam search는 비슷한 응답을 여러 개 내놓기 쉽다. 예를 들어 “I don’t know”와 “I really don’t know”처럼 표면만 조금 다른 후보들이 상위 beam을 차지할 수 있다. 저자들은 이를 피하기 위해 각 beam 사이의 dissimilarity를 목적함수에 추가하고, 특히 다른 beam에서 이미 선택된 단어를 반복 선택하는 것을 페널티 주는 **hamming-based diversity**를 사용한다.

논문에 따르면 각 turn에서 후보 수는 **$K=5$**를 사용한다. 저자들은 5개보다 적으면 좋은 후보를 놓칠 수 있고, 너무 많으면 사용자가 읽고 고르기 번거롭다고 설명한다. 따라서 이 논문의 사용자 인터페이스 설계까지 포함한 실용적 판단이 반영되어 있다.

### 3.5 One-shot Learning과 Learning Rate

사용자 피드백을 얼마나 빨리 모델에 반영할지는 Adam optimizer의 initial learning rate로 조절한다. 저자들은 learning rate가 충분히 크면 **one-shot learning**이 일어나, 사용자의 feedback이 곧바로 동일 prompt에 대한 가장 유력한 예측으로 바뀐다고 설명한다. 반대로 learning rate가 너무 작으면 여러 번의 “nudges”가 필요하다.  

이 부분은 이 논문의 성격을 잘 보여준다. 전통적 deep learning 논문처럼 장기 안정성만 보는 것이 아니라, 사람과 실시간 상호작용하면서 **즉각적인 적응성**을 얼마나 확보할 것인가를 중요한 설계 변수로 본다.

### 3.6 학습이 사실상 하는 일

형식적으로는 online active learning이지만, 실제 update는 매우 간단하다. 사용자가 선택한 응답 또는 직접 수정한 응답을 **target sentence**로 삼고, 이에 대해 cross-entropy loss를 역전파한다. 즉, hand-crafted reward function도, policy gradient도, 복잡한 credit assignment도 없다. 저자들이 말하는 “reinforcement”는 엄밀한 RL reward라기보다 **human-preferred supervision signal**에 더 가깝다. 이 점이 논문의 장점이자 한계다. 구현은 단순하고 직관적이지만, 장기 대화 보상 최적화와는 다르다.

## 4. Experiments and Findings

### 4.1 평가 설정

저자들은 총 **200개의 test prompt**를 만들고, 여기에 대해 세 모델의 응답을 비교한다.

* **SL1**: 1단계 supervised learning만 수행
* **SL2**: 2단계 supervised learning까지 수행
* **SL2+oAL**: SL2 이후 online active learning까지 수행

이 응답들을 **5명의 human judge**가 평가하며, 평가지표는 다음 네 가지다.

* syntactical coherence
* relevance to the prompt
* interestingness
* user engagement

평가는 각 축마다 0 또는 1의 binary score를 부여하고 평균을 낸다.

### 4.2 주요 정량 결과

Figure 2a에 대한 저자 설명에 따르면, **SL2+oAL은 네 축 중 세 축에서 다른 모델보다 14–21% 더 좋은 성능**을 보인다. 이는 online active learning이 단순 문법성뿐 아니라 relevance, interestingness, engagement를 높이는 데 효과가 있음을 시사한다. 특히 논문은 사람들이 직접 선택한 피드백이 반영되므로, 모델이 semantically coherent하고 relevant한 답뿐 아니라 더 흥미로운 응답을 배우게 된다고 주장한다.

### 4.3 Learning Rate 실험

저자들은 one-shot learning이 가능하도록 여러 learning rate를 실험한다. 결과적으로 **너무 큰 learning rate는 품질을 떨어뜨린다**고 보고한다. 이유는 새 데이터에 대해 파라미터 변화가 너무 커져, 모델이 이전에 배운 내용을 잊어버리는 instability가 생기기 때문이다. 논문은 **0.005**가 stability와 one-shot adaptation 사이의 적절한 균형이라고 말한다.

이 결과는 중요하다. 이 논문은 “빠르게 배운다”는 점을 강조하지만, 실제로는 무작정 빠른 적응이 좋은 게 아니라 **기존 능력 보존과 새 피드백 반영 사이의 균형**이 필요함을 보여준다.

### 4.4 Training Interaction 수에 따른 변화

또 다른 실험에서는 interaction 수를 바꿔 가며 품질 향상을 본다. 저자들은 모델이 인간과 계속 대화할수록 **서서히 개선**되며, 약 300 interaction 이후 curve가 plateau처럼 보여도 gradient는 여전히 작지만 0은 아니라고 설명한다. 즉, 개선은 느리지만 계속 축적된다. 저자들은 이를 “인간이 언어를 배우는 방식과 유사하다”고 해석한다.

### 4.5 정성 비교

Table 1의 예시를 보면, SL1은 종종 “No.” 같이 지나치게 짧거나 문맥과 안 맞는 응답을 내는 반면, SL2는 더 적절해지고, SL2+oAL은 더 재치 있거나 상호작용적인 답을 보인다. 예를 들어 “Do you have any kids?”에 대해 SL1은 “No.”, SL2는 “I have no!”처럼 어색한 답을 내지만, SL2+oAL은 “None that are really close to me.”처럼 상대적으로 자연스럽고 캐릭터가 느껴지는 답을 한다. 이런 정성 예시는 active learning이 단순 accuracy 향상보다 **대화 스타일의 질감**을 바꾸는 데 유효하다는 논문의 메시지를 강화한다.

### 4.6 Mood / Persona Customization

Table 2는 논문의 흥미로운 실험이다. 저자들은 SL2 모델 세 개를 각각 **cheerful**, **gloomy**, **rude/sarcastic** 스타일로 100 interaction씩 훈련시킨다. 그 결과 같은 입력에도 mood가 다르게 반영된 응답이 나온다. 예를 들어 “How do you feel?”에 cheerful 모델은 “Amazing, and you?”, gloomy 모델은 “I’m not in the mood.”, rude 모델은 “Buzz off.”라고 답한다. 저자들은 이를 통해 active learning이 단순 정답 교정이 아니라 **persona/style shaping**에도 사용될 수 있다고 주장한다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **reward engineering 없이 인간 피드백을 직접 학습 신호로 썼다**는 점이다. open-domain dialogue에서 좋은 보상함수를 만드는 건 매우 어렵다. 이 논문은 그 문제를 우회해, 사람이 “좋은 응답”을 직접 골라주는 방식으로 해결한다. 설계가 단순하면서도 직관적이다.

두 번째 강점은 **diverse candidate generation + human selection** 조합이다. 후보를 한 개만 내면 사용자는 수동적으로 평가만 해야 하지만, 여러 개를 내고 고르게 하면 더 효율적으로 preference를 반영할 수 있다. hamming-diverse beam search는 이 상호작용 메커니즘을 성립시키는 중요한 기술적 장치다.

세 번째 강점은 **customized persona, mood, conversational style**까지 학습 가능하다는 실험적 시연이다. 이는 오늘날 personalization이나 alignment 관점에서도 흥미롭다. 단순히 “정답 같은 응답”이 아니라 **사용자가 원하는 말투와 정체성**을 반영할 수 있다는 점을 미리 보여줬다.

### 한계

하지만 한계도 분명하다. 첫째, 이 논문이 말하는 active learning은 일반적인 pool-based active learning과는 상당히 다르다. 실제로는 **human preference-based incremental supervised learning**에 더 가깝다. 따라서 “active learning”이라는 이름이 직관적이긴 해도, 고전적 active learning 연구와 동일선상에서 비교하긴 어렵다. 이 평가는 논문의 절차 설명 자체에 근거한 해석이다.

둘째, 사람 피드백에 강하게 의존한다. 실험은 흥미롭지만, 지속적으로 사용자에게 후보 5개를 읽고 고르게 하는 것은 실제 서비스에서 비용이 크다. 논문도 5개 이상은 사용자에게 번거롭다고 인정한다. 즉, scale-up 가능성에는 제약이 있다.

셋째, 평가는 상당 부분 **subjective human evaluation**에 의존한다. dialogue quality를 측정하기 위해 이는 타당하지만, 실험 재현성과 비교 가능성 측면에서는 제한이 있다. 또한 예시 응답 중 일부는 여전히 어색하거나 과도하게 캐릭터화되어 있어, 품질 향상이 곧바로 안정적 일반화로 이어진다고 보긴 어렵다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “최고의 dialogue model”을 만든 데 있다기보다, **사용자 피드백을 실시간으로 흡수하는 neural dialogue training loop**를 제안한 데 있다. 오늘날 RLHF, direct preference optimization, online adaptation 같은 흐름과 직접 같지는 않지만, “사람이 고른 응답을 즉시 모델에 반영해 conversational behavior를 shaping한다”는 아이디어는 상당히 선구적이다. 특히 open-domain dialogue에서 reward design의 어려움을 정면으로 문제 삼았다는 점에서 역사적 의미가 있다.

## 6. Conclusion

이 논문은 Seq2Seq dialogue model 위에 **offline two-phase supervised learning + online human-in-the-loop active learning**을 결합한 대화 생성 프레임워크를 제안한다. 모델은 먼저 일반 대화와 short-text 대화 데이터로 기본기를 익히고, 이후 실제 사용자와 상호작용하며 여러 후보 응답 중 사용자가 선호한 답을 학습한다. hamming-diverse beam search는 후보의 다양성을 보장하고, 적절한 learning rate는 one-shot adaptation을 가능하게 한다. 실험 결과, 이 구조는 coherence, relevance, interestingness, engagement를 개선하고, 나아가 cheerful, gloomy, rude 같은 style/persona까지 학습할 수 있음을 보였다.

연구적으로 이 논문은 open-domain dialogue에서 hand-crafted reward의 한계를 넘어서려는 초기 시도 중 하나로 가치가 있다. 실용적으로는 사람 피드백 비용, 평가의 주관성, 장기적 안정성 문제가 남지만, “좋은 대화는 사람이 가장 잘 안다”는 철학을 모델 학습 과정에 직접 넣었다는 점에서 의미 있는 작업이다.
