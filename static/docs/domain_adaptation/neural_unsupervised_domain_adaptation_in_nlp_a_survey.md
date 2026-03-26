# Neural Unsupervised Domain Adaptation in NLP - A Survey

* **저자**: Alan Ramponi, Barbara Plank
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2006.00632](https://arxiv.org/abs/2006.00632)

## 1. 논문 개요

이 논문은 NLP에서의 **neural unsupervised domain adaptation(UDA)** 연구를 체계적으로 정리한 survey이다. 핵심 문제의식은 매우 분명하다. 현대 NLP 모델, 특히 deep neural network와 pre-trained language model은 라벨이 충분한 환경에서는 뛰어난 성능을 내지만, 학습 데이터가 만들어진 환경과 실제 적용 환경이 다를 때 성능이 크게 하락한다. 논문은 이 현상을 **domain shift**로 설명하며, 특히 **타깃 도메인에 라벨이 전혀 없고 비라벨 데이터만 있는 상황**을 다루는 UDA가 실무적으로 더 중요하고 더 어렵다고 본다.

논문이 설정하는 기본 문제는 다음과 같다. source domain에서는 라벨된 데이터가 있고, target domain에서는 라벨이 없으며, 두 도메인의 입력 분포가 다르다. 즉, $P_S(X) \neq P_T(X)$ 이다. 이런 상황에서 source에서 학습한 모델이 target에서도 잘 작동하도록 만드는 것이 domain adaptation의 목표다. 논문은 이 문제를 단순한 성능 개선 기법의 문제가 아니라, 더 넓게는 **training distribution 밖으로의 generalization**, 즉 **out-of-distribution generalization**과 연결된 핵심 과제로 본다.

이 survey의 중요성은 세 가지다. 첫째, 비지도 도메인 적응의 neural 방법들을 체계적으로 정리한다. 둘째, domain이라는 개념 자체가 NLP에서 지나치게 느슨하게 사용되어 왔다는 점을 비판하고, 이를 더 넓은 **variety** 개념으로 재해석한다. 셋째, 기존 연구가 지나치게 sentiment analysis에 치우쳐 있고, 다양한 NLP 작업 전반으로 충분히 확장되지 못했음을 지적하면서 향후 연구 방향을 제시한다. 따라서 이 논문은 단순한 방법 정리뿐 아니라, NLP 일반화 문제를 보는 관점 자체를 재구성하려는 survey라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 UDA 방법을 단순히 알고리즘 나열로 정리하지 않고, **model-centric**, **data-centric**, **hybrid**라는 세 축으로 재구성하는 데 있다. 저자들은 기존 연구를 크게 다음과 같이 본다. 모델 구조나 loss를 바꾸어 domain-invariant representation을 학습하려는 접근은 model-centric이고, pseudo-labeling이나 data selection, pre-training처럼 데이터를 어떻게 활용하느냐에 초점을 맞춘 접근은 data-centric이다. 그리고 둘을 섞은 방법은 hybrid로 본다.

이 분류는 survey의 실질적 기여다. 기존의 많은 survey가 vision 쪽 domain adaptation, MT, 혹은 pre-neural NLP adaptation에 초점을 두었다면, 이 논문은 **neural UDA in NLP**에 한정해 최근 흐름을 한 프레임 안에 넣는다. 특히 pre-training의 부상을 단순한 “모델이 커졌다”는 현상이 아니라, domain adaptation의 한 형태로 읽어내는 점이 인상적이다. 즉, domain-adaptive pre-training(DAPT), task-adaptive pre-training(TAPT) 같은 최근 기법을 UDA의 연장선에서 이해한다.

또 하나의 중요한 직관은, 저자들이 “domain”이라는 용어 자체를 문제 삼는다는 점이다. 논문은 실제 NLP 데이터의 차이가 단순히 뉴스/트위터처럼 명시적인 도메인 이름으로 깔끔히 나뉘지 않는다고 본다. 장르, 주제, 문체, 사회언어학적 속성, 데이터 수집 방식, annotation bias 같은 다양한 잠재 요인이 얽혀 있으므로, 데이터는 사실상 고차원 **variety space** 위의 샘플이라는 것이다. 이 관점은 단순한 source-to-target 적응을 넘어서, 왜 robust generalization이 어려운지를 더 근본적으로 설명한다.

요약하면, 이 논문의 핵심 아이디어는 “UDA의 방법론을 분류하고 정리한다”는 수준을 넘어서, **NLP에서 adaptation을 바라보는 단위를 domain에서 variety로 확장하고, 이를 out-of-distribution generalization 문제와 연결한다**는 데 있다.

## 3. 상세 방법 설명

이 논문은 새로운 단일 알고리즘을 제안하는 방법 논문이 아니라 survey이므로, 하나의 통합 아키텍처나 하나의 손실 함수가 제시되지는 않는다. 대신 각 방법군의 학습 원리와 대표 기법을 체계적으로 설명한다. 따라서 이 섹션에서는 논문이 제시한 UDA 방법론의 구조를 정리한다.

### 3.1 문제 설정과 수식적 배경

논문은 먼저 domain adaptation의 형식적 정의를 제시한다. 도메인은 $ \mathcal{D} = {\mathcal{X}, P(X)} $로 정의된다. 여기서 $ \mathcal{X} $는 feature space이고, $P(X)$는 입력의 주변 분포다. 작업(task)은 $ \mathcal{T} = {\mathcal{Y}, P(Y|X)} $로 정의된다. 여기서 $ \mathcal{Y} $는 label space이고, $P(Y|X)$는 조건부분포다.

Domain adaptation에서는 source domain과 target domain이 같은 task를 공유하지만 입력 분포가 다르다. 즉, source와 target에 대해
$$
P_S(X) \neq P_T(X)
$$
이다. 논문은 이것을 주로 **covariate shift**로 설명한다. 반면 label shift, 즉
$$
P_S(Y) \neq P_T(Y)
$$
도 관련 문제이지만, 타깃 라벨이 없다는 UDA 설정에서는 이보다 입력 분포 차이에 초점을 둔다고 명시한다.

즉, 목적은 source의 라벨 데이터와 target의 비라벨 데이터를 이용해 함수 $f$를 학습하고, 이 $f$가 target domain에서 잘 일반화하도록 만드는 것이다.

### 3.2 전체 방법론 분류

논문은 UDA를 세 가지 범주로 나눈다.

첫째, **model-centric methods**는 feature space, loss function, regularization, model architecture를 직접 바꿔서 domain shift를 완화하려는 접근이다.

둘째, **data-centric methods**는 pseudo-labeling, data selection, pre-training처럼 데이터를 더 잘 활용하여 적응하는 방법이다.

셋째, **hybrid methods**는 위 두 계열을 혼합한다. 예를 들어 adversarial loss와 pseudo-labeling을 함께 쓰거나, pivot 방법과 contextual embeddings를 함께 쓰는 경우다.

이 분류는 단순한 taxonomy가 아니라, 각 접근이 어디에 개입하는지를 명확히 보여준다. model-centric은 “모델 안에서 domain gap을 줄이는 것”이고, data-centric은 “학습에 들어가는 신호를 더 유리하게 구성하는 것”이다.

### 3.3 Model-centric: Feature-centric methods

#### Pivot-based methods

초기 대표 기법은 **SCL(Structural Correspondence Learning)**, **SFA(Spectral Feature Alignment)**이다. 핵심 아이디어는 source와 target 양쪽에서 공통적으로 의미 있는 feature, 즉 **pivot**를 찾아 shared feature space를 만든다는 것이다. 예를 들어 sentiment classification에서는 “great”, “bad” 같은 단어가 여러 도메인에 걸쳐 공통으로 나타날 수 있고, 이를 매개로 도메인 간 feature correspondence를 학습한다.

Neural 계열에서는 이 pivot 아이디어를 autoencoder나 language model과 결합한다. 예를 들어 **AE-SCL**은 autoencoder를 사용해 non-pivot에서 pivot를 예측하는 latent representation을 학습하고, 그 표현을 downstream task에 활용한다. 하지만 이런 표현은 문맥 의존적이지 않다는 한계가 있다. 이를 보완하기 위해 **PBLM(Pivot-Based Language Modeling)**이 등장하며, LSTM 기반 language model이 pivot와 non-pivot의 존재를 예측하도록 하여 더 구조적인 표현을 학습한다. 이후 **TRL-PBLM**은 적은 pivot로 시작해 점차 더 많은 pivot를 사용하는 iterative refinement를 통해 정확도와 안정성을 높인다.

논문은 이러한 pivot 기반 neural UDA가 대부분 **sentiment classification에 집중**되어 있다고 지적한다. 즉, 개념적으로 흥미롭지만 범용성은 아직 충분히 검증되지 않았다.

#### Autoencoder-based methods

Autoencoder 기반 접근은 입력을 복원하는 reconstruction objective를 통해 domain-transferrable latent representation을 학습하려는 계열이다. 대표적으로 **SDA(Stacked Denoising Autoencoder)**는 입력에 noise를 넣고 이를 복원하도록 학습함으로써 보다 robust한 representation을 학습한다. 직관은, 노이즈가 섞인 입력에서도 원래 의미를 복원하게 하면 특정 도메인의 표면적 특징에 덜 과적합된 표현을 얻을 수 있다는 것이다.

이후 **MSDA(Marginalized SDA)**는 noise를 명시적으로 샘플링하는 대신 이를 수학적으로 marginalize하여 속도와 확장성을 개선했다. 또 structured dropout이나 adversarial regularization을 접목한 변형도 제안되었다.

다만 논문은 autoencoder 방식의 약점을 분명히 지적한다. 이 방법은 입력 복원에는 강하지만, 학습된 표현이 꼭 **언어적 구조나 task-specific 신호를 잘 반영한다는 보장은 없다**. 즉, reconstruction objective가 NLP 작업 성능과 직접 정렬되지 않을 수 있다.

### 3.4 Model-centric: Loss-centric methods

#### Domain adversarial methods

이 survey에서 가장 널리 쓰였다고 평가되는 방법은 **DANN(Domain-Adversarial Neural Network)**이다. DANN의 핵심은 두 목표를 동시에 최적화하는 것이다. 하나는 source labeled data에 대해 task predictor가 잘 작동하도록 하는 것이고, 다른 하나는 feature extractor가 source와 target을 domain classifier가 구분하지 못하도록 만드는 것이다.

개념적으로는 다음과 같은 구조다.

* feature extractor가 입력 텍스트를 representation으로 변환한다.
* task classifier는 이 representation으로 source task loss를 최소화한다.
* domain classifier는 이 representation이 source인지 target인지 맞히려 한다.
* gradient reversal layer를 사용해 feature extractor는 domain classifier를 혼란시키는 방향으로 업데이트된다.

이렇게 하면 representation이 **task에는 유용하지만 domain에는 비식별적(domain-invariant)**이 되도록 유도된다.

수식 자체를 논문이 자세히 전개하지는 않았지만, 원리는 다음처럼 이해할 수 있다. 전체 목적은 task loss $L_{\text{task}}$는 최소화하고, domain classification loss $L_{\text{domain}}$는 feature extractor 입장에서 최대화하는 것이다. 즉,
$$
\min_{\theta_f,\theta_y} L_{\text{task}}(\theta_f,\theta_y) - \lambda L_{\text{domain}}(\theta_f,\theta_d)
$$
와 비슷한 형태로 볼 수 있다. 여기서 $\theta_f$는 feature extractor, $\theta_y$는 task classifier, $\theta_d$는 domain classifier의 파라미터이며, gradient reversal layer는 사실상 domain loss의 부호를 뒤집어 feature extractor에 전달하는 역할을 한다. 이 식은 survey 원문에 완전한 수식 형태로 적혀 있지는 않지만, 논문이 설명하는 DANN의 학습 원리를 가장 직관적으로 풀어쓴 것이다.

논문은 DANN의 장점으로 **확장성**, **범용성**, **다양한 NLP 작업으로의 적용성**을 든다. 실제로 sentiment analysis뿐 아니라 POS tagging, parsing, NER 관련 작업, relation extraction, stance detection, duplicate question detection 등으로 확장되었다고 정리한다.

하지만 한계도 분명하다. DANN은 기본적으로 shared representation만 잘 학습하며, domain-specific information을 충분히 활용하지 못할 수 있다. 또한 domain classifier가 너무 강해지면 학습이 불안정해지고 gradient 문제가 생길 수 있다.

이를 완화하기 위한 대안으로 **Wasserstein distance** 기반 adversarial learning이 소개된다. 이 계열은 domain classifier로 이진 판별을 하는 대신 source와 target representation 분포 사이의 Wasserstein distance를 줄인다. 논문은 duplicate question detection 연구를 예로 들며, 성능은 비슷하되 Wasserstein 쪽이 더 안정적일 수 있다고 설명한다.

#### Domain Separation Networks

**DSN(Domain Separation Networks)**은 DANN의 shared-only 한계를 보완하려는 시도다. shared encoder 외에 source private encoder와 target private encoder를 별도로 두어, 공통 정보와 도메인 고유 정보를 분리한다. 직관적으로는 “공통으로 옮겨야 하는 정보”와 “각 도메인에만 속하는 정보”를 따로 보존하는 셈이다.

하지만 논문은 DSN도 완전한 해결책은 아니라고 본다. private representation이 주로 decoder 재구성에만 사용되고, 최종 classifier는 여전히 shared representation 위에서 학습되는 경우가 많기 때문이다. NLP에서는 이를 변형한 **GSN(Genre Separation Network)**이 relation extraction에 적용되었다고 정리한다.

#### Reweighting methods

이 계열은 도메인 차이를 representation 수준에서 숨기기보다, source의 각 instance가 target과 얼마나 비슷한지에 따라 가중치를 주는 방법이다. 대표적으로 **MMD(Maximum Mean Discrepancy)**, **KMM(Kernel Mean Matching)** 등이 언급된다. KMM은 reproducing kernel Hilbert space에서 source와 target의 평균이 가까워지도록 source instance의 weight를 조정한다.

직관적으로는 “source 데이터 전체를 똑같이 믿지 말고, target과 닮은 source 샘플을 더 많이 반영하자”는 생각이다. 논문은 이 방법이 NLP에서는 오래전부터 제안되었지만 neural setting에서의 강력한 효과는 아직 충분히 입증되지 않았다고 본다.

### 3.5 Data-centric methods

#### Pseudo-labeling

Pseudo-labeling은 target unlabeled data에 현재 모델이 예측한 label을 붙여서, 그것을 마치 gold label처럼 다시 학습에 사용하는 방식이다. self-training, co-training, tri-training, temporal ensembling 등이 여기에 포함된다.

핵심 직관은 단순하다. 처음에는 source labeled data로 모델을 만든 뒤, 그 모델이 target의 일부 샘플에 대해 충분히 신뢰할 수 있는 예측을 하면 이를 추가 감독 신호로 사용한다. 이렇게 하면 target 분포에 더 가까운 데이터를 이용해 decision boundary를 보정할 수 있다.

논문은 특히 **tri-training**과 같은 고전적 기법이 neural 시대에도 강한 baseline임을 강조한다. 또한 contextualized representation과 결합한 self-training, adaptive ensembling 같은 최근 변형도 소개한다. 중요한 점은 pseudo-labeling이 매우 실용적이지만, 잘못된 pseudo label이 누적되면 error reinforcement가 일어날 수 있다는 위험을 항상 내포한다는 점이다. 이 한계는 survey 본문에서 직접 길게 전개되지는 않지만, bootstrapping 계열 방법의 일반적인 성격상 암묵적으로 드러난다.

#### Data selection

Data selection은 방대한 source 혹은 pre-training corpus 중에서 target에 더 유사한 데이터를 골라 쓰는 접근이다. perplexity, Jensen-Shannon divergence, cosine similarity, distance metric 등이 기준으로 사용된다.

이 접근의 장점은 복잡한 모델 수정보다 **어떤 데이터를 학습시키느냐**에 직접 개입한다는 점이다. 특히 pre-training 시대에는 “더 큰 데이터가 항상 좋은가”보다 “타깃과 더 관련된 데이터를 쓰는 것이 중요한가”가 핵심 질문이 되는데, 논문은 여기에 대해 상당히 긍정적이다. 즉, target과 가까운 unlabeled data를 고르는 일 자체가 adaptation의 중요한 축이라고 본다.

#### Pre-training and adaptive pre-training

논문에서 가장 현대적인 비중을 차지하는 부분이다. 저자들은 pre-training을 세밀하게 나눈다.

첫째는 일반적인 **pre-training**이다. 예를 들어 BERT처럼 대규모 일반 코퍼스에서 self-supervised objective로 encoder를 학습한 뒤, downstream task에 fine-tuning하는 방식이다.

둘째는 **adaptive pre-training**이다. 이것은 다시 두 가지로 나뉜다.

하나는 **multi-phase pre-training**으로, 일반 코퍼스에서 pre-train한 뒤 domain-specific unlabeled corpus나 task-specific unlabeled corpus로 추가 pre-training하는 방식이다. 논문은 이를 broad-domain $\succ$ domain-specific $\succ$ task-specific의 점점 더 가까운 분포로 이동하는 과정으로 설명한다. 여기에 **DAPT(Domain-Adaptive Pre-Training)**, **TAPT(Task-Adaptive Pre-Training)**, BioBERT, AdaptaBERT 등이 포함된다.

다른 하나는 **auxiliary-task pre-training**으로, intermediate labeled task를 거치는 방식이다. 예를 들어 **STILTs**처럼 본 작업 이전에 관련된 중간 작업으로 supervised transfer를 수행해 표현을 더 안정적으로 만든다.

이 survey의 중요한 해석은, 이러한 adaptive pre-training 역시 UDA의 주요 흐름이라는 점이다. 즉, target unlabeled data에 masked language modeling을 수행하는 것 자체가 매우 강력한 adaptation이며, domain relevance가 큰 역할을 한다는 것이다. 논문은 이를 근거로 “도메인 혹은 variety는 large pretrained model 시대에도 여전히 중요하다”고 주장한다.

### 3.6 Hybrid methods

Hybrid 방법은 model-centric과 data-centric 신호를 함께 쓴다. 예를 들어 adversarial loss와 semi-supervised objective를 결합하거나, pivot 기반 접근과 pseudo-labeling을 결합하거나, contextualized embedding과 pivot 정보를 결합하는 식이다. 또한 multi-task learning과 pseudo-labeling을 결합한 **multi-task tri-training**, consistency loss와 temporal curriculum을 결합한 **adaptive ensembling**도 포함된다.

이 계열의 의미는 단순하다. 실제 adaptation 문제는 단일 원리만으로 풀기 어렵기 때문에, representation alignment와 label propagation, auxiliary task, self-ensembling 등을 함께 써야 한다는 것이다. 논문은 hybrid 연구가 앞으로 더 늘어날 가능성이 크다고 시사한다.

## 4. 실험 및 결과

이 논문은 survey이므로 단일 실험을 새로 수행하지 않는다. 따라서 일반적인 실험 논문처럼 하나의 dataset, metric, baseline, score table을 제시하지 않는다. 대신 기존 UDA 연구를 **작업별, 방법별**로 정리한 Table 1이 사실상 핵심 “결과 요약” 역할을 한다.

가장 중요한 실증적 관찰은 다음과 같다.

첫째, **sentiment analysis에 연구가 과도하게 집중되어 있다**. Table 1을 보면 neural pivot-based methods, SDA/MSDA 계열, DANN 계열, adaptive pre-training, 일부 hybrid 기법까지 상당수가 sentiment analysis에서 검증되었다. 논문은 이를 column bias라고 해석한다.

둘째, **여러 작업에 걸친 폭넓은 비교가 부족하다**. 어떤 방법은 한 작업에서만 시험되었고, 같은 방법군끼리도 동일 조건에서 공정 비교한 경우가 많지 않다. 논문은 이를 row sparsity라고 부른다. 즉, 특정 방법이 실제로 일반적인 UDA 원리로 강한지, 아니면 특정 작업에만 유리한지 판단하기 어렵다.

셋째, **DANN 계열은 가장 넓게 활용된 neural UDA 방법**이다. survey에 따르면 sentiment classification뿐 아니라 language identification, natural language inference, POS tagging, dependency parsing, relation extraction, machine reading comprehension, stance detection, duplicate question detection 등으로 적용 범위가 확장되었다. 이 점에서 DANN은 범용성 측면의 대표 기준선으로 기능한다.

넷째, **pseudo-labeling과 tri-training 같은 고전 기법이 neural 시대에도 강한 baseline**임이 재확인된다. 논문은 최근 연구가 오히려 이런 전통적 기법을 다시 불러와 contextualized representation과 결합하고 있다고 평가한다. 이는 “새로운 neural trick이 항상 고전 기법보다 우월하지는 않다”는 중요한 메시지다.

다섯째, **adaptive pre-training은 매우 유망한 흐름**으로 묘사된다. Han and Eisenstein, Gururangan et al., Li et al.의 결과를 바탕으로 저자들은 domain-relevant unlabeled data를 이용한 추가 pre-training이 high-resource와 low-resource 조건 모두에서 효과적이라고 정리한다. 즉, 큰 사전학습 모델이 있다고 해서 domain 문제가 사라지는 것이 아니라, 오히려 어떤 데이터로 추가 적응하느냐가 중요해진다.

여섯째, 논문은 adversarial learning의 성능만이 아니라 **학습 안정성**도 중요한 평가 기준으로 다룬다. 예를 들어 Wasserstein 방법은 DANN과 비슷한 성능을 내면서 더 안정적일 수 있다고 소개된다. 또한 fine-tuning 자체의 brittleness, 즉 seed와 data order에 따라 결과가 크게 흔들리는 문제도 적응 연구에서 중요한 현실적 변수로 지적한다.

따라서 이 survey에서 “실험 결과”의 핵심은 숫자 하나가 아니라, **어떤 방법이 어떤 작업에 편중되어 검증되었는지**, **무엇이 범용적으로 강한 baseline인지**, **pre-training 시대에도 domain relevance가 남아 있는지**를 정리한 메타 수준의 결론이라고 보는 것이 정확하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 neural UDA in NLP라는 비교적 넓고 복잡한 주제를 단순한 역사 나열이 아니라, 명확한 분류 체계로 정리했다는 점이다. model-centric, data-centric, hybrid라는 구조는 연구자에게 “지금 내가 고치려는 것은 모델인가, 데이터인가, 아니면 둘 다인가”를 분명히 보게 한다. 특히 pre-training과 adaptive pre-training을 UDA의 핵심 흐름 안으로 포함시킨 것은 2020년 시점에서 매우 적절한 정리다.

또 다른 강점은 “domain” 개념을 비판적으로 재검토했다는 점이다. 많은 survey는 방법을 정리하는 데 그치지만, 이 논문은 왜 domain adaptation 자체가 어려운지를 데이터의 복합적 이질성, 잠재 요인, corpus bias 차원에서 설명한다. **variety space**라는 관점은 이후 robustness, fairness, OOD generalization, dataset documentation 논의와도 잘 연결된다.

또한 논문은 survey이면서도 단순히 긍정적으로 정리하지 않는다. sentiment analysis 편향, 단일 작업 중심 검증, benchmark의 부재, 학습 안정성 문제, X scarcity까지 현실적인 병목을 명확히 짚는다. 이런 점에서 문헌 정리와 비판적 해석의 균형이 좋다.

반면 한계도 있다. 첫째, survey 특성상 개별 방법의 수식적 세부사항이나 구현 차이를 깊게 파고들지는 않는다. 예를 들어 DANN, DSN, PBLM, DAPT 등은 원리 수준에서는 충분히 설명되지만, 실제 objective의 차이와 실패 사례를 아주 세밀하게 비교하는 수준은 아니다. 따라서 방법을 직접 구현하거나 재현하려는 독자에게는 원 논문들을 추가로 읽어야 한다.

둘째, 논문 자체가 지적하듯, 당시 문헌 자체가 특정 작업에 치우쳐 있기 때문에 survey도 그 편향을 완전히 벗어나기 어렵다. 즉, “NLP 전반의 UDA”를 다루지만 실제 evidence base는 sentiment analysis와 일부 sequence labeling, parsing 쪽에 더 두텁다.

셋째, survey는 variety라는 개념을 강하게 주장하지만, 이를 operationalize하는 구체적 평가 프레임워크까지 제시하지는 않는다. 다시 말해 variety가 중요하다는 철학적·개념적 주장은 설득력 있지만, 실제 benchmark design이나 model selection 차원에서 이를 어떻게 표준화할지는 여전히 열린 문제로 남는다.

넷째, 본문에 언급된 일부 방향, 예를 들어 instance reweighting의 neural setting 효과나 broader unlabeled distribution release의 필요성 등은 아직 “유망하다”는 수준이지, 강한 경험적 합의로 정리되지는 않는다. 논문도 이를 조심스럽게 다루며, 명확하지 않은 부분은 단정하지 않는다.

비판적으로 해석하면, 이 논문은 “UDA의 정답”을 제시하는 것이 아니라, **NLP 일반화 문제를 다시 묻는 survey**다. 그래서 장점은 넓은 시야에 있고, 단점은 실험적 결론의 날카로움이 개별 실험 논문보다 약할 수밖에 없다는 점이다. 그러나 survey의 목적을 생각하면 이는 자연스러운 한계다.

## 6. 결론

이 논문은 NLP에서의 neural unsupervised domain adaptation 연구를 model-centric, data-centric, hybrid 접근으로 체계화하고, 각 계열의 대표 방법과 적용 작업을 정리한 중요한 survey이다. 핵심 기여는 세 가지로 요약할 수 있다.

첫째, 비지도 도메인 적응의 대표 방법들을 한 프레임 안에서 정리했다. pivot-based methods, autoencoder methods, adversarial learning, reweighting, pseudo-labeling, data selection, adaptive pre-training, hybrid methods가 각각 어떤 원리로 작동하는지 큰 그림을 제공한다.

둘째, domain이라는 개념을 더 일반적인 **variety**로 재해석함으로써, NLP 데이터의 이질성을 단순한 도메인 이름 이상으로 이해해야 한다고 주장했다. 이는 단지 adaptation을 넘어 robustness와 out-of-distribution generalization 문제로 이어진다.

셋째, 기존 연구의 편향과 부족한 benchmark 구조를 비판하고, 다중 작업·다중 적응 설정, 더 풍부한 비라벨 데이터 공개, annotation divergence 연구, X scarcity 대응 같은 향후 과제를 제안했다.

실제 적용 측면에서 보면, 이 논문은 “타깃 라벨이 없을 때 무엇을 할 수 있는가”에 대한 매우 실용적인 지도를 제공한다. 작은 규모의 target unlabeled data가 있다면 adversarial learning이나 pseudo-labeling, adaptive pre-training을 고려할 수 있고, 더 넓게는 어떤 데이터가 target variety와 가까운지 판단하는 data selection 문제도 중요함을 시사한다. 향후 연구 측면에서는, 단일 도메인 적응보다 더 넓은 OOD generalization, unknown target robustness, scarce-data adaptation으로 확장되는 흐름의 출발점 역할을 한다.

결국 이 논문의 가장 중요한 메시지는 다음과 같다. **큰 모델이 있어도 domain shift 문제는 사라지지 않으며, NLP는 여전히 데이터의 variety를 이해하고 그 차이를 견디는 방향으로 발전해야 한다.**
