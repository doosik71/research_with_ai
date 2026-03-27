# Domain Adaptation and Multi-Domain Adaptation for Neural Machine Translation: A Survey

* **저자**: Danielle Saunders
* **발표연도**: 2021
* **arXiv**: [https://arxiv.org/abs/2104.06951](https://arxiv.org/abs/2104.06951)

## 1. 논문 개요

이 논문은 Neural Machine Translation, 즉 NMT에서 **domain adaptation**과 **multi-domain adaptation**을 체계적으로 정리한 survey 논문이다. 핵심 문제의식은 매우 분명하다. 현대 NMT는 대규모 병렬 데이터가 충분할 때 매우 강력한 성능을 보이지만, 학습 때 보지 못한 새로운 domain으로 들어가면 번역 품질이 급격히 떨어진다. 예를 들어 뉴스 데이터로 학습한 모델은 biomedical 텍스트나 특정 고객의 기술 문서, 혹은 새롭게 등장한 사회적 이슈를 반영한 문장들에서 약해질 수 있다. 논문은 이 문제를 단순히 “새 데이터에 맞춰 조금 더 학습하면 된다”는 수준으로 보지 않는다. 실제로 fine-tuning은 강력한 기본선이지만, 충분한 in-domain bilingual data가 없을 수 있고, 데이터가 있어도 overfitting이나 catastrophic forgetting 때문에 기존 성능을 잃을 수 있다는 점을 강조한다.

저자는 domain adaptation 문제를 “기존 번역 시스템을 처음부터 다시 학습하지 않고, 특정 특성을 가진 문장들에 대해 성능을 개선하려는 상황”으로 넓게 정의한다. 여기서 domain은 단순히 데이터 출처(provenance)만이 아니라 topic, genre, style, formality 같은 다양한 언어적 속성을 포함한다. 이어서 multi-domain adaptation은 여기에 더해 **하나의 시스템이 여러 domain에서 동시에 잘 동작해야 하는 상황**으로 정의된다. 즉, 이 논문은 단일 target domain에 맞춘 적응뿐 아니라, 여러 domain을 함께 다루는 현실적 시나리오까지 포괄한다.

이 문제의 중요성은 학문적으로도 크고 실용적으로도 크다. 실제 서비스 환경에서는 원래의 학습 데이터 전체와 동일한 계산 자원을 늘 사용할 수 없고, 새로운 고객 도메인이나 새롭게 등장한 언어 사용 양식에 빠르게 대응해야 한다. 또한 도메인 특화 번역은 단순한 BLEU 개선을 넘어서 용어 일관성, 형식성 제어, 문맥 일치, 편향 완화 등 실제 사용성에 직결된다. 따라서 이 논문은 단순 리뷰를 넘어, NMT 적응 연구 전체를 설계 공간별로 분류하고 각 접근법의 장단점과 적용 시나리오를 정리하는 역할을 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 domain adaptation을 단일 기법으로 보지 않고, **NMT 시스템 개발 과정의 어느 단계에서 개입하느냐**에 따라 체계적으로 분류하는 데 있다. 저자는 adaptation 기법을 크게 네 부류로 나눈다. 첫째는 **data-centric adaptation**으로, 어떤 데이터를 선택하거나 생성해 적응에 사용할 것인지의 문제다. 둘째는 **architecture-centric adaptation**으로, domain label, domain-specific subnetwork, adapter 같은 구조적 수정을 통해 적응하는 방식이다. 셋째는 **training-scheme adaptation**으로, regularization, curriculum learning, instance weighting, minimum risk training, meta-learning 등 학습 절차 자체를 조정하는 방식이다. 넷째는 **inference-time adaptation**으로, 모델 파라미터를 바꾸지 않고 ensemble, rescoring, constrained decoding, terminology injection 등 추론 과정에서 domain specificity를 부여하는 방식이다.

이 분류의 장점은 매우 실용적이라는 점이다. 예를 들어 상용 번역 시스템 사용자처럼 모델 구조를 바꿀 수 없고 adaptation data만 조정할 수 있는 사람도 있고, 반대로 연구자처럼 데이터는 고정되어 있지만 아키텍처나 학습 알고리즘을 바꿀 수 있는 경우도 있다. 이 논문은 바로 이런 현실적 제약을 반영해, “어떤 상황에서 어떤 adaptation 방법이 가능한가”라는 관점으로 survey를 전개한다.

또 하나의 중요한 차별점은 **single-domain adaptation과 multi-domain adaptation을 명확히 구분**한다는 점이다. 기존 연구는 종종 특정 test domain이 알려져 있고 그 domain이 다른 domain과 명확히 분리되어 있다고 가정했다. 그러나 저자는 실제 언어 domain은 겹치고 흐릿하며, test 문장의 provenance를 모를 수도 있고, 문장이 여러 domain 특성을 동시에 가질 수도 있다고 지적한다. 따라서 domain label이 항상 완전한 답이 아니며, domain inference나 overlapping domain 처리, unseen domain에 대한 few-shot 또는 zero-shot 대응도 중요하다고 본다.

마지막으로 이 논문은 domain adaptation을 전통적인 topic/genre 문제에만 가두지 않는다. low-resource translation, gender bias mitigation, document-level translation 같은 별도의 MT 연구 문제들도 adaptation 관점으로 재해석할 수 있다고 주장한다. 이 확장은 이 survey의 중요한 학술적 기여다. 즉, adaptation은 단지 “뉴스에서 의학으로 옮겨가는 문제”가 아니라, **모델이 특정한 언어적 조건이나 요구사항에 맞게 동작을 바꾸는 넓은 프레임**이라는 것이 이 논문의 핵심 메시지다.

## 3. 상세 방법 설명

이 논문은 survey이므로 하나의 새로운 모델을 제안하지 않는다. 대신 NMT의 기본 구성과 adaptation 기법이 어디에 작용하는지를 설명하기 위해 Transformer 기반 번역 모델의 입력 표현, 학습 목적, 추론 절차를 정리하고, 각 적응 기법이 이 구조 위에 어떻게 얹히는지를 설명한다.

먼저 NMT는 source sentence $\mathbf{x}$와 target sentence $\mathbf{y}$를 입력으로 받아, 조건부 확률 $P(\mathbf{y}|\mathbf{x})$를 모델링한다. 일반적인 학습 목표는 Maximum Likelihood Estimation이며, 이는 cross-entropy loss 최소화와 동치다. 논문은 이를 다음과 같이 정리한다.

$$
\hat{\theta}=\operatorname_{argmax}_{\theta}\log P(\mathbf{y}|\mathbf{x};\theta)
$$

또는 토큰 단위의 cross-entropy 관점에서는

$$
\hat{\theta}=\operatorname_{argmin}_{\theta}\sum_{j=1}^{|y|}-\log P(y_j|\mathbf{y}_{1:j-1},\mathbf{x};\theta)
$$

가 된다. 즉, 정답 target sequence의 각 토큰이 높은 확률을 갖도록 파라미터 $\theta$를 업데이트한다.

논문이 강조하는 첫 번째 기본선은 **fine-tuning**이다. 이미 generic domain에서 사전학습된 모델이 있을 때, 작은 in-domain 병렬 데이터에 대해 같은 목적함수로 추가 학습을 계속하는 방식이다. 계산 비용이 작고 효과가 커서 사실상 adaptation의 기본선 역할을 한다. 하지만 fine-tuning은 세 가지 핵심 문제를 가진다고 정리한다. 첫째, in-domain data가 부족할 수 있다. 둘째, 새로운 domain에 적응하는 동안 기존 domain 성능이 급락하는 catastrophic forgetting이 발생할 수 있다. 셋째, adaptation set이 작거나 반복적이면 overfitting이 생겨 아주 유사한 문장에만 잘 맞고 조금만 달라져도 무너질 수 있다.

이러한 문제를 해결하기 위한 기법들을 저자는 다음처럼 구조화한다.

### 데이터 중심 방법

가장 먼저 **추가 자연 데이터 선택**이 있다. 이때 discrete lexical overlap 기반 방법과 continuous representation 기반 방법, 그리고 외부 모델을 이용한 relevance scoring 방법으로 나눈다. 예를 들어 n-gram overlap, TF-IDF, fuzzy matching, cross-entropy difference, sentence embedding similarity 등을 이용해 generic corpus에서 pseudo in-domain data를 추출할 수 있다. 핵심 직관은 “test domain과 비슷한 문장을 더 많이 보여주면 adaptation이 쉬워진다”는 것이다. 특히 용어 일관성이나 세부 lexical choice가 중요한 경우, test 문장과 유사한 문장을 선택하는 것이 큰 도움이 된다.

다음으로 **데이터 필터링**이 있다. 이미 in-domain이라고 여겨지는 데이터도 실제로는 noisy하거나 잘못 정렬되었을 수 있으므로, clean data만 남기도록 filtering하는 것이다. adaptation data가 작을수록 이런 정제가 더 중요하다고 논문은 지적한다.

그다음은 **monolingual data를 이용한 synthetic bilingual data 생성**이다. 가장 대표적인 것이 back-translation이다. target-side monolingual in-domain 문장을 모은 뒤, 역방향 모델로 synthetic source를 만들어 병렬 데이터를 구성한다. 반대로 forward translation도 가능하다. back-translation은 자연스러운 target 문장을 유지한다는 점에서 자주 사용되지만, synthetic text 자체가 translationese라는 별도 domain처럼 작동할 수 있어 주의가 필요하다고 설명한다. 저자는 tagged back-translation이 이 문제를 줄일 수 있다는 기존 연구도 소개한다.

또한 작은 adaptation set을 **noising**하거나 **simplification**해서 data augmentation하는 방식도 다룬다. 예를 들어 source 문장에 인위적 노이즈를 넣어 더 다양한 입력에 강하게 만들거나, target complexity를 조절할 수 있다. 마지막으로 lexicon이나 template 기반의 **fully synthetic adaptation data**도 설명한다. 특히 domain-specific terminology가 중요한 환경에서는 bilingual dictionary나 lexicon이 강력한 수단이 될 수 있다.

### 구조 중심 방법

구조 측면에서는 먼저 **domain label과 domain control**이 등장한다. 가장 간단한 방식은 source 또는 target에 domain tag를 붙이는 것이다. 예를 들어 `<bio>` 같은 tag를 source 앞에 붙여 “이 문장은 biomedical domain이다”라는 신호를 모델에 준다. 이 tag는 단순 token일 수도 있고, 별도의 domain embedding으로 각 단어 임베딩에 결합될 수도 있다. 이 방식은 여러 domain을 하나의 모델에서 다룰 때 특히 유용하다. 다만 domain이 겹치거나 fuzzy한 경우 잘못된 tag가 오히려 해가 될 수 있다는 점도 함께 논의한다.

다음은 **domain-specific subnetworks**다. 여기에는 domain-specific embedding, domain classifier, domain-specific softmax bias, 혹은 domain마다 별도의 encoder/decoder/attention 모듈을 두는 방식이 포함된다. 직관적으로는 모델 일부를 domain마다 특화시키는 것이다. 그러나 domain 수가 많아질수록 파라미터와 계산 비용이 커지고, domain을 이산적이고 분리된 실체로 가정하게 되는 한계가 있다.

이 논문이 특히 비중 있게 다루는 것은 **adapter layers**다. adapter는 기존 Transformer 층 사이에 넣는 작은 추가 모듈로, 사전학습 파라미터는 고정하고 adapter만 학습한다. 이 방식은 forgetting이 구조적으로 적고, 저장해야 할 파라미터가 적어 매우 실용적이다. 논문은 adapter가 noisy domain처럼 full fine-tuning이 쉽게 overfit되는 상황에서 오히려 더 잘 작동할 수 있다고 정리한다. 반면 adapter의 크기와 tuning step 수는 신중히 선택해야 한다고 말한다.

### 학습 절차 중심 방법

학습 절차 쪽에서 가장 핵심은 **regularization**이다. catastrophic forgetting을 줄이기 위해 사전학습 파라미터에서 너무 멀어지지 않게 제약을 거는 방식이다. 논문은 일반적인 regularized loss를 다음과 같이 제시한다.

$$
\hat{\theta}=\operatorname_{argmin}_{\theta}\left[L_{CE}(\mathbf{x},\mathbf{y};\theta)+\Lambda \sum_j F_j(\theta_j-\theta_j^{PT})^2\right]
$$

여기서 $\theta_j^{PT}$는 pre-training 당시의 파라미터이고, $\Lambda$는 이전 domain을 얼마나 보존할지 결정하는 가중치다. $F_j=1$이면 단순 L2 regularization이 되고, $F_j$를 Fisher information으로 두면 **Elastic Weight Consolidation, EWC**가 된다. EWC의 직관은 “이전 domain에서 중요한 파라미터는 덜 바꾸고, 덜 중요한 파라미터를 더 바꾸자”는 것이다. 논문은 EWC가 단일 domain 적응뿐 아니라 연속적인 multi-domain adaptation에서도 forgetting 완화에 효과적이었다고 소개한다.

또한 일부 파라미터만 학습하고 나머지는 **freeze**하는 방식도 설명한다. 이는 adaptation 자유도를 줄여 forgetting을 완화하지만, 새 domain 성능이 full fine-tuning보다 낮아질 수 있다. 반면 sparse parameter selection이나 factorized parameter adaptation처럼 적은 파라미터만 선택적으로 조정하면서도 성능 저하를 최소화하는 연구들도 소개된다.

이 외에도 **knowledge distillation**, **curriculum learning**, **instance weighting**을 중요한 adaptation 학습 절차로 설명한다. curriculum learning은 generic에서 in-domain으로 점진적으로 샘플 분포를 이동시키는 전략으로 볼 수 있고, mixed fine-tuning은 generic data 일부를 adaptation 과정에 계속 섞어 forgetting과 overfitting을 완화한다. instance weighting은 문장별 domain relevance에 따라 loss 가중치를 다르게 주는 방식이다.

비-MLE 학습으로는 **Minimum Risk Training, MRT**와 **meta-learning**을 다룬다. MRT는 번역 샘플 집합에 대해 BLEU 같은 평가 척도 기반 risk를 최소화하는 방식이다. 논문은 적응 데이터가 작고 noisy할 때 MLE보다 exposure bias를 줄이는 데 도움될 수 있다고 본다. MRT 목표는 대략 다음과 같이 주어진다.

$$
\hat{\theta}=\operatorname_{argmin}_{\theta}\sum_{s=1}^{S}\sum_{n=1}^{N}\Delta(\mathbf{y}_n^{(s)},\mathbf{y}^{(s)*})
\frac{P(\mathbf{y}_n^{(s)}|\mathbf{x}^{(s)};\theta)^{\alpha}}
{\sum_{n'}P(\mathbf{y}_{n'}^{(s)}|\mathbf{x}^{(s)};\theta)^{\alpha}}
$$

여기서 $\Delta(\cdot)$는 보통 $1-\text{sBLEU}$ 같은 cost다. 즉, reference와 더 가까운 샘플이 더 낮은 risk를 갖도록 학습한다.

meta-learning은 “새 domain에 몇 step만에 빠르게 적응할 수 있는 초기 파라미터”를 학습하는 접근이다. 논문은 특히 adapter와 결합한 meta-learning이 few-shot adaptation에 유망하다고 정리하지만, 전체 Transformer 파라미터를 meta-learn하면 원래 domain을 잃어버릴 수 있다고 설명한다.

### 추론 중심 방법

추론 단계에서는 파라미터를 바꾸지 않고 domain 적응을 달성하는 여러 방법을 다룬다. 먼저 **multi-domain ensembling**이 있다. 여러 domain-specific model을 준비한 뒤, source 문장이나 partial hypothesis에 따라 동적으로 ensemble weight를 바꾸는 것이다. 논문은 Bayesian interpolation 기반의 domain-adaptive ensembling을 소개하며, 각 time step에서 어떤 domain model이 더 적합한지 posterior를 통해 가중치를 조정할 수 있다고 설명한다.

그다음은 **retrieval-based ensembling**, 예를 들어 $k$NN-MT 같은 방식이다. decoder state와 in-domain datastore의 nearest neighbor target token 분포를 결합해, domain-specific context를 비모수적으로 끌어오는 방식이다.

또한 **rescoring**과 **constrained decoding**도 중요하게 다룬다. 먼저 generic model이 후보 번역을 만들고, 이후 domain-specific language model이나 phrase lattice, terminology constraint를 이용해 다시 점수를 매기거나 출력을 제한한다. 마지막으로 **terminology tagging**과 **priming** 같은 pre/post-processing 방식도 소개한다. 즉, source에 용어 태그나 유사 문장 예시를 넣어 모델이 원하는 terminology나 phrasing을 따르도록 만드는 것이다. 이는 black-box commercial MT에도 적용 가능할 수 있다는 점에서 실용적이다.

## 4. 실험 및 결과

이 논문은 survey이므로 하나의 통일된 실험 설정을 제안하고 거기서 새 SOTA를 보고하는 논문은 아니다. 따라서 일반적인 실험 섹션처럼 하나의 데이터셋, 하나의 baseline, 하나의 metric 표를 중심으로 읽으면 안 된다. 대신 저자는 각 adaptation 계열의 연구 결과를 종합해, 어떤 조건에서 어떤 방법이 유리한지 서술적으로 정리한다. 이 점은 이 논문을 이해할 때 중요하다. 즉, 이 논문의 “결과”는 단일 모델 성능 수치가 아니라, **문헌 전체를 종합한 실천적 결론**이다.

저자가 반복적으로 강조하는 실험적 관찰은 다음과 같다. 첫째, **simple fine-tuning은 여전히 매우 강한 baseline**이다. 적은 계산으로 큰 in-domain 성능 향상을 얻을 수 있으며, 많은 제안 방법들은 결국 simple fine-tuning보다 어떤 점에서 더 나은지로 평가된다. 둘째, **mixed fine-tuning, regularization, adapters** 같은 방식은 forgetting이나 overfitting을 줄이는 데 유리하다. 셋째, **back-translation과 monolingual data 활용**은 병렬 in-domain data가 부족할 때 특히 중요하다. 넷째, **tagging, terminology constraint, priming** 등은 정량 점수뿐 아니라 실제 용어 제어와 출력 controllability 측면에서 의미가 있다.

데이터 관련 결과에서는 n-gram overlap, fuzzy matching, sentence embedding, cross-entropy difference 등을 활용한 data selection이 in-domain terminology와 lexical choice를 향상시킨다는 점을 여러 선행연구가 보여준다고 정리한다. 단, 너무 미세한 per-sentence adaptation은 matching이 부정확할 경우 overfitting을 유발할 수 있어, 전체 test set 수준의 selection이 더 안정적일 수 있다고 언급한다.

synthetic data 쪽에서는 back-translation이 domain adaptation에도 강력하지만, synthetic data 비율이 높으면 translationese에 적응하는 착시가 생길 수 있다는 결과를 소개한다. 따라서 tagged back-translation이나 authentic data와의 교대 학습이 중요하다고 정리한다. forward translation은 별도 역방향 모델이 없어도 되는 장점이 있지만, synthetic target의 오류를 모델이 다시 학습할 위험이 있다. 인간 평가 관점에서는 back-translation 쪽이 더 자연스럽다고 보고한 연구도 인용한다.

구조적 방법에서는 domain tag와 domain feature가 simple fine-tuning보다 in-domain 성능을 개선하는 경우가 있고, 특히 domain feature embedding이 단순 discrete tag보다 조금 더 낫다는 결과들을 소개한다. 그러나 domain이 서로 겹칠 때는 tag가 오히려 일반 domain과의 공유 이점을 방해할 수 있다는 결과도 함께 제시한다. adapter에 대해서는 parameter efficiency가 매우 뛰어나고 forgetting이 거의 없으며, noisy domain에서는 full fine-tuning보다 더 나은 결과가 가능하다고 정리한다.

학습 절차에서는 EWC와 mixed fine-tuning이 repeatedly strong한 해결책으로 제시된다. EWC는 generic-to-specific adaptation뿐 아니라 sequential multi-domain adaptation에서도 forgetting을 줄였고, mixed fine-tuning은 generic data replay 효과로 강한 성능을 보인다. instance weighting과 curriculum learning도 domain relevance를 정교하게 반영하는 수단으로 효과적이지만, 실제 효용은 relevance estimation 품질에 크게 좌우된다. MRT는 adaptation data가 noisy하거나 domain mismatch가 심할 때 exposure bias와 metric mismatch를 줄이는 방향으로 도움이 된다고 정리한다. meta-learning은 few-shot adaptation에 유망하지만, meta-training domain 구성이 부적절하면 오히려 overfit될 수 있다.

추론 단계의 결과로는 multi-domain ensemble이 unknown test domain에서 상당히 합리적인 기본선이 될 수 있고, 특히 domain adaptive weighting이 정적 uniform ensemble보다 낫다고 소개한다. retrieval-based inference는 별도의 NMT 모델 추가 없이도 in-domain token prior를 활용해 큰 성능 향상을 낼 수 있는 최근 흐름으로 정리된다. terminology injection과 constrained decoding은 BLEU 개선뿐 아니라 실제로 사용자가 원하는 용어를 반드시 넣어야 하는 번역 환경에서 특히 중요하다.

논문 말미의 case study도 결과 해석의 일부다. 저자는 low-resource translation, gender bias mitigation, document-level translation에서 adaptation framing이 실제로 유효하게 쓰였음을 보여준다. 예를 들어 gender bias 문제는 남성/여성 관련 번역 패턴을 일종의 domain처럼 보고 adaptation, tag, prompting, constrained decoding을 적용할 수 있다. document-level translation 역시 문서 자체를 하나의 작은 domain처럼 보고 lexicon adaptation, document tag, context-aware module adaptation 등을 사용할 수 있다. 즉, 논문의 실험적 메시지는 “domain adaptation 기술은 특정 topic adaptation을 넘어 더 넓은 MT 문제들에 재사용 가능하다”는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **taxonomy가 명확하고 실용적**이라는 점이다. 많은 survey가 방법을 나열하는 데 그치지만, 이 논문은 adaptation 개입 지점을 데이터, 구조, 학습, 추론으로 나눠 독자가 자신의 현실 제약에 맞게 접근법을 고를 수 있게 한다. 이는 연구자뿐 아니라 실제 시스템 개발자에게도 큰 가치가 있다.

둘째, 이 논문은 **single-domain과 multi-domain을 구분하면서도 둘의 연결관계**를 잘 설명한다. 단지 여러 domain label을 붙이는 수준이 아니라, domain overlap, unseen domain, unknown domain at inference, continual learning 같은 현실적인 문제를 함께 다룬다. 이 때문에 survey가 단순 정리 이상으로, adaptation 문제 설정 자체를 재정의하는 역할을 한다.

셋째, 저자는 **fine-tuning의 강함과 한계를 동시에 균형 있게 다룬다**. 새로운 기법을 과도하게 미화하지 않고, 왜 fine-tuning이 여전히 강력한 baseline인지, 그리고 forgetting과 overfitting이 어떤 식으로 나타나는지 구체적으로 설명한다. 이 점은 학술적으로 매우 중요하다. adaptation 연구는 작은 수치 개선을 내세우기 쉬운데, 이 논문은 그 배경 문제를 구조적으로 보여준다.

넷째, low-resource, gender bias, document-level MT를 adaptation 관점에서 해석한 부분은 매우 통찰적이다. 이는 adaptation을 단순한 topic transfer 문제가 아니라 **모델 동작을 원하는 언어 조건으로 조정하는 범용 프레임**으로 끌어올린다.

한편 한계도 있다. 가장 먼저, 이 논문은 survey이기 때문에 **하나의 통일된 실험 기준으로 방법들을 직접 비교하지 않는다**. 따라서 어떤 방법이 항상 우월한지 결론을 내리기는 어렵다. 이는 survey 논문의 본질적 한계이기도 하다. 서로 다른 논문들이 서로 다른 데이터셋, 언어쌍, 평가 지표, 실험 프로토콜을 사용하기 때문에, 독자는 비교 결과를 읽을 때 맥락 차이를 항상 염두에 두어야 한다.

둘째, 논문이 broad coverage를 추구하다 보니 **각 개별 기법의 수학적 세부사항이나 구현 난이도**는 깊게 들어가지 못하는 부분이 있다. 예를 들어 adapter 구조나 EWC 적용의 실제 구현 차이, retrieval datastore 설계 같은 세부 엔지니어링 요소는 논문 목적상 압축적으로만 소개된다.

셋째, 논문 시점의 한계도 있다. 이 survey는 2021년 arXiv 논문으로, 당시까지의 NMT adaptation 문헌을 정리한다. 따라서 이후 크게 확장된 large language model 기반 translation adaptation, instruction tuning, parameter-efficient fine-tuning의 최신 계열까지 포함하지는 않는다. 물론 이것은 논문의 결함이라기보다 시점상의 자연스러운 제약이다.

비판적으로 보면, 저자는 domain 개념의 fuzziness와 overlap을 충분히 강조하면서도, 실제 많은 기법이 여전히 discrete domain assumption에 크게 의존한다는 사실을 인정한다. 이 tension은 논문이 의도적으로 드러내는 문제이기도 하다. 즉, survey는 이산적 tag나 subnetworks가 실용적이라고 소개하면서도, 동시에 실제 언어 domain은 그보다 훨씬 더 연속적이고 중첩적이라는 점을 분명히 한다. 따라서 이 논문을 읽고 나면 “domain adaptation의 다음 단계는 더 정교한 domain representation”이라는 질문이 자연스럽게 남는다.

## 6. 결론

이 논문은 NMT에서 domain adaptation과 multi-domain adaptation을 폭넓고 체계적으로 정리한 survey로서, 핵심 기여는 세 가지로 요약할 수 있다. 첫째, domain adaptation을 data, architecture, training, inference라는 네 축으로 분류해 연구 지형을 구조화했다. 둘째, single-domain adaptation과 multi-domain adaptation을 구분하면서, forgetting, overfitting, unknown domain, continual learning 같은 실제 문제들을 연결해 설명했다. 셋째, low-resource translation, gender bias mitigation, document-level translation 같은 별도 연구 주제들도 adaptation 프레임으로 재해석할 수 있음을 보여주었다.

실용적인 관점에서 이 논문은 “새 domain 번역을 잘하고 싶을 때 무엇을 건드릴 수 있는가?”라는 질문에 매우 좋은 안내서다. bilingual in-domain data가 거의 없으면 monolingual data와 synthetic data를 활용할 수 있고, 기존 성능을 잃고 싶지 않으면 regularization이나 mixed fine-tuning, adapters를 고려할 수 있으며, 모델을 바꾸기 어렵다면 inference-time terminology control이나 rescoring, priming 같은 방법을 사용할 수 있다. 즉, adaptation은 하나의 기법이 아니라 여러 제약 조건 속에서 선택하는 설계 공간이라는 점을 잘 보여준다.

향후 연구 측면에서도 이 논문은 방향성을 제시한다. 저자는 특히 **extremely fine-grained adaptation**, **unsupervised adaptation**, **efficiency**, **intentional forgetting**을 중요한 미래 과제로 본다. 이는 오늘날에도 여전히 설득력 있는 전망이다. 실제 서비스에서는 사용자별, 문서별, 심지어 문장별 personalization 요구가 커지고 있으며, 그에 따라 적은 파라미터로 빠르게 적응하는 방법과, 원하지 않는 행동을 선택적으로 버리는 방법이 점점 중요해지고 있다.

종합하면, 이 논문은 새로운 adaptation 알고리즘을 제안하는 논문은 아니지만, NMT adaptation 연구 전체를 이해하는 데 매우 높은 가치를 갖는다. 특히 domain adaptation을 단순한 성능 개선 기술이 아니라 **번역 시스템의 동작을 상황에 맞게 통제하는 핵심 프레임**으로 제시했다는 점에서, 이후의 실제 응용과 후속 연구에 중요한 기반을 제공하는 survey라고 평가할 수 있다.
