# Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks

* **저자**: Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2004.10964](https://arxiv.org/abs/2004.10964)

## 1. 논문 개요

이 논문은 대규모 범용 pretrained language model이 이미 매우 큰 규모의 이질적 코퍼스로 학습되었더라도, 목표 작업이 속한 **domain**과 **task**에 맞게 추가 pretraining을 계속하는 것이 여전히 유의미한지 체계적으로 검증한다. 저자들은 RoBERTa를 기본 모델로 사용하고, biomedical, computer science, news, reviews의 네 개 domain과 여덟 개 분류 작업에서 성능 변화를 비교한다.

연구 문제는 비교적 명확하다. 이미 160GB 이상의 광범위한 텍스트로 학습된 강력한 language model이 존재할 때, 별도의 domain 특화 적응이 여전히 필요한가 하는 질문이다. 더 구체적으로는 세 가지 하위 질문이 있다. 첫째, 목표 task가 속한 domain의 unlabeled corpus로 추가 pretraining을 하면 성능이 개선되는가. 둘째, 더 작은 규모이지만 task와 직접적으로 연결된 unlabeled task corpus로 추가 pretraining을 하면 추가 이득이 있는가. 셋째, domain 전체 코퍼스나 대규모 human-curated unlabeled data가 없을 때, task와 유사한 데이터를 자동 선택하여 비슷한 효과를 얻을 수 있는가.

이 문제가 중요한 이유는, 최근의 NLP가 “더 크고 더 일반적인 pretrained LM”에 크게 의존하고 있기 때문이다. 만약 이런 범용 모델이 사실상 모든 textual variation을 충분히 포괄하지 못한다면, 실제 응용에서는 무조건 더 큰 범용 모델만 추구하는 대신, **적절한 domain/task 적응 절차**가 필요하다. 반대로 적응이 별 의미가 없다면, 여러 specialized LM을 만들 필요가 줄어든다. 이 논문은 이 질문에 대해 단일 사례가 아니라 여러 domain과 여러 데이터 규모 조건에서 비교적 폭넓은 실험을 수행했다는 점에서 의의가 크다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 pretrained LM의 적응 대상을 **domain 분포**와 **task 분포**로 분리해서 생각하는 데 있다. 논문 Figure 1의 설명대로, 실제 supervised task 데이터는 더 넓은 target domain 내부의 특정한 task distribution에서 관측된 샘플이다. 따라서 좋은 표현을 얻으려면 단순히 일반 영어 전체를 잘 모델링하는 것만으로는 부족할 수 있고, 먼저 넓은 domain에 맞춘 뒤 다시 task에 맞추는 다단계 적응이 효과적일 수 있다는 것이 저자들의 기본 직관이다.

이 관점에서 저자들은 두 가지 적응 방식을 정의한다. 첫 번째는 **DAPT (Domain-Adaptive Pretraining)** 으로, 대규모 domain-specific unlabeled corpus에 대해 RoBERTa의 masked language modeling을 계속 수행하는 방식이다. 두 번째는 **TAPT (Task-Adaptive Pretraining)** 으로, supervised task의 unlabeled training set 자체에 대해 추가 pretraining을 수행하는 방식이다. 논문의 핵심 주장은 이 둘이 경쟁 관계라기보다 **상보적**이라는 것이다. 즉, domain에 대한 넓은 적응과 task에 대한 좁고 직접적인 적응이 각각 다른 종류의 이점을 제공한다.

기존 접근과의 차별점은 몇 가지가 있다. 먼저, prior work는 주로 한 domain에서만 효과를 보거나, 더 작은 규모 혹은 덜 다양한 pretrained LM에서의 적응만 보였는데, 이 논문은 **강력한 범용 모델 RoBERTa**를 기준으로 여러 domain을 동시에 비교한다. 또한 단순히 “domain 적응이 좋다”에서 끝나지 않고, task 적응과의 관계, cross-task transfer, human-curated unlabeled data의 가치, 그리고 자동 데이터 선택까지 연결한다. 즉, 이 논문은 continued pretraining을 하나의 단편적 기법이 아니라 **실무적 적응 전략의 설계 공간 전체**로 다룬다.

## 3. 상세 방법 설명

전체 파이프라인은 복잡한 새로운 아키텍처를 제안하는 것이 아니라, 기존 RoBERTa에 대해 **추가 pretraining 단계를 설계**하는 것이다. 기본 출발점은 off-the-shelf RoBERTa-base이며, downstream classification에서는 최종 layer의 `[CLS]` representation을 task-specific feedforward layer에 넣어 예측한다. 즉, 분류 헤드는 표준적인 구조이고, 핵심은 사전학습 단계를 어떻게 추가하느냐에 있다.

RoBERTa 자체의 pretraining objective는 masked language modeling이다. 논문은 이 objective를 그대로 유지하며, domain 또는 task 데이터에 대해 추가 학습한다. 손실 함수의 형태는 token prediction에 대한 cross-entropy loss이며, 본문에서 새로운 목적 함수가 제안되지는 않는다. 따라서 방법론의 핵심은 새로운 수식 설계보다는 **어떤 unlabeled data 분포에 대해 기존 LM objective를 다시 적용할 것인가**에 있다. 논문이 제공하는 중요한 메시지는, 같은 objective라도 학습 데이터 분포를 바꾸면 downstream task 성능이 크게 달라질 수 있다는 점이다.

### 3.1 Domain-Adaptive Pretraining (DAPT)

DAPT는 RoBERTa를 큰 규모의 domain-specific unlabeled corpus에 대해 계속 pretraining하는 단계다. 실험에 사용한 네 domain은 다음과 같다.

BioMed는 S2ORC의 biomedical full-text papers 268만 편으로 약 7.55B tokens, 47GB 규모다.
CS는 S2ORC의 computer science papers 222만 편으로 약 8.10B tokens, 48GB다.
News는 RealNews 기사 1190만 편으로 약 6.66B tokens, 39GB다.
Reviews는 Amazon reviews 2475만 건으로 약 2.11B tokens, 11GB다.

저자들은 각 domain에 대해 RoBERTa를 12.5K steps 동안 학습한다. 본문과 appendix에 따르면 이는 사실상 각 domain dataset에 대해 single pass에 해당하도록 맞춘 설정이다. 하드웨어는 single v3-8 TPU를 사용했고, effective batch size는 2048이다. 학습률 스케줄과 optimizer는 RoBERTa 설정을 따르며, Adam, warmup linear scheduler, 최대 learning rate 0.0005 등을 사용했다.

이 단계의 직관은 간단하다. RoBERTa가 원래 다양한 영어 텍스트를 보았더라도, biomedical 논문이나 computer science 논문처럼 특수한 용어 분포와 문체를 충분히 내재화하지 못했을 수 있다. 이 경우 domain corpus로 masked LM objective를 더 수행하면, 표현이 해당 domain의 lexical/statistical regularity에 더 잘 맞춰진다.

### 3.2 Domain Similarity 분석

저자들은 DAPT 이전에 target domain이 RoBERTa의 원래 pretraining domain과 얼마나 비슷한지 정량화하려 시도한다. 방법은 매우 단순하지만 해석 가능하다. 각 domain의 held-out 문서 샘플에서 stopword를 제외한 상위 10K unigram vocabulary를 만들고, vocabulary overlap을 비교한다. RoBERTa 원래 pretraining corpus 전체는 공개되지 않았기 때문에, BookCorpus, Stories, Wikipedia, RealNews와 유사한 출처에서 50K 문서를 샘플링해 pretraining-domain vocabulary를 근사한다.

이 분석에서 News와 Reviews는 RoBERTa pretraining domain과 상당한 vocabulary overlap을 보였고, CS와 BioMed는 더 멀리 떨어져 있었다. 저자들은 이를 바탕으로 “더 멀리 떨어진 domain일수록 DAPT의 이득이 클 것”이라는 가설을 세운다. 실제 실험 결과도 대체로 이 직관과 맞아떨어진다.

다만 저자들은 vocabulary overlap이 완전한 설명은 아니라고 조심스럽게 말한다. 예를 들어 Reviews와 News는 단어 중첩이 상당하지만, 실제 데이터 생성 과정과 문서 길이, 문체 차이 같은 요인이 추가로 작동할 수 있다. Appendix의 cross-domain masked LM loss 분석에서도 이러한 복잡성이 드러난다.

### 3.3 Task-Adaptive Pretraining (TAPT)

TAPT는 더 작은 규모지만 task와 직접 연결된 unlabeled training data에 대해 추가 pretraining하는 방식이다. 예를 들어 ChemProt 분류를 하려면 ChemProt training set의 텍스트를 라벨 없이 다시 읽히면서 masked LM을 수행한다. 이 방법은 domain corpus 전체를 사용하는 DAPT보다 훨씬 저렴하지만, task relevance가 더 높다.

TAPT의 학습 설정은 DAPT와 유사하되, 데이터 규모가 훨씬 작으므로 12.5K steps 대신 **100 epochs** 학습을 사용한다. 또한 매 epoch마다 masking probability 0.15로 다른 token을 랜덤 masking하여 artificial augmentation 효과를 준다. 작은 데이터셋의 한계를 보완하려는 의도다. 데이터가 5K 미만인 task에서는 gradient accumulation을 포함한 batch size 256을 사용했다. learning rate는 더 작은 batch일 때 0.0001을 사용한다.

이 방식의 핵심은 domain 전체보다도 더 좁은 **task distribution**에 맞게 표현을 적응시키는 것이다. 논문은 같은 biomedical domain 안에서도 ChemProt와 RCT가 다르고, 같은 news domain 안에서도 HyperPartisan과 AGNews가 다를 수 있다고 본다. 따라서 “같은 domain”이라는 이유만으로 충분하지 않을 수 있으며, task-specific unlabeled corpus가 추가 이득을 줄 수 있다.

### 3.4 DAPT + TAPT

저자들은 DAPT와 TAPT를 조합한 다단계 적응도 실험한다. 절차는 RoBERTa에서 시작해 먼저 DAPT를 수행하고, 이어서 TAPT를 수행한 뒤 fine-tuning하는 것이다. 이 순서가 중요한데, 저자들은 반대로 TAPT 후 DAPT를 하면 task-relevant corpus에 대한 정보가 catastrophic forgetting으로 약화될 수 있다고 추정한다. 그러나 이는 실험적으로 직접 검증한 것은 아니며, 본문에서 speculation이라고 명시한다.

이 조합은 가장 계산량이 크지만, domain awareness와 task awareness를 동시에 확보한다는 장점이 있다. 실제로 Table 5에서는 거의 모든 task에서 최고 성능을 낸다. 논문의 중요한 결론 중 하나가 바로 “DAPT와 TAPT는 대체 관계가 아니라 누적적으로 이득을 준다”는 점이다.

### 3.5 Cross-Task Transfer와 Task 분포의 차이

TAPT가 특정 task에만 특화되는지 보기 위해 저자들은 같은 domain 내부 다른 task의 unlabeled data로 pretraining한 뒤 현재 task를 fine-tuning하는 **Transfer-TAPT**를 실험한다. 예를 들어 RCT 데이터로 task-adaptive pretraining한 모델을 ChemProt에 fine-tuning하는 식이다.

결과는 대체로 좋지 않다. Table 6에 따르면, 대부분의 경우 Transfer-TAPT는 원래 해당 task에 대한 TAPT보다 낮은 성능을 낸다. 이는 TAPT가 domain 일반성보다 task specificity를 강하게 학습한다는 뜻이다. 다시 말해 같은 domain 안에서도 task distribution은 꽤 다를 수 있으며, 이것이 DAPT만으로 충분하지 않고 TAPT가 별도로 유용한 이유를 설명한다.

### 3.6 Human-curated unlabeled data와 자동 데이터 선택

논문은 TAPT를 task training set의 unlabeled data에만 제한하지 않는다. 실제 annotation pipeline에서는 대개 더 큰 unlabeled pool이 먼저 있고, 그중 일부만 라벨링된다. 저자들은 이 더 큰 pool이 task distribution과 매우 유사하다고 보고, 이를 **Curated-TAPT**로 활용한다. RCT-500, HyperPartisan, IMDB에서 이 설정을 실험했으며, 결과적으로 기존 TAPT나 DAPT+TAPT보다 더 좋은 성능을 보였다.

이후 저자들은 human-curated pool이 없을 때를 가정하여 자동 데이터 선택을 제안한다. 여기서 사용한 방법은 VAMPIRE라는 lightweight bag-of-words language model로 task 문장과 domain 문장을 공통 embedding space에 사상한 뒤, 각 task sentence에 대해 domain corpus에서 $k$개의 nearest neighbors를 찾는 것이다. 검색은 FAISS와 cosine similarity를 사용한다. 이렇게 선택된 domain 문장들과 원래 task 데이터를 합쳐 pretraining하는 방식을 $k$nn-TAPT라고 부른다. 비교를 위해 같은 수의 문장을 무작위로 뽑는 rand-TAPT도 사용한다.

이 방법론의 핵심은 매우 큰 domain corpus 전체로 비싼 DAPT를 하지 않고도, task와 가까운 부분집합만 가져와 더 효율적인 적응을 수행하려는 것이다. 본문에는 새로운 복잡한 최적화 식은 없지만, 개념적으로는 “task distribution에 가까운 unlabeled examples를 retrieval로 보강한 TAPT”라고 이해하면 된다.

## 4. 실험 및 결과

### 4.1 실험 설정

논문은 네 개 domain에서 두 개씩, 총 여덟 개 분류 task를 평가한다.

BioMed domain에는 ChemProt와 RCT가 있다. ChemProt는 relation classification이며 train labeled examples는 4,169개다. RCT는 abstract sentence role classification이고 18,040개의 labeled training examples를 가진 high-resource task다.

CS domain에는 ACL-ARC와 SciERC가 있다. ACL-ARC는 citation intent classification으로 1,688개의 labeled training examples를 사용하고, SciERC는 relation classification으로 3,219개를 사용한다.

News domain에는 HyperPartisan과 AGNews가 있다. HyperPartisan은 low-resource setting으로 labeled train이 515개이고 추가 unlabeled train 5,000개가 있다. AGNews는 topic classification으로 labeled train 115,000개의 high-resource task다.

Reviews domain에는 Helpfulness와 IMDB가 있다. Helpfulness는 review helpfulness 분류로 labeled train 115,251개, IMDB는 sentiment classification으로 labeled train 20,000개와 unlabeled train 50,000개가 있다.

평가 지표는 대부분 macro-F1이며, ChemProt와 RCT는 prior work를 따라 micro-F1을 사용한다. 결과는 5개 random seed 평균과 표준편차를 보고한다. 이는 작은 데이터셋에서 seed variance가 크다는 문제를 인식하고 재현성을 높이려는 선택이다.

### 4.2 DAPT의 효과

Table 3에 따르면 DAPT는 거의 전 domain에서 baseline RoBERTa보다 성능을 향상시킨다.

ChemProt는 81.9에서 84.2로 상승했고, RCT는 87.2에서 87.6으로 올랐다.
ACL-ARC는 63.0에서 75.4로 크게 상승했고, SciERC는 77.3에서 80.8로 개선되었다.
HyperPartisan은 86.6에서 88.2로 상승했다.
AGNews는 93.9에서 93.9로 변화가 거의 없었다.
Helpfulness는 65.1에서 66.5로, IMDB는 95.0에서 95.4로 상승했다.

특히 ACL-ARC처럼 CS 논문 도메인 특성이 강한 저자 의도 분류에서는 이득이 매우 크다. 반면 AGNews는 RoBERTa pretraining data에 news가 이미 포함되어 있었기 때문에 추가 DAPT의 이득이 거의 없었다고 해석할 수 있다. 논문도 이런 방향으로 설명한다. 하지만 같은 news domain 안에서도 HyperPartisan에는 효과가 있었기 때문에, domain overlap이 높더라도 task 특성에 따라 추가 적응의 이득은 달라질 수 있다.

또한 domain relevance를 검증하기 위해 저자들은 각 task에 대해 **irrelevant domain**으로 적응한 모델과도 비교했다. 예를 들어 News task에는 CS LM, Reviews task에는 BioMed LM을 사용한다. 이 실험에서 대체로 DAPT가 irrelevant-domain adaptation보다 훨씬 좋았고, irrelevant adaptation은 종종 baseline RoBERTa보다도 나쁜 결과를 냈다. 이는 단순히 “추가 데이터에 더 노출되면 좋다”가 아니라, **relevant data에 노출되어야 좋다**는 점을 보여준다.

### 4.3 TAPT의 효과

Table 5는 TAPT가 모든 task에서 baseline을 일관되게 개선함을 보여준다.

ChemProt는 81.9에서 82.6으로, RCT는 87.2에서 87.7로 개선된다.
ACL-ARC는 63.0에서 67.4로 개선되고, SciERC는 77.3에서 79.3으로 오른다.
HyperPartisan은 86.6에서 90.4로 크게 상승한다.
AGNews는 93.9에서 94.5, Helpfulness는 65.1에서 68.5, IMDB는 95.0에서 95.5로 개선된다.

흥미로운 점은 TAPT가 비용은 더 낮지만, 일부 task에서는 DAPT와 맞먹거나 오히려 더 좋다는 것이다. 예를 들어 RCT, HyperPartisan, AGNews, Helpfulness, IMDB에서는 TAPT가 DAPT보다 우수하다. 이는 task relevance가 매우 강력한 신호라는 뜻이다. domain 전체를 보는 것보다, 작은 규모라도 task에 직접 맞닿은 데이터가 더 효율적인 경우가 있다는 메시지다.

### 4.4 DAPT + TAPT의 효과

DAPT와 TAPT를 순차적으로 결합한 결과는 Table 5에서 거의 모든 task의 최고 성능을 낸다.

ChemProt는 84.4, RCT는 87.8, ACL-ARC는 75.6, SciERC는 81.3, HyperPartisan은 90.0, AGNews는 94.6, Helpfulness는 68.7, IMDB는 95.6이다.

이 결과는 다단계 적응의 논리를 잘 뒷받침한다. DAPT는 보다 넓은 domain regularity를 학습하고, TAPT는 그 위에서 task distribution에 더욱 정밀하게 맞춘다. 따라서 task에 따라 어느 하나만으로도 성능 향상이 있지만, 둘을 결합하면 대체로 누적 이득이 발생한다.

다만 state-of-the-art와 비교하면 항상 최고는 아니다. 예를 들어 RCT의 비교 대상 SoTA는 92.9로 본 논문보다 훨씬 높고, HyperPartisan도 Longformer 기반 SoTA 94.8에는 미치지 못한다. 그러나 이 논문의 목적은 새로운 task-specific SOTA architecture를 제안하는 것이 아니라, **적응 전략 자체의 일반성과 유효성**을 입증하는 데 있다. 같은 기본 백본 RoBERTa를 유지한 상태에서 일관된 개선을 보였다는 점이 중요하다.

### 4.5 Cross-task transfer 결과

Table 6은 TAPT가 다른 task로는 잘 옮겨가지 않는다는 점을 보여준다. 예를 들어 BioMed에서 ChemProt용 TAPT는 82.6인데, 다른 task의 data로 Transfer-TAPT하면 80.4가 된다. News에서 HyperPartisan는 90.4에서 82.2로 크게 하락하고, AGNews는 94.5에서 93.9로 소폭 감소한다. Reviews에서도 Helpfulness는 68.5에서 65.0으로 떨어진다.

이 결과는 TAPT가 강력하지만 task-specific하다는 것을 뜻한다. 같은 domain이라도 task들 사이의 데이터 분포 차이가 적지 않으며, task에 지나치게 맞춘 pretraining은 다른 task에는 오히려 독이 될 수 있다. 이는 실무적으로도 중요하다. 한 domain 안의 여러 task에 대해 재사용 가능한 intermediate LM을 만들고 싶다면, TAPT보다는 DAPT가 더 적합할 가능성이 있다.

### 4.6 Human-curated data의 가치

Table 7에서 Curated-TAPT는 세 개 task 모두에서 strong baseline보다 더 좋아진다. RCT-500에서는 TAPT 79.8, DAPT+TAPT 83.0, Curated-TAPT 83.4, DAPT+Curated-TAPT 83.8이다. HyperPartisan에서는 Curated-TAPT가 90.4 내지 92.1 수준의 향상을 보이며, IMDB도 95.7 내지 95.8까지 오른다.

특히 저자들이 강조하는 포인트는, RCT 전체 labeled corpus를 쓰는 설정의 DAPT+TAPT 성능에 비해, RCT-500처럼 labeled data가 극도로 적은 상황에서도 Curated-TAPT를 사용하면 상당한 수준까지 따라간다는 점이다. 논문은 “fully labeled RCT corpus를 이용한 DAPT+TAPT 성능의 95% 수준을, labeled data 0.3%만으로 달성한다”고 해석한다. 이 수치는 task-distribution unlabeled pool의 실질적 가치를 강하게 보여준다.

### 4.7 자동 데이터 선택 결과

Table 8의 자동 데이터 선택 결과를 보면, $k$nn-TAPT는 모든 경우에 기본 TAPT보다 낫다. 또한 대체로 random selection보다도 좋다. ChemProt에서는 TAPT 82.6에 비해 50nn-TAPT, 150nn-TAPT, 500nn-TAPT가 모두 83.2~83.3 정도로 개선된다. RCT-500에서는 79.8에서 80.8, 81.2, 81.7로 $k$가 커질수록 성능이 증가한다. ACL-ARC도 67.4에서 70.7, 73.3, 75.5로 상승하여 DAPT 75.4에 근접하거나 약간 넘는다.

이는 간단한 retrieval 기반 data augmentation만으로도 DAPT에 가까운 효과를 부분적으로 얻을 수 있음을 의미한다. 특히 computational resource가 부족할 때, domain 전체 수천만 문서를 다시 pretrain하는 대신 task와 가까운 문장만 선택해서 학습하는 전략이 상당히 현실적이라는 점을 보여준다.

### 4.8 계산 비용 비교

Table 9는 RCT-500 기준으로 adaptation 비용을 비교한다. TAPT는 0.2K steps, 500 docs, 80KB 저장공간만으로 가능하다. 반면 DAPT는 12.5K steps, 25M docs, 47GB 저장공간이 필요하다. 저자 표현대로 TAPT는 single v3-8 TPU 기준 DAPT보다 거의 60배 빠르고, 저장공간 요구량은 580만 배 정도 작다.

이 비교는 논문의 메시지를 실용적으로 강화한다. 최고 성능만 보면 DAPT+TAPT가 좋지만, 비용 대비 효율을 생각하면 TAPT나 Curated-TAPT, 혹은 $k$nn-TAPT가 매우 매력적이다. 즉, 이 논문은 “무조건 가장 큰 적응이 최고”가 아니라, **자원 상황에 맞는 적응 전략 선택**의 필요성을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 continued pretraining을 단일 기법이 아니라 설계 공간으로 본다는 점이다. DAPT, TAPT, DAPT+TAPT, Curated-TAPT, $k$nn-TAPT를 같은 프레임 안에서 비교하며, 각각의 장점과 비용을 함께 보여준다. 이 때문에 연구 결과가 단순히 “이 방법이 좋다” 수준을 넘어, 실제 적용 시 어떤 전략을 선택해야 할지 판단하는 데 도움이 된다.

또 다른 강점은 실험 범위다. 네 domain, 여덟 task, high-resource와 low-resource 설정, irrelevant-domain control, cross-task transfer, human-curated augmentation, automatic data selection까지 포함한다. 이 정도로 다양한 축을 같은 논문에서 일관되게 비교한 점은 설득력이 높다. 또한 random seed 평균과 표준편차를 제시하고, development 결과와 구현 details를 appendix에 제공한 점도 실험 보고의 성실성으로 볼 수 있다.

세 번째 강점은 결과 해석이 단순한 성능 비교를 넘어선다는 점이다. domain similarity 분석, domain overlap 사례, cross-domain masked LM loss, transfer-TAPT 실패 사례 등을 통해 “domain”과 “task”의 관계를 더 세밀하게 해석하려 한다. 이는 단순한 benchmark 논문보다 개념적 통찰을 더 제공한다.

그러나 한계도 분명하다. 첫째, 본 논문은 RoBERTa-base를 중심으로 실험하며, 결과를 “어떤 pretrained LM에도 일반적으로 적용 가능하다”고 결론에서 말하지만, 실제로 여러 backbone에 대해 광범위하게 검증한 것은 아니다. 따라서 다른 구조나 더 큰 모델에서 같은 크기의 효과가 나오는지는 본문만으로 확정할 수 없다.

둘째, domain similarity 분석은 주로 unigram overlap과 masked LM loss에 의존한다. 이는 직관적이지만 domain의 의미론적, 문체적, 구조적 차이를 충분히 설명하지는 못한다. 실제로 저자들 스스로 Reviews와 News의 관계에서 단순한 overlap 지표만으로는 설명되지 않는 현상을 보고한다. 따라서 “무엇이 domain인가”라는 더 근본적인 질문에는 아직 부분적으로만 답한다.

셋째, task-adaptive pretraining의 성공은 task unlabeled data가 충분히 task distribution을 대표한다는 가정에 기대고 있다. 하지만 실제 산업 환경에서는 training set이 편향되어 있거나 미래 test distribution과 어긋날 수 있다. 논문은 이 가능성을 깊게 다루지 않는다. 다시 말해 TAPT는 강력하지만, 잘못된 task distribution에 과적응할 위험도 있다.

넷째, HyperPartisan처럼 작은 데이터셋에서는 seed variance가 매우 커서 degenerate seeds를 discard and resample했다고 appendix에서 밝힌다. 이 처리는 실용적으로는 이해되지만, 방법 비교의 엄밀성 측면에서는 해석에 주의가 필요하다. 어떤 기준으로 degenerate seed를 제거했는지, 이것이 다른 설정과 얼마나 공정하게 비교되는지는 본문에서 충분히 상세히 논의되지 않는다.

다섯째, 논문은 catastrophic forgetting 가능성을 언급하며 DAPT 후 TAPT 순서를 채택하지만, 반대 순서나 더 정교한 curriculum 방식은 직접 실험하지 않는다. 따라서 “왜 이 조합이 최선인가”에 대한 메커니즘 수준 설명은 아직 제한적이다.

비판적으로 보면, 이 논문은 새로운 모델 구조를 제안하지 않음에도 매우 영향력이 큰데, 그 이유는 범용 LM 시대에 오히려 데이터 분포 설계의 중요성을 다시 부각했기 때문이다. 다만 이 결론은 동시에 “좋은 adaptation을 위해 좋은 unlabeled data를 어떻게 확보할 것인가”라는 더 어려운 문제를 남긴다. 결국 모델보다 데이터 선택이 중요하다고 말하면서, 그 데이터 선택의 품질이 성능을 좌우하게 되기 때문이다.

## 6. 결론

이 논문은 대규모 범용 pretrained LM이 존재하더라도, 추가 pretraining을 멈추지 말고 목표 **domain**과 **task**에 맞게 계속 적응시켜야 한다는 점을 실험적으로 설득력 있게 보여준다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, DAPT와 TAPT를 여러 domain과 task에서 체계적으로 비교하여 둘 다 유효함을 보였다. 둘째, 두 방법을 결합하면 대부분의 경우 최고 성능을 얻는다는 점을 확인했다. 셋째, human-curated unlabeled data와 retrieval 기반 자동 데이터 선택이 저비용 대안으로 유망함을 제시했다.

실제 적용 측면에서 이 연구는 매우 중요하다. 의료, 과학기술, 법률, 고객 리뷰, 뉴스처럼 언어 분포가 뚜렷하게 다른 환경에서는, 단순히 범용 foundation model을 fine-tuning하는 것만으로 충분하지 않을 수 있다. 적절한 unlabeled corpus를 활용한 multi-phase adaptive pretraining은 적은 labeled data 환경에서도 성능을 크게 높일 수 있다. 특히 DAPT로 reusable domain LM을 만든 뒤, 각 task마다 TAPT를 추가하는 전략은 조직 차원의 모델 운영에서도 현실적인 설계가 될 수 있다.

향후 연구 방향도 자연스럽다. 어떤 domain 정의가 가장 유용한지, task와 domain의 경계를 어떻게 수학적으로 더 잘 모델링할지, 더 큰 최신 LM에서도 같은 경향이 유지되는지, 그리고 automatic data selection을 더 정교하게 개선할 수 있는지가 후속 과제가 된다. 이 논문은 “더 큰 모델”만이 답이 아니라, **더 적절한 데이터와 더 적절한 적응 절차**가 equally important하다는 점을 분명히 한 작업이라고 평가할 수 있다.
