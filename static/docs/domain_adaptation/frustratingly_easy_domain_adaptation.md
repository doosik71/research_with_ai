# Frustratingly Easy Domain Adaptation

* **저자**: Hal Daumé III
* **발표연도**: 2009
* **arXiv**: <https://arxiv.org/abs/0907.1815>

## 1. 논문 개요

이 논문은 **fully supervised domain adaptation** 문제를 매우 단순한 방식으로 해결하는 방법을 제안한다. 문제 설정은 다음과 같다. 하나의 **source domain**에는 라벨이 많이 달린 데이터가 있고, 실제로 성능을 내고 싶은 **target domain**에는 라벨이 적게 달린 데이터가 있다. 이때 source만 쓰면 target에 잘 맞지 않을 수 있고, target만 쓰면 데이터가 너무 적어서 일반화가 어렵다. 따라서 두 도메인의 정보를 동시에 활용하되, 두 도메인이 완전히 같지도 않고 완전히 다르지도 않다는 점을 반영하는 학습 방법이 필요하다.

논문의 핵심 목표는 이 문제를 새로운 복잡한 학습 알고리즘으로 푸는 것이 아니라, **표준 supervised learning 문제로 변환**해서 기존의 maxent, SVM, perceptron 같은 일반적인 분류기를 그대로 사용할 수 있게 만드는 것이다. 저자는 이를 위해 feature space를 아주 간단히 확장하는 전처리 방법을 제안한다. 논문 제목에 들어간 “Frustratingly Easy”라는 표현은, 기존의 복잡한 domain adaptation 기법들보다 훨씬 단순한데도 성능이 매우 좋다는 점을 강조한다.

이 문제가 중요한 이유는 NLP를 비롯한 실제 머신러닝 응용에서 데이터 분포 차이, 즉 **domain shift**가 매우 흔하기 때문이다. 예를 들어 뉴스 기사에서 학습한 모델을 biomedical text, 블로그, 음성 인식 출력, 웹 포럼 같은 다른 텍스트 도메인에 적용하면 성능이 크게 떨어질 수 있다. 따라서 domain adaptation은 실제 배포 가능한 시스템을 만들기 위해 매우 중요한 문제다.

또한 이 논문은 실용성 측면에서도 의미가 크다. 제안 방법은 논문에서 강조하듯이 전처리 수준에서 구현 가능하며, 실제로 아주 짧은 Perl 스크립트로 구현할 수 있을 정도로 간단하다. 그럼에도 여러 sequence labeling 태스크에서 당시 강력한 baseline들과 state-of-the-art 방법을 능가하거나 비슷한 성능을 보였다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 직관적이다. 원래의 각 feature를 하나만 쓰지 않고, 이를 세 가지 버전으로 나눈다.

첫째는 **general feature**로, source와 target 모두에 공통으로 적용되는 정보다. 둘째는 **source-specific feature**로, source domain에서만 의미를 갖는 정보다. 셋째는 **target-specific feature**로, target domain에서만 의미를 갖는 정보다. 이렇게 하면 모델은 어떤 feature가 두 도메인에 공통적으로 유효한지, 아니면 특정 도메인에서만 유효한지를 자동으로 학습할 수 있다.

이 설계의 직관은 논문에서 제시한 예로 잘 설명된다. part-of-speech tagging에서 “the”라는 단어는 거의 모든 도메인에서 determiner로 동작하므로 공통 feature로 다루는 것이 자연스럽다. 반면 “monitor”라는 단어는 WSJ 뉴스 도메인에서는 동사일 가능성이 크지만, 하드웨어 리뷰 도메인에서는 명사일 가능성이 크다. 이런 경우 “monitor”라는 원래 feature 하나만으로는 두 도메인의 차이를 표현하기 어렵지만, source-specific과 target-specific 복사본을 따로 두면 서로 다른 의미를 따로 학습할 수 있다.

기존 접근과의 차별점은, prior model이나 Daumé III and Marcu의 복잡한 latent-variable 기반 모델처럼 **학습 절차 자체를 바꾸지 않고**, 입력 표현만 바꾼다는 점이다. 특히 prior model은 먼저 source 모델을 학습하고, 그 가중치를 target 학습의 prior처럼 사용하는 **순차적 접근**이다. 반면 이 논문의 방법은 source와 target을 하나의 확장된 feature space에서 **jointly optimization**한다. 즉 공통 성분과 도메인별 성분의 trade-off를 단일 학습 과정에서 한 번에 조절한다.

또 하나의 중요한 차별점은 이 방법이 **multi-domain adaptation**으로 자연스럽게 확장된다는 것이다. 두 도메인일 때는 feature space를 세 배로 늘리면 되지만, $K$개 도메인이 있으면 각 도메인용 복사본과 공통 복사본 하나를 둬서 총 $(K+1)$배로 확장하면 된다. 즉 구조 자체가 매우 일반적이다.

## 3. 상세 방법 설명

### 3.1 문제 설정

입력 공간을 $\mathcal{X}$, 출력 공간을 $\mathcal{Y}$라고 하자. source domain의 분포를 $\mathcal{D}^{s}$, target domain의 분포를 $\mathcal{D}^{t}$로 둔다. source 데이터셋 $D^{s}$에는 $N$개의 예제가 있고, target 데이터셋 $D^{t}$에는 $M$개의 예제가 있다. 일반적으로 $N \gg M$이다. 목표는 target domain에서의 expected loss가 낮은 함수 $h:\mathcal{X}\rightarrow\mathcal{Y}$를 학습하는 것이다.

논문은 설명을 위해 $\mathcal{X}=\mathbb{R}^{F}$, $\mathcal{Y}={-1,+1}$를 가정하지만, 실제 아이디어는 이진 분류에만 국한되지 않고 더 일반적인 supervised learning에 적용될 수 있다고 설명한다.

### 3.2 기존 baseline과 prior work

논문은 먼저 몇 가지 기본 baseline을 정리한다.

**SrcOnly**는 source 데이터만으로 학습한다.
**TgtOnly**는 target 데이터만으로 학습한다.
**All**은 두 데이터를 단순히 합쳐서 학습한다.
**Weighted**는 source 데이터가 너무 많아 target 신호가 묻히는 문제를 완화하기 위해 source 예제에 가중치를 낮게 주는 방식이다.
**Pred**는 source 모델의 예측값을 target 모델의 feature로 추가한다.
**LinInt**는 source 모델과 target 모델의 예측을 선형 결합한다.

그 다음으로 stronger baseline으로 **Prior model**이 소개된다. 이 방법은 source에서 학습한 가중치 $\boldsymbol{w}^{s}$를 target 모델의 prior처럼 사용한다. 일반적인 regularization이 $\lambda |\boldsymbol{w}|_2^2$라면, Prior model은 이를 다음처럼 바꾼다.

$$
\lambda |\boldsymbol{w}-\boldsymbol{w}^{s}|_2^2
$$

이 의미는 target 모델의 가중치가 source 모델의 가중치와 비슷하도록 선호하되, target 데이터가 필요하면 벗어날 수 있게 한다는 것이다.

또 다른 기존 방법은 source-specific, target-specific, general component를 latent하게 나누는 복잡한 모델이다. 논문에 따르면 이 모델은 EM 알고리즘을 사용하며 성능은 좋지만 구현이 어렵고 속도가 훨씬 느리다.

### 3.3 제안 방법: Feature Augmentation

이 논문의 핵심은 원래의 feature vector $\boldsymbol{x}\in\mathbb{R}^{F}$를 세 부분으로 복제해, 확장된 공간 $\breve{\mathcal{X}}=\mathbb{R}^{3F}$로 매핑하는 것이다.

source 데이터에 대해서는 다음 변환을 사용한다.

$$
\Phi^{s}(\boldsymbol{x}) = \langle \boldsymbol{x}, \boldsymbol{x}, \boldsymbol{0} \rangle
$$

target 데이터에 대해서는 다음 변환을 사용한다.

$$
\Phi^{t}(\boldsymbol{x}) = \langle \boldsymbol{x}, \boldsymbol{0}, \boldsymbol{x} \rangle
$$

여기서 첫 번째 블록은 **general**, 두 번째 블록은 **source-specific**, 세 번째 블록은 **target-specific** feature를 의미한다.

이 표현의 의미를 쉽게 말하면 다음과 같다. source 예제는 “공통 특징”과 “source 전용 특징”을 동시에 갖고, target 예제는 “공통 특징”과 “target 전용 특징”을 동시에 갖는다. 따라서 학습 알고리즘은 하나의 weight vector 안에서 공통 패턴과 도메인 특수 패턴을 동시에 조절할 수 있다.

예를 들어 “the”처럼 두 도메인에서 동일하게 유용한 feature는 general 부분의 weight가 커지면 된다. 반대로 “monitor”처럼 도메인에 따라 역할이 달라지는 feature는 source-specific과 target-specific 부분에 다른 weight가 실리면 된다.

논문은 이 확장이 다소 중복적이라고도 언급한다. 더 작은 표현도 가능하지만, 현재의 3-way 복제가 분석과 multi-domain 확장 측면에서 더 다루기 쉽다고 설명한다.

### 3.4 Kernelized version

논문은 모든 실험에서 선형 모델을 사용했지만, 이 방법의 kernelized 해석도 제시한다. 원래 kernel을 $K(x,x')$라고 하자. 확장된 공간에서의 kernel $\breve{K}(x,x')$는 같은 도메인인지 다른 도메인인지에 따라 달라진다.

같은 도메인일 때는

$$
\breve{K}(x,x') = 2K(x,x')
$$

다른 도메인일 때는

$$
\breve{K}(x,x') = K(x,x')
$$

즉 정리하면,

$$
\breve{K}(x,x')=
\begin{cases}
2K(x,x') & \text{same domain} \
K(x,x') & \text{different domain}
\end{cases}
$$

이 결과는 매우 직관적이다. kernel을 similarity라고 보면, **같은 도메인에 속한 두 샘플은 기본적으로 다른 도메인 샘플보다 두 배 더 비슷하다고 간주**하는 셈이다. 논문은 이 해석을 통해 target test sample에 대해 target training sample의 영향력이 source sample보다 더 크게 반영된다고 설명한다. 즉 완전히 source를 버리지는 않지만, target 쪽에 더 무게를 둔다.

### 3.5 Prior model과의 관계

저자는 이 방법이 Prior model과 꽤 비슷하다고 분석한다. 만약 일반 weight를 $w_g$, source 예측에 쓰이는 유효 weight를 $w_s$, target 예측에 쓰이는 유효 weight를 $w_t$라고 보면, feature-augmentation된 전체 weight의 $\ell_2$ regularization은 대략 다음 꼴로 쓸 수 있다.

$$
|w_g|^2 + |w_s - w_g|^2 + |w_t - w_g|^2
$$

이 식에서 자유 변수 $w_g$를 적절히 선택하면, 결과적으로 $w_s$와 $w_t$가 서로 크게 다르지 않도록 압박하는 regularizer가 된다. 즉 형태상으로는 Prior model의 “source와 target 가중치가 유사하도록 유도”하는 효과와 비슷하다.

하지만 논문은 왜 feature augmentation이 더 나을 수 있는지를 두 가지로 설명한다.

첫째, source와 target을 **순차적으로**가 아니라 **동시에 joint하게** 최적화한다.
둘째, 공통/도메인별 가중치의 균형을 사람이 하이퍼파라미터로 강하게 조정하기보다, 단일 supervised learner가 데이터에 맞춰 자연스럽게 조절하게 만든다.

즉 이 방법은 prior를 수동으로 설계하는 대신, 표현 공간을 바꿔서 학습 알고리즘이 적절한 구조를 스스로 찾게 한다.

### 3.6 Multi-domain adaptation

두 도메인일 때는 원래 $F$차원 feature를 $3F$차원으로 확장했다. 도메인이 $K$개라면 각 도메인 전용 feature block이 $K$개, 그리고 공통 block이 하나 필요하므로 전체 차원은 다음과 같다.

$$
\mathbb{R}^{(K+1)F}
$$

즉 각 샘플은 자신의 도메인에 해당하는 전용 block과 공통 block만 활성화시키고, 나머지는 모두 0으로 둔다. 이 구조는 여러 source domain이 있을 때도 자연스럽게 적용 가능하다.

### 3.7 학습 절차

논문에서 제안한 방법 자체는 별도의 복잡한 optimization 절차를 요구하지 않는다. 실제 절차는 다음처럼 이해할 수 있다.

먼저 source와 target의 모든 입력 feature를 위 식대로 확장한다.
그 다음 확장된 데이터를 하나의 supervised training set으로 만들어 표준 learner에 넣는다.
이후 학습된 weight vector는 일반 feature, source 전용 feature, target 전용 feature를 동시에 포함하게 된다.
테스트 시에는 target 예제에 대해 $\Phi^t(\boldsymbol{x})$ 변환을 적용한 뒤 예측한다.

즉 “domain adaptation 알고리즘”이라기보다, **domain adaptation 문제를 표준 지도학습 입력 형식으로 바꾸는 변환 규칙**이라고 보는 것이 더 정확하다.

## 4. 실험 및 결과

### 4.1 실험 설정

논문은 다양한 sequence labeling 태스크에서 제안 방법을 평가한다. 사용된 태스크는 named-entity recognition, shallow parsing, part-of-speech tagging, recapitalization이다.

구체적인 데이터셋은 다음과 같다.

ACE-NER는 ACE 2005의 여러 도메인(bn, bc, nw, wl, un, cts)으로 구성된 named entity recognition 태스크다.
CoNLL-NE는 2006 ACE 데이터를 source, CoNLL 2003 NER 데이터를 target으로 쓰는 설정이다.
PubMed-POS는 WSJ Penn Treebank를 source, PubMed abstract를 target으로 쓰는 POS tagging 문제다.
CNN-Recap은 newswire를 source, ASR system output을 target으로 하는 recapitalization 태스크다.
Treebank-Chunk와 Treebank-Brown은 shallow parsing 태스크이며, WSJ, Switchboard, Brown corpus의 여러 세부 도메인을 포함한다.

논문은 각 데이터셋의 train/dev/test 크기와 feature 수를 자세히 표로 제시한다. 예를 들어 PubMed-POS는 source 데이터가 약 95만 개, target train은 1만 1천여 개이고, CNN-Recap은 source가 200만 개로 매우 크다. 즉 source와 target의 데이터 크기 차이가 큰, domain adaptation에 전형적인 환경을 실험적으로 다룬다.

모든 실험에서 sequence labeling을 위해 **Searn** 알고리즘과 그 내부 classifier로 **averaged perceptron**을 사용했다. structural feature에 대해서는 second-order Markov assumption을 사용했고, 평가는 단순성과 통계적 유의성 검정을 위해 주로 **label accuracy**를 사용했다. 논문은 chunking에서 F1보다 accuracy를 택한 이유도 명시적으로 설명한다.

### 4.2 비교 대상

비교 대상은 다음과 같다.

SrcOnly, TgtOnly, All, Weighted, Pred, LinInt 같은 강한 baseline들이 포함되었고, stronger prior work로 Prior model이 들어갔다. 또한 본문 표에는 없지만, 추가 비교로 Daumé III and Marcu (2006)의 **MegaM** 모델도 따로 실험했다고 설명한다.

제안 방법은 표에서 **Augment**라고 표시된다.

### 4.3 주요 정량 결과

논문 Table 2의 수치는 **error rate**이며, 낮을수록 좋다.

전반적으로 저자는 Brown 세부 도메인을 제외하면 **Augment가 거의 항상 최고 성능**이라고 주장한다. 실제 표를 보면 다음과 같은 대표 사례들이 있다.

* **ACE-NER**

  * bn: Prior 2.06, Augment 1.98
  * nw: Prior 3.68, Augment 3.39
  * un: Prior 2.03, Augment 1.91
  * cts: Prior 0.34, Augment 0.32

* **PubMed-POS**

  * SrcOnly 12.02, TgtOnly 4.15, Prior 3.99, Augment 3.61
    source만 쓰면 매우 나쁘고, target 소량 데이터만 써도 많이 개선되며, Augment가 추가로 크게 개선한다.

* **CNN-Recap**

  * SrcOnly 10.29, TgtOnly 3.82, Prior 3.35, Augment 3.37
    여기서는 Prior가 아주 근소하게 더 좋다. 다만 차이는 매우 작고, 논문은 통계적 유의성 기준에서 묶여 있을 가능성을 함께 고려한다.

* **CoNLL target**

  * SrcOnly 2.49, TgtOnly 2.95, All 1.80, Weighted 1.75, LinInt 1.77, Prior 1.89, Augment 1.76
    CoNLL에서는 Weighted가 가장 낮고 Augment는 거의 비슷한 수준이다.

* **Treebank-Brown**

  * SrcOnly 6.35, TgtOnly 5.75, Prior 4.72, Augment 4.65
    여기서는 Augment가 가장 좋다.

논문의 핵심 메시지는 Brown의 세부 하위도메인(br-cf, br-cg 등)을 개별적으로 target으로 둘 때는 Augment가 일관되게 강하지 않다는 점을 제외하면, 대부분의 실험에서 단순 baseline뿐 아니라 당시의 강력한 Prior 모델도 이긴다는 것이다.

### 4.4 결과 해석

저자는 중요한 경험적 패턴 하나를 지적한다. Augment가 잘 안 되는 경우는 대체로 **SrcOnly가 TgtOnly보다 더 좋은 경우**와 겹친다는 것이다. 이는 source와 target이 실제로 꽤 비슷해서, 굳이 feature space를 분리해 도메인 차이를 강조하지 않아도 되는 상황일 수 있다. 다시 말해 이 방법은 “두 도메인이 어느 정도 다르지만, 그래도 일부 공통 구조는 공유한다”는 조건에서 특히 유리하다.

논문 초록과 discussion에서도 비슷한 조건을 강조한다. 즉 target 데이터가 너무 적어서 target-only로는 부족하지만, 그렇다고 source-only가 충분히 좋은 것도 아닌 상황, 곧 **“target data가 조금 있어서 source-only보다 약간 더 잘할 수 있는 경우”**에 가장 적합하다는 것이다.

### 4.5 MegaM과의 비교

논문은 표에는 넣지 않았지만 MegaM 모델도 추가 비교했다. 대체로 MegaM은 각 태스크에서 최고 모델들과 비슷한 수준이었고, 일부 Brown 세부 도메인에서는 Augment보다 낫기도 했다. 하지만 학습 시간이 훨씬 길고, 하이퍼파라미터 튜닝을 위해 여러 번의 cross-validation이 필요했다. 반면 Augment는 훨씬 단순하고 빠르다. 따라서 성능만이 아니라 **효율성과 구현 용이성**까지 고려하면 제안 방법의 실용적 장점이 크다고 볼 수 있다.

### 4.6 모델 해석 분석

논문은 단순히 성능표만 제시하지 않고, ACE-NER에서 학습된 weight를 분석해 제안 방법이 실제로 “상식적인” 도메인 구조를 학습하는지 살펴본다.

예를 들어 `/Aa+/` 패턴, 즉 첫 글자만 대문자이고 나머지는 소문자인 경우는 일반적으로 entity를 나타내는 강한 신호다. 실제로 general domain weight에서 긍정적인 경향이 나타난다. 그러나 broadcast news에서는 capitalization 정보가 없어서 이 feature가 큰 의미가 없고, usenet에서는 이메일 주소나 URL 때문에 오히려 음의 weight를 갖는 경향이 있다고 해석한다.

또 `/bush/`라는 단어 feature는 기본적으로 사람 이름(PER) 쪽으로 가중치가 가지만, conversational domain에서는 식물 의미로도 많이 등장할 수 있어서 다르게 학습된다. `/the/`의 현재 위치와 이전 위치 feature도 도메인별로 역할이 달라지며, 특정 도메인에서 흔한 표현들 때문에 weight가 달라진다고 설명한다.

이 분석은 제안 방법이 단순히 차원을 늘려 overparameterization 효과만 얻는 것이 아니라, 실제로 **공통 구조와 도메인 특수 구조를 분리해서 의미 있게 학습**하고 있음을 뒷받침하려는 시도다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **압도적인 단순성**이다. 새로운 최적화 알고리즘, latent variable model, 복잡한 inference 없이도 domain adaptation을 수행할 수 있다. 기존 supervised learner에 바로 결합할 수 있고, 구현 부담이 매우 낮다. 이 단순성은 연구용뿐 아니라 실제 시스템 적용에도 큰 장점이다.

둘째 강점은 **강한 경험적 성능**이다. 논문은 여러 sequence labeling 태스크에서 baseline들을 꾸준히 이기거나 최소한 매우 경쟁력 있는 결과를 보여준다. 특히 Prior model처럼 이미 강한 비교 방법보다도 종종 더 좋은 결과를 낸다.

셋째는 **모델 해석 가능성**이다. general/source/target weight로 분해되므로 어떤 feature가 공통적으로 작동하는지, 어떤 feature가 도메인마다 다르게 작동하는지를 직접 볼 수 있다. 이는 단순한 black-box adaptation보다 해석 측면에서 이점이 있다.

넷째는 **확장성**이다. multi-domain adaptation으로 거의 같은 아이디어를 바로 확장할 수 있다는 점은 실용적이다. 실제 데이터는 source가 하나가 아니라 여러 하위 도메인으로 나뉘는 경우가 많기 때문이다.

다만 한계도 분명하다. 우선 이 논문은 **fully supervised adaptation**만 다룬다. 즉 target에 라벨된 데이터가 조금이라도 있어야 한다. 실제로 많은 domain adaptation 상황에서는 unlabeled target data만 있는 semi-supervised 또는 unsupervised setting이 더 중요할 수 있는데, 이 논문은 그 경우를 다루지 않는다.

또한 이 방법은 도메인 차이를 단지 “공통 + 도메인 전용 weight” 구조로 표현한다. 이는 매우 단순하고 강력하지만, 더 복잡한 구조적 차이나 feature correspondence를 명시적으로 모델링하지는 않는다. 예를 들어 structural correspondence learning처럼 서로 다른 표현 간 대응 관계를 적극적으로 찾지는 않는다.

실험 결과 측면에서도 모든 경우에 항상 우월하지는 않다. 특히 Brown corpus의 세부 도메인들에서는 성능이 불안정하고, 어떤 경우에는 Prior나 다른 방법보다 못하다. 논문은 이를 source와 target이 충분히 유사한 경우로 해석하지만, 그 자체로 보면 제안 방법의 적용 조건이 있다는 의미다.

이론적 분석도 제한적이다. 논문은 “이 방법이 learning을 더 어렵게 만들지는 않는다”는 정도의 직관과 Prior model과의 유사성은 제시하지만, 왜 어떤 상황에서 더 잘 작동하는지에 대한 완전한 이론은 제공하지 못한다. 저자도 discussion에서 이 부분을 future work로 명시한다.

비판적으로 보면, 실험이 모두 sequence labeling 중심이라 방법의 범용성은 직관적으로는 높지만, 논문 내부 증거는 특정 NLP 태스크에 치우쳐 있다. 또 정확도 중심 평가를 사용했기 때문에, 일부 태스크에서 더 일반적으로 쓰이는 지표와의 비교가 부족하다고 느낄 수도 있다. 하지만 논문은 왜 accuracy를 사용했는지 이유를 분명히 밝혔다.

## 6. 결론

이 논문은 domain adaptation을 위해 놀랄 만큼 단순한 방법을 제안한다. 핵심은 각 feature를 **general**, **source-specific**, **target-specific** 버전으로 복제하여, source와 target 데이터를 서로 다른 방식으로 확장된 feature space에 매핑하는 것이다. 이렇게 하면 별도의 복잡한 adaptation 알고리즘 없이도 기존 supervised learner가 공통 지식과 도메인 특수 지식을 함께 학습할 수 있다.

실험적으로 이 방법은 다양한 sequence labeling 태스크에서 매우 강한 baseline들과 당시의 state-of-the-art 방법에 대해 경쟁력 있거나 그보다 좋은 결과를 보였다. 특히 target 데이터가 조금 있지만 충분하지 않은, 전형적인 fully supervised adaptation 환경에서 강력한 성능을 보인다.

이 연구의 중요한 의의는, domain adaptation이 반드시 복잡한 모델 설계를 필요로 하지 않을 수 있음을 보여준 데 있다. 좋은 표현 설계만으로도 강력한 적응 효과를 얻을 수 있다는 점은 이후 representation learning과 transfer learning 연구 전반에 큰 시사점을 준다. 실제 응용 측면에서도 구현이 매우 쉬워, NLP뿐 아니라 다른 supervised learning 문제에서 빠르게 시험해볼 수 있는 실용적 기법으로 가치가 크다.

향후 연구 관점에서는, 이 방법이 왜 잘 작동하는지에 대한 더 강한 이론적 설명, 도메인 간 유사도를 나타내는 하이퍼파라미터 $\alpha$ 같은 일반화, 그리고 semi-supervised 혹은 unsupervised adaptation으로의 확장이 중요한 후속 과제가 될 것이다. 논문 자체도 이러한 방향을 명시적으로 남겨 두고 있다. 전체적으로 이 논문은 “단순한 아이디어가 얼마나 강력할 수 있는가”를 보여주는 대표적인 domain adaptation 연구라고 평가할 수 있다.
