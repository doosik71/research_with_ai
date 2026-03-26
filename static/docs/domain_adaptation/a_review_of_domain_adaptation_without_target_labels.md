# A review of domain adaptation without target labels

* **저자**: Wouter M. Kouw, Marco Loog
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1901.05335](https://arxiv.org/abs/1901.05335)

## 1. 논문 개요

이 논문은 **target domain에 label이 전혀 없는 상황**에서, source domain의 labeled data와 target domain의 unlabeled data만으로 어떻게 target domain에서 잘 일반화되는 classifier를 만들 수 있는지를 체계적으로 정리한 **survey/review 논문**이다. 즉, 새로운 알고리즘 하나를 제안하는 논문이라기보다, 기존 연구들을 공통 원리와 가정에 따라 재구성하고, 어떤 조건에서 domain adaptation이 가능하거나 실패하는지를 이론과 함께 설명하는 데 목적이 있다.

논문이 다루는 핵심 문제는 매우 현실적이다. 실제 데이터는 거의 항상 training environment와 deployment environment가 다르다. 병원마다 의료영상 장비가 다르고, 카메라나 센서가 다르고, 문서 도메인마다 단어 분포가 다르며, 시뮬레이터에서 만든 로봇 데이터와 실제 환경 데이터도 다르다. 이런 경우 source에서 학습한 분류기를 그대로 target에 적용하면 성능이 크게 떨어질 수 있다. 이 논문은 바로 이런 **distribution shift** 하에서의 학습을 domain adaptation의 관점으로 정리한다.

저자들은 문제를 단순히 “분포가 다르다”로만 보지 않는다. 어떤 방식으로 적응하는지가 중요하다고 보고, 기존 방법들을 크게 세 가지로 분류한다. 첫째는 **sample-based methods**로, source sample마다 target에 대한 중요도를 다르게 주는 방식이다. 둘째는 **feature-based methods**로, source와 target이 더 비슷해지도록 feature space를 바꾸거나 공통 representation을 학습하는 방식이다. 셋째는 **inference-based methods**로, 모델 파라미터 추정이나 추론 규칙 자체에 adaptation을 직접 넣는 방식이다. 이 분류는 단순한 정리가 아니라, 이후 이론적 가정과 일반화 보장 조건을 연결하는 역할을 한다.

이 논문의 중요성은 두 가지다. 하나는 개별 논문들을 나열하는 대신, domain adaptation의 핵심 가정들이 무엇이고 각 가정이 어떤 형태의 error bound를 가능하게 하는지 보여준다는 점이다. 다른 하나는 “label 없는 target adaptation에는 no free lunch가 있다”는 메시지를 분명히 한다는 점이다. 즉, target label이 없으므로 어떤 방법도 항상 성공할 수 없고, 반드시 어떤 구조적 가정이 필요하다는 점을 강조한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **domain adaptation 방법들을 “무엇을 바꾸는가”에 따라 재조직하는 것**이다. 많은 기존 survey는 application 영역이나 알고리즘 family 중심으로 정리하는 반면, 이 논문은 adaptation이 작동하는 위치를 기준으로 본다. 즉, 개별 sample의 영향력을 바꾸는가, feature representation 자체를 바꾸는가, 아니면 parameter inference 과정 자체를 바꾸는가로 나눈다. 이 관점은 여러 방법들 사이의 공통 구조를 드러내는 데 유용하다.

논문이 제시하는 중요한 직관은 다음과 같다. source와 target의 차이는 단순히 nuisance가 아니라, classifier가 target에서 실패하는 직접적 원인이다. 그러므로 adaptation은 결국 다음 중 하나를 해야 한다.
첫째, source data를 target처럼 보이게 다시 가중한다.
둘째, source와 target을 공통된 표현 공간으로 옮긴다.
셋째, 불확실한 target 상황을 추론 과정에 직접 반영해 더 보수적이거나 더 안정적인 estimator를 만든다.

또 다른 핵심 메시지는 **가정과 일반화 보장의 연결**이다. 예를 들어 covariate shift를 가정하면 importance weighting이 정당화되고, 특정한 invariant component가 존재한다고 가정하면 domain-invariant representation 학습이 이론적으로 뒷받침된다. robust partition 가정을 두면 robustness 기반 bound가 가능하고, source가 덜 informative한 영역의 risk가 작다고 가정하면 PAC-Bayes 계열 bound가 가능해진다. 즉, 방법은 다양하지만, 결국 성능 보장은 특정 가정 위에서만 성립한다.

기존 접근과의 차별점은 이 논문이 deep method만 강조하지 않는다는 데 있다. 오히려 deep adversarial adaptation도 더 큰 분류 체계의 일부로 다루며, 고전적인 importance weighting, subspace alignment, optimal transport, empirical Bayes, PAC-Bayes까지 함께 놓고 비교한다. 그래서 분야를 넓게 조망하게 해 주는 장점이 있다.

## 3. 상세 방법 설명

이 논문은 survey이므로 하나의 단일 파이프라인을 제안하지 않는다. 대신 다음과 같은 이론적 프레임과 방법군을 제시한다.

### 3.1 문제 설정과 기본 정의

논문은 source domain과 target domain을 같은 feature-label space 위의 서로 다른 확률분포로 정의한다.
source domain은 $p_{\mathcal S}(x,y)$, target domain은 $p_{\mathcal T}(x,y)$이다.
source에서는 labeled samples $(x_i, y_i)$가 있고, target에서는 unlabeled samples $z_j$만 있다.

목표는 target sample의 label을 맞추거나, 더 일반적으로 target distribution에서 새로 나오는 sample의 label을 잘 예측하는 classifier $h$를 학습하는 것이다. 이때 risk는 다음처럼 정의된다.

$$
R(h) = \mathbb{E}[\ell(h(x), y)]
$$

여기서 $\ell$은 loss function이다. error는 $0/1$ loss에 대한 특수한 경우다.

이 논문은 특히 **single-source / single-target / no target labels** 설정만 다룬다. multi-source adaptation이나 semi-supervised domain adaptation은 범위 밖이라고 명시한다.

### 3.2 도메인 차이 측정

domain adaptation에서는 source와 target이 얼마나 다른지를 수치화하는 discrepancy measure가 중요하다. 논문은 여러 분포 거리 중 특히 두 가지를 자주 사용한다.

첫째는 **$\mathcal H \Delta \mathcal H$ divergence**이다.

$$
\mathrm{D}_{\mathcal{H}\Delta\mathcal{H}}[p_{\mathcal S}, p_{\mathcal T}] = 2 \sup_{h,h' \in \mathcal H} \left| \Pr_{\mathcal S}[h(x)\neq h'(x)] - \Pr_{\mathcal T}[h(x)\neq h'(x)] \right|
$$

이 값은 두 classifier가 source와 target에서 얼마나 다르게 disagreement하는지를 본다. 값이 크면 두 domain이 classifier 관점에서 많이 다르다는 뜻이다.

둘째는 **Rényi divergence**이다.

$$
\mathrm{D}_{\mathcal R^\alpha}[p_{\mathcal T}, p_{\mathcal S}] = \frac{1}{\alpha-1}\log_2 \int_{\mathcal X} \frac{p_{\mathcal T}^\alpha(x)}{p_{\mathcal S}^{\alpha-1}(x)}dx
$$

이 divergence는 importance weighting 분석에서 자주 등장한다. source와 target의 density ratio가 얼마나 극단적인지를 반영한다고 이해하면 된다.

### 3.3 일반화 오차 관점

논문은 adaptation이 없을 때조차 target error가 무엇에 의해 제한되는지를 먼저 설명한다. source classifier의 target generalization error는 대략 다음 세 요소에 의해 결정된다.

1. source와 target 둘 다에서 잘 작동하는 이상적인 joint hypothesis의 error
2. 두 domain 간 discrepancy
3. hypothesis class complexity와 sample size

대표적으로 다음과 같은 bound를 소개한다.

$$
e_{\mathcal T}(\hat h_{\mathcal S}) - e_{\mathcal T}(h^*_{\mathcal T})
\le
2e^*_{\mathcal S,\mathcal T}
+
\mathrm{D}_{\mathcal H \Delta \mathcal H}(p_{\mathcal S}, p_{\mathcal T})
+
4\sqrt{
\frac{2}{n}
\left(
\nu \log(2(n+1)) + \log \frac{8}{\delta}
\right)
}
$$

이 식의 의미는 간단하다. source에서 잘 학습했다고 해서 target에서도 잘되는 것이 아니다. domain discrepancy가 크면 bound가 커지고, source와 target을 동시에 설명할 수 있는 classifier 자체가 없다면 adaptation도 어렵다.

### 3.4 Sample-based methods

이 계열은 source sample을 그대로 쓰지 않고, target에 더 relevant한 sample이 더 크게 반영되도록 **weighting**하는 방식이다.

#### 3.4.1 Data importance-weighting

이 방법은 **covariate shift**를 가정한다. 즉,

$$
p_{\mathcal S}(y|x) = p_{\mathcal T}(y|x)
$$

는 유지되지만,

$$
p_{\mathcal S}(x) \neq p_{\mathcal T}(x)
$$

일 수 있다고 본다. 그러면 target risk는 source distribution 위의 weighted expectation으로 바꿀 수 있다.

$$
R_{\mathcal T}(h) = \sum_y \int_{\mathcal X} \ell(h(x), y) \frac{p_{\mathcal T}(x,y)}{p_{\mathcal S}(x,y)} p_{\mathcal S}(x,y)dx
$$

covariate shift 가정으로 posterior를 소거하면 weight는

$$
w(x) = \frac{p_{\mathcal T}(x)}{p_{\mathcal S}(x)}
$$

가 된다. 즉, target에서 자주 나오고 source에서는 드문 sample일수록 weight가 커진다.

이 접근의 핵심은 weight 추정이다. 논문은 세 부류를 설명한다.

첫째, **indirect estimation**은 source와 target의 density를 각각 추정한 뒤 ratio를 취한다. Gaussian 같은 parametric model이나 kernel density estimation을 사용할 수 있다.
둘째, **direct estimation**은 density ratio를 직접 최적화한다. 대표적으로 KMM, KLIEP, Least-Squares Importance Fitting이 있다.
셋째, classifier와 weight를 동시에 학습하는 방법도 있다.

예를 들어 **Kernel Mean Matching (KMM)**은 weighted source와 target의 MMD를 줄이는 방식이다.

$$
\mathrm{D}_{\text{MMD}}[w,p_{\mathcal S}(x),p_{\mathcal T}(x)] = |\mathbb{E}_{\mathcal S}[w\phi(x)] - \mathbb{E}_{\mathcal T}[\phi(x)]|_{\mathcal H}
$$

실제로는 sample weight들의 quadratic program으로 풀린다.

논문은 이 계열의 중요한 한계도 강조한다. domain이 너무 다르면 weight variance가 커지고, 몇 개 sample에 학습이 과도하게 집중될 수 있다. 또한 expected squared weight가 유한해야 수렴성이 보장된다. 다시 말해 support mismatch가 심하면 importance weighting은 불안정해진다.

#### 3.4.2 Class importance-weighting

이 방법은 **prior shift / label shift**를 가정한다. 즉,

$$
p_{\mathcal S}(x|y) = p_{\mathcal T}(x|y)
$$

는 유지되지만 class prior는 달라질 수 있다고 본다. 그러면 weight는

$$
w(y) = \frac{p_{\mathcal T}(y)}{p_{\mathcal S}(y)}
$$

가 된다.

문제는 target label이 없기 때문에 $p_{\mathcal T}(y)$를 직접 알 수 없다는 점이다. 그래서 BBSE처럼 source에서 confusion matrix를 추정하고, target의 predicted class frequency를 이용해 target prior를 역추정하는 방법이 소개된다. 이 계열은 class imbalance나 cost-sensitive learning과도 연결된다.

### 3.5 Feature-based methods

이 계열은 source를 target처럼 보이게 변환하거나, source와 target을 공통 representation으로 보내는 방식이다.

#### 3.5.1 Subspace mappings

핵심 가정은 domain-specific noise가 있지만 공통 subspace 구조가 있다는 것이다. 대표적인 예가 **Subspace Alignment**다. source와 target 각각에서 PCA basis를 구해 $C_{\mathcal S}, C_{\mathcal T}$라 하면, 정렬 행렬은

$$
M = C_{\mathcal S}^\top C_{\mathcal T}
$$

가 된다. source를 source subspace에 투영한 뒤, $M$으로 target subspace 방향에 맞게 정렬해 학습한다.

이 계열은 manifold 관점으로 확장된다. 예를 들어 **Geodesic Flow Kernel**은 source subspace에서 target subspace까지의 중간 subspace들을 모두 고려한다. 각 중간 subspace의 projection을 적분해 하나의 kernel을 만든다.

#### 3.5.2 Optimal transport

optimal transport는 source와 target을 probability measure로 보고, source mass를 target mass로 옮기는 최소 비용 plan을 찾는다. Wasserstein distance가 핵심이다.

$$
\mathrm{D}_{\mathcal W}[p_{\mathcal S}(x), p_{\mathcal T}(x)] = \inf_{\gamma \in \Gamma} \int_{\mathcal X \times \mathcal X} d(x,z), d\gamma(x,z)
$$

여기서 $\gamma$는 source와 target marginal을 연결하는 coupling이다. 실제로는 empirical distribution 위에서 transport matrix를 구하고, barycentric mapping으로 source sample을 target 쪽으로 이동시킨다.

optimal transport의 장점은 graph structure나 joint distribution structure를 regularization에 자연스럽게 넣을 수 있다는 점이다. 또한 Wasserstein distance 기반의 generalization 분석도 가능하다.

#### 3.5.3 Domain-invariant spaces

이 계열의 목표는 source와 target을 아예 **domain-invariant representation**으로 보내는 것이다. 단순히 source를 target으로 맞추는 것이 아니라, 둘 다 새로운 space로 보내 그 공간에서는 domain 차이가 줄어들도록 한다.

논문은 **Conditional Invariant Components** 개념을 소개한다. 변환 $t(x)$를 통해 얻어진 성분이 각 class에 대해

$$
p_{\mathcal S}(t(x)|y) = p_{\mathcal T}(t(x)|y)
$$

를 만족하면, 그 성분은 class-conditional invariant하다고 본다. 이때 transformed source와 transformed target 사이의 MMD가 작으면 transformed source error와 target error의 차이를 제어할 수 있다.

대표 방법으로 **Transfer Component Analysis (TCA)**가 나온다. joint kernel 위에서 domain mean discrepancy를 줄이는 방향으로 component를 찾는다. 목적은 trace minimization 형태다.

$$
\underset{C}{\text{minimize}}
\quad
\mathrm{tr}(C^\top K L K C)
\quad
\text{s.t.}
\quad
C^\top K H K C = I
$$

이는 trivial solution을 피하면서 domain discrepancy가 작은 projection을 학습하려는 것이다.

또 다른 방법으로 DME, DICA 등이 있으며, 일부는 MMD kernel 자체를 학습한다.

#### 3.5.4 Deep domain adaptation

deep learning에서는 autoencoder 기반 방법과 adversarial 방법이 대표적이다.

가장 유명한 예는 **DANN (Domain-Adversarial Neural Network)**이다. 하나의 feature extractor 위에 두 개의 loss를 둔다. 하나는 source label classification loss, 다른 하나는 source-vs-target domain classification loss다. 학습 시 feature extractor는 label prediction에는 유리하고 domain discrimination에는 불리한 representation을 만들도록 업데이트된다.

이 논문은 DANN의 핵심 직관을 **proxy $\mathcal A$-distance**로 설명한다.

$$
\mathrm{D}_{\mathcal A}[x,z] = 2(1 - 2\hat e(x,z))
$$

여기서 $\hat e(x,z)$는 source와 target을 구분하는 classifier의 cross-validation error다. domain classifier가 source와 target을 잘 구분하지 못할수록, 두 domain이 representation space에서 더 비슷하다고 본다.

다만 논문은 중요한 한계를 지적한다. 데이터 marginal을 맞춘다고 해서 class-conditional까지 자동으로 맞는 것은 아니다. 그래서 adversarial alignment가 항상 올바른 semantic alignment를 보장하지는 않는다.

#### 3.5.5 Correspondence learning

이 부분은 주로 NLP의 high-dimensional sparse feature를 다룬다. source에서 sentiment를 잘 나타내는 단어가 target에는 없을 수 있다. 그러면 두 domain에 공통으로 자주 등장하는 pivot word를 기준으로 correlated features를 찾아, 서로 대응되는 feature를 구성한다. Structural Correspondence Learning이 대표적이다. 이 접근은 domain-specific vocabulary 문제를 feature correspondence 문제로 바꾼다.

### 3.6 Inference-based methods

이 계열은 adaptation을 feature나 sample 단계가 아니라 **estimation/inference 단계에 직접 집어넣는다.**

#### 3.6.1 Algorithmic robustness

robust algorithm은 feature space를 여러 region으로 나누고, 각 region 안에서는 sample 하나가 빠져도 loss 변화가 작도록 한다. domain shift를 training-to-test sample replacement로 해석하면, robust classifier는 target에서도 안정적일 수 있다.

논문은 posterior shift를 완화된 형태로 표현하는 **$\lambda$-shift**를 소개한다. $\lambda=0$이면 posterior가 완전히 같고, $\lambda=1$이면 크게 다를 수 있다. 이 관점은 strict covariate shift보다 덜 강한 가정이다.

#### 3.6.2 Minimax estimators

이 방법은 target uncertainty를 adversary로 모델링한다. classifier는 worst-case target condition에서도 risk가 작도록 학습한다.

예를 들어 Robust Bias-Aware classifier는 adversary가 target posterior를 고르되 source feature statistics와 일치해야 한다는 제약 아래 risk를 최대화하고, classifier는 그 최대 risk를 최소화한다.

또 **Target Contrastive Robust (TCR)** 계열은 source classifier보다 target risk를 확실히 줄일 수 있는 방향으로만 adaptation한다. target soft label의 worst-case를 상정해도 source classifier보다 더 나쁜 쪽으로 가지 않게 설계하는 것이 핵심이다. 이는 negative transfer를 명시적으로 피하려는 접근으로 이해할 수 있다.

#### 3.6.3 Self-learning

self-learning은 source로 초기 classifier를 만든 뒤, target에 pseudo-label을 붙여 다시 학습하는 iterative 방법이다. DASVM, co-training, EM 기반 Ad-REM, BDA 등이 소개된다.

이 방법의 핵심 난점은 어떤 pseudo-label이 신뢰할 만한가이다. DASVM은 margin 근처 sample을 점진적으로 교체하고, Ad-REM은 class balance를 강제해 예측 불안정을 줄이며, BDA는 pseudo-label을 이용해 conditional distribution까지 맞추는 방향으로 확장한다.

논문은 self-learning의 실용적 장점도 말한다. pseudo-labeled target을 validation처럼 써서 hyperparameter tuning을 하는 **reverse validation** 전략이 가능하다는 점이다.

#### 3.6.4 Empirical Bayes

Bayesian 관점에서는 source domain을 prior knowledge로 본다. 즉, source로 prior distribution을 추정하고, target unlabeled 혹은 제한된 구조 정보와 함께 posterior inference를 수행한다. 예를 들어 NLP에서 단어 간 상관구조를 source 대규모 corpus로 추정해 prior covariance에 넣을 수 있다.

이 방법의 장점은 source가 parameter search space를 줄여준다는 데 있다. 반대로 source와 target이 너무 다르면 informative prior가 오히려 잘못된 방향으로 posterior를 끌어 negative effect를 낼 수 있다.

#### 3.6.5 PAC-Bayes

PAC-Bayes는 hypothesis space 위의 prior $\pi$와 posterior $\rho$를 두고, posterior에서 뽑은 classifier들의 disagreement와 source error를 함께 고려한다. 논문은 domain discrepancy를 hypothesis disagreement 차이로 정의한다.

$$
\mathrm{D}_\rho[p_{\mathcal T}, p_{\mathcal S}] = |d_{\mathcal S}(\rho) - d_{\mathcal T}(\rho)|
$$

또 새로운 bound에서는 source가 informative하지 않은 target 영역의 risk $\eta_{\mathcal T \setminus \mathcal S}$를 따로 둔다. 이를 계산 가능한 선형 분류기 형태로 내린 것이 **DALC**다. 최종 objective는 대략 target disagreement, source classification error, parameter norm의 trade-off이다.

$$
\hat{\theta} = \arg\min_\theta , \tau_1 \sum_{j=1}^m \tilde{\Phi}!\left(\frac{z_j\theta}{|z_j|}\right) + \tau_2 \sum_{i=1}^n \Phi^2!\left(\frac{y_i x_i\theta}{|x_i|}\right) + |\theta|^2
$$

직관적으로는 target에서 지나치게 확신하는 결정을 줄이고, source error와 complexity를 함께 통제하는 형태다.

## 4. 실험 및 결과

이 논문은 **새로운 단일 모델의 실험 논문이 아니라 review 논문**이므로, 하나의 통합 benchmark에서 특정 방법이 최고 성능을 냈다는 식의 실험 결과를 제시하지 않는다. 따라서 일반적인 연구 논문처럼 “데이터셋-기준선-정량 성능표” 중심의 실험 섹션을 기대하면 안 된다. 이 점은 명시적으로 이해해야 한다.

대신 논문은 여러 응용 영역에서 domain adaptation이 왜 중요한지를 motivating example로 보여준다. 예를 들면 의료영상에서 병원 간 scanner 차이, computer vision에서 dataset bias, robotics에서 simulation-to-real gap, speech recognition에서 speaker variation, NLP에서 domain-specific word distribution, bioinformatics에서 organism 차이, astronomy에서 selection bias 등이 소개된다. 이 예시들은 domain adaptation이 단순히 이론적 문제가 아니라, 실제로 매우 다양한 scientific/engineering setting에서 나타난다는 점을 설득력 있게 전달한다.

또한 이 논문은 “실험 결과” 대신 **이론적 결과와 방법적 비교**를 중심으로 독자가 실질적인 판단을 할 수 있게 한다. 예를 들어 importance weighting은 covariate shift와 support overlap이 어느 정도 있어야 하고, deep adversarial 방법은 marginal matching만으로 class-conditional alignment를 보장하지 못하며, minimax나 PAC-Bayes 계열은 보수적으로 negative transfer를 줄일 수 있다는 식의 실질적 비교 포인트를 제공한다.

다시 말해, 이 논문의 결과는 “새로운 SOTA 수치”가 아니라 다음과 같은 메타 수준의 결론에 가깝다.
첫째, unlabeled target adaptation은 반드시 가정에 의존한다.
둘째, 그 가정은 방법마다 다르며, 성공 조건도 다르다.
셋째, discrepancy를 줄인다고 해서 항상 target classification이 좋아지는 것은 아니다.
넷째, support mismatch와 class-conditional mismatch가 실제 실패의 핵심 원인이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **분야 전체를 구조적으로 정리한 점**이다. sample-based, feature-based, inference-based라는 분류는 단순하면서도 설명력이 높다. 많은 개별 방법이 사실상 어디를 조작하는지에 따라 이해될 수 있음을 보여주고, 이로 인해 독자는 새로운 논문을 읽을 때도 그 방법이 어느 계열에 속하는지 빠르게 파악할 수 있다.

두 번째 강점은 **가정 중심의 설명**이다. 저자들은 domain adaptation을 만능 해결책처럼 포장하지 않는다. 오히려 covariate shift, prior shift, invariant components, robustness partition, source-support assumption 같은 가정이 있어야만 error bound가 가능하다고 분명히 말한다. 이는 실무자나 연구자에게 매우 중요한 태도다. 특히 “가정을 검증할 target label이 없기 때문에 classifier selection이 본질적으로 어렵다”는 점을 솔직하게 다룬다.

세 번째 강점은 **이론과 직관의 균형**이다. 논문은 $\mathcal H \Delta \mathcal H$ divergence, Rényi divergence, Wasserstein distance, PAC-Bayes bound 등 수학적 도구를 다루지만, 왜 그런 quantity가 필요한지 직관도 함께 설명한다. 예를 들어 importance weight variance가 커지면 소수 sample에 학습이 쏠린다는 설명은 이론과 실패 사례를 잘 연결해 준다.

네 번째 강점은 **응용 영역의 폭**이다. computer vision, medical imaging, speech, NLP, robotics, bioinformatics, astronomy, fairness까지 연결하면서 domain adaptation의 범용성을 잘 보여준다. 특히 fairness를 distribution matching과 연결하는 부분은 흥미롭다.

하지만 한계도 분명하다.

첫째, 이 논문은 review이므로 각 방법의 실제 empirical ranking이나 실전 가이드라인을 강하게 제시하지는 못한다. 어떤 상황에서 어떤 방법을 우선 시도해야 하는지에 대해서는 통찰은 있지만, 구체적 선택 기준은 상대적으로 약하다.

둘째, “가정 검증의 어려움”을 강조하지만, 실무에서 쓸 수 있는 hypothesis test나 diagnosis protocol은 제한적으로만 논의된다. 즉, 문제의 핵심 난점은 잘 설명하지만, 완전한 해결책은 아니다.

셋째, deep domain adaptation을 포함하긴 하지만, 2019년 시점의 survey이므로 이후 발전한 contrastive adaptation, source-free adaptation, test-time adaptation, diffusion 기반 transfer, foundation model 시대의 adaptation 같은 후속 흐름은 포함되지 않는다. 물론 이는 논문의 잘못이라기보다 시점의 한계다.

넷째, feature matching에 대한 비판은 중요하지만, 실제로 어떤 조건에서 semantic misalignment가 심각해지는지에 대한 실증적 비교는 깊지 않다. 즉, “marginal matching is not enough”라는 메시지는 강하지만, 이를 체계적으로 분해하는 실험적 프레임은 제공하지 않는다.

비판적으로 보면, 이 논문의 가장 중요한 공헌은 “무엇이 잘 되는가”를 말하는 것보다 “왜 잘 안 될 수 있는가”를 정리한 데 있다. 이는 매우 건강한 기여지만, 초심자 입장에서는 다소 보수적으로 느껴질 수도 있다. 그러나 바로 그 점 때문에 이 논문은 domain adaptation 입문과 재정리에 매우 유용하다.

## 6. 결론

이 논문은 target label이 없는 domain adaptation을 하나의 통일된 시각으로 정리한다. 핵심 결론은 간단하다. **source에서 target으로의 일반화는 가능할 수 있지만, 반드시 어떤 구조적 가정이 필요하며, 그 가정이 어떤 형태인지에 따라 적절한 방법이 달라진다.**

저자들은 기존 방법을 세 갈래로 정리했다. sample-based 방법은 source sample의 중요도를 조절하고, feature-based 방법은 source와 target의 표현을 맞추거나 invariant space를 학습하며, inference-based 방법은 추론 절차 자체를 더 robust하거나 더 보수적으로 만든다. 이 정리는 단순한 taxonomy가 아니라, 각 방법이 기대는 이론적 조건과 실패 원인을 이해하는 데 직접 연결된다.

실제 적용 측면에서 이 논문이 주는 가장 중요한 교훈은 다음과 같다. 첫째, domain discrepancy가 작다고 해서 자동으로 class semantics가 맞는 것은 아니다. 둘째, support mismatch는 많은 방법의 근본적 한계다. 셋째, negative transfer를 피하려면 assumption test와 diagnostics가 매우 중요하다. 넷째, causal structure에 대한 이해가 향후 adaptation strategy selection에 큰 역할을 할 수 있다.

향후 연구 관점에서 보면, 이 논문은 세 방향을 강하게 시사한다. 하나는 가정의 타당성을 더 잘 진단하는 test 개발이다. 다른 하나는 더 해석 가능하고 실패 모드를 드러내는 adaptation 방법의 설계다. 마지막 하나는 causal information이나 environment variable을 활용해 어떤 shift가 발생했는지 더 명확히 식별하는 방향이다.

정리하면, 이 논문은 domain adaptation의 개별 기법을 배우기 위한 자료이면서 동시에, 이 분야를 과도하게 낙관적으로 보지 않도록 해 주는 균형 잡힌 survey다. 특히 “label 없는 target adaptation에는 보장이 아니라 가정이 먼저다”라는 메시지를 명확하게 남긴다는 점에서, 지금 읽어도 여전히 가치가 크다.
