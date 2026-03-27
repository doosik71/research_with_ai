# Unsupervised Domain Adaptation Based on Source-guided Discrepancy

* **저자**: Seiichi Kuroki, Nontawat Charoenphakdee, Han Bao, Junya Honda, Issei Sato, Masashi Sugiyama
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1809.03839](https://arxiv.org/abs/1809.03839)

## 1. 논문 개요

이 논문은 unsupervised domain adaptation에서 source domain과 target domain 사이의 차이를 어떻게 측정할 것인가라는 핵심 문제를 다룬다. 설정은 전형적이다. source domain에는 라벨이 있는 데이터가 있고, target domain에는 입력만 있으며 라벨은 없다. 목표는 source의 감독 정보를 활용해 target에서 잘 동작하는 classifier를 학습하는 것이다.

문제는 source와 target의 분포가 다르면, source에서 잘 작동하는 모델이 target에서 성능이 급격히 떨어질 수 있다는 점이다. 따라서 두 도메인의 차이를 정량화하는 discrepancy measure가 필요하다. 기존 연구에는 maximum mean discrepancy, KL divergence, Rényi divergence, Wasserstein distance 같은 일반적인 분포 거리뿐 아니라, hypothesis class를 반영하는 domain adaptation 전용 discrepancy도 있었다. 그러나 저자들은 기존 방법들에 두 가지 큰 한계가 있다고 본다. 첫째, 이론적 보장은 있으나 계산이 매우 비싸다. 둘째, 계산은 쉽지만 target 일반화 성능에 대한 이론적 보장이 없다. 셋째, 어떤 방법은 target label이 필요해 unsupervised setting에 쓸 수 없다.

이 논문의 핵심 목표는 이런 한계를 동시에 줄이는 새로운 discrepancy measure를 제안하는 것이다. 저자들은 이를 **source-guided discrepancy**, 줄여서 **S-disc**라고 부른다. 이 measure는 source label 정보를 적극적으로 사용한다. 그 결과, target label이 없어도 계산 가능하고, 기존의 $\mathcal{X}$-disc보다 더 타이트한 일반화 오차 경계를 제공하며, 계산도 훨씬 효율적으로 할 수 있다고 주장한다.

즉, 이 논문은 단순히 새 척도를 제안하는 수준이 아니라, 다음 세 가지를 함께 달성하려고 한다. 첫째, unsupervised domain adaptation에 맞는 discrepancy 정의를 제시한다. 둘째, binary classification의 $0$-$1$ loss에 대해 실제 계산 가능한 알고리즘을 준다. 셋째, finite-sample consistency와 generalization bound를 통해 이 척도의 이론적 타당성을 보인다. 마지막으로 실험을 통해 기존 척도보다 더 좋은 source selection 신호를 준다는 점을 입증하려 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 기존 discrepancy가 “너무 보수적”이라는 문제의식에서 출발한다. 대표적인 기존 척도인 $\mathcal{X}$-disc는 다음처럼 정의된다.

$$
\mathrm{disc}_{\mathcal{H}}^{\ell}(P_T, P_S) = \sup_{h,h' \in \mathcal{H}} \left| R_T^{\ell}(h,h') - R_S^{\ell}(h,h') \right|
$$

여기서는 hypothesis 두 개 $h, h'$를 모두 최악의 경우로 고른다. 즉, 도메인 차이를 “가장 나쁜 두 classifier 쌍” 기준으로 측정한다. 저자들은 이것이 두 가지 문제를 만든다고 본다. 하나는 generalization bound가 느슨해질 수 있다는 것이고, 다른 하나는 계산량이 커진다는 것이다. 특히 hypothesis pair 전체를 뒤져야 하므로 실제 계산이 어렵다.

반면 S-disc는 이 두 함수 중 하나를 임의의 최악의 hypothesis가 아니라, **source domain에서의 true risk minimizer** $h_S^*$로 고정한다. 즉, source에서 가장 잘 맞는 분류기와 다른 hypothesis를 비교해 도메인 차이를 측정한다. 정의는 다음과 같다.

$$
\varsigma_{\mathcal{H}}^{\ell}(P_{D_1}, P_{D_2}) = \sup_{h \in \mathcal{H}} \left| R_{D_1}^{\ell}(h,h_S^*) - R_{D_2}^{\ell}(h,h_S^*) \right|
$$

이렇게 하면 두 가지 직관이 생긴다. 첫째, domain adaptation에서 실제로 중요한 것은 source에서 의미 있는 classifier가 target에서도 비슷하게 작동하는지이지, hypothesis class 전체에서 아무 두 classifier나 골라 최악 차이를 보는 것이 아니다. 둘째, source label을 활용하면 source에서 좋은 classifier를 anchor처럼 잡을 수 있다. 그래서 보다 task-relevant한 discrepancy를 얻을 수 있다.

저자들이 강조하는 차별점은 명확하다. 기존 $d_{\mathcal{H}}$는 계산은 쉽지만 label 정보를 쓰지 않아 좋은 source와 나쁜 source를 구분하지 못할 수 있다. 반면 $\mathcal{Y}$-disc는 이론적으로 더 직접적이지만 target label이 필요해 unsupervised setting에서는 쓸 수 없다. S-disc는 source label만으로 계산되며, target label 없이도 task-aware discrepancy를 측정한다. 즉, **unsupervised setting에서 쓸 수 있으면서, 계산 가능하고, 이론적으로도 더 타이트한 bound를 주는 절충안**이 아니라 오히려 더 적합한 measure로 제시된다.

핵심 부등식도 이 논문의 직관을 잘 드러낸다.

$$
\left| R_T^{\ell}(h,h_S^*) - R_S^{\ell}(h,h_S^*) \right| \le \varsigma_{\mathcal{H}}^{\ell}(P_T,P_S) \le \mathrm{disc}_{\mathcal{H}}^{\ell}(P_T,P_S)
$$

즉, S-disc는 $\mathcal{X}$-disc보다 항상 작거나 같다. 따라서 같은 구조의 generalization bound 안에 들어갈 때, S-disc를 쓰면 더 타이트한 bound를 기대할 수 있다.

## 3. 상세 방법 설명

논문은 먼저 unsupervised domain adaptation 문제를 형식화한다. 각 도메인은 입력 분포와 labeling function의 쌍 $(P_D, f_D)$로 표현된다. source domain은 $(P_S, f_S)$, target domain은 $(P_T, f_T)$이다. 주어진 데이터는 source의 labeled sample $\mathcal{S}={(x_j^S, y_j^S)}_{j=1}^{n_S}$와 target의 unlabeled sample $\mathcal{T}={x_i^T}_{i=1}^{n_T}$이다.

학습 목표는 hypothesis class $\mathcal{H}$에서 target risk

$$
R_T^{\ell}(h,f_T) = \mathbb{E}_{x \sim P_T}[\ell(h(x), f_T(x))]
$$

를 작게 만드는 $h$를 찾는 것이다.

### S-disc의 정의와 성질

제안된 S-disc는 source의 최적 hypothesis $h_S^*$를 기준점으로 쓴다.

$$
h_S^* = \arg\min_{h \in \mathcal{H}} R_S^{\ell}(h,f_S)
$$

이를 이용해 S-disc를 다음처럼 정의한다.

$$
\varsigma_{\mathcal{H}}^{\ell}(P_{D_1},P_{D_2}) = \sup_{h \in \mathcal{H}} \left| R_{D_1}^{\ell}(h,h_S^*) - R_{D_2}^{\ell}(h,h_S^*) \right|
$$

저자들은 이 measure가 triangular inequality와 symmetry는 만족하지만, 일반적으로 엄밀한 distance는 아니라고 설명한다. 즉, 서로 다른 두 분포라도 S-disc가 0이 될 수 있다. 이는 S-disc가 “분포 전체의 차이”보다는 “source-optimal classifier 기준으로 봤을 때 task-relevant한 차이”를 측정하기 때문이다.

### $0$-$1$ loss에서의 계산 가능 형태

논문의 중요한 공헌은 binary classification과 symmetric hypothesis class에서 S-disc를 효율적으로 계산하는 공식을 제시한 점이다. 여기서 symmetric hypothesis class란 $h \in \mathcal{H}$이면 $-h \in \mathcal{H}$도 포함하는 경우다.

Theorem 2에 따르면 empirical S-disc는 다음과 같이 쓸 수 있다.

$$
\varsigma_{\mathcal{H}}^{\ell_{01}}(\widehat{P}_T,\widehat{P}_S) = 1 - \min_{h \in \mathcal{H}} J_{\ell_{01}}(h)
$$

여기서

$$
J_{\ell}(h) = \frac{1}{n_S}\sum_{j=1}^{n_S}\ell(h(x_j^S), h_S^*(x_j^S)) + \frac{1}{n_T}\sum_{i=1}^{n_T}\ell(h(x_i^T), -h_S^*(x_i^T))
$$

이다.

이 식의 의미는 매우 중요하다. source 샘플에는 $h_S^*$가 예측한 label을 붙이고, target 샘플에는 그 반대 부호의 pseudo-label을 붙인다. 그러면 S-disc 계산이 “source와 target을 서로 반대 label로 분리하는 cost-sensitive classification” 문제로 바뀐다. 다시 말해, source 쪽은 $h_S^*$와 같은 방향으로 맞추고, target 쪽은 $h_S^*$와 반대 방향으로 맞추도록 하는 classifier를 찾는 문제다. 이 최적값이 클수록 두 도메인을 $h_S^*$ 기준으로 더 다르게 본다는 뜻이고, 따라서 discrepancy가 크다.

### Algorithm 1의 절차

논문은 이 결과를 바탕으로 3단계 알고리즘을 제안한다.

첫 단계는 **source learning**이다. source labeled data로 classifier $\widehat{h}_S$를 학습한다. 실제로는 $h_S^*$를 직접 알 수 없으므로 empirical risk minimizer인 $\widehat{h}_S$를 사용한다.

둘째 단계는 **pseudo labeling**이다. source 입력들에는 $\mathrm{sign}(\widehat{h}_S(x))$를 붙이고, target 입력들에는 $-\mathrm{sign}(\widehat{h}_S(x))$를 붙인다. 즉,

$$
\widetilde{\mathcal{S}} = {(x,\mathrm{sign}\circ\widehat{h}_S(x)) \mid x \in \mathcal{S}_{\mathcal{X}}}
$$

$$
\widetilde{\mathcal{T}} = {(x,-\mathrm{sign}\circ\widehat{h}_S(x)) \mid x \in \mathcal{T}}
$$

를 만든다.

셋째 단계는 **cost-sensitive learning**이다. 새 classifier $h''$를 학습하여 pseudo-labeled source와 pseudo-labeled target을 비용 가중 하에 분류한다. 이때 source 쪽 가중치는 $1/n_S$, target 쪽은 $1/n_T$이다. 그리고 마지막에

$$
\varsigma_{\mathcal{H}}^{\ell}(\widehat{P}_T,\widehat{P}_S) = 1 - J_{\ell_{01}}(h'')
$$

를 반환한다.

### surrogate loss와 계산 복잡도

$0$-$1$ loss 직접 최소화는 계산적으로 어렵기 때문에, 실제 학습에는 hinge loss를 surrogate로 쓴다.

$$
\ell_{\text{hinge}}(y,y') = \max(0, 1 - yy')
$$

중요한 점은 surrogate loss는 학습 단계에만 쓰이고, 최종 S-disc 값 자체는 여전히 $0$-$1$ loss 기반 정의에 따라 계산된다는 것이다. 저자들은 SVM의 SMO 알고리즘을 쓰면 전체 계산 복잡도가 $O((n_T+n_S)^3)$라고 설명한다.

이는 기존 $\mathcal{X}$-disc의 계산보다 크게 유리하다. 논문 부록에서는 $\mathcal{X}$-disc를 hinge loss와 semidefinite relaxation으로 계산하는 방법도 제시하지만, 그 복잡도는 $O((n_T+n_S+d)^8)$로 훨씬 크다. 따라서 실용적인 관점에서 S-disc는 계산 가능한 이론적 discrepancy measure라는 위치를 갖는다.

### 이론 분석: consistency와 convergence

논문은 Rademacher complexity를 이용해 empirical S-disc estimator의 deviation bound를 제시한다. 핵심 결과는 다음과 같다.

손실 함수 $\ell$이 상수 $M$으로 upper bounded일 때, 확률 $1-\delta$ 이상으로

$$
\begin{aligned}
\left| \varsigma_{\mathcal{H}}^{\ell}(\widehat{P}_T,\widehat{P}_S) - \varsigma_{\mathcal{H}}^{\ell}(P_T,P_S) \right| \le & \; 2\mathfrak{R}_{P_T,n_T}(\ell \circ (\mathcal{H}\otimes\mathcal{H})) \\
& + 2\mathfrak{R}_{P_S,n_S}(\ell \circ (\mathcal{H}\otimes\mathcal{H})) \\
& + M\sqrt{\frac{\log \frac{4}{\delta}}{2n_T}} \\
& + M\sqrt{\frac{\log \frac{4}{\delta}}{2n_S}}
\end{aligned}
$$

가 성립한다.

이 결과는 empirical estimator가 sample 수가 늘면 true S-disc로 수렴한다는 뜻이다. 특히 linear-in-parameter model에 대해 Rademacher complexity가 $1/\sqrt{n}$ 오더로 제어되므로, $0$-$1$ loss의 경우 S-disc 추정 오차는

$$
\mathcal{O}_p\left(n_T^{-1/2} + n_S^{-1/2}\right)
$$

로 수렴한다. 이는 통계적으로 표준적인 좋은 수렴 속도다.

### 일반화 오차 경계

이 논문의 이론적 핵심은 target regret bound다. 손실이 triangle inequality를 만족하면, 임의의 hypothesis $h$에 대해

$$
R_T^{\ell}(h,f_T) - R_T^{\ell}(h_T^*,f_T) \le R_S^{\ell}(h,h_S^*) + R_T^{\ell}(h_S^*,h_T^*) + \varsigma_{\mathcal{H}}^{\ell}(P_T,P_S)
$$

가 성립한다.

이 bound의 세 항은 각각 의미가 분명하다. 첫째 항은 source에서 $h$가 source-optimal classifier와 얼마나 다른가를 본다. 둘째 항은 source-optimal classifier와 target-optimal classifier가 target에서 얼마나 다른가를 나타내는데, 이는 domain gap의 본질적 난이도에 해당한다. 셋째 항이 바로 source-target discrepancy이다.

기존 $\mathcal{X}$-disc 기반 bound는 마지막 항이 $\mathrm{disc}_{\mathcal{H}}^{\ell}(P_T,P_S)$로 바뀐 동일한 형태인데, S-disc가 항상 그보다 작거나 같으므로 더 타이트한 bound가 된다. 이 논문의 주장은 바로 이 지점에 있다. 즉, 단순히 계산만 쉬운 대체 척도가 아니라, **기존 theoretical bound의 구조를 유지하면서 마지막 discrepancy term을 더 작게 만들 수 있는 척도**라는 점이다.

Theorem 8에서는 finite-sample 버전도 제시한다. 거기서는 empirical source loss와 empirical S-disc, 그리고 $1/\sqrt{n_S}$, $1/\sqrt{n_T}$ 차수의 complexity term이 들어간다. 결과적으로 sample 수가 충분히 크면 실제로 중요한 항은 empirical source error, source-target optimal classifier 차이, empirical S-disc라는 해석을 제시한다.

## 4. 실험 및 결과

논문은 세 종류의 실험을 통해 S-disc를 평가한다. 첫 번째는 toy illustration, 두 번째는 computation time 비교, 세 번째는 empirical convergence와 source selection이다.

### 4.1 Toy example: 기존 $d_{\mathcal{H}}$의 실패 사례

첫 실험은 2차원 Gaussian toy data로 구성된다. source는 두 개 $S_1, S_2$이고 target 하나가 있으며, 각 클래스별로 200개 샘플을 생성한다. 분포 평균은 다음과 같이 설정된다.

* $S_1$의 class 0 평균은 $(-5,-5)$, class 1 평균은 $(5,-5)$
* $S_2$의 class 0 평균은 $(0,3)$, class 1 평균은 $(2,-3)$
* target의 class 0 평균은 $(-5,-3)$, class 1 평균은 $(5,-3)$

모든 분포의 공분산은 identity이다. 실험에는 linear kernel SVM을 사용한다.

결과는 다음과 같다.

* $S$-disc 기준: $\varsigma(\widehat{P}_T,\widehat{P}_{S_1}) = 0.27$, $\varsigma(\widehat{P}_T,\widehat{P}_{S_2}) = 0.49$
* $d_{\mathcal{H}}$ 기준: $d_{\mathcal{H}}(\widehat{P}_T,\widehat{P}_{S_1}) = 0.69$, $d_{\mathcal{H}}(\widehat{P}_T,\widehat{P}_{S_2}) = 0.49$

즉, $d_{\mathcal{H}}$는 $S_2$가 target에 더 가깝다고 판단하지만, S-disc는 $S_1$이 더 좋은 source라고 판단한다.

실제 target loss를 보면 $S_1$에서 학습한 classifier의 loss는 $0.0$이고, $S_2$에서 학습한 classifier의 loss는 $0.49$이다. 따라서 이 예제에서는 S-disc의 ranking이 실제 target 성능과 일치하고, $d_{\mathcal{H}}$는 잘못된 판단을 한다.

저자들의 해석은 분명하다. $d_{\mathcal{H}}$는 source와 target을 입력 공간에서 얼마나 잘 구분할 수 있는지만 보기 때문에, support가 조금만 떨어져 있어도 서로 완전히 다른 domain처럼 볼 수 있다. 하지만 실제 예측 경계나 risk minimizer가 비슷할 수 있다. 반면 S-disc는 source label 구조를 반영하기 때문에, classification task 관점에서 더 올바른 source quality를 측정한다.

### 4.2 Computation time 비교

두 번째 실험에서는 S-disc, $d_{\mathcal{H}}$, $\mathcal{X}$-disc의 계산 시간을 비교한다. 2차원 데이터, source와 target 각각 200개 샘플을 사용한다. $\mathcal{X}$-disc는 논문 부록의 semidefinite relaxation 방식으로 계산하고, S-disc와 $d_{\mathcal{H}}$는 linear SVM으로 계산한다.

결과는 Figure 2에서 제시되며, $\mathcal{X}$-disc는 계산 시간이 매우 커서 사실상 비실용적이다. 반면 S-disc와 $d_{\mathcal{H}}$는 둘 다 계산 가능하다. 논문은 이를 통해 “이론적으로 타당하면서 실제 계산도 가능한 discrepancy”라는 S-disc의 위치를 강조한다.

여기서 수치값이 본문에 구체적으로 적혀 있지는 않다. 따라서 정확한 초 단위 비교는 이 텍스트만으로는 알 수 없고, 저자들이 Figure 2를 통해 상대적 추세만 보여주었다고 보는 것이 정확하다.

### 4.3 Empirical convergence

세 번째 실험 중 하나는 MNIST 기반 empirical convergence 비교다. task는 odd/even binary classification이다. source domain은 두 개다.

* $S_1$: 원래 MNIST
* $S_2$: digit 0부터 7까지만 포함한 편향된 MNIST
* target: MNIST

각 도메인 샘플 수를 1,000, 2,000, ..., 20,000으로 늘려가며 discrepancy 추정값이 어떻게 변하는지 본다. logistic regression을 사용했다.

논문에 따르면 두 discrepancy 모두 $S_1$이 $S_2$보다 더 나은 source라고 판단한다. 그러나 중요한 차이는 수렴 양상이다. $S_1$은 target과 사실상 같은 분포이므로 이상적으로 discrepancy가 0에 가까워져야 한다. S-disc는 이에 더 잘 맞게 수렴하지만, $d_{\mathcal{H}}$는 훨씬 느리게 수렴하거나 0이 아닌 값에 머무는 경향을 보인다. 이는 S-disc가 실제로 더 안정적으로 도메인 차이를 포착한다는 경험적 근거로 제시된다.

### 4.4 Source selection

가장 실용적인 실험은 source selection이다. source domain은 10개이며, 그중 5개는 clean grayscale MNIST-M이고, 나머지 5개는 Gaussian noise를 섞은 noisy grayscale MNIST-M이다. target은 MNIST다. task는 역시 odd/even classification이고 logistic regression을 사용한다.

목표는 discrepancy 값이 작은 순으로 source를 정렬했을 때, 상위 5개 안에 clean source가 얼마나 많이 들어가는지를 보는 것이다. 즉, 좋은 source를 골라내는 ranking problem이다. 샘플 수는 클래스당 $200$부터 $4000$까지 증가시키고, noise level은 $\epsilon = 30, 40, 50$으로 바꾼다. 각 설정은 15회 반복해 평균 score를 낸다.

결과는 S-disc가 샘플 수가 커질수록 더 좋은 성능을 보인다는 것이다. 반면 $d_{\mathcal{H}}$는 noisy source와 clean source를 거의 구분하지 못한다. 논문은 특히 $d_{\mathcal{H}}$가 항상 1을 반환했다고 적고 있다. 이는 MNIST-M과 MNIST를 전부 “관계없는 도메인”처럼 본다는 뜻이다. 따라서 실제 adaptation에서 중요한 source ranking 문제에서는 $d_{\mathcal{H}}$가 지나치게 거칠고, S-disc가 더 유의미한 신호를 제공한다고 해석할 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의, 이론, 알고리즘, 실험이 서로 잘 맞물려 있다는 점이다. 저자들은 단순히 새로운 discrepancy를 제안하는 데 그치지 않고, 왜 기존 척도가 불충분한지, 새 척도가 어떤 점에서 더 domain adaptation 목적에 맞는지, 그리고 그것이 이론적으로도 어떤 이득을 주는지를 일관되게 보여준다. 특히 source label을 적극적으로 활용한다는 발상은 unsupervised domain adaptation의 제약을 어기지 않으면서도 task-aware discrepancy를 만든다는 점에서 설득력이 있다.

또 다른 강점은 generalization bound의 구조가 매우 명료하다는 점이다. target regret가 source estimation term, optimal classifier mismatch term, discrepancy term으로 분해되며, 여기서 discrepancy term을 S-disc로 더 타이트하게 바꾼다는 논리는 이론적으로 깔끔하다. 실제로 $\varsigma_{\mathcal{H}}^{\ell}(P_T,P_S) \le \mathrm{disc}_{\mathcal{H}}^{\ell}(P_T,P_S)$라는 관계는 제안의 핵심 타당성을 잘 뒷받침한다.

실용성 측면에서도 장점이 있다. $\mathcal{X}$-disc는 계산이 비싸고, $d_{\mathcal{H}}$는 이론적 보장이 약하다. S-disc는 이 둘의 중간이 아니라, 계산 효율성과 이론적 보장을 동시에 어느 정도 만족한다. 특히 binary classification에서 cost-sensitive classification으로 환원하는 아이디어는 구현 가능성이 높다.

다만 한계도 분명하다. 첫째, 이 논문의 계산 알고리즘과 주요 이론은 사실상 **binary classification**에 초점이 맞춰져 있다. 일반 loss에 대한 정의와 이론은 제시되지만, 실제 효율적 계산법은 $0$-$1$ loss와 symmetric hypothesis class에서 전개된다. multi-class나 structured prediction, deep neural network 기반 복잡한 가설 공간으로 직접 확장하는 방법은 본문에 구체적으로 제시되지 않는다.

둘째, S-disc의 정의는 source-optimal classifier $h_S^*$를 중심에 놓는다. 이는 source label을 쓰는 장점이 있지만, 동시에 source domain이 매우 편향되어 있거나 source label 구조가 target과 본질적으로 다를 때는 이 기준점이 적절하지 않을 수 있다. 이 문제는 bound의 두 번째 항인 $R_T^{\ell}(h_S^*, h_T^*)$에 반영되는데, 이 항은 실제로는 관측할 수 없다. 즉, 이론적으로는 중요한 항이지만 실제 source selection이나 model selection에서 직접 제어하기 어렵다.

셋째, 실험은 주로 얕은 모델이나 비교적 단순한 설정에서 수행되었다. SVM과 logistic regression 기반 결과는 제안의 직관을 잘 보여주지만, 현대 deep domain adaptation 기법과 직접 경쟁하는 형태의 대규모 실험은 이 텍스트에 나타나지 않는다. 따라서 최신 representation learning 기반 adaptation 방법에서 S-disc가 어떻게 사용될지는 논문만으로는 아직 열려 있다.

넷째, $\mathcal{Y}$-disc와 달리 target label이 없어 계산 가능하다는 점은 장점이지만, 결국 S-disc도 source classifier에 의존한다. source classifier가 부정확하면 pseudo-label 기반 discrepancy estimation도 영향을 받을 수 있다. 논문은 consistency를 제시하지만, source 학습 자체가 어려운 경우나 hypothesis misspecification이 심한 경우의 영향은 자세히 분석하지 않는다.

비판적으로 보면, 이 논문은 “더 나은 discrepancy measure”를 성공적으로 제안했지만, domain adaptation 전체를 해결하는 방법을 제시하는 것은 아니다. 즉, S-disc는 source selection이나 domain closeness 평가에는 강력하지만, representation alignment나 feature transformation을 직접 학습하는 adaptation 알고리즘은 아니다. 따라서 실제 적용에서는 S-disc를 단독 방법으로 쓰기보다, source weighting, source selection, 또는 adaptation objective의 auxiliary criterion으로 활용하는 방향이 자연스럽다.

## 6. 결론

이 논문은 unsupervised domain adaptation에서 사용할 수 있는 새로운 discrepancy measure인 **source-guided discrepancy (S-disc)**를 제안한다. 핵심은 source label 정보를 활용해 source-optimal classifier를 기준점으로 삼음으로써, 기존 $\mathcal{X}$-disc보다 더 task-relevant하고 더 타이트한 discrepancy를 측정하는 것이다.

논문의 주요 기여는 세 가지로 정리할 수 있다. 첫째, target label 없이도 계산 가능한 새로운 discrepancy 정의를 제시했다. 둘째, binary classification과 $0$-$1$ loss에 대해 이를 cost-sensitive classification 문제로 환원하는 효율적 계산 알고리즘을 제안했다. 셋째, estimator의 consistency와 $\mathcal{O}_p(n_T^{-1/2}+n_S^{-1/2})$ 수렴 속도를 보였고, 기존 $\mathcal{X}$-disc 기반보다 더 타이트한 target generalization bound를 도출했다. 실험적으로도 toy example, empirical convergence, source selection에서 S-disc가 기존 $d_{\mathcal{H}}$보다 더 유용한 signal을 주는 것을 확인했다.

실제 적용 측면에서 이 연구는 특히 **source selection**, **multi-source adaptation에서의 source weighting**, 그리고 **adaptation 전에 어떤 source가 target에 적합한지 판단하는 진단 도구**로 중요할 가능성이 크다. 또한 discrepancy를 입력 분포 수준이 아니라 task와 classifier 수준에서 정의해야 한다는 관점을 강화한다는 점에서도 의미가 있다.

향후 연구로는 multi-class 확장, deep representation learning과의 결합, end-to-end adaptation objective 내에서 S-disc를 직접 최적화하는 방법, 그리고 source-optimal classifier의 품질이 discrepancy estimation에 미치는 영향 분석 등이 자연스럽게 이어질 수 있다. 이 논문은 domain adaptation의 “도메인 차이”를 더 똑똑하게 정의하려는 시도이며, 이론과 실용성의 균형을 잘 잡은 작업으로 평가할 수 있다.
