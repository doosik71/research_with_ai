# Domain-Adversarial Neural Networks

* **저자**: Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand
* **발표연도**: 2014
* **arXiv**: [https://arxiv.org/abs/1412.4446](https://arxiv.org/abs/1412.4446)

## 1. 논문 개요

이 논문은 **domain adaptation** 문제를 위해 제안된 신경망 학습 방법인 **DANN (Domain-Adversarial Neural Network)** 을 소개한다. 문제 설정은 비교적 명확하다. 학습 시점에는 **source domain** 에서만 라벨이 있고, **target domain** 에는 라벨이 없거나 매우 적다. 그런데 실제로 성능을 내야 하는 곳은 target domain이다. 예를 들어 영화 리뷰로 학습한 감성 분류기를 책 리뷰에 적용하려고 할 때, source와 target의 분포가 다르기 때문에 일반적인 supervised learning만으로는 target 성능이 크게 떨어질 수 있다.

논문의 핵심 목표는, **source에서 분류에 유용한 표현은 유지하면서도, 그 표현만 보고는 데이터가 source에서 왔는지 target에서 왔는지 구별하기 어렵게 만드는 것**이다. 저자들은 이것이 domain adaptation 이론과 직접 연결된다고 주장한다. 즉, 좋은 표현은 단순히 분류에 도움이 되는 것만이 아니라, **domain 간 차이를 드러내지 않는 표현**이어야 한다는 것이다.

이 문제의 중요성은 매우 크다. 실제 데이터는 거의 항상 학습 분포와 테스트 분포가 다르며, 특히 새로운 도메인마다 라벨을 다시 수집하는 비용이 크다. 따라서 target에 라벨이 거의 없거나 전혀 없는 상황에서 source의 지식을 이전할 수 있는 방법은 실용성과 학술적 가치가 모두 높다. 이 논문은 이 문제를 신경망의 목적함수 안에 직접 넣고, adversarial 학습으로 해결하려 했다는 점에서 의미가 크다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음 한 문장으로 요약할 수 있다. **“좋은 domain-invariant representation은 label은 잘 예측하지만 domain은 잘 예측하지 못해야 한다.”** 저자들은 이 생각을 기존 domain adaptation 이론, 특히 Ben-David 계열의 일반화 bound에 기대어 정당화한다.

논문이 강조하는 이론적 배경은 target risk가 대략 다음 세 가지에 의해 좌우된다는 것이다. 첫째, source에서의 분류 오차. 둘째, source와 target 사이의 분포 차이. 셋째, 두 도메인에 동시에 잘 맞는 classifier가 실제로 존재하는가를 나타내는 항이다. 여기서 학습 알고리즘이 직접 제어할 수 있는 것은 주로 첫째와 둘째다. 따라서 source 분류 오차를 낮추는 동시에, source와 target이 표현 공간에서 서로 잘 구분되지 않도록 만들면 target 성능이 좋아질 가능성이 커진다.

이 논문의 차별점은 그 아이디어를 **신경망 내부 representation에 대해 직접 최적화**했다는 점이다. 기존의 많은 domain adaptation 연구는 선형 모델 중심이었고, 비선형 표현 학습에서는 mSDA 같은 robust representation이 강력한 성능을 보였다. 하지만 이 논문은 “노이즈에 robust한 표현”과 “domain을 구분하기 어려운 표현”은 같은 것이 아니라고 본다. 다시 말해, **robustness와 domain invariance는 별개의 원리이며 서로 보완적**일 수 있다는 것이다.

구체적으로 DANN은 하나의 hidden representation을 두 가지 서로 반대되는 목적에 동시에 노출시킨다. 분류기(classifier)는 이 representation으로 source label을 잘 맞추려고 하고, domain regressor는 이 representation으로 source/target domain을 잘 맞추려고 한다. 그런데 hidden layer는 domain regressor를 **헷갈리게 만드는 방향으로** 업데이트된다. 이 적대적 경쟁 구조가 바로 DANN의 핵심이다.

## 3. 상세 방법 설명

전체 구조는 비교적 단순한 **one-hidden-layer neural network** 위에 domain regressor를 추가한 형태다. 크게 세 부분으로 나눌 수 있다.

첫째는 **feature extractor** 역할의 hidden layer이다. 입력 $x$가 들어오면 hidden representation $h(x)$를 만든다.

둘째는 **label predictor** 이다. 이 모듈은 source 샘플에 대해서만 사용되며, hidden representation으로부터 class label을 예측한다.

셋째는 **domain regressor** 이다. 이 모듈은 source와 target 양쪽 샘플을 모두 받아서, 해당 샘플이 source인지 target인지 판별하려고 한다.

논문에서 hidden layer와 classifier는 다음처럼 정의된다.

$$
\begin{aligned}
h(x) &= \mathrm{sigm}(b + Wx) \\
f(x) &= \mathrm{softmax}(c + Vh(x))
\end{aligned}
$$

여기서 $W, b$는 hidden layer의 파라미터이고, $V, c$는 label predictor의 파라미터다. $\mathrm{sigm}$은 sigmoid activation이고, $\mathrm{softmax}$는 클래스 확률을 출력하는 함수다. 각 성분 $f_y(x)$는 입력 $x$가 클래스 $y$일 조건부 확률로 해석된다.

source의 labeled sample $S={(x_i^s, y_i^s)}_{i=1}^{m}$ 가 주어졌을 때, 기본적인 분류 손실은 정답 클래스의 negative log-likelihood이다.

$$
\mathcal{L}(f(x), y) = \log \frac{1}{f_y(x)}
$$

즉, 일반적인 supervised NN은 source 데이터에 대해 다음 목적을 최소화한다.

$$
\min_{W,V,b,c}\left[\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(f(x_i^s), y_i^s)\right]
$$

여기까지는 표준 분류 신경망이다.

이제 논문의 핵심이 들어간다. target unlabeled sample $T={x_i^t}_{i=1}^{m'}$ 가 있을 때, 저자들은 hidden representation $h(S)$ 와 $h(T)$ 사이의 domain 차이를 줄이는 regularizer를 추가한다. 이 regularizer는 Ben-David의 $\mathcal{H}$-divergence 이론에서 출발한다. $\mathcal{H}$-divergence는 어떤 hypothesis class $\mathcal{H}$ 가 source와 target을 얼마나 잘 구분할 수 있는지를 나타내는 양이다. 정의는 다음과 같다.

$$
d_{\mathcal{H}}(\mathcal{D}_S^X,\mathcal{D}_T^X) = 2\sup_{\eta \in \mathcal{H}} \left| \Pr_{x^s \sim \mathcal{D}_S^X}[\eta(x^s)=1] - \Pr_{x^t \sim \mathcal{D}_T^X}[\eta(x^t)=1] \right|
$$

직관적으로 말하면, 어떤 classifier가 source와 target을 쉽게 구별할 수 있다면 divergence가 크고, 구별하기 어렵다면 divergence가 작다.

논문은 이 divergence를 직접 계산하지 않고, **source/target 판별 문제를 푸는 domain classifier의 성능**으로 근사한다. 이를 위해 domain label $z$를 source면 1, target이면 0으로 두고, hidden representation 위에서 logistic regressor를 학습한다.

$$
p(z=1 \mid \phi)=o(\phi)=\mathrm{sigm}(d + u^\top \phi)
$$

여기서 $\phi$는 $h(x^s)$ 또는 $h(x^t)$ 이고, $u,d$는 domain regressor의 파라미터다.

그 다음 전체 목적함수는 다음 minimax 형태가 된다.

$$
\min_{W,V,b,c} \left[ \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(f(x_i^s), y_i^s) + \lambda \max_{u,d} \left( -\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}^d(o(x_i^s),1) -\frac{1}{m'}\sum_{i=1}^{m'}\mathcal{L}^d(o(x_i^t),0) \right) \right]
$$

여기서 $\lambda > 0$ 는 분류 성능과 domain confusion 사이의 균형을 조절하는 하이퍼파라미터다. domain loss $\mathcal{L}^d$ 는 표준 binary cross-entropy다.

$$
\mathcal{L}^d(o(x),z) = -z\log(o(x))-(1-z)\log(1-o(x))
$$

이 목적함수의 의미를 쉬운 말로 풀면 이렇다.

분류기 쪽 파라미터 $W,V,b,c$ 는

* source label을 잘 맞추도록 학습되어야 하고,
* 동시에 hidden representation이 domain regressor에게 유리하지 않도록 학습되어야 한다.

반면 domain regressor의 파라미터 $u,d$ 는

* source와 target을 최대한 잘 구분하도록 학습된다.

즉, hidden representation과 domain regressor가 서로 경쟁한다. domain regressor는 representation 안에서 domain 신호를 찾으려 하고, hidden layer는 그 신호를 지우려 한다. 이 adversarial 구조 때문에 최종 표현은 label 정보는 남기고 domain 정보는 줄이는 방향으로 유도된다.

학습 알고리즘은 hard EM처럼 번갈아 최적화할 수도 있지만, 논문은 **단순한 SGD로도 충분히 잘 된다**고 보고한다. 한 번의 업데이트에서 source 예제 하나와 target 예제 하나를 사용한다. source 예제로는 classification gradient와 domain gradient를 계산하고, target 예제로는 domain gradient만 계산한다. 중요한 점은,

* 일반 신경망 파라미터 $W,V,b,c$ 는 **gradient descent**
* domain regressor 파라미터 $u,d$ 는 **gradient ascent**

로 업데이트된다는 것이다. 즉, domain regressor는 자기 손실을 줄이는 방향으로 움직이고, hidden representation은 domain regressor의 성공을 방해하는 방향으로 역전파된다. 오늘날 널리 알려진 **gradient reversal** 아이디어의 원형이 바로 여기 있다. 다만 이 논문 텍스트에서는 별도의 layer 이름으로 formalize하기보다, 업데이트 부호를 반대로 적용하는 SGD 절차로 설명한다.

또 하나 중요한 실무적 요소는 **early stopping** 이다. source labeled sample의 90%를 학습, 10%를 validation으로 나누고, validation risk가 최소일 때 학습을 멈춘다. 이는 target label이 없는 기본 setting에서는 비교적 자연스러운 선택이다. 다만 Amazon 실험에서는 하이퍼파라미터 선택을 위해 target의 매우 작은 labeled validation set 100개를 썼다는 점이 별도로 명시되어 있다.

## 4. 실험 및 결과

실험은 크게 세 부분으로 구성된다. toy problem, Amazon reviews sentiment classification, 그리고 mSDA와의 결합이다. 추가로 이론과 연결하기 위해 Proxy A-distance도 측정한다.

### 4.1 Toy problem

첫 번째 실험은 inter-twinning moons의 변형 문제다. source는 두 개의 반달 모양 클래스이고, target은 같은 구조를 갖지만 전체가 $35^\circ$ 회전되어 있다. source에는 각 클래스당 150개씩 총 300개 labeled 예제가 있고, target에는 300개 unlabeled 예제가 있다.

비교 대상은 DANN과 표준 NN이다. 두 모델은 hidden layer 크기 15로 동일한 구조를 사용한다. 논문은 비교를 공정하게 하기 위해 표준 NN에서도 domain regressor 자체는 같이 학습시키되, **그 gradient가 hidden layer로 들어가지 않게** 막는다. 즉, domain classifier는 존재하지만 representation을 domain-invariant하게 만들지는 않는 버전이다.

이 toy 실험에서 논문은 네 가지 관점을 그림으로 보여준다.

첫째, **label classification boundary** 에서 표준 NN은 source는 잘 맞추지만 회전된 target에는 충분히 적응하지 못한다. 반면 DANN은 source와 target 모두를 더 잘 분리하는 경계를 학습한다.

둘째, **hidden representation의 PCA 시각화** 에서 DANN은 target 점들이 source 점들과 더 잘 섞여 나타난다. 반면 표준 NN은 target 점들이 별도 클러스터를 이루는 경향이 있다. 이는 DANN의 표현이 domain을 덜 드러낸다는 시각적 증거다.

셋째, **domain classification boundary** 에서 DANN의 domain regressor는 source와 target을 거의 구별하지 못한다. 표준 NN에서는 어느 정도 domain 분리가 가능하다. 이는 DANN이 실제로 domain confusion을 유도했음을 보여준다.

넷째, **hidden neurons의 결정면** 을 보면 표준 NN의 뉴런들은 domain rotation 정보까지 포착하는 패턴을 보이는 반면, DANN에서는 그런 패턴이 줄어든다. 즉, representation 차원에서 domain-specific signal이 제거된다는 해석이 가능하다.

이 toy 결과는 정량 표보다는 시각화 중심이지만, 제안 방법의 직관을 매우 선명하게 보여주는 역할을 한다.

### 4.2 Amazon reviews sentiment classification

핵심 정량 실험은 Amazon reviews dataset이다. 도메인은 books, dvd, electronics, kitchen 네 개이며, 각 리뷰는 unigram과 bigram 기반 **5,000차원 특징 벡터**로 표현된다. 라벨은 이진 감성 분류로, 평점이 3 이하이면 0, 4 또는 5이면 1이다.

총 12개의 domain adaptation task를 수행한다. 예를 들어 books $\to$ dvd는 books를 source, dvd를 target으로 사용하는 설정이다. 각 task에서

* source labeled 2,000개
* target unlabeled 2,000개
  를 학습에 사용하고,
* 별도 target test set 3,000~6,000개
  에서 평가한다.

비교 모델은 세 가지다.

* **DANN**
* **NN**: 같은 구조지만 domain-adversarial regularizer 없음
* **SVM**: linear kernel

하이퍼파라미터 선택은 grid search로 수행하며, 아주 작은 target labeled validation set 100개를 사용해 가장 낮은 target validation risk의 모델을 선택한다. 이 점은 완전한 unsupervised adaptation은 아니고, model selection 단계에 제한된 target supervision이 들어간 설정이라고 이해해야 한다.

표 1의 원본 입력 공간 결과를 보면, DANN은 12개 task 중 다수에서 NN과 SVM보다 더 낮은 error rate를 보인다. 몇 가지 예를 들면 다음과 같다.

* books $\to$ electronics: DANN 0.246, NN 0.251, SVM 0.256
* dvd $\to$ books: DANN 0.247, NN 0.261, SVM 0.269
* electronics $\to$ kitchen: DANN 0.148, NN 0.149, SVM 0.163
* kitchen $\to$ books: DANN 0.283, NN 0.288, SVM 0.325

모든 task에서 항상 압도적인 것은 아니다. 예를 들어 books $\to$ dvd에서는 NN이 0.199로 DANN 0.201보다 약간 낫고, kitchen $\to$ electronics에서는 SVM이 0.158로 DANN/NN의 0.161보다 낫다. 그러나 전체적으로 보면 DANN이 평균적으로 더 우세하다.

논문은 이를 **Pairwise Poisson binomial test** 로 정리한다. 원본 데이터 기준으로 DANN이

* NN보다 더 좋을 확률: **0.90**
* SVM보다 더 좋을 확률: **0.97**

이라고 보고한다. 저자들의 해석은 명확하다. NN과 DANN의 차이는 사실상 domain adaptation regularizer뿐이므로, 성능 향상은 곧 **domain-invariant representation 유도가 실제 target generalization에 도움이 된다**는 증거라는 것이다.

### 4.3 mSDA representation과의 결합

다음 실험은 이 논문의 중요한 메시지를 더 강화한다. 저자들은 당시 강력한 성능을 내던 **mSDA (marginalized stacked denoising autoencoders)** 와 DANN을 결합한다.

mSDA는 source와 target의 unlabeled 데이터를 모두 이용해 robust feature representation을 학습한다. corruption probability 50%, 5 layers를 사용했고, 원본 입력과 5개 layer 출력을 concat하여 최종 **30,000차원 표현**을 만든다.

그 위에서 다시 DANN, NN, SVM을 실행한다. 결과는 표 1의 “mSDA representation” 열에 정리되어 있다. 여기서도 DANN은 대체로 매우 강하다. 예를 들면,

* books $\to$ electronics: DANN 0.197, NN 0.228, SVM 0.244
* dvd $\to$ electronics: DANN 0.181, NN 0.234, SVM 0.220
* kitchen $\to$ books: DANN 0.222, NN 0.226, SVM 0.234
* electronics $\to$ dvd: DANN 0.216, NN 0.228, SVM 0.261

역시 일부 task에서는 NN이나 SVM이 소폭 앞서는 경우가 있다. 예를 들어 books $\to$ dvd에서는 NN 0.171이 DANN 0.176보다 약간 낮고, electronics $\to$ books에서는 SVM 0.229가 DANN 0.237보다 좋다. 그러나 전체적으로는 DANN이 가장 낫다.

Poisson binomial test 결과도 이를 뒷받침한다. mSDA representation 위에서 DANN이

* NN보다 더 좋을 확률: **0.82**
* SVM보다 더 좋을 확률: **0.88**

이다.

이 결과의 의미는 꽤 중요하다. mSDA는 노이즈에 robust한 representation을 학습하고, DANN은 domain discrimination을 억제하는 representation을 학습한다. 논문은 이 둘이 동일한 원리가 아니라 **상보적(complementary)** 이라고 주장한다. 실험 결과는 이 주장을 강하게 지지한다.

### 4.4 Proxy A-distance 분석

논문은 단순히 분류 성능만 보여주지 않고, 이론적 핵심인 **domain similarity** 자체를 측정하려고 한다. 이를 위해 Proxy A-distance (PAD)를 사용한다.

PAD는 source와 target representation을 얼마나 잘 구별할 수 있는지를 반영한다. 먼저 source/target representation으로 domain classification dataset $U$를 만들고, 이를 반으로 나눈 뒤, 한쪽으로 linear SVM을 학습하고 다른 쪽에서 error $\epsilon$을 측정한다. PAD는 다음처럼 계산된다.

$$
\hat d_A = 2(1 - 2\epsilon)
$$

이 값이 크면 source와 target이 잘 구분된다는 뜻이고, 작으면 구분이 어렵다는 뜻이다.

논문은 세 가지 PAD 결과를 보고한다.

첫째, raw data와 DANN representation을 비교하면, **DANN representation의 PAD가 더 낮다**. 즉, 원본 공간보다 domain 차이가 줄어든다.

둘째, hidden size를 100으로 고정하고 DANN과 표준 NN을 비교하면, **DANN이 NN보다 일관되게 더 낮은 PAD** 를 만든다. 이는 DANN의 adversarial regularizer가 representation의 domain distinguishability를 실제로 낮춘다는 직접 증거다.

셋째, mSDA representation 자체는 오히려 raw data보다 PAD가 더 커질 수 있다. 즉, mSDA는 target task 성능은 올리지만, Ben-David식 divergence 관점에서는 꼭 domain을 더 비슷하게 만들지는 않는다. 그런데 mSDA 위에 DANN을 얹으면 PAD가 다시 크게 떨어진다. 논문은 이것이 mSDA + DANN 성능 향상의 한 이유일 수 있다고 해석한다.

이 부분은 논문의 이론-실험 연결에서 매우 중요하다. 단순히 “성능이 좋아졌다”가 아니라, **제안한 메커니즘이 실제로 representation의 domain discriminability를 낮췄다**는 것을 보여주기 때문이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **이론적 동기와 신경망 구현이 매우 직접적으로 연결되어 있다**는 점이다. 많은 domain adaptation 방법이 경험적으로는 잘 작동하더라도 왜 그런지 설명이 약한 경우가 있는데, 이 논문은 Ben-David의 target risk bound에서 출발하여 “source risk와 domain divergence를 동시에 제어하자”는 목적을 명확히 세운다. 그리고 그 목적을 hidden representation에 대한 adversarial objective로 구현했다.

또 하나의 강점은 **개념적 단순성** 이다. 구조는 기본적인 one-hidden-layer network에 domain classifier 하나를 얹은 수준이며, 학습도 복잡한 alternating optimization 대신 SGD로 처리한다. 이후 많은 후속 연구에서 gradient reversal layer 형태로 널리 채택될 수 있었던 이유도 이 단순함에 있다.

세 번째 강점은 **실험 설계의 설득력** 이다. toy problem은 메커니즘을 시각적으로 설명하고, Amazon reviews는 실제 benchmark에서의 정량 성능을 보여준다. 거기에 PAD 분석까지 더해져, 단순 accuracy 향상뿐 아니라 representation 수준에서 무엇이 바뀌었는지를 확인한다. 특히 mSDA와의 결합 실험은 DANN이 기존 강력한 표현학습법과도 경쟁하기보다 보완 관계에 있음을 보여준다.

하지만 한계도 분명하다. 먼저, 이 논문은 **비교적 단순한 구조와 binary classification setting** 에 초점을 맞춘다. hidden layer도 하나뿐이고, task도 감성 이진분류 중심이다. 저자들도 결론에서 deeper architecture, multi-source adaptation, 다른 learning task로의 확장을 future work로 제시한다. 따라서 이 논문만으로 복잡한 비전 문제나 대규모 다중 클래스 문제에서의 일반성을 확정하기는 어렵다.

둘째, 실험에서 **하이퍼파라미터 선택에 target labeled validation set 100개를 사용**한다. 이는 논문 텍스트에 명시되어 있다. 따라서 완전히 label-free한 unsupervised domain adaptation setting이라고 보기보다, model selection 수준의 제한된 target supervision이 들어간 구성이다. 이 점은 결과 해석에서 주의가 필요하다.

셋째, 이론적 bound의 $\beta$ 항은 여전히 중요하지만 실제로는 제어하기 어렵다. 논문도 이를 인정한다. 즉, source와 target에 동시에 잘 맞는 classifier가 애초에 존재하지 않는다면, representation을 domain-invariant하게 만드는 것만으로는 충분하지 않다. domain invariance가 항상 좋은 것은 아니며, label-discriminative 정보까지 과도하게 제거할 위험도 있다. 이 논문은 그런 trade-off를 $\lambda$로 조절하지만, 언제 어떤 값이 좋은지에 대한 더 깊은 분석은 제공하지 않는다.

넷째, PAD 해석에는 흥미로운 긴장이 있다. mSDA는 PAD를 줄이지 않아도 target 성능을 향상시킬 수 있는데, 이는 단순한 domain indistinguishability만으로 adaptation 전체를 설명하기 어렵다는 뜻이기도 하다. 다시 말해, 논문이 제시한 이론적 시각은 강력하지만, 실제 adaptation 성능은 representation의 robustness, class structure 보존, optimization dynamics 같은 여러 요소의 결합 결과일 수 있다. 저자들도 이를 직접적으로 깊게 파고들지는 않는다.

비판적으로 보면, 이 논문은 오늘날 관점에서 매우 영향력이 크지만, 원문 자체는 아직 **초기 형태의 adversarial adaptation** 에 가깝다. domain classifier도 단순 logistic regressor이고, 실험 범위도 제한적이다. 그럼에도 불구하고 중요한 이유는, 이후의 수많은 adversarial domain adaptation 연구가 사실상 이 아이디어를 확장한 것이기 때문이다.

## 6. 결론

이 논문은 domain adaptation을 위해 **label prediction에는 유용하지만 domain prediction에는 불리한 hidden representation** 을 학습하는 DANN을 제안했다. 핵심은 source 분류 손실과 domain confusion 목적을 하나의 minimax 학습 문제로 묶는 것이며, 이를 단순한 SGD로 구현했다는 점이다.

실험적으로 DANN은 toy problem에서 domain-invariant representation의 직관을 잘 보여주었고, Amazon reviews benchmark에서는 표준 NN과 선형 SVM보다 전반적으로 더 나은 성능을 보였다. 또한 mSDA와 결합했을 때도 추가 향상을 보여, **robust representation과 domain-invariant representation이 상보적** 일 수 있음을 보였다. PAD 분석은 제안 방법이 실제로 source와 target의 표현 차이를 줄인다는 증거를 제공한다.

이 연구의 중요한 의의는 단순히 한 benchmark에서 성능이 좋았다는 데 있지 않다. 더 본질적으로는, **representation learning과 domain adaptation theory를 adversarial learning으로 연결하는 설계 틀**을 제시했다는 점에 있다. 이후의 gradient reversal, adversarial adaptation, fairness-aware representation learning 같은 여러 흐름과도 개념적으로 이어진다. 실제 응용 측면에서도 센서 변화, 장비 차이, 데이터 수집 환경 변화처럼 학습-테스트 분포가 달라지는 많은 문제에 이 아이디어가 유용할 가능성이 크다.

결국 이 논문은 “좋은 transfer representation이란 무엇인가?”라는 질문에 대해, **“class에는 informative하고 domain에는 uninformative한 표현”** 이라는 매우 강력한 답을 제시한 초기 대표작이라고 볼 수 있다.
