# Mind the Class Weight Bias: Weighted Maximum Mean Discrepancy for Unsupervised Domain Adaptation

* **저자**: Hongliang Yan, Yukang Ding, Peihua Li, Qilong Wang, Yong Xu, Wangmeng Zuo
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1705.00609](https://arxiv.org/abs/1705.00609)

## 1. 논문 개요

이 논문은 unsupervised domain adaptation, 즉 source domain에는 라벨이 있고 target domain에는 라벨이 없는 상황에서, 널리 쓰이던 MMD(Maximum Mean Discrepancy) 기반 정렬 방법의 중요한 약점을 지적한다. 저자들의 핵심 문제 제기는 간단하다. 기존 MMD 기반 방법들은 source와 target의 전체 분포를 가깝게 만들려고 하지만, 이 과정에서 각 클래스의 비율, 즉 class prior distribution 또는 class weight가 서로 다를 수 있다는 사실을 거의 고려하지 않았다.

논문은 이 문제를 **class weight bias**라고 부른다. 예를 들어 source에서는 어떤 숫자 클래스가 자주 나오고 target에서는 거의 나오지 않을 수 있다. 또는 source에 있던 몇몇 클래스가 target에는 거의 없거나 전혀 없는 경우도 생길 수 있다. 이런 상황에서 일반적인 MMD는 “두 도메인의 전체 평균 임베딩을 맞추는 것”만 보므로, 실제로는 각 클래스의 조건부 분포를 잘 맞추기보다 source의 클래스 비율을 target에도 강제로 유지하도록 유도할 수 있다. 저자들은 이것이 domain adaptation 성능을 악화시킨다고 주장한다.

이 문제의 중요성은 실제 응용에서 class prior shift가 매우 흔하다는 점에 있다. 표본 수집 방식이 달라지거나 적용 시나리오가 달라지면 클래스 비율은 자연스럽게 달라진다. 따라서 domain discrepancy를 줄이는 척도 자체가 이런 변화에 둔감하거나 오히려 잘못된 방향으로 작동하면, 학습된 representation이 genuinely domain-invariant하지 못하게 된다. 이 논문은 바로 이 지점에서 기존 MMD를 재해석하고, class prior가 다른 상황에서도 더 타당한 discrepancy measure를 만들기 위해 weighted MMD(WMMD)를 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 “source와 target의 전체 분포를 그대로 맞추는 것이 아니라, source를 먼저 재가중(reweighting)해서 target의 class prior와 맞춘 뒤 그 재가중된 source와 target을 비교해야 한다”는 것이다.

기존 MMD는 source 분포 $p_s(x)$와 target 분포 $p_t(x)$ 사이의 차이를 줄인다. 하지만 이 두 분포는 각 클래스 조건부 분포의 mixture이므로, 클래스 비율이 다르면 전체 분포 차이 안에 “도메인 차이”와 “클래스 비율 차이”가 섞여 들어간다. 논문은 이 중에서 우리가 줄이고 싶은 것은 클래스 조건부 분포의 차이지, 클래스 비율의 차이가 아니라고 본다. 따라서 source 쪽 각 클래스에 auxiliary weight를 두어 source의 class prior를 target과 유사하게 만든 reference source distribution을 구성하고, 그 reference와 target 사이의 discrepancy를 측정하자는 것이 핵심이다.

이 접근의 차별점은 sample-specific weight를 학습하거나 일부 source 샘플만 고르는 기존 reweighting/selection 계열과 달리, **class-specific auxiliary weight**를 도입한다는 점이다. 즉, 개별 샘플마다 서로 다른 weight를 주는 것이 아니라 “클래스 단위”로 source의 기여도를 조절한다. 이 설계는 논문이 겨냥하는 문제가 class weight bias이기 때문에 매우 직접적이다.

또 하나의 핵심은 target label이 없다는 UDA의 제약을 CEM(Classification EM) 방식으로 풀었다는 점이다. target의 pseudo-label을 현재 모델로 추정하고, 그 pseudo-label로 target class prior를 추정한 다음, 그 비율로 source class weight를 보정한다. 다시 말해 이 방법은 단순히 새로운 regularizer 하나를 제안하는 데서 끝나지 않고, target label이 없는 상태에서 그 regularizer를 실제로 학습할 수 있는 절차까지 함께 제안한다.

## 3. 상세 방법 설명

논문의 방법은 크게 세 부분으로 이해할 수 있다. 첫째, 기존 MMD가 왜 class weight bias를 처리하지 못하는지 확률분포 관점에서 설명한다. 둘째, weighted MMD를 정의한다. 셋째, 이를 CNN 기반 UDA 모델인 WDAN(Weighted Domain Adaptation Network)에 넣고 CEM으로 최적화한다.

### 3.1 기존 MMD의 한계

MMD는 RKHS 상에서 두 분포의 mean embedding 차이를 재는 비모수적 거리이다. 논문은 MMD를 다음과 같이 정리한다.

$$
\mathrm{MMD}^2(s,t)=\left| \mathbb{E}_{x^s \sim s}[\phi(x^s)]-\mathbb{E}_{x^t \sim t}[\phi(x^t)] \right|_{\mathcal H}^2
$$

실제 샘플 집합 $\mathcal D_s={x_i^s}_{i=1}^M$, $\mathcal D_t={x_j^t}_{j=1}^N$에 대해서는 다음과 같은 empirical estimator를 쓴다.

$$
\mathrm{MMD}^2(\mathcal D_s,\mathcal D_t)=\left| \frac{1}{M}\sum_{i=1}^M \phi(x_i^s)-\frac{1}{N}\sum_{j=1}^N \phi(x_j^t) \right|_{\mathcal H}^2
$$

문제는 이 식이 클래스 정보를 전혀 직접 반영하지 않는다는 점이다. 분포를 mixture 형태로 쓰면

$$
p_u(x^u)=\sum_{c=1}^{C} w_c^u, p_u(x^u \mid y^u=c), \quad u \in {s,t}
$$

이다. 여기서 $w_c^u$는 domain $u$에서 클래스 $c$의 prior, 즉 class weight이다. 만약 모든 클래스에 대해 $w_c^s=w_c^t$라면 전체 분포를 맞추는 것이 어느 정도 타당할 수 있다. 하지만 이 가정이 깨지면 전체 분포 차이에는 “조건부 분포 차이”뿐 아니라 “prior 차이”까지 함께 들어간다. 저자들은 바로 이 때문에 MMD가 잘못된 방향으로 최소화될 수 있다고 주장한다.

직관적으로 말하면, MMD를 줄이는 과정은 target representation을 source의 class ratio에 맞추는 방향으로도 작동할 수 있다. 논문 그림 설명에 따르면, 이 경우 target 샘플 일부가 잘못된 클래스 쪽으로 끌릴 수 있다. 즉, MMD 최소화가 항상 “좋은 도메인 불변 표현”을 의미하지는 않는다.

### 3.2 Weighted MMD의 정의

이를 해결하기 위해 논문은 source의 class conditional distribution은 유지하되 class prior는 target 쪽에 맞춘 새로운 reference source distribution $p_{s,\alpha}(x^s)$를 정의한다.

먼저 각 클래스별 auxiliary weight를

$$
\alpha_c=\frac{w_c^t}{w_c^s}
$$

로 둔다. 그러면 reference source distribution은

$$
p_{s,\alpha}(x^s)=\sum_{c=1}^{C} \alpha_c, w_c^s, p_s(x^s \mid y^s=c)
$$

로 쓸 수 있다. 이 식은 결국 source의 각 클래스 contribution을 target의 class prior에 맞게 재조정한 것이다. 이때 $\alpha_c w_c^s = w_c^t$가 되므로, reference source는 target과 같은 class prior를 갖게 된다.

이를 바탕으로 weighted MMD empirical estimator는 다음과 같이 정의된다.

$$
\mathrm{MMD}_w^2(\mathcal D_s,\mathcal D_t)=
\left|
\frac{1}{\sum_{i=1}^{M}\alpha_{y_i^s}}
\sum_{i=1}^{M}\alpha_{y_i^s}\phi(x_i^s)
-\frac{1}{N}\sum_{j=1}^{N}\phi(x_j^t)
\right|_{\mathcal H}^2
$$

이 식의 의미는 매우 분명하다. source 샘플의 평균 임베딩을 그냥 단순 평균하지 않고, 각 샘플이 속한 클래스의 auxiliary weight로 가중평균한다. 그러면 source 쪽 mean embedding이 “target의 prior를 반영한 source 평균”으로 바뀐다. 따라서 이 discrepancy는 prior mismatch를 줄이는 대신, class-conditional alignment에 더 집중하게 된다.

### 3.3 Linear-time approximation

기존 MMD는 pairwise kernel 계산 때문에 quadratic cost가 들 수 있어 CNN mini-batch SGD에 비효율적이다. 이를 위해 논문은 기존 linear-time unbiased estimator를 weighted version으로 확장한다.

원래 linear MMD는 quad-tuple $z_i=(x_{2i-1}^s, x_{2i}^s, x_{2i-1}^t, x_{2i}^t)$에 대해

$$
\mathrm{MMD}_{l}^2(s,t)=\frac{2}{M}\sum_{i=1}^{M/2} h_l(z_i)
$$

형태를 갖고,

$$
h_l(z_i)=k(x_{2i-1}^s,x_{2i}^s)+k(x_{2i-1}^t,x_{2i}^t)-k(x_{2i-1}^s,x_{2i}^t)-k(x_{2i}^s,x_{2i-1}^t)
$$

로 정의된다.

논문은 이를 weighted 형태로 바꾸어

$$
\mathrm{MMD}_{l,w}^2(\mathcal D_s,\mathcal D_t)=\frac{2}{M}\sum_{i=1}^{M/2} h_{l,w}(z_i)
$$

를 사용하고,

$$
\begin{aligned}
h_{l,w}(z_i)= & \; \alpha_{y_{2i-1}^s}\alpha_{y_{2i}^s}k(x_{2i-1}^s,x_{2i}^s) \\
& +k(x_{2i-1}^t,x_{2i}^t) \\
& -\alpha_{y_{2i-1}^s}k(x_{2i-1}^s,x_{2i}^t) \\
& -\alpha_{y_{2i}^s}k(x_{2i}^s,x_{2i-1}^t)
\end{aligned}
$$

로 쓴다.

여기서 중요한 점은 source-source 항에는 두 source 샘플의 클래스 weight가 곱으로 들어가고, source-target 교차항에는 대응하는 source 샘플의 클래스 weight만 들어간다는 점이다. 이 설계는 source mean embedding 자체를 weighted mean으로 바꾼 결과에 대응한다. 논문은 이 approximation 덕분에 WMMD를 mini-batch SGD에 넣을 수 있다고 설명한다.

### 3.4 WDAN의 목적함수

WMMD만 정의해서는 충분하지 않다. target label이 없기 때문이다. 이를 위해 논문은 semi-supervised logistic regression의 아이디어를 차용해 WDAN을 제안한다. 목적함수는 다음과 같다.

$$
\min_{\mathbf W,{\hat y_j}_{j=1}^{N},\boldsymbol{\alpha}}
\frac{1}{M}\sum_{i=1}^{M}\ell(x_i^s,y_i^s;\mathbf W)
+\gamma \frac{1}{N}\sum_{j=1}^{N}\ell(x_j^t,\hat y_j^t;\mathbf W)
+\lambda \sum_{l=l_1}^{l_2}\mathrm{MMD}_{l,w}(\mathcal D_s^l,\mathcal D_t^l)
$$

여기서 첫 번째 항은 source supervised loss이고, 두 번째 항은 pseudo-label이 붙은 target에 대한 loss이며, 세 번째 항은 여러 adaptation layer에서의 weighted MMD regularizer이다. $\lambda$는 discrepancy regularization의 강도를, $\gamma$는 target pseudo-label loss의 비중을 조절한다.

이 목적식은 논문 방법의 성격을 잘 보여준다. 단순히 domain discrepancy만 줄이는 것이 아니라, source discriminative learning과 target pseudo-supervision도 함께 사용한다. 또한 weighted MMD는 주로 higher layers에 삽입한다. 이는 deep network의 상위 계층일수록 task-specific feature가 강해져 domain bias가 커진다는 기존 연구의 관찰을 따른 것이다.

### 3.5 CEM 기반 학습 절차

WDAN 최적화의 가장 중요한 부분은 target label이 없는 상태에서 $\alpha$를 추정하는 것이다. 논문은 CEM 구조로 이를 해결한다.

#### E-step

현재 모델 파라미터 $\mathbf W$가 주어졌을 때, target 샘플 $x_j^t$의 클래스 posterior를 softmax 출력으로 정의한다.

$$
p(y_j^t=c \mid x_j^t)=g_c(x_j^t;\mathbf W)
$$

즉, 별도의 복잡한 posterior model이 아니라 현재 CNN classifier의 softmax 출력을 그대로 posterior로 사용한다.

#### C-step

posterior가 주어지면 pseudo-label을 가장 확률이 높은 클래스로 할당한다.

$$
\hat y_j=\arg\max_c p(y_j^t=c \mid x_j^t)
$$

이후 indicator function $\mathbf 1_c(\hat y_j)$를 사용해 target class proportion을 추정한다.

$$
\hat w_c^t=\frac{\sum_j \mathbf 1_c(\hat y_j)}{N}
$$

그리고 auxiliary weight를

$$
\alpha_c=\frac{\hat w_c^t}{w_c^s}
$$

로 갱신한다.

이 단계의 의미는 분명하다. target의 true class prior를 알 수 없으므로, 현재 모델이 추정한 pseudo-label 분포를 기반으로 target prior를 대신 추정한다. 따라서 $\alpha$는 학습이 진행되면서 계속 업데이트된다.

#### M-step

$\alpha$와 pseudo-label을 고정한 상태에서 모델 파라미터 $\mathbf W$를 SGD로 갱신한다. 이때 사용하는 loss는

$$
\mathcal L(\mathbf W)=
\frac{1}{M}\sum_{i=1}^{M}\ell(x_i^s,y_i^s;\mathbf W)
+\gamma \frac{1}{N}\sum_{j=1}^{N}\ell(x_j^t,\hat y_j^t;\mathbf W)
+\lambda \sum_{l=l_1}^{l_2}\mathrm{MMD}_{l,w}(\mathcal D_s^l,\mathcal D_t^l)
$$

이다.

논문은 quad-tuple 기반으로 각 layer feature에 대해 gradient를 계산할 수 있음을 설명한다. 예를 들어 source-side feature에 대한 WMMD term의 gradient는 kernel derivative와 class weight가 곱해진 형태가 된다. 핵심은 weighted term도 미분 가능하므로 backpropagation에 자연스럽게 들어간다는 점이다.

### 3.6 아키텍처 수준의 구현

논문은 여러 CNN architecture에 이를 적용한다. AlexNet에서는 마지막 세 개 fully connected layer에, GoogLeNet에서는 마지막 inception과 fully connected layer들에, LeNet에서는 마지막 fully connected layer에 WMMD regularizer를 둔다. 이는 기존 DAN 계열과 유사한 adaptation layer 설계이지만, regularizer가 MMD가 아니라 weighted MMD라는 점이 다르다.

이렇게 보면 WDAN은 완전히 새로운 backbone이라기보다, DAN류의 deep adaptation framework를 class prior shift까지 고려하도록 확장한 모델이라고 이해하는 것이 적절하다.

## 4. 실험 및 결과

논문은 네 가지 대표 벤치마크에서 WDAN을 평가한다. Office-10+Caltech-10, Office-31, ImageCLEF, Digit Recognition이며, backbone으로는 AlexNet, GoogLeNet, VGGNet-16, LeNet을 사용한다. 구현은 Caffe로 수행했고, batch size는 64이다. $\lambda$와 $\gamma$는 주어진 후보 집합에서 cross-validation으로 선택했다고 밝힌다. auxiliary weight는 초기값 $\alpha_c=1$로 시작한다.

### 4.1 Office-10+Caltech-10

이 데이터셋은 10개 공통 클래스를 가진 네 도메인 A, W, D, C로 구성되며, 총 6개의 adaptation task를 만든다. 논문은 AlexNet, GoogLeNet, VGGNet-16 기반 결과를 제시한다.

AlexNet 계열 평균 성능을 보면 DAN이 87.3%, WDAN이 89.2%, WDAN*가 89.8%이다. 여기서 WDAN*는 source와 target의 ground-truth class distribution을 prior로 사용한 이상적 상한선에 가까운 설정이다. 일반 WDAN이 DAN보다 1.9%p 높고, oracle prior를 쓴 WDAN*와도 차이가 크지 않다는 점은 pseudo-label 기반 class prior 추정이 실제로 어느 정도 잘 작동한다는 근거로 해석할 수 있다.

GoogLeNet에서는 DAN 92.3%, WDAN 93.2%로 0.9%p 향상되었고, VGGNet-16에서는 DAN 92.4%, WDAN 93.1%로 0.7%p 향상되었다. 즉, backbone이 바뀌어도 WDAN의 이점이 일관되게 유지된다. 이는 제안 방법이 특정 architecture에만 맞는 트릭이 아니라 discrepancy metric 수준의 개선이라는 논문의 주장과 잘 맞는다.

### 4.2 ImageCLEF

ImageCLEF subset에서는 Caltech256(C), Bing(B), PASCAL VOC2012(P)의 세 도메인을 사용해 6개 task를 만든다. GoogLeNet 기반 비교에서 baseline GoogLeNet은 평균 69.9, DDC는 70.1, DAN은 70.4, WDAN은 71.3을 얻는다.

절대 향상폭은 크지 않아 보일 수 있지만, 논문이 다루는 문제는 기존 domain alignment 틀을 완전히 바꾸는 것이 아니라 MMD의 blind spot을 해결하는 것이므로, 여러 task 평균에서 약 0.9%p의 일관된 개선은 충분히 의미 있는 결과로 제시된다. 특히 C→P에서 DAN 62.2, WDAN 65.0으로 상대적으로 큰 향상이 보인다. 이는 class weight bias가 더 심하거나 representation transfer가 더 까다로운 task에서 제안 방법이 더 유리할 가능성을 시사한다.

### 4.3 Digit Recognition

MNIST(M), SVHN(S), USPS(U) 세 데이터셋으로 네 개의 task를 구성하고 LeNet 기반으로 평가한다. 결과는 다음과 같다. LeNet 평균 45.5, SA 46.8, DAN 53.5, WDAN 57.2이다.

특히 M→S에서 DAN 19.3, WDAN 23.4이고, U→M에서도 DAN 60.5, WDAN 65.4이다. 이는 class prior shift를 고려한 정렬이 비교적 단순한 digit benchmark에서도 실제 효과를 낸다는 점을 보여준다. 논문은 DAN 대비 평균 3.7%p 향상을 보고하며, 이는 다른 데이터셋보다 더 큰 폭이다. digit datasets 간에는 appearance gap뿐 아니라 sampling distribution 차이도 커서 class weight bias 문제가 더 민감하게 작용했을 수 있다.

### 4.4 Office-31

31개 클래스를 가진 Office-31은 더 많은 클래스 수에서의 확장성을 보는 실험이다. AlexNet 기반 평균 정확도는 AlexNet 66.7, DAN 70.0, WDAN 72.1이다. 특히 W→D에서 DAN 95.2, WDAN 98.7로 큰 향상이 나타난다. 논문은 이를 통해 WMMD가 클래스 수가 많아져도 여전히 유효하다고 주장한다.

### 4.5 하이퍼파라미터 분석

논문은 $W \rightarrow C$ task에서 $\lambda$ 값을 변화시키며 WDAN과 DAN의 민감도를 비교한다. 결과 설명에 따르면 WDAN은 전반적으로 DAN보다 consistently better이며, WDAN의 최적은 $\lambda=0.4$, DAN의 최적은 $\lambda=0.1$이다. 또 WDAN은 $\lambda<1.2$일 때, DAN은 $\lambda<1.0$일 때 baseline AlexNet보다 좋다고 한다.

이 분석은 regularization strength가 너무 크면 오히려 discriminative learning이 약화될 수 있음을 보여준다. 즉, domain alignment와 classification performance의 균형이 중요하다는 일반적 사실을 재확인한다. 동시에 같은 $\lambda$ 범위에서 WDAN이 더 높은 성능을 유지한다는 점은 weighted alignment가 더 안정적임을 시사한다.

### 4.6 Class Weight Bias에 대한 강건성

논문에서 가장 설득력 있는 분석 중 하나는 target domain의 class distribution을 인위적으로 바꾸며 class weight bias 수준을 조절한 실험이다. PASCAL VOC2012와 Caltech256에서 airplane, motorbike 두 클래스만 택해 binary adaptation task를 만들고, source는 각 클래스 0.5로 고정한 뒤 target 비율만 점차 바꾼다.

결과 설명에 따르면 bias가 커질수록 DAN의 성능은 크게 감소한다. 반면 WDAN은 훨씬 더 robust하다. 이 실험은 논문의 핵심 주장을 직접 검증한다. 단순히 여러 벤치마크에서 성능이 조금 좋아졌다는 수준이 아니라, “왜 좋아지는가”에 대한 원인 검증 실험을 따로 제공한다는 점이 중요하다.

### 4.7 Feature Visualization

D→C task에서 DAN과 WDAN이 학습한 target feature를 t-SNE로 시각화한 결과, WDAN이 클래스 간 separation을 더 잘 유지한다고 논문은 설명한다. 저자들의 해석은 DAN이 class weight bias까지 같이 최소화하려 하기 때문에 클래스 구조를 희생할 수 있는 반면, WDAN은 prior effect를 제거하고 alignment하므로 class discrepancy distance를 더 잘 보존한다는 것이다.

정량 실험만으로는 representation 구조를 직접 보기 어렵기 때문에, 이 시각화는 제안 방법이 단순히 decision boundary tuning이 아니라 feature geometry에도 영향을 준다는 보조 근거로 작동한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 MMD 기반 domain adaptation에서 오랫동안 당연한 전제로 놓여 있던 “source와 target의 class prior가 비슷하다”는 가정을 정면으로 문제 삼았다는 점이다. 즉, 기존 방법을 조금 더 세게 학습시키거나 network를 더 깊게 만드는 식의 확장이 아니라, discrepancy metric의 개념적 타당성을 다시 점검했다는 점에서 학문적 의미가 있다.

두 번째 강점은 제안이 이론적 직관과 구현 가능성을 함께 가진다는 점이다. weighted MMD는 확률분포 mixture 관점에서 자연스럽게 도출되며, 동시에 linear-time approximation으로 mini-batch SGD에 넣을 수 있다. 즉, 수학적으로 그럴듯하면서도 실제 deep UDA pipeline에 바로 통합 가능하다.

세 번째 강점은 실험 설계가 논문의 주장과 직접 연결된다는 점이다. 여러 benchmark에서의 평균 향상뿐 아니라, class weight bias를 인위적으로 조절하는 별도의 robustness experiment를 수행했다. 이는 제안 방법이 왜 필요한지를 실험적으로도 보여준다.

반면 한계도 분명하다. 가장 중요한 한계는 target class prior 추정이 pseudo-label 품질에 크게 의존한다는 점이다. 논문은 CEM으로 이를 반복적으로 갱신하지만, 초기 classifier가 매우 부정확한 경우 잘못된 pseudo-label이 잘못된 $\hat w_c^t$를 만들고, 이것이 다시 잘못된 $\alpha_c$로 이어질 위험이 있다. 논문은 WDAN*와의 비교를 통해 추정이 꽤 잘 된다고 보이지만, pseudo-label noise가 심한 상황에서의 failure mode를 체계적으로 분석하지는 않았다.

또 다른 한계는 이 방법이 기본적으로 source와 target이 동일한 label space를 공유한다는 전형적 closed-set UDA 가정 위에 서 있다는 점이다. 논문 초반에 imbalanced cross-domain data를 special case로 언급하지만, 실제 formulation은 여전히 source와 target의 클래스 집합이 같다고 보는 쪽에 가깝다. 몇몇 클래스가 target에 거의 없거나 아예 없는 경우에 어느 정도 도움이 될 수는 있어도, open-set 또는 partial domain adaptation 문제를 명시적으로 다루지는 않는다.

또한 논문은 class-specific weighting을 도입했지만, class 내부에서의 multimodality나 finer-grained sample importance 차이는 고려하지 않는다. 즉, 어떤 클래스 안에서도 일부 source 샘플은 target과 잘 맞고 일부는 전혀 맞지 않을 수 있는데, WMMD는 같은 클래스에 같은 weight를 준다. 따라서 sample-level transferability까지 포착하는 방법과 비교하면 표현력이 제한될 수 있다.

비판적으로 보면, 이 논문은 “prior mismatch가 MMD를 왜곡한다”는 중요한 통찰을 제공하지만, 그 해결책이 pseudo-label 기반 prior estimation에 전적으로 기대는 만큼 초기 적응 단계의 안정성 문제가 숨어 있다. 논문은 CEM이 stationary value로 수렴할 수 있다고 언급하지만, 실제 deep non-convex setting에서의 수렴 특성은 강하게 보장되지 않는다. 다만 제공된 실험 범위에서는 이러한 문제가 성능 개선을 가릴 만큼 심각하지 않았던 것으로 보인다.

## 6. 결론

이 논문은 unsupervised domain adaptation에서 널리 쓰이던 MMD가 class weight bias를 고려하지 못한다는 점을 핵심 문제로 제기하고, 이를 해결하기 위한 weighted MMD를 제안했다. 방법의 본질은 source를 class-specific auxiliary weight로 재가중하여 target class prior를 반영한 reference source distribution을 만든 뒤, 그 분포와 target을 정렬하는 것이다. 이를 CNN 기반 WDAN으로 구현하고, pseudo-label과 auxiliary weight를 번갈아 갱신하는 CEM 기반 학습 절차를 제시했다.

실험적으로 WDAN은 Office-10+Caltech-10, ImageCLEF, Digit Recognition, Office-31에서 일관되게 DAN을 능가했다. 특히 class weight bias를 직접 조절한 실험은 기존 MMD 기반 적응이 bias에 취약하고, WMMD가 그 문제를 완화한다는 논문의 핵심 메시지를 잘 뒷받침한다.

이 연구의 의미는 단순히 DAN의 변형을 하나 더 제시한 데 있지 않다. 더 중요한 점은 domain discrepancy를 설계할 때 “무엇을 같게 만들어야 하는가”를 다시 묻는 시각을 제공했다는 데 있다. 실제 응용에서는 class prior shift가 흔하므로, 이 논문의 아이디어는 이후의 class-aware adaptation, partial adaptation, label-shift-aware transfer 같은 방향으로 자연스럽게 이어질 수 있다. 다만 target prior 추정의 정확도와 pseudo-label noise에 대한 의존성은 앞으로 더 정교한 방식으로 보완될 필요가 있다.

전반적으로 이 논문은 MMD 기반 UDA의 약점을 명확히 짚고, 개념적으로 타당하며 실용적으로도 구현 가능한 개선안을 제시한 의미 있는 연구로 볼 수 있다. 특히 “domain alignment는 전체 분포를 무조건 맞추는 것이 아니라, task-relevant한 구조를 보존하면서 맞춰야 한다”는 교훈을 분명하게 보여준다는 점에서 이후 연구에도 중요한 시사점을 가진다.
