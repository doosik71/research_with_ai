# Contrastive Adaptation Network for Unsupervised Domain Adaptation

* **저자**: Guoliang Kang, Lu Jiang, Yi Yang, Alexander G. Hauptmann
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1901.00976](https://arxiv.org/abs/1901.00976)

## 1. 논문 개요

이 논문은 Unsupervised Domain Adaptation, 즉 UDA 문제를 다룬다. UDA는 source domain에는 라벨이 있지만 target domain에는 라벨이 전혀 없는 상황에서, source에서 학습한 모델이 target에서도 잘 동작하도록 만드는 문제다. 이 문제의 핵심 난점은 두 도메인의 데이터 분포가 다르다는 domain shift에 있다. 같은 클래스라도 source와 target에서 시각적 특성이 달라질 수 있기 때문에, source에서만 학습한 분류기는 target에서 성능이 크게 떨어질 수 있다.

기존의 많은 UDA 방법은 source와 target의 분포 차이를 줄이기 위해 MMD(Maximum Mean Discrepancy)나 그 변형을 사용해 domain discrepancy를 최소화했다. 그러나 이 논문은 그러한 접근이 대체로 class-agnostic, 즉 클래스 정보를 충분히 반영하지 않는다는 점을 문제로 지적한다. 단순히 전체 분포만 맞추면, 서로 다른 클래스의 샘플이 잘못 정렬될 수 있다. 예를 들어 target의 어떤 클래스 샘플이 source의 다른 클래스 쪽으로 끌려가도 전체 분포 관점에서는 discrepancy가 줄어든 것처럼 보일 수 있다. 이 경우 target에서 decision boundary가 불안정해지고 generalization이 나빠진다.

논문의 목표는 이러한 문제를 해결하기 위해 class-aware alignment를 수행하는 새로운 domain adaptation 프레임워크를 제안하는 것이다. 저자들은 같은 클래스끼리는 source와 target 간 거리를 줄이고, 다른 클래스끼리는 도메인을 넘어 더 잘 분리되도록 만드는 Contrastive Domain Discrepancy, 즉 CDD를 제안한다. 그리고 이 목적함수를 실제 딥러닝 학습에 적용하기 위해 Contrastive Adaptation Network, CAN을 설계한다.

이 문제는 중요하다. 실제 응용에서는 target domain 라벨을 얻기 어려운 경우가 많고, 특히 산업 환경에서는 새로운 장비, 조명, 카메라, 시뮬레이션 대 실제 이미지 간 차이처럼 domain shift가 흔하게 발생한다. 따라서 단순히 domain-level 정렬이 아니라 class-aware한 정렬을 수행해 더 discriminative한 표현을 만드는 것은 UDA의 실제 성능을 높이는 데 직접적으로 연결된다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 "도메인 전체를 한꺼번에 맞추는 것"이 아니라 "클래스를 고려해 맞춰야 한다"는 데 있다. 기존 MMD 기반 방법은 source와 target의 전체 feature distribution 차이를 줄이지만, 어떤 샘플이 어느 클래스에 속하는지를 고려하지 않는다. 저자들은 이것이 adaptation 실패의 중요한 원인이라고 본다.

이를 위해 제안된 개념이 Contrastive Domain Discrepancy이다. CDD는 두 가지 항을 동시에 고려한다. 첫째는 intra-class domain discrepancy로, 같은 클래스의 source와 target 샘플이 서로 가깝도록 만드는 항이다. 둘째는 inter-class domain discrepancy로, 서로 다른 클래스의 source와 target 샘플은 더 멀어지도록 만드는 항이다. 즉, 같은 클래스는 도메인을 넘어 정렬하고, 다른 클래스는 더 분리되게 만들어 feature space를 더 discriminative하게 만든다.

이 점이 기존 접근과의 가장 큰 차별점이다. 단순한 domain confusion이나 global discrepancy minimization은 class label을 무시하기 때문에 misalignment가 가능하다. 반면 CAN은 클래스 수준의 조건부 분포 차이를 직접 다룬다. 또한 단지 "같은 클래스끼리 모으는" 데서 멈추지 않고, "다른 클래스는 더 밀어낸다"는 contrastive 관점을 도입해 decision boundary 주변의 애매한 표현을 줄인다.

또 하나 중요한 아이디어는 target label이 없다는 UDA의 본질적 한계를 우회하기 위한 alternating optimization이다. CDD를 계산하려면 target sample의 클래스 정보가 필요하다. 그러나 실제 target label은 알 수 없으므로, 저자들은 현재 feature representation을 바탕으로 target을 clustering하여 pseudo label hypothesis를 만든다. 그런 다음 이 pseudo label을 사용해 CDD를 계산하고, 다시 network를 업데이트한다. 즉, label hypothesis와 feature representation을 번갈아 개선하는 구조다.

정리하면 이 논문의 핵심은 다음과 같이 이해할 수 있다. 첫째, domain adaptation은 class-aware해야 한다. 둘째, same-class alignment와 different-class separation을 동시에 최적화해야 더 discriminative한 target representation을 얻을 수 있다. 셋째, unlabeled target 문제는 clustering 기반 pseudo label과 alternating optimization으로 다룬다. 넷째, mini-batch 학습에서 각 클래스가 양 도메인에 모두 나타나도록 class-aware sampling을 사용해 CDD 추정을 안정화한다.

## 3. 상세 방법 설명

### 3.1 문제 설정

source domain 데이터는 $\mathcal{S}={(\mathbf{x}_1^s,y_1^s),\dots,(\mathbf{x}_{N_s}^s,y_{N_s}^s)}$로 주어지고, target domain 데이터는 $\mathcal{T}={\mathbf{x}_1^t,\dots,\mathbf{x}_{N_t}^t}$로 주어진다. source label $y^s$는 알려져 있지만 target label $y^t$는 알 수 없다. 목표는 $\mathcal{S}$와 $\mathcal{T}$를 이용해 target의 예측 $\hat{y}^t$를 정확히 수행하는 것이다.

딥네트워크 $\Phi_\theta$에서 특정 층 $l$의 표현은 $\phi_l(\mathbf{x})$로 표기한다. 논문은 특히 task-specific fully connected layer의 표현에 adaptation을 적용한다. 이는 convolutional layer는 비교적 transferable하고, FC layer는 더 domain-specific하다는 기존 UDA 관찰을 따른 것이다.

### 3.2 MMD 복습과 한계

논문은 먼저 MMD를 복습한다. MMD는 두 분포 $P$와 $Q$의 평균 임베딩 차이를 RKHS에서 측정하는 방식이다. 정의는 다음과 같다.

$$
\mathcal{D}_{\mathcal{H}}(P,Q)\triangleq \sup_{f\sim\mathcal{H}} \left(\mathbb{E}_{\mathbf{X}^s}[f(\mathbf{X}^s)]-\mathbb{E}_{\mathbf{X}^t}[f(\mathbf{X}^t)]\right)_{\mathcal{H}}
$$

실제 mini-batch에서는 층 $l$에 대해 empirical kernel mean embedding을 이용해 MMD의 제곱을 추정한다.

$$
\begin{aligned}
\hat{\mathcal{D}}^{mmd}_{l} = & \frac{1}{n_s^2}\sum_{i=1}^{n_s}\sum_{j=1}^{n_s}k_l(\phi_l(\mathbf{x}_i^s),\phi_l(\mathbf{x}_j^s)) \\
& + \frac{1}{n_t^2}\sum_{i=1}^{n_t}\sum_{j=1}^{n_t}k_l(\phi_l(\mathbf{x}_i^t),\phi_l(\mathbf{x}_j^t)) \\
& - \frac{2}{n_sn_t}\sum_{i=1}^{n_s}\sum_{j=1}^{n_t}k_l(\phi_l(\mathbf{x}_i^s),\phi_l(\mathbf{x}_j^t))
\end{aligned}
$$

이 식은 source와 target의 전체 분포 차이를 줄이지만, 클래스 조건부 구조를 반영하지 않는다. 그래서 서로 다른 클래스가 잘못 정렬될 여지가 있다.

### 3.3 Contrastive Domain Discrepancy

저자들은 source와 target의 conditional distribution 차이를 클래스별로 따로 측정한다. 이를 위해 클래스 마스크 함수 $\mu_{cc'}(y,y')$를 도입한다.

$$
\mu_{cc'}(y,y') = \begin{cases}
1 & \text{if } y=c,; y'=c' \\
0 & \text{otherwise}
\end{cases}
$$

두 클래스 $c_1, c_2$에 대해, 클래스-조건부 domain discrepancy를 다음과 같이 정의한다.

$$
\hat{\mathcal{D}}^{c_1c_2}(\hat{y}_1^t,\hat{y}_2^t,\dots,\hat{y}_{n_t}^t,\phi)=e_1+e_2-2e_3
$$

여기서 $e_1$, $e_2$, $e_3$는 각각 source 내부, target 내부, 그리고 source-target 사이의 같은 조건부 클래스 조합에 대한 kernel 평균이다.

$$
e_1= \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \frac{ \mu_{c_1c_1}(y_i^s,y_j^s),k(\phi(\mathbf{x}_i^s),\phi(\mathbf{x}_j^s)) }{ \sum_{i=1}^{n_s}\sum_{j=1}^{n_s}\mu_{c_1c_1}(y_i^s,y_j^s) }
$$

$$
e_2= \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \frac{ \mu_{c_2c_2}(\hat{y}_i^t,\hat{y}_j^t),k(\phi(\mathbf{x}_i^t),\phi(\mathbf{x}_j^t)) }{ \sum_{i=1}^{n_t}\sum_{j=1}^{n_t}\mu_{c_2c_2}(\hat{y}_i^t,\hat{y}_j^t) }
$$

$$
e_3= \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \frac{ \mu_{c_1c_2}(y_i^s,\hat{y}_j^t),k(\phi(\mathbf{x}_i^s),\phi(\mathbf{x}_j^t)) }{ \sum_{i=1}^{n_s}\sum_{j=1}^{n_t}\mu_{c_1c_2}(y_i^s,\hat{y}_j^t) }
$$

이 정의는 두 경우로 해석된다. $c_1=c_2=c$이면 같은 클래스에 대한 source-target discrepancy, 즉 intra-class discrepancy가 된다. 반면 $c_1\neq c_2$이면 서로 다른 클래스 사이 discrepancy, 즉 inter-class discrepancy가 된다.

최종 CDD는 intra-class discrepancy의 평균에서 inter-class discrepancy의 평균을 빼는 형태다.

$$
\hat{\mathcal{D}}^{cdd} = \underbrace{ \frac{1}{M}\sum_{c=1}^{M}\hat{\mathcal{D}}^{cc}(\hat{y}_{1:n_t}^t,\phi) }_{intra} - \underbrace{ \frac{1}{M(M-1)}\sum_{c=1}^{M}\sum_{\substack{c'=1 \\ c'\neq c}}^{M}\hat{\mathcal{D}}^{cc'}(\hat{y}_{1:n_t}^t,\phi) }_{inter}
$$

이 식의 의미는 매우 직관적이다. 첫 번째 항은 같은 클래스의 source-target 분포 차이를 줄이고, 두 번째 항 앞의 마이너스 부호는 서로 다른 클래스의 domain discrepancy를 크게 만들어 전체 목적함수 관점에서 inter-class separation을 유도한다. 논문은 이를 통해 class-aware alignment와 discriminative feature learning을 동시에 수행한다고 주장한다.

### 3.4 Contrastive Adaptation Network의 전체 목적함수

CAN은 ImageNet pretrained backbone, 예를 들면 ResNet-50이나 ResNet-101 위에 task-specific FC layer를 얹고, FC layer들의 표현에 대해 CDD를 계산한다. 여러 FC layer에 대한 CDD를 모두 더한 값은 다음과 같다.

$$
\hat{\mathcal{D}}^{cdd}_{\mathcal{L}}=\sum_{l=1}^{L}\hat{\mathcal{D}}^{cdd}_{l}
$$

또한 source labeled data에 대해서는 표준 cross-entropy loss를 적용한다.

$$
\ell^{ce} = -\frac{1}{n_s'} \sum_{i'=1}^{n_s'} \log P_\theta(y_{i'}^s|\mathbf{x}_{i'}^s)
$$

따라서 전체 학습 목적함수는 다음과 같다.

$$
\ell = \ell^{ce} + \beta \hat{\mathcal{D}}^{cdd}_{\mathcal{L}}
$$

여기서 $\beta$는 discrepancy penalty의 가중치다. 이 목적함수를 최소화하면, source 분류 성능을 유지하면서 same-class source-target alignment는 강화되고 different-class separation도 강화된다.

논문에서 주목할 점은 source data sampling을 두 목적에 대해 독립적으로 수행한다는 것이다. 즉, cross-entropy를 위한 source mini-batch와 CDD 계산을 위한 source mini-batch가 다를 수 있다. 이는 class-aware sampling을 더 유연하게 하기 위한 설계다.

### 3.5 Alternating Optimization

이 방법의 핵심 난제는 target label이 없다는 점이다. 저자들은 이를 직접 예측해서 매 iteration마다 동시에 쓰는 대신, clustering과 network update를 번갈아 수행한다.

각 loop에서 먼저 현재 network parameter $\theta$를 고정한 채 target feature를 clustering하여 pseudo label hypothesis를 얻는다. 그 후 이 pseudo label을 사용해 CDD를 계산하고, back-propagation으로 $\theta$를 업데이트한다. 즉, label estimation과 feature adaptation을 교대로 수행한다.

target clustering에는 spherical K-means를 사용한다. 각 sample은 첫 번째 task-specific layer의 입력 활성값 $\phi_1(\cdot)$로 표현된다. 예를 들어 ResNet에서는 global average pooling output이 이에 해당한다.

각 source 클래스 중심은 다음과 같이 계산된다.

$$
O^{sc} = \sum_{i=1}^{N_s} \mathbf{1}_{y_i^s=c} \frac{\phi_1(\mathbf{x}_i^s)}{|\phi_1(\mathbf{x}_i^s)|}
$$

target의 클래스 중심 $O^{tc}$는 같은 클래스의 source 중심 $O^{sc}$로 초기화된다. 거리 측정은 cosine dissimilarity를 사용한다.

$$
dist(\mathbf{a},\mathbf{b}) = \frac{1}{2} \left( 1- \frac{\langle \mathbf{a},\mathbf{b}\rangle}{|\mathbf{a}||\mathbf{b}|} \right)
$$

이후 clustering은 두 단계를 반복한다. 첫째, 각 target sample을 가장 가까운 중심에 할당해 pseudo label을 부여한다.

$$
\hat{y}_i^t \leftarrow \arg\min_c dist(\phi_1(\mathbf{x}_i^t), O^{tc})
$$

둘째, 할당된 pseudo label에 따라 target 중심을 다시 계산한다.

$$
O^{tc} \leftarrow \sum_{i=1}^{N_t} \mathbf{1}_{\hat{y}_i^t=c} \frac{\phi_1(\mathbf{x}_i^t)}{|\phi_1(\mathbf{x}_i^t)|}
$$

이 과정을 수렴하거나 최대 반복 횟수에 도달할 때까지 반복한다.

### 3.6 Ambiguous sample과 ambiguous class 처리

저자들은 pseudo label noise가 adaptation에 해가 될 수 있음을 인지하고, clustering 후 신뢰도가 낮은 데이터와 클래스를 제거한다.

먼저 ambiguous data는 자신이 속한 cluster center에서 너무 먼 target sample이다. 다음 조건을 만족하는 샘플만 남긴다.

$$
\tilde{\mathcal{T}} = { (\mathbf{x}^t,\hat{y}^t) \mid dist(\phi_1(\mathbf{x}^t),O^{t(\hat{y}^t)}) \lt D_0 ,\; \mathbf{x}^t\in\mathcal{T} }
$$

여기서 $D_0\in[0,1]$는 threshold다.

또한 어떤 클래스가 너무 적은 수의 confident target sample만 포함한다면, 그 클래스는 현재 loop에서 제외한다. 선택된 클래스 집합은 다음과 같다.

$$
\mathcal{C}_{T_e} = \left \{ c \mid \sum_i \mathbf{1}_{\hat{y}_i^t=c} \gt N_0,\; c\in{0,1,\dots,M-1} \right \}
$$

즉, 충분한 confident target sample이 있는 클래스만 adaptation에 참여시킨다. 논문은 학습이 진행될수록 더 많은 클래스와 샘플이 포함되며, 이를 progressive learning 관점에서 해석한다.

### 3.7 Class-Aware Sampling

보통의 mini-batch sampling은 클래스를 고려하지 않는다. 그러나 CDD를 계산하려면 한 클래스에 대해 source와 target 양쪽 샘플이 모두 mini-batch에 포함되어야 intra-class discrepancy를 안정적으로 추정할 수 있다. 그렇지 않으면 어떤 클래스는 source만 있고 target이 없어서 discrepancy를 계산하지 못한다.

이를 해결하기 위해 저자들은 class-aware sampling, CAS를 제안한다. 각 loop에서 선택된 클래스 집합 $\mathcal{C}_{T_e}$ 중 일부 $\mathcal{C}'_{T_e}$를 랜덤하게 뽑고, 이 클래스들에 대해 source와 target 양 도메인에서 샘플을 함께 뽑는다. 이렇게 하면 mini-batch마다 각 선택 클래스에 대해 source-target 대응이 존재할 가능성이 높아져 CDD 계산이 효율적이고 안정적이 된다.

### 3.8 알고리즘 흐름

논문에 제시된 Algorithm 1을 자연스럽게 정리하면 다음과 같다.

먼저 source 전체를 forward해 클래스 중심 $O^{sc}$를 구한다. 그 다음 이를 이용해 target cluster center를 초기화한다. 이후 target 전체에 대해 spherical K-means를 수행해 pseudo label을 만든다. 클러스터 중심과 거리가 먼 ambiguous sample과, confident sample 수가 부족한 ambiguous class를 제거한다. 그런 다음 $K$번의 network update 단계에서 class-aware sampling으로 mini-batch를 구성해 CDD를 계산하고, 별도로 source batch로 cross-entropy를 계산한 뒤, 두 손실을 합친 목적함수로 back-propagation을 수행한다. 이 loop를 여러 번 반복한다.

핵심은 pseudo label을 매 step마다 즉시 바꾸는 것이 아니라, 일정한 loop 단위로 clustering을 수행한 후 그 결과를 이용해 몇 번의 gradient update를 하는 비동기적 구조라는 점이다. 논문은 이 설계가 더 stable하고 efficient하다고 설명한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 두 개의 대표적인 UDA benchmark를 사용한다.

첫째는 Office-31이다. 총 4,110장의 이미지와 31개 클래스로 구성되며, Amazon(A), Webcam(W), DSLR(D)의 세 도메인을 포함한다. 대표적인 6개 adaptation task는 $A\rightarrow W$, $D\rightarrow W$, $W\rightarrow D$, $A\rightarrow D$, $D\rightarrow A$, $W\rightarrow A$이다. 도메인 간 데이터 수가 불균형하며, A는 2,817장, W는 795장, D는 498장이다.

둘째는 VisDA-2017 classification benchmark다. synthetic-to-real adaptation을 다루며 12개 클래스로 구성된다. 총 약 280k 이미지가 있고, training은 152,397장의 synthetic 이미지, validation은 55,388장의 real 이미지, test는 72,372장의 real 이미지로 구성된다. 이 데이터셋은 domain gap이 크기 때문에 더 도전적인 벤치마크다.

비교 대상은 RevGrad, DAN, JAN 같은 class-agnostic alignment 계열과, MADA, MCD, ADR 같이 클래스 정보나 decision boundary를 더 반영하는 방법들이다. backbone은 Office-31에서 ResNet-50, VisDA-2017에서 ResNet-101을 사용했다. Batch normalization 파라미터는 도메인별로 분리하고, 그 외 파라미터는 source와 target 간 공유한다.

최적화는 SGD with momentum 0.9를 사용한다. 학습률은 $\eta_p=\frac{\eta_0}{(1+ap)^b}$ 형태로 조정된다. convolutional layer의 초기 학습률은 0.001, task-specific FC layer는 0.01이다. $\beta$는 0.3으로 설정되었다. 일부 Office-31 task에서는 $(D_0,N_0)=(0.05,3)$를 사용해 ambiguous target 샘플과 클래스를 필터링했고, 다른 task에서는 필터링을 사용하지 않았다.

### 4.2 Office-31 결과

Office-31 결과에서 CAN은 모든 6개 task에서 가장 높은 평균 성능을 보인다. 평균 정확도는 다음과 같다.

* Source-finetune: 76.1
* RevGrad: 82.2
* DAN: 80.4
* JAN: 84.3
* MADA: 85.2
* Ours (intra only): 89.5
* **Ours (CAN): 90.6**

특히 $A\rightarrow W$에서 94.5, $A\rightarrow D$에서 95.0, $D\rightarrow A$에서 78.0, $W\rightarrow A$에서 77.0을 기록했다. 논문은 평균 기준으로 CAN이 JAN보다 절대값 6.3%p, MADA보다 5.4%p 높다고 강조한다.

이 결과는 두 가지를 보여준다. 첫째, 단순 source fine-tuning보다 UDA가 확실한 이점을 가진다. 둘째, class-aware adaptation이 class-agnostic adaptation보다 더 강력하다. 셋째, class-aware 방법들 중에서도 intra-only보다 full CAN이 더 좋기 때문에, same-class alignment뿐 아니라 different-class separation도 중요하다는 점을 뒷받침한다.

### 4.3 VisDA-2017 결과

VisDA-2017 validation set에서 CAN의 평균 정확도는 87.2%다. 주요 비교 결과는 다음과 같다.

* Source-finetune: 49.4
* RevGrad: 57.4
* DAN: 62.8
* JAN: 65.7
* MCD: 71.9
* ADR: 74.8
* SE: 84.3
* Ours (intra only): 83.9
* **Ours (CAN): 87.2**

클래스별로 보면 airplane 97.0, bicycle 87.2, horse 97.8, knife 96.2, skateboard 96.3, truck 59.9 등을 기록했다. 일부 클래스는 여전히 어렵지만 전체 평균은 매우 높다. 특히 당시 VisDA-2017 competition 1위였던 self-ensembling(SE)의 84.3%보다 2.9%p 높다.

논문은 test set에도 single model만으로 87.4%를 기록했다고 설명한다. 이는 leaderboard 2위 방법의 87.7%와 비교 가능한 성능이다. 저자들은 ensemble이나 추가 data augmentation을 사용하지 않았음을 강조한다. 따라서 vanilla ResNet-101 backbone 위에서도 제안 기법이 강력하다고 해석할 수 있다.

### 4.4 정성 분석

논문은 Office-31의 $W\rightarrow A$ task에 대해 t-SNE 시각화를 제시한다. JAN과 CAN을 비교했을 때, CAN의 target 표현은 더 높은 intra-class compactness와 더 큰 inter-class margin을 보인다고 설명한다. 이는 CDD의 목적과 정성적으로 일치한다. 다만 제공된 텍스트에는 그림 자체의 세부 패턴이 포함되어 있지 않으므로, 시각적 배치에 대한 더 구체적인 묘사는 할 수 없다.

### 4.5 Ablation study: inter-class discrepancy의 효과

가장 직접적인 ablation은 "intra only"와 "CAN" 비교다. "intra only"는 intra-class discrepancy만 최소화하고 inter-class discrepancy 최대화는 제거한 버전이다. Office-31에서는 평균 89.5에서 90.6으로, VisDA-2017에서는 83.9에서 87.2로 성능이 오른다.

이는 inter-class term이 단순한 부가 요소가 아니라 성능 향상에 실질적으로 기여함을 보여준다. 저자들의 해석은 다음과 같다. intra-class discrepancy만 줄이는 것으로는 adaptation이 완전하지 않고, 여전히 source에 과적합된 결정경계가 남을 수 있다. inter-class discrepancy를 키우면 클래스 사이 간격이 커져 target에서 더 robust한 분류가 가능해진다.

### 4.6 Ablation study: AO와 CAS의 효과

Table 3은 alternative optimization(AO)과 class-aware sampling(CAS)의 기여를 보여준다.

* Office-31 평균: w/o AO 88.1, w/o CAS 89.1, CAN 90.6
* VisDA-2017 평균: w/o AO 77.5, w/o CAS 81.6, CAN 87.2

두 구성 요소 모두 중요하지만, 특히 VisDA-2017처럼 더 어려운 설정에서는 AO와 CAS가 큰 차이를 만든다. AO를 제거하면 pseudo label noise를 더 직접적으로 받게 되고, CAS를 제거하면 mini-batch에서 CDD 추정 효율이 떨어진다.

흥미로운 점은 AO 없이도 class-agnostic baseline보다 성능이 좋다는 것이다. 즉, 제안된 CDD 자체가 어느 정도 label noise에 robust하다는 점을 시사한다. 저자들은 이것이 MMD 기반 평균 임베딩 통계가 개별 noisy label에 덜 민감하기 때문이라고 해석한다.

### 4.7 Ablation study: pseudo label 활용 방식 비교

Table 4는 pseudo label을 어떻게 사용하는지가 중요함을 보여준다.

* pseudo0: initial clustering의 pseudo label을 고정한 채 학습
* pseudo1: clustering으로 pseudo label을 갱신하면서 pseudo-labeled target cross-entropy로 학습
* CAN: clustering 갱신 + CDD 기반 학습

결과는 다음과 같다.

* pseudo0 평균: 79.8
* pseudo1 평균: 83.4
* CAN 평균: 86.1

pseudo0는 initial clustering 정확도와 사실상 같았다고 논문은 설명한다. 이는 deep network의 capacity가 pseudo label을 그대로 외워버릴 수 있음을 뜻한다. pseudo1은 반복적으로 pseudo label을 갱신해 성능이 좋아지지만, CAN보다 낮다. 이는 단순히 pseudo-labeled target에 cross-entropy를 거는 것보다, class-aware discrepancy를 직접 최적화하는 것이 더 효과적임을 보여준다.

### 4.8 학습 중 CDD와 정확도 변화

논문은 ground-truth target label로 계산한 CDD-G의 변화를 관찰한다. JAN은 초기에 조금 감소하다가 높은 수준에서 빠르게 안정화되며, contrastive discrepancy를 충분히 줄이지 못한다. 반면 CAN은 pseudo label이 완벽하지 않음에도 training이 진행될수록 CDD가 꾸준히 감소한다. 이와 함께 정확도도 증가한다.

이 관찰은 두 가지를 말해준다. 첫째, clustering 기반 target label hypothesis가 ground-truth CDD를 추적하는 reasonable proxy로 작동한다. 둘째, 정말로 CDD를 줄이는 것이 target accuracy 향상과 연결된다.

### 4.9 Hyper-parameter sensitivity

논문은 balance weight $\beta$의 민감도를 예시 task들에서 분석한다. 결과는 전반적으로 bell-shaped curve를 보인다고 설명한다. $\beta$가 너무 작으면 adaptation 효과가 약하고, 어느 범위까지는 성능이 증가하다가, 너무 크면 오히려 감소한다. 그러나 저자들은 CAN이 상당히 넓은 범위의 $\beta$에서 baseline보다 좋은 성능을 보여 비교적 안정적이라고 주장한다. 제공된 텍스트에는 정확한 수치표가 없으므로, 구체적인 최적 범위는 적시할 수 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 UDA의 domain alignment를 class-aware 관점에서 재정의했다는 점이다. 기존 global discrepancy minimization이 가진 misalignment 문제를 명확히 지적하고, intra-class alignment와 inter-class separation을 동시에 다루는 CDD라는 설계를 제시했다. 이 아이디어는 단순하고 직관적이면서도, MMD 기반 프레임워크 위에 자연스럽게 얹을 수 있어 기존 이론 및 구현과도 잘 연결된다.

두 번째 강점은 실제 학습 가능하도록 세부 메커니즘을 잘 설계했다는 점이다. UDA에서는 target label이 없기 때문에 클래스 정보를 쓰는 것이 항상 어렵다. 이 논문은 clustering 기반 pseudo label hypothesis, ambiguous sample/class filtering, class-aware sampling, alternating optimization을 결합해 이 문제를 실용적으로 해결했다. 즉, 핵심 아이디어뿐 아니라 그것이 동작하도록 만드는 학습 절차까지 함께 제안했다는 점이 좋다.

세 번째 강점은 실험적 설득력이다. Office-31과 VisDA-2017에서 모두 강한 baseline을 넘는 성능을 보였고, ablation study도 비교적 체계적이다. 단지 최종 숫자만 보고 끝나는 것이 아니라, inter-class term의 효과, AO의 효과, CAS의 효과, pseudo label 활용 방식의 차이까지 분석했다. 이는 제안 요소들이 실제로 필요한지 검증하려는 태도를 보여준다.

네 번째 강점은 feature quality에 대한 설명 가능성이다. t-SNE와 CDD 변화 곡선을 통해, CAN이 왜 더 좋은지에 대한 정성적 직관도 제시한다. 단순히 "성능이 좋다"가 아니라 "같은 클래스는 더 조밀해지고 다른 클래스는 더 멀어진다"는 구조적 해석이 가능하다.

반면 한계도 분명하다. 가장 본질적인 한계는 pseudo label 품질에 여전히 의존한다는 점이다. 논문은 CDD가 label noise에 어느 정도 robust하다고 주장하지만, target clustering이 매우 부정확한 초반 단계나 class imbalance가 심한 상황에서는 성능이 흔들릴 수 있다. 이를 줄이기 위해 ambiguous sample/class filtering을 넣었지만, 이는 반대로 어려운 샘플을 한동안 학습에서 배제하는 효과도 낳는다.

두 번째 한계는 clustering 기반 alternating optimization이 추가적인 계산 비용과 구현 복잡도를 가져온다는 점이다. 매 loop마다 전체 target에 대해 clustering을 수행해야 하고, source class center 계산도 필요하다. 논문은 stable and efficient하다고 설명하지만, 실제 대규모 환경에서의 비용이나 scalability에 대한 정량 분석은 제공된 텍스트에 없다.

세 번째 한계는 kernel 기반 discrepancy 추정과 mini-batch 통계의 안정성 문제다. CDD는 클래스별 통계를 써야 하므로, mini-batch 구성에 민감할 수 있다. CAS가 이를 완화하지만, 결국 충분한 batch diversity와 클래스별 샘플 수가 확보되어야 한다. 특히 클래스 수가 매우 많거나 long-tail 분포일 때 어떻게 동작하는지는 이 논문 텍스트만으로는 확인할 수 없다.

네 번째 한계는 적용 범위다. 논문은 image classification 기반 UDA에 초점을 맞추며, segmentation이나 detection 같은 더 복잡한 structured prediction 문제에 대한 직접 검증은 없다. 또한 backbone은 ResNet 계열로 제한되어 있고, 현대적인 self-supervised pretraining이나 transformer backbone과의 결합 가능성은 이 논문 시점상 다뤄지지 않았다.

비판적으로 보면, 이 논문은 "조건부 분포 정렬"과 "판별적 표현 학습"을 잘 결합했지만, target pseudo label 생성 자체를 더 강건하게 만드는 방향보다는 일단 clustering 품질이 어느 정도 괜찮다고 가정하는 부분이 있다. 또한 CDD의 성공이 얼마나 초기 source feature quality에 의존하는지, 혹은 domain gap이 훨씬 더 큰 상황에서도 안정적인지에 대한 심층 분석은 제한적이다. 그럼에도 불구하고, 제공된 근거 범위 안에서는 방법론적 설계와 성능 향상이 상당히 설득력 있다.

## 6. 결론

이 논문은 UDA에서 단순한 domain-level alignment가 아니라 class-aware alignment가 필요하다는 문제의식을 바탕으로, Contrastive Domain Discrepancy와 Contrastive Adaptation Network를 제안했다. 핵심은 같은 클래스의 source-target 분포 차이는 줄이고, 다른 클래스 사이의 차이는 키우는 것이다. 이를 통해 target feature를 더 discriminative하게 만들고, 잘못된 정렬과 불안정한 decision boundary 문제를 완화한다.

방법론적으로는 CDD라는 새로운 discrepancy metric, clustering 기반 target label hypothesis 생성, ambiguous target filtering, class-aware sampling, alternating optimization이 유기적으로 결합되어 있다. 결과적으로 Office-31과 VisDA-2017에서 strong baseline을 넘는 성능을 얻었고, ablation study를 통해 각 구성 요소의 효과도 입증했다.

이 연구의 중요한 의미는 UDA에서 "무조건 분포를 맞춘다"는 관점을 넘어서, "무엇과 무엇을 맞출 것인가"를 클래스 수준에서 더 정교하게 정의했다는 데 있다. 이후의 domain adaptation 연구에서 class-conditional alignment, pseudo label refinement, discriminative feature learning을 결합하는 흐름에 중요한 아이디어를 제공한 논문으로 볼 수 있다. 실제 적용 측면에서도 synthetic-to-real adaptation, 카메라/환경 변화 대응, 라벨이 부족한 신규 배포 환경 적응 같은 문제에 유용한 관점을 제공한다.
