# Adversarially Robust One-class Novelty Detection

- **저자**: Shao-Yuan Lo, Poojan Oza, Vishal M. Patel
- **발표연도**: 2021
- **arXiv**: <https://arxiv.org/abs/2108.11168>

## 1. 논문 개요

이 논문은 one-class novelty detection, 즉 정상 클래스 하나만을 학습한 뒤 입력이 그 정상 클래스에 속하는지 아니면 새로운 클래스에 속하는지를 판별하는 문제에서, 적대적 공격(adversarial attack)에 대한 강건성(adversarial robustness)을 다룬다. 저자들은 먼저 기존의 deep auto-encoder 기반 novelty detector들이 adversarial example에 매우 취약하다는 점을 보인다. 그리고 이미지 분류를 위해 제안된 대표적인 방어 기법들, 예를 들어 PGD-based adversarial training, feature denoising, self-supervised auxiliary task 기반 방법들이 novelty detection 맥락에서는 충분히 효과적이지 않다고 주장한다.

이 논문의 핵심 문제의식은 다음과 같다. novelty detector는 본질적으로 “정상 클래스만 잘 복원하고 비정상 클래스는 잘 복원하지 못하도록” 동작해야 한다. 그런데 adversarial perturbation이 입력에 들어오면, 정상 샘플의 reconstruction error를 부당하게 키우거나 비정상 샘플의 reconstruction error를 부당하게 줄여서 novelty score를 왜곡할 수 있다. 그 결과 정상 데이터가 이상으로, 이상 데이터가 정상으로 오분류된다. 이러한 실패는 보안 민감한 응용에서 특히 심각하다.

문제의 중요성은 실제 응용 범위에서 분명하다. anomaly detection, novelty detection, one-class classification은 제조 결함 탐지, 감시 영상 이상 탐지, 의료 이상 징후 탐지처럼 “이상 사례를 충분히 수집하기 어려운” 문제에서 널리 쓰인다. 이런 시스템이 공격에 취약하다면, 정상 샘플을 이상처럼 보이게 하거나 반대로 이상 샘플을 정상처럼 보이게 만들어 시스템 전체의 신뢰성을 깨뜨릴 수 있다. 따라서 이 논문은 단순히 novelty detection의 성능을 높이는 것이 아니라, adversarial 환경에서도 novelty score가 의미를 유지하도록 만드는 문제를 정면으로 다룬다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 novelty detector의 latent space를 “강하게 제약하고 정제해도” 성능을 유지할 수 있다는 task-specific property를 적극적으로 활용하는 데 있다. 일반적인 image classification에서는 latent representation이 여러 클래스를 구분할 수 있을 정도로 풍부한 semantic information을 유지해야 하므로, feature space를 과하게 압축하거나 조작하면 정확도가 쉽게 떨어진다. 반면 one-class novelty detection은 정상 클래스의 정보만 잘 보존하면 되므로, latent space를 훨씬 더 강하게 정리하고 제한할 수 있다.

저자들은 이 점을 이용해 Principal Latent Space, 즉 **PrincipaLS**라는 방어 모듈을 제안한다. 이 방법은 encoder가 만든 latent feature를 그대로 decoder에 넘기지 않고, incrementally-trained cascade PCA를 통해 principal latent subspace로 투영한 뒤 다시 복원하여 decoder에 넣는다. 이렇게 하면 adversarial perturbation으로 오염된 latent representation에서 정상 클래스의 주된 구조만 남기고 잡음 성분을 제거할 수 있다는 것이 논문의 직관이다.

좀 더 구체적으로 보면, PrincipaLS는 두 단계로 adversarial noise를 제거한다. 먼저 **Vector-PCA**가 채널 방향의 latent vector를 하나의 principal latent vector로 요약한다. 이 단계에서 원래 채널별 feature의 대부분이 하나의 주성분 방향으로 대체되므로, 채널 방향에 흩어진 perturbation이 크게 제거된다. 그 다음 **Spatial-PCA**가 spatial map 상에 남아 있는 residual perturbation을 다시 저차원 principal map으로 정제한다. 즉, 첫 번째 단계는 “vector space에서의 강한 정리”, 두 번째 단계는 “spatial domain에서의 추가 정제”에 해당한다.

기존 방법과의 차별점도 분명하다. 기존 방어는 주로 classification task를 위해 고안되었고, 입력 변환, feature denoising, adversarial training, self-supervised purification 등에 의존한다. 하지만 이 논문은 novelty detection의 목적 자체가 “정상 클래스 분포만 강하게 모델링하는 것”이라는 점을 이용하여 latent space를 훨씬 공격적으로 제한하는 방식을 채택한다. 즉, 방어를 일반적인 robustness 기법으로 보지 않고, novelty detection의 목적 함수와 표현 구조에 맞춘 task-specific defense로 설계했다는 점이 가장 큰 차별점이다.

## 3. 상세 방법 설명

논문의 기본 대상은 AE-style novelty detector이다. 입력 이미지 $\mathbf{X}$가 encoder $Enc$를 거쳐 latent representation $\mathbf{Z}$가 되고, decoder $Dec$가 이를 복원해 $\hat{\mathbf{X}}$를 만든다. novelty score는 보통 reconstruction error, 즉 $|\hat{\mathbf{X}}-\mathbf{X}|^2$로 계산된다. 정상 데이터는 잘 복원되어 낮은 score를 가져야 하고, 비정상 데이터는 잘 복원되지 않아 높은 score를 가져야 한다.

### 공격 문제 정식화

저자들은 novelty detection에 맞는 adversarial attack objective를 먼저 정의한다. PGD를 예로 들면 adversarial example은 다음과 같이 갱신된다.

$$
\mathbf{X}_{t+1} = Proj_{\mathbf{X},\epsilon}^{L_\infty} \left \{ \mathbf{X}_t + \alpha \cdot sign\left(\nabla_{\mathbf{X}_t}\mathcal{L}(\hat{\mathbf{X}}^t,\mathbf{X}_t,y)\right) \right \}
$$

여기서 $\hat{\mathbf{X}}^t = Dec(Enc(\mathbf{X}_t))$이고, $y \in {-1,1}$이다. $y=1$은 known class, $y=-1$은 novel class를 뜻한다. 손실은 다음과 같다.

$$
\mathcal{L}(\hat{\mathbf{X}}^t,\mathbf{X}_t,y)=
y , |\hat{\mathbf{X}}^t-\mathbf{X}_t|^2
$$

이 식의 의미는 명확하다. 정상 데이터($y=1$)에 대해서는 reconstruction error를 키우는 방향으로 공격하고, 비정상 데이터($y=-1$)에 대해서는 reconstruction error를 줄이는 방향으로 공격한다. novelty detection을 망가뜨리려면 바로 이 두 방향이 가장 자연스럽다. 저자들은 이 formulation이 기존 ARAE나 APAE 계열에서 사용한 더 제한적인 공격보다 강하다고 주장한다.

### PCA 기초 정의

논문은 PCA를 latent space 변환의 핵심 연산으로 사용한다. 데이터 행렬 $\mathbf{X}\in\mathbb{R}^{n\times d}$에 대해 mean vector $\boldsymbol{\mu}$와 principal component matrix $\mathbf{U}$를 구한다. 그중 앞의 $k$개 주성분만 유지한 $\tilde{\mathbf{U}}$를 사용하면, forward PCA와 inverse PCA는 다음처럼 정의된다.

$$
f(\mathbf{X};\boldsymbol{\mu},\tilde{\mathbf{U}})=
(\mathbf{X}-\boldsymbol{\mu})\tilde{\mathbf{U}}
$$

$$
g(\mathbf{X}_{PCA};\boldsymbol{\mu},\tilde{\mathbf{U}})=
\mathbf{X}_{PCA}\tilde{\mathbf{U}}^\top+\boldsymbol{\mu}
$$

즉, 원본 특징을 저차원 주성분 공간으로 보낸 뒤, 다시 원래 차원으로 복원할 수 있다. 이때 복원된 표현은 원본의 모든 정보를 보존하지 않지만, 주요 분산 방향의 구조는 유지한다. 저자들은 이 성질을 이용해 adversarial perturbation이 섞인 성분은 버리고 정상 클래스의 주요 구조만 남기려 한다.

### PrincipaLS 전체 구조

입력 adversarial image $\mathbf{X}_{adv}$가 encoder를 통과하면 latent space $\mathbf{Z}_{adv}=Enc(\mathbf{X}_{adv})\in\mathbb{R}^{s\times v}$를 얻는다. 여기서 $s=h\times w$는 spatial dimension이고, $v$는 channel 수에 해당하는 vector dimension이다. 논문은 encoder의 마지막 activation을 sigmoid로 바꾸어 latent value를 $[0,1]$ 구간으로 제한한다.

그 다음 PrincipaLS는 두 단계 PCA를 수행한다.

#### 1) Vector-PCA

먼저 latent space의 vector dimension, 즉 channel 방향에 대해 PCA를 수행한다. 평균 latent vector와 첫 번째 principal latent vector를 구한다.

$$
{\boldsymbol{\mu}_V,\tilde{\mathbf{U}}_V}=h_V(\mathbf{Z}_{adv}, k_V=1)
$$

논문은 항상 $k_V=1$로 둔다. 즉 채널 방향에서 단 하나의 principal vector만 남긴다. 이후 latent space는 다음처럼 Vector-PCA space로 투영된다.

$$
\mathbf{Z}_V = f_V(\mathbf{Z}_{adv};\boldsymbol{\mu}_V,\tilde{\mathbf{U}}_V)
$$

이 결과 $\mathbf{Z}_V \in \mathbb{R}^{s\times 1}$이 되어, 각 spatial 위치마다 “그 위치의 latent vector가 principal latent vector를 얼마나 스케일링하는가”만 남는다. 논문이 강조하는 방어 메커니즘은 여기서 나온다. 원래 adversarial perturbation이 채널별로 복잡하게 퍼져 있었더라도, 이제 각 위치는 하나의 principal vector 방향만 허용되므로 perturbation이 들어갈 자유도가 급격히 줄어든다.

#### 2) Spatial-PCA

이후 단일 채널 map인 $\mathbf{Z}_V$에 대해 spatial dimension 쪽 PCA를 수행한다. 평균 Vector-PCA map과 principal Vector-PCA maps를 계산한다.

$$
{\boldsymbol{\mu}_S,\tilde{\mathbf{U}}_S}=h_S(\mathbf{Z}_V^\top, k_S)
$$

그리고 이를 Spatial-PCA space로 변환한다.

$$
\mathbf{Z}_S^\top = f_S(\mathbf{Z}_V^\top;\boldsymbol{\mu}_S,\tilde{\mathbf{U}}_S)
$$

여기서 $k_S$는 hyperparameter이며, 실험에서는 $k_S=8$을 기본값으로 쓴다. 이 단계는 spatial domain에서의 추가적인 denoising 역할을 한다. 이미 Vector-PCA로 크게 정리된 표현에서, spatially coherent한 정상 구조만 더 남기고 residual perturbation을 줄이는 셈이다.

#### 3) 역변환과 principal latent space 복원

그 다음 inverse Spatial-PCA, inverse Vector-PCA를 통해 다시 원래 latent dimension으로 복원한다.

$$
\hat{\mathbf{Z}}_V^\top = g_S(\mathbf{Z}_S^\top;\boldsymbol{\mu}_S,\tilde{\mathbf{U}}_S)
$$

$$
\mathbf{Z}_{PrincipaLS}=g_V(\hat{\mathbf{Z}}_V;\boldsymbol{\mu}_V,\tilde{\mathbf{U}}_V)
$$

이렇게 얻은 $\mathbf{Z}_{PrincipaLS}$가 principal latent space이다. 최종 복원 이미지는

$$
\hat{\mathbf{X}}_{adv}=Dec(\mathbf{Z}_{PrincipaLS})
$$

로 계산되며, reconstruction error를 novelty score로 사용한다.

### Incremental training

PrincipaLS의 중요한 특징은 PCA basis를 고정된 사후처리로 얻는 것이 아니라, 학습 과정과 함께 **incrementally** 업데이트한다는 점이다. 즉 principal components ${\boldsymbol{\mu}_V,\tilde{\mathbf{U}}_V,\boldsymbol{\mu}_S,\tilde{\mathbf{U}}_S}$는 training step마다 exponential moving average 방식으로 갱신된다.

$$
{\boldsymbol{\mu}_V^t,\tilde{\mathbf{U}}_V^t}=
(1-\eta_V){\boldsymbol{\mu}_V^{t-1},\tilde{\mathbf{U}}_V^{t-1}}
+\eta_V \cdot h_V(\mathbf{Z}_{adv}^t)
$$

$$
{\boldsymbol{\mu}_S^t,\tilde{\mathbf{U}}_S^t}=
(1-\eta_S){\boldsymbol{\mu}_S^{t-1},\tilde{\mathbf{U}}_S^{t-1}}
+\eta_S \cdot h_S((\mathbf{Z}_V^t)^\top)
$$

여기서 $\eta_V, \eta_S$는 EMA learning rate이다. 이 구조의 의미는 network weights와 principal latent components가 서로 적응해 간다는 데 있다. encoder가 latent distribution을 바꾸면 PCA basis도 따라가고, 반대로 PCA basis가 latent를 제약하면 network weights도 그 구조에 맞게 학습된다. 저자들은 이를 mutual learning으로 해석한다.

### 방어 메커니즘 해석

논문은 PrincipaLS가 왜 방어가 되는지를 비교적 명확히 설명한다.

첫째, Vector-PCA 단계에서 모든 latent vector는 미리 학습된 principal latent vector의 scaling factor로 다시 표현된다. principal vector 자체는 학습된 parameter처럼 동작하며 adversarial perturbation에 직접 흔들리지 않는다. 따라서 perturbation이 실릴 수 있는 공간이 대폭 줄어든다.

둘째, 남은 perturbation은 단일 채널 map의 scaling factor들 안에만 존재할 수 있다. 그러면 Spatial-PCA가 이 작은 subspace를 다시 principal maps로 정제하여 residual adversary를 더 제거한다.

셋째, 이렇게 얻어진 principal latent space는 known class distribution에 더 가까운 representation이므로 decoder는 공격을 받아도 known class 스타일의 복원을 수행하게 된다. 그 결과 정상 데이터는 여전히 비교적 낮은 reconstruction error를 유지하고, 비정상 데이터는 known class로 억지 복원되면서 큰 reconstruction error를 낸다.

논문은 여기에 PGD-based adversarial training도 함께 사용한다. 다만 저자들의 주장은 “AT만으로는 부족하고, PrincipaLS 같은 latent purification이 함께 있어야 novelty detection에서 강건성이 크게 좋아진다”는 것이다.

## 4. 실험 및 결과

실험은 매우 광범위하게 구성되어 있다. 데이터셋은 MNIST, Fashion-MNIST, CIFAR-10, MVTec-AD, ShanghaiTech(SHTech) 다섯 가지를 사용한다. 이미지 기반 toy dataset부터 실제 anomaly detection dataset, 그리고 video anomaly detection dataset까지 포함해 범위를 넓혔다. 평가 지표는 주로 AUROC이며, MNIST, F-MNIST, CIFAR-10에서는 10개 known class 각각에 대해 one-class setting을 만들고 평균 AUROC(mAUROC)를 보고한다. MVTec-AD는 15개 category 평균, ShanghaiTech는 frame-level AUROC를 사용한다.

baseline defense는 PGD-AT, FD, SAT, RotNet-AT, SOAP, APAE를 포함한다. novelty detector backbone은 vanilla AE, VAE, AAE, ALOCC, GPND, ARAE, Puzzle-AE까지 총 7종이다. 저자들은 backbone architecture를 통일해 공정한 비교를 시도한다. encoder는 4개 convolution layer와 pooling으로 구성되고, decoder는 이를 mirror하는 구조다.

### 주요 결과 1: 기존 novelty detector는 공격에 매우 취약하다

Table I의 가장 중요한 메시지는 방어가 없는 경우 성능이 크게 붕괴한다는 점이다. 예를 들어 vanilla AE 기준으로 MNIST clean mAUROC는 0.964이지만, PGD 공격에서는 0.051까지 떨어진다. F-MNIST는 0.892에서 0.088, CIFAR-10은 0.550에서 0.034, MVTec-AD는 0.667에서 0.032, SHTech는 0.523에서 0.034 수준으로 내려간다. 이는 기존 novelty detector가 adversarial perturbation 앞에서 사실상 무력하다는 뜻이다.

### 주요 결과 2: 분류용 defense는 novelty detection에서 제한적이다

PGD-AT와 FD는 분명 개선을 주지만 충분하지 않다. 예를 들어 MNIST에서 PGD-AT는 PGD 공격 하 mAUROC를 0.357까지, FD는 0.366까지 올린다. 그러나 PrincipaLS는 0.706으로 훨씬 높다. F-MNIST에서는 0.368, 0.379에 비해 PrincipaLS가 0.613이다. CIFAR-10에서도 0.145, 0.147에 비해 0.246이다. 이 차이는 모든 데이터셋에서 반복된다.

특히 SAT나 RotNet-AT는 novelty detection 환경에서 추가적인 이점이 작거나 불안정하다. SOAP는 일부 설정에서 괜찮지만 전체적으로 일관되지 않다. APAE는 anomaly detection용으로 설계되었음에도 논문이 제안한 더 강한 공격 프로토콜에서는 상대적으로 약한 성능을 보인다.

### 주요 결과 3: PrincipaLS는 다양한 공격 전반에서 가장 일관되게 강하다

Table I에서 PrincipaLS는 FGSM, PGD, MI-FGSM, MultAdv, AF, black-box까지 거의 모든 경우에서 가장 높은 혹은 매우 높은 수준의 mAUROC를 보인다. 특히 MNIST에서는 평균 0.775, F-MNIST 0.673, CIFAR-10 0.309, MVTec-AD 0.337, SHTech 0.249를 기록한다. absolute value 자체는 dataset 난이도에 따라 낮을 수 있지만, 모든 방어 중 가장 일관되게 우수하다는 점이 핵심이다.

### 주요 결과 4: 다양한 novelty detector backbone에 일반화된다

Table II는 AE뿐 아니라 VAE, AAE, ALOCC, GPND, ARAE, Puzzle-AE에 PrincipaLS를 붙였을 때의 결과를 보여준다. 예를 들어 MNIST under PGD에서 VAE는 0.739, GPND는 0.741, ALOCC는 0.693 등으로 모두 다른 baseline defense보다 우수하다. F-MNIST에서도 전 backbone에서 0.599~0.629 수준으로 비교적 고르게 강건성을 높인다. CIFAR-10처럼 어려운 데이터셋에서도 거의 모든 backbone에서 약 0.24~0.25 수준을 달성한다. 이는 PrincipaLS가 특정 AE 구조에만 맞춘 trick이 아니라, AE-style novelty detector 전반에 꽤 보편적으로 붙을 수 있다는 근거다.

### 주요 결과 5: clean data 성능도 오히려 좋아진다

이 논문에서 인상적인 부분 중 하나는 robust defense가 clean performance를 희생하지 않는다는 주장이다. Table III에서 MNIST clean mAUROC는 No Defense 0.964에 비해 PrincipaLS 0.973, F-MNIST는 0.892에 비해 0.922, CIFAR-10은 0.550에 비해 0.578이다. 즉 PrincipaLS는 robustness만 높이는 것이 아니라 clean novelty detection 성능도 개선한다.

저자들의 해석은 이렇다. principal latent components는 known class latent space만을 기준으로 학습되므로, novel class image의 latent를 known class manifold 쪽으로 강제로 투영한다. 그러면 decoder는 novel image를 known class처럼 복원하려 하고, 그 결과 reconstruction error가 커진다. 반면 known class image는 원래부터 그 latent manifold와 잘 맞기 때문에 손상이 적다. 즉 정상과 이상의 reconstruction error gap이 clean setting에서도 더 잘 형성된다.

### 주요 결과 6: 계산 비용 증가가 비교적 작다

Table IV에 따르면 CIFAR-10 입력에서 No Defense는 $18.0\times10^3$ FPS, PrincipaLS는 $15.6\times10^3$ FPS로 약 13.3% 감소한다. FD는 62.2%, SOAP는 82.2%, APAE는 77.8% 감소하므로, PrincipaLS는 상당히 가벼운 편이다. 저자들은 backbone이 더 깊어질수록 상대적 overhead는 더 줄어들 것이라고 주장한다.

### 분석 실험

논문은 추가적인 분석도 풍부하다.

첫째, ablation study에서 Vector-PCA alone만 써도 PGD-AT보다 훨씬 좋아진다. MNIST 기준 PGD-AT 0.357, Vector-PCA 0.566, Vector-PCA+FD 0.582, full PrincipaLS 0.706이다. 이는 첫 번째 PCA만으로도 큰 효과가 있고, Spatial-PCA가 이를 더 밀어 올린다는 뜻이다.

둘째, $k_V$와 $k_S$의 trade-off를 분석한다. 일반적으로 더 큰 $k$는 더 많은 semantic information을 남겨 clean accuracy에는 유리하지만, 동시에 adversarial information도 남겨 robustness에는 불리하다. 그래서 novelty detection에서는 $k_V=1$, $k_S=8$이 가장 균형이 좋다고 결론 내린다.

셋째, attack iteration 수 $t_{max}$와 perturbation size $\epsilon$를 바꾸어도 PrincipaLS가 비교적 안정적으로 우세함을 보인다. 특히 $\epsilon$이 커질수록 공격 강도가 세지는 것은 자연스럽지만, 그 상황에서도 PrincipaLS가 baseline보다 낫다.

넷째, latent stability 측면에서 adversarial example과 clean counterpart의 latent L2 distance를 측정했을 때, PrincipaLS는 다른 defense보다 세 자릿수 이상 작다고 보고한다. 이는 latent space가 훨씬 덜 흔들린다는 정성적 근거다.

다섯째, reconstruction error histogram과 reconstructed image 사례도 제시한다. 방어가 없으면 공격 후 정상과 이상의 reconstruction error 분포가 뒤엉켜 novelty detection이 무너진다. PGD-AT는 정상 쪽은 어느 정도 회복하지만 이상 쪽 분리가 충분하지 않다. 반면 PrincipaLS는 anomalous sample을 known class 쪽으로 복원해 더 큰 reconstruction error gap을 유지한다는 것이 Figure 7과 Figure 8의 메시지다.

### adaptive / knowledgeable attack 평가

논문은 Athalye류의 obfuscated gradient 비판을 의식하여 추가 sanity check를 제시한다. iterative attack이 one-step attack보다 강하고, white-box가 black-box보다 강하며, distortion budget이 커질수록 공격이 더 강해진다는 점 등을 확인한다. 또한 PrincipaLS의 내부 메커니즘을 공격자에게 알려준 knowledgeable attack 두 종류를 설계하지만, 오히려 auxiliary loss가 커질수록 mAUROC가 증가했다고 보고한다. 저자들은 이를 근거로 “PrincipaLS를 안다고 해도 더 강한 공격을 쉽게 만들기 어렵다”고 주장한다.

다만 이 부분은 논문 내부 결과에 근거한 주장이지, 모든 adaptive attack 가능성을 완전히 배제했다는 뜻은 아니다. 논문도 특정 두 형태의 auxiliary loss를 시험한 것이다.

### 추가 논의: image classification에는 잘 맞지 않음

논문은 PrincipaLS를 CIFAR-10 image classification의 ResNet-18에도 붙여 본다. 결과는 예상대로 좋지 않다. clean accuracy와 PGD accuracy 모두 PGD-AT나 FD보다 낮다. 이는 novelty detection에서는 허용되던 aggressive latent compression이 classification에서는 semantic discrimination을 훼손하기 때문이다. 이 결과는 역설적으로 논문의 주장, 즉 “novelty detection에는 전용 방어가 필요하다”는 점을 강화한다.

### multiple class novelty detection 확장

또한 MNIST에서 digit 0과 digit 2를 정상 클래스로 두는 multiple class novelty detection 실험도 수행한다. 여기서도 PrincipaLS는 clean AUROC와 PGD AUROC를 모두 개선하며, PGD-AT나 FD보다 크게 낫다. 저자들은 이를 근거로 one-class뿐 아니라 multi-known-class novelty detection에도 가능성이 있다고 말한다. 다만 이 부분은 비교적 제한적인 설정 하나로만 제시되므로, 일반성은 추가 검증이 필요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 novelty detection의 고유한 구조를 정확히 짚고, 그에 맞는 defense mechanism을 설계했다는 점이다. 많은 adversarial defense 연구가 classification 중심으로 전개되어 왔는데, 이 논문은 novelty detection에서는 latent space를 더 강하게 제한해도 된다는 task-specific insight를 명시적으로 활용한다. 단순히 기존 defense를 가져다 쓰는 것이 아니라, 왜 그들이 충분히 맞지 않는지까지 실험으로 보여준 뒤 새로운 구조를 제안한 점이 설득력 있다.

또 다른 강점은 실험 범위가 넓다는 것이다. 공격 종류가 많고, 데이터셋도 toy image에서 real anomaly dataset, video anomaly dataset까지 포함하며, novelty detector backbone도 7개나 비교한다. 이 정도면 “특정 한 조건에서만 우연히 잘 된 방법”이라는 비판을 어느 정도 막아 준다. 특히 clean performance까지 좋아졌다는 결과는 실용적 관점에서 매우 매력적이다.

모듈이 가볍고 attachable하다는 점도 장점이다. decoder 앞에 latent purification 모듈을 붙이는 형태라서 AE-style architecture 전반에 비교적 쉽게 적용 가능하다는 인상을 준다. 실제로 Table II의 일반화 결과도 이를 뒷받침한다.

하지만 한계도 분명하다. 첫째, 논문은 AE-style reconstruction 기반 novelty detection에 강하게 의존한다. energy-based method, hypersphere-based one-class method, diffusion-based anomaly detector처럼 reconstruction이 중심이 아닌 계열에 대해서는 직접적인 적용 가능성이 명시되지 않는다. 따라서 “novelty detection 전체”에 대한 방어라고 일반화하기는 어렵다.

둘째, PrincipaLS의 핵심은 known class manifold로의 강한 투영인데, 이 방식은 정상 클래스 구조가 latent PCA로 비교적 잘 요약될 수 있을 때 유리하다. 정상 클래스가 매우 복잡하거나 multi-modal하며, 단일 혹은 제한된 principal component 구조로 잘 표현되지 않는 경우에는 정보 손실이 더 커질 수 있다. 논문은 one-class, 그리고 제한적 다중 클래스 설정에서는 효과를 보였지만, 더 복잡한 실제 분포에서의 실패 가능성은 충분히 논의하지 않는다.

셋째, adaptive attack에 대한 분석은 존재하지만 완전한 의미의 최강 adaptive evaluation이라고 보기는 어렵다. 논문은 두 종류의 PrincipaLS-knowledgeable attack을 제안해 실험했으나, adaptive attack space는 훨씬 넓다. 따라서 “공격자가 PrincipaLS를 알아도 conventional white-box attack이 가장 강하다”는 결론은 논문이 시도한 공격군 내에서는 맞지만, 일반적 보장으로 받아들이면 과할 수 있다.

넷째, 발표연도나 공식 arXiv URL이 제공된 텍스트에는 명시되지 않았고, 본문에서 Figure 3, Figure 4, Figure 5, Figure 6, Figure 7, Figure 8, Figure 9의 정량적 축값이나 세부 곡선은 텍스트만으로 완전히 확인할 수 없다. 따라서 이 보고서는 제공된 텍스트에 나온 설명과 표 수치를 중심으로 해석했으며, 그림의 시각적 세부 패턴까지는 추측하지 않았다.

비판적으로 보면, 이 논문의 가장 강한 메시지는 “novelty detection은 classification과 다른 robust defense가 필요하다”는 점이다. 이 메시지는 대체로 잘 전달된다. 다만 PrincipaLS가 왜 이론적으로 optimal한지, 혹은 PCA 기반 정제가 왜 다른 저차원 구조보다 근본적으로 우수한지에 대한 엄밀한 이론은 제공되지 않는다. 결국 논문의 설득력은 주로 empirical evidence에 기반한다. 그 empirical evidence는 상당히 강하지만, 더 일반적 이론 분석이 있었다면 더 완성도 높은 논문이 되었을 것이다.

## 6. 결론

이 논문은 one-class novelty detection이 adversarial attack에 매우 취약하다는 점을 체계적으로 보여주고, 이를 해결하기 위한 task-specific defense로 **PrincipaLS**를 제안한다. PrincipaLS는 encoder의 latent space에 대해 Vector-PCA와 Spatial-PCA를 연쇄적으로 적용하여 principal latent space를 만들고, 이 과정에서 adversarial perturbation을 제거하면서 known class distribution만을 강하게 유지한다. 또한 incremental training을 통해 PCA basis와 network weights를 함께 적응시켜 end-to-end 학습이 가능하도록 구성했다.

실험적으로는 다양한 white-box, black-box 공격과 여러 데이터셋, 여러 novelty detector backbone에서 강한 일관성을 보이며 기존 defense보다 우수한 adversarial robustness를 달성했다. 더 흥미로운 점은 clean performance도 종종 개선되었다는 것이다. 이는 novelty detection의 목적 자체와 잘 맞물리는 representation control이었기 때문에 가능한 결과로 해석된다.

실제 적용 측면에서 이 연구는 제조 불량 검출, 감시 이상 탐지, 희귀 이벤트 탐지처럼 anomaly/novelty detection이 중요한 시스템에서 의미가 크다. 향후 연구로는 reconstruction 기반이 아닌 다른 novelty detection 계열로의 확장, 더 강한 adaptive attack 평가, 더 복잡한 multi-modal 정상 분포에서의 검증, 그리고 video나 multimodal setting에서의 구조적 확장이 중요해 보인다. 제공된 텍스트에 근거하면, 이 논문은 adversarially robust novelty detection이라는 비교적 덜 탐구된 문제를 본격적으로 정식화하고 강한 baseline과 방법론을 제시한, 실험적으로도 설득력 있는 연구라고 평가할 수 있다.
