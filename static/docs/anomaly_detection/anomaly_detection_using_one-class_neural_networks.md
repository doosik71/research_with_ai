# Anomaly Detection using One-Class Neural Networks

- **저자**: Raghavendra Chalapathy (University of Sydney, CMCRC), Aditya Krishna Menon (Data61/CSIRO, ANU), Sanjay Chawla (QCRI, HBKU)
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1802.06360

## 1. 논문 개요

이 논문은 비지도 이상 탐지에서 오래 사용되어 온 One-Class SVM(OC-SVM)의 핵심 목적함수를 신경망 학습 안으로 직접 집어넣어, **표현 학습 자체가 이상 탐지에 맞게 조정되도록 만드는** 방법을 제안한다. 논문의 문제의식은 분명하다. 기존의 대표적 deep anomaly detection 접근 중 하나는 autoencoder나 사전학습 네트워크로 feature를 먼저 만들고, 그 feature를 별도의 이상 탐지기인 OC-SVM에 넣는 하이브리드 방식이다. 하지만 이 방식에서는 hidden representation이 이상 탐지 목적에 의해 직접 유도되지 않기 때문에, 표현은 일반적일 수 있어도 anomaly detection에 최적화되었다고 보기 어렵다.

저자들은 바로 이 지점을 연구 문제로 삼는다. 즉, **“복잡한 고차원 데이터에서 이상 탐지에 정말 유리한 표현을 어떻게 학습할 것인가?”**라는 질문에 대해, 단순히 feature extractor와 detector를 분리하지 않고, one-class objective가 hidden layer representation을 직접 밀어붙이도록 설계한다. 이 문제는 중요하다. 실제 이상 탐지는 사기 탐지, 침입 탐지, 고장 진단, 자율주행 안전성 등에서 핵심이며, 특히 이미지나 시계열처럼 결정 경계가 비선형적이고 데이터 차원이 높은 경우 shallow method만으로는 한계가 자주 드러난다.

이 논문의 핵심 주장은 다음과 같다. OC-NN은 deep network의 representation learning 능력과 one-class classification의 목적을 결합하여, 정상 데이터들을 원점(origin)으로부터 분리하는 초평면을 학습하도록 만든다. 이로 인해 hidden layer가 anomaly detection task에 특화된 방향으로 훈련된다. 저자들은 synthetic data, MNIST, CIFAR-10, 그리고 GTSRB adversarial stop sign 데이터셋에서 이 접근을 평가했고, 복잡한 데이터에서는 기존 shallow baseline보다 좋은 경우가 있으며, state-of-the-art deep one-class 계열과도 경쟁력 있는 성능을 보인다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 간단하지만 중요하다. 기존 OC-SVM은 feature space에서 정상 데이터가 원점과 분리되도록 초평면을 학습한다. 그런데 feature space 자체가 anomaly detection에 맞게 설계되어 있지 않으면, 초평면이 아무리 좋아도 성능에 한계가 있다. 반대로 deep network는 복잡한 비선형 표현을 만들 수 있지만, 그 표현이 reconstruction이나 일반 분류 목적에 맞춰져 있으면 이상 탐지에는 최적이 아닐 수 있다. 따라서 저자들은 **OC-SVM의 목적을 neural network 내부 표현과 직접 연결**한다.

구체적으로 보면, 전통적인 OC-SVM은 $ \langle w, \Phi(x_n) \rangle $ 형태의 score를 사용한다. 여기서 $\Phi(x)$는 kernel-induced feature mapping이다. 이 논문은 이 부분을 $ \langle w, g(Vx_n) \rangle $로 바꾼다. 즉, fixed feature map $\Phi(\cdot)$ 대신, 신경망의 hidden representation $g(Vx_n)$를 사용한다. 이 변화의 의미는 매우 크다. 이제 representation이 외부에서 주어진 것이 아니라, one-class objective를 줄이도록 직접 학습된다. 다시 말해, anomaly detection에 유리한 hidden feature가 만들어진다.

기존 접근과의 차별점도 바로 여기에 있다. 논문이 비판하는 hybrid OC-SVM 접근은 deep feature extraction과 anomaly detection이 분리되어 있다. autoencoder는 reconstruction을 잘 하도록 학습되고, 그 결과 얻은 hidden representation을 별도 OC-SVM에 넣는다. 그런데 이 구조에서는 anomaly detection 결과가 hidden layer 학습에 되먹임되지 않는다. OC-NN은 이 단절을 없애고, one-class loss가 representation learning을 직접 통제한다는 점에서 차별화된다.

또 하나의 아이디어는 최적화 방식이다. 제안 목적함수는 non-convex이므로 한 번에 전역 최적해를 구하기 어렵다. 저자들은 alternating minimization을 사용한다. 먼저 threshold 역할을 하는 $r$를 고정하고 network parameter를 학습한 뒤, 다시 현재 score 분포로부터 $r$를 갱신한다. 흥미로운 점은 이 $r$ 갱신이 단순한 수치최적화가 아니라 **score들의 $\nu$-quantile을 선택하는 문제와 동치**라는 것이다. 이 덕분에 알고리즘 구조가 매우 명확해진다.

## 3. 상세 방법 설명

논문은 가장 단순한 형태의 feed-forward neural network를 기준으로 설명한다. 기본 구조는 입력에서 hidden layer로 가는 가중치 행렬 $V$, hidden activation 함수 $g(\cdot)$, 그리고 hidden에서 출력으로 가는 가중치 $w$로 이루어진다. activation은 linear 또는 sigmoid를 사용할 수 있다고 적혀 있으나, 실험에서는 단일 hidden layer와 linear activation이 좋은 결과를 냈다고 보고한다.

전통적인 OC-SVM 목적함수는 다음과 같다.

$$
\min_{w,r} \frac{1}{2}|w|_2^2 + \frac{1}{\nu N}\sum_{n=1}^{N}\max\bigl(0, r - \langle w,\Phi(x_n)\rangle \bigr) - r
$$

여기서 $w$는 초평면의 방향, $r$은 bias 혹은 threshold 역할을 하는 값이며, $\nu \in (0,1)$는 얼마나 많은 점이 초평면 안쪽에 들어가도 되는지 조절하는 하이퍼파라미터다. 직관적으로는 허용할 이상치 비율 또는 false positive와 margin 사이의 trade-off를 조절한다.

논문은 이 식에서 kernel feature map $\Phi(x)$를 neural representation으로 대체한다. 그러면 OC-NN의 목적함수는 다음과 같이 된다.

$$
\min_{w,V,r} \frac{1}{2}|w|_2^2 + \frac{1}{2}|V|_F^2 + \frac{1}{\nu N}\sum_{n=1}^{N}\max\bigl(0, r - \langle w, g(Vx_n)\rangle \bigr) - r
$$

이 식을 항목별로 보면 다음과 같다.

첫째, $\frac{1}{2}|w|_2^2$는 출력 가중치의 regularization이다. 둘째, $\frac{1}{2}|V|_F^2$는 입력-은닉 가중치의 regularization이다. 셋째, hinge와 비슷한 $\max(0, r - \hat y_n)$ 항은 각 샘플의 score가 threshold $r$보다 작으면 패널티를 준다. 여기서

$$
\hat y_n = \langle w, g(Vx_n)\rangle
$$

이다. 즉, 정상 샘플은 가능한 한 $\hat y_n \ge r$가 되도록 밀어 올리고 싶다. 마지막의 $-r$ 항은 무작정 $r$를 작게 만들어 모든 샘플을 쉽게 만족시키는 것을 방지하고, 초평면을 원점에서 멀리 두도록 유도한다. 이는 OC-SVM의 기본 해석과 같다.

이 목적함수의 중요한 의미는 다음과 같다. 신경망은 단지 입력을 압축하는 것이 아니라, **정상 데이터가 큰 score를 받고 이상 데이터는 작은 score를 받는 방향으로 표현을 바꾸도록 훈련된다.** 이 점이 autoencoder 기반 reconstruction 방식과 가장 크게 다르다. reconstruction 기반 방법은 “입력을 얼마나 잘 복원하느냐”를 본다. 반면 OC-NN은 “정상 데이터가 one-class 기준에서 얼마나 안정적으로 분리되느냐”를 본다.

### 학습 절차

저자들은 alternating minimization을 사용한다.

먼저 $r$를 고정하고, $w$와 $V$를 backpropagation으로 학습한다. 이때 최적화되는 식은 다음처럼 생각할 수 있다.

$$
\arg\min_{w,V} \frac{1}{2}|w|_2^2 + \frac{1}{2}|V|_F^2 + \frac{1}{\nu N}\sum_{n=1}^{N}\ell(y_n,\hat y_n(w,V))
$$

여기서

$$
\ell(y,\hat y)=\max(0,y-\hat y), \qquad y_n = r, \qquad \hat y_n = \langle w,g(Vx_n)\rangle
$$

이다. 즉, 현재 threshold $r$보다 낮은 score를 받는 샘플에만 loss가 걸린다.

그 다음에는 현재의 $w, V$가 주어졌다고 보고 $r$를 업데이트한다. 이때의 문제는

$$
\arg\min_r \left( \frac{1}{\nu N}\sum_{n=1}^{N}\max(0, r-\hat y_n) \right) - r
$$

이다. 논문은 이 문제의 해가 단순히 ${\hat y_n}_{n=1}^N$의 $\nu$-quantile임을 보인다. 즉,

$$
r = \text{$\nu$-quantile of } {\hat y_n}_{n=1}^N
$$

이다.

이 결과는 꽤 직관적이다. $\nu$는 허용할 이상치 비율과 연결되므로, score 분포의 특정 분위수를 threshold로 잡는 것이 자연스럽다. 논문은 미분을 통해

$$
\frac{1}{N}\sum_{n=1}^{N}\mathbf{1}[\hat y_n < r] = \nu
$$

가 되어야 한다는 결론을 도출한다. 이는 곧 “$r$보다 작은 score가 전체의 $\nu$ 비율이 되도록 하라”는 뜻이고, 바로 $\nu$-quantile의 정의와 맞닿는다.

### 알고리즘 흐름

논문에 제시된 알고리즘을 쉬운 말로 정리하면 다음과 같다.

초기 threshold $r^{(0)}$를 정한다. 그다음 반복적으로 현재 $r$를 기준으로 network weight $(w,V)$를 backpropagation으로 학습한다. 학습된 network로 모든 샘플의 score $\hat y_n$를 계산하고, 이 score들의 $\nu$-quantile로 새로운 $r$를 설정한다. 이 과정을 수렴할 때까지 반복한다. 마지막에는 각 샘플에 대해

$$
S_n = \hat y_n - r
$$

를 계산하고, $S_n \ge 0$이면 정상, $S_n < 0$이면 이상으로 판단한다.

즉, decision score가 threshold보다 얼마나 위에 있는지를 보는 구조다. score가 음수이면 정상 집합의 경계 안에 충분히 들어오지 못했다는 뜻이므로 anomaly로 처리된다.

### 사전학습 autoencoder와의 관계

이 논문은 개념적으로는 raw input에도 적용 가능하다고 설명하지만, 실험에서는 먼저 deep autoencoder를 학습해 representation을 얻은 뒤, 그 encoder weights를 초기값으로 사용한다. 중요한 점은 이 encoder weights를 고정하지 않는다는 것이다. 다시 말해, pretrained encoder를 가져오지만 이후 OC-NN objective에 맞게 계속 미세조정한다. 따라서 단순한 하이브리드 방식과 달리 feature extractor가 anomaly objective의 영향을 받는다.

이 설계는 practical하다. 처음부터 완전히 무작위 초기화로 one-class objective만 학습하는 것보다, autoencoder 기반 representation으로 시작하면 optimization이 더 안정적일 수 있다. 동시에 fine-tuning을 허용하므로 anomaly detection에 맞는 방향으로 표현을 수정할 수 있다.

### 논문이 명확히 말하지 않는 부분

다만 이 논문은 one-hidden-layer feed-forward 구조로 식을 설명하면서, 실제 이미지 실험에서는 CNN encoder와 transfer된 representation을 함께 사용한다. 따라서 이론식과 실제 구현 사이에는 다소 추상화 수준 차이가 있다. 하지만 논문의 핵심은 “표현 $g(Vx)$가 anomaly objective의 직접적 제약을 받는다”는 데 있으므로, deeper architecture로의 일반화는 개념적으로 자연스럽다. 다만 deeper network 전체에 대해 어떤 최적화 안정성 문제가 있는지, initialization이 결과에 얼마나 민감한지 등은 본문에서 충분히 분석되지는 않는다.

## 4. 실험 및 결과

논문은 synthetic, MNIST, CIFAR-10, GTSRB의 네 종류 설정을 사용한다. 실험 목적은 간단한 데이터에서 제안식이 제대로 작동하는지 확인하고, 이어서 복잡한 이미지 및 adversarial example 탐지 상황에서도 경쟁력이 있는지 보는 것이다.

비교 대상은 OC-SVM/SVDD, Isolation Forest, KDE 같은 shallow method와, DCAE, AnoGAN, Soft-Bound Deep SVDD, One-Class Deep SVDD, RCAE 같은 deep anomaly detection method들이다. 구현은 OC-NN, DCAE, RCAE는 Keras와 TensorFlow 기반이고, OC-SVM과 Isolation Forest는 공개 구현을 사용했다고 적혀 있다.

평가 지표는 주로 AUC로 제시된다. 표에는 평균 AUC와 표준편차가 10개 random seed 기준으로 보고되어 있다.

### 데이터셋과 설정

Synthetic 데이터는 512차원에서 정상 190개, 이상 10개로 구성된다. 정상은 $\mu=0, \sigma=2$의 정규분포에서, 이상은 $\mu=0, \sigma=10$의 정규분포에서 생성했다. 분산이 훨씬 큰 점들을 이상으로 보는 단순하지만 직관적인 설정이다.

MNIST는 한 클래스를 정상으로 두고 나머지 9개 클래스를 이상으로 취급하는 one-class classification 설정이다. 학습은 정상 클래스 위주로 하고, 훈련 셋에 정상 클래스 수의 1% 수준의 anomaly를 포함시킨 것으로 설명되어 있다. 테스트에서는 여러 클래스가 섞여 있다.

CIFAR-10 역시 같은 방식인데, anomaly 비율은 정상 클래스 수의 10%로 두었다. 이미지 전처리는 global contrast normalization을 $L_1$ norm으로 수행하고, 이후 min-max scaling으로 $[0,1]$ 범위로 맞춘다.

GTSRB 실험은 더 응용적이다. stop sign 클래스를 정상으로 보고, Boundary Attack으로 만든 adversarial example을 이상으로 본다. 자율주행 안전성과 연결되는 설정이라는 점에서 의미가 있다. 여기서는 정상 1050개와 adversarial 100개를 합쳐 학습 데이터셋을 구성한다.

### 네트워크 구조

MNIST와 CIFAR-10에서는 LeNet 스타일 CNN을 사용한다. MNIST는 두 개의 convolutional module과 마지막 32-unit dense layer, CIFAR-10은 세 개의 convolutional module과 128-unit dense layer를 사용한다. GTSRB 역시 세 개의 convolutional module과 마지막 32-unit dense layer를 사용한다.

OC-NN의 feed-forward 부분에 대해서는 데이터셋별 best-performing hidden layer 구성이 별도 표로 제시된다. 예를 들어 Synthetic은 512 입력에 128 hidden, 출력 1이고, MNIST는 32 입력에 32 hidden, CIFAR-10은 128 입력에 32 hidden, GTSRB는 128 입력에 32 hidden과 optional layer 16이 사용되었다고 적혀 있다. 이 부분은 encoder output 차원과 feed-forward head 구조를 함께 요약한 것으로 보인다.

논문은 단일 hidden layer와 linear activation이 가장 좋은 결과를 냈다고 말한다. 또한 $\nu$는 각 데이터셋의 outlier proportion에 맞춰 설정했다고 밝힌다. 즉, threshold 분위수와 실제 anomaly 비율을 연결했다.

### Synthetic 결과

Synthetic 데이터에서는 OC-NN이 10개의 anomaly를 거의 확실하게 음수 decision score로 분리했다고 설명한다. 히스토그램 그림을 통해 anomalous point의 score가 정상과 분리되어 나타났으며, 고전적인 OC-SVM과 거의 동등한 성능을 보인다고 해석한다.

이 결과의 의미는 제안 목적함수 자체가 적어도 단순 setting에서는 one-class separation을 제대로 구현한다는 데 있다. 다만 이는 비교적 쉬운 문제이므로, 진짜 평가는 고차원 이미지 데이터에서의 일반화 성능이다.

### MNIST 결과

MNIST에서는 RCAE가 전체적으로 매우 강한 성능을 보인다. 표를 보면 각 정상 클래스 0~9에 대해 RCAE가 거의 모두 98~100에 가까운 AUC를 기록하며, OC-NN도 상당히 높지만 대부분 RCAE보다는 낮다. 예를 들어 class 0에서는 OC-NN이 97.60, RCAE가 99.92이고, class 2에서는 OC-NN이 87.32, RCAE가 98.01이다. 전반적으로 MNIST에서는 reconstruction 기반 robust autoencoder가 매우 강력하다는 점이 드러난다.

이는 MNIST가 구조가 단순하고 숫자 모양의 reconstruction이 비교적 안정적인 데이터이기 때문으로 해석할 수 있다. 즉, “잘 복원되는 정상 이미지 vs 잘 복원되지 않는 이상 이미지”의 구분이 잘 먹히는 상황에서는 RCAE 계열이 유리할 수 있다. OC-NN도 높은 성능을 내지만, 이 데이터셋에서는 최상위는 아니다.

### CIFAR-10 결과

CIFAR-10은 훨씬 어렵다. 전체적으로 AUC가 50대 후반에서 70대 초반 수준으로 내려간다. 이는 클래스 간 시각적 다양성이 크고, 한 클래스를 정상으로 두었을 때 다른 클래스가 반드시 “이상답게” 멀리 떨어져 있지 않기 때문이다.

표를 보면 RCAE가 여러 클래스에서 가장 높거나 강한 성능을 보이고, OC-NN은 일부 클래스에서 shallow baseline과 Deep SVDD 계열보다 낫다. 논문은 특히 Automobile, Bird, Deer처럼 global contrast가 약한 클래스에서 OC-NN이 shallow method와 Deep SVDD를 능가한다고 언급한다. 실제 표에서 Automobile은 OC-NN 61.97로 OCSVM/KDE 63.03보다는 약간 낮지만 Deep SVDD 계열보다는 낫고, Bird는 OC-NN 63.66으로 Deep SVDD 계열보다 높으며, Deer는 OC-NN 67.40으로 일부 deep one-class baseline보다 낫다. 반면 Frog, Truck처럼 전역 구조가 더 뚜렷한 클래스에서는 OC-SVM/SVDD 같은 shallow method가 더 강한 경우도 있다.

이 결과는 꽤 흥미롭다. 즉, deep one-class objective가 항상 shallow kernel method를 이기는 것은 아니다. 데이터의 전역 구조가 강하고 class-specific pattern이 비교적 단순하면, kernel 기반 방법이 여전히 경쟁력이 있다. 반대로 더 복잡하고 특징의 정교한 조정이 필요한 경우에는 OC-NN의 task-aware representation learning이 장점이 될 수 있다.

또한 논문은 One-Class Deep SVDD가 soft-boundary variant보다 두 데이터셋에서 약간 더 낫다고 말한다. 이는 deep one-class 계열 안에서도 경계를 어떻게 잡는지가 중요하다는 점을 보여 준다.

### GTSRB adversarial stop sign 결과

이 실험은 자율주행 맥락에서 가장 실용적이다. 목표는 stop sign 이미지 중 adversarial example을 탐지하는 것이다. 표에 따르면 성능은 다음과 같이 요약된다. OC-SVM/SVDD 52.5, KDE 51.5, IF 53.37, DCAE 79.1, Soft-Bound Deep SVDD 67.53, One-Class Deep SVDD 67.08, OC-NN 63.53, RCAE 87.39 정도다.

여기서는 OC-NN이 shallow method보다는 훨씬 낫지만, Deep SVDD보다도 약간 낮고 RCAE보다는 크게 뒤진다. 따라서 이 실험만 놓고 보면 제안 방법이 최고 성능이라고 할 수는 없다. 논문도 결과 서술에서 RCAE가 모든 다른 deep model보다 뛰어났다고 인정한다. OC-NN이 찾아낸 이상 샘플은 주로 perspective가 이상하거나 crop이 잘못된 이미지들로 설명된다.

이 결과는 중요한 메시지를 준다. adversarial example 탐지는 단순한 one-class separation만으로 충분하지 않을 수 있다. 픽셀 수준 왜곡, 국소 구조 변화, 재구성 가능성 같은 요소가 크게 작용할 수 있어 robust autoencoder가 더 강할 수 있다. 즉, OC-NN은 representation을 anomaly objective에 맞춘다는 장점이 있지만, 모든 anomaly 유형에서 최고의 선택은 아니다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의가 매우 명확하다는 점이다. hybrid 방식의 약점을 정확히 짚고, “representation learning이 anomaly objective의 영향을 받아야 한다”는 방향을 수학적 목적함수로 구현했다. 이는 단순한 응용이 아니라, OC-SVM의 구조를 neural network 형태로 재해석한 시도라는 점에서 의미가 있다. 특히 $r$ 업데이트가 $\nu$-quantile과 동치라는 정리는 알고리즘을 이해하기 쉽게 만들고, alternating minimization의 설계를 수학적으로 정당화한다.

또 다른 강점은 방법의 해석 가능성이다. 최종 score는 $S_n = \hat y_n - r$로 정의되고, 양수/음수에 따라 정상과 이상을 나눈다. reconstruction error처럼 간접적인 값이 아니라, one-class decision boundary에 대한 상대적 위치를 직접 본다는 점에서 의미가 분명하다. 또한 pretrained autoencoder를 활용하면서도 encoder를 고정하지 않고 fine-tune한다는 설계는 practical한 절충안이다.

실험적으로는 shallow method와 deep method를 폭넓게 비교했고, synthetic부터 image, adversarial setting까지 평가했다는 점이 장점이다. 제안법이 최소한 특정 복잡 데이터셋에서는 shallow baseline보다 유리하고, deep one-class 계열과도 경쟁할 수 있음을 보여 준다.

하지만 한계도 분명하다. 첫째, 논문 제목과 전체 주장만 보면 제안법이 state-of-the-art를 전반적으로 능가하는 듯 보일 수 있으나, 실제 표를 보면 그렇지 않다. MNIST와 GTSRB에서는 RCAE가 훨씬 강하고, CIFAR-10에서도 클래스별 우세가 엇갈린다. 따라서 OC-NN의 실증적 기여는 “항상 최고 성능”이라기보다, **one-class objective를 end-to-end representation learning과 결합한 새로운 방향의 유효성 제시**에 더 가깝다.

둘째, 목적함수는 non-convex이므로 전역 최적 보장이 없다. 논문도 이를 인정한다. 그런데 실제로 initialization, encoder pretraining quality, hidden dimension, activation choice에 얼마나 민감한지는 충분히 분석하지 않았다. 특히 실험에서는 autoencoder pretraining을 사실상 전제하고 있어, “순수한 OC-NN 자체의 힘”과 “좋은 initialization의 도움”이 얼마나 분리되는지 명확하지 않다.

셋째, 실험 설정에서 anomaly 비율 $\nu$를 실제 outlier proportion에 맞추어 주는 부분은 현실에서는 다소 강한 가정일 수 있다. 실제 비지도 이상 탐지에서는 anomaly 비율을 정확히 알기 어려운 경우가 많다. 따라서 $\nu$ misspecification에 얼마나 robust한지 분석이 있었다면 더 좋았을 것이다.

넷째, 방법 설명은 one-hidden-layer feed-forward network 중심인데, 실제 이미지 실험은 CNN encoder와 transfer learning 구조를 사용한다. 큰 틀에서는 일관되지만, 이론식과 최종 구현 사이의 간격이 있어 처음 읽는 독자에게는 다소 혼동될 수 있다.

비판적으로 보면, 이 논문은 이후 등장한 Deep SVDD류와 같은 deep one-class learning 흐름과 문제의식을 공유하면서도, “hyperplane separation from origin”이라는 OC-SVM 관점을 신경망화했다는 점이 특징이다. 다만 empirical superiority는 데이터셋에 따라 제한적이므로, 기여의 핵심은 성능 절대치보다 **설계 철학과 목적함수의 정식화**에 두는 것이 공정하다.

## 6. 결론

이 논문은 anomaly detection에서 deep representation learning과 one-class objective를 분리하지 말고 하나의 학습 문제로 묶어야 한다는 관점에서 출발한다. 제안된 OC-NN은 OC-SVM 유사 손실을 신경망 목적함수로 사용하여, hidden representation이 anomaly detection에 특화되도록 학습한다. 또한 alternating minimization을 통해 네트워크 가중치와 threshold $r$를 번갈아 최적화하고, $r$가 score들의 $\nu$-quantile과 동치임을 보였다.

실험 결과는 제안법이 shallow baseline보다 복잡한 데이터셋에서 유리한 경우가 있고, deep one-class 계열과도 경쟁력이 있음을 보여 준다. 다만 모든 실험에서 최고 성능을 내는 것은 아니며, 특히 MNIST와 GTSRB에서는 RCAE가 더 우수했다. 그럼에도 이 연구의 의미는 크다. anomaly detection에서 **표현 학습이 탐지 목적과 직접 연결되어야 한다**는 메시지를 명확하게 제시했고, 이후의 deep one-class learning 연구들과 연결되는 중요한 아이디어를 제공하기 때문이다.

실제 적용 측면에서는 고차원 이미지나 비선형 구조가 중요한 데이터에서, 단순한 shallow detector보다 task-aware deep representation이 필요한 상황에 유용할 가능성이 있다. 향후 연구로는 deeper architecture에 대한 안정적 학습, anomaly ratio를 모를 때의 robust optimization, self-supervised pretraining과의 결합, 그리고 adversarial example처럼 미세한 분포 변화에 대한 성능 향상이 중요한 후속 과제가 될 것이다.
