# Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation

* **저자**: Jian Liang, Dapeng Hu, Jiashi Feng
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2002.08546](https://arxiv.org/abs/2002.08546)

## 1. 논문 개요

이 논문은 **source data에 직접 접근하지 않고도 unsupervised domain adaptation(UDA)를 수행할 수 있는가**라는 매우 실용적인 질문을 다룬다. 기존 UDA 방법들은 대체로 adaptation 단계에서 source 데이터와 target 데이터를 함께 사용한다. 그러나 실제 환경에서는 source 데이터가 여러 장치에 분산되어 있거나, 개인정보 또는 민감정보를 포함하고 있어 재전송이 어렵고, 법적·운영적 이유로 외부 접근이 제한되는 경우가 많다. 저자들은 바로 이 지점을 문제로 삼고, **훈련된 source model만 제공되는 상황**에서 target domain에 적응하는 방법을 제안한다.

논문이 다루는 핵심 연구 문제는 다음과 같다. source 데이터가 전혀 없고, unlabeled target 데이터만 있으며, source 쪽에서 전달받을 수 있는 것은 오직 잘 학습된 모델 하나뿐일 때, 어떻게 그 모델의 지식을 target domain으로 옮길 수 있는가? 일반적인 feature distribution alignment 방식은 source feature distribution을 관찰할 수 있어야 작동하는데, 이 설정에서는 source 샘플 자체가 없기 때문에 직접적인 distribution matching이 불가능하다. 따라서 이 문제는 단순한 UDA 변형이 아니라, **source-free adaptation** 또는 **hypothesis transfer 기반 adaptation**이라는 새로운 제약 조건을 가진 어려운 문제다.

이 문제의 중요성은 명확하다. 논문에서도 예시로 들듯이 source dataset은 수십 MB에서 수천 MB까지 커질 수 있지만, source model은 훨씬 작다. 예를 들어 Digits에서는 source dataset이 33.2MB인데 source model은 0.9MB이고, VisDA-C에서는 source dataset이 7884.8MB인데 source model은 172.6MB이다. 즉, 모델만 전달하는 것은 통신 효율 측면에서 유리하고, 데이터 비공개 요구도 더 잘 만족시킨다. 병원 데이터, 감시 카메라 데이터, 개인 기기 데이터처럼 원천 데이터 공유가 곤란한 환경에서 특히 의미가 크다.

저자들은 이를 위해 **SHOT(Source HypOthesis Transfer)** 라는 간단하면서도 범용적인 representation learning framework를 제안한다. SHOT의 핵심은 source model을 **feature encoder**와 **classifier(hypothesis)** 로 분리해서 보고, adaptation 시에는 source classifier를 고정한 채 target-specific encoder만 학습하는 것이다. 즉, source data 자체가 아니라 **source hypothesis가 담고 있는 class 구조와 decision boundary 정보**를 활용해 target representation을 재배치한다. 이를 위해 정보 최대화(information maximization)와 self-supervised pseudo-labeling을 결합한다.

정리하면, 이 논문은 단순히 성능 개선만을 노린 것이 아니라, **실제 배포 가능한 UDA 설정**을 제안하고, 그 설정에서 작동하는 비교적 단순한 해법을 설계했으며, closed-set뿐 아니라 partial-set, open-set, multi-source, multi-target 등 여러 시나리오로 확장 가능함을 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음과 같다. source 데이터는 없지만, 잘 훈련된 source classifier는 여전히 source domain의 클래스 구조와 분리 기준을 내포하고 있다. 그렇다면 adaptation의 목표는 source와 target의 raw feature distribution을 직접 맞추는 것이 아니라, **target feature가 source classifier에 잘 들어맞도록 만드는 것**이라고 볼 수 있다. 즉, source classifier를 일종의 고정된 “판별 기준”으로 보고, target encoder를 조정하여 target 샘플들이 이 판별 기준 아래에서 잘 분리되도록 만드는 것이다.

기존 UDA 방법과의 가장 큰 차이는 바로 여기에 있다. 기존 방법들은 주로 다음 두 부류였다. 하나는 MMD 같은 moment matching을 통해 source와 target의 feature distribution을 통계적으로 맞추는 방법이고, 다른 하나는 domain discriminator를 이용해 adversarial하게 도메인 차이를 줄이는 방법이다. 두 방식 모두 adaptation 중에 source 데이터를 접근할 수 있어야 한다. 반면 SHOT은 source 데이터 없이도 작동하도록 설계되었다. 따라서 source-free라는 제약 자체가 기존 방식과 구조적으로 다르다.

SHOT의 두 번째 핵심 아이디어는 **information maximization(IM)** 이다. 저자들은 이상적인 target 출력이 두 가지 성질을 가져야 한다고 본다. 첫째, 각 샘플의 예측은 확신 있게 one-hot에 가까워야 한다. 둘째, 전체 target dataset 차원에서는 특정 클래스 하나로 몰리지 않고, 예측 분포가 다양해야 한다. 첫 번째만 있으면 모든 샘플이 같은 클래스로 가는 trivial solution이 발생할 수 있다. 두 번째만 있으면 각 샘플의 결정이 불안정해질 수 있다. 그래서 SHOT은 sample-wise entropy minimization과 dataset-wise diversity maximization을 동시에 사용한다.

세 번째 핵심 아이디어는 **self-supervised pseudo-labeling** 이다. 단순히 source classifier의 argmax 출력을 pseudo label로 쓰면 domain shift 때문에 잡음이 크다. 저자들은 이를 보완하기 위해 target domain 내부에서 클래스별 prototype, 즉 centroid를 만들고, 각 target 샘플을 가장 가까운 centroid에 할당하는 방식으로 pseudo label을 다시 정제한다. 이 방식은 target 데이터의 전역 구조(global structure)를 활용한다는 점에서, 단순한 prediction-based pseudo labeling보다 더 안정적이라고 주장한다.

결국 SHOT은 다음과 같은 설계 철학을 가진다. **source hypothesis는 고정된 지식 저장소로 두고, target encoder만 학습하여 target representation을 그 hypothesis에 맞춘다.** 여기에 정보 최대화로 “예측이 날카롭고 다양하게” 되도록 하고, self-supervised pseudo labels로 “틀린 방향으로 정렬되는 문제”를 줄인다. 이 조합이 SHOT의 핵심이다.

## 3. 상세 방법 설명

### 3.1 문제 설정

논문은 $K$-class classification 문제를 가정한다. source domain에는 라벨이 있는 데이터 ${(x_s^i, y_s^i)}_{i=1}^{n_s}$ 가 있고, target domain에는 라벨이 없는 데이터 ${x_t^i}_{i=1}^{n_t}$ 가 있다. 일반적인 UDA에서는 source 데이터와 target 데이터를 모두 활용할 수 있지만, 이 논문에서는 adaptation 시점에 source 데이터는 폐기되고 오직 학습된 source function $f_s$ 와 target 입력만 남는다.

목표는 source model $f_s:\mathcal{X}_s \to \mathcal{Y}_s$ 를 이용해 target function $f_t:\mathcal{X}_t \to \mathcal{Y}_t$ 를 학습하는 것이다. 이때 source와 target의 task는 본질적으로 같다고 가정하지만, 입력 분포는 다르다.

### 3.2 Source model 학습

먼저 source model은 일반적인 supervised learning 방식으로 학습한다. 기본 손실은 cross-entropy이다.

$
\mathcal{L}_{src}(f_s;\mathcal{X}_s,\mathcal{Y}_s) = -\mathbb{E}_{(x_s,y_s)} \sum*{k=1}^{K} q_k \log \delta_k(f_s(x_s))
$

여기서 $\delta_k(\cdot)$ 는 softmax의 $k$번째 출력이고, $q_k$ 는 정답 클래스에 대해서만 1인 one-hot label이다.

저자들은 source model 자체의 discriminability를 높이고 이후 target alignment를 돕기 위해 **label smoothing** 도 사용한다. label smoothing이 적용된 손실은 다음과 같다.

$
\mathcal{L}_{src}^{ls}(f_s;\mathcal{X}_s,\mathcal{Y}_s) = -\mathbb{E}_{(x_s,y_s)} \sum*{k=1}^{K} q_k^{ls} \log \delta_k(f_s(x_s))
$

여기서 smoothed label은

$
q_k^{ls}=(1-\alpha)q_k+\alpha/K
$

이며, 논문에서는 $\alpha=0.1$ 로 둔다.

이 단계의 목적은 단순히 source classifier를 만드는 것이 아니라, **target adaptation에 잘 활용될 수 있는 source hypothesis** 를 만드는 것이다.

### 3.3 모델 분해: encoder와 hypothesis

source model은 두 모듈로 나뉜다.

* feature encoding module: $g_s:\mathcal{X}_s \to \mathbb{R}^d$
* classifier module: $h_s:\mathbb{R}^d \to \mathbb{R}^K$

즉,

$
f_s(x)=h_s(g_s(x))
$

이다.

SHOT에서는 adaptation 시 target model도 같은 classifier를 사용한다. 즉,

$
h_t = h_s
$

로 두고, classifier는 **freeze** 한다. 대신 target encoder $g_t$ 만 학습한다. 최종 target model은

$
f_t(x)=h_t(g_t(x))
$

가 된다.

이 설계는 source hypothesis가 source domain의 class-level structure를 담고 있다는 가정에 기반한다. source data 자체는 볼 수 없지만, classifier의 weight와 decision surface는 source에서 어떤 특징이 어떤 클래스에 대응되는지에 대한 정보를 유지하고 있다. 그러므로 target encoder가 이 classifier에 맞는 표현을 만들 수 있다면 adaptation이 가능하다는 발상이다.

### 3.4 Information Maximization: SHOT-IM

source data가 없으므로 source feature distribution $p(g_s(x_s))$ 를 직접 추정해 target distribution과 맞출 수는 없다. 대신 저자들은 target 출력의 바람직한 형태를 직접 강제한다. 좋은 adaptation이 일어났다면, target 샘플들은 source classifier를 통과했을 때 **개별 샘플 수준에서는 확신 있게 분류**되고, **전체 데이터 수준에서는 한 클래스에 붕괴하지 않고 다양하게 분포**해야 한다.

이를 위해 두 개의 항을 사용한다.

첫 번째는 entropy minimization이다.

$
\mathcal{L}_{ent}(f_t;\mathcal{X}_t) = -\mathbb{E}_{x_t \in \mathcal{X}_t} \sum_{k=1}^{K} \delta_k(f_t(x_t)) \log \delta_k(f_t(x_t))
$

이 항은 각 샘플의 출력 분포가 날카롭고 one-hot에 가깝도록 만든다. 즉, 모델이 target 샘플에 대해 더 확신 있게 예측하도록 유도한다.

두 번째는 diversity-promoting term이다.

$
\mathcal{L}_{div}(f_t;\mathcal{X}_t) = \sum_{k=1}^{K}\hat{p}_k \log \hat{p}_k
$

여기서

$
\hat{p}=
\mathbb{E}_{x_t \in \mathcal{X}_t}[\delta(f_t(x_t))]
$

이다. 즉, target 전체에 대한 평균 softmax 출력이다. 이 항은 target 데이터 전체의 예측 분포가 한 클래스에 치우치지 않도록 만든다. 논문은 이를

$
D_{KL}(\hat{p}, \frac{1}{K}\mathbf{1}_K)-\log K
$

로도 쓸 수 있음을 보인다. 결국 평균 예측 분포가 균등분포에 가깝도록 유도하는 것이다.

이 두 항을 함께 쓰는 이유는 명확하다. entropy minimization만 쓰면 모든 샘플이 하나의 클래스에 강하게 몰리는 해가 가능하다. diversity term은 이 붕괴를 막는다. 그래서 SHOT-IM은 target feature가 source classifier에 맞게 정렬되도록 하지만, 동시에 전체 클래스 사용이 편중되지 않게 한다.

### 3.5 Self-supervised pseudo-labeling

정보 최대화만으로는 target feature가 잘 정렬되더라도 **잘못된 클래스 방향으로 정렬될 위험**이 있다. 논문은 t-SNE 시각화를 통해, SHOT-IM이 source-only보다 훨씬 나아지지만 여전히 일부 target 데이터가 잘못된 source hypothesis에 매칭될 수 있다고 설명한다.

이를 줄이기 위해 self-supervised pseudo-labeling을 도입한다.

#### 1단계: soft prediction을 이용한 초기 centroid 계산

먼저 현재 target model의 출력과 feature를 바탕으로 각 클래스의 초기 centroid를 계산한다.

$
c_k^{(0)} =
\frac{
\sum_{x_t \in \mathcal{X}_t}
\delta_k(\hat{f}_t(x_t)) , \hat{g}_t(x_t)
}{
\sum*{x_t \in \mathcal{X}_t}
\delta_k(\hat{f}_t(x_t))
}
$

이 식은 일종의 weighted k-means centroid와 비슷하다. 각 샘플이 클래스 $k$ 에 속할 soft probability를 weight로 사용해 centroid를 만든다. 직관적으로는 “현재 모델이 class $k$ 일 가능성이 높다고 보는 샘플들”의 평균 feature를 취하는 것이다.

#### 2단계: nearest centroid로 pseudo label 생성

그다음 각 target feature를 가장 가까운 centroid에 할당하여 pseudo label을 만든다.

$
\hat{y}_t = \arg\min_k D_f(\hat{g}_t(x_t), c_k^{(0)})
$

여기서 $D_f(a,b)$ 는 cosine distance이다.

즉, 단순히 classifier output의 argmax를 쓰는 것이 아니라, **target domain 내부에서 형성된 class prototype과의 거리**를 이용해 pseudo label을 만든다.

#### 3단계: hard pseudo label 기반 centroid 재계산

이후 새 pseudo label을 이용해 centroid를 다시 계산한다.

$
c_k^{(1)} = \frac{ \sum_{x_t \in \mathcal{X}_t} \mathds{1}(\hat{y}_t=k),\hat{g}_t(x_t) }{ \sum_{x_t \in \mathcal{X}_t} \mathds{1}(\hat{y}_t=k) }
$

그리고 다시

$
\hat{y}_t = \arg\min_k D_f(\hat{g}_t(x_t), c_k^{(1)})
$

로 pseudo label을 갱신한다.

논문은 centroid와 pseudo label을 여러 번 갱신할 수 있지만, 실험상 한 번만 업데이트해도 충분히 좋은 성능을 보인다고 한다.

핵심은 이 pseudo label이 source classifier의 즉각적인 noisy output에만 의존하지 않고, **target 데이터 자체가 가진 구조를 반영해서 정제된다**는 점이다. 그래서 저자들은 이를 self-supervised pseudo labels라고 부른다.

### 3.6 최종 학습 목적함수

최종적으로 SHOT은 고정된 classifier $h_t=h_s$ 와 학습 가능한 encoder $g_t$ 를 사용하며, 다음 손실을 최소화한다.

$
\mathcal{L}(g_t) = \mathcal{L}_{ent}(h_t \circ g_t;\mathcal{X}_t) + \mathcal{L}_{div}(h_t \circ g_t;\mathcal{X}_t) - \beta, \mathbb{E}_{(x_t,\hat{y}_t)} \sum_{k=1}^{K} \mathds{1}_{[k=\hat{y}_t]} \log \delta_k(h_t(g_t(x_t)))
$

마지막 항은 pseudo label을 정답처럼 사용한 cross-entropy 형태다. 식 앞의 부호가 음수이므로, 실제로는 pseudo label에 대해 예측 확률을 높이도록 학습한다. $\beta>0$ 는 균형 하이퍼파라미터이며, 논문에서는 대부분의 실험에서 $\beta=0.3$, Digits에서는 $\beta=0.1$ 을 사용한다.

즉, 최종 objective는 다음 세 요소로 구성된다.

첫째, 각 sample prediction을 날카롭게 만드는 entropy minimization.
둘째, 전체 target prediction 분포가 collapse하지 않도록 하는 diversity regularization.
셋째, self-supervised pseudo labels를 이용해 class-consistent feature structure를 강화하는 pseudo-label supervision.

### 3.7 알고리즘 흐름

논문 말미의 Algorithm 1을 따르면 학습 절차는 다음과 같다.

먼저 source model $f_s=g_s \circ h_s$ 를 준비한다. adaptation 시작 시에는 classifier를 고정하고, source encoder의 파라미터를 복사해 target encoder 초기값으로 사용한다. 그 후 각 epoch마다 현재 target model로부터 self-supervised pseudo labels를 생성하고, mini-batch 단위로 target 데이터를 샘플링하여 pseudo label과 함께 최종 loss로 $g_t$ 를 업데이트한다. 이 과정을 정해진 epoch 수만큼 반복한다.

즉, 학습은 매 epoch마다 “현재 representation으로 pseudo label 생성 → pseudo label을 이용한 encoder 갱신”의 반복 구조를 가진다.

### 3.8 네트워크 설계 선택

저자들은 source hypothesis의 품질이 adaptation 성능에 매우 중요하다고 보고, 몇 가지 구조적 선택도 함께 논의한다.

첫째, **weight normalization(WN)** 을 마지막 FC classifier layer에 사용한다. softmax에서 class weight vector의 norm이 분류에 영향을 주기 때문에, 각 클래스 weight의 norm을 통제하여 feature와 class prototype의 거리를 더 일관되게 만들려는 의도다.

둘째, **batch normalization(BN)** 을 bottleneck 뒤에 둔다. 저자들은 BN이 서로 다른 도메인의 평균과 분산을 정규화해 internal dataset shift를 줄이는 데 도움을 준다고 본다.

셋째, 앞서 말한 **label smoothing(LS)** 을 source 학습에 사용한다. 이는 source feature를 더 조밀하고 잘 분리된 cluster로 만들고, 이후 target alignment가 쉬워지도록 돕는다.

이 세 요소는 SHOT의 이론적 핵심은 아니지만, 실험에서 실제 성능 향상에 기여하는 중요한 구현 요소로 다뤄진다.

## 4. 실험 및 결과

### 4.1 실험 설정과 비교 대상

논문은 SHOT의 범용성을 검증하기 위해 다양한 UDA 시나리오를 평가한다. 데이터셋은 Digits, Office, Office-Home, VisDA-C, Office-Caltech 등이며, closed-set뿐 아니라 partial-set, open-set, multi-source, multi-target까지 포함한다.

Digits는 SVHN, MNIST, USPS 간 적응을 다루고, Office와 Office-Home은 object recognition의 대표적인 benchmark다. VisDA-C는 synthetic-to-real 적응으로 규모가 크고 난도가 높다. Office-Caltech는 multi-source 및 multi-target 평가에 사용된다.

비교 대상은 ADDA, CDAN, ADR, CyCADA, CAT, SWD, DANN, DAN, SAFN, BSP, TransNorm, IWAN, SAN, ETN, OSBP, STA, DADA, FADA 등 당시 대표적인 domain adaptation 방법들이다. 중요한 점은 이들 대부분이 adaptation 중 source data에 접근한다는 것이다. 따라서 SHOT은 더 불리한 정보 조건에서도 경쟁하거나 더 좋은 성능을 내는지를 보여주는 것이 핵심이다.

### 4.2 구현 세부사항

Digits에는 LeNet-5 계열 네트워크를 사용하고, object recognition에는 ResNet-50 또는 ResNet-101 backbone을 사용한다. object recognition에서는 마지막 FC 대신 256차원 bottleneck과 task-specific classifier를 두며, bottleneck 뒤에 BN, 마지막 FC에 WN을 넣는다.

최적화는 mini-batch SGD with momentum 0.9, weight decay $1e^{-3}$ 를 사용한다. 새로 추가된 layer는 backbone보다 10배 큰 learning rate로 학습한다. 기본 learning rate는 대부분 $1e^{-2}$ 이고, VisDA-C는 $1e^{-3}$ 이다. scheduler는

$
\eta = \eta_0 \cdot (1+10p)^{-0.75}
$

를 사용한다. batch size는 64이고, pseudo labels는 매 epoch 갱신한다.

논문은 target augmentation으로 ten-crop ensemble 같은 추가 기법은 쓰지 않았다고 명시한다. 따라서 결과는 비교적 단순한 평가 프로토콜 위에서 얻어졌다.

### 4.3 Digits 결과

Digits의 closed-set UDA에서 SHOT은 매우 강력한 성능을 보인다.

* S→M: SHOT 98.9, SHOT-IM 99.0, Oracle 99.4
* U→M: SHOT 98.0, SHOT-IM 97.6
* M→U: SHOT 97.9, SHOT-IM 97.7, Oracle 98.0
* 평균: SHOT 98.3, SHOT-IM 98.2

이 평균 98.3은 표에 있는 대부분의 기존 방법보다 높다. 예를 들어 SWD가 98.0, CDAN+E가 94.3, ADR이 93.8이다. source model only는 79.3에 불과하므로, source data 없이도 adaptation만으로 큰 폭의 향상을 얻었다는 점이 중요하다.

저자들은 특히 M→U에서 SHOT이 target-supervised oracle과 거의 비슷한 수준이라고 언급한다. 이는 source domain인 MNIST가 비교적 크고, 따라서 source hypothesis 자체가 강해 domain gap 완화에 유리하기 때문일 수 있다고 해석한다.

### 4.4 Office 결과

작은 규모의 Office benchmark에서는 평균 88.6을 기록한다. 이는 CDAN+TransNorm의 89.3보다는 약간 낮지만, source-only 79.3보다 훨씬 높다. 세부적으로는 D→A와 W→A 같은 더 어려운 task에서 좋은 결과를 보였다.

논문은 SHOT이 target domain이 너무 작을 때는 불리할 수 있다고 설명한다. 이유는 SHOT이 source classifier를 고정한 상태에서 target domain 자체의 구조를 이용해 pseudo labels와 representation을 학습해야 하므로, target 샘플 수가 적을수록 centroid 기반 self-supervision이 불안정할 수 있기 때문이다. 이 해석은 논문 본문에 직접 제시되어 있다.

### 4.5 Office-Home 결과

Office-Home은 논문의 가장 설득력 있는 결과 중 하나다. 평균 정확도는 다음과 같다.

* Source model only: 60.2
* SHOT-IM: 70.5
* SHOT(full): 71.8

기존 최고 성능으로 제시된 CDAN+TransNorm은 67.6이므로, SHOT은 약 4.2%p 향상시킨다. 또한 12개 task 중 10개에서 최고 성능을 기록했다고 논문은 설명한다. 특히 medium-sized target domain에서는 target 구조를 이용한 SHOT의 방식이 매우 효과적임을 보여준다.

이는 단순히 source-free setting에서도 “어느 정도 된다” 수준이 아니라, **당시 state-of-the-art를 넘어서는 결과**를 냈다는 점에서 중요하다.

### 4.6 VisDA-C 결과

VisDA-C에서는 per-class 평균 정확도 82.9를 달성한다.

* Source model only: 46.6
* SHOT-IM: 80.4
* SHOT(full): 82.9

기존 방법들인 SWD 76.4, SAFN 76.1, CDAN+BSP 75.9보다 높다. 특히 truck 클래스에서 좋은 성능을 보인다고 논문이 강조한다. source-only가 46.6에 그친 것을 보면, 이 데이터셋에서는 adaptation의 기여가 매우 크다.

또한 SHOT은 SHOT-IM보다 consistently 낫다. 이는 self-supervised pseudo-labeling이 단순 IM 위에 실제로 유의미한 개선을 더한다는 정성적 주장과 정량적 결과가 맞아떨어짐을 의미한다.

### 4.7 Ablation Study

Table 6은 이 논문의 핵심 요소들이 각각 얼마나 중요한지 보여준다.

Office / Office-Home / VisDA-C 평균을 보면,

* Source model only: 79.3 / 60.2 / 46.6
* naive pseudo-labeling: 83.0 / 64.1 / 76.6
* self-supervised PL: 87.6 / 68.9 / 80.7
* $\mathcal{L}_{ent}$ only: 83.5 / 55.5 / 63.3
* $\mathcal{L}_{ent}+\mathcal{L}_{div}$: 87.3 / 70.5 / 80.4
* $\mathcal{L}_{ent}+\mathcal{L}_{div}$ + naive PL: 87.5 / 70.3 / 82.9
* $\mathcal{L}_{ent}+\mathcal{L}_{div}$ + self-supervised PL: 88.6 / 71.8 / 82.9

여기서 중요한 관찰은 세 가지다.

첫째, naive pseudo-labeling보다 self-supervised pseudo-labeling이 항상 낫다. 이는 centroid 기반 정제가 실제로 noise를 줄여준다는 논문의 핵심 주장과 일치한다.

둘째, $\mathcal{L}_{ent}$ 만 사용하면 특히 Office-Home에서 55.5로 매우 낮다. 즉, entropy minimization alone은 불충분하고, 심지어 해를 망칠 수 있다.

셋째, $\mathcal{L}_{div}$ 를 함께 넣으면 성능이 크게 오른다. 이는 diversity-promoting objective가 collapse를 막고 IM의 실질적 성능을 뒷받침한다는 증거다.

Figure 3에서는 BN, WN, LS의 기여도도 본다. source model only는 BN의 도움을 특히 많이 받고, WN과 LS는 source-only에서는 때때로 감소를 보일 수 있지만 SHOT-IM까지 포함하면 전체적으로 상호보완적이라고 보고한다. 결국 세 요소를 모두 썼을 때 SHOT-IM이 가장 좋았다.

### 4.8 Partial-set과 Open-set 결과

SHOT은 closed-set만을 위한 방법이 아니라, 변형된 UDA 시나리오에도 확장된다.

#### Partial-set DA

Office-Home partial-set DA에서 평균 정확도는

* SAFN: 71.8
* SHOT-IM: 76.8
* SHOT: 79.3

으로 매우 크게 향상된다.

논문 부록에 따르면 partial-set에서는 target이 source의 일부 클래스만 포함하므로, closed-set에서 사용하던 diversity term $\mathcal{L}_{div}$ 가 적절하지 않다. 균등한 class usage를 강제하면 오히려 없는 클래스까지 활성화하려 들 수 있기 때문이다. 그래서 PDA에서는 $\mathcal{L}_{div}$ 를 제거하고, centroid가 너무 작은 경우 empty cluster처럼 버린다. 이 수정은 문제 구조에 잘 맞는 합리적 조정이다.

#### Open-set DA

Office-Home open-set DA에서는 평균 정확도

* STA: 69.5
* SHOT-IM: 71.5
* SHOT: 72.8

을 달성한다.

open-set에서는 target domain에 source에 없던 unknown class가 있다. classifier를 완전히 새로 학습하지 않고는 unknown을 직접 모델링하기 어렵기 때문에, 논문은 entropy 기반 uncertainty를 계산해 2-class k-means로 known/unknown을 나누는 thresholding 전략을 사용한다. uncertainty가 큰 cluster를 unknown으로 보고, 그 샘플들은 centroid 업데이트와 entropy objective 계산에서 제외한다. 이는 기존 SHOT 틀을 크게 바꾸지 않으면서 open-set으로 확장한 실용적 장치다.

### 4.9 Multi-source와 Multi-target 결과

Office-Caltech에서 SHOT은 multi-source와 multi-target에서도 최고 성능을 기록한다.

* Multi-source 평균: SHOT 97.7
* Multi-target 평균: SHOT 96.5

multi-source에서는 source별로 hypothesis를 따로 학습하고 각각 target에 적응시킨 뒤, 최종 score를 합산한다. multi-target에서는 여러 target domain을 단순 결합해 하나의 target처럼 취급한다. 여기서도 FADA 같은 federated DA 기반 접근보다 SHOT이 더 낫다.

이 확장은 흥미롭지만, 본문 설명은 비교적 간단하다. 세부적인 이론이나 더 정교한 결합 전략은 본문에 충분히 설명되어 있지 않다. 따라서 이 부분은 “SHOT의 기본 틀이 여러 시나리오에 확장 가능하다”는 실험적 증거로 이해하는 것이 적절하다.

### 4.10 Off-the-shelf source model에 대한 특수 사례

논문은 source model을 직접 학습할 수 없는 경우도 실험한다. ImageNet으로 사전학습된 ResNet-50만 있는 상황에서 ImageNet→Caltech partial DA를 수행한 결과,

* ResNet-50: 69.7
* ETN: 83.2
* SHOT-IM: 81.7
* SHOT: 83.3

을 얻었다.

즉, source dataset에 접근하지 못하고, 심지어 source model을 직접 학습한 것도 아니어도, SHOT은 여전히 강하게 작동할 수 있음을 보인다. 이것은 실사용 관점에서 매우 중요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정 자체가 매우 실용적이라는 점이다. 많은 domain adaptation 논문은 benchmark에서는 강하지만 실제 배포 환경을 충분히 반영하지 못한다. 반면 이 논문은 “source data를 접근할 수 없다”는 제약을 중심에 놓고, 프라이버시와 통신 효율 문제를 정면으로 다룬다. 이 설정은 today’s deployment setting과 잘 맞는다.

두 번째 강점은 방법이 비교적 단순하다는 점이다. SHOT은 generator-discriminator 구조처럼 복잡한 adversarial training을 요구하지 않는다. source classifier를 freeze하고 target encoder를 조정하는 구조는 직관적이며 구현도 어렵지 않다. 손실도 entropy, diversity, pseudo-label cross-entropy라는 익숙한 구성이다. 그럼에도 성능은 여러 benchmark에서 매우 강하다.

세 번째 강점은 self-supervised pseudo-labeling 설계다. 단순 argmax pseudo labeling보다 target structure를 더 잘 반영하는 centroid 기반 refinement를 도입해, domain shift로 인한 noisy label 문제를 완화했다. ablation이 이를 뒷받침한다. 특히 $\mathcal{L}_{div}$ 의 중요성을 명확히 보여준 점도 설득력이 있다.

네 번째 강점은 범용성이다. closed-set뿐 아니라 partial-set, open-set, multi-source, multi-target으로 확장되며, 심지어 off-the-shelf pre-trained model에도 적용 가능함을 보인다. 즉, SHOT은 특정 벤치마크용 특수 해법이 아니라 비교적 일반적인 틀로 보인다.

반면 한계도 분명하다. 가장 중요한 한계는 SHOT이 **source hypothesis가 충분히 좋은 구조를 담고 있다는 가정**에 크게 의존한다는 점이다. source classifier가 약하거나, source 학습이 불충분하거나, target과의 semantic gap이 너무 크면 frozen classifier가 오히려 adaptation의 병목이 될 수 있다. 논문은 source model 품질이 SHOT-IM 성능에 영향을 준다고 직접 언급한다.

또한 SHOT은 target domain 내의 구조를 이용해 pseudo label을 정제하므로, **target 데이터가 너무 작거나 불균형할 경우** 불안정할 수 있다. 실제로 Office처럼 작은 benchmark에서는 일부 task에서 최고 성능이 아니었고, 저자도 target domain이 작을 때 hypothesis learning이 어렵다고 해석한다.

또 다른 한계는 이론적 보장보다 경험적 성능에 무게가 실려 있다는 점이다. 왜 frozen classifier + target encoder optimization이 언제 잘 작동하는지에 대한 일반적 이론은 제시되지 않는다. 예를 들어 source hypothesis가 유지해야 할 성질, target shift의 종류별 민감도, pseudo-label refinement의 수렴 성질 등은 깊게 다뤄지지 않는다.

partial-set과 open-set 확장도 실용적이지만, 다소 heuristic하다. PDA에서는 diversity term을 제거하고 tiny centroid를 버리며, ODA에서는 entropy thresholding과 2-class k-means를 사용한다. 이는 잘 작동하지만, 보다 principled한 open-world modeling이라고 보기는 어렵다.

비판적으로 보면, SHOT은 “source data 없이도 adaptation이 가능하다”는 메시지를 강하게 전달하지만, 실제로는 **source classifier에 묻어 있는 source-domain class geometry를 최대한 재사용하는 방법**이다. 따라서 truly source-agnostic한 방법이라기보다는, source-trained decision boundary를 target 쪽에서 최대한 활용하는 방식이라고 보는 것이 정확하다. 그럼에도 논문이 설정한 문제에는 매우 잘 맞는 해법이다.

## 6. 결론

이 논문은 source data에 접근할 수 없는 현실적 제약 아래에서 unsupervised domain adaptation을 수행하는 새로운 관점을 제시한다. 핵심 기여는 **SHOT(Source HypOthesis Transfer)** 라는 framework를 통해, source classifier를 고정하고 target encoder만 학습하는 형태의 source-free UDA를 효과적으로 구현했다는 점이다. 이를 위해 information maximization으로 예측의 확신도와 다양성을 동시에 유도하고, centroid 기반 self-supervised pseudo-labeling으로 target 구조를 활용한 정제를 수행한다.

실험적으로 SHOT은 Digits, Office-Home, VisDA-C 등 여러 benchmark에서 매우 경쟁력 있는 성능을 보였고, 특히 medium-scale 및 large-scale 환경에서는 기존 state-of-the-art를 능가했다. 또한 partial-set, open-set, multi-source, multi-target, off-the-shelf pre-trained source model 등 다양한 변형 시나리오에서도 유연하게 확장되었다.

이 연구의 실제적 의미는 크다. 데이터 프라이버시, 분산 저장, 통신 비용 제약이 중요한 환경에서는 source data 자체를 재활용하기 어렵다. SHOT은 이럴 때 **모델만 전달하고도 adaptation이 가능하다**는 강한 가능성을 보여준다. 이후 source-free domain adaptation 분야가 활발히 전개된 점을 생각하면, 이 논문은 그 흐름을 촉진한 중요한 작업으로 볼 수 있다.

향후 연구 측면에서는 source hypothesis의 품질 조건, pseudo-label refinement의 안정성, unknown class 처리의 더 원리적인 설계, semantic gap이 큰 경우의 robustness 등을 더 정교하게 다룰 필요가 있다. 그럼에도 이 논문은 “source data가 꼭 필요하냐”는 질문에 대해, 적어도 많은 경우에는 “그렇지 않다”는 설득력 있는 답을 제시한 연구라고 평가할 수 있다.
