# Multi-Adversarial Domain Adaptation

* **저자**: Zhongyi Pei, Zhangjie Cao, Mingsheng Long, Jianmin Wang
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1809.02176](https://arxiv.org/abs/1809.02176)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 다룬다. 즉, 라벨이 있는 source domain 데이터와 라벨이 없는 target domain 데이터가 주어졌을 때, source에서 학습한 분류기가 target에서도 잘 동작하도록 만드는 것이 목표다. 문제의 핵심은 두 도메인의 데이터 분포가 다르다는 점이며, 이 차이를 보통 **domain shift** 또는 **dataset bias**라고 부른다.

기존의 deep domain adaptation 방법들, 특히 adversarial learning 기반 방법들은 source와 target을 구분하지 못하도록 feature를 학습함으로써 도메인 차이를 줄이려 했다. 대표적으로 단일 domain discriminator를 사용하는 방법은 전체 source 분포와 전체 target 분포를 맞추는 데 초점을 둔다. 하지만 실제 데이터 분포는 하나의 단순한 덩어리가 아니라 클래스별 혹은 하위 군집별로 나뉜 **multimode structure**를 가진다. 따라서 전체 분포만 거칠게 맞추면, 서로 대응되어야 할 클래스끼리 정렬되는 대신, 예를 들어 source의 cat이 target의 dog와 잘못 정렬되는 식의 **false alignment**가 발생할 수 있다.

이 논문은 바로 이 문제를 해결하기 위해 **MADA (Multi-Adversarial Domain Adaptation)** 를 제안한다. 핵심 생각은 하나의 discriminator로 전체를 맞추지 말고, **클래스별로 여러 개의 domain discriminator를 두어 더 세밀한 정렬을 수행하자**는 것이다. 이를 통해 저자들은 두 가지를 동시에 달성하려 한다. 첫째, 관련 있는 클래스 구조를 더 잘 맞추어 **positive transfer**를 강화한다. 둘째, 서로 관계없는 구조가 잘못 맞춰지는 것을 막아 **negative transfer**를 줄인다.

이 문제는 중요하다. 실제 응용에서는 target domain의 라벨을 충분히 확보하기 어렵기 때문에, source에서 배운 모델을 target으로 옮기는 기술이 매우 중요하다. 그런데 단순한 전역 정렬만으로는 실제 복잡한 클래스 구조를 제대로 반영하지 못하므로, 이 논문이 제안하는 fine-grained alignment는 domain adaptation의 실질적 성능 개선에 직접 연결된다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 매우 명확하다. **도메인 간 정렬은 전체 분포 수준이 아니라 클래스 혹은 모드 수준에서 이루어져야 한다**는 것이다. 기존 adversarial domain adaptation은 source와 target 전체를 하나의 discriminator로만 구분하지 못하게 만든다. 이 경우 도메인 구분은 사라질 수 있지만, 클래스 경계가 뒤섞일 위험이 있다. 논문 Figure 1은 바로 이 점을 직관적으로 보여준다. source의 한 클래스가 target의 전혀 다른 클래스와 정렬되면, feature는 domain-invariant일 수 있어도 분류에는 치명적이다.

MADA의 차별점은 **하나의 domain discriminator를 여러 개의 class-wise discriminator로 분해**한 데 있다. 클래스 수가 $K$개이면 discriminator도 $K$개를 둔다. 각 discriminator는 특정 클래스와 관련된 source/target 샘플을 정렬하는 역할을 맡는다. 그런데 target 데이터에는 라벨이 없으므로, 어떤 target 샘플이 어떤 discriminator에 들어가야 하는지 직접 알 수 없다. 이를 해결하기 위해 저자들은 **label predictor의 softmax 출력**을 사용한다. 즉, 어떤 샘플이 클래스 $k$일 확률 $\hat{y}_i^k$가 높으면, 그 샘플은 $k$번째 discriminator에 더 크게 반영된다.

이 설계는 hard assignment가 아니라 **soft assignment**라는 점이 중요하다. target 샘플이 아직 확실히 어느 클래스인지 모를 때도, 각 discriminator에 확률적으로 기여하게 만들 수 있다. 저자들이 강조하는 장점은 세 가지다. 첫째, target 데이터에 대한 부정확한 hard class assignment를 피할 수 있다. 둘째, 관련 없는 클래스 discriminator에는 낮은 가중치만 전달되므로 false alignment를 줄일 수 있다. 셋째, discriminator들이 서로 다른 파라미터를 가지면서 클래스별 정렬을 수행하기 때문에, 보다 정교한 multimode alignment가 가능하다.

즉, 이 논문의 본질은 **domain adversarial learning에 discriminative structure를 직접 통합**했다는 점에 있다. 기존 RevGrad가 “도메인을 못 구분하게 만드는 것”에 초점을 뒀다면, MADA는 “클래스별로 대응되는 구조끼리 도메인을 못 구분하게 만드는 것”에 더 가깝다. 이 차이가 성능 향상의 핵심 이유로 제시된다.

## 3. 상세 방법 설명

논문은 unsupervised domain adaptation 설정을 따른다. source domain은 $\mathcal{D}_s={(\mathbf{x}_i^s,\mathbf{y}_i^s)}_{i=1}^{n_s}$ 로 주어지며 라벨이 있다. target domain은 $\mathcal{D}_t={\mathbf{x}_j^t}_{j=1}^{n_t}$ 로 주어지며 라벨이 없다. source와 target은 서로 다른 joint distribution $P(\mathbf{X}^s,\mathbf{Y}^s)$ 와 $Q(\mathbf{X}^t,\mathbf{Y}^t)$ 에서 왔고, 일반적으로 $P \neq Q$이다. 목표는 feature extractor $G_f$ 와 label predictor $G_y$ 를 학습해 target risk를 줄이는 것이다.

전체 구조는 세 부분으로 이해할 수 있다. 먼저 **feature extractor** $G_f$ 가 입력 이미지 $\mathbf{x}$ 를 feature $\mathbf{f}=G_f(\mathbf{x})$ 로 변환한다. 다음으로 **label predictor** $G_y$ 가 feature를 받아 클래스 예측 $\hat{\mathbf{y}}=G_y(\mathbf{f})$ 를 만든다. 마지막으로 **여러 개의 domain discriminator** $G_d^1,\dots,G_d^K$ 가 각 클래스별로 source인지 target인지 구분하도록 학습된다. feature extractor는 discriminator들을 속이도록 반대로 학습된다. 이 adversarial 구조는 Gradient Reversal Layer (GRL)로 구현된다.

먼저 비교 기준이 되는 기존 domain adversarial network의 목적함수는 다음과 같다.

$$
C_0(\theta_f,\theta_y,\theta_d) = \frac{1}{n_s}\sum_{\mathbf{x}_i \in \mathcal{D}_s} L_y\big(G_y(G_f(\mathbf{x}_i)), y_i\big) - \frac{\lambda}{n}\sum_{\mathbf{x}_i \in (\mathcal{D}_s \cup \mathcal{D}_t)} L_d\big(G_d(G_f(\mathbf{x}_i)), d_i\big)
$$

여기서 첫 번째 항은 source 라벨 분류 loss이고, 두 번째 항은 domain discrimination loss이다. $n=n_s+n_t$ 이고, $\lambda$ 는 두 목적 사이의 trade-off를 조절하는 하이퍼파라미터다. 학습은 saddle-point 문제로 이루어진다. 즉, $\theta_f,\theta_y$ 는 전체 목적함수를 최소화하고, $\theta_d$ 는 domain loss를 최대화하는 방향으로 움직인다. feature extractor는 domain 정보를 지우고, discriminator는 그것을 찾으려 하기 때문에 adversarial game이 형성된다.

문제는 이 방식이 오직 **하나의 discriminator**만 사용한다는 것이다. 따라서 전체 분포는 맞출 수 있어도, 내부 multimode structure는 반영하지 못한다. 이를 해결하기 위해 MADA는 클래스별 domain discriminator를 도입한다.

논문의 핵심 수식은 식 (3)과 식 (4)이다. 먼저 domain loss를 클래스별로 나누면,

$$
L_d = \frac{1}{n} \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in \mathcal{D}_s \cup \mathcal{D}_t} L_d^k\big(G_d^k(\hat{y}_i^k G_f(\mathbf{x}_i)), d_i\big)
$$

가 된다.

이 식의 의미를 쉽게 설명하면 다음과 같다.

각 샘플 $\mathbf{x}_i$ 는 label predictor를 통해 클래스 확률 벡터 $\hat{\mathbf{y}}_i$ 를 얻는다. 그중 $k$번째 클래스에 대한 확률이 $\hat{y}_i^k$ 이다. 이 확률을 feature $G_f(\mathbf{x}_i)$ 에 곱해 $k$번째 discriminator의 입력으로 넣는다. 즉, 어떤 샘플이 클래스 $k$ 일 가능성이 높을수록 $k$번째 discriminator 학습에 더 큰 비중으로 기여한다.

이 설계는 사실상 attention처럼 동작한다. 논문도 이 점을 직접 언급한다. 샘플이 여러 클래스 discriminator에 동시에 soft하게 참여하되, 자신과 관련이 낮은 discriminator에는 거의 영향을 주지 않는다. 따라서 target 데이터처럼 라벨이 없는 상황에서도 확률적이고 유연한 class-conditional alignment가 가능하다.

이를 포함한 MADA의 전체 목적함수는 다음과 같다.

$$
C(\theta_f,\theta_y,\theta_d^k|_{k=1}^{K}) = \frac{1}{n_s} \sum_{\mathbf{x}_i \in \mathcal{D}_s} L_y\big(G_y(G_f(\mathbf{x}_i)), y_i\big) - \frac{\lambda}{n} \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in \mathcal{D}} L_d^k\big(G_d^k(\hat{y}_i^k G_f(\mathbf{x}_i)), d_i\big)
$$

여기서 $\mathcal{D}=\mathcal{D}_s \cup \mathcal{D}_t$ 이다. 최적화 문제는 다음과 같은 minimax 형태다.

$$
(\hat{\theta}_f,\hat{\theta}_y) = \arg\min_{\theta_f,\theta_y} C(\theta_f,\theta_y,\theta_d^k|_{k=1}^{K})
$$

$$
(\hat{\theta}_d^1,\dots,\hat{\theta}_d^K) = \arg\max_{\theta_d^1,\dots,\theta_d^K} C(\theta_f,\theta_y,\theta_d^k|_{k=1}^{K})
$$

직관적으로 보면, label predictor는 source 분류를 잘해야 하고, feature extractor는 각 클래스별 discriminator가 source/target을 구분하지 못하게 해야 한다. 그러나 이때 모든 데이터를 하나로 섞는 것이 아니라, 각 샘플이 “자신이 속할 가능성이 높은 클래스들”에만 맞춰져 정렬된다.

학습 절차는 mini-batch SGD로 수행된다. 구현은 Caffe 기반이며, AlexNet과 ResNet의 ImageNet pre-trained model에서 fine-tuning한다. convolution/pooling layer도 fine-tuning하고, classifier layer는 scratch에서 학습하므로 learning rate를 lower layer보다 10배 크게 둔다. learning rate schedule은 RevGrad와 같은 progressive schedule을 사용한다.

학습률은

$$
\eta_p = \frac{\eta_0}{(1+\alpha p)^\beta}
$$

로 조정하며, $p$ 는 0에서 1까지 선형적으로 증가하는 training progress, $\eta_0=0.01$, $\alpha=10$, $\beta=0.75$ 이다.

또한 도메인 adversarial 강도를 조절하는 $\lambda$ 는 논문 중간에서는 실험 전체에 대해 $\lambda=1$로 고정한다고 설명하지만, 학습 초반 noisy activation을 줄이기 위해 실제 SGD에서는 다음과 같은 점진적 스케줄을 곱해 사용했다고 적고 있다.

$$
\frac{2}{1+\exp(-\delta p)} - 1
$$

여기서 $\delta=10$ 이다. 즉, 실제 학습에서는 adversarial 효과를 초반에는 약하게 두고 점차 강화해 안정성을 높인다. 이 부분은 문장상 다소 혼동될 수 있지만, 논문이 말하는 바는 “기본 하이퍼파라미터로서 $\lambda$ 는 1로 두되, 훈련 안정화를 위해 adversarial weight를 progress-dependent schedule로 조절한다”는 취지로 이해하는 것이 타당하다.

계산 복잡도에 대해서는, discriminator 수가 늘어나지만 각 discriminator가 전체 네트워크에서 차지하는 비율이 작기 때문에 전체 복잡도는 RevGrad와 유사하다고 주장한다. 다만 구체적인 FLOPs나 wall-clock time 수치는 본문에 제시되지 않았다.

## 4. 실험 및 결과

논문은 두 개의 대표적 visual domain adaptation 벤치마크에서 MADA를 평가한다. 첫째는 **Office-31**, 둘째는 **ImageCLEF-DA**이다.

Office-31은 Amazon (A), Webcam (W), DSLR (D) 세 도메인으로 구성되며 총 4,652장, 31개 클래스이다. 실험 태스크는 $A \rightarrow W$, $D \rightarrow W$, $W \rightarrow D$, $A \rightarrow D$, $D \rightarrow A$, $W \rightarrow A$ 의 6개다. 이 데이터셋은 도메인 크기와 난이도가 서로 다르기 때문에 domain adaptation의 표준 벤치마크로 자주 쓰인다.

ImageCLEF-DA는 Caltech-256 \(C), ImageNet (I), Pascal VOC (P) 세 도메인에서 공통 12개 클래스를 추출해 만든 데이터셋이다. 각 도메인과 각 클래스의 이미지 수가 균형적이어서, Office-31과는 다른 성격의 평가를 제공한다. 실험 태스크는 $I \rightarrow P$, $P \rightarrow I$, $I \rightarrow C$, $C \rightarrow I$, $C \rightarrow P$, $P \rightarrow C$ 의 6개다.

비교 방법은 shallow transfer learning의 TCA, GFK와 deep transfer learning의 DDC, DAN, RTN, RevGrad이다. shallow method에는 SVM을 사용하며, deep method는 AlexNet 또는 ResNet backbone을 기반으로 한다. 평가 지표는 target domain classification accuracy이며, 세 번의 random experiment 평균과 standard error를 보고한다.

Office-31 결과를 보면, AlexNet backbone에서 MADA의 평균 정확도는 **77.1%** 로 RevGrad의 **74.1%**, RTN의 **73.7%**, DAN의 **71.7%** 보다 높다. 특히 어려운 태스크에서 개선 폭이 크다. 예를 들어 $A \rightarrow W$ 에서 RevGrad는 73.0%인데 MADA는 **78.5%** 이고, $D \rightarrow A$ 에서는 RevGrad 52.4% 대비 MADA **56.0%**, $W \rightarrow A$ 에서는 RevGrad 50.4% 대비 MADA **54.5%** 이다. 이는 도메인 차이가 큰 어려운 전이에서 multimode-aware alignment가 더 효과적임을 보여준다.

ResNet backbone에서는 개선이 더 크게 나타난다. 평균 정확도는 RevGrad **82.2%** 대비 MADA **85.2%** 다. 특히 $A \rightarrow W$ 에서 82.0%에서 **90.0%** 로 크게 상승하고, $A \rightarrow D$ 도 79.7%에서 **87.8%** 로 상승한다. 이 결과는 backbone이 강해질수록 class-wise adversarial alignment의 이점이 더 잘 드러날 수 있음을 시사한다.

ImageCLEF-DA에서도 MADA는 대부분의 태스크에서 최고 혹은 최고 수준의 성능을 보인다. AlexNet 기반 평균 정확도는 RevGrad **78.2%** 대비 MADA **79.8%** 이고, ResNet 기반 평균 정확도는 RevGrad **85.0%** 대비 MADA **85.8%** 이다. 절대적인 개선 폭은 Office-31보다 작지만, 이 데이터셋은 클래스 균형이 잘 맞춰져 있어 본래 adaptation이 상대적으로 덜 어렵다는 점을 감안할 수 있다.

논문은 **negative transfer** 상황을 따로 검증한 점이 중요하다. Office-31의 31개 source 클래스 중 target에는 25개 클래스만 남도록 6개 클래스를 제거하여, source에 target과 무관한 클래스가 존재하도록 더 어려운 설정을 만든다. 이 경우 adversarial alignment가 잘못 작동하면 source의 irrelevant class가 target에 강제로 맞춰져 negative transfer가 발생하기 쉽다.

이 설정에서 Table 3 결과는 매우 인상적이다. AlexNet baseline의 평균은 **68.4%**, RevGrad는 오히려 **66.6%** 로 baseline보다 나빠진다. 즉, 단일 discriminator adversarial adaptation이 negative transfer를 실제로 유발할 수 있음을 보여준다. 반면 MADA는 **73.7%** 로 두 방법 모두를 크게 앞선다. 예를 들어 $A \rightarrow W$ 에서 AlexNet 58.2%, RevGrad 65.1%에 비해 MADA는 **70.8%** 이고, $W \rightarrow A$ 에서는 AlexNet 47.3%, RevGrad 42.9%, MADA **54.2%** 이다. 이 실험은 MADA의 가장 강력한 실증적 근거 중 하나다. 논문이 주장하는 “negative transfer 완화”가 단순 구호가 아니라, 불일치 클래스가 존재하는 더 일반적인 상황에서 실제 성능 이점으로 이어진다는 점을 보여준다.

정성 분석도 제시된다. Figure 3의 t-SNE 시각화에서 RevGrad는 source와 target을 섞는 데는 성공하지만 클래스 간 분리가 충분하지 않다고 해석된다. 반면 MADA는 도메인 정렬과 클래스 분리를 동시에 더 잘 보인다고 주장한다. 이는 MADA가 단순히 domain-invariant feature만 만드는 것이 아니라, **discriminative하면서도 transferable한 feature** 를 만든다는 저자들의 해석을 뒷받침한다.

또한 논문은 proxy $\mathcal{A}$-distance를 이용해 분포 차이를 측정한다. 이는 source/target 구분 classifier의 일반화 오차 $\epsilon$ 으로부터

$$
d_{\mathcal{A}} = 2(1-2\epsilon)
$$

로 계산된다. Figure 4(b)에서 MADA feature의 $d_{\mathcal{A}}$ 가 ResNet과 RevGrad보다 더 작게 나타난다고 보고한다. 즉, MADA가 cross-domain discrepancy를 더 효과적으로 줄였다는 해석이다.

Sharing strategy 분석도 있다. 여러 discriminator 사이에서 파라미터를 얼마나 공유할지 비교한 결과, discriminator 파라미터를 많이 공유할수록 성능이 떨어졌다. 이는 이 논문의 핵심 주장을 강화한다. 즉, 다중 discriminator를 두더라도 결국 같은 파라미터를 공유해버리면 fine-grained alignment 능력이 약화되고, 진정한 class-wise alignment의 장점이 줄어든다는 것이다.

마지막으로 convergence 분석에서 MADA는 RevGrad와 비슷한 수준의 안정적 수렴 특성을 보이면서도 전 과정에서 더 낮은 test error를 유지한다고 보고한다. 다만 수렴 속도나 시간 복잡도에 대한 정량적 수치는 제시되지 않고, 주로 곡선과 서술로 설명한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 매우 잘 맞아떨어진다는 점이다. domain adaptation에서 단순 전역 정렬만으로는 부족하며, multimode structure를 고려해야 한다는 문제의식이 분명하고, 이를 class-wise discriminator라는 비교적 직관적이고 구현 가능한 방식으로 해결했다. 아이디어가 단순한 편이지만, 왜 필요한지와 어떤 효과를 기대하는지가 명확하다.

또 다른 강점은 **negative transfer**를 실험적으로 정면으로 다뤘다는 점이다. 많은 domain adaptation 논문이 평균 accuracy 개선만 보여주고 끝나는 반면, 이 논문은 source와 target label space가 부분적으로 불일치하는 더 어려운 상황에서 기존 adversarial adaptation이 실패할 수 있음을 보여준다. 그리고 MADA가 이 상황을 더 잘 견딘다는 점을 Table 3로 증명한다. 이는 실용적으로도 중요하다. 실제 응용에서는 source와 target의 클래스 구성이 완전히 일치하지 않을 가능성이 높기 때문이다.

세 번째 강점은 기존 adversarial adaptation과의 관계가 명료하다는 점이다. MADA는 RevGrad류 방법을 완전히 버리는 것이 아니라, 그 위에 discriminative structure를 추가하는 방향으로 확장한다. 따라서 개념적으로 이해하기 쉽고, 기존 프레임워크에 통합하기도 상대적으로 수월하다.

하지만 한계도 있다. 첫째, 이 방법은 **label predictor의 신뢰도**에 상당히 의존한다. target 데이터는 라벨이 없으므로, 각 샘플이 어떤 class-wise discriminator에 얼마나 기여할지 결정하는 정보는 전적으로 $\hat{\mathbf{y}}$ 에서 나온다. 만약 초반 예측이 많이 틀리면, 잘못된 discriminator에 샘플이 흘러들어갈 수 있다. 논문은 soft assignment와 progressive training으로 이를 완화한다고 보지만, 이 문제가 완전히 해소되었다고 보기는 어렵다.

둘째, 클래스 수 $K$ 에 비례해 discriminator 수가 늘어나므로, 클래스가 매우 많은 문제에서는 구조가 무거워질 수 있다. 논문은 discriminator의 계산 비중이 작아 전체 복잡도는 RevGrad와 비슷하다고 설명하지만, 이는 Office-31이나 ImageCLEF-DA 같은 비교적 소규모 분류 문제에 대한 주장에 가깝다. 대규모 클래스 공간이나 복잡한 데이터셋에서의 확장성은 본문만으로는 충분히 검증되지 않았다.

셋째, 이 논문은 multimode structure를 사실상 **class posterior**로 근사한다. 즉, 모드의 정의가 클래스와 거의 동일하게 취급된다. 그러나 실제 데이터에서 모드는 클래스보다 더 미세한 하위 군집일 수도 있다. 예를 들어 같은 클래스 안에도 pose, style, background에 따라 여러 서브모드가 존재할 수 있다. MADA는 클래스 수준의 정렬은 잘 다루지만, 클래스 내부 서브모드까지 명시적으로 모델링하지는 않는다. 이 점은 논문에서 직접 해결한 문제가 아니다.

넷째, 실험은 당시 표준 벤치마크에서 강한 결과를 보였지만, 모두 분류 중심이며 데이터셋 규모도 비교적 제한적이다. detection, segmentation, open-set adaptation 같은 더 일반적인 설정에 얼마나 잘 확장되는지는 이 논문에서 다루지 않는다. 따라서 기여는 분명하지만 적용 범위는 본문 실험 범위 내에서 해석하는 것이 안전하다.

비판적으로 보면, MADA의 핵심은 “class-conditional alignment”를 adversarial 방식으로 구현한 것으로 이해할 수 있다. 이 아이디어는 매우 타당하지만, target pseudo-label의 품질에 취약할 수 있으며, 결국 정렬의 성공 여부가 classifier 신뢰도와 긴밀히 연결된다. 그럼에도 불구하고 논문은 이 구조가 실제로 전역 정렬보다 낫다는 충분한 실험적 증거를 제시했다는 점에서 설득력이 있다.

## 6. 결론

이 논문은 unsupervised domain adaptation에서 단일 discriminator 기반 adversarial alignment의 한계를 지적하고, 이를 해결하기 위해 **Multi-Adversarial Domain Adaptation (MADA)** 를 제안했다. 핵심 기여는 클래스별 multiple domain discriminator를 도입하고, 각 샘플이 label predictor의 출력 확률에 따라 관련 discriminator에 soft하게 기여하도록 설계한 것이다. 이를 통해 source와 target의 **multimode structure**를 더 정교하게 정렬하고, 잘못된 모드 정렬로 인한 **negative transfer**를 줄이려 했다.

실험적으로 MADA는 Office-31과 ImageCLEF-DA에서 당시 강력한 baseline들을 능가했으며, 특히 어려운 transfer task와 partial label mismatch 상황에서 큰 강점을 보였다. 이는 단순히 domain-invariant feature를 만드는 것만으로는 충분하지 않고, **discriminative structure를 고려한 fine-grained alignment** 가 필요하다는 메시지를 강하게 전달한다.

향후 연구 관점에서 보면, 이 논문은 이후의 class-conditional adaptation, multi-discriminator adaptation, pseudo-label-aware alignment 연구들에 중요한 발판이 되는 아이디어를 제공한다. 실제 적용 측면에서도, source와 target의 구조가 단순하지 않고 일부 클래스 불일치나 복잡한 하위 모드가 존재하는 환경에서, MADA의 관점은 여전히 유효하다. 즉, 이 논문은 domain adaptation을 “전체 분포 맞추기”에서 “의미 있는 구조끼리 맞추기”로 한 단계 더 정교화한 중요한 작업으로 평가할 수 있다.
