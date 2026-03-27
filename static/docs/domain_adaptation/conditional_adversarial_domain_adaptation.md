# Conditional Adversarial Domain Adaptation

* **저자**: Mingsheng Long, Zhangjie Cao, Jianmin Wang, Michael I. Jordan
* **발표연도**: 2018
* **arXiv**:

## 1. 논문 개요

이 논문은 unsupervised domain adaptation 문제를 다룬다. 구체적으로는, label이 있는 source domain과 label이 없는 target domain 사이에 분포 차이(domain shift)가 존재할 때, source에서 학습한 분류기가 target에서도 잘 작동하도록 만드는 것이 목표다. 기존의 adversarial domain adaptation 방법들은 주로 feature representation 자체만 정렬하려고 했는데, 저자들은 이것만으로는 분류 문제의 본질적인 multimodal structure를 충분히 반영하지 못한다고 본다. 예를 들어 클래스가 여러 개일 때, 서로 다른 클래스의 feature들이 도메인 사이에서 잘못 섞여 정렬되면 domain discriminator는 속일 수 있어도 실제 분류 성능은 나빠질 수 있다.

논문이 제기하는 핵심 연구 문제는 두 가지다. 첫째, 단순히 feature distribution만 맞추는 adversarial adaptation은 class-aware한 정렬이 아니기 때문에 multimodal 분포를 제대로 맞추지 못할 수 있다. 둘째, classifier prediction을 조건 정보로 사용하면 더 좋은 정렬이 가능하지만, prediction이 불확실한 샘플까지 동일하게 사용하면 오히려 transfer를 해칠 수 있다. 따라서 저자들은 feature와 class prediction의 결합 구조를 활용하면서도, prediction uncertainty를 제어하는 새로운 conditional adversarial adaptation 프레임워크가 필요하다고 주장한다.

이 문제는 매우 중요하다. 실제 응용에서는 학습 데이터와 테스트 데이터의 분포가 달라지는 일이 흔하며, 특히 새로운 환경에서는 target label을 얻기 어렵다. 따라서 label 없는 target domain으로의 robust한 transfer는 컴퓨터 비전뿐 아니라 넓은 범위의 머신러닝 응용에서 중요한 주제다. 이 논문은 기존 adversarial adaptation의 약점을 분명히 짚고, 보다 discriminative하면서도 transferable한 정렬 방식을 제안한다는 점에서 의미가 크다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 domain discriminator를 단순히 feature $f$에만 의존하게 하지 않고, classifier prediction $g$까지 함께 조건으로 넣어 학습시키는 것이다. 저자들의 관점에서 $g$는 단순한 출력 벡터가 아니라, 각 샘플이 어떤 class mode에 속하는지를 드러내는 discriminative information이다. 따라서 $f$와 $g$를 함께 사용하면, source와 target의 단순한 marginal feature alignment가 아니라 보다 class-sensitive한 joint alignment가 가능해진다.

이 아이디어를 구현한 모델이 CDAN, 즉 Conditional Domain Adversarial Network이다. 기존 DANN류 방법은 feature만 정렬하기 때문에 “도메인은 비슷해졌지만 클래스 구조는 망가지는” under-matching 혹은 mode mismatch 문제가 생길 수 있다. 반면 CDAN은 domain discriminator가 $(f,g)$의 joint variable을 보게 함으로써 “어떤 feature가 어떤 class prediction과 결합되는가”까지 반영한다. 논문은 특히 이 결합을 단순 concatenation이 아니라 multilinear conditioning으로 설계해야 multimodal structure를 충분히 포착할 수 있다고 강조한다.

또 하나의 핵심 아이디어는 entropy conditioning이다. target domain의 예측은 초기에 불확실한 경우가 많기 때문에, 불확실한 prediction을 그대로 domain alignment에 강하게 반영하면 잘못된 정렬이 일어날 수 있다. 이를 막기 위해, 저자들은 entropy가 낮은, 즉 상대적으로 confident한 샘플에 더 큰 가중치를 주어 domain discriminator가 “easy-to-transfer” 예제에 집중하도록 만든다. 이 점이 CDAN+E의 핵심이며, 실험에서도 기본 CDAN보다 더 좋은 성능을 보인다.

기존 접근과의 차별점은 명확하다. 논문에서 비교하는 여러 선행 연구는 feature와 class를 따로 정렬하거나, classifier output만 별도로 보는 방식이었다. 그러나 이 논문은 feature와 prediction의 cross-covariance dependency 자체를 하나의 conditional discriminator 안에서 모델링해야 multimodal alignment가 가능하다고 주장한다. 즉, “feature 정렬”과 “class 정보 활용”을 분리된 모듈로 보는 대신, 둘의 상호작용을 직접 학습 대상으로 삼는 것이 차별점이다.

## 3. 상세 방법 설명

전체 시스템은 source 분류 손실과 domain adversarial 손실을 함께 최적화하는 minimax 구조로 이루어진다. 입력 $\mathbf{x}$는 feature extractor $F$를 거쳐 feature representation $\mathbf{f}=F(\mathbf{x})$를 만들고, classifier $G$를 거쳐 prediction $\mathbf{g}=G(\mathbf{x})$를 만든다. 여기서 source domain에는 label이 있으므로 분류 손실을 계산할 수 있고, source와 target 모두에 대해서는 domain discriminator $D$를 학습시켜 source인지 target인지 구분하게 한다. 다만 CDAN에서는 $D$가 feature만 보지 않고, $f$와 $g$를 조건화한 표현 $T(h)$를 입력으로 받는다. 여기서 $h=(f,g)$이다.

먼저 source classification objective는 다음과 같다.

$$
\mathcal{E}(G)=\mathbb{E}_{(\mathbf{x}_i^s,\mathbf{y}_i^s)\sim \mathcal{D}_s}
L(G(\mathbf{x}_i^s),\mathbf{y}_i^s)
$$

여기서 $L(\cdot,\cdot)$는 cross-entropy loss이다. 이것은 source data에서 정확한 분류기를 만들기 위한 항이다.

기존 adversarial adaptation과 비슷하게, domain discriminator는 source sample과 target sample을 구분하려고 하며, 그 손실은 다음과 같다.

$$
\mathcal{E}(D,G)=
-\mathbb{E}_{\mathbf{x}_i^s\sim \mathcal{D}_s}\log[D(\mathbf{f}_i^s,\mathbf{g}_i^s)]
-\mathbb{E}_{\mathbf{x}_j^t\sim \mathcal{D}_t}\log[1-D(\mathbf{f}_j^t,\mathbf{g}_j^t)]
$$

이 식의 의미는 간단하다. discriminator $D$는 source의 joint representation에는 1을, target의 joint representation에는 0을 내도록 학습된다. 반대로 feature extractor와 classifier는 이 discriminator를 속이도록 학습되어, source와 target의 joint distribution이 구분되지 않게 만든다.

전체 minimax objective는 다음과 같다.

$$
\min_G \mathcal{E}(G)-\lambda \mathcal{E}(D,G), \quad
\min_D \mathcal{E}(D,G)
$$

즉, $G$는 source classification error를 줄이면서 동시에 domain discriminator가 source와 target을 구분하지 못하도록 만들어야 한다. $\lambda$는 분류 성능과 domain invariance 사이의 trade-off를 조절하는 hyper-parameter이다.

이 논문의 가장 중요한 방법론적 기여는 conditioning strategy이다. 가장 단순한 방식은 $f$와 $g$를 이어붙인 $\mathbf{f}\oplus \mathbf{g}$를 discriminator에 넣는 concatenation이다. 하지만 저자들은 이것이 두 변수의 독립적 병치에 가깝고, feature와 class prediction 사이의 multiplicative interaction을 제대로 반영하지 못한다고 지적한다.

이를 해결하기 위해 제안한 것이 multilinear conditioning이다. 조건화 함수는 다음처럼 정의된다.

$$
T_{\otimes}(f,g)=f\otimes g
$$

여기서 $\otimes$는 outer product이다. 이 표현은 각 feature 차원과 각 class prediction 차원의 곱을 모두 포함하므로, feature-class 간 상호작용을 풍부하게 담는다. 논문은 이 표현이 class-conditional distribution의 평균들을 사실상 담아낼 수 있기 때문에, multimodal structure를 포착하는 데 유리하다고 설명한다. 직관적으로 말하면, 단순 concatenation은 “이 샘플의 feature는 무엇인가, prediction은 무엇인가”를 나란히 보여줄 뿐이지만, outer product는 “어떤 feature 성분이 어떤 class prediction과 결합되는가”를 직접 표현한다.

문제는 차원 폭발이다. $f$의 차원을 $d_f$, $g$의 차원을 $d_g$라고 하면 $f\otimes g$의 차원은 $d_f\times d_g$가 되므로 매우 커질 수 있다. 이를 해결하기 위해 논문은 randomized multilinear map을 사용한다.

$$
T_{\odot}(f,g)=\frac{1}{\sqrt{d}}(R_f f)\odot(R_g g)
$$

여기서 $\odot$는 element-wise product이고, $R_f$, $R_g$는 한 번 샘플링한 후 고정하는 random matrix이다. 이 식은 고차원 outer product를 직접 만들지 않고도, 그 inner product 구조를 근사하도록 설계되었다. 논문은 theorem을 통해 $T_{\odot}$가 $T_{\otimes}$의 inner product를 unbiased하게 근사함을 보인다. 기대값은 정확히 맞고, 분산은 random matrix의 4차 모멘트에 의해 제어된다고 설명한다. 실용적으로는 다음 규칙을 쓴다.

$$
T(h)= \begin{cases}
T_{\otimes}(f,g), & d_f\times d_g \le 4096 \\
T_{\odot}(f,g), & \text{otherwise}
\end{cases}
$$

즉, 차원이 manageable하면 정확한 multilinear map을 쓰고, 너무 크면 randomized approximation을 쓴다.

이를 반영한 CDAN objective는 다음과 같이 쓸 수 있다.

$$
\min_G
\mathbb{E}_{(\mathbf{x}_i^s,\mathbf{y}_i^s)\sim \mathcal{D}_s}
L(G(\mathbf{x}_i^s),\mathbf{y}_i^s)
+
\lambda\Big(
\mathbb{E}_{\mathbf{x}_i^s\sim \mathcal{D}_s}\log[D(T(h_i^s))]
+
\mathbb{E}_{\mathbf{x}_j^t\sim \mathcal{D}_t}\log[1-D(T(h_j^t))]
\Big)
$$

$$
\max_D
\mathbb{E}_{\mathbf{x}_i^s\sim \mathcal{D}_s}\log[D(T(h_i^s))]
+
\mathbb{E}_{\mathbf{x}_j^t\sim \mathcal{D}_t}\log[1-D(T(h_j^t))]
$$

이제 entropy conditioning을 보자. 예측 분포의 불확실성은 entropy로 측정한다.

$$
H(g)=-\sum_{c=1}^{C} g_c\log g_c
$$

entropy가 크면 prediction이 불확실한 것이고, 작으면 confident한 것이다. 저자들은 각 샘플에 다음 가중치를 곱한다.

$$
w(H(g))=1+e^{-H(g)}
$$

이 가중치는 entropy가 낮을수록 크고, entropy가 높을수록 작아진다. 따라서 confident한 샘플이 domain adversarial training에서 더 큰 역할을 하게 된다. 이를 적용한 CDAN+E objective는 domain discriminator 관련 항마다 $w(H(g))$를 곱한 형태다. 이 설계의 의도는 명확하다. target prediction이 아직 불안정한 샘플은 alignment에 덜 반영하고, 이미 어느 정도 class structure가 드러난 샘플을 우선적으로 맞추자는 것이다.

학습 절차는 end-to-end backpropagation으로 가능하다. 논문에 따르면 전체 시스템은 몇 줄의 코드로 구현 가능할 정도로 간단하며, discriminator 쪽 계수 $\lambda$는 training progress에 따라 점차 키우는 progressive strategy를 사용한다. 학습률은 DANN 계열에서 사용하던 annealing schedule을 따른다. 또한 source classifier는 ImageNet pre-trained backbone을 fine-tuning하되, 새로 추가한 classifier layer는 더 큰 learning rate로 학습한다.

이 논문은 이론적 분석도 제공한다. 핵심은 target risk $\epsilon_Q(G)$가 source risk와 distribution discrepancy, 그리고 이상적 가설 $G^*$의 error로 upper bound된다는 domain adaptation theory를 바탕으로, CDAN의 discriminator가 feature-prediction joint variable 위의 discrepancy를 줄이는 역할을 한다는 점이다. 기존 feature-only discrepancy 대신, proxy joint distributions $P_G=(f,G(f))$와 $Q_G=(f,G(f))$ 사이의 $\Delta$-distance를 정의하고, 충분히 expressive한 domain discriminator family 아래에서 CDAN training이 이 discrepancy의 상한을 줄인다고 설명한다. 요약하면, 이론적으로도 CDAN은 단순 marginal feature alignment보다 더 task-relevant한 discrepancy를 줄이는 구조라는 것이 저자들의 주장이다.

## 4. 실험 및 결과

실험은 다섯 종류의 대표적인 domain adaptation benchmark에서 수행되었다. Office-31, ImageCLEF-DA, Office-Home, Digits, VisDA-2017이다. 이 구성은 비교적 쉬운 데이터셋부터 매우 challenging한 synthetic-to-real 데이터셋까지 포괄한다. 따라서 특정 환경에서만 잘 되는 방법이 아니라, 다양한 domain gap 상황에서 일관적으로 강한지를 확인할 수 있다.

비교 대상은 DAN, RTN, DANN, ADDA, JAN, UNIT, GTA, CyCADA 등 당시 대표적인 deep domain adaptation 방법들이다. 평가는 unsupervised domain adaptation 프로토콜에 따라 수행되며, source의 labeled data 전체와 target의 unlabeled data 전체를 사용한다. 주요 평가지표는 classification accuracy이고, 세 번의 random experiment 평균으로 보고한다. backbone은 AlexNet과 ResNet-50을 사용한다. 중요한 점은 비교 방법들이 backbone은 비슷하고 adaptation module이 다르므로, 성능 차이가 방법론 차이에서 왔다는 해석이 가능하다는 것이다.

Office-31 결과를 보면, CDAN과 CDAN+E는 거의 모든 transfer task에서 기존 방법보다 강하다. AlexNet 기반 평균 accuracy는 JAN이 76.0인데, CDAN은 77.0, CDAN+E는 77.7이다. ResNet-50 기반에서는 JAN이 84.3, GTA가 86.5인데, CDAN은 86.6, CDAN+E는 87.7이다. 특히 A→W와 A→D 같은 어려운 task에서 개선 폭이 크다. 예를 들어 ResNet-50에서 A→W는 JAN 85.4, GTA 89.5, CDAN 93.1, CDAN+E 94.1이다. A→D는 JAN 84.7, GTA 87.7, CDAN 89.8, CDAN+E 92.9이다. 이는 source와 target이 많이 다를수록 conditional alignment의 이점이 더 크다는 논문의 주장을 잘 뒷받침한다.

ImageCLEF-DA에서는 전체적인 개선 폭이 상대적으로 작다. 논문도 그 이유를 설명하는데, 이 데이터셋은 세 도메인이 서로 더 유사하고 클래스 균형도 잘 맞아서 adaptation 자체가 비교적 쉬운 편이라는 것이다. 그래도 ResNet-50 평균 accuracy에서 JAN 85.8 대비 CDAN 87.1, CDAN+E 87.7로 개선이 있다. 즉, 쉬운 데이터셋에서는 절대적 향상이 작지만, 여전히 conditional adaptation의 우위가 유지된다.

Office-Home 결과는 이 논문의 강점을 가장 잘 보여준다. 이 데이터셋은 클래스 수가 65개로 많고, 도메인 간 차이도 크며, 전반적으로 어렵다. ResNet-50 평균 accuracy를 보면 DANN 57.6, JAN 58.3, CDAN 63.8, CDAN+E 65.8이다. 이는 단순한 소폭 향상이 아니라 상당히 큰 개선이다. 논문은 이를 “category-agnostic alignment의 한계”로 해석한다. 클래스가 많고 구조가 복잡할수록 단순 feature alignment는 classification-friendly하지 않을 수 있고, CDAN처럼 multimodal structure를 반영하는 alignment가 더 유리하다는 것이다.

Digits와 VisDA-2017에서도 강한 결과를 보인다. Digits에서는 generative pixel-level adaptation 방법들이 강력한 baseline인데, CDAN+E는 평균 94.3으로 CyCADA 94.2와 비슷하거나 약간 높고, UNIT 93.4보다 높다. VisDA-2017에서는 JAN 61.6, GTA 69.5, CDAN 66.8, CDAN+E 70.0이다. 즉, feature-level method임에도 불구하고 CDAN+E가 매우 복잡한 generative adaptation 기법과 경쟁하거나 능가한다.

Ablation study도 중요하다. 먼저 entropy conditioning의 효과는 일관적이다. 거의 모든 주요 결과표에서 CDAN+E가 CDAN보다 높다. 이는 entropy-based reweighting이 uncertain target examples의 악영향을 줄이고 transferability를 높인다는 논문 주장을 실험적으로 지지한다. 또 randomized multilinear map의 sampling 방식도 비교했는데, Office-31 ResNet 기준으로 w/o random sampling이 평균 87.7로 가장 높고, uniform sampling은 87.0, gaussian sampling은 86.4였다. 즉, 정확한 multilinear conditioning이 가장 좋지만, randomized version도 상당히 잘 작동한다.

Conditioning strategy 비교에서도 논문의 설계 근거가 뚜렷하다. 단순 concatenation 기반 DANN-[f,g]는 좋은 성능을 내지 못한다고 보고한다. 이는 concatenation이 feature와 class prediction 사이의 cross-covariance를 충분히 표현하지 못하기 때문이라고 설명한다. 반면 multilinear conditioning은 이 상호작용을 포착해 더 좋은 adaptation을 만든다. 또한 entropy weight $e^{-H(g)}$가 prediction correctness와 잘 대응된다고 분석하여, confidence-based weighting의 타당성을 보인다.

Distribution discrepancy 분석에서는 $\mathcal{A}$-distance를 사용한다. Figure 2(c)에 따르면 CDAN feature의 $\mathrm{dist}_{\mathcal{A}}$가 ResNet이나 DANN보다 더 작다. 이는 CDAN이 source-target gap을 더 효과적으로 줄인다는 뜻이다. Convergence 분석에서는 CDAN이 DANN보다 더 빨리 수렴하며, 정확한 multilinear version이 randomized version보다 더 빠르게 수렴한다고 보고한다.

마지막으로 t-SNE visualization은 정성적 해석을 제공한다. ResNet feature는 source-target이 잘 섞이지 않고, DANN은 도메인 정렬은 다소 좋아지지만 class discrimination이 충분하지 않다. CDAN-f와 특히 CDAN-fg는 source-target alignment가 더 잘 되면서 class cluster도 더 뚜렷하다. 즉, 단순히 도메인만 섞는 것이 아니라 분류 친화적으로 정렬된다는 것이 시각적으로 드러난다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 adversarial domain adaptation의 실패 원인을 비교적 정확하게 짚었다는 점이다. 기존 방법들이 feature marginal alignment에 치우쳐 있다는 비판은 설득력이 있으며, 실제로 multiclass classification에서는 class-conditional structure가 매우 중요하다. CDAN은 이 문제를 feature와 prediction의 joint interaction으로 다루고, 그것도 단순 concatenation이 아니라 multilinear conditioning으로 설계했다는 점에서 방법론적으로 탄탄하다.

두 번째 강점은 성능과 단순성의 균형이다. 논문은 “a few lines of codes”라고 표현할 정도로 구현이 비교적 간단하면서도, 다섯 개의 benchmark에서 강력한 성능을 보였다. 특히 Office-Home과 VisDA처럼 어려운 환경에서 개선 폭이 크다는 점은 실질적 가치가 높다. 복잡한 pixel-level generative model을 쓰지 않으면서도 그에 필적하거나 앞서는 결과를 냈다는 점도 인상적이다.

세 번째 강점은 불확실성 제어를 함께 도입했다는 것이다. classifier prediction을 조건 정보로 쓴다는 아이디어만으로는 noisy target prediction 문제가 남는데, entropy conditioning을 통해 이 문제를 자연스럽게 보완했다. 이 설계는 직관적일 뿐 아니라 ablation으로도 검증되었다.

네 번째 강점은 이론적 설명을 제공했다는 점이다. 많은 domain adaptation 논문이 실험 중심으로 끝나는 반면, 이 논문은 왜 joint variable 위의 discriminator가 target risk bound 관점에서 의미가 있는지 formalism을 통해 설명하려고 한다. 물론 실제 deep network 전체의 복잡한 학습 dynamics를 완전히 설명하는 것은 아니지만, 적어도 방법의 방향성에 대한 논리적 근거를 제공한다.

한계도 있다. 첫째, classifier prediction $g$를 조건 정보로 사용하는 방식은 본질적으로 현재 classifier의 품질에 의존한다. 논문은 entropy conditioning으로 이를 완화하지만, 초기 학습 단계에서 prediction이 매우 불안정한 경우 conditioning 자체가 noise를 전달할 가능성은 여전히 있다. 이 위험을 어느 정도까지 줄일 수 있는지는 데이터 특성과 학습 안정성에 따라 달라질 수 있다.

둘째, multilinear conditioning은 표현력이 높지만 차원 증가 문제가 있어서 randomized approximation이 필요하다. 논문은 이를 효율적으로 처리했지만, 근사 오차가 실제 학습에 미치는 영향은 데이터와 backbone에 따라 달라질 수 있다. 실험에서는 잘 작동했지만, 모든 상황에서 동일하다고 단정할 수는 없다.

셋째, 이 논문은 주로 image classification 기반의 unsupervised domain adaptation에 초점을 맞춘다. segmentation이나 detection 같은 structured output task에 일반적으로 어떻게 확장되는지는 본문에서 직접적으로 자세히 다루지 않는다. 관련 선행연구를 언급하긴 하지만, 본 논문의 실험적 검증 범위는 주로 classification이다.

넷째, 이론 분석은 representation space가 고정되어 있다고 두고 discriminator family가 충분히 expressive하다고 가정한다. 이는 domain adaptation 이론에서 흔한 전개이지만, 실제 end-to-end deep learning 상황을 그대로 반영한다고 보기는 어렵다. 즉, 이론은 방향성을 주는 수준이지 전체 현상을 엄밀하게 설명하는 완전한 보장은 아니다.

비판적으로 보면, 이 논문의 가장 큰 메시지는 “feature만 맞추면 부족하다”는 것이다. 이 주장은 이후 많은 class-conditional adaptation 연구의 출발점이 되기에 충분히 중요하다. 다만 classifier prediction을 condition으로 쓰는 방법은 pseudo-label 기반 방법들과 마찬가지로 self-reinforcing bias를 만들 가능성이 있고, 이 문제를 더 정교하게 제어하는 후속 연구가 필요하다고 볼 수 있다.

## 6. 결론

이 논문은 Conditional Domain Adversarial Network, 즉 CDAN을 제안하여 adversarial domain adaptation을 단순한 feature alignment에서 joint feature-prediction alignment로 확장했다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, classifier prediction을 domain discriminator의 조건 정보로 사용해 multimodal distribution alignment를 가능하게 했다. 둘째, concatenation 대신 multilinear conditioning을 사용해 feature와 class prediction의 cross-covariance를 포착했다. 셋째, entropy conditioning을 도입해 uncertain target examples의 악영향을 줄이고 transferability를 높였다.

실험 결과는 이 아이디어가 단순한 이론적 제안이 아니라 실제로 강력하게 작동함을 보여준다. 특히 어려운 데이터셋에서 더 큰 향상을 보였다는 점은, class-aware joint alignment가 domain adaptation의 중요한 방향임을 시사한다. 이 연구는 이후 class-conditional alignment, confidence-aware adaptation, joint distribution adaptation 같은 주제들에 직접적인 영향을 준 중요한 논문으로 볼 수 있다.

실제 적용 측면에서도 의미가 크다. 의료 영상, 자율주행, 산업 비전처럼 학습 환경과 배포 환경의 차이가 큰 분야에서는, source와 target의 단순 feature alignment만으로는 부족할 수 있다. 이 논문은 모델의 prediction 자체를 adaptation 신호로 활용하는 방법을 보여주었고, 이는 실제 배포 환경에서 더 robust한 transfer learning 시스템을 설계하는 데 중요한 통찰을 제공한다.

무엇보다도, 이 논문은 adversarial adaptation의 “무엇을 맞출 것인가”라는 질문에 대해 feature alone이 아니라 feature와 prediction의 관계를 맞춰야 한다는 강한 답을 제시한다. 그 점에서 CDAN은 단순한 성능 개선 방법을 넘어, domain adaptation의 관점을 한 단계 진전시킨 연구라고 평가할 수 있다.
