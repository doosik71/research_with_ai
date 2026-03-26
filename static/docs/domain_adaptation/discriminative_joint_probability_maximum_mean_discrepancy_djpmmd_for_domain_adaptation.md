# Discriminative Joint Probability Maximum Mean Discrepancy (DJP-MMD) for Domain Adaptation

* **저자**: Wen Zhang and Dongrui Wu
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1912.00320](https://arxiv.org/abs/1912.00320)

## 1. 논문 개요

이 논문은 domain adaptation에서 source domain과 target domain 사이의 분포 차이를 더 정확하게 측정하는 새로운 기준으로 **DJP-MMD (Discriminative Joint Probability Maximum Mean Discrepancy)**를 제안한다. 기존의 많은 방법은 두 도메인 사이의 차이를 marginal distribution discrepancy와 conditional distribution discrepancy의 합, 또는 가중합으로 근사해 왔다. 대표적으로 TCA는 marginal discrepancy만 맞추고, JDA는 marginal과 conditional discrepancy를 함께 맞추며, BDA는 둘의 비중을 조절한다. 그러나 저자들은 이런 방식이 본질적으로 **joint probability distribution discrepancy**를 직접 측정하는 것이 아니라는 점을 지적한다.

논문이 다루는 핵심 문제는 다음과 같다. source domain에는 레이블이 충분하지만, target domain에는 레이블이 없거나 매우 부족한 상황에서, 두 도메인의 분포가 다르면 source에서 학습한 분류기가 target에서 잘 작동하지 않는다. 기존 MMD 기반 방법은 같은 클래스끼리 두 도메인을 가깝게 만드는 데 집중했지만, 서로 다른 클래스 사이를 더 분리해야 한다는 **discriminability** 문제를 충분히 반영하지 못했다. 즉, “같은 것은 가깝게”만 고려하고 “다른 것은 멀게”는 약하게 다룬 셈이다.

이 문제가 중요한 이유는 domain adaptation의 최종 목적이 단순한 분포 정렬이 아니라 **target domain에서의 분류 정확도 향상**이기 때문이다. 도메인 간 정렬이 잘 되어도 클래스 간 경계가 흐려지면 실제 분류 성능은 제한될 수 있다. 따라서 논문은 transferability와 discriminability를 동시에 고려하는 분포 정렬이 더 바람직하다고 주장한다.

저자들은 이를 위해 joint probability discrepancy를 직접 정의하고, 같은 클래스 간 joint discrepancy는 줄이고 다른 클래스 간 joint discrepancy는 키우는 방식의 DJP-MMD를 설계한다. 이후 이 기준을 하나의 domain adaptation 프레임워크인 **JPDA (Joint Probability Distribution Adaptation)** 안에 삽입해, 기존 TCA/JDA/BDA와 비교한다. 실험 결과, 여섯 개 이미지 분류 데이터셋에서 평균적으로 더 높은 정확도를 보였다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. 기존 방법은 source와 target의 차이를 다음처럼 본다.

$$
d(\mathcal{D}_s,\mathcal{D}_t) \approx d(P(X_s),P(X_t)) + d(P(Y_s|X_s),P(Y_t|X_t))
$$

또는 여기에 가중치를 둔 형태로 본다. 하지만 저자들은 실제로 분포를 완전히 기술하는 것은 $P(X,Y)$이며, 따라서 더 자연스러운 비교 대상은 **joint probability distribution**이라고 본다. 즉, marginal과 conditional을 따로 계산해서 더하는 대신, joint를 직접 다뤄야 한다는 주장이다.

여기서 중요한 이론적 전환은 다음과 같다. 기존 방식은 사실상 $P(Y|X)P(X)$를 다루는데, 이것을 계산 가능한 형태로 바꾸는 과정에서 다시 conditional distribution $P(X|Y)$를 통해 근사한다. 이 과정에는 두 가지 근사가 들어간다. 첫째, posterior와 marginal의 의존성이 무시된다. 둘째, 계산이 어려운 $P(Y|X)$ 대신 $P(X|Y)$를 사용한다. 저자들은 이러한 근사 대신, 처음부터 Bayesian law에 따라

$$
P(X,Y)=P(X|Y)P(Y)
$$

를 사용하면 joint probability discrepancy를 직접 계산할 수 있다고 본다. 이 점이 논문의 첫 번째 차별점이다.

두 번째 핵심 아이디어는 **분포 정렬의 목적을 transferability와 discriminability의 동시 최적화로 바꾸는 것**이다. 같은 클래스의 source-target 샘플은 가까워져야 하고, 다른 클래스의 source-target 샘플은 오히려 멀어져야 한다. 이를 수식으로 분해하면, 같은 클래스끼리의 joint probability discrepancy를 모은 항 $\mathcal{M}_T$는 줄이고, 다른 클래스끼리의 discrepancy를 모은 항 $\mathcal{M}_D$는 키우는 방향이 된다. 논문은 이를

$$
d(\mathcal{D}_s,\mathcal{D}_t)=\mathcal{M}_T-\mu \mathcal{M}_D
$$

로 정의한다. 여기서 $\mu>0$는 trade-off parameter이다. 이 식은 domain adaptation 문제를 “같은 클래스 도메인 정렬”과 “서로 다른 클래스 분리”의 결합 문제로 재해석한 것이다.

기존 접근과의 차별점은 다음과 같이 정리할 수 있다. TCA는 marginal만 맞춘다. JDA는 marginal과 class-conditional을 함께 맞춘다. BDA는 둘의 균형을 조절한다. 반면 DJP-MMD는 애초에 joint probability를 직접 다루며, 그 안에서 같은 클래스와 다른 클래스를 분리해서 취급한다. 즉, 단순히 두 종류의 discrepancy를 더하는 것이 아니라, **분류에 유리한 방향으로 discrepancy 구조 자체를 재설계**한다는 점이 핵심이다.

## 3. 상세 방법 설명

논문은 먼저 문제를 전형적인 unsupervised domain adaptation 설정으로 놓는다. source domain $\mathcal{D}_s$에는 $n_s$개의 labeled sample ${( \mathbf{x}_{s,i}, y_{s,i})}_{i=1}^{n_s}$가 있고, target domain $\mathcal{D}_t$에는 $n_t$개의 unlabeled sample ${\mathbf{x}_{t,j}}_{j=1}^{n_t}$가 있다. feature space와 label space는 두 도메인에서 동일하다고 가정한다. 목표는 feature mapping $h$ 또는 선형 사상 $A$를 찾아, 변환된 source와 target이 공통 subspace에서 잘 정렬되도록 만드는 것이다.

일반적인 domain adaptation 목적함수는 다음처럼 쓸 수 있다.

$$
\min_h \ d_{S,T} + \lambda \mathcal{R}(h)
$$

여기서 $d_{S,T}=d(P(X_s,Y_s),P(X_t,Y_t))$는 source-target discrepancy이고, $\mathcal{R}(h)=|h|_F^2$는 복잡도 제어용 regularization이다. 즉, 논문은 처음부터 discrepancy의 대상이 joint probability라고 명시한다.

### 전통적 MMD 해석의 재검토

기존 MMD 기반 domain adaptation은 보통 다음과 같은 형태를 사용한다.

$$
d(\mathcal{D}_s,\mathcal{D}_t)
\approx
\mu_1 d(P(X_s),P(X_t))
+
\mu_2 d(P(X_s|Y_s),P(X_t|Y_t))
$$

이는 marginal discrepancy와 conditional discrepancy를 따로 계산한 뒤 결합한 것이다. 저자들은 이것이 joint discrepancy의 직접 계산이 아니라 근사라고 본다. 특히 conditional term도 실제 posterior $P(Y|X)$가 아니라 class-conditional $P(X|Y)$를 쓰고 있으므로, joint probability discrepancy를 정확하게 반영하지 못한다고 비판한다.

선형 매핑 $h(\mathbf{x})=A^\top \mathbf{x}$를 쓰면, 기존 conditional MMD는 클래스별 평균 차이의 합으로 쓸 수 있다.

$$
\sum_{c=1}^{C} \left| \frac{1}{n_s^c}\sum_{i=1}^{n_s^c}A^\top \mathbf{x}_{s,i}^c - \frac{1}{n_t^c}\sum_{j=1}^{n_t^c}A^\top \mathbf{x}_{t,j}^c \right|_2^2
$$

여기서 핵심은 분모가 각 클래스 샘플 수 $n_s^c, n_t^c$라는 점이다. 특히 target class count $n_t^c$는 pseudo-label에서 추정되므로 불확실성이 있다.

### DJP-MMD의 정의

논문은 joint probability discrepancy를 Bayesian law에 따라 $P(X|Y)P(Y)$의 비교로 정의한다. 이를 같은 클래스끼리의 항과 다른 클래스끼리의 항으로 분해하면

$$
d(\mathcal{D}_s,\mathcal{D}_t) \equiv \mathcal{M}_T + \mathcal{M}_D
$$

처럼 쓸 수 있다. 여기서 $\mathcal{M}_T$는 same-class source-target discrepancy이고, $\mathcal{M}_D$는 different-class source-target discrepancy이다.

하지만 분류를 잘하려면 $\mathcal{M}_D$를 줄이는 것이 아니라 오히려 키워서 클래스 간 분리를 강화해야 한다. 그래서 최종적으로 논문은 discriminative discrepancy를

$$
d(\mathcal{D}_s,\mathcal{D}_t)=\mathcal{M}_T-\mu \mathcal{M}_D
$$

로 둔다. 즉, optimization은 $\mathcal{M}_T$는 작게, $\mathcal{M}_D$는 크게 만드는 방향으로 간다.

### Transferability 항 $\mathcal{M}_T$

같은 클래스에 대한 joint probability discrepancy는 다음과 같이 정의된다.

$$
\mathcal{M}_T = \sum_{c=1}^{C} \left| \mathbb{E}[f(\mathbf{x}_s)|y_s^c]P(y_s^c) - \mathbb{E}[f(\mathbf{x}_t)|y_t^c]P(y_t^c) \right|^2
$$

여기서 $f(\mathbf{x})=A^\top \mathbf{x}$이다. empirical form으로 바꾸면,

$$
\mathbb{E}[f(\mathbf{x}_s)|y_s^c]P(y_s^c) = \frac{1}{n_s}\sum_{i=1}^{n_s^c} A^\top \mathbf{x}_{s,i}^c
$$

이고 target도 유사하게

$$
\mathbb{E}[f(\mathbf{x}_t)|y_t^c]P(y_t^c) = \frac{1}{n_t}\sum_{j=1}^{n_t^c} A^\top \mathbf{x}_{t,j}^c
$$

가 된다. 따라서 최종적으로

$$
\mathcal{M}_T = \sum_{c=1}^{C} \left| \frac{1}{n_s}\sum_{i=1}^{n_s^c} A^\top \mathbf{x}_{s,i}^c - \frac{1}{n_t}\sum_{j=1}^{n_t^c} A^\top \mathbf{x}_{t,j}^c \right|_2^2
$$

가 된다.

이 식의 중요한 차이는 기존 conditional MMD와 달리 분모가 $n_s^c, n_t^c$가 아니라 전체 샘플 수 $n_s, n_t$라는 점이다. 저자들은 이것이 joint probability의 prior $P(Y^c)$를 포함한 형태이기 때문에 더 자연스럽고, 특히 target에서 전체 샘플 수 $n_t$는 정확히 알고 있으므로 클래스별 수를 직접 분모로 쓰는 것보다 안정적이라고 주장한다.

### Discriminability 항 $\mathcal{M}_D$

다른 클래스 간 discrepancy는 다음과 같이 정의된다.

$$
\mathcal{M}_D = \sum_{c \neq \hat{c}}\sum_{\hat{c}=1}^{C} \left| \mathbb{E}[f(\mathbf{x}_s)|y_s^c]P(y_s^c) - \mathbb{E}[f(\mathbf{x}_t)|y_t^{\hat{c}}]P(y_t^{\hat{c}}) \right|^2
$$

이를 empirical form으로 바꾸면

$$
\mathcal{M}_D = \sum_{c \neq \hat{c}}\sum_{\hat{c}=1}^{C} \left| \frac{1}{n_s}\sum_{i=1}^{n_s^c}A^\top \mathbf{x}_{s,i}^c - \frac{1}{n_t}\sum_{j=1}^{n_t^{\hat{c}}}A^\top \mathbf{x}_{t,j}^{\hat{c}} \right|_2^2
$$

가 된다. 이 항은 source의 클래스 $c$와 target의 다른 클래스 $\hat{c}$ 사이 평균 feature를 멀게 만드는 역할을 한다. 따라서 최적화에서 이 항을 빼주는 것은, 결과적으로 **다른 클래스 간 마진을 넓히는 효과**를 낸다.

### 행렬 형태와 최적화

논문은 source one-hot label matrix를 $Y_s$, target pseudo-label one-hot matrix를 $\hat{Y}_t$로 두고, 이를 이용해 transferability 항을 더 압축된 행렬 형태로 쓴다.

$$
\mathcal{M}_T = |A^\top X_s N_s - A^\top X_t N_t|_F^2
$$

여기서

$$
N_s=\frac{Y_s}{n_s}, \qquad N_t=\frac{\hat{Y}_t}{n_t}
$$

이다. 각 열은 클래스별 mean mapped feature를 나타낸다.

다음으로 discriminability를 위해 $F_s$와 $\hat{F}_t$를 구성한다. 직관적으로는 각 클래스에 대해 “자기 자신을 제외한 다른 클래스들과의 조합”을 모두 펼쳐놓은 행렬이다. 이를 통해

$$
\mathcal{M}_D = |A^\top X_s M_s - A^\top X_t M_t|_F^2
$$

로 쓸 수 있고,

$$
M_s=\frac{F_s}{n_s}, \qquad M_t=\frac{\hat{F}_t}{n_t}
$$

이다.

따라서 기본 DJP-MMD 최적화 문제는

$$
\min_A |A^\top X_s N_s - A^\top X_t N_t|_F^2 - \mu |A^\top X_s M_s - A^\top X_t M_t|_F^2
$$

가 된다.

### JPDA 프레임워크

저자들은 이 DJP-MMD를 단독으로 두지 않고, 기존 TCA/JDA 계열과 유사하게 정규화와 PCA 보존 제약을 넣어 **JPDA**를 구성한다.

최종 목적함수는

$$
\min_A |A^\top X_s N_s - A^\top X_t N_t|_F^2 - \mu |A^\top X_s M_s - A^\top X_t M_t|_F^2 + \lambda |A|_F^2
$$

subject to

$$
A^\top X H X^\top A = I
$$

이다. 여기서 $X=[X_s, X_t]$, $H=I-\mathbf{1}_n$는 centering matrix이다. 제약식은 projected data의 분산 구조를 어느 정도 유지하는 역할을 한다. 이는 고전적인 TCA/JDA 계열에서 자주 쓰이는 형태다.

이를 Lagrangian으로 바꾸면

$$
\mathcal{J} = \operatorname{tr}\left( A^\top ( X(R_{\min}-\mu R_{\max})X^\top + \lambda I ) A \right) + \operatorname{tr}\left( \eta ( I - A^\top X H X^\top A ) \right)
$$

가 되고, 여기서 $R_{\min}$은 transferability에 해당하는 matrix, $R_{\max}$는 discriminability에 해당하는 matrix이다. 미분해서 0으로 두면 generalized eigen-decomposition 문제가 된다.

$$
\left(X(R_{\min}-\mu R_{\max})X^\top+\lambda I\right)A = \eta XHX^\top A
$$

저자들은 이 식의 **trailing eigenvectors**를 선택해 projection matrix $A$를 구성한다. 이후 source의 projected feature $A^\top X_s$로 classifier를 학습하고, 이를 target의 projected feature $A^\top X_t$에 적용한다.

### 학습 절차와 pseudo-label 업데이트

JPDA는 iterative algorithm이다. 입력은 source/target feature matrix, source one-hot label matrix, subspace dimension $p$, trade-off parameter $\mu$, regularization $\lambda$, iteration 수 $T$이다. 매 반복마다 다음 절차를 수행한다.

먼저 현재 target pseudo-label을 이용해 $R_{\min}$과 $R_{\max}$를 구성한다. 다음으로 generalized eigen-decomposition을 풀어 $A$를 구한다. 그 뒤 projected source feature로 classifier를 학습하고, projected target에 예측을 수행해 새로운 pseudo-label $\hat{Y}_t$를 얻는다. 이 pseudo-label은 다음 반복에서 다시 joint probability matrix 구성에 사용된다.

즉, 이 알고리즘은 “정렬 → 분류 → pseudo-label 갱신 → 다시 정렬” 구조를 갖는다. 논문에서는 classifier로 1-nearest neighbor를 사용했다.

### Kernelization

비선형 도메인 시프트를 다루기 위해 저자들은 RKHS 상으로의 kernelization도 제시한다. feature를 직접 쓰는 대신 $\phi(\mathbf{x})$를 사용하고, 이에 대응하는 kernel matrix $K_s, K_t, K$를 만든다. 그러면 목적함수는

$$
\min_A |A^\top K_s N_s - A^\top K_t N_t|_F^2 - \mu |A^\top K_s M_s - A^\top K_t M_t|_F^2 + \lambda |A|_F^2
$$

subject to

$$
A^\top K H K^\top A = I
$$

가 된다. 최적화 방식은 선형 경우와 유사하다고 설명한다. 논문은 deep model을 설계하지는 않았고, kernelized shallow DA 프레임워크 수준에서 제안한다.

### 계산 복잡도

논문은 주요 비용을 generalized eigen-decomposition과 MMD matrix construction으로 본다. $T$가 iteration 수, $p$가 subspace dimension일 때 전체 이론적 복잡도는

$$
\mathbf{O}(Tp d^2 + Tn^2 + Tdn)
$$

로 제시된다. 여기서 $n=n_s+n_t$이다. 저자들은 실험에서 JPDA가 JDA/BDA보다 더 빠른 경우가 많다고 보고한다.

## 4. 실험 및 결과

실험은 여섯 개의 이미지 분류 벤치마크 데이터셋에서 수행되었다. 데이터셋은 Office+Caltech, COIL, Multi-PIE, MNIST, USPS이다. 실험 유형은 모두 cross-domain visual adaptation이며, source는 labeled, target은 unlabeled인 unsupervised DA 설정이다.

Office+Caltech는 object recognition 벤치마크로, Caltech (C), Amazon (A), Webcam (W), DSLR (D) 네 도메인으로 구성된다. 서로 다른 두 도메인을 source와 target으로 선택해 총 12개 transfer task를 만든다. COIL은 20개 객체, 1,440장 이미지로 구성되며 COIL1과 COIL2로 분할해 두 도메인 문제를 만든다. Multi-PIE는 face recognition 데이터셋으로 C05, C07, C09, C27, C29 다섯 pose subset을 서로 다른 도메인으로 보아 총 20개 transfer task를 구성한다. USPS와 MNIST는 digit recognition 문제에 사용된다.

비교 대상은 TCA, JDA, BDA, 그리고 제안 방법 JPDA이다. 저자들은 이들 방법이 regularization 구조는 유사하고, 핵심 차이는 사용하는 MMD metric에 있다고 본다. 따라서 성능 차이를 MMD 설계의 차이로 해석할 수 있다고 주장한다. classifier는 모두 1-nearest neighbor를 사용했다. 공통 설정으로 $p=100$, $T=10$을 사용했고, regularization $\lambda$는 Office+Caltech에서 1, 그 외 데이터셋에서는 0.1을 사용했다. JPDA의 trade-off parameter는 $\mu=0.1$이다.

평가 지표는 target domain classification accuracy이다.

정량 결과에서 가장 눈에 띄는 점은 전체 평균 정확도이다. TCA는 47.22%, JDA는 57.37%, BDA는 57.18%, JPDA는 **60.68%**로 보고되었다. 즉, 평균 기준으로 JPDA가 가장 높았다. 특히 JDA와 BDA가 비슷하거나 BDA가 오히려 약간 낮은 경우도 있었는데, 이는 논문이 비판한 “marginal/conditional 가중합이 항상 더 좋은 것은 아니다”라는 주장과 연결된다.

Multi-PIE에서는 여러 task에서 개선폭이 비교적 컸다. 예를 들어 C05→C09는 TCA 41.79, JDA 54.23, BDA 52.82, JPDA 66.67로 JPDA가 크게 앞섰다. C29→C07도 TCA 29.90, JDA 42.05, BDA 43.22, JPDA 51.32로 상승폭이 뚜렷하다. 물론 모든 task에서 절대적으로 최고는 아니며, 예컨대 C05→C27에서는 JDA 84.50, BDA 83.03, JPDA 83.99로 JDA가 약간 높다. 따라서 논문은 “대부분의 작업에서 우수”하다고 보는 것이 정확하지, 모든 경우에 항상 최고라고 말할 수는 없다.

Office+Caltech에서도 전반적으로 JPDA가 우수했다. 예를 들어 C→A는 38.20, 44.78, 44.57, **47.60**이고, C→W는 38.64, 41.69, 40.34, **45.76**이다. 다만 A→D에서는 BDA 40.76, JDA 39.49, JPDA 36.94로 JPDA가 뒤지는 task도 있다. W→D에서는 JDA와 BDA가 89.17로 JPDA 88.54보다 약간 높다. 이런 점은 제안 방법이 평균적으로 강하지만 task별 편차는 남아 있음을 보여준다.

COIL에서는 COIL1→COIL2가 88.47, 89.31, 89.44, **92.08**, COIL2→COIL1이 85.83, 88.47, 88.33, **89.86**으로 제안 방법이 명확히 좋다. USPS+MNIST에서는 USPS→MNIST에서 JDA 59.65, BDA 59.90, JPDA 59.20으로 약간 뒤졌지만, MNIST→USPS에서는 **68.94**로 최고였다. 따라서 전체적으로는 일관된 평균 개선이 존재하지만, digit task 일부에서는 개선이 제한적이다.

정성 결과로는 t-SNE 시각화를 제시했다. Caltech→Amazon 전이에서 원시 feature는 source와 target이 많이 섞이고 클래스 간 분리도 불완전하다. JDA/BDA 이후에는 어느 정도 정렬되지만 클래스 2와 3에서 분리도가 충분하지 않다고 설명한다. JPDA는 source-target alignment를 유지하면서도 클래스 간 분리가 더 선명하다고 주장한다. 이는 $\mathcal{M}_D$를 통한 discriminability 강화 효과를 시각적으로 보여주려는 것이다.

수렴 분석에서도 JPDA는 빠르게 수렴했다고 보고한다. Multi-PIE 20개 task 평균 기준으로 iteration 수가 증가할 때 평균 MMD distance가 더 작아지고 정확도는 더 높아졌다고 한다. 즉, 이 방법이 iterative pseudo-label refinement를 사용하더라도 불안정하지 않고, 비교적 빠르게 안정화된다고 해석한다.

시간 복잡도의 실험적 비교도 흥미롭다. 예를 들어 C05→C07에서 TCA 2.58초, JDA 94.46초, BDA 107.47초, JPDA 46.12초이다. C→A에서는 JDA 31.61초, BDA 34.73초, JPDA 30.65초이고, MNIST→USPS에서는 JDA 9.04초, BDA 13.58초, JPDA 8.41초이다. 즉, TCA보다는 느리지만, JDA/BDA보다는 빠른 경우가 많다. 특히 BDA가 느린 이유는 $\mathcal{A}$-distance 계산을 위해 $C+1$개의 classifier를 학습해야 하기 때문이라고 설명한다.

파라미터 민감도 실험에서는 $\mu$와 $\lambda$에 대한 robustness를 확인했다. 논문은 JPDA가 $\mu \in [0.001, 0.2]$와 $\lambda \in [0.01, 10]$ 범위에서 만족스러운 성능을 보인다고 주장한다. 이는 방법이 극단적으로 민감한 하이퍼파라미터 튜닝에 의존하지 않는다는 근거로 제시된다.

Ablation study에서는 세 가지 MMD를 비교한다. 기존 joint MMD, transferability만 고려하는 JP-MMD, transferability와 discriminability를 모두 고려하는 DJP-MMD이다. 결과적으로 JP-MMD가 기존 joint MMD보다 좋았고, DJP-MMD가 다시 그보다 좋았다. 이는 논문의 핵심 주장인 “joint probability를 직접 쓰는 것 자체가 이득이며, 거기에 discriminability까지 추가하면 더 좋다”를 실험적으로 뒷받침한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 재정의가 명확하다는 점이다. 기존 MMD 기반 DA를 단순히 변형한 것이 아니라, 왜 marginal+conditional의 합이 joint probability discrepancy의 자연스러운 계산이 아닐 수 있는지를 이론적으로 짚고, $P(X|Y)P(Y)$ 형태의 직접적 계산으로 옮겨간다. 이 관점 전환은 단순한 트릭이라기보다, domain discrepancy 정의 자체를 다시 세우는 시도라는 점에서 의미가 있다.

두 번째 강점은 transferability와 discriminability를 동시에 다룬다는 점이다. 많은 DA 논문이 도메인 간 정렬만 강조하는데, 이 논문은 분류 목적에서는 클래스 간 분리 역시 중요하다는 점을 명확히 드러낸다. 특히 $\mathcal{M}_D$를 별도 항으로 정의하고 이를 목적함수에 반영한 것은, 이후의 discriminative DA 관점과도 잘 맞닿아 있다.

세 번째 강점은 방법이 비교적 단순하다는 점이다. 제안 방법은 deep network가 아니라 기존 subspace-based DA 틀 안에 들어간다. generalized eigen-decomposition으로 최적화할 수 있고, kernelization도 자연스럽게 된다. 구현 난이도가 높지 않으면서 실험적으로 평균 성능 향상을 보였다는 점은 실용적인 장점이다.

네 번째 강점은 비교 실험이 비교적 공정하다는 점이다. TCA, JDA, BDA와 동일한 분류기와 유사한 regularization 틀을 사용해, MMD 정의 차이에 따른 성능 변화를 보려 했다. 또한 평균 성능, t-SNE 시각화, 수렴성, 시간 복잡도, 파라미터 민감도, ablation study까지 포함해 논문의 주장들을 다각도로 검증하려 했다.

반면 한계도 분명하다. 첫째, target pseudo-label 품질에 크게 의존한다. $\hat{Y}_t$는 반복적으로 갱신되지만, 초기 pseudo-label이 많이 틀리면 $\mathcal{M}_T$와 $\mathcal{M}_D$ 계산 자체가 왜곡될 수 있다. 논문은 iterative refinement를 사용하지만, pseudo-label noise가 얼마나 심할 때 성능이 무너지는지는 자세히 분석하지 않았다.

둘째, discriminability 항이 항상 안정적으로 작동하는지에 대한 이론적 분석은 충분하지 않다. 서로 다른 클래스 간 거리를 키우는 것은 직관적으로 좋지만, 실제로는 클래스 구조가 복잡하거나 도메인 간 class imbalance가 큰 경우 과도한 분리가 오히려 정렬을 방해할 수도 있다. 논문은 이 점을 깊게 다루지 않는다.

셋째, deep learning setting으로의 확장은 제안만 있고 실제 검증은 없다. 논문은 shallow feature 기반 subspace adaptation 수준에서 결과를 보였고, future work로 deep learning과 adversarial learning으로의 확장을 언급한다. 따라서 오늘날의 강력한 deep UDA 기법과 직접 비교해 얼마나 경쟁력 있는지는 이 논문만으로는 알 수 없다.

넷째, 실험이 이미지 분류에 한정되어 있다. domain adaptation은 segmentation, detection, time-series, BCI, NLP 등 다양한 응용이 있는데, 본 논문의 실험은 비교적 전통적인 visual classification 벤치마크에 집중되어 있다. 따라서 일반성은 추가 검증이 필요하다.

다섯째, 일부 task에서는 JDA/BDA보다 개선이 없거나 오히려 뒤진다. 즉, 평균적으로 우수하다는 결론은 타당하지만, “항상 더 좋다”라고 보기는 어렵다. 특히 digit recognition이나 일부 Office task에서는 개선폭이 작거나 음수인 경우도 있다. 이 점은 방법의 장점이 특정 데이터 구조에서 더 크게 드러날 수 있음을 시사한다.

비판적으로 보면, 이 논문은 “joint probability를 직접 다룬다”는 이론적 메시지가 강점이지만, 실제 최종 형태는 여전히 클래스별 평균 차이를 사용한 MMD 기반 정렬이다. 따라서 표현력이 근본적으로 완전히 새로운 것은 아니며, 좋은 pseudo-label과 좋은 feature 표현이 성능을 좌우한다는 점에서는 기존 shallow DA와 구조적으로 비슷하다. 그럼에도 불구하고, same-class alignment와 different-class separation을 함께 넣은 설계는 분류 관점에서 분명 의미 있는 개선이다.

## 6. 결론

이 논문은 domain adaptation에서 분포 차이를 측정하는 기존 방식의 한계를 짚고, 이를 대체하는 **DJP-MMD**를 제안했다. 핵심은 marginal discrepancy와 conditional discrepancy를 따로 더하는 대신, $P(X|Y)P(Y)$ 관점에서 **joint probability discrepancy를 직접 계산**하는 것이다. 여기에 더해, 같은 클래스의 source-target discrepancy는 줄이고 다른 클래스의 source-target discrepancy는 키우는 방식으로 transferability와 discriminability를 동시에 강화했다.

이를 구현한 JPDA 프레임워크는 generalized eigen-decomposition으로 최적화 가능하며, kernelization도 지원한다. 여섯 개 이미지 분류 벤치마크에서 TCA, JDA, BDA 대비 평균적으로 더 높은 정확도를 보였고, 시각화와 ablation study를 통해 joint probability 기반 정렬과 discriminability 항의 기여를 함께 보여주었다.

실제 적용 측면에서 이 연구는 전통적인 subspace-based DA 방법을 더 분류 친화적으로 설계하는 방법을 제시했다는 의미가 있다. 특히 pseudo-label 기반 iterative adaptation, class-aware alignment, discriminative regularization이라는 관점은 이후 deep UDA로도 자연스럽게 이어질 수 있다. 향후 연구에서는 논문이 제안한 대로 deep learning이나 adversarial learning 안에 DJP-MMD를 통합해, 더 복잡한 feature 표현과 함께 검증하는 방향이 중요할 것이다.

전반적으로 이 논문은 새로운 대규모 모델을 제안한 것은 아니지만, **분포 정렬의 정의를 더 정확하고 분류 친화적으로 재구성했다는 점**에서 가치가 있다. domain adaptation에서 “무엇을 맞출 것인가”를 다시 묻고, 그 답을 same-class alignment와 different-class separation의 결합으로 제시했다는 점이 이 연구의 핵심 기여라고 볼 수 있다.
