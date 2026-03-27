# Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation

* **저자**: Chao Chen, Zhihong Chen, Boyuan Jiang, Xinyu Jin
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1808.09347](https://arxiv.org/abs/1808.09347)

## 1. 논문 개요

이 논문은 **Unsupervised Deep Domain Adaptation**에서 흔히 사용하는 “도메인 정렬(domain alignment)”만으로는 충분하지 않다는 문제의식에서 출발한다. 기존 방법들은 source와 target의 feature distribution 차이를 줄이는 데 집중하지만, 저자들은 이것만으로는 domain shift가 완전히 사라지지 않으므로, target sample 중 일부가 여전히 decision boundary 근처나 class center에서 멀리 떨어진 위치에 남게 된다고 본다. 그 결과, source에서 학습한 classifier가 target에서 오분류를 일으키기 쉽다.

논문의 핵심 목표는 이 한계를 줄이기 위해 **도메인 정렬과 판별적 특징 학습(discriminative feature learning)을 동시에 수행**하는 것이다. 즉, source와 target이 공유하는 feature space를 domain-invariant하게 만드는 동시에, 그 공간 안에서 feature들이 **같은 클래스끼리는 더 조밀하게(intra-class compactness), 다른 클래스끼리는 더 멀게(inter-class separability)** 배치되도록 학습한다. 저자들은 이런 구조가 단순히 classification 성능만 높이는 것이 아니라, 오히려 domain alignment 자체도 더 잘 되게 도와준다고 주장한다.

이 문제는 실제로 중요하다. 도메인 적응에서는 target label이 없기 때문에, alignment가 애매하게 된 상태에서 classifier가 잘못된 경계를 가지면 target 쪽 성능이 크게 무너질 수 있다. 특히 source와 target의 차이가 큰 어려운 transfer task에서 이런 문제가 더 두드러진다. 이 논문은 그 틈을 파고들어, “alignment 이후의 feature geometry”까지 같이 설계해야 transfer가 더 잘 된다는 점을 강조한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 간단하지만 강력하다. **좋은 domain adaptation은 단지 source와 target을 섞어 놓는 것만으로 끝나지 않고, 그 섞인 공간이 분류에 유리한 구조를 가져야 한다**는 것이다.

기존의 discrepancy-based 방법이나 adversarial adaptation 방법은 대체로 “source와 target feature의 분포 차이”를 줄이는 데 초점을 둔다. 그러나 저자들은 분포 차이가 줄어들더라도, 각 클래스의 feature cluster가 퍼져 있거나 서로 충분히 떨어져 있지 않다면, target sample은 여전히 boundary 근처에 걸쳐 오분류될 수 있다고 본다. 따라서 **alignment와 discrimination을 함께 최적화해야 한다**는 것이 JDDA의 기본 아이디어다.

이때 흥미로운 점은, 저자들이 target label이 없는 상황에서 target feature에 직접 discriminative constraint를 거는 대신, **source의 labeled feature를 더 discriminative하게 만들면 정렬된 target feature도 간접적으로 더 discriminative해질 것**이라고 본다는 점이다. 즉, target에 pseudo-label을 붙여 강제로 묶는 방식이 아니라, source의 supervision을 이용해 shared space의 구조를 먼저 정교하게 만들고, alignment를 통해 target이 그 구조를 따라오도록 한다.

저자들은 이를 위해 두 가지 discriminative learning 전략을 제안한다. 하나는 **Instance-Based discriminative loss**이고, 다른 하나는 **Center-Based discriminative loss**이다. 전자는 샘플 쌍(pair) 수준에서 같은 클래스는 가깝게, 다른 클래스는 margin 이상 멀게 만드는 방식이고, 후자는 각 클래스 중심(center)을 도입해 샘플-중심 거리와 중심-중심 거리를 제어하는 방식이다. 논문은 이 두 방식 모두 효과적이지만, 계산 효율과 수렴 속도 측면에서는 Center-Based 방식이 더 유리하다고 설명한다.

## 3. 상세 방법 설명

전체 구조는 **shared weights를 갖는 two-stream CNN**이다. 한 스트림은 labeled source data를 입력받고, 다른 스트림은 unlabeled target data를 입력받는다. 두 스트림은 같은 파라미터 $\Theta$를 공유한다. 논문에서 중요한 특징은 bottleneck layer에 대해 두 가지 추가 제약을 건다는 점이다. 하나는 source와 target의 분포 차이를 줄이는 **domain discrepancy loss**이고, 다른 하나는 source feature를 더 잘 뭉치고 더 잘 분리되게 만드는 **discriminative loss**이다.

전체 학습 목표는 다음과 같다.

$$
\mathcal{L}(\Theta \mid X_s, Y_s, X_t) = \mathcal{L}_s + \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_d
$$

여기서 $\mathcal{L}_s$는 source supervised classification loss이고, $\mathcal{L}_c$는 domain discrepancy loss, $\mathcal{L}_d$는 저자들이 새로 도입한 discriminative loss이다. $\lambda_1$, $\lambda_2$는 각각 domain alignment와 discriminative learning의 중요도를 조절하는 하이퍼파라미터다.

### 3.1 Source classification loss

source domain에는 label이 있으므로 일반적인 supervised classification loss를 사용한다.

$$
\mathcal{L}_s = \frac{1}{n_s}\sum_{i=1}^{n_s} c(\Theta \mid x_i^s, y_i^s)
$$

즉, source 데이터에 대해 softmax 기반 분류 손실을 최소화한다. 이 부분은 기존 deep classifier와 동일한 역할을 한다.

### 3.2 Domain discrepancy loss: CORAL

도메인 정렬에는 **CORAL (Correlation Alignment)** 이 사용된다. CORAL은 source와 target feature의 covariance를 맞추는 방법이다. 논문은 bottleneck layer의 source feature $H_s$와 target feature $H_t$에 대해 다음 loss를 사용한다.

$$
\mathcal{L}_c = CORAL(H_s, H_t) \frac{1}{4L^2} \left| Cov(H_s) - Cov(H_t) \right|_F^2
$$

여기서 $L$은 bottleneck layer의 feature dimension이고, $|\cdot|_F^2$는 squared Frobenius norm이다. 각 covariance는 mini-batch 단위로 계산된다.

$$
Cov(H_s)=H_s^\top J_b H_s, \qquad Cov(H_t)=H_t^\top J_b H_t
$$

또한

$$
J_b = I_b - \frac{1}{b}\mathbf{1}\mathbf{1}^\top
$$

는 배치 중심화(centralization)를 위한 행렬이다. 즉, 이 항은 source와 target의 2차 통계량을 맞춰 domain gap을 줄이는 역할을 한다.

### 3.3 Instance-Based discriminative loss

첫 번째 제안은 **샘플 간 거리 기반** discriminative loss다. 같은 클래스 샘플끼리는 일정 거리 이하로 붙게 만들고, 다른 클래스 샘플끼리는 일정 거리 이상 떨어지게 만든다. 두 source feature $h_i^s$, $h_j^s$에 대해 loss는 다음과 같이 정의된다.

같은 클래스일 때:

$$
\mathcal{J}_d^I(h_i^s, h_j^s) = \max(0, |h_i^s-h_j^s|_2 - m_1)^2
$$

다른 클래스일 때:

$$
\mathcal{J}_d^I(h_i^s, h_j^s) = \max(0, m_2 - |h_i^s-h_j^s|_2)^2
$$

즉, 같은 클래스는 거리가 $m_1$보다 크면 벌점을 주고, 다른 클래스는 거리가 $m_2$보다 작으면 벌점을 준다. 전체 loss는 모든 source sample pair에 대해 합산한다.

$$
\mathcal{L}_d^I = \sum_{i,j=1}^{n_s}\mathcal{J}_d^I(h_i^s, h_j^s)
$$

논문은 이를 pairwise distance matrix $D^H$와 indicator matrix $L$을 사용해 더 간단히 쓴다.

$$
\mathcal{L}_d^I = \alpha |\max(0, D^H-m_1)^2 \circ L|_{sum} + |\max(0, m_2-D^H)^2 \circ (1-L)|_{sum}
$$

여기서 $\alpha$는 intra-class와 inter-class 항의 비중을 조절하는 계수다. 이 방식은 직관적으로 분명하다. label이 같은 source feature는 서로 뭉치게 하고, 다른 label은 margin을 두고 벌어지게 하므로 feature space가 더 분리된다. 다만 모든 샘플 쌍 간 거리를 계산해야 하므로 계산량이 크다.

### 3.4 Center-Based discriminative loss

두 번째 제안은 계산 효율을 높이기 위해 **class center**를 도입하는 방식이다. 각 샘플이 자기 클래스 중심에 가까워지도록 만들고, 서로 다른 클래스 중심끼리는 멀어지도록 만든다.

$$
\mathcal{L}_d^C = \beta \sum_{i=1}^{n_s}\max(0,|h_i^s-c_{y_i}|_2^2-m_1) + \sum_{i,j=1,i\neq j}^{c} \max(0,m_2-|c_i-c_j|_2^2)
$$

첫 번째 항은 **intra-class compactness**를 담당한다. 샘플 $h_i^s$가 자기 클래스 중심 $c_{y_i}$에서 너무 멀면 벌점을 준다. 두 번째 항은 **inter-class separability**를 담당한다. 서로 다른 클래스 중심 $c_i$, $c_j$가 너무 가까우면 벌점을 준다.

중요한 실무적 문제는 “global class center”를 어떻게 유지하느냐이다. 전체 dataset의 feature 평균을 매 스텝마다 정확히 구하기는 어렵기 때문에, 논문은 mini-batch 기반으로 center를 점진적으로 업데이트한다. 클래스 $j$에 대한 center update는 다음과 같다.

$$
\Delta c_j = \frac{\sum_{i=1}^{b}\delta(y_i=j)(c_j-h_i^s)} {1+\sum_{i=1}^{b}\delta(y_i=j)}
$$

$$
c_j^{t+1}=c_j^t-\gamma \cdot \Delta c_j^t
$$

여기서 $\gamma$는 center update learning rate다. 첫 반복에서는 batch class center로 초기화하고, 이후 각 배치가 들어올 때마다 점진적으로 global center를 갱신한다.

논문은 이 loss를 더 간단히 다음처럼 정리한다.

$$
\mathcal{L}_d^C = \beta |\max(0,H^c-m_1)|_{sum} + |\max(0,m_2-D^c)\circ M|_{sum}
$$

여기서 $H^c$는 각 샘플과 해당 class center 사이 거리 항을 모은 것이고, $D^c$는 class center들 간 pairwise distance matrix다.

### 3.5 학습 절차

JDDA-I는 다음 loss를 사용한다.

$$
\mathcal{L} = \mathcal{L}_s+\lambda_1\mathcal{L}_c+\lambda_2^I\mathcal{L}_d^I
$$

JDDA-C는 다음 loss를 사용한다.

$$
\mathcal{L} = \mathcal{L}_s+\lambda_1\mathcal{L}_c+\lambda_2^C\mathcal{L}_d^C
$$

JDDA-I는 모든 항이 입력 feature에 대해 미분 가능하므로 일반적인 backpropagation으로 학습할 수 있다. JDDA-C는 네트워크 파라미터뿐 아니라 global class center도 같이 업데이트해야 한다.

저자들의 주장은 다음과 같이 정리할 수 있다. 첫째, discriminative source feature는 target alignment를 더 쉽게 만든다. 둘째, 정렬된 공간에서 클래스 간 간격이 커지면 target sample이 cluster edge 쪽에 있어도 오분류 가능성이 줄어든다. 즉, $\mathcal{L}_d$는 단순히 source training accuracy를 높이는 regularizer가 아니라, domain adaptation의 geometry를 개선하는 역할을 한다.

## 4. 실험 및 결과

논문은 두 종류의 벤치마크에서 실험한다. 하나는 **Office-31**, 다른 하나는 **digital recognition dataset**이다. Office-31은 Amazon, Webcam, DSLR의 세 도메인으로 구성되며 총 6개의 transfer task를 평가한다. digital recognition 쪽은 SVHN, MNIST, MNIST-M, USPS, synthetic digits를 사용하고, 네 가지 transfer pair를 평가한다.

비교 대상은 DDC, DAN, DANN, CMD, ADDA, CORAL 등 당시 대표적인 deep domain adaptation 방법들이다. Office-31에서는 ImageNet pretrained ResNet-50을 fine-tuning했고, digital recognition에서는 modified LeNet을 사용했다. 모든 방법은 TensorFlow와 Adam optimizer로 구현했고, 배치 크기는 총 256, 각 도메인에서 128개씩 샘플링했다. 학습률은 $\eta=10^{-4}$, center update용 learning rate는 $\gamma=0.5$로 설정했다. adaptation factor $\lambda$는 고정값 대신 training progress에 따라 증가하는 progressive schedule을 사용했다.

하이퍼파라미터는 JDDA-I에 대해 $\lambda_2^I = 0.03$, JDDA-C에 대해 $\lambda_2^C = 0.01$로 고정했고, margin은 전 실험에서 $m_1=0$, $m_2=100$으로 두었다. 이 설정은 SVHN $\rightarrow$ MNIST에서 먼저 고른 뒤 다른 task에도 그대로 사용했다고 설명한다.

### 4.1 Office-31 결과

Office-31 평균 정확도는 다음과 같다. ResNet은 75.0, DDC는 75.8, DAN은 78.5, DANN은 76.7, CMD는 77.5, CORAL은 78.0, JDDA-I는 79.2, JDDA-C는 80.2를 기록했다. 특히 JDDA-C가 전체 평균 최고 성능을 보였다. 어려운 task인 A $\rightarrow$ W에서는 JDDA-C가 82.6으로 CORAL의 79.3, DAN의 78.3보다 높았고, W $\rightarrow$ A에서도 JDDA-C가 66.7로 CORAL의 63.4, DAN의 64.2보다 높았다. 논문은 이 결과를 바탕으로, 제안 방법이 특히 도메인 차이가 큰 어려운 task에서 더 유리하다고 해석한다.

쉬운 task인 D $\rightarrow$ W나 W $\rightarrow$ D에서는 기존 방법과 큰 차이가 나지 않지만, JDDA는 여기서도 동급 이상의 성능을 유지한다. 따라서 저자들은 “정렬과 판별성의 결합”이 작은 규모의 실전 adaptation setting에서도 유효하다고 주장한다.

### 4.2 Digital recognition 결과

digital recognition에서는 성능 향상이 더 크다. 평균 정확도는 Modified LeNet 71.6, DDC 79.0, DAN 81.0, DANN 78.5, CMD 88.6, ADDA 85.4, CORAL 91.0, JDDA-I 93.8, JDDA-C 94.3이다. 여기서도 JDDA-C가 최고 평균 성능이다.

특히 SVHN $\rightarrow$ MNIST에서는 CORAL이 89.5인데 JDDA-I는 93.1, JDDA-C는 94.2를 기록했다. MNIST $\rightarrow$ MNIST-M에서는 CORAL 81.6, JDDA-I 87.5, JDDA-C 88.4로 개선 폭이 크다. 논문은 이 두 task가 특히 어려운 이유를 설명한다. SVHN은 배경, 블러, 기울기, 대비, 회전 변화가 심하고, MNIST-M은 MNIST digit mask와 무작위 자연 이미지 배경을 합성한 데이터라서 MNIST와 시각적 차이가 크기 때문이다. 따라서 JDDA가 이런 어려운 setting에서 크게 좋아진 것은, 단순 정렬을 넘어 feature structure를 정리한 효과로 해석할 수 있다.

### 4.3 Feature visualization

논문은 정성적 분석으로 2D feature plot과 t-SNE visualization을 제시한다. source-only에 discriminative loss를 넣지 않은 경우 feature가 길게 늘어진 strip 형태로 나타나지만, discriminative loss를 넣으면 같은 클래스끼리 더 조밀한 cluster를 이루고 cluster 간 간격도 커진다고 설명한다. 이는 $\mathcal{L}_d$가 실제로 feature geometry를 바꾼다는 근거다.

또한 SVHN $\rightarrow$ MNIST에서 CORAL만 사용한 경우보다 JDDA를 사용한 경우가, class 기준으로 보았을 때 inter-class gap이 더 선명하고, domain 기준으로 보았을 때 source와 target category alignment도 더 잘 된다고 해석한다. 즉, discriminative source space가 결국 target alignment 품질도 높인다는 논문의 핵심 주장을 시각화 결과가 뒷받침한다.

### 4.4 Convergence와 parameter sensitivity

저자들은 convergence curve도 비교한다. JDDA는 target test error를 더 낮게 만들 뿐 아니라 더 안정적으로 수렴한다고 보고하며, 특히 JDDA-C가 가장 빠르게 수렴한다고 말한다. 그 이유는 global cluster information을 고려하기 때문이라고 설명한다.

또한 $\lambda_2$에 대한 민감도 분석에서 성능은 bell-shaped curve를 보인다. 즉, discriminative loss의 비중이 너무 작아도 효과가 약하고, 너무 커도 domain alignment를 방해해 성능이 떨어진다. 논문은 이를 통해 **alignment와 discrimination의 균형**이 중요하다는 점을 강조한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 매우 설득력 있다는 점이다. 기존 domain adaptation 연구가 “도메인 간 차이 감소”에 과도하게 집중해 왔다는 비판은 타당하며, 실제로 classification은 결국 feature space 안의 class structure에 의해 결정된다. 이 논문은 그 사실을 명확히 짚고, **domain-invariant하면서도 discriminative한 representation**이라는 더 정교한 목표를 제시했다.

또 다른 강점은 방법이 비교적 단순하고 기존 프레임워크에 쉽게 결합된다는 점이다. CORAL 같은 기존 alignment loss 위에 discriminative loss를 얹는 구조이므로 구현 난도가 지나치게 높지 않다. 특히 Center-Based 버전은 pairwise distance를 전부 계산하는 방식보다 훨씬 효율적이고, 실험에서도 빠른 수렴을 보였다.

실험도 설득력이 있다. Office-31과 digit adaptation이라는 서로 다른 성격의 벤치마크에서 일관되게 성능 향상을 보였고, 어려운 task에서 개선 폭이 크다는 점은 논문의 주장을 잘 뒷받침한다. 단지 평균 정확도만 조금 높아진 것이 아니라, 어떤 상황에서 왜 개선되는지를 함께 설명한다는 점이 좋다.

하지만 한계도 있다. 첫째, discriminative loss는 **source label에만 의존**한다. 따라서 source 구조가 target에도 잘 전이된다는 가정이 필요하다. 논문은 이를 “shared feature space에서 두 도메인이 잘 정렬된다면 target도 자동으로 discriminative해질 것”이라고 설명하지만, 클래스 조건부 분포가 크게 다르거나 class-wise mismatch가 심한 상황에서는 이 가정이 약해질 수 있다. 논문은 이런 극단적 경우를 별도로 분석하지는 않는다.

둘째, target에 대해 explicit한 class-level alignment는 하지 않는다. 다시 말해 conditional alignment나 pseudo-label refinement 같은 장치는 없다. 따라서 이후 등장한 class-aware adaptation, contrastive UDA, clustering 기반 UDA와 비교하면, target semantic structure를 직접 활용하는 수준은 제한적이다. 물론 이것은 이 논문 시점에서는 자연스러운 설계지만, 후속 연구와 비교할 때는 분명한 범위 제한이다.

셋째, 하이퍼파라미터 민감도는 존재한다. 저자들도 $\lambda_2$가 너무 크거나 작으면 성능이 나빠진다고 직접 보여 준다. 즉, discriminative regularization은 무조건 세게 거는 것이 아니라 alignment와 균형을 잡아야 한다. 이는 실전 적용 시 task별 tuning 부담이 될 수 있다.

넷째, 논문 추출문만으로는 일부 시각화 그림의 정확한 좌표 구조나 세부 축 설명까지는 복원할 수 없다. 따라서 Figure 3, Figure 4, Figure 5, Figure 6에 대한 해석은 논문 본문 설명에 근거한 것이며, 그림 자체의 세부 수치까지 확인한 것은 아니다.

종합하면, 이 논문은 “domain alignment만으로는 부족하다”는 점을 명확히 제기하고, 이를 간단하면서도 효과적인 discriminative loss 설계로 연결했다는 점에서 의미가 크다. 다만 target 자체의 구조를 적극적으로 이용하지 않는다는 점에서는 후속 발전 여지가 남아 있다.

## 6. 결론

이 논문은 unsupervised deep domain adaptation에서 **도메인 정렬(domain alignment)** 과 **판별적 특징 학습(discriminative feature learning)** 을 함께 수행해야 한다는 관점을 제시한다. 구체적으로 source classification loss와 CORAL 기반 domain discrepancy loss에 더해, source feature가 같은 클래스끼리는 더 조밀하고 다른 클래스끼리는 더 멀어지도록 만드는 discriminative loss를 추가했다. 이 discriminative loss는 Instance-Based와 Center-Based 두 가지 형태로 제안되었고, 실험적으로 두 방식 모두 기존 방법들을 능가했으며, 특히 JDDA-C가 계산 효율과 수렴 속도 면에서도 유리했다.

논문의 주요 기여는 세 가지로 요약할 수 있다. 첫째, deep domain adaptation에서 discriminative feature learning을 본격적으로 결합한 초기 시도 중 하나라는 점이다. 둘째, source label만으로도 shared feature space의 구조를 개선해 target adaptation까지 돕는다는 설계를 보여 주었다는 점이다. 셋째, 실제 벤치마크에서 어려운 transfer task일수록 그 효과가 크게 나타난다는 점을 실험으로 입증했다는 점이다. 이 논문의 아이디어는 이후의 class-aware alignment, center/contrastive 기반 UDA, clustering-assisted adaptation 같은 흐름과도 자연스럽게 연결된다.

실제 적용 관점에서 보면, 이 연구는 “좋은 adaptation representation은 domain-invariant할 뿐 아니라 class-discriminative해야 한다”는 중요한 원칙을 남긴다. 향후 연구는 이 원칙을 더 발전시켜, target pseudo-label의 신뢰도를 높이거나, conditional alignment를 결합하거나, contrastive learning과 통합하는 방향으로 확장할 수 있다. 그런 의미에서 JDDA는 단순한 성능 개선 기법을 넘어서, domain adaptation에서 feature geometry를 어떻게 다뤄야 하는지에 대한 중요한 관점을 제공한 논문이라고 볼 수 있다.
