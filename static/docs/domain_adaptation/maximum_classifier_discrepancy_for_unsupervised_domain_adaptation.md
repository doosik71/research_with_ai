# Maximum Classifier Discrepancy for Unsupervised Domain Adaptation

* **저자**: Kuniaki Saito, Kohei Watanabe, Yoshitaka Ushiku, Tatsuya Harada
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1712.02560](https://arxiv.org/abs/1712.02560)

## 1. 논문 개요

이 논문은 unsupervised domain adaptation, 즉 source domain에는 라벨이 있고 target domain에는 라벨이 없는 상황에서 target 성능을 높이는 새로운 적응 방법을 제안한다. 문제의 핵심은 source와 target의 분포 차이 때문에 source에서 잘 학습한 분류기가 target에서는 잘 작동하지 않는다는 점이다. 예를 들어 웹에서 수집한 이미지로 학습한 분류기는 실제 카메라로 촬영한 이미지에서는 조명, 노이즈, 시점 차이 때문에 성능이 크게 떨어질 수 있다.

기존의 대표적 접근은 domain classifier를 두고, feature generator가 source와 target feature를 구분하지 못하도록 학습하는 adversarial adaptation 방식이었다. 그러나 논문은 이러한 방식에 두 가지 중요한 문제가 있다고 지적한다. 첫째, domain classifier는 단지 “source냐 target이냐”만 구분할 뿐, 실제 분류 문제의 decision boundary를 고려하지 않는다. 따라서 target feature가 클래스 경계 근처의 애매한 위치에 와도 domain-level에서는 잘 맞춰졌다고 간주될 수 있다. 둘째, 두 도메인의 feature distribution을 완전히 일치시키려는 목표 자체가 도메인 고유 특성 때문에 지나치게 어렵거나 비현실적일 수 있다.

이 논문의 목표는 단순히 도메인 분포를 뭉뚱그려 맞추는 대신, task-specific decision boundary를 활용하여 target feature가 source support 안쪽의 “분류 가능한 영역”으로 들어오도록 만드는 것이다. 이를 위해 저자들은 두 개의 task classifier 사이의 예측 불일치(discrepancy)를 이용해 target 샘플이 source support 바깥에 있는지를 탐지하고, generator가 այդ 불일치를 줄이도록 학습하는 방법을 제안한다. 이 문제는 domain adaptation의 본질인 “어떻게 target을 source의 분류 가능 영역으로 끌어들일 것인가”에 직접 답하려는 시도라는 점에서 중요하다.

## 2. 핵심 아이디어

이 논문의 핵심 직관은 매우 명확하다. source 데이터에서 잘 학습된 서로 다른 두 classifier가 있을 때, source support 안쪽에 위치한 샘플이라면 두 classifier의 예측이 대체로 비슷해야 한다. 반면 source support 바깥에 있거나 class boundary 근처에 놓인 target 샘플은 두 classifier가 서로 다른 예측을 내릴 가능성이 높다. 따라서 두 classifier의 출력 차이를 보면, 어떤 target 샘플이 “안전한 영역”에 있고 어떤 샘플이 “위험한 영역”에 있는지 간접적으로 알 수 있다.

기존 adversarial DA는 domain discriminator를 두고 domain confusion을 유도한다. 반면 이 논문은 discriminator 역할을 domain classifier가 아니라 task-specific classifier 두 개가 맡도록 바꾼다. 이 차이가 본질적이다. 기존 방식은 domain gap 자체만 줄이려 하고, 어떤 feature가 실제 class decision boundary와 어떤 관계에 있는지 잘 반영하지 못한다. 반대로 제안 방법은 classification task에 직접 연결된 classifier의 disagreement를 사용하므로, target feature를 더 “분별력 있는(discriminative)” 위치로 이동시키려 한다.

구체적으로는 다음의 minimax 구조를 가진다. 먼저 두 classifier는 source에서는 정확하게 분류하도록 유지하면서, target에서는 서로 최대한 다르게 예측하도록 학습된다. 이렇게 하면 target 중 source support 바깥에 있는 샘플들이 더 강하게 드러난다. 그런 다음 generator는 같은 target 샘플들에 대해 두 classifier의 출력을 다시 비슷하게 만들도록 학습된다. 이 반복 과정은 결국 target feature를 source support에 더 가깝게 이동시키는 효과를 낸다.

즉, 이 논문의 차별점은 “도메인 자체를 속이는 것”이 아니라 “클래스 경계를 고려하는 분류기 불일치”를 통해 적응을 수행한다는 데 있다. 논문 제목의 Maximum Classifier Discrepancy는 바로 이 핵심 전략, 즉 두 classifier의 차이를 의도적으로 키워서 문제 샘플을 드러내고 다시 줄이면서 generator를 교정하는 과정을 가리킨다.

## 3. 상세 방법 설명

논문은 모델을 세 부분으로 나눈다. 첫째는 feature generator $G$이고, 둘째와 셋째는 각각 classifier $F_1$, $F_2$이다. 입력 이미지 $\mathbf{x}_s$ 또는 $\mathbf{x}_t$는 먼저 $G$를 거쳐 feature로 변환되고, 그 feature가 두 classifier에 전달된다. 각 classifier는 $K$개 클래스에 대한 logits를 출력하고, softmax를 거쳐 확률 분포를 만든다. 이를 각각 $p_1(\mathbf{y}|\mathbf{x})$, $p_2(\mathbf{y}|\mathbf{x})$로 표기한다.

저자들이 정의하는 핵심 quantity는 discrepancy이다. 이는 두 classifier의 target 예측 분포가 얼마나 다른지를 나타낸다. 논문에서 사용한 discrepancy loss는 다음과 같다.

$$
d(p_1, p_2)=\frac{1}{K}\sum_{k=1}^{K} |{p_1}_k - {p_2}_k|
$$

여기서 ${p_1}_k$와 ${p_2}_k$는 클래스 $k$에 대한 두 classifier의 확률 출력이다. 즉, 각 클래스별 확률 차이의 절댓값을 평균낸 $L_1$ 거리이다. 저자들은 이 선택이 이론적 연결성과 실험적 안정성 측면에서 적절하다고 주장하며, $L_2$ distance는 잘 작동하지 않았다고 명시한다.

전체 학습은 세 단계로 이루어진다.

### Step A: source supervised learning

먼저 $G$, $F_1$, $F_2$를 모두 사용하여 source 샘플을 잘 분류하도록 학습한다. 이는 source domain에서 discriminative feature와 classifier를 확보하기 위한 기본 단계다. 목적함수는 일반적인 softmax cross-entropy이다.

$$
\min_{G,F_1,F_2} \mathcal{L}(X_s, Y_s)
$$

논문에 제시된 supervised classification loss는 source 데이터 $(\mathbf{x}_s, y_s)$에 대해 정답 클래스의 로그 확률을 최대화하는 형태이다. 이 단계가 없으면 classifier 자체가 task를 제대로 배우지 못하므로 이후 discrepancy를 활용하는 전략이 무너진다.

### Step B: classifier update to maximize discrepancy

이 단계에서는 generator $G$를 고정한 뒤, 두 classifier $F_1, F_2$를 업데이트한다. 목적은 두 classifier가 source는 계속 잘 분류하면서, target에서는 가능한 한 서로 다른 예측을 내도록 만드는 것이다. 이를 위해 source classification loss는 유지하고, target discrepancy loss는 크게 만들도록 학습한다. 논문 표기상 objective는 다음과 같다.

$$
\min_{F_1,F_2} \mathcal{L}(X_s, Y_s) - \mathcal{L}_{adv}(X_t)
$$

여기서

$$
\mathcal{L}_{adv}(X_t)=\mathbb{E}_{\mathbf{x}_t \sim X_t} \left[d\left(p_1(\mathbf{y}|\mathbf{x}_t), p_2(\mathbf{y}|\mathbf{x}_t)\right)\right]
$$

이다. 이 목적식은 source loss는 줄이고, target discrepancy는 키우는 효과를 가진다. 중요한 점은 source classification loss를 함께 넣는다는 것이다. 논문은 이 항을 제거하면 성능이 크게 떨어졌다고 직접 보고한다. 이유는 classifier가 무의미하게 서로 다른 예측을 하도록 망가지는 것을 막고, source에서의 정상적인 class boundary를 유지해야 하기 때문이다.

### Step C: generator update to minimize discrepancy

이번에는 두 classifier를 고정하고 generator만 업데이트한다. generator는 target 샘플에 대해 두 classifier의 출력 차이를 최소화하도록 학습된다.

$$
\min_G \mathcal{L}_{adv}(X_t)
$$

이 단계의 의미는 명확하다. 현재 classifier들이 “이 샘플은 source support 바깥이라 서로 다르게 본다”고 표시한 target feature를 generator가 움직여, 두 classifier가 다시 비슷하게 보도록 만드는 것이다. 논문은 이 업데이트를 한 mini-batch에 대해 $n$번 반복할 수 있게 두었고, $n$은 generator와 classifier 간의 균형을 조절하는 hyper-parameter이다.

### 전체 알고리즘의 해석

세 단계를 이어서 보면 다음과 같다. Step A는 source에서 분류 가능한 기준면을 만든다. Step B는 target 중 경계 바깥, 혹은 source support 바깥에 있는 샘플을 classifier discrepancy로 드러낸다. Step C는 generator가 այդ 샘플을 더 안전한 위치로 옮긴다. 이 과정을 반복하면 target feature가 source support 내부로 점차 끌려오게 된다.

이 논문에서 특히 중요한 점은 “완전한 distribution matching”이 목표가 아니라는 것이다. 저자들은 t-SNE 시각화를 통해 source와 target이 완전히 겹치지는 않더라도, class-wise로 더 분별력 있게 정렬되면 적응이 성공할 수 있음을 보여준다. 즉, 필요한 것은 전역 분포 일치가 아니라 decision boundary 관점에서의 유효한 정렬이다.

### 이론적 통찰

논문은 Ben-David 등의 domain adaptation theory와의 연결도 제시한다. target error는 source error, domain divergence, 그리고 ideal joint hypothesis의 shared error $\lambda$에 의해 upper bound된다는 고전적 결과를 인용한다.

$$
\forall h \in H,;
R_{\mathcal{T}}(h)\leq R_{\mathcal{S}}(h)+\frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S},\mathcal{T})+\lambda
\leq
R_{\mathcal{S}}(h)+\frac{1}{2}d_{\mathcal{H}}(\mathcal{S},\mathcal{T})+\lambda
$$

기존 DANN류 방법은 주로 $d_{\mathcal{H}}(\mathcal{S},\mathcal{T})$를 줄이는 방향이다. 반면 본 논문은 $d_{\mathcal{H}\Delta\mathcal{H}}$에 주목한다. 이 항은 두 가설 $h, h'$ 사이의 disagreement를 기반으로 정의되므로, source에서 잘 맞는 두 classifier를 가정하면 source 쪽 disagreement는 작고, 결국 target에서의 disagreement를 줄이는 것이 중요해진다. 논문은 이를 바탕으로 자신들의 minimax 문제가 이론적으로 $\mathcal{H}\Delta\mathcal{H}$-distance와 밀접하게 연결된다고 설명한다.

다만 이 부분은 엄밀한 새로운 정리(proof)를 제시하기보다는, 제안한 방법이 기존 이론과 어떻게 맞닿아 있는지를 직관적으로 보여주는 수준에 가깝다. 즉, 논문의 이론 파트는 엄밀한 수학적 보장보다는 방법의 동기를 정당화하는 역할을 한다고 보는 것이 적절하다.

### 보충 자료의 추가 사항

보충 자료에서는 VisDA 실험에 class balance loss를 추가로 사용했다고 설명한다. 이 항은 target 샘플이 특정 클래스에만 치우치지 않도록 하는 목적이다.

$$
\mathbb{E}_{\mathbf{x}_t \sim X_t}\sum_{k=1}^{K}\log p(y=k|\mathbf{x}_t)
$$

논문은 이 loss에 $\lambda=0.01$을 곱해 Step 2와 Step 3에 추가했다고 밝힌다. 중요한 점은 이 class balance loss가 본 방법의 핵심 메커니즘은 아니며, VisDA에서 성능을 더 안정화하기 위한 보조 항이라는 것이다.

또한 supplementary에서는 gradient reversal layer(GRL)를 적용하면 classifier discrepancy를 키우는 업데이트와 generator discrepancy를 줄이는 업데이트를 한 번에 처리할 수 있다고 설명한다. 원 논문 본문에서는 3-step 학습을 제안했지만, 보충 자료는 이 과정을 더 간단한 one-step adversarial update로 바꿀 가능성도 보여준다.

## 4. 실험 및 결과

논문은 toy experiment, digit classification, traffic sign classification, large-scale object classification, semantic segmentation까지 매우 폭넓게 실험한다. 이는 제안 방법이 특정 작은 벤치마크에서만 통하는 것이 아니라 여러 문제 설정에서 비교적 일관되게 동작함을 보이기 위한 구성이다.

### Toy dataset 실험

가장 먼저 두 개의 intertwined moons 문제를 사용해 방법의 작동 원리를 시각적으로 설명한다. source는 두 개의 반달 모양 클래스로 구성되고, target은 source를 회전시켜 생성한다. source only 모델에서는 target 쪽에서 decision boundary가 잘 맞지 않고, classifier 둘 사이 차이도 크지 않다. Step C 없이 classifier discrepancy만 키운 모델에서는 target에서 두 classifier가 크게 갈라진다. 반면 제안 방법은 discrepancy를 키운 뒤 다시 generator가 이를 줄이도록 학습하여, target 샘플이 두 classifier가 합의하는 영역으로 이동하는 모습을 보인다.

이 toy 결과의 의미는 단순한 시각화 이상이다. 논문이 주장하는 “discrepancy는 source support 밖의 샘플을 드러내고, generator는 그 샘플을 support 안으로 끌어들인다”는 메커니즘이 실제 decision boundary 수준에서 어떻게 작동하는지 보여준다.

### Digit 및 traffic sign adaptation

논문은 MNIST, SVHN, USPS, SYN SIGNS, GTSRB를 사용해 여러 adaptation 시나리오를 테스트한다. 주요 결과는 Table 1에 정리되어 있다.

SVHN $\rightarrow$ MNIST에서 source only는 67.1%인데, 제안 방법은 $n=4$일 때 96.2%를 기록한다. 이는 DANN의 71.1%, DSN의 82.7%, ADDA의 76.0%보다 크게 높다. 논문이 주장하는 강점이 가장 극적으로 드러나는 결과 중 하나다. 이 setting은 도메인 차이가 큰 대표 사례인데, 단순 분포 정렬보다 classifier discrepancy 기반 정렬이 더 유리하다는 해석이 가능하다.

SYN SIGNS $\rightarrow$ GTSRB에서는 source only 85.1%, DANN 88.7%, DSN 93.1%, 제안 방법은 최대 94.4%를 기록한다. 여기서는 개선 폭이 SVHN $\rightarrow$ MNIST만큼 극적이지는 않지만, 여전히 기존 방법들보다 우수하다.

MNIST $\rightarrow$ USPS, MNIST $\rightarrow$ USPS* 및 USPS $\rightarrow$ MNIST에서도 제안 방법은 매우 강한 성능을 보인다. 예를 들어 USPS $\rightarrow$ MNIST에서는 $n=4$에서 94.1%를 기록해 ADDA 90.1%, CoGAN 89.1%, DRCN 73.7%보다 높다. 전반적으로 $n$을 늘릴수록 성능이 좋아지는 경향도 관찰된다.

저자들은 discrepancy loss가 감소할수록 accuracy가 상승하는 관계를 Figure 5로 제시한다. 이는 단순히 수치가 좋다는 것뿐 아니라, 제안한 학습 신호가 실제 target 정확도 향상과 상관이 있음을 보여주는 정성적 증거다.

### VisDA object classification

VisDA는 synthetic-to-real object adaptation의 대규모 벤치마크다. 여기서 논문은 ResNet101 backbone을 사용해 12개 클래스 분류를 수행한다. Table 2를 보면 source only의 mean accuracy는 52.4, MMD는 61.1, DANN은 57.4인데, 제안 방법은 $n=4$에서 71.9를 달성한다. 이는 평균 기준으로 상당한 폭의 향상이다.

클래스별로 봐도 강점이 뚜렷하다. 예를 들어 knife 클래스는 source only 17.9, MMD 42.9, DANN 29.5인데, 제안 방법은 최대 79.6에 이른다. person은 31.2에서 76.9까지, truck은 8.5에서 29.7 또는 25.8까지 향상된다. 모든 클래스에서 균일하게 좋아지는 것은 아니지만, 전반적으로 source only나 MMD/DANN 대비 훨씬 안정적인 상승을 보인다.

논문은 특히 MMD와 DANN이 일부 클래스에서는 source only보다 더 나쁜 결과를 낸다고 지적한다. 반면 제안 방법은 모든 클래스에서 source only보다 낫다고 주장한다. 이 점은 실제 응용에서 중요하다. 평균 성능도 중요하지만, adaptation이 특정 클래스 성능을 심각하게 망가뜨리지 않는지도 중요한데, 이 논문은 그 안정성을 장점으로 내세운다.

### Semantic segmentation

이 논문이 강한 인상을 주는 이유 중 하나는 분류를 넘어 semantic segmentation에도 방법을 적용했다는 점이다. source는 GTA5 또는 Synthia 같은 synthetic urban scene, target은 Cityscapes이다. 평가 지표는 mIoU이며, backbone으로 VGG-16 기반 FCN-8s와 DRN-105를 사용했다.

#### GTA5 $\rightarrow$ Cityscapes

Table 3에서 VGG-16 기반 source only는 mIoU 24.9, DANN은 별도로 보고되지 않고, 제안 방법은 최대 28.8이다. VGG-16에서는 상승 폭이 크지 않지만 분명한 개선이 보인다.

DRN-105에서는 개선이 더 크다. source only 22.2, DANN 32.8에 비해 제안 방법은 $k=2$에서 39.7, $k=3$에서 38.9, $k=4$에서 38.1을 기록한다. 특히 road, sidewalk, building, car 같은 주요 class에서 상당한 향상이 나타난다. 예를 들어 car는 source only 53.6, DANN 53.3인데 제안 방법은 최대 84.1까지 오른다. person도 37.0에서 58.5까지 향상된다.

#### Synthia $\rightarrow$ Cityscapes

Table 4에서도 DRN-105 기준 source only는 23.4, DANN은 32.5, 제안 방법은 최대 37.3을 기록한다. synthetic-to-real segmentation에서도 classifier discrepancy 기반 adaptation이 강하게 작동함을 보여준다.

이 실험은 논문의 중요성을 크게 높여준다. 많은 domain adaptation 논문이 작은 분류 벤치마크에 머무르는데, 이 논문은 dense prediction task인 segmentation까지 확장했다. 이는 제안 아이디어가 단순한 이미지-level classification trick이 아니라 feature alignment 원리로서 더 일반적일 수 있음을 시사한다.

### Supplementary의 GRL 결과

보충 자료는 gradient reversal layer를 적용한 one-step 학습 버전도 제시한다. GTA5 $\rightarrow$ Cityscapes에서 DRN-105 + GRL은 mIoU 39.9를 기록해, 본문 3-step 학습의 최고치 39.7과 비슷하거나 약간 높다. Synthia $\rightarrow$ Cityscapes에서도 완전히 최고는 아니지만 비슷한 수준을 유지한다. 이는 방법의 핵심 아이디어가 특정 학습 절차에만 의존하는 것이 아니라, 더 실용적인 adversarial training 형태로도 유지될 수 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation 문제를 decision boundary 관점에서 다시 정식화했다는 점이다. 기존 방법들이 “source와 target을 구분 못 하게 만들면 된다”는 식으로 접근했다면, 이 논문은 “target이 source의 분류 가능한 support 안에 들어와야 한다”는 더 직접적이고 task-aware한 목표를 세운다. 이 차이는 단순한 구현상의 차이가 아니라, adaptation의 목적 함수 자체를 바꾸는 수준의 기여다.

또 다른 강점은 방법이 직관적이면서도 실험적으로 강력하다는 점이다. generator 하나와 classifier 두 개라는 비교적 간단한 구성으로, digits부터 VisDA, semantic segmentation까지 일관된 개선을 보인다. 특히 SVHN $\rightarrow$ MNIST와 VisDA, Cityscapes 계열 실험은 이 방법이 domain gap이 큰 환경에서도 강하게 작동할 수 있음을 보여준다.

이론과의 연결도 장점이다. 엄밀한 새 이론을 전개한 것은 아니지만, $\mathcal{H}\Delta\mathcal{H}$-distance 관점에서 classifier discrepancy를 해석한 것은 방법의 설계 이유를 잘 설명해 준다. 단지 heuristic이 아니라 기존 domain adaptation theory와 논리적으로 닿아 있다는 점이 설득력을 높인다.

그러나 한계도 분명하다. 첫째, 방법은 source에서 정확하고 서로 다른 두 classifier를 안정적으로 유지할 수 있다는 가정에 기대고 있다. 실제로 classifier 둘이 충분히 다르지 않으면 discrepancy 신호가 약해질 수 있고, 반대로 지나치게 다르면 source decision boundary 자체가 불안정해질 수 있다. 논문은 서로 다른 initialization을 통해 classifier 차이를 확보한다고 하지만, 이 다양성이 얼마나 충분한지에 대한 체계적 분석은 많지 않다.

둘째, hyper-parameter $n$에 대한 의존성이 있다. generator를 몇 번 반복 업데이트할지에 따라 성능이 달라지며, 논문은 여러 값에서 성능 향상을 보이지만 최적 값 선택은 데이터셋에 따라 달라진다. supplementary에서 GRL 버전으로 이를 완화하려 하지만, 본문 제안의 기본형은 여전히 조정이 필요한 구조다.

셋째, 이 논문은 target가 truly multimodal하거나 source support가 target 전체를 충분히 감싸지 못하는 경우에 어떤 일이 벌어지는지를 깊게 다루지 않는다. 제안 방법은 target를 source support 안으로 끌어오려는 방향인데, 만약 target에 source에 없는 새로운 mode가 존재한다면 이 전략은 지나치게 source 중심적으로 작동할 수 있다. 이런 상황에서 adaptation이 오히려 representation을 왜곡할 가능성은 논문에서 논의되지 않는다.

넷째, 이론 파트는 motivating insight로는 충분하지만, 제안한 discrepancy loss와 실제 target risk 감소 사이를 엄밀하게 연결하는 새 정리나 일반화 bound를 제공하지는 않는다. 따라서 이론적 설명은 강한 보장이라기보다 타당한 해석으로 읽는 것이 맞다.

비판적으로 보면, 이 논문은 “decision boundary를 고려한 adaptation”이라는 중요한 방향을 열었지만, classifier disagreement가 항상 좋은 uncertainty proxy인지, 그리고 source support 밖에 있는 모든 target 샘플을 동일하게 다뤄도 되는지는 이후 연구에서 더 정교하게 다뤄질 필요가 있다. 그럼에도 당시 맥락에서는 매우 영향력 있는 문제 제기이자 실용적 해법이었다고 평가할 수 있다.

## 6. 결론

이 논문은 unsupervised domain adaptation에서 domain classifier 기반 전통적 분포 정렬의 한계를 지적하고, 두 task-specific classifier의 discrepancy를 활용하는 새로운 adversarial adaptation 프레임워크를 제안한다. 핵심은 target 샘플 중 source support 바깥에 있는 것들을 classifier disagreement로 탐지하고, generator가 그 불일치를 줄이도록 학습함으로써 target feature를 더 분별력 있는 위치로 이동시키는 것이다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, decision boundary를 고려하는 classifier discrepancy 기반 adaptation을 제안했다. 둘째, 이 아이디어가 $\mathcal{H}\Delta\mathcal{H}$-distance와 연결됨을 보여주었다. 셋째, digits, traffic signs, VisDA, semantic segmentation에 이르기까지 다양한 벤치마크에서 강력한 성능을 입증했다.

이 연구는 이후 domain adaptation 분야에서 decision boundary, classifier disagreement, uncertainty-aware alignment를 중요한 주제로 부상시키는 데 큰 역할을 했다. 실제 응용 측면에서도 synthetic-to-real recognition, 도시 장면 segmentation, 라벨이 부족한 산업 비전 문제 등에서 활용 가능성이 크다. 향후 연구는 이 아이디어를 더 안정적인 학습 절차, 더 강한 이론, partial/open-set adaptation 같은 더 어려운 설정으로 확장하는 방향으로 이어질 수 있다.
