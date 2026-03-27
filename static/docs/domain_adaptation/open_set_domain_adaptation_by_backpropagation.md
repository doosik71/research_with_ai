# Open Set Domain Adaptation by Backpropagation

* **저자**: Kuniaki Saito, Shohei Yamamoto, Yoshitaka Ushiku, Tatsuya Harada
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1804.10427](https://arxiv.org/abs/1804.10427)

## 1. 논문 개요

이 논문은 **Open Set Domain Adaptation (OSDA)** 문제를 다룬다. 일반적인 unsupervised domain adaptation은 source와 target이 같은 클래스 집합을 공유한다고 가정하는 **closed-set** 환경을 전제로 한다. 그러나 실제 환경에서는 unlabeled target 데이터 안에 source에는 없는 새로운 클래스가 섞여 있을 수 있다. 이 논문은 그러한 클래스들을 **unknown class**라고 부르고, source에 없는 unknown target 샘플을 잘 걸러내면서도 source와 공유되는 **known class**는 올바르게 적응시키는 방법을 제안한다.

이 연구가 다루는 핵심 문제는 두 가지이다. 첫째, target 데이터에는 라벨이 없기 때문에 어떤 샘플이 unknown인지 직접 알 수 없다. 둘째, domain adaptation에서는 보통 source와 target의 feature distribution을 맞추는데, OSDA에서는 unknown target까지 source 쪽으로 끌어당기면 오히려 잘못된 정렬이 일어난다. 즉, 모든 target 샘플을 source에 맞추는 기존 distribution matching 방식은 open set 상황에서 구조적으로 부적절하다.

기존 OSDA 연구 가운데 일부는 unknown source 샘플도 함께 사용하는 설정을 가정했다. 하지만 이 논문은 그보다 더 어려운 설정을 제시한다. 즉, **source에는 known class만 존재하고, target은 unlabeled이며 known과 unknown이 섞여 있지만 unknown source 샘플은 전혀 주어지지 않는다.** 이 가정은 실제 적용 측면에서 더 자연스럽고, 따라서 문제의 실용성이 높다.

논문의 목표는 단순히 unknown을 threshold로 걸러내는 것이 아니라, 학습 과정에서 feature generator가 target 샘플을 **known source 쪽으로 정렬할지**, 아니면 **unknown으로 밀어낼지**를 선택할 수 있게 만드는 것이다. 저자들은 이를 adversarial learning으로 구현하며, 실험을 통해 digits, Office, VisDA 데이터셋에서 기존 baseline보다 좋은 성능을 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **classifier는 target 샘플에 대해 unknown class 확률이 특정 중간값 $t$가 되도록 학습하고, feature generator는 그 값을 $t$에서 멀어지게 만들도록 적대적으로 학습한다.** 이 과정에서 generator는 target 샘플마다 두 가지 선택지를 갖게 된다. 하나는 해당 샘플을 known source feature 쪽으로 옮겨 unknown 확률을 낮추는 것이고, 다른 하나는 오히려 unknown 쪽으로 밀어 unknown 확률을 높이는 것이다. 저자들은 이 두 갈래 선택이 unknown target과 known target을 자연스럽게 분리하게 만든다고 주장한다.

이 점이 기존 접근과 가장 크게 다르다. 기존 MMD나 domain classifier 기반 adaptation은 기본적으로 source와 target의 분포를 맞추려 한다. 그런데 OSDA에서는 unknown target이 source에 대응되는 클래스가 없으므로, 이들을 source 쪽으로 정렬하는 순간 feature space가 오염된다. 논문은 이 문제를 피하기 위해 “모든 target을 맞춘다”가 아니라, “target 일부는 맞추고 일부는 거부한다”는 방향으로 문제를 다시 설계했다.

또 하나의 중요한 차별점은 **unknown source 샘플이 필요 없다는 점**이다. 기존 open set recognition은 unknown 샘플을 직접 보거나, threshold 기반으로 사후 처리하는 경우가 많았다. 반면 이 논문은 unlabeled target 안에 unknown이 섞여 있다는 사실만 이용하고, 어떤 샘플이 unknown인지 명시적 supervision 없이 학습을 진행한다. 즉, unknown에 대한 직접 레이블 없이도 feature separation을 유도하려는 점이 핵심이다.

저자들의 직관은 다음과 같이 요약할 수 있다. classifier가 target을 unknown으로 약하게 밀어내는 경계면을 만들고, generator는 그 경계면으로부터 각 target 샘플을 멀리 보내려고 한다. 그러면 source와 닮은 known target은 source 쪽으로 정렬되고, source와 잘 맞지 않는 target은 unknown 영역으로 밀려난다. 이 구조는 단순한 domain confusion이 아니라, **domain alignment와 unknown rejection을 동시에 수행하는 적대적 학습**이다.

## 3. 상세 방법 설명

논문은 두 개의 네트워크를 사용한다. 첫째는 입력 이미지 $\mathbf{x}_s$ 또는 $\mathbf{x}_t$를 feature로 바꾸는 **feature generator $G$**이고, 둘째는 그 feature를 받아 총 $K+1$개의 클래스 확률을 출력하는 **classifier $C$**이다. 여기서 $K$는 source의 known class 수이고, 마지막 $K+1$번째 차원은 unknown class에 해당한다. 따라서 classifier는 일반적인 closed-set 분류기보다 하나 더 많은 출력을 갖는다.

입력 $\mathbf{x}$에 대해 classifier는 logit ${l_1, \dots, l_{K+1}}$를 출력하고, softmax를 통해 각 클래스 확률을 만든다. 논문에서 사용하는 확률 표기는 다음과 같다.

$$
p(y=j|\mathbf{x}) = \frac{\exp(l_j)}{\sum_{k=1}^{K+1}\exp(l_k)}
$$

여기서 $j=1,\dots,K$는 known class이고, $j=K+1$은 unknown class이다. 이 정의 자체는 표준 softmax 분류와 같다. 중요한 것은 이 $K+1$번째 출력에 어떤 학습 신호를 주느냐이다.

먼저 source 샘플 $\mathbf{x}_s$와 레이블 $y_s$에 대해서는 일반적인 supervised classification을 수행한다. 즉, source는 known class만 포함하므로 cross-entropy loss를 그대로 사용한다.

$$
L_s(\mathbf{x}_s, y_s) = -\log p(y=y_s|\mathbf{x}_s)
$$

또는 논문 표현대로,

$$
p(y=y_s|\mathbf{x}_s) = (C \circ G(\mathbf{x}_s))_{y_s}
$$

이 항은 source에서 분류 성능을 유지하게 해 주며, known class decision boundary를 학습하는 기본 역할을 한다.

이제 핵심은 target loss이다. target은 unlabeled이므로 어떤 샘플이 known인지 unknown인지 모른다. 논문은 target 샘플에 대해 classifier가 unknown 확률 $p(y=K+1|\mathbf{x}_t)$를 어떤 특정 값 $t$에 맞추도록 학습시킨다. 이를 위해 binary cross-entropy 형태의 adversarial loss를 도입한다.

$$
L_{adv}(\mathbf{x}_t) = - t \log p(y=K+1|\mathbf{x}_t) - (1-t)\log(1-p(y=K+1|\mathbf{x}_t))
$$

여기서 $0 < t < 1$이며, 논문의 실험에서는 주로 $t=0.5$를 사용한다. 이 식의 의미를 쉽게 설명하면, classifier는 target 샘플을 확실한 known도 아니고 확실한 unknown도 아닌 **애매한 경계 상태**에 두려고 한다. $t=0.5$라면 unknown 확률을 0.5로 만들도록 유도하는 셈이다. 즉, classifier가 target 전반에 대해 일종의 **pseudo decision boundary**를 만든다.

이 상태에서 generator는 classifier를 속이도록 학습된다. 최종 목적함수는 classifier와 generator에 대해 서로 다르게 정의된다.

Classifier에 대해서는

$$
\min_C \; L_s(\mathbf{x}_s, y_s) + L_{adv}(\mathbf{x}_t)
$$

Generator에 대해서는

$$
\min_G \; L_s(\mathbf{x}_s, y_s) - L_{adv}(\mathbf{x}_t)
$$

를 최적화한다.

여기서 중요한 구조는 generator 쪽에 $-L_{adv}$가 붙는다는 점이다. classifier는 target의 unknown 확률을 $t$에 맞추려 하고, generator는 반대로 그 값을 $t$에서 벗어나게 하려고 한다. 이때 generator가 할 수 있는 행동은 두 가지다. 어떤 target 샘플에 대해서는 unknown 확률을 낮춰 known source feature와 더 잘 맞도록 만들 수 있고, 다른 샘플에 대해서는 unknown 확률을 높여 아예 unknown으로 거부되도록 만들 수 있다. 저자들이 강조하는 “two options”가 바로 이것이다.

왜 굳이 $t=1$이 아니라 $t \in (0,1)$를 쓰는가도 논문의 핵심 논리이다. 만약 classifier가 target에 대해 항상 $p(y=K+1|\mathbf{x}_t)=1$이 되도록 학습된다면, generator는 그것을 속이기 위해 unknown 확률을 무조건 낮추는 방향으로 움직일 가능성이 크다. 그러면 결국 모든 target을 source 쪽으로 맞추는 기존 distribution matching과 비슷한 동작이 된다. 반면 $t=0.5$와 같은 중간값을 쓰면 generator는 각 샘플별로 **unknown 쪽으로 보내는 것이 더 쉬운지**, 아니면 **known 쪽으로 끌어오는 것이 더 쉬운지**를 선택할 수 있다. 그래서 unknown/known 분리가 가능해진다.

학습 구현은 gradient reversal layer를 사용한다. 이는 Domain-Adversarial Training 계열에서 자주 쓰는 방식으로, forward에서는 아무 일도 하지 않지만 backward에서는 gradient 부호를 뒤집는다. 그래서 classifier와 generator를 한 번의 backpropagation 흐름 안에서 서로 반대 목적을 갖도록 업데이트할 수 있다. 논문은 minibatch마다 source와 target을 샘플링한 뒤 $L_s$와 $L_{adv}$를 계산하고, gradient reversal을 통해 $G$와 $C$를 동시에 업데이트하는 절차를 제시한다.

논문은 기존 open set recognition과의 차이도 세 가지로 정리한다. 첫째, 기존 방법은 unknown 샘플을 학습 중에 직접 쓰지 못하는 경우가 많지만, 이 방법은 unlabeled target 안에 unknown이 포함되어 있으므로 feature extractor가 간접적으로 unknown rejection을 학습할 수 있다. 둘째, 기존 OSVM류는 고정 threshold로 unknown을 판단하는 반면, 이 방법은 샘플마다 다른 representation과 classifier 출력을 통해 사실상 **샘플별로 다른 경계**를 형성한다. 셋째, feature extractor가 pseudo decision boundary의 방향 정보를 받기 때문에 각 샘플이 그 경계에서 얼마나 떨어져야 하는지까지 반영해 representation을 학습할 수 있다.

전체적으로 보면, 이 방법은 closed-set domain adaptation의 domain confusion 아이디어를 open set 상황에 맞게 바꾸면서, **“정렬(alignment)”과 “거부(rejection)”를 같은 feature space 안에서 동시에 배우도록 만든 설계**라고 요약할 수 있다.

## 4. 실험 및 결과

논문은 Office, VisDA, Digits 데이터셋에서 실험을 수행했고, 추가로 semi-supervised open set recognition 실험도 제시한다. backbone으로는 Office와 VisDA에서 AlexNet과 VGGNet pretrained on ImageNet을 사용했고, 해당 실험에서는 backbone 자체는 업데이트하지 않았다고 적혀 있다. FC8 뒤에 hidden unit 100개의 fully connected layer를 두었고, Batch Normalization과 Leaky-ReLU를 사용했다. 최적화는 momentum SGD, learning rate $1.0 \times 10^{-3}$, momentum 0.9이다. Digits에서는 별도 CNN 구조와 Adam optimizer를 사용했고 learning rate는 $2.0 \times 10^{-5}$이다.

비교 baseline은 세 가지이다. 첫째는 **OSVM**으로, source only로 학습한 CNN feature 위에 open set SVM을 적용한다. 둘째는 **MMD + OSVM**으로, domain alignment를 MMD로 수행한 뒤 feature에 OSVM을 적용한다. 셋째는 **BP + OSVM**으로, domain classifier 기반 adaptation by backpropagation을 수행한 뒤 OSVM을 쓴다. 즉, baseline 설계는 “adaptation은 기존 방식으로 하고, unknown rejection은 OSVM에 맡긴다”는 구조다. 이와 비교해 제안법은 representation learning 단계 자체에서 unknown separation을 학습한다는 점이 다르다.

### Office 데이터셋: 10 shared class + 1 unknown, 총 11-class 분류

첫 번째 Office 실험은 기존 OSDA 프로토콜을 따른다. 전체 31개 클래스 중 10개를 shared known class로 쓰고, 나머지 일부를 unknown으로 사용한다. 평가 지표는 class-average accuracy이며, **OS**는 unknown을 포함한 전체 클래스 평균 정확도, **OS*(10)**은 shared known class들만 대상으로 한 평균 정확도이다.

AlexNet 기반 결과에서 제안법은 unknown source 샘플이 없는 조건에서도 평균적으로 가장 좋은 수준의 성능을 보인다. 예를 들어 평균 성능은 OS 80.4, OS*80.2로 보고되며, OSVM은 40.6/37.1, MMD+OSVM은 34.5/28.5, BP+OSVM은 29.5/22.7에 그친다. 비교를 위해 제시된 기존 ATI-$\lambda$ + OSVM은 75.0의 평균 OS를 보이므로, unknown source가 없는 조건에서 제안법이 더 강하다. VGGNet에서는 격차가 더 크며, 제안법은 평균 OS 88.0, OS* 88.5를 기록한다. 같은 조건의 OSVM, MMD+OSVM, BP+OSVM은 모두 평균 OS가 66점대에 머문다.

이 결과가 중요한 이유는 단순히 평균이 높아서가 아니다. 논문은 OSVM 계열이 known target 샘플을 unknown으로 과하게 보내는 경향이 있다고 해석한다. 실제로 OS와 OS*를 비교하면 baseline은 known class 정확도도 불안정하고, domain alignment를 추가해도 성능이 안정적으로 좋아지지 않는다. 이는 unknown target이 섞인 상태에서 전체 target distribution을 source 쪽으로 맞추는 전략이 유효하지 않다는 저자들의 문제 제기와 맞아떨어진다.

t-SNE 시각화도 이 해석을 뒷받침한다. source only나 MMD, BP에서는 target unknown이 source/known 영역에 섞여 들어가는 경향이 보이지만, 제안법에서는 unknown target이 known target 및 source known과 분리되는 패턴이 관찰된다고 서술한다. 논문의 메시지는 단순하다. **기존 distribution matching은 open set에서 unknown까지 정렬해 버리고, 제안법은 unknown을 떼어내는 feature를 배운다.**

### unknown 비율과 $t$ 값 분석

논문은 DSLR $\rightarrow$ Amazon 적응에서 unknown target 비율을 바꾸는 실험도 수행한다. unknown 비율이 늘어나면 성능은 감소하지만, 제안법은 전반적으로 안정적으로 동작한다고 보고한다. 이는 적어도 제안법이 특정 unknown 비율에만 과도하게 맞춰진 방식은 아니라는 뜻이다.

더 중요한 분석은 $t$ 값에 대한 ablation이다. 논문은 $t$가 1에 가까워질수록 generator가 전체 target을 source 쪽으로 맞추는 기존 distribution matching과 비슷하게 행동한다고 설명한다. 실제 결과에서도 $t$가 증가할수록 OS와 OS*가 감소한다. 저자 해석에 따르면, 이는 unknown과 known을 구별하는 representation을 배우지 못하기 때문이다. 반대로 $t=0.5$ 근처에서는 unknown rejection과 known alignment의 균형이 가장 잘 맞는다. 이는 이 논문의 가장 핵심적인 empirical evidence다. 즉, 방법의 효과가 단순 adversarial training 때문이 아니라, **중간 목표값 $t$를 둔 설계**에서 온다는 점을 보여준다.

또한 Webcam $\rightarrow$ DSLR 적응에서 unknown class 확률의 histogram을 보면, 학습 초반에는 known/unknown 모두 낮은 확률 영역에 몰려 있지만, 500 epoch 이후에는 unknown 샘플은 높은 unknown 확률, known 샘플은 낮은 unknown 확률 쪽으로 분리된다. 이는 제안법이 실제로 feature separation을 일으킨다는 정성적 증거로 사용된다.

### Office 데이터셋: 20 shared class + 1 unknown, 총 21-class 분류

두 번째 Office 실험은 known class 수를 20개로 늘린 더 어려운 설정이다. 이 경우 VGGNet 기반 결과만 제시된다. 평균 성능은 제안법이 OS 74.7, OS* 74.6, ALL 76.1로 가장 높다. 비교 baseline은 OSVM 61.8/61.7/61.5, MMD+OSVM 58.9/58.2/62.3, BP+OSVM 59.8/59.1/63.2 수준이다.

특히 A-W 적응에서는 일부 OS, OS* 수치만 보면 다른 방법이 더 나아 보이는 지점이 있지만, 논문은 **ALL** 지표를 함께 봐야 한다고 말한다. ALL은 클래스 평균이 아니라 전체 샘플에 대한 정확도이며, 제안법이 이 지표에서 우수하다. 저자 해석은 기존 방법들이 target 샘플을 known class 중 하나로 과하게 분류하는 경향이 있다는 것이다. 즉, class-average만 보면 착시가 생길 수 있고, 실제 sample-level behavior에서는 제안법이 더 균형 잡힌 예측을 한다는 주장이다.

### VisDA 데이터셋

VisDA에서는 synthetic-to-real adaptation을 다룬다. 12개 카테고리 중 vehicle 계열 6개를 known, 나머지 6개를 unknown으로 둔다. 평가 지표는 클래스별 정확도와 전체 평균이다. AlexNet과 VGGNet 두 설정 모두에서 제안법이 평균적으로 가장 좋다.

AlexNet에서는 평균 정확도 Avg 58.5, known 평균 Avg known 54.8을 기록하며, baseline 중 최고인 BP+OSVM의 Avg 44.7, Avg known 45.1보다 좋다. VGGNet에서는 제안법이 Avg 65.2, Avg known 61.1을 기록하고, BP+OSVM은 55.5/57.8, MMD+OSVM은 54.4/56.0, OSVM은 52.5/54.9이다. 특히 unknown class 정확도는 VGGNet 기준 89.7로 매우 높다.

논문은 이 데이터셋에서 unknown이 vehicle이 아닌 다른 객체들이라 외형 차이가 커서 unknown rejection이 상대적으로 쉬웠을 수 있다고 해석한다. 반면 known 클래스인 car의 정확도가 상대적으로 낮은 등의 현상도 보인다. 정성 예시를 보면, 객체가 가려져 있거나 여러 객체가 함께 있는 이미지에서는 known이 unknown으로 오분류되기도 한다. 또 person이나 horse가 motorcycle로 잘못 분류되는 경우도 있는데, 저자들은 motorcycle 이미지에 사람과 배경이 함께 자주 등장해 appearance가 비슷하게 작용했을 가능성을 언급한다. 이런 분석은 모델이 단순 class semantics보다 visual co-occurrence에 영향을 받을 수 있음을 보여준다.

### Digits 데이터셋

Digits 실험에서는 SVHN, USPS, MNIST 사이의 adaptation을 다룬다. 숫자 0~4를 known, 5~9를 unknown으로 설정한다. 평가 지표는 OS, OS*, ALL, UNK이다. 여기서 UNK는 unknown 클래스 정확도로 보인다.

세 가지 적응 시나리오의 평균 결과를 보면, 제안법은 OS 82.4, OS* 81.7, ALL 84.5, UNK 85.9를 기록한다. OSVM은 59.1/57.7/61.7/65.7, MMD+OSVM은 68.0/68.8/66.3/58.4, BP+OSVM은 60.4/69.4/44.5/15.3이다. 특히 BP+OSVM은 UNK가 매우 낮아, distribution matching이 unknown target 탐지에 심각한 악영향을 줄 수 있음을 보여준다.

세부적으로 보면 USPS-MNIST와 MNIST-USPS에서는 제안법이 매우 강한 성능을 보인다. 반면 SVHN-MNIST는 다른 두 시나리오보다 어렵다. 논문은 이를 domain gap이 훨씬 크기 때문이라고 설명한다. 즉, open set 문제뿐 아니라 기본 domain shift 자체가 클 때는 여전히 어려움이 있다.

feature visualization에서도 제안법은 unknown target(5~9)을 source/known에서 분리하고 known target은 source known과 정렬하는 모습을 보였다고 서술한다. 반면 BP는 대부분의 target feature를 source 쪽으로 맞추려 해 unknown이 섞여 버린다. 이 시각화는 논문의 핵심 주장과 다시 연결된다.

### Semi-supervised open set recognition 응용

마지막으로 논문은 domain shift가 없는 open set recognition에도 방법을 적용한다. 이 설정에서는 labeled known sample과 unlabeled sample이 같은 domain에 존재한다. 숫자 0~4를 known, 5~9를 unknown으로 두고, known은 클래스당 1000개만 labeled로 사용한다.

이 실험에서는 OSVM도 강한데, 그 이유는 domain shift가 없어서 source only feature 자체가 잘 작동하기 때문이다. 그럼에도 제안법은 MNIST 쪽에서는 OSVM보다 좋고, SVHN에서는 classifier 학습이 불안정한 모습을 보인다. 흥미롭게도 **Ours + OSVM**을 쓰면 평균적으로 가장 좋은 결과가 나온다. 논문은 이를 두고, 제안법이 unknown rejection에 유리한 representation은 잘 학습했지만, 같은-domain semi-supervised setting에서는 classifier 자체는 충분히 좋게 학습되지 않았을 수 있다고 해석한다. 이 부분은 제안법의 representation learning 효과와 classifier calibration 문제가 분리될 수 있음을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 매우 일관된다는 점이다. 저자들은 OSDA에서 왜 기존 domain alignment가 실패하는지를 분명히 지적하고, 그 해결책으로 “일부 target은 맞추고 일부는 거부해야 한다”는 구조를 제안했다. 그리고 이를 구현하는 수단으로 classifier에는 $t$로 약한 unknown 경계를 만들게 하고, generator에는 그 경계에서 멀어지게 하는 adversarial objective를 부여했다. 문제 진단, 방법 설계, 실험 결과가 한 방향을 보고 있다.

둘째 강점은 **unknown source 샘플이 필요 없다는 실용성**이다. 기존 OSDA 설정 중 일부는 unknown source를 사용했는데, 현실에서는 어떤 unknown을 source에 충분히 수집하는 것 자체가 어렵다. 이 논문은 unlabeled target만으로 unknown rejection을 유도하므로 실전 적용 가능성이 높다.

셋째 강점은 실험 범위가 비교적 넓다는 점이다. Office, VisDA, Digits처럼 domain gap과 클래스 구조가 다른 여러 데이터셋에서 일관된 경향을 보여 주며, 단순 accuracy 표뿐 아니라 t-SNE, unknown ratio, $t$ 값 변화, histogram까지 제시해 방법의 작동 원리를 여러 각도에서 뒷받침한다.

다만 한계도 분명하다. 첫째, 이 방법은 결국 target 안에 unknown이 어느 정도 섞여 있고, 그 unknown이 known source와 representation 상에서 분리 가능하다는 전제를 간접적으로 깔고 있다. unknown이 known과 시각적으로 매우 유사하거나, target의 known/unknown 분포가 극단적으로 복잡할 경우 성능이 얼마나 유지되는지는 이 논문만으로는 충분히 알 수 없다.

둘째, $t$의 선택이 성능에 중요한 영향을 준다. 논문은 $t=0.5$를 주로 사용하며, $t$ 변화 실험도 제시하지만, 왜 0.5가 이론적으로 최적인지에 대한 강한 증명은 없다. 즉, 이 방법은 직관적으로 설득력 있고 실험적으로도 잘 작동하지만, $t$의 의미와 최적 조건에 대한 이론적 분석은 제한적이다.

셋째, baseline 비교가 약간 제한적이다. 비교 대상의 상당수는 “adaptation 후 OSVM” 구조이고, 그 결과는 OSVM threshold 민감도에 영향을 받을 수 있다. 논문이 제시한 시점에서는 합리적인 비교였지만, 이후 더 강한 open set 혹은 partial/open-world adaptation 기법들과 비교하면 위치가 달라질 수 있다. 물론 이것은 후속 연구 관점의 비판이지, 논문 내부 증거를 깨는 문제는 아니다.

넷째, 실험 설명 중 일부는 다소 압축적이다. 예를 들어 classifier calibration 문제나 semi-supervised setting에서 Ours와 Ours+OSVM의 차이가 왜 크게 나는지에 대한 분석은 제한적이다. 또한 backbone을 업데이트하지 않은 Office/VisDA 설정이 최선인지, 다른 backbone이나 end-to-end fine-tuning에서 어떤 양상이 나올지까지는 다루지 않는다.

종합하면, 이 논문은 문제 설정의 현실성과 방법의 단순성, 그리고 empirical effectiveness 측면에서 강점을 가진다. 반면 unknown의 성질이 더 복잡한 경우, 혹은 하이퍼파라미터 $t$에 대한 이론적 근거 부족은 남는 과제라고 볼 수 있다.

## 6. 결론

이 논문은 open set domain adaptation에서 **unknown source 샘플 없이도** unknown target을 거부하고 known target은 source known과 정렬할 수 있는 adversarial learning 방법을 제안했다. 방법의 핵심은 classifier가 target의 unknown 확률을 $t$로 맞추는 경계를 만들고, generator가 그 경계에서 target 샘플을 멀어지게 학습한다는 점이다. 이 구조 덕분에 generator는 각 target 샘플을 known 쪽으로 정렬할지 unknown으로 거부할지 선택할 수 있게 되고, 결과적으로 unknown/known separation이 가능해진다.

실험적으로도 논문은 Office, VisDA, Digits에서 기존 OSVM, MMD+OSVM, BP+OSVM보다 전반적으로 우수한 성능을 보여 주었다. 특히 기존 distribution matching 방식이 OSDA에서 왜 실패하는지를 정량 결과와 feature visualization으로 설득력 있게 보여 준 점이 인상적이다.

이 연구의 의의는 단순히 한 가지 성능 향상 기법을 제안한 데 있지 않다. 더 중요한 점은 domain adaptation을 open set 상황으로 확장할 때, “domain invariance만 추구해서는 안 된다”는 방향 전환을 명확히 제시했다는 데 있다. 이후 open set, partial set, universal domain adaptation으로 이어지는 연구 흐름에서 이 논문은 중요한 초기 아이디어로 볼 수 있다. 실제 적용 측면에서도, 자동 수집된 unlabeled target 데이터 안에 새로운 클래스가 섞여 있는 상황은 매우 흔하므로, 이 논문이 제시한 관점은 여전히 의미가 크다.
