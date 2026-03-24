# Deep Domain Confusion: Maximizing for Domain Invariance

* **저자**: Eric Tzeng, Judy Hoffman, Ning Zhang, Kate Saenko, Trevor Darrell
* **발표연도**: 2014
* **arXiv**: [https://arxiv.org/abs/1412.3474](https://arxiv.org/abs/1412.3474)

## 1. 논문 개요

이 논문은 visual domain adaptation 문제를 다룬다. 즉, 어떤 도메인(source domain)에서 충분한 라벨 데이터로 학습한 분류기가 다른 도메인(target domain)로 옮겨졌을 때 성능이 떨어지는 현상을 해결하려는 연구다. 저자들은 이 문제의 핵심 원인을 dataset bias 또는 domain shift로 본다. 같은 객체 분류 문제라도 Amazon 상품 이미지, Webcam 이미지, DSLR 이미지처럼 데이터가 수집된 환경이 달라지면 입력 분포가 달라지고, 그 결과 source에서 잘 작동하던 모델이 target에서는 성능이 떨어진다.

당시까지의 많은 domain adaptation 방법은 shallow feature나 shallow model에 머무르는 경우가 많았고, deep CNN을 활용하는 경우에도 주로 pre-trained CNN feature를 그대로 가져다 쓰거나, target의 소량 라벨에 맞춰 fine-tuning하는 방식이 일반적이었다. 하지만 target domain의 라벨 수가 매우 적거나 아예 없는 경우에는 단순 fine-tuning이 어렵고, 오히려 source 분포에 과적합되기 쉽다.

이 논문은 이런 한계를 해결하기 위해, 분류 성능을 유지하면서 동시에 source와 target의 feature 분포를 최대한 비슷하게 만드는 CNN 구조를 제안한다. 구체적으로는 adaptation layer를 추가하고, 그 위에서 MMD(Maximum Mean Discrepancy) 기반의 domain confusion loss를 함께 최적화한다. 목표는 “의미적으로는 잘 구분되면서도, 도메인 관점에서는 서로 구별되지 않는 표현”을 학습하는 것이다.

이 문제의 중요성은 매우 크다. 실제 응용에서는 새로운 도메인마다 대규모 라벨 데이터를 다시 모으기 어렵다. 따라서 기존에 학습된 강력한 표현을 유지하면서, 적은 라벨 또는 무라벨 환경에서도 새로운 도메인으로 잘 일반화되는 표현을 학습하는 것은 실용적으로도, 학문적으로도 중요한 과제다. 이 논문은 deep representation learning과 domain adaptation을 하나의 학습 목표로 결합했다는 점에서 의미가 크다.

## 2. 핵심 아이디어

논문의 핵심 직관은 간단하다. 좋은 분류 표현은 클래스별로 잘 분리되어야 하지만, 동시에 domain shift에는 둔감해야 한다. 즉, 같은 클래스라면 Amazon 이미지든 Webcam 이미지든 feature 공간에서 가깝게 모이도록 만들어야 한다. 저자들은 이를 “classification loss는 낮추고, domain discrepancy는 줄이는 것”으로 정식화한다.

기존 접근과의 차별점은 크게 세 가지다.

첫째, 단순히 pre-trained CNN의 어느 중간층을 feature로 쓸지 경험적으로 고르는 것이 아니라, MMD를 이용해 어느 층이 source와 target 사이의 discrepancy가 가장 작은지 측정하고, 이를 adaptation layer의 위치 선택에 사용한다. 논문에서는 fully connected 7층인 $fc7$ 뒤에 adaptation layer를 두는 것이 가장 적절하다고 판단했다.

둘째, adaptation layer의 차원 수도 임의로 정하지 않고, 여러 차원을 시도한 뒤 source와 target 사이의 MMD가 가장 작아지는 차원을 선택한다. 이 논문에서는 256차원이 최종 선택되었다.

셋째, 가장 중요한 차별점은 representation selection에만 그치지 않고, 학습 과정 자체에 domain confusion loss를 넣어 CNN 파라미터를 직접 fine-tuning한다는 점이다. 즉, 분류 목적만으로 학습한 CNN에서 feature를 가져오는 것이 아니라, 처음부터 “분류 가능하면서 domain-invariant한 표현”이 되도록 joint loss로 학습한다.

이 아이디어의 본질은 다음과 같이 볼 수 있다. source 라벨을 이용해 semantic discrimination을 확보하고, 동시에 source와 target의 feature 평균 차이를 줄여 domain mismatch를 완화한다. 그 결과, target에 라벨이 거의 없거나 없어도 source에서 학습한 classifier가 더 잘 전이될 수 있다.

## 3. 상세 방법 설명

### 전체 구조

저자들은 Krizhevsky et al.의 AlexNet 계열 CNN을 기반으로 사용한다. 이 네트워크는 5개의 convolution/pooling layer와 3개의 fully connected layer를 가지며, 마지막 fully connected layer의 차원은 ${4096, 4096, |C|}$이다. 여기에 추가로 낮은 차원의 bottleneck adaptation layer를 삽입한다.

구조적으로는 source와 target 데이터를 동시에 받는 두 개의 CNN branch처럼 보이지만, 실제로는 **가중치를 공유(shared weights)** 한다. 즉, source용 CNN과 target용 CNN이 별도로 존재하는 것이 아니라 동일한 feature extractor를 함께 학습하는 형태다.

adaptation layer 위에서 네트워크는 두 갈래의 목적을 가진다.

하나는 labeled data에 대한 classification branch다. 이 branch는 클래스 예측이 잘 되도록 분류 손실을 계산한다.

다른 하나는 source와 target 전체 데이터를 이용하는 domain confusion branch다. 이 branch는 adaptation layer에서 나온 source feature와 target feature의 분포 차이를 MMD로 계산한다.

결국 하나의 representation이 두 조건을 동시에 만족하도록 학습된다. 클래스는 잘 구분해야 하고, 도메인은 잘 구분되지 않아야 한다.

### MMD 기반 domain confusion

논문에서 domain discrepancy를 측정하기 위해 사용한 핵심 도구는 MMD다. 어떤 representation $\phi(\cdot)$가 주어졌을 때, source set $X_S$와 target set $X_T$ 사이의 empirical MMD는 다음과 같이 정의된다.

$$
\text{MMD}(X_S, X_T) = \left\lVert \frac{1}{|X_S|}\sum_{x_s \in X_S}\phi(x_s) - \frac{1}{|X_T|}\sum_{x_t \in X_T}\phi(x_t) \right\rVert
$$

이 식의 의미는 매우 직관적이다. source feature들의 평균과 target feature들의 평균이 얼마나 다른지를 재는 것이다. 두 평균이 가까우면 feature 공간에서 두 도메인이 비슷하게 보인다고 해석할 수 있다. 논문은 이를 domain confusion의 척도로 사용한다. 즉, MMD가 작을수록 더 domain-invariant한 표현이라고 본다.

### 전체 학습 목적 함수

논문이 제안하는 전체 손실은 classification loss와 MMD regularization의 합이다.

$$
\mathcal{L} = \mathcal{L}_C(X_L, y) + \lambda , \text{MMD}^2(X_S, X_T)
$$

여기서 $\mathcal{L}_C(X_L, y)$는 라벨이 있는 데이터 $X_L$과 정답 $y$에 대한 classification loss다. 논문 본문에서는 구체적인 분류 손실의 수식을 따로 적지는 않았지만, CNN 분류 문맥상 softmax 기반 classification loss로 이해하는 것이 자연스럽다. 다만 논문이 명시적으로 식을 적어주지는 않았으므로, 여기서는 “labeled data에 대한 classification loss”라고만 보는 것이 정확하다.

두 번째 항은 MMD 제곱항이다. 이는 source와 target의 feature 분포 차이를 줄이기 위한 정규화 항이다. $\lambda$는 두 목적 사이의 균형을 조절하는 하이퍼파라미터다. $\lambda$가 너무 작으면 domain confusion 효과가 거의 없고, 너무 크면 모든 샘플이 지나치게 비슷한 feature로 몰리면서 분류력이 약해질 수 있다. 논문에서는 $\lambda = 0.25$를 사용했다.

이 목적 함수의 해석은 분명하다. 첫 번째 항은 “의미 있는 semantic separation”을 만들고, 두 번째 항은 “도메인 간 불일치”를 줄인다. 따라서 최종 representation은 discriminative하면서도 domain-invariant한 특성을 갖게 된다.

### adaptation layer의 역할

adaptation layer는 낮은 차원의 bottleneck layer다. 저자들의 직관은 이렇다. 너무 고차원의 fully connected feature는 source domain의 세부적인 특성까지 많이 담고 있어서 overfitting을 유발할 수 있다. 따라서 더 낮은 차원의 적절한 bottleneck을 도입하면, source-specific nuance를 덜 담고 좀 더 일반화 가능한 feature를 만들 수 있다.

또한 domain confusion loss를 adaptation layer 위에 직접 걸기 때문에, 이 층은 단순한 차원 축소 이상의 역할을 한다. 즉, 도메인 간 차이를 줄이도록 직접 regularize되는 핵심 표현 공간이 된다.

### adaptation layer 위치 선택

이 논문은 adaptation layer를 어디에 넣을지 임의로 정하지 않는다. 먼저 pre-trained CNN의 각 fully connected layer representation에 대해 source와 target의 MMD를 계산한다. MMD가 작은 층일수록 두 도메인에 대해 더 invariant한 표현이라고 해석한다.

실험 결과, $fc7$ representation이 가장 적절한 층으로 선택되었다. 그래서 이후 모든 실험에서는 adaptation layer를 $fc7$ 뒤에 삽입했다.

이 선택 방식의 의미는 크다. 보통 깊은 층으로 갈수록 semantic separation은 좋아지지만 domain-specific bias도 남을 수 있다. 반면 너무 낮은 층은 일반적이지만 분류에 충분히 적합하지 않을 수 있다. 저자들은 MMD를 사용해 이 trade-off를 데이터 기반으로 선택했다.

### adaptation layer 차원 선택

adaptation layer의 차원 역시 grid search와 MMD를 이용해 선택한다. Amazon$\rightarrow$Webcam 작업에서 64부터 4096까지 2의 거듭제곱 단위로 다양한 차원을 실험했다. 각 차원으로 학습한 뒤 source와 target 간 MMD를 계산하고, MMD가 가장 낮은 차원을 선택했다.

논문에서는 256차원을 최종 선택했다. 흥미로운 점은 256이 반드시 테스트 정확도를 절대적으로 최대화한 값은 아니었다는 것이다. 저자들은 이 점을 솔직하게 언급한다. 다만 MMD 기반 선택이 지나치게 작은 차원이나 지나치게 큰 차원 같은 극단을 피하게 해 주며, 전체적으로 합리적인 선택을 제공한다고 해석한다. 즉, MMD는 완벽한 oracle은 아니지만 유용한 model selection criterion이라는 주장이다.

### supervised와 unsupervised adaptation

이 구조는 supervised adaptation과 unsupervised adaptation 모두에 사용할 수 있다.

supervised adaptation에서는 target domain에도 소량의 labeled data가 있으므로, classification loss를 source와 target labeled data 모두에 대해 계산할 수 있다.

unsupervised adaptation에서는 target domain에 labeled data가 없으므로, classification loss는 source labeled data에 대해서만 계산된다. 하지만 MMD loss는 라벨이 필요 없기 때문에 source와 target 전체 데이터를 모두 이용할 수 있다.

따라서 이 방법의 장점은, target 라벨이 없더라도 domain discrepancy를 줄이는 방향으로 representation을 계속 조정할 수 있다는 점이다.

### 학습 절차

학습은 standard backpropagation으로 진행된다. 기존 pre-trained CNN의 하위 층은 복사해오고, adaptation layer와 classifier는 새로 학습해야 하므로 이 둘의 learning rate를 하위 층보다 10배 크게 설정했다. 이는 새로 추가된 층이 빠르게 적응하도록 하기 위한 선택이다.

학습 중에는 minibatch 단위로 source와 target 데이터를 함께 사용해 MMD를 계산하고, labeled 데이터에 대해서는 classification loss를 계산한다. 두 손실을 합친 joint loss를 기준으로 전체 네트워크를 미세조정한다.

요약하면, 이 방법은 다음 흐름으로 이해할 수 있다.

먼저 pre-trained CNN에서 MMD를 기준으로 adaptation layer의 위치를 정한다. 다음으로 여러 bottleneck 차원을 실험해 MMD가 가장 작은 차원을 정한다. 마지막으로 classification loss와 MMD regularization을 함께 사용해 전체 네트워크를 fine-tuning한다. 이 세 단계 전체가 domain invariance를 중심으로 설계되어 있다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

실험은 Office dataset에서 수행되었다. 이 데이터셋은 Amazon, DSLR, Webcam의 세 도메인으로 구성되며, 31개 object category를 포함한다. 예를 들면 keyboard, file cabinet, laptop 같은 사무 환경 객체들이다. 가장 큰 도메인은 2817장의 라벨 이미지를 가진다.

논문은 표준 평가 설정을 따른다. 주요 transfer task는 세 가지다.

Amazon$\rightarrow$Webcam
DSLR$\rightarrow$Webcam
Webcam$\rightarrow$DSLR

각 task마다 5개의 random train/test split에 대해 평균과 standard error를 보고한다.

학습 프로토콜은 기존 연구와 동일하게 맞췄다. Amazon이 source일 때는 클래스당 20장의 source 예시를 사용하고, Webcam이나 DSLR이 source일 때는 클래스당 8장을 사용한다. supervised adaptation에서는 target domain에서 클래스당 3장의 labeled sample을 사용한다. unsupervised adaptation에서는 target labeled sample을 사용하지 않는다.

### adaptation layer 위치 평가

저자들은 먼저 “MMD가 layer selection에 실제로 도움이 되는가?”를 검증한다. pre-trained CNN의 각 fully connected layer feature를 추출하고, source-target 간 MMD를 계산한다. 그리고 각 layer representation에 대해 간단한 adaptation baseline으로 target 성능을 측정한다.

Figure 3의 핵심 결론은, MMD와 target test accuracy가 대체로 역관계를 보인다는 점이다. 즉, MMD가 작은 층이 실제 adaptation 성능도 더 좋았다. 이 실험에서 $fc7$이 가장 좋은 층으로 선택되었고, $fc6$은 가장 나쁜 층으로 나타났다. 이는 MMD가 단순 이론적 척도가 아니라 실제 representation selection에도 유용하다는 근거를 제공한다.

### adaptation layer 차원 선택 결과

Figure 4에서는 adaptation layer의 차원 수를 달리하면서 MMD와 test accuracy를 비교한다. Amazon$\rightarrow$Webcam이 가장 어려운 task이기 때문에 이 task를 기준으로 차원을 선택했다. 64, 128, 256, 512, 1024, 2048, 4096 같은 다양한 차원을 실험한 것으로 읽힌다.

결과적으로 256차원이 선택되었다. 논문은 이 값이 test accuracy를 절대적으로 최대로 만드는 값은 아니지만, 지나친 극단을 피하고 전반적으로 좋은 trade-off를 제공한다고 설명한다. 이 부분은 논문의 정직한 서술이다. 즉, MMD가 완벽한 selection metric이라고 과장하지 않고, practical한 기준으로 유용하다고 주장한다.

### supervised adaptation 성능

Table 1은 supervised adaptation 결과를 보여준다. 제안 방법은 모든 transfer task에서 비교 방법보다 가장 높은 성능을 기록한다.

Amazon$\rightarrow$Webcam에서는 84.1 $\pm$ 0.6
DSLR$\rightarrow$Webcam에서는 95.4 $\pm$ 0.4
Webcam$\rightarrow$DSLR에서는 96.3 $\pm$ 0.3
평균은 91.9이다.

비교 대상 중 눈에 띄는 것들을 보면, DaNN은 평균 69.4, DLID는 73.3, SA는 59.9 수준이다. DeCAF6 S+T는 일부 task에서 매우 높지만 전체 평균 비교는 제시되지 않았다. 그래도 저자들의 방법이 당시 state-of-the-art를 상당 폭 넘어섰다는 점은 분명하다.

특히 Webcam$\rightarrow$DSLR에서 96.3%라는 수치는 매우 높다. 논문은 이 경우가 주로 pose, resolution, lighting의 비교적 작은 변화에 해당하며, 이런 종류의 bias에 대해서는 사실상 거의 invariant한 representation을 학습했다고 해석한다.

### unsupervised adaptation 성능

Table 2는 더 어려운 unsupervised adaptation 결과다. 여기서는 target labeled data가 전혀 없다.

제안 방법의 결과는 다음과 같다.

Amazon$\rightarrow$Webcam에서는 59.4 $\pm$ 0.8
DSLR$\rightarrow$Webcam에서는 92.5 $\pm$ 0.3
Webcam$\rightarrow$DSLR에서는 91.7 $\pm$ 0.8
평균은 81.2이다.

비교 방법과 비교하면, DaNN의 평균은 59.9, DLID는 60.0, SA는 40.8, GFK는 36.4다. Amazon$\rightarrow$Webcam처럼 도메인 차이가 큰 어려운 과제에서도 제안 방법이 59.4%를 달성해 이전 deep method 대비 뚜렷한 개선을 보인다.

이 결과는 논문의 핵심 주장을 강하게 뒷받침한다. target label이 전혀 없어도, source classification과 domain confusion을 함께 최적화한 표현은 이전 방법보다 훨씬 잘 전이된다.

### regularization 효과 분석

Figure 5는 regularized fine-tuning과 unregularized fine-tuning의 학습 곡선을 비교한다. Amazon$\rightarrow$Webcam unsupervised split에서 초반 700 iteration 동안의 변화를 보여준다.

흥미롭게도 초기에는 regularization이 없는 방법이 더 빨리 성능이 오른다. 하지만 곧 source 데이터에 과적합되기 시작하고, target test accuracy가 떨어진다. 반면 MMD regularization을 넣은 방법은 초기 학습 속도는 느리지만, 과적합을 막기 때문에 최종적으로 더 높은 accuracy에 도달한다.

이 실험은 제안 기법의 역할을 매우 선명하게 보여준다. MMD regularizer는 단순히 source-target 평균을 맞추는 부가 항이 아니라, 실제로 deep fine-tuning 과정에서 overfitting을 억제하는 효과적인 regularizer라는 것이다.

### 정성적 시각화

Figure 6에서는 t-SNE embedding으로 learned representation과 원래 pre-trained $fc7$ representation을 비교한다. Amazon 이미지는 파란색, Webcam 이미지는 초록색으로 표시된다.

저자들의 해석에 따르면, 제안 방법의 representation에서는 클래스별로 더 조밀한 cluster가 형성되면서도, 같은 클래스 안에서 서로 다른 도메인 샘플이 잘 섞인다. 반면 원래 $fc7$ 공간에서는 같은 클래스라도 Amazon과 Webcam이 서로 다른 cluster로 나뉘는 경우가 많다. 예시로 monitor 클래스가 언급된다. 기존 $fc7$에서는 Amazon monitor와 Webcam monitor가 분리되지만, 제안 표현에서는 같은 cluster로 섞인다.

이 정성 결과는 정량 결과와 잘 맞아떨어진다. 단순히 accuracy만 오른 것이 아니라, feature 공간 자체가 domain invariance를 더 잘 반영하도록 재구성되었음을 보여준다.

### 역사적 진전 비교

Figure 7은 Office dataset에서의 historical progress를 요약한다. hand-crafted feature 기반 방법들은 blue circle, deep representation 기반 방법들은 red square로 표시된다. 저자들은 Amazon$\rightarrow$Webcam supervised task에서 DeCAF 대비 3.4%p 개선, unsupervised task에서 5.5%p 개선을 보고한다.

이 비교는 이 논문이 단순 소폭 개선이 아니라, 당시 deep feature 기반 domain adaptation 연구 흐름에서 의미 있는 성능 도약을 제공했다는 점을 강조한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 deep representation learning과 domain adaptation을 하나의 명시적 목적 함수로 결합했다는 점이다. 이전에는 pre-trained CNN feature를 가져다 쓰거나, 별도 adaptation 기법을 shallow하게 얹는 경우가 많았다. 하지만 이 논문은 분류 성능과 domain invariance를 동시에 학습하겠다는 목표를 end-to-end fine-tuning 형태로 구현했다. 이후 등장하는 adversarial domain adaptation 계열 연구들을 떠올리면, 이 논문은 그 이전 단계에서 매우 중요한 방향을 제시한 셈이다.

또 다른 강점은 MMD를 단순 loss로만 쓰지 않고, layer depth selection과 layer width selection에도 사용했다는 점이다. 즉, model selection criterion과 training objective를 같은 원리로 묶었다. 이는 방법 전체의 설계 일관성을 높인다.

실험 측면에서도 강점이 있다. supervised와 unsupervised 두 setting을 모두 평가했고, Office benchmark의 세 가지 주요 transfer task에 대해 일관되게 strong result를 보였다. 특히 학습 곡선과 t-SNE visualization을 통해 왜 성능이 좋아지는지까지 설명하려고 한 점이 좋다.

하지만 한계도 분명하다.

첫째, MMD가 두 도메인의 평균 차이를 줄이는 데는 유용하지만, 더 복잡한 고차 통계 차이나 class-conditional alignment까지 충분히 반영하는지는 이 논문만으로는 알기 어렵다. 실제로 source와 target 전체 평균만 맞추면, 클래스 구조까지 완벽히 정렬된다고 보장할 수는 없다. 논문도 이 부분을 깊게 분석하지는 않는다.

둘째, adaptation layer 차원 선택에서 저자들 스스로 인정하듯이 MMD가 반드시 최고의 test accuracy를 주는 차원을 고르지는 않는다. 즉, MMD는 유용한 heuristic이지만 완전한 model selection oracle은 아니다. Figure 4의 irregularity에 대해서도 finer sampling이 필요할 수 있다고 적고 있다.

셋째, 실험이 Office dataset에 집중되어 있다. 당시에는 표준 벤치마크였지만, 데이터 규모가 작고 도메인 종류도 제한적이다. 따라서 더 큰 규모나 더 복잡한 shift에서도 같은 효과가 유지되는지는 논문 안에서 직접 증명되지 않는다.

넷째, 논문은 classification loss의 구체적 형태나 optimization의 세부 구현을 매우 자세히 설명하지는 않는다. 예를 들어 MMD 계산의 minibatch 안정성이나 batch composition의 영향 같은 실무적 세부사항은 본문에 충분히 드러나지 않는다. 따라서 재현성 측면에서는 다소 아쉬움이 있다.

비판적으로 보면, 이 방법은 domain discrepancy를 줄이는 원리가 비교적 단순하고 직관적이어서 장점이 크지만, 그만큼 alignment의 표현력이 제한될 가능성도 있다. 이후 연구들이 adversarial objective나 conditional alignment, moment matching의 확장형을 탐구하게 된 이유도 여기서 찾을 수 있다. 그럼에도 불구하고 이 논문은 “deep network를 직접 domain-invariant하게 학습한다”는 핵심 방향을 분명히 제시했다는 점에서 높은 가치를 가진다.

## 6. 결론

이 논문은 domain adaptation에서 중요한 두 요구, 즉 분류 성능과 도메인 불변성(domain invariance)을 동시에 만족시키는 deep CNN 학습 방법을 제안했다. 핵심은 adaptation layer를 추가하고, classification loss에 MMD 기반 domain confusion term을 더해 joint optimization을 수행하는 것이다. 또한 MMD를 이용해 adaptation layer의 위치와 차원을 선택하는 절차까지 제안함으로써, 설계와 학습이 하나의 원리로 연결되도록 했다.

실험적으로는 Office benchmark의 supervised 및 unsupervised adaptation setting 모두에서 당시 최고 수준의 성능을 달성했다. 특히 source에 과적합되기 쉬운 fine-tuning 과정에서 MMD regularization이 실제로 일반화를 개선한다는 점을 정량적, 정성적으로 설득력 있게 보여주었다.

이 연구의 의의는 단순히 Office dataset 성능 향상에 있지 않다. 더 중요한 점은 deep feature가 강력하다는 사실만으로는 domain shift 문제가 완전히 해결되지 않으며, representation learning 단계에서부터 domain invariance를 직접 최적화해야 한다는 메시지를 명확히 제시했다는 것이다. 실제 응용에서는 새로운 센서, 새로운 카메라, 새로운 환경으로 데이터 분포가 계속 바뀌기 때문에, 이런 접근은 매우 실용적이다. 또한 이후의 adversarial domain adaptation, discrepancy minimization 기반 transfer learning 연구들에 중요한 발판이 된다는 점에서도 이 논문은 의미 있는 기여를 한다.
