# Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation

* **저자**: Jichang Li, Guanbin Li, Yemin Shi, Yizhou Yu
* **발표연도**: 2021
* **arXiv**: [https://arxiv.org/abs/2104.09415](https://arxiv.org/abs/2104.09415)

## 1. 논문 개요

이 논문은 Semi-Supervised Domain Adaptation, 즉 SSDA 문제를 다룬다. SSDA는 source domain에는 충분한 라벨이 있고, target domain에는 아주 적은 수의 labeled sample과 많은 unlabeled sample이 있을 때, target domain에서 잘 동작하는 분류기를 학습하는 문제다. 기존 UDA보다 현실적인 설정이며, target domain에 클래스당 1장 또는 3장 정도의 labeled sample만 있어도 성능 향상이 가능하다는 점에서 중요하다.

논문이 제기하는 핵심 문제는 다음과 같다. 기존 SSDA 방법은 보통 source와 target 사이의 정렬, 즉 inter-domain adaptation에 초점을 맞추거나, 또는 target 내부 클래스 구조를 정리하는 intra-domain adaptation을 부분적으로만 다룬다. 그런데 실제로는 두 문제가 동시에 해결되어야 한다. source 쪽 labeled data의 양이 훨씬 많기 때문에 학습된 표현은 source 중심으로 치우치기 쉽고, 그 결과 target의 unlabeled sample이 target의 labeled sample 주변에 안정적으로 모이지 못하거나, source와 target의 같은 클래스끼리도 제대로 정렬되지 못하는 문제가 생긴다.

이 논문은 이러한 문제를 해결하기 위해 Cross-Domain Adaptive Clustering, 즉 CDAC를 제안한다. 핵심 목표는 unlabeled target feature들을 클래스별 cluster로 형성하게 만들고, 동시에 그 cluster를 source domain의 대응 클래스 cluster와 정렬하는 것이다. 저자들은 이를 통해 inter-domain gap과 intra-domain gap을 동시에 줄이겠다고 주장한다.

문제의 중요성은 명확하다. SSDA에서 성능 병목은 단순히 source와 target를 전체 분포 수준에서 맞추는 것으로 해결되지 않는다. target 내부에서 클래스별 구조가 무너지면, 소수의 labeled target sample이 있어도 그 정보가 충분히 퍼지지 못한다. 따라서 target 내부 clustering과 cross-domain alignment를 함께 설계하는 것은 매우 실질적인 기여다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 sample-wise alignment나 전체 distribution-wise alignment 대신, **cluster-wise feature alignment**를 학습의 중심 단위로 삼는 것이다. 즉, target domain의 unlabeled sample들을 feature space에서 클래스별 cluster로 모이게 만들고, 그 cluster 단위로 source와 target를 정렬한다.

이 아이디어는 두 단계의 직관으로 이해할 수 있다.

첫째, target 내부에서 같은 클래스에 속할 가능성이 높은 unlabeled sample들은 feature 관점에서 서로 가까워져야 한다. 이를 위해 저자들은 unlabeled target sample 쌍에 대해 pairwise similarity label을 구성하고, 그에 맞게 classifier prediction이 일치하도록 학습시키는 adversarial adaptive clustering loss를 제안한다. 이 손실은 target 내부의 class-wise sub-distribution을 더 응집되게 만드는 역할을 한다.

둘째, 단순히 target 내부 clustering만 하면 source dominance 문제가 남는다. 논문은 feature extractor와 classifier 사이에 minimax 구조를 도입한다. feature extractor는 clustering loss를 **최소화**하여 target 내부 cluster를 잘 만들도록 하고, classifier는 같은 loss를 **최대화**하는 방향으로 학습시켜 source에 과도하게 치우친 feature bias를 줄이고 domain-invariant representation을 유도한다. 논문은 이 점을 gradient reversal layer로 구현한다고 설명한다.

여기에 더해, target에 labeled sample이 너무 적기 때문에 cluster core가 불안정할 수 있다는 문제가 있다. 이를 보완하기 위해 high-confidence pseudo labeling을 사용한다. confidence가 높은 unlabeled target sample을 의사 라벨로 target의 “사실상 labeled set”에 편입시켜, 각 클래스의 target cluster core를 더 robust하게 만든다. 이 pseudo label은 단순한 self-training이 아니라, adversarial clustering이 더 잘 작동하도록 cluster core를 강화하는 보조 장치로 쓰인다.

기존 접근과의 차별점은 논문이 직접 강조한다. 기존 SSDA 방법들이 주로 sample-wise feature alignment에 의존했다면, CDAC는 cluster-wise alignment를 정면으로 도입하고, 그것을 adversarial하게 학습하며, pseudo labeling으로 cluster core를 강화한다. 즉, intra-domain 구조 형성과 inter-domain 정렬을 하나의 학습 프레임 안에서 결합한 점이 이 논문의 차별점이다.

## 3. 상세 방법 설명

전체 모델은 feature extractor $\mathcal{G}$와 classifier $\mathcal{F}$로 구성된다. backbone은 AlexNet 또는 ResNet34를 사용한다. classifier는 bias가 없는 linear network와 normalization layer로 구성되며, 논문은 이것이 spherical feature space를 형성해 같은 클래스 샘플들의 feature variance를 줄이는 데 유리하다고 설명한다.

입력 이미지 $x$에 대한 예측은 다음과 같다.

$p(x)=\sigma(\mathcal{F}(\mathcal{G}(x)))$

여기서 $\sigma(\cdot)$는 softmax 함수이며, 결과는 클래스 확률 벡터 $\mathbf{p}$다.

먼저 기본적인 supervised 학습은 source labeled set $\mathcal{S}$와 target labeled set $\mathcal{L}$에 대해 standard cross-entropy loss로 수행된다.

$\mathbf{L}_{\mathbf{CE}}=-\sum_{(x,y)\in\mathcal{S}\cup\mathcal{L}} y \log(p(x))$

이 손실은 source와 target의 labeled data를 이용해 기본 분류 경계를 학습하게 해 준다. 하지만 이것만으로는 unlabeled target sample들이 target 내부에서 구조적으로 잘 정렬된다고 보장할 수 없다.

### 3.1 Adversarial Adaptive Clustering Loss

이 논문의 핵심은 $\mathbf{L}_{\mathbf{AAC}}$이다. 아이디어는 unlabeled target mini-batch 안에서 sample pair를 만들고, 두 샘플이 같은 클래스일 가능성이 높으면 서로 연결된 것으로 간주하는 것이다.

이를 위해 저자들은 feature magnitude 기준으로 정렬했을 때의 top-$k$ feature index를 사용한다. 두 unlabeled target sample $x_i^u$, $x_j^u$가 같은 top-$k$ index 집합을 가지면 pairwise similarity label을 1로, 아니면 0으로 둔다. 논문에서는 $k=5$로 둔다.

$s_{ij}=\mathbb{1}{\text{top}k(\mathcal{G}(x_i^u))=\text{top}k(\mathcal{G}(x_j^u))}$

이 $s_{ij}$는 “두 샘플이 feature 관점에서 같은 cluster에 속하는가”를 나타내는 이진 표지다.

그 다음, classifier의 예측 분포가 이 pairwise similarity와 일관되도록 binary cross-entropy 형태의 손실을 구성한다. 논문 원문 추출본에서는 식 (4) 내부에 `missing`이라는 깨진 토큰이 들어가 있어 표기가 일부 손상되어 있지만, 문맥상 핵심은 $\mathbf{p}_i^\mathsf{T}\mathbf{p}_j'$ 같은 **예측 분포 간 inner product**를 similarity score로 사용한다는 점이다. 즉, 원본 이미지 $x_i^u$의 prediction과 증강된 이미지 $x_j'$의 prediction이 같은 클래스를 가리키는지를 점수화한다.

의미적으로는 다음과 같다.

* $s_{ij}=1$이면 두 prediction의 inner product가 커지도록 한다.
* $s_{ij}=0$이면 그 inner product가 작아지도록 한다.

따라서 이 손실을 최소화하면, target 내부에서 비슷한 샘플은 같은 클래스로 분류되고, 다른 샘플은 다른 클래스로 분리되도록 feature structure가 정리된다. 즉, unlabeled target feature가 cluster를 이루게 된다.

### 3.2 왜 adversarial인가

논문은 단순히 $\mathbf{L}_{\mathbf{AAC}}$를 최소화하는 것만으로는 충분하지 않다고 본다. source labeled data가 압도적으로 많아서 학습 표현이 source domain 중심으로 편향될 수 있기 때문이다. 이런 상황에서 target 내부 clustering만 강하게 밀면 오히려 source 편향이 강화되어 overfitting이 심해질 수 있다고 주장한다.

그래서 저자들은 gradient reversal layer를 사용해 feature extractor와 classifier가 $\mathbf{L}_{\mathbf{AAC}}$에 대해 서로 반대 방향으로 최적화되게 만든다.

feature extractor는 다음을 최소화한다.

$\theta_{\mathcal{G}}^*=\arg\min_{\theta_{\mathcal{G}}} \mathbf{L}_{\mathbf{CE}}+\lambda \mathbf{L}_{\mathbf{AAC}}$

classifier는 다음을 최소화한다.

$\theta_{\mathcal{F}}^*=\arg\min_{\theta_{\mathcal{F}}} \mathbf{L}_{\mathbf{CE}}-\lambda \mathbf{L}_{\mathbf{AAC}}$

즉,

* $\mathcal{G}$는 $\mathbf{L}_{\mathbf{AAC}}$를 줄여 target feature가 잘 뭉치도록 만들고,
* $\mathcal{F}$는 사실상 $\mathbf{L}_{\mathbf{AAC}}$를 키우는 방향으로 작동하여, source 중심 편향을 줄이고 더 domain-invariant한 feature를 유도한다.

이 minimax 구조가 논문 제목의 “Cross-Domain Adaptive Clustering”에 해당하는 부분이다. target 내부 clustering과 source-target 간 cluster-wise alignment를 동시에 유도하는 장치라고 볼 수 있다.

### 3.3 Pseudo Labeling

SSDA에서는 target labeled sample이 클래스당 1개 또는 3개 정도이므로, target cluster core가 매우 약하다. 이 문제를 해결하기 위해 pseudo labeling loss $\mathbf{L}_{\mathbf{PL}}$를 도입한다.

mini-batch의 unlabeled target sample $x_j^u$에 대해 현재 모델의 예측 $\mathbf{p}_j$를 구하고, 그 argmax를 hard pseudo label $\hat{y}_j^u$로 만든다.

$\hat{y}_j^u=\arg\max(\mathbf{p}_j)$

이후 같은 샘플의 또 다른 augmentation인 $x_j''$에 대해 예측을 수행하고, confidence가 threshold $\tau$ 이상일 때만 pseudo label을 사용해 supervised cross-entropy를 건다.

$\mathbf{L}_{\mathbf{PL}}=-\sum*{j=1}^{M}\mathbb{1}{\max(\mathbf{p}_j)\ge\tau}\cdot \hat{y}_j^u \log(\mathbf{p}(x_j''))$

여기서 $\tau$는 pseudo label 채택 임계값이며, 실험에서는 $\tau=0.95$를 사용한다.

이 손실의 역할은 단순하다. 신뢰할 수 있는 unlabeled target sample을 추가적인 labeled target sample처럼 활용해 target 쪽 cluster core를 더 안정적으로 만드는 것이다. 논문은 이것이 adversarial clustering을 보조해 더 강한 target cluster를 형성한다고 해석한다.

### 3.4 Consistency Loss와 전체 손실

논문은 unlabeled target image마다 두 개의 augmentation을 사용한다. 하나는 AAC용, 다른 하나는 pseudo labeling용이다. 그러면 두 augmentation에 대한 예측이 지나치게 달라질 수 있으므로 consistency loss를 둔다.

$\mathbf{L}_{\mathbf{Con}}=w(t)\sum_{j=1}^{M} |\mathbf{p}_j'-\mathbf{p}_j''|^2$

여기서 $w(t)$는 ramp-up 함수다.

$w(t)=\nu e^{-5(1-\frac{t}{T})^2}$

$\nu$는 계수, $t$는 현재 step, $T$는 ramp-up 총 step 수다. 이 함수는 학습 초반보다 후반으로 갈수록 consistency loss의 영향을 더 자연스럽게 키우는 역할을 한다.

결국 전체 최적화는 다음과 같다.

$\theta_{\mathcal{G}}^*=\arg\min_{\theta_{\mathcal{G}}} \mathbf{L}_{\mathbf{CE}}+\lambda\mathbf{L}_{\mathbf{AAC}}+\mathbf{L}_{\mathbf{PL}}+\mathbf{L}_{\mathbf{Con}}$

$\theta_{\mathcal{F}}^*=\arg\min_{\theta_{\mathcal{F}}} \mathbf{L}_{\mathbf{CE}}-\lambda\mathbf{L}_{\mathbf{AAC}}+\mathbf{L}_{\mathbf{PL}}+\mathbf{L}_{\mathbf{Con}}$

정리하면 학습 절차는 다음과 같이 이해할 수 있다. 먼저 labeled source와 labeled target에 대해 supervised learning을 하고, 동시에 unlabeled target에 대해서는 AAC로 cluster 구조를 형성하며, pseudo labeling으로 cluster core를 강화하고, consistency loss로 augmentation 간 출력을 안정화한다. 그 과정에서 feature extractor와 classifier는 AAC 항에 대해 adversarial하게 상호작용한다.

### 3.5 아키텍처와 구현상의 특징

논문이 명시한 구현 요소는 다음과 같다.

* backbone은 AlexNet 또는 ResNet34
* feature extractor는 ImageNet pretrained initialization 사용
* classifier의 linear layer는 random initialization
* augmentation은 RandAugment 사용
* $\lambda=1.0$, $\nu=30.0$, $\tau=0.95$
* optimizer, learning rate, mini-batch size 등 나머지 설정은 MME를 따른다

다만 논문 추출 텍스트만으로는 optimizer 종류, learning rate 값, batch size의 정확한 수치까지는 확인할 수 없다. 논문은 “MME와 동일하다”고만 적고, 제공된 본문에는 그 구체값이 포함되어 있지 않다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 설정

논문은 세 가지 대표 벤치마크에서 CDAC를 평가한다.

첫째, DomainNet이다. 여기서는 Real, Clipart, Painting, Sketch의 4개 도메인만 선택하며, 총 126개 클래스를 사용한다. 1-shot과 3-shot 설정을 모두 평가한다.

둘째, Office-Home이다. Real, Clipart, Art, Product의 4개 도메인과 65개 클래스로 구성되며, 3-shot 설정에서 평가한다.

셋째, Office이다. DSLR, Webcam, Amazon의 3개 도메인과 31개 클래스로 이루어져 있고, AlexNet backbone에서 1-shot과 3-shot을 평가한다.

비교 기준선은 MME, UODA, BiAT, Meta-MME, APE, S+T, DANN, Ent 등이다. S+T는 source labeled와 target labeled만으로 학습한 모델이고, DANN과 Ent는 원래 UDA 방법이지만 소량의 labeled target supervision을 추가해 재학습한 버전이다.

평가 지표는 분류 정확도, 즉 accuracy(%)다.

### 4.2 DomainNet 결과

DomainNet은 가장 어려운 실험군으로 보인다. 클래스 수가 많고 도메인 차이도 크기 때문이다.

AlexNet 기준으로 CDAC의 평균 정확도는 1-shot에서 52.1%, 3-shot에서 56.2%다. 이는 기존 강한 baseline인 APE, BiAT, MME보다 모두 높다. 예를 들어 3-shot 평균은 APE 48.9%, BiAT 49.4%, Meta-MME 48.8%보다 뚜렷하게 높다.

ResNet34 기준으로는 성능 차이가 더 분명하다. CDAC의 평균 정확도는 1-shot 73.6%, 3-shot 76.0%이며, 이는 UODA의 68.5%, 71.2%, APE의 67.6%, 71.7%, BiAT의 67.1%, 69.7%보다 높다. 즉 backbone이 stronger할수록 CDAC의 clustering 기반 정렬 효과가 더 잘 드러난다고 해석할 수 있다.

각 adaptation scenario에서도 CDAC가 매우 안정적이다. 예를 들어 ResNet34 3-shot에서:

* $R \rightarrow C$: 79.6
* $R \rightarrow P$: 75.1
* $P \rightarrow C$: 79.3
* $C \rightarrow S$: 69.9
* $S \rightarrow P$: 73.4
* $R \rightarrow S$: 72.5
* $P \rightarrow R$: 81.9

으로 대부분 시나리오에서 최고 성능이다. 특히 difficult transfer로 보이는 $C \rightarrow S$나 $R \rightarrow S$에서도 baseline보다 큰 폭의 개선이 있다.

### 4.3 Office-Home 결과

Office-Home 3-shot에서 AlexNet 기준 평균 정확도는 CDAC 56.8%로, APE 55.6%, BiAT 56.4%, MME 55.2%보다 높다. 절대 차이는 DomainNet보다 작지만, 다양한 도메인 전이 전반에서 우수한 평균을 보여준다.

ResNet34 기준 평균 정확도는 CDAC 74.2%이며, APE 74.0%, MME 73.1%, Ent 71.9%보다 높다. 차이는 크지 않지만 mean accuracy에서 최고다. 개별 시나리오를 보면 모든 항목에서 무조건 최고는 아니지만, 전체적으로 매우 균형 잡힌 성능을 보인다. 예를 들어 $C \rightarrow R$에서는 80.2, $C \rightarrow P$에서는 81.4로 매우 높은 수치를 기록한다.

이 결과는 CDAC가 DomainNet처럼 어려운 대규모 데이터셋뿐 아니라, 상대적으로 전형적인 Office-Home에서도 잘 일반화된다는 점을 보여준다.

### 4.4 Office 결과

Office는 상대적으로 작은 벤치마크이며 AlexNet만 사용했다. 여기서도 CDAC는 가장 좋은 mean accuracy를 기록한다.

* 1-shot 평균: 63.1
* 3-shot 평균: 70.0

비교하면 MME는 56.5 / 67.6, BiAT는 56.3 / 68.4, APE는 3-shot에서 68.3이다. 즉 작은 데이터셋에서도 CDAC의 이점이 유지된다. 특히 1-shot에서 improvement가 더 크게 보이는데, 이는 극소량 target label 상황에서 pseudo labeling과 clustering이 더 큰 도움을 준다는 논문의 주장과 잘 맞는다.

### 4.5 Ablation Study

Table 4는 DomainNet, ResNet34, 3-shot 설정에서 각 손실 항의 역할을 검증한다. SSDA baseline은 $\mathbf{L}_{\mathbf{CE}}$만 사용한 모델이며 평균 정확도 60.0이다.

여기에 $\mathbf{L}_{\mathbf{AAC}}$만 더하면 평균 67.6으로 오른다. 즉 +7.6p 개선이다. 이는 adversarial adaptive clustering만으로도 큰 효과가 있음을 보여준다.

반대로 $\mathbf{L}_{\mathbf{PL}}$만 더하면 평균 73.4가 된다. baseline 대비 +13.4p 개선이다. 이 수치만 보면 pseudo labeling의 기여가 매우 크다. 논문 본문도 target cluster core 강화가 중요하다고 강조한다.

$\mathbf{L}_{\mathbf{AAC}}+\mathbf{L}_{\mathbf{PL}}$를 함께 쓰면 평균 75.3으로 더 오른다. 여기에 $\mathbf{L}_{\mathbf{Con}}$까지 추가한 최종 모델은 평균 76.0으로 최고 성능을 낸다.

이 ablation은 몇 가지 메시지를 준다.

첫째, AAC와 pseudo labeling은 둘 다 유효하며 상호보완적이다.
둘째, pseudo labeling의 효과가 특히 강하게 나타난다.
셋째, consistency loss는 단독 핵심이라기보다는 전체 구조를 안정화하는 마무리 역할에 가깝다.

동일한 경향은 UDA setting에서도 나타난다. 예를 들어 UDA baseline은 평균 58.6인데, AAC만 추가하면 64.0, PL만 추가하면 68.2, AAC+PL이면 72.8, 전체는 73.0이다. 이는 CDAC의 핵심 아이디어가 SSDA뿐 아니라 UDA적 설정에서도 유효하다는 보조 증거다.

### 4.6 정성적 분석

논문은 Cluster Core Distance, 즉 CCD를 사용해 source와 target의 같은 클래스 cluster 중심 사이 거리를 측정한다. CCD가 작을수록 cross-domain cluster alignment가 잘 되었다는 뜻이다. 결과적으로 CDAC가 S+T나 CDAC w/o PL보다 더 작은 CCD로 수렴한다고 보고한다. 이는 AAC가 실제로 cluster-wise alignment를 유도한다는 정성적 근거다.

또한 pseudo labeling 품질 분석에서는 DomainNet의 두 시나리오에서 best performance 시점 기준으로 전체 training example 대비 각각 최대 59.9%, 63.8% 정도에 대해 correct pseudo-label이 부여되었다고 보고한다. 이 수치는 pseudo labeling이 실제로 충분히 많은 unlabeled target data를 활용하게 해 준다는 점을 보여준다.

t-SNE 시각화에서는 학습이 진행될수록 target feature들이 target cluster core로 모이고, 동시에 source의 대응 cluster 쪽으로 가까워지는 모습을 제시한다. 예시로 “bus” 클래스에 대해 epoch가 진행되면서 source-target feature distribution이 점점 가까워진다고 설명한다. 물론 t-SNE는 정성적 도구이므로 과도한 해석은 피해야 하지만, 논문의 주장과 일관된 시각적 증거를 제공한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법 설계가 잘 맞물린다는 점이다. SSDA에서 중요한 것은 source-target 정렬만이 아니라 target 내부 클래스 구조를 잘 세우는 것이다. CDAC는 바로 այդ 지점을 겨냥한다. cluster-wise alignment라는 관점은 SSDA의 구조적 어려움을 정확히 짚고 있다.

둘째 강점은 방법이 비교적 단순하면서도 효과가 크다는 점이다. backbone을 바꾸지 않고, 기존 feature extractor-classifier 구조 위에 AAC, pseudo labeling, consistency loss를 올리는 방식이기 때문에 구현 부담이 과도하지 않다. 실험 결과도 세 개 벤치마크에서 일관되게 강하다.

셋째 강점은 ablation이 설득력 있다는 점이다. 각 손실 항의 역할이 표로 분리되어 제시되고, pseudo labeling과 AAC가 각각 그리고 함께 의미 있는 개선을 만든다는 것이 수치로 뒷받침된다. CCD와 t-SNE 분석도 방법의 직관을 보조한다.

하지만 한계도 분명하다.

첫째, AAC의 pairwise similarity label이 top-$k$ feature index 일치 여부에 의존한다는 점은 다소 heuristic하다. 이 기준이 왜 가장 적절한지, 또는 다른 similarity criterion과 비교해 얼마나 robust한지는 제공된 본문만으로 충분히 검증되지 않는다.

둘째, 식 (4)의 정확한 표기와 similarity score의 세부 정의가 현재 제공된 추출 텍스트에서는 일부 손상되어 있다. 문맥상 prediction inner product를 사용하는 것은 분명하지만, 원 논문의 정확한 수식 레이아웃과 모든 기호 정의를 여기서 완전히 복원할 수는 없다. 따라서 수식 수준의 완전한 재현에는 원 PDF 확인이 필요하다.

셋째, pseudo labeling의 효과가 매우 크다는 것은 장점이지만 동시에 위험 요소이기도 하다. high-confidence threshold $\tau=0.95$로 노이즈를 줄이려 했지만, 도메인 차이가 극심하거나 초반 모델이 불안정한 경우 잘못된 pseudo-label이 cluster core를 오염시킬 가능성이 있다. 논문은 정성 분석으로 이를 보완하지만, failure case나 class imbalance 상황에 대한 상세 논의는 제공된 텍스트에 명확히 나오지 않는다.

넷째, 계산량 측면에서 unlabeled target mini-batch 내 모든 pair를 고려하는 구조는 배치 크기에 따라 비용이 커질 수 있다. 논문은 성능 중심으로 설명하며, pairwise comparison의 비용이나 scalability에 대한 별도 분석은 제공하지 않는다.

다섯째, 실험은 classification benchmark 중심이다. segmentation이나 detection처럼 더 복잡한 structured prediction 문제로의 확장 가능성은 직접 검증되지 않았다. 논문이 “중요할 가능성”은 보여주지만, 실제 일반화 범위는 아직 제한적이다.

비판적으로 보면, 이 논문의 핵심 성공 요인은 cluster-wise adversarial alignment 그 자체와 pseudo labeling에 의한 target supervision 확대가 결합된 데 있다. 그런데 ablation 수치상 pseudo labeling의 단독 기여가 매우 크기 때문에, CDAC의 진정한 차별점이 AAC인지, 아니면 strong pseudo-labeling framework와의 결합인지 더 세밀한 분석이 있었으면 좋았을 것이다. 그럼에도 AAC를 추가했을 때 추가 개선이 분명히 존재하므로, 논문의 중심 주장 자체는 여전히 설득력이 있다.

## 6. 결론

이 논문은 SSDA에서 source 중심 편향과 target 내부 구조 불안정이라는 두 문제를 동시에 다루기 위해 CDAC를 제안했다. 핵심은 unlabeled target feature를 cluster로 조직하는 adversarial adaptive clustering loss와, target cluster core를 강화하는 pseudo labeling의 결합이다. 여기에 consistency loss를 더해 augmentation 간 예측 안정성을 확보했다.

방법론적으로 보면 CDAC는 단순한 sample-wise alignment에서 벗어나 class-aware한 cluster-wise alignment를 전면에 내세웠다는 점에서 의미가 있다. 또한 feature extractor와 classifier를 minimax 방식으로 학습해 target representation이 source에 과도하게 끌려가지 않도록 설계한 점도 SSDA 맥락에서 타당하다.

실험적으로는 DomainNet, Office-Home, Office 전반에서 state-of-the-art 수준의 성능을 보고했고, ablation과 정성 분석도 방법의 유효성을 뒷받침한다. 특히 labeled target sample이 극히 적은 1-shot, 3-shot 환경에서 강한 성능을 보인다는 점은 실제 응용 측면에서 중요하다.

향후 연구 측면에서는 pairwise similarity 정의를 더 학습 가능하게 만들거나, pseudo-label noise에 더 강한 방식으로 확장하는 방향이 유망해 보인다. 또한 이 아이디어를 classification beyond, 예를 들어 semantic segmentation 같은 structured task로 옮기는 것도 의미 있을 것이다. 전체적으로 이 논문은 SSDA에서 cluster structure를 중심에 둔 설계가 얼마나 효과적일 수 있는지를 잘 보여주는 강한 작업이다.
