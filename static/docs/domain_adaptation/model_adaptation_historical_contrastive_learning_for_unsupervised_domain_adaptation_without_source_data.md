# Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data

* **저자**: Jiaxing Huang, Dayan Guan, Aoran Xiao, Shijian Lu
* **발표연도**: 2021
* **arXiv**: [https://arxiv.org/abs/2110.03374](https://arxiv.org/abs/2110.03374)

## 1. 논문 개요

이 논문은 **source data에 접근할 수 없는 상태에서** 이미 source domain으로 학습된 모델을 target domain에 적응시키는 문제를 다룬다. 저자들은 이를 일반적인 UDA(Unsupervised Domain Adaptation)와 구분하여 **UMA(Unsupervised Model Adaptation)** 또는 **source-free domain adaptation**에 해당하는 설정으로 본다. 핵심 전제는, 기존 UDA처럼 labeled source data와 unlabeled target data를 동시에 볼 수 없고, 오직 source로 사전학습된 모델만 target 쪽으로 가져와 적응시켜야 한다는 점이다.

이 문제가 중요한 이유는 분명하다. 실제 환경에서는 source data가 매우 크고, 외부 반출이 어렵고, 개인정보나 지식재산권 문제 때문에 공유가 제한되는 경우가 많다. 논문은 Table 1을 통해 dataset 크기와 source-trained model 크기를 비교하며, source data 전체를 옮기는 것보다 **학습된 모델만 전달하는 것이 훨씬 작고 효율적**이라는 점을 강조한다. 예를 들어 semantic segmentation, object detection, classification에 해당하는 여러 데이터셋의 저장 용량은 수천 MB에서 수만 MB 수준인데, 대응하는 source-trained model은 수백 MB 수준이다. 즉, 실제 배포 관점에서 UMA는 충분히 실용적인 설정이다.

하지만 source data가 없다는 것은 단순한 편의가 아니라 학습 난이도를 급격히 높인다. 일반적인 domain adaptation은 source와 target 분포를 같이 보면서 정렬할 수 있지만, UMA에서는 source 쪽 supervision이 사라져 모델이 target에 맞춰지는 과정에서 원래 source에서 배운 decision boundary나 semantic structure를 잃어버리기 쉽다. 저자들은 이를 사실상 **source hypothesis를 잊어버리는 문제**, 즉 catastrophic forgetting과 유사한 형태로 본다.

이를 해결하기 위해 제안한 방법이 **Historical Contrastive Learning (HCL)**이다. 이 방법의 기본 생각은 source data를 직접 저장하거나 재사용할 수는 없지만, **과거 시점의 모델(historical models)** 은 source hypothesis의 흔적을 담고 있으므로, 이를 일종의 memory처럼 활용해 현재 모델이 target에 적응하면서도 원래의 판별 능력을 유지하도록 만들 수 있다는 것이다. HCL은 이 아이디어를 두 가지 축으로 구체화한다. 하나는 instance 수준의 판별 표현을 학습하는 **HCID(Historical Contrastive Instance Discrimination)**, 다른 하나는 pseudo label 기반으로 category 수준의 판별 표현을 학습하는 **HCCD(Historical Contrastive Category Discrimination)** 이다.

요약하면, 이 논문은 “source data 없이도 domain adaptation이 가능한가?”라는 질문에 대해, 단순히 entropy minimization이나 pseudo-labeling만 하는 것이 아니라, **historical source hypothesis를 contrastive learning과 self-training에 연결하는 memory 기반 해법**을 제시한다는 점에서 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 매우 명확하다. **source data 자체는 사라졌지만, source-trained model과 적응 과정에서 생성된 과거 모델들은 source domain의 지식과 decision structure를 부분적으로 보존하고 있다.** 따라서 현재 모델이 target 데이터만 보고 학습할 때, 이 historical models를 참조 대상으로 삼으면 source 쪽 의미 구조를 잃지 않으면서 적응할 수 있다.

기존 source-free/UMA 계열 접근은 대체로 세 가지 흐름으로 볼 수 있다. 첫째, source-trained classifier를 고정하거나 source hypothesis를 직접 활용해 target에서 information maximization을 수행하는 방식이다. 둘째, generator나 GAN으로 source-like sample을 복원하려는 방식이다. 셋째, self-entropy reduction이나 pseudo-label self-training으로 target prediction을 안정화하는 방식이다. 이 논문은 이러한 흐름과 달리, **과거 모델을 explicit memory로 삼아 현재 표현과 비교하는 contrastive formulation**을 도입했다는 점에서 차별적이다.

저자들이 제안한 차별점은 두 층위에서 이해할 수 있다.

첫째, **instance level**에서의 차별성이다. HCID는 현재 모델이 만든 query embedding과 historical models가 만든 key embedding을 대비시켜, 같은 샘플의 다른 표현은 가깝게, 다른 샘플의 표현은 멀어지게 학습한다. 중요한 점은 positive key가 단순한 현재 배치 내부의 augmentation이 아니라, **historical model이 만든 representation**이라는 것이다. 이 구조 덕분에 target 데이터의 discriminative representation을 학습하면서도, representation이 source hypothesis에서 지나치게 멀어지지 않도록 제어할 수 있다.

둘째, **category level**에서의 차별성이다. HCCD는 target sample에 pseudo label을 부여하되, 그 pseudo label의 신뢰도를 현재 모델과 historical model들의 prediction consistency로 조절한다. 즉, 단순히 현재 모델이 높은 confidence를 보였다고 해서 강하게 학습하지 않고, **과거 모델들과도 예측이 일관된 샘플일수록 더 큰 가중치**를 준다. 이는 noisy pseudo label 문제를 완화하면서 self-training을 보다 안정적으로 만든다.

결국 HCL의 핵심은, UMA에서 부족한 source supervision을 **historical source hypothesis**로 보완한다는 데 있다. 그리고 이 보완은 단순 regularization이 아니라, 하나는 contrastive instance learning, 다른 하나는 consistency-aware self-training이라는 두 개의 상보적인 학습 신호로 구현된다. 저자들은 HCID가 unseen data에 잘 일반화되는 instance-discriminative representation을 만들고, HCCD가 downstream recognition objective에 정렬된 category-discriminative representation을 만든다고 해석한다. 이 두 성질이 결합되면 representation space가 더 잘 구조화되고, 결국 segmentation, detection, classification 전반에서 성능 향상으로 이어진다는 것이 논문의 주장이다.

## 3. 상세 방법 설명

### 전체 구조

논문에서의 모델 적응 과정은 대략 다음처럼 이해할 수 있다. 먼저 source domain에서 학습된 초기 모델 $G^0$ 또는 encoder $E^0$가 있다. 이후 target domain의 unlabeled data만 사용하여 모델을 점진적으로 적응시키는데, 이때 현재 모델 $G^t$ 또는 $E^t$만 쓰지 않고, 이전 epoch 또는 이전 단계에서 저장된 historical model들 $G^{t-m}, E^{t-m}$ 도 함께 참조한다. 현재 모델은 target에 맞도록 변화해야 하지만, historical models는 과거의 source hypothesis를 담고 있기 때문에 일종의 anchor 역할을 한다.

HCL은 크게 두 손실로 구성된다.

1. **HCID**: embedding space에서 instance discrimination을 수행한다.
2. **HCCD**: output probability space에서 category discrimination을 수행한다.

즉, 하나는 feature-level, 다른 하나는 prediction-level에서 historical information을 사용한다.

### 3.1 Historical Contrastive Instance Discrimination (HCID)

HCID의 목적은 unlabeled target data로부터 **instance-discriminative representation**을 배우는 것이다. 여기서 “instance discrimination”은 같은 샘플의 표현은 서로 가깝게, 다른 샘플의 표현은 멀게 학습하는 contrastive learning의 전형적 목표를 뜻한다.

논문은 query sample $x_q$ 와 key sample 집합 $X_k={x_{k_0}, x_{k_1}, \dots, x_{k_N}}$ 를 둔다. 현재 모델의 encoder $E^t$ 는 query를 인코딩하여

$$
q^t = E^t(x_q)
$$

를 만든다. 반면 historical encoder $E^{t-m}$ 는 각 key sample을 인코딩하여

$$
k_n^{t-m} = E^{t-m}(x_{k_n}), \quad n=0,\dots,N
$$

을 만든다.

여기서 positive key는 query sample의 augmentation에 해당하는 샘플이고, 나머지는 negative key다. 그러면 HCID의 손실은 다음과 같이 주어진다.

$$
\mathcal{L}_{\text{HisNCE}} = \sum_{x_q \in X_{tgt}} -\log \frac{ \exp(q^t \cdot k_+^{t-m}/\tau), r_+^{t-m} }{ \sum_{i=0}^{N} \exp(q^t \cdot k_i^{t-m}/\tau), r_i^{t-m} }
$$

이 식은 형태상 InfoNCE와 유사하지만, 중요한 차이가 있다.

첫째, key를 현재 모델이 아니라 **historical model이 생성한다**.
둘째, 각 key마다 **reliability weight** $r_i^{t-m}$ 가 존재한다.

온도 파라미터 $\tau$ 는 일반 contrastive learning과 같은 역할을 하며, similarity 분포의 sharpness를 조절한다. 핵심은 $r_i^{t-m}$ 인데, 저자들은 이것을 historical embedding의 신뢰도라고 해석한다. 즉, 과거 모델이 만든 모든 representation이 equally useful한 것은 아니며, 잘 학습된 historical embedding을 더 기억하고, 불안정하거나 품질이 낮은 embedding의 영향은 줄이고자 한다. 논문은 이 reliability를 **classification entropy**로 추정한다고 설명한다. 엔트로피가 낮은 예측은 상대적으로 확신이 높으므로 더 신뢰할 수 있는 key로 간주하는 것으로 이해할 수 있다.

이 손실의 의미를 쉬운 말로 풀면 다음과 같다. 현재 모델이 target 샘플을 어떻게 표현해야 할지 배울 때, 완전히 새롭게 representation space를 만들지 않고, **과거 모델들이 같은 샘플을 어떻게 보았는지**를 positive reference로 삼는다. 동시에 다른 샘플들의 historical representation과는 멀어지도록 하여 discrimination을 유지한다. 이렇게 하면 target 분포에 맞춘 표현 학습과 source hypothesis 보존을 동시에 추구할 수 있다.

논문은 또 하나의 중요한 설명을 덧붙인다. 만약 $m=0$ 이고 모든 reliability가 1이라면, 즉 $r_i^{t-m}=1$ 이면, 이 식은 사실상 일반 InfoNCE의 특수한 경우가 된다. 따라서 HCID는 기존 contrastive learning을 UMA 설정에 맞게 일반화한 형태라고 볼 수 있다. 또한 실제 구현에서는 하나의 historical model만 쓰는 것이 아니라 **복수의 historical models에 대해 Eq. (1)을 여러 번 계산**할 수 있다고 설명한다. 이는 memory 범위를 더 넓혀 과거 representation을 더 풍부하게 활용할 수 있음을 뜻한다.

### 3.2 Historical Contrastive Category Discrimination (HCCD)

HCID가 feature space의 instance discrimination에 초점을 둔다면, HCCD는 output space에서 **category-discriminative representation**을 학습하게 한다. 논문은 이를 self-training의 한 형태로 본다. 하지만 일반 pseudo-label self-training과 달리, pseudo label의 기여도를 **historical consistency**로 조절한다는 점이 핵심이다.

먼저 unlabeled sample $x$ 에 대해, 현재 모델과 historical model은 각각

$$
p^t = G^t(x), \quad p^{t-m} = G^{t-m}(x)
$$

와 같은 $K$-class probability vector를 출력한다. 현재 모델의 예측으로 pseudo label을 생성하고,

$$
\hat{y} = \mathbf{\Gamma}(p^t)
$$

historical consistency는

$$
h_{con} = 1 - \text{Sigmoid}\left(|p^t - p^{t-m}|_1\right)
$$

로 계산한다.

이 식의 의미는 직관적이다. 현재 모델과 historical model의 class probability 분포가 서로 비슷하면 $|p^t - p^{t-m}|*1$ 가 작아지고, 따라서 $h*{con}$ 은 커진다. 반대로 예측이 크게 다르면 consistency가 낮다고 판단해 $h_{con}$ 이 작아진다.

이후 HCCD는 weighted cross-entropy 형태의 self-training loss를 사용한다.

$$
\mathcal{L}_{\text{HisST}} = -\sum_{x \in X_{tgt}} h_{con}\times \hat{y}\log p_x
$$

여기서 본질은 pseudo label 자체보다, 그 pseudo label을 **얼마나 믿을 것인가**를 historical consistency로 정한다는 점이다. 현재 모델만 보고 confidence가 높다고 해서 바로 강하게 학습하면 오류가 증폭될 수 있다. 그러나 현재와 과거 모델이 비슷한 결론에 도달한 샘플은 더 안정적으로 학습된 샘플일 가능성이 높다. 따라서 그런 샘플의 self-training 손실을 더 크게 반영하고, 불일치가 큰 샘플은 손실 기여를 줄인다.

논문 표현을 빌리면, prediction consistency가 높은 샘플은 **well-learnt sample**, 낮은 샘플은 **poorly-learnt sample**으로 간주된다. 이 설계는 noisy pseudo labels의 영향력을 줄이고, category-level decision boundary를 안정적으로 개선하는 역할을 한다.

### 3.3 HCID와 HCCD의 상보성

저자들은 HCID와 HCCD가 **orthogonal self-supervision signals**를 제공한다고 해석한다. HCID는 샘플 단위의 구분 가능성을 강조하고, HCCD는 클래스 단위의 정렬 가능성을 강조한다. 전자는 “서로 다른 샘플은 구분되어야 한다”는 representation learning 관점이고, 후자는 “같은 클래스는 같은 쪽으로 모여야 한다”는 classification 관점이다.

이 상보성은 논문의 실험에서도 확인된다. semantic segmentation의 ablation에서 HCID만 써도, HCCD만 써도 각각 의미 있는 성능을 내지만, 두 방법을 결합한 HCL이 가장 높은 성능을 얻는다. 따라서 HCL은 단일 기법이라기보다, **instance-level contrastive regularization과 category-level consistency-aware self-training의 결합**으로 보는 것이 더 정확하다.

### 3.4 이론적 통찰

논문은 HCID와 HCCD에 대해 각각 probabilistic model 관점의 해석과 수렴성 주장을 제시한다.

* **Proposition 1**: HCID는 Expectation Maximization으로 최적화되는 maximum likelihood 문제로 모델링될 수 있다.
* **Proposition 2**: HCID는 특정 조건에서 수렴한다.
* **Proposition 3**: HCCD는 Classification Expectation Maximization으로 최적화되는 classification maximum likelihood 문제로 볼 수 있다.
* **Proposition 4**: HCCD 역시 특정 조건에서 수렴한다.

다만 제공된 본문에는 증명 자체는 포함되어 있지 않고, **appendix에 증명이 있다**고만 언급되어 있다. 따라서 이 보고서에서는 그 수렴 조건이나 증명의 세부 전개를 재구성할 수는 없다. 중요한 것은 저자들이 이 방법을 단순 heuristic이 아니라, 확률적 최적화 문제와 연결되는 방식으로 정당화하려 했다는 점이다.

## 4. 실험 및 결과

이 논문은 HCL을 특정 task 하나에만 적용하지 않고, **semantic segmentation, object detection, image classification**의 세 가지 대표적 computer vision task에 걸쳐 평가한다. 또한 일반 closed-set adaptation뿐 아니라 partial-set, open-set adaptation까지 확장하여, 방법의 일반성을 강조한다.

### 데이터셋과 설정

semantic segmentation에서는 **GTA5 $\rightarrow$ Cityscapes** 와 **SYNTHIA $\rightarrow$ Cityscapes** 를 사용한다. 두 source 데이터셋은 synthetic urban scene이고, target은 real-world Cityscapes다. 평가지표는 일반적으로 **mIoU** 이다.

object detection에서는 **Cityscapes $\rightarrow$ Foggy Cityscapes** 와 **Cityscapes $\rightarrow$ BDD100k** 를 사용한다. 평가지표는 클래스별 AP와 **mAP** 이다.

image classification에서는 **VisDA17** 과 **Office-31** 을 사용한다. 추가로 **Office-Home** 에 대해 partial-set DA와 open-set DA까지 실험한다. 분류 문제에서는 task별 accuracy와 평균 accuracy가 핵심 지표다.

모델 backbone은 task별로 표준 구성을 따른다. segmentation은 DeepLab-V2, detection은 Faster R-CNN, classification은 VisDA17에 ResNet-101, Office-31에 ResNet-50을 사용했다. optimizer는 모두 SGD 계열이며, task별 learning rate schedule이 제시되어 있다. 이 점은 방법의 성능 향상이 backbone 교체보다는 학습 전략 자체에서 왔음을 보여준다.

### 4.1 Semantic Segmentation 결과

#### GTA5 $\rightarrow$ Cityscapes

Table 2에서 제안한 **HCL 단독 모델**은 mIoU **48.1**을 기록한다. 이는 source-free baseline인 UR의 **45.1**, SFDA의 **45.8**보다 높다. 또한 각각의 기존 source-free 방법에 HCL을 결합한 **UR + HCL = 49.1**, **SFDA + HCL = 49.3**은 더 높은 성능을 보여 HCL의 보완적 성격을 입증한다.

흥미로운 점은 source data를 사용하는 일반 UDA 방법들과 비교해도 HCL이 경쟁력 있다는 것이다. 예를 들어 CRST는 47.1, CrCDA는 48.6인데, source-free임에도 불구하고 **+HCL 설정은 이 범위를 넘거나 비슷한 수준**에 도달한다. 이는 source data가 없어도 historical model memory만으로 상당한 adaptation 효과를 낼 수 있음을 시사한다.

Ablation도 중요하다. Table 2 하단에서 **HCID 단독은 45.6**, **HCCD 단독은 46.7**, **HCL 결합은 48.1**이다. 즉 둘 중 하나만 써도 성능 향상이 있지만, 결합 시 추가 이득이 생긴다. 이는 논문의 핵심 주장인 instance-level signal과 category-level signal의 상보성을 뒷받침한다.

#### SYNTHIA $\rightarrow$ Cityscapes

Table 3에서도 비슷한 경향이 나타난다. **HCL 단독은 mIoU 43.5, mIoU* 50.2**를 기록한다. 기존 source-free 방법 UR은 39.6/45.0, SFDA는 42.4/48.7이다. 여기에 HCL을 결합하면 각각 **44.1/51.1**, **45.0/51.9**로 더 좋아진다.

absolute number 자체는 GTA5보다 낮지만, 이는 SYNTHIA $\rightarrow$ Cityscapes가 일반적으로 더 까다로운 설정이기 때문으로 해석할 수 있다. 중요한 것은 여기서도 HCL의 경향이 안정적으로 유지된다는 점이다.

### 4.2 Object Detection 결과

#### Cityscapes $\rightarrow$ Foggy Cityscapes

Table 4에서 source-free baseline인 SFOD는 mAP **33.5**를 기록한다. 제안한 **HCL 단독**은 **39.7**, 그리고 **SFOD + HCL**은 **40.3**이다. 즉 HCL이 detection에서도 유효하며, source-free baseline 대비 꽤 큰 폭의 향상을 준다.

비-source-free UDA 방법들과 비교하면, CRDA는 37.4, MLDA와 CAFA는 36.0 수준이다. 즉 source data 없이도 HCL은 오히려 더 높은 수치를 보인다. 이것은 논문의 가장 인상적인 결과 중 하나다. 단순히 source-free setting에서 “그럭저럭 된다”가 아니라, **실제 source-dependent UDA baseline들과도 대등하거나 우세한 결과**를 보이기 때문이다.

#### Cityscapes $\rightarrow$ BDD100k

Table 5에서도 동일한 패턴이 반복된다. SFOD는 mAP **29.0**, HCL 단독은 **30.3**, SFOD + HCL은 **31.1**이다. 수치 향상 폭은 Foggy Cityscapes보다는 조금 작지만, source-free adaptation에서 일관된 개선을 보인다. 클래스별로도 car, bus, mcycle, bicycle 등 다수 항목에서 개선이 나타난다.

### 4.3 Image Classification 결과

#### VisDA17

Table 6에서 HCL 단독은 평균 accuracy **83.5**를 기록한다. 이는 기존 source-free baseline인 3C-GAN의 **81.6**, SHOT의 **80.4**보다 높다. 또 3C-GAN + HCL은 **84.2**, SHOT + HCL은 **83.9**로 더 개선된다.

클래스별 수치를 보면 특히 Car, Motor, Truck 같은 일부 클래스에서 HCL의 향상이 두드러진다. 반면 어떤 클래스에서는 다른 방법이 더 높을 수도 있다. 즉 HCL이 모든 클래스를 동일하게 압도한다기보다는, **평균적으로 더 안정된 class transfer**를 보여준다고 보는 것이 적절하다.

#### Office-31

Table 7에서 HCL 단독 평균은 **89.8**이고, 3C-GAN은 **89.6**, SHOT은 **87.3**이다. 여기에 HCL을 결합하면 3C-GAN + HCL은 **90.6**, SHOT + HCL은 **90.1**이 된다. Office-31은 비교적 작은 고전 벤치마크이기 때문에 절대 개선 폭은 크지 않지만, 여러 task 평균에서 일관되게 우세하다.

### 4.4 Partial-set / Open-set Adaptation 결과

Table 8은 HCL이 단순 closed-set adaptation뿐 아니라, class mismatch가 있는 더 어려운 설정에도 일반화될 수 있음을 보여준다.

partial-set DA에서는 HCL 평균이 **79.6**, SHOT은 **76.8**, SHOT + HCL은 **80.1**이다. 즉 target 클래스가 source 클래스의 부분집합인 상황에서도 HCL이 효과적이다.

open-set DA에서는 HCL 평균이 **72.6**, SHOT은 **71.5**, SHOT + HCL은 **73.2**이다. open-set은 target에 unknown class가 섞여 있어 훨씬 어렵지만, 여기서도 historical consistency를 활용한 방식이 성능 향상에 기여한다.

### 4.5 정성적 결과와 표현 시각화

논문은 GTA5 $\rightarrow$ Cityscapes task에서 t-SNE 시각화(Figure 3)를 제공한다. 저자들의 해석에 따르면 HCL은 UR, SFDA보다 **더 instance-discriminative하면서도 category-discriminative한 feature representation**을 형성한다. 즉, 같은 클래스의 샘플은 잘 모이고, 다른 샘플들 사이의 구분도 유지되는 구조다. 이는 HCID와 HCCD의 결합 효과와 잘 맞는다.

또 Figure 4의 qualitative segmentation 비교에서는 sidewalk, road, sky 같은 영역에서 HCL이 더 나은 segmentation 결과를 보인다고 설명한다. 제공된 텍스트만으로 이미지를 직접 확인할 수는 없지만, 저자들은 정량 결과뿐 아니라 시각적 품질에서도 HCL이 우수하다고 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정과 방법 설계가 매우 잘 맞아떨어진다**는 점이다. UMA의 본질적 어려움은 source data 부재로 인해 source hypothesis를 유지하기 어렵다는 데 있는데, HCL은 이 문제를 정면으로 겨냥한다. source data 대신 historical models를 memory로 사용한다는 발상은 단순하고도 설득력이 있다. 특히 “모델은 남아 있지만 데이터는 없다”는 실제 배포 상황과 잘 맞기 때문에, 이론적 흥미뿐 아니라 실용적 가치도 크다.

두 번째 강점은 방법이 **task-agnostic** 하다는 점이다. segmentation, detection, classification 모두에서 일관된 개선을 보였고, partial-set/open-set 같은 학습 설정 변화에도 대응했다. 많은 domain adaptation 기법은 특정 task 구조에 강하게 의존하는데, HCL은 representation-level과 prediction-level의 일반적 원리를 사용하기 때문에 범용성이 높다.

세 번째 강점은 **보완 가능성(complementarity)** 이다. HCL은 standalone method로도 성능이 좋지만, 기존 source-free 방법에 추가 모듈처럼 붙여도 성능을 높인다. 이는 실제 연구나 응용에서 매우 유리하다. 기존 파이프라인을 전부 바꾸지 않아도, historical contrast 구조를 더해 성능을 개선할 수 있기 때문이다.

네 번째로, 방법론의 구조가 해석 가능하다. HCID는 “과거 representation을 참조하는 contrastive learning”, HCCD는 “과거 예측과의 일관성을 반영한 pseudo-label self-training”으로 이해할 수 있어, black-box heuristic처럼 보이지 않는다. 여기에 이론적 정당화까지 제시하려 한 점도 강점이다.

반면 한계도 분명하다.

가장 먼저, **historical model 저장과 관리 비용**이 실제로 얼마나 드는지는 본문에서 충분히 정량화되지 않았다. source data보다 모델이 작다는 것은 분명하지만, 여러 시점의 historical models를 얼마나 오래 저장해야 하는지, 몇 개를 쓰는 것이 최적이며 메모리와 연산 비용은 어떻게 증가하는지에 대한 상세 분석은 제공된 텍스트에 나타나지 않는다. 실무 적용 시에는 이 부분이 중요할 수 있다.

둘째, reliability score $r_i^{t-m}$ 를 classification entropy로 추정한다고 하지만, 이 선택이 왜 최선인지에 대한 비교 실험은 제공된 텍스트만으로는 확인되지 않는다. 다른 uncertainty measure나 calibration-aware weighting과 비교했을 때 어떠한지 알 수 없다.

셋째, HCCD의 pseudo-label weighting은 현재 모델과 historical model의 prediction consistency를 사용한다. 그런데 만약 historical model 자체가 특정 편향을 강하게 갖고 있거나, source-target gap가 매우 커서 과거 예측이 구조적으로 틀리는 경우에는 consistency가 오히려 잘못된 방향을 강화할 가능성도 있다. 논문은 전반적으로 좋은 결과를 보였지만, **historical hypothesis가 항상 유익한가**에 대한 실패 사례 분석은 제공된 텍스트에 없다.

넷째, 이론적 부분은 proposition 형태로 제시되지만, 핵심 증명은 appendix에 있고 현재 제공된 본문에는 없다. 따라서 독자로서는 “EM-like interpretation이 정확히 어떻게 구성되는지”, “수렴 조건이 실제 학습 설정에서 얼마나 현실적인지”를 본문만으로 파악하기 어렵다. 이 점은 이론 기여를 평가할 때 제한 요소다.

다섯째, 이 논문은 source-free adaptation 문제를 매우 잘 다루지만, 문제 설정상 여전히 **초기 source-trained model의 품질**에 강하게 의존한다. source-trained model이 약하거나 source domain 자체가 target과 지나치게 동떨어진 경우에도 HCL이 안정적으로 동작하는지는 제공된 텍스트만으로 확인할 수 없다.

종합적으로 보면, 이 논문은 핵심 아이디어와 실험적 설득력은 강하지만, historical memory의 관리 전략, reliability 설계의 대안 비교, failure mode 분석은 더 보강될 여지가 있다.

## 6. 결론

이 논문은 source data에 접근할 수 없는 UMA 설정에서, **historical source hypothesis를 활용한 contrastive learning 기반 적응 기법 HCL**을 제안했다. 제안 방법은 두 요소로 이루어진다. HCID는 현재 모델의 query와 historical model의 key를 비교하는 방식으로 instance-discriminative representation을 학습하고, HCCD는 현재와 과거 모델의 prediction consistency를 기반으로 pseudo label을 re-weighting하여 category-discriminative representation을 학습한다. 이 두 신호는 서로 다른 수준의 자기지도 학습을 제공하며, 함께 사용할 때 가장 좋은 성능을 낸다.

실험적으로 HCL은 semantic segmentation, object detection, image classification 전반에서 source-free baseline을 일관되게 능가했고, 경우에 따라 source data를 사용하는 일반 UDA 방법과도 경쟁 가능한 수준을 달성했다. 또한 partial-set 및 open-set adaptation에서도 효과를 보여, 특정 task에 한정되지 않는 일반성을 드러냈다.

이 연구의 가장 중요한 의미는, source-free adaptation을 단지 “source 없이도 흉내 내는 adaptation” 수준이 아니라, **historical models를 memory 자원으로 활용하는 본격적인 학습 프레임워크**로 확장했다는 데 있다. 실제 산업 환경에서 데이터 이동이 제한되는 상황은 점점 더 많아지고 있으므로, HCL의 아이디어는 model deployment, on-device adaptation, privacy-preserving transfer learning 같은 맥락에서 실제 응용 가능성이 크다. 앞으로는 이 논문의 결론에서 언급하듯, 이러한 memory-based learning이 다른 transfer learning 문제나 continual adaptation, test-time adaptation과 결합되며 더 넓게 발전할 가능성이 있다.
