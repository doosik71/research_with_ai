# Deep Hashing Network for Unsupervised Domain Adaptation

* **저자**: Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, Sethuraman Panchanathan
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1706.07522](https://arxiv.org/abs/1706.07522)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation**과 **deep hashing**을 하나의 통합 프레임워크로 결합한 연구이다. 문제 설정은 전형적인 비지도 도메인 적응과 같다. 즉, source domain에는 라벨이 있지만 target domain에는 라벨이 없고, 학습된 모델은 보지 못한 target 샘플을 정확히 분류해야 한다. 논문은 이 문제를 단순히 “도메인 차이를 줄이는 분류기 학습”으로만 보지 않고, 동시에 **target 데이터를 compact binary hash code로 표현하는 문제**까지 함께 해결하려고 한다.

저자들이 제시하는 핵심 주장은 다음과 같다. 일반적인 deep domain adaptation은 마지막 층에서 class probability를 출력하지만, 이 논문은 마지막에 **확률 벡터 대신 binary hash code**를 출력하도록 네트워크를 바꾼다. 이렇게 하면 저장과 검색 효율이 좋아질 뿐 아니라, 라벨이 없는 target 데이터에 대해 별도의 entropy 기반 정렬 손실을 정의할 수 있고, 추론 시에도 test 샘플의 hash code를 source 샘플의 hash code와 비교하여 보다 안정적으로 클래스를 결정할 수 있다는 것이다.

또한 이 논문은 새로운 벤치마크 데이터셋인 **Office-Home**을 함께 소개한다. 기존 Office, Office-Caltech 같은 도메인 적응 데이터셋은 deep learning 기반 적응 기법을 평가하기에는 작고 다양성이 부족하다는 문제의식이 있었고, Office-Home은 더 많은 카테고리와 더 큰 규모를 제공하여 이 한계를 보완하려고 한다.

정리하면, 이 논문의 목표는 두 가지이다. 첫째, source의 라벨과 target의 비라벨 데이터를 함께 사용해 target 분류 성능을 높이는 것, 둘째, 그 과정에서 retrieval에도 유용한 compact hash representation을 학습하는 것이다. 이는 실제 환경에서 “라벨은 다른 도메인에만 있고, 현재 도메인에는 라벨이 없으며, 저장과 검색 비용도 중요한 경우”를 겨냥한 문제 설정으로 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **도메인 적응을 위한 feature alignment**, **source에서의 supervised hashing**, **target에서의 unsupervised entropy minimization**을 하나의 deep CNN 안에서 동시에 최적화하는 것이다. 저자들은 이 모델을 **DAH (Domain Adaptive Hashing)** 라고 부른다.

직관적으로 보면, 좋은 hash code는 같은 클래스끼리는 비슷하고 다른 클래스끼리는 달라야 한다. source domain에서는 라벨이 있으므로 이 조건을 직접 줄 수 있다. 반면 target domain은 라벨이 없기 때문에, target 샘플이 source의 여러 클래스 중 **정확히 하나의 클래스와만 강하게 정렬되도록** 유도하는 방식이 필요하다. 저자들은 이를 위해 각 target 샘플이 각 source 클래스에 속할 확률을 정의하고, 그 확률 분포가 one-hot에 가까워지도록 **entropy loss**를 최소화한다.

기존 deep domain adaptation과의 차별점은 두 층위에서 드러난다. 첫째, 단순히 transferable feature를 학습하는 데서 그치지 않고, 마지막 표현을 **binary code 학습 문제**로 바꿨다. 둘째, target supervision이 없는 상황에서 hash code가 source 클래스 구조를 반영하도록 하기 위해, “분류 loss” 대신 **source class-level similarity를 이용한 entropy 기반 목표**를 설계했다. 논문은 이것이 기존 unsupervised hashing이나 일반 domain adaptation과 다른 새로운 결합이라고 주장한다.

또 하나의 중요한 차별점은 prediction 방식이다. 일반적인 분류기는 마지막 softmax 확률로 예측하지만, DAH는 hash representation을 중심으로 target 샘플을 source category들과 비교하여 분류한다. 즉, 최종 표현 자체가 classification과 retrieval 양쪽에 모두 쓰이는 representation이 된다.

## 3. 상세 방법 설명

### 전체 구조

모델은 기본적으로 VGG-F 기반 CNN 위에 구성된다. 네트워크는 conv1–conv5, fc6, fc7을 사용하고, 마지막 표준 fc8 대신 **hash-fc8** 층을 둔다. 이 층의 출력은 $d$차원 실수 벡터이며, 최종적으로 sign 함수를 적용해 $d$비트 hash code를 얻는다.

입력과 출력은 다음과 같이 정리된다.

* source 데이터: $\mathcal{D}_s={(\mathbf{x}_i^s, y_i^s)}_{i=1}^{n_s}$
* target 데이터: $\mathcal{D}_t={\mathbf{x}_i^t}_{i=1}^{n_t}$
* 네트워크 출력: $\psi(\mathbf{x}) \in \mathbb{R}^d$
* 이진 hash code: $\mathbf{h}=\operatorname{sgn}(\psi(\mathbf{x}))$

즉, 네트워크는 입력 이미지를 받아 compact continuous representation을 만들고, 이를 binary code로 양자화한다.

### 3.1 도메인 차이 감소: MK-MMD

도메인 적응을 위해 저자들은 fully connected 층들인 $\mathcal{F}={\textit{fc6}, \textit{fc7}, \textit{fc8}}$ 에서 source와 target의 표현 분포 차이를 줄인다. 사용한 도구는 **multi-kernel maximum mean discrepancy (MK-MMD)** 이다.

전체 MK-MMD 손실은 다음과 같다.

$$
\mathcal{M}(\mathcal{U}_s,\mathcal{U}_t)=\sum_{l\in\mathcal{F}} d_k^2(\mathcal{U}_s^l,\mathcal{U}_t^l)
$$

여기서 $\mathcal{U}_s^l$와 $\mathcal{U}_t^l$는 각 층 $l$에서 source와 target의 출력 표현 집합이다. 각 층의 MMD는 RKHS에서 평균 임베딩 간 거리로 정의된다.

$$
d_k^2(\mathcal{U}_s^l,\mathcal{U}_t^l)=\left| \mathbb{E}[\phi(\mathbf{u}^{s,l})]-\mathbb{E}[\phi(\mathbf{u}^{t,l})] \right|_{\mathcal{H}_k}^2
$$

쉽게 말하면, source와 target feature의 평균 구조가 커널 공간에서 비슷해지도록 만드는 항이다. 저자들은 단일 커널이 아니라 여러 PSD kernel의 convex combination을 써서 보다 안정적으로 분포 차이를 줄이려 한다. 이는 당시 DAN, RTN 등에서도 널리 사용되던 방식과 맥락이 같다.

중요한 점은 논문이 convolution layer는 비교적 generic하다고 보고, fully connected layer 쪽을 주된 adaptation 대상이라고 본다는 점이다. 따라서 fc6–fc8에서만 MK-MMD를 적용한다.

### 3.2 source용 supervised hash loss

source domain에는 라벨이 있으므로, 같은 클래스 샘플은 유사한 hash code를, 다른 클래스 샘플은 다른 hash code를 가지게 해야 한다. 이를 위해 논문은 pairwise similarity 기반의 supervised hashing objective를 사용한다.

두 이진 hash code $\mathbf{h}_i,\mathbf{h}_j \in {-1,+1}^d$에 대해, Hamming distance는 내적과 다음 관계를 가진다.

$$
\operatorname{dist}_H(\mathbf{h}_i,\mathbf{h}_j)=\frac{1}{2}(d-\mathbf{h}_i^\top \mathbf{h}_j)
$$

즉, 내적이 크면 Hamming distance가 작고, 따라서 더 비슷한 code이다. source 샘플 쌍에 대해 similarity label $s_{ij}\in{0,1}$를 정의하면, 같은 클래스일수록 $\mathbf{h}_i^\top \mathbf{h}_j$가 커져야 한다.

논문은 이를 likelihood로 바꾼다.

$$
p(s_{ij}|\mathbf{h}_i,\mathbf{h}_j)= \begin{cases} \sigma(\mathbf{h}_i^\top \mathbf{h}_j), & s_{ij}=1 \\ 1-\sigma(\mathbf{h}_i^\top \mathbf{h}_j), & s_{ij}=0 \end{cases}
$$

여기서 $\sigma(x)=\frac{1}{1+e^{-x}}$이다. 이 정의는 “같은 클래스면 내적이 커야 하고, 다른 클래스면 내적이 작아야 한다”는 직관을 그대로 probability model로 만든 것이다.

이에 따른 음의 로그 가능도는 다음과 같다.

$$
\mathcal{L}(\mathbf{H}) = -\sum_{s_{ij}\in\mathcal{S}} \left( s_{ij}\mathbf{h}_i^\top \mathbf{h}_j - \log(1+\exp(\mathbf{h}_i^\top \mathbf{h}_j)) \right)
$$

하지만 $\mathbf{h}_i$는 이진 벡터이므로 직접 최적화가 어렵다. 그래서 저자들은 이를 연속 출력 $\mathbf{u}_i=\psi(\mathbf{x}_i)\in\mathbb{R}^d$로 relaxation한다. 이때 tanh를 써서 출력이 $[-1,1]$ 범위에 머물게 하고, binarization 오차를 줄이기 위한 **quantization loss**를 추가한다.

최종 source hashing loss는 다음과 같다.

$$
\mathcal{L}(\mathcal{U}_s)= -\sum_{s_{ij}\in\mathcal{S}} \left( s_{ij}\mathbf{u}_i^\top \mathbf{u}_j - \log(1+\exp(\mathbf{u}_i^\top \mathbf{u}_j)) \right) + \sum_{i=1}^{n_s}|\mathbf{u}_i-\operatorname{sgn}(\mathbf{u}_i)|_2^2
$$

첫 번째 항은 클래스 구조를 보존하는 pairwise supervised hashing이고, 두 번째 항은 연속 출력이 실제 binary code에 가깝도록 만드는 regularizer이다.

### 3.3 target용 unsupervised entropy loss

target domain에는 라벨이 없으므로 source처럼 pairwise supervised objective를 줄 수 없다. 대신 논문은 각 target 출력 $\mathbf{u}_i^t$가 source의 어떤 클래스와 가까운지를 확률적으로 정의한다.

각 클래스 $j$에 대해 source의 해당 클래스 샘플 $K$개를 고르고, target 샘플이 클래스 $j$에 속할 확률을 다음과 같이 둔다.

$$
p_{ij} = \frac{\sum_{k=1}^{K}\exp\left((\mathbf{u}_i^t)^\top \mathbf{u}_k^{s_j}\right)} {\sum_{l=1}^{C}\sum_{k=1}^{K}\exp\left((\mathbf{u}_i^t)^\top \mathbf{u}_k^{s_l}\right)}
$$

이 식의 의미는 단순하다. target 샘플의 표현이 source의 특정 클래스에 속한 여러 샘플들과 비슷하면, 그 클래스의 확률이 커진다. 한 개의 prototype이 아니라 여러 source 샘플과의 유사도를 합친다는 점이 특징이다. 저자들은 이것이 더 robust한 category assignment를 만든다고 본다.

그 다음 확률 벡터 $\mathbf{p}_i=[p_{i1},\dots,p_{iC}]^\top$의 entropy를 줄인다.

$$
\mathcal{H}(\mathcal{U}_s,\mathcal{U}_t) = -\frac{1}{n_t} \sum_{i=1}^{n_t}\sum_{j=1}^{C} p_{ij}\log p_{ij}
$$

이 값을 최소화하면, 각 target 샘플의 클래스 확률 분포가 퍼지지 않고 one-hot에 가까워진다. 즉, target 샘플이 source 클래스 중 하나에만 뚜렷하게 정렬되도록 유도한다. 이것은 label이 없는 target에 대해 pseudo-label을 직접 생성하는 방식은 아니지만, 결과적으로는 low-entropy class assignment를 강제하는 것과 비슷한 효과를 가진다.

### 3.4 최종 목적함수

최종적으로 DAH는 세 손실을 함께 최적화한다.

$$
\mathcal{J} = \mathcal{L}(\mathcal{U}_s) + \gamma \mathcal{M}(\mathcal{U}_s,\mathcal{U}_t) + \eta \mathcal{H}(\mathcal{U}_s,\mathcal{U}_t)
$$

여기서,

* $\mathcal{L}$: source supervised hash loss
* $\mathcal{M}$: source-target distribution alignment를 위한 MK-MMD
* $\mathcal{H}$: target low-entropy alignment loss

$\gamma$는 도메인 정렬의 비중을, $\eta$는 target entropy loss의 비중을 조절한다.

이 구조의 해석은 매우 명확하다. source에서는 클래스 구조를 지키는 hash code를 배우고, target에서는 source 클래스 중 하나로 잘 정렬되게 만들며, 동시에 두 도메인의 표현 분포를 fully connected 층들에서 가깝게 만든다. 따라서 representation learning, domain adaptation, hashing이 모두 한 목적함수 안에 통합된다.

### 3.5 네트워크 및 학습 절차

논문은 ImageNet으로 사전학습된 VGG-F를 기반으로 fine-tuning한다. domain adaptation 상황은 데이터가 많지 않기 때문에 처음부터 깊은 CNN을 학습하지 않고 pretrained model을 적응시키는 전략을 택한다.

구체적으로 conv1–conv5, fc6, fc7은 fine-tuning하고, 마지막 fc8은 hash-fc8으로 교체한다. hash-fc8은 다음 구조를 가진다.

$$
\textit{hash-fc8} := {\textit{fc8} \rightarrow \textit{batch-norm} \rightarrow \tanh()}
$$

tanh는 출력을 $[-1,1]$ 범위로 제한해 hashing relaxation에 도움을 주지만, 포화 때문에 vanishing gradient 문제가 있을 수 있다. 이를 완화하려고 저자들은 tanh 앞에 batch normalization을 넣었다고 설명한다.

학습은 standard backpropagation으로 수행되며, supplementary material에는 각 손실의 미분식이 제시되어 있다. 다만 본문 수준에서 중요한 것은 세 손실 모두 미분 가능하도록 설계되었고, batch 단위로 함께 최적화된다는 점이다.

### 3.6 추론

학습이 끝난 뒤에는 target 샘플의 출력으로부터 hash code를 만들고, source category와의 유사도에 기반하여 확률 $p(y|\mathbf{h})$를 계산한다. 논문은 구현상 식 (6)의 구조를 사용해 class probability를 정의하고, 가장 큰 확률을 주는 클래스를 예측으로 선택한다.

즉, 마지막 예측은 단순 softmax classifier가 아니라 **source hash/code structure를 기준으로 한 category assignment**라고 보는 편이 정확하다.

## 4. 실험 및 결과

### 데이터셋

논문은 두 데이터셋에서 실험한다.

첫째는 기존 **Office** 데이터셋이다. Amazon (A), DSLR (D), Webcam (W) 3개 도메인으로 구성되고, 총 약 4,100장의 이미지와 31개 카테고리를 가진다. 저자들은 모든 source-target 조합에 대해 6개의 transfer task를 평가한다.

둘째는 논문이 새로 제안한 **Office-Home** 데이터셋이다. Art (Ar), Clipart (Cl), Product (Pr), Real-World (Rw)의 4개 도메인, 65개 카테고리, 총 약 15,500장 이미지로 이루어진다. 모든 source-target 조합을 고려하여 12개의 transfer task를 실험한다.

Office-Home이 중요한 이유는 도메인 적응용 객체 인식 벤치마크로서 더 큰 규모와 더 많은 클래스 수를 제공하기 때문이다. 논문은 기존 Office나 Office-Caltech이 deep adaptation 기법을 검증하기에 충분히 크지 않다고 지적한다.

### 구현 세부사항

DAH는 MatConvNet으로 구현되었다. conv1–conv5, fc6, fc7은 hash-fc8보다 10배 낮은 learning rate로 fine-tuning한다. 학습률은 $10^{-4}$에서 $10^{-5}$ 범위로 조절하고, 300 epoch 동안 학습하며, momentum은 0.9, weight decay는 $5\times10^{-4}$이다.

해시 길이는 기본적으로 $d=64$를 사용한다. target entropy loss의 가중치는 $\eta=1$이다. MK-MMD의 가중치 $\gamma$는 source와 target를 구분하는 binary domain classifier의 validation error가 가장 커지도록 선택했다고 적혀 있다. 이는 두 도메인이 가장 구분되지 않도록 만드는 heuristic이다.

또한 클래스당 $K=5$개의 source 샘플을 사용하여 target entropy probability를 정의한다. source similarity matrix에서 like-pair와 unlike-pair의 수가 불균형하므로, 유사한 pair의 가중치를 높이기 위해 similarity matrix 값은 ${0,10}$으로 설정했다고 보고한다.

### 4.1 비지도 도메인 적응 성능

비교 대상은 shallow adaptation 방법인 GFK, TCA, CORAL, JDA와 deep adaptation 방법인 DAN, DANN이다. 추가로 DAH에서 entropy loss를 제거한 변형인 **DAH-e**도 비교한다.

#### Office 결과

Office 데이터셋의 평균 정확도는 다음과 같다.

* GFK: 62.32
* TCA: 64.54
* CORAL: 66.04
* JDA: 69.37
* DAN: 72.13
* DANN: 75.15
* DAH-e: 72.31
* DAH: 73.04

이 결과에서 DAH는 shallow baselines보다 확실히 좋고 DAN보다도 약간 높지만, **DANN보다는 낮다**. 논문도 이를 정직하게 인정하며, 클래스 수가 적은 Office에서는 domain adversarial training이 더 효과적일 수 있다고 해석한다.

#### Office-Home 결과

Office-Home에서 평균 정확도는 다음과 같다.

* GFK: 32.40
* TCA: 30.34
* CORAL: 37.91
* JDA: 36.97
* DAN: 43.46
* DANN: 44.94
* DAH-e: 42.69
* DAH: 45.54

여기서는 **DAH가 가장 높은 평균 정확도**를 기록한다. 특히 카테고리 수가 65개로 많은 환경에서, 단순히 도메인을 섞는 것보다 클래스 구조를 반영하는 hash-based alignment가 더 유리하다는 것이 저자들의 해석이다.

DAH와 DAH-e를 비교하면, DAH가 평균적으로 더 좋다. 이는 target entropy loss가 실제로 target 샘플을 source category 구조에 더 잘 정렬시키며, 성능 향상에 기여한다는 근거로 제시된다.

### 4.2 feature analysis

논문은 fc7 출력의 quality도 분석한다. 사용한 도구는 두 가지이다.

첫째는 **$\mathcal{A}$-distance** 이다. 이는 source와 target를 구분하는 이진 분류기의 일반화 오차 $\epsilon$로부터 근사적으로 $2(1-2\epsilon)$ 형태로 계산되는 도메인 discrepancy 측정치이다. 값이 작을수록 두 도메인이 더 잘 정렬된 것으로 볼 수 있다.

둘째는 **t-SNE visualization** 이다. Art와 Clipart의 10개 카테고리에 대해 Deep feature, DAN feature, DAH feature를 시각화했다.

논문에 따르면 DAH feature가 Deep, DAN보다 더 작은 도메인 discrepancy를 보였고, t-SNE에서도 source와 target 간 overlap과 category clustering이 가장 잘 나타났다. 이는 DAH가 단순히 분류 정확도만 올린 것이 아니라, 실제 representation space에서도 더 잘 정렬된 구조를 만든다는 정성적 증거로 제시된다.

다만 제공된 텍스트에는 Figure 3의 정확한 수치값은 포함되어 있지 않으므로, 이 부분은 **정성적 비교 중심으로만 해석하는 것이 적절**하다.

### 4.3 Unsupervised Domain Adaptive Hashing 성능

이 논문은 분류뿐 아니라 hashing 자체의 품질도 평가한다. 여기서 핵심 질문은 “target에 라벨이 없는 상황에서, 다른 관련 도메인의 라벨을 활용하여 좋은 hash code를 만들 수 있는가?”이다.

비교 시나리오는 네 가지이다.

첫째, 완전 비지도 hashing이다. ITQ, KMeans hashing, BA, BDNN과 비교한다.
둘째, source 라벨만 사용하지만 domain adaptation은 하지 않는 **NoDA**이다. 이는 DPSH를 source에 학습한 뒤 target에 그대로 적용한 경우다.
셋째, 본 논문의 **DAH**이다.
넷째, target 라벨을 직접 사용하는 supervised hashing **SuH**이다. 이는 upper bound 역할을 한다.

평가는 precision-recall curve와 mean average precision (mAP)으로 수행된다.

#### mAP @64 bits

표 4의 평균 mAP는 다음과 같다.

* NoDA: 0.278
* ITQ: 0.370
* KMeans: 0.322
* BA: 0.301
* BDNN: 0.382
* DAH: 0.480
* SuH: 0.707

DAH는 모든 비지도/준비지도 방식보다 높은 평균 mAP를 보인다. 특히 source에서 학습한 모델을 그대로 target에 적용하는 NoDA보다 큰 폭으로 낫다. 이는 **라벨이 다른 도메인에만 있을 때, domain mismatch를 무시하면 hashing 성능이 매우 나빠진다**는 점을 보여준다.

흥미로운 부분은 DAH가 완전 비지도 hashing 기법들보다도 우수하다는 점이다. 이는 source 라벨 정보를 domain adaptation과 함께 사용하는 것이, target 내부 구조만 보는 비지도 hashing보다 훨씬 유리하다는 뜻이다. 물론 target 라벨을 직접 쓰는 SuH가 여전히 가장 높지만, DAH는 그 upper bound에 가장 근접한 비지도 적응 기반 방법으로 제시된다.

### 4.4 supplementary 결과: 해시 길이 변화

추가 실험에서 저자들은 $d=16$, $64$, $128$ bit를 비교했다.

Office-Home 분류 정확도의 평균은 다음과 같다.

* DAH-16: 31.36
* DAH-64: 45.54
* DAH-128: 46.26

즉, 16비트는 표현력이 부족해 성능이 크게 떨어지고, 128비트는 64비트보다 소폭 좋아진다. 다만 hashing mAP에서는 bit 수가 늘었다고 항상 좋아지지는 않는다고 보고한다. 실제로 supplementary의 설명에는 128비트에서 mAP가 무조건 개선되지 않는다고 명시되어 있다. 이는 hashing에서 **code length 증가가 retrieval 품질 증가를 자동 보장하지 않는다**는 점을 보여준다.

또한 특정 경우, 예를 들어 Real-World의 128비트 설정에서는 DAH가 SuH보다 더 좋았다고 언급한다. 저자들은 이를 domain adaptation이 때로는 더 generalizable한 representation을 학습하게 만들 수 있다는 흥미로운 신호로 해석한다. 다만 이는 일부 사례에 대한 관찰이며, 일반 법칙으로 확대하는 것은 조심해야 한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정의 실용성이다. 실제 환경에서는 현재 도메인에 라벨이 없고, 유사하지만 다른 도메인에는 라벨이 있으며, 동시에 저장 및 검색 비용도 중요할 수 있다. DAH는 바로 그 상황을 겨냥해 **분류와 해싱을 동시에 해결하는 통합 모델**을 제시한다. 이는 단지 성능 숫자를 조금 올린 것보다 더 큰 기여로 볼 수 있다.

두 번째 강점은 목적함수 설계가 명확하다는 점이다. source의 supervised pairwise hashing, target의 entropy minimization, 그리고 source-target distribution alignment가 서로 역할이 구분되어 있다. 각 손실이 왜 필요한지 설명도 비교적 설득력 있다. 특히 target entropy loss는 라벨 없는 target을 source category 구조에 붙이기 위한 장치로서 논문의 핵심 설계 중 하나다.

세 번째 강점은 Office-Home 데이터셋의 도입이다. 이후 domain adaptation 연구에서 Office-Home은 매우 널리 쓰이는 벤치마크가 되었기 때문에, 데이터셋 기여만으로도 이 논문의 영향력은 상당하다. 제공된 본문만 보더라도 저자들은 기존 벤치마크의 소규모 문제를 분명히 인식하고 있으며, 더 현실적인 규모의 데이터셋을 만들려 했다.

반면 한계도 분명하다. 첫째, target entropy minimization은 각 target 샘플이 source의 어느 한 클래스에 명확히 속한다는 가정을 은근히 내포한다. 즉, **target label space가 source label space와 완전히 공유된다**는 전제가 사실상 필요하다. open-set adaptation이나 partial domain adaptation 같은 더 어려운 상황에서는 그대로 적용되기 어렵다. 논문 텍스트에는 이러한 확장 설정에 대한 논의는 없다.

둘째, target category probability $p_{ij}$를 계산할 때 각 클래스당 $K$개의 source 샘플을 사용한다. 이 방식은 직관적이지만, 샘플 선택 방식이나 클래스 내부 다양성에 따라 민감할 수 있다. 논문은 $K=5$를 사용했다고만 설명하며, 이 선택의 민감도 분석은 제공된 텍스트 안에서는 충분하지 않다.

셋째, 분류 측면에서는 Office에서 DANN을 넘지 못한다. 즉, 제안 방법이 모든 상황에서 state of the art라고 말하기는 어렵다. 논문도 클래스 수가 적은 환경에서는 adversarial alignment가 더 유리할 수 있다고 해석한다. 따라서 DAH의 강점은 “보편적 우월성”보다는 “클래스 수가 많고 hashing이 중요한 환경에서의 장점”으로 이해하는 편이 더 정확하다.

넷째, hashing의 장점을 강조하지만, 실제 deployment 관점에서 hash code를 사용한 분류가 softmax보다 얼마나 더 robust한지에 대한 이론적 분석은 강하지 않다. 본문은 직관과 실험으로 설득하지만, 왜 hash-based category assignment가 어떤 조건에서 특히 유리한지에 대한 더 깊은 분석은 부족하다.

다섯째, 제공된 텍스트 기준으로는 계산 비용과 메모리 효율의 실제 절감량, 혹은 inference latency 비교는 제시되지 않는다. hashing을 제안하는 중요한 동기 중 하나가 storage/retrieval efficiency인데, 실제 시스템 관점의 정량 비교는 보이지 않는다. 따라서 “효율성”의 주장은 개념적으로는 타당하지만, 본문에 나타난 실험은 주로 정확도와 mAP 중심이다.

비판적으로 보면, DAH는 매우 흥미로운 결합이지만, 현대적 관점에서는 target pseudo-labeling, contrastive learning, prototype alignment, adversarial/domain confusion 등과 비교했을 때 더 정교한 클래스 수준 alignment 방법들이 이후 많이 등장했다. 그럼에도 당시 시점에서는 hashing과 domain adaptation을 깊은 네트워크 안에서 결합했다는 점이 분명한 참신성으로 읽힌다.

## 6. 결론

이 논문은 **deep hashing**과 **unsupervised domain adaptation**을 통합한 DAH를 제안하고, source 라벨과 target 비라벨 데이터를 함께 사용하여 target 분류와 target hash code 학습을 동시에 수행했다. 핵심은 source에서는 supervised pairwise hashing으로 클래스 구조를 학습하고, target에서는 entropy minimization으로 source 클래스 중 하나에 뚜렷하게 정렬되도록 하며, 중간 표현에서는 MK-MMD로 도메인 차이를 줄이는 것이다.

실험적으로는 Office-Home에서 강력한 성능을 보였고, hashing 품질 평가에서도 NoDA나 일반 비지도 hashing보다 더 높은 mAP를 기록했다. 또한 Office-Home 데이터셋 자체를 새로 제안한 점도 이 연구의 중요한 공헌이다.

실제 적용 측면에서 이 연구는 “현재 도메인에는 라벨이 없지만, 관련 도메인의 라벨은 존재하고, compact representation도 필요한 상황”에 특히 의미가 있다. 예를 들어 대규모 이미지 저장, 검색, 분류를 동시에 고려해야 하는 시스템에서 유용할 가능성이 있다. 향후 연구로는 open-set/partial-set domain adaptation, 더 강한 클래스 수준 prototype 학습, 혹은 contrastive objective와의 결합 등이 자연스러운 확장 방향으로 보인다.

전체적으로 이 논문은 완전히 모든 실험 조건에서 최고 성능을 보인다고 말할 수는 없지만, **문제 정의의 실용성, 목적함수 설계의 일관성, Office-Home 데이터셋의 기여** 때문에 학술적 가치가 충분한 논문으로 평가할 수 있다. 특히 domain adaptation을 단순 feature alignment가 아니라 **compact discriminative representation learning**의 관점으로 확장했다는 점에서 의미가 크다.
