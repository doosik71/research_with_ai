# Contrastive Domain Adaptation

- **저자**: Mamatha Thota (University of Lincoln), Georgios Leontidis (University of Aberdeen)
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2103.15566

## 1. 논문 개요

이 논문은 contrastive self-supervised learning을 domain adaptation 문제에 적용하는 새로운 프레임워크인 **Contrastive Domain Adaptation (CDA)**을 제안한다. 핵심 연구 문제는 다음과 같다: 소스 도메인과 타겟 도메인 양쪽 모두에 레이블이 없고, ImageNet 사전학습 가중치도 사용하지 않는 완전 비지도(fully unsupervised) 설정에서, 서로 다른 확률 분포를 따르는 두 도메인 간에 일반화 가능한 표현(representation)을 학습할 수 있는가?

기존의 대부분의 Unsupervised Domain Adaptation(UDA) 방법들은 소스 도메인의 레이블 정보에 의존하거나, ImageNet으로 사전학습된 backbone을 전제로 한다. 이는 레이블 획득 비용이 높거나 사전학습 환경을 갖추기 어려운 현실적인 시나리오에서 적용에 제약이 있다. 이 논문은 레이블이 전혀 없는 소스 및 타겟 데이터만을 활용하여 도메인 불변(domain-invariant)이고 클래스 구별(class-discriminative)이 가능한 특징 표현을 학습하는 접근법을 탐구한다는 점에서 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심적인 직관은, contrastive learning이 의미적으로 유사한 샘플을 가깝게, 상이한 샘플을 멀리 밀어내는 방식으로 학습하므로, 이를 domain adaptation에 적절히 확장하면 도메인 간 시각적 차이(domain shift)에도 불구하고 도메인 불변 표현을 학습할 수 있다는 것이다.

이를 위해 논문은 세 가지 설계 아이디어를 결합한다. 첫째, contrastive loss를 소스와 타겟 도메인 각각에 독립적으로 적용하여 서로 다른 도메인의 샘플이 섞여 생기는 false negative 문제와 도메인 간 거리 확대 문제를 방지한다. 둘째, 레이블 없이 false negative로 의심되는 negative 쌍을 유사도 기반으로 식별하고 제거하는 **False Negative Removal (FNR)** 전략을 도입한다. 셋째, **Maximum Mean Discrepancy (MMD)**를 손실 함수에 추가하여 소스와 타겟 특징 분포 간의 거리를 명시적으로 최소화한다.

기존 contrastive learning 기반 UDA 방법들(예: CAN [18], JCL [32])과의 주요 차별점은, 이들이 소스 레이블이나 ImageNet 사전학습 가중치에 의존하는 반면, CDA는 그 어떤 레이블도 사전학습 가중치도 사용하지 않는다는 점이다.

## 3. 상세 방법 설명

### 전체 파이프라인

CDA는 ResNet-50을 base encoder $f(\cdot)$로, 2개의 hidden layer를 가진 비선형 MLP를 projection head $g(\cdot)$로 사용하며, 두 네트워크 모두 완전히 무작위 초기화(random initialization)에서 학습된다. 학습이 완료된 후에는 projection head를 제거하고, encoder의 표현 위에 선형 분류기(linear classifier)를 올려 소스 레이블만으로 downstream 평가를 수행한다.

미니배치 내 각 소스 이미지 $x^s$에 대해 두 가지 augmentation을 생성하여 anchor 쌍 $a_1^s, a_2^s$를 만들고, encoder와 projection head를 통과시켜 표현 벡터 $z_1^s = g(f(a_1^s))$, $z_2^s = g(f(a_2^s))$를 얻는다. 타겟 도메인에 대해서도 동일한 과정을 독립적으로 수행한다.

### Contrastive Loss for DA

표준 NT-Xent loss는 다음과 같이 정의된다:

$$L_{CONT} = -\log \frac{\exp(\text{sim}(z_i, z_j)/T)}{\sum_{k=1}^{2N} \mathbf{1}_{(k \neq i)} \exp(\text{sim}(z_i, z_k)/T)}$$

여기서 $\text{sim}(u, v) = u^T v / \|u\| \|v\|$는 cosine similarity이고, $T$는 temperature 파라미터이다.

만약 이 손실을 소스와 타겟 이미지가 혼재한 미니배치에 그대로 적용하면, 다른 도메인에서 온 동일 클래스 이미지가 negative로 처리되어 두 도메인의 표현이 오히려 멀어지는 문제가 발생한다. 이를 방지하기 위해 논문은 소스와 타겟 각각에 대해 독립적으로 contrastive loss를 계산하고 합산한다:

$$L_{CONT\_DA} = L_{CONT\_S} + L_{CONT\_T}$$

### False Negative Removal (FNR)

레이블이 없는 상황에서 false negative란, 미니배치 내에서 anchor와 다른 이미지로 취급되지만 실제로는 같은 클래스인 샘플을 말한다. 이러한 false negative는 contrastive loss 계산 시 모순된 gradient를 발생시켜 수렴을 방해한다.

제안된 방법은 각 anchor $i$에 대해 모든 negative 쌍의 유사도를 계산한 후, 유사도가 가장 높은 상위 $k$개의 negative를 제거하는 방식이다. 이를 반영한 손실 함수는 다음과 같다:

$$L_{FNR} = -\log \frac{\exp(\text{sim}(z_i, z_j)/T)}{\sum_{k=1}^{2N} \mathbf{1}_{(k \neq i, k \notin S_i)} \exp(\text{sim}(z_i, z_k)/T)}$$

여기서 $S_i$는 anchor $i$와 유사하다고 판단되어 제거된 negative 샘플의 집합이다. 이를 도메인 적응 설정에 맞게 소스와 타겟 각각에 적용하면:

$$L_{FNR\_DA} = L_{FNR\_S} + L_{FNR\_T}$$

실험에서는 배치 크기 512 기준으로 anchor당 1개(FNR₁) 또는 2개(FNR₂)의 false negative를 제거하는 두 가지 설정을 테스트하였다. 이 방법의 장점은 추가적인 support view나 별도 네트워크 없이 현재 미니배치의 유사도 정보만으로 작동하기 때문에 추가적인 계산 비용이 발생하지 않는다는 것이다.

### Maximum Mean Discrepancy (MMD)

도메인 정렬을 위해 MMD를 특징 공간에서 소스와 타겟 분포 간의 거리를 측정하는 추가 손실로 활용한다. MMD의 제곱 추정치는 다음과 같이 계산된다:

$$L_{MMD} = \frac{1}{N^2} \sum_{i,i'} k(x_i^s, x_{i'}^s) - \frac{2}{NM} \sum_{i,j} k(x_i^s, x_j^t) + \frac{1}{M^2} \sum_{j,j'} k(x_j^t, x_{j'}^t)$$

여기서 $k(\cdot, \cdot)$은 Reproducing Kernel Hilbert Space (RKHS)에 대응하는 universal kernel이며, $N$과 $M$은 각각 소스와 타겟 샘플 수이다. 직관적으로는, 두 분포가 동일하다면 고차원 특징 공간에서의 평균 임베딩이 일치해야 한다는 원리를 이용한 것이다.

최종 학습 시에는 $L_{FNR\_DA}$와 $L_{MMD}$를 함께 역전파하여 encoder와 projection head를 업데이트한다.

### 4-View 확장

Contrastive Multiview Coding [38]에서 영감을 받아, anchor당 augmentation을 2개에서 4개로 늘리는 실험도 수행하였다(CDAx4aug). 이 경우 소스에서 2쌍, 타겟에서 2쌍의 contrastive loss를 모두 역전파한다.

## 4. 실험 및 결과

### 데이터셋 및 평가

실험은 세 가지 표준 digit domain adaptation 벤치마크에서 수행되었다. MNIST→USPS (M→U)는 도메인 이동이 비교적 작은 설정이며, SVHN→MNIST (S→M)는 컬러 street view 숫자에서 흑백 손글씨로의 큰 분포 차이를 가지며, MNIST→MNISTM (M→MM)은 MNIST 숫자를 컬러 배경에 blending한 타겟 도메인으로 역시 상당한 도메인 이동이 존재한다. 평가 지표는 타겟 도메인에서의 분류 정확도(accuracy)이며, 소스 레이블만으로 학습한 선형 분류기를 사용한다.

### 주요 정량 결과

SimClr-Base(소스만으로 학습 후 타겟에 적용)는 M→U/S→M/M→MM에서 각각 92.0/31.7/34.9%로 평균 53.1%에 불과하여 도메인 이동에 취약함을 보인다.

도메인별 독립 contrastive loss를 적용한 CDA-Base는 평균 71.7%로 약 19% 향상되었으며, FNR을 추가한 CDA_FNR1(74.0%)과 CDA_FNR2(75.5%)는 각각 평균 2.3%, 3.8% 추가 향상을 보였다. MMD를 추가한 CDA-MMD는 평균 76.2%, FNR과 MMD를 모두 결합한 CDA_FNR-MMD는 평균 76.8%로 CDA-Base 대비 5.1% 향상을 달성하였다.

4-view 설정인 CDAx4aug는 평균 76.8%, CDAx4aug_FNR은 평균 77.5%로, 추가 뷰가 학습에 도움이 됨을 보였다. 그러나 4-view에 MMD를 결합한 경우(CDAx4aug-MMD, CDAx4aug_FNR-MMD)에는 추가 augmentation으로 인한 노이즈가 MMD 수렴을 방해하여 2-view 대비 성능 향상이 제한적이었다.

SOTA 비교(Table 3) 측면에서, 소스 레이블을 사용하는 DANN(73.8%, S→M), DAN(71.1%), ADDA(76.0%)와 비교하여 CDA_FNR-MMD는 S→M에서 76.2%로 경쟁력 있는 성능을 보였다. M→U에서는 94.2%로 모든 비교 방법을 상회하였다. 다만 M→MM에서는 DANN(76.6%), DAN(76.9%) 대비 60.2%로 낮은 수치를 보이는데, 이는 레이블 없이 학습하는 본 방법의 한계로 해석된다.

## 5. 강점, 한계

### 강점

논문에서 실험적으로 뒷받침되는 주요 강점은 세 가지다. 첫째, 레이블과 ImageNet 사전학습 모두 없이도 기존 지도학습 기반 UDA 방법들과 경쟁 가능한 성능을 보인다는 점에서 실용적 가치가 있다. 둘째, FNR이 추가 계산 비용 없이 일관된 성능 향상을 가져온다. 셋째, 도메인별 독립 contrastive loss라는 단순한 수정만으로 SimClr-Base 대비 큰 성능 격차를 만들어내는 설계가 효율적이다.

### 한계 및 비판적 해석

몇 가지 중요한 한계가 존재한다. 첫째, 실험이 단순한 digit 데이터셋에 한정되어 있어 실제 고해상도 이미지(예: Office-31, DomainNet 등)에서의 일반화 가능성이 검증되지 않았다. 둘째, M→MM에서 레이블을 사용하는 방법들에 비해 성능 차이가 크게 벌어지는데, 이는 도메인 이동이 클 때 레이블 없이 도메인 불변 표현을 학습하는 것이 여전히 어렵다는 것을 시사한다. 셋째, 4-view에서 MMD 결합이 오히려 성능을 저하시키는 현상은 단순히 "augmentation 노이즈" 탓으로만 설명하고 있어, 이에 대한 심층 분석이 부족하다. 넷째, false negative 판단 기준으로 유사도 상위 $k$개를 단순 제거하는 방식은 실제 false negative와 hard negative를 구분하지 못할 수 있으며, 최적의 $k$ 값 선택에 대한 이론적 근거가 제시되지 않았다.

## 6. 결론

이 논문은 contrastive learning을 완전 비지도 domain adaptation 설정으로 확장하는 CDA 프레임워크를 제안하며, (1) 도메인별 독립 contrastive loss, (2) false negative removal, (3) MMD 기반 분포 정렬, (4) multi-view 학습 확장이라는 네 가지 기여를 제시한다.

실용적 관점에서, 이 연구는 레이블 획득이 어렵거나 소스 데이터 레이블 자체에 접근이 제한된 시나리오(예: 의료 영상, 위성 데이터 분석)에서 가치 있는 방향을 제시한다. 향후 연구 측면에서는 대규모 이미지 데이터셋에 대한 확장 검증, false negative 판별의 이론적 정교화, 그리고 multi-view와 MMD 결합 시 발생하는 수렴 문제의 해결이 중요한 과제로 남아 있다.
