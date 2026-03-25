# MADAN: Multi-source Adversarial Domain Aggregation Network for Domain Adaptation

* **저자**: Sicheng Zhao, Bo Li, Xiangyu Yue, Pengfei Xu, Kurt Keutzer
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2003.00820](https://arxiv.org/abs/2003.00820)

## 1. 논문 개요

이 논문은 **multi-source domain adaptation (MDA)** 문제를 다룬다. 즉, 라벨이 있는 여러 개의 source domain과 라벨이 없거나 매우 적은 하나의 target domain이 있을 때, source와 target 사이의 domain shift를 줄여 target 성능을 높이는 것이 목표다. 기존의 많은 domain adaptation 연구는 단일 source를 가정하거나, 여러 source가 있어도 이를 단순히 합쳐서 하나의 source처럼 취급했다. 그러나 실제로는 서로 다른 source들 사이에도 분포 차이가 존재하기 때문에, 이들을 무작정 합치면 오히려 서로 간섭하여 학습이 비효율적이거나 suboptimal해질 수 있다.

논문이 문제 삼는 핵심은 두 가지다. 첫째, 기존 multi-source 방법들은 주로 **feature-level alignment**에만 집중하여 고수준 특징 분포만 맞추는데, 이는 semantic segmentation처럼 픽셀 단위 예측이 필요한 정밀한 작업에는 부족하다. 둘째, 기존 방법들은 각 source와 target의 쌍별 정렬만 고려하고, **여러 source들끼리의 misalignment**는 충분히 다루지 않는다. 즉, 각 source가 target 쪽으로 어느 정도 이동하더라도 source들끼리는 여전히 서로 다른 방향에 놓일 수 있다.

이 문제는 특히 자율주행용 synthetic-to-real adaptation 같은 실제 응용에서 중요하다. 예를 들어 GTA, SYNTHIA 같은 synthetic 데이터는 라벨링이 쉽지만, Cityscapes나 BDDS 같은 real-world 데이터는 라벨링 비용이 매우 높다. 따라서 여러 synthetic source를 잘 활용해 real target에 적응시키는 것은 실용적 가치가 크다.

이 논문은 이를 위해 **MADAN (Multi-source Adversarial Domain Aggregation Network)** 을 제안한다. 핵심 방향은 다음과 같다. 각 source를 target 스타일에 맞게 **pixel-level로 translation**하여 adapted domain을 만들고, 이렇게 만들어진 여러 adapted domain들을 다시 서로 가깝게 **aggregation**하며, 마지막으로 aggregated domain과 target 사이를 **feature-level로 정렬**한다. segmentation에서는 여기에 더해 **category-level alignment**와 **context-aware generation**을 추가한 **MADAN+** 를 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은, multi-source adaptation을 잘 하려면 단순히 feature space에서만 source-target을 맞추는 것으로는 부족하고, 먼저 각 source를 target처럼 보이게 만드는 **중간 adapted domain**을 만든 뒤, 그 adapted domain들 자체를 하나의 더 통일된 분포로 모아야 한다는 점이다.

기존 방법과의 차별점은 크게 세 가지다.

첫째, **pixel-level alignment**를 본격적으로 multi-source setting에 도입했다. 각 source마다 target 스타일로 이미지를 변환하는 generator를 두고, CycleGAN류의 cycle consistency를 사용해 source 이미지의 시각적 appearance를 target에 가깝게 바꾼다. 이를 통해 단순 feature matching보다 더 직접적으로 domain gap을 줄인다.

둘째, source-target 정렬만으로 끝나지 않고, 서로 다른 source에서 온 adapted domain들끼리도 더 비슷해지도록 **domain aggregation**을 수행한다. 이를 위해 논문은 **Sub-domain Aggregation Discriminator (SAD)** 와 **Cross-domain Cycle Discriminator (CCD)** 를 제안한다. SAD는 adapted domains 간 구분이 어렵도록 만들고, CCD는 한 source에서 target으로 갔다가 다시 다른 source 쪽으로 돌아오는 경로를 이용해 adapted domain 간 일관성을 더 강하게 요구한다.

셋째, semantic consistency를 정적으로 보지 않고 **Dynamic Semantic Consistency (DSC)** 로 바꾼다. 기존 CyCADA류에서는 source 이미지와 translated 이미지를 같은 task model에 통과시켜 의미가 보존되도록 강제했다. 그러나 translated image는 이미 target-like domain에 가까워졌기 때문에 source 전용 task model로 보는 것이 적절하지 않을 수 있다. 이에 저자들은 현재 학습 중인 task model을 사용해 translated image의 semantics를 비교하게 하여, semantic consistency 기준이 점진적으로 source domain에서 target-adapted domain 쪽으로 이동하도록 설계했다. 이 점이 이 논문의 이론적·실용적 핵심 중 하나다.

segmentation용 MADAN+의 추가 아이디어도 분명하다. 글로벌 feature 정렬만으로는 클래스별 출현 빈도 불균형을 다루기 어렵기 때문에 **category-level alignment (CLA)** 를 넣고, image translation 단계에서는 전역 의미와 지역 detail을 동시에 보존하기 위해 **context-aware generation (CAG)** 을 도입한다. 특히 segmentation은 경계, 작은 물체, 위치 구조가 중요하므로 이 확장이 설득력 있다.

## 3. 상세 방법 설명

### 3.1 문제 설정

논문은 여러 개의 labeled source domain $S_1, S_2, \dots, S_M$ 과 하나의 unlabeled target domain $T$ 를 가정한다. 각 source domain $S_i$ 에서는 입력과 라벨 쌍 $(X_i, Y_i)$ 가 주어지고, target domain에서는 입력 $X_T$ 만 주어진다. 모든 도메인은 같은 입력 공간과 같은 라벨 공간을 공유한다고 가정한다. 즉, homogeneous setting과 closed-set setting이다.

목표는 $(X_i, Y_i)_{i=1}^M$ 와 $X_T$ 만 사용하여, target 샘플의 라벨을 잘 예측하는 모델 $F$ 를 학습하는 것이다.

### 3.2 전체 구조

MADAN은 세 구성요소로 이루어진다.

첫 번째는 **Dynamic Adversarial Image Generation (DAIG)** 이다. 각 source 이미지를 target처럼 보이게 translation하여 pixel-level alignment를 수행한다.

두 번째는 **Adversarial Domain Aggregation (ADA)** 이다. 서로 다른 source에서 생성된 adapted domains가 서로 더 비슷한 하나의 aggregated domain이 되도록 만든다.

세 번째는 **Feature-aligned Task Learning (FTL)** 이다. aggregated adapted domain의 라벨을 이용해 task network를 학습하고, 동시에 feature-level에서 aggregated domain과 target을 adversarial하게 정렬한다.

직관적으로 보면, 먼저 “각 source를 target처럼 보이게” 만들고, 그 다음 “이 adapted source들끼리도 서로 비슷하게” 만들며, 마지막으로 “그 위에서 task model이 target feature에도 잘 작동하도록” 학습하는 구조다.

### 3.3 Dynamic Adversarial Image Generation (DAIG)

각 source $S_i$ 에 대해 generator $G_{S_i \rightarrow T}$ 를 두어 source image를 target 스타일로 변환한다. 그리고 discriminator $D_T$ 는 진짜 target 이미지와 translated source 이미지를 구분하도록 학습된다. 논문에 적힌 GAN loss는 다음 형태다.

$$
\mathcal{L}_{GAN}^{S_i \rightarrow T}(G_{S_i \rightarrow T}, D_T, X_i, X_T) = \mathbb{E}_{\mathbf{x}_i \sim X_i} \log D_T(G_{S_i \rightarrow T}(\mathbf{x}_i)) + \mathbb{E}_{\mathbf{x}_T \sim X_T} \log (1 - D_T(\mathbf{x}_T))
$$

표기 방향은 일반적인 GAN 식과 조금 다르게 보일 수 있으나, 요지는 **translated source가 target처럼 보이도록** generator와 discriminator를 adversarial하게 학습한다는 것이다.

또한 source에서 target으로 가는 매핑만 있으면 문제가 under-constrained 되므로, 역방향 generator $G_{T \rightarrow S_i}$ 와 source discriminator $D_i$ 도 둔다. 그리고 두 방향이 서로 모순되지 않도록 **cycle consistency loss** 를 둔다.

$$
\mathcal{L}_{cyc}^{S_i \leftrightarrow T} = \mathbb{E}_{\mathbf{x}_i \sim X_i} \left| G_{T \rightarrow S_i}(G_{S_i \rightarrow T}(\mathbf{x}_i)) - \mathbf{x}_i \right|_1 + \mathbb{E}_{\mathbf{x}_T \sim X_T} \left| G_{S_i \rightarrow T}(G_{T \rightarrow S_i}(\mathbf{x}_T)) - \mathbf{x}_T \right|_1
$$

이 식은 source를 target으로 갔다가 다시 source로 돌아오면 원본과 비슷해야 하고, 반대로 target을 source로 갔다가 다시 target으로 돌아와도 원본과 비슷해야 함을 뜻한다. 이렇게 해야 translation이 단순히 무작위 스타일 변경이 아니라 구조를 어느 정도 보존한다.

### 3.4 Dynamic Semantic Consistency (DSC)

논문이 특히 강조하는 부분이다. 단순 cycle consistency만으로는 semantic 정보 보존이 충분하지 않다. 예를 들어 segmentation에서는 자동차가 하늘로 바뀌면 안 된다. 기존 CyCADA는 source 이미지와 translated 이미지를 같은 pretrained source task model에 넣고 semantic consistency를 강제했다.

하지만 저자들은 이것이 완전히 적절하지 않다고 본다. translated image는 source 스타일이 아니라 target 스타일에 더 가까우므로, source 전용 모델 $F_i$ 로 그 의미를 해석하는 것은 translation을 방해할 수 있다. 이상적으로는 target용 모델 $F_T$ 가 필요하지만, target 라벨이 없으므로 불가능하다.

그래서 논문은 현재 adapted domain에서 학습 중인 task model $F$ 자체를 $F_A$ 로 사용한다. 즉, source 원본 $\mathbf{x}_i$ 는 source-pretrained model $F_i$ 로 보고, translated image $G_{S_i \rightarrow T}(\mathbf{x}_i)$ 는 현재 학습 중인 adapted-domain task model $F_A = F$ 로 본다. 두 예측 분포가 비슷해지도록 KL divergence를 최소화한다.

$$
\mathcal{L}_{DSC}^{S_i}(G_{S_i \rightarrow T}, X_i, F_i, F_A) = \mathbb{E}_{\mathbf{x}_i \sim X_i} KL\left( F_A(G_{S_i \rightarrow T}(\mathbf{x}_i)) \;|\; F_i(\mathbf{x}_i) \right)
$$

쉽게 말해, “source 이미지가 가진 의미를 번역된 이미지도 유지하되, 그 의미 판단 기준은 점점 target-adapted model 쪽으로 이동”하게 한다. 논문은 이것이 두 가지 이점을 가진다고 말한다. 첫째, generator가 더 target-like 하면서도 semantic을 보존하는 이미지를 만들도록 유도한다. 둘째, 별도 모델을 추가하지 않고 task model 파라미터를 공유할 수 있어 효율적이다.

### 3.5 Adversarial Domain Aggregation (ADA)

여러 source를 각각 target으로 translation하면, 각 source는 target 쪽으로는 가까워질 수 있지만 adapted domain들끼리는 여전히 다를 수 있다. 예를 들어 Source A에서 온 adapted image들과 Source B에서 온 adapted image들이 target 스타일 일부는 공유해도 서로 다른 artifact를 남길 수 있다. 이를 해결하기 위해 domain aggregation을 수행한다.

#### 3.5.1 Sub-domain Aggregation Discriminator (SAD)

각 source $S_i$ 에 대해 discriminator $D_A^i$ 를 두어, “$S_i$ 에서 온 adapted domain”과 “나머지 source들에서 온 adapted domain들”을 구분하도록 학습한다. generator 쪽은 이를 속이도록 학습되므로 결과적으로 adapted domains 간 구분이 어려워진다.

논문 식은 다음과 같다.

$$
\mathcal{L}_{SAD}^{S_i} = \mathbb{E}_{\mathbf{x}_i \sim X_i} \log D_A^i(G_{S_i \rightarrow T}(\mathbf{x}_i)) + \frac{1}{M-1} \sum_{j \neq i} \mathbb{E}_{\mathbf{x}_j \sim X_j} \log \left(1 - D_A^i(G_{S_j \rightarrow T}(\mathbf{x}_j))\right)
$$

핵심은 source별 adapted domains가 서로 분리된 sub-domain으로 남지 않게 만드는 것이다.

#### 3.5.2 Cross-domain Cycle Discriminator (CCD)

CCD는 더 흥미로운 장치다. 다른 source $S_j$ 에서 target으로 번역된 이미지 $G_{S_j \rightarrow T}(X_j)$ 를 다시 $G_{T \rightarrow S_i}$ 로 $S_i$ 쪽으로 보내, 이 결과가 실제 $S_i$ 이미지와 구분되는지를 discriminator $D_i$ 가 판단한다.

$$
\mathcal{L}_{CCD}^{S_i} = \mathbb{E}_{\mathbf{x}_i \sim X_i} \log D_i(\mathbf{x}_i) + \frac{1}{M-1} \sum_{j \neq i} \mathbb{E}_{\mathbf{x}_j \sim X_j} \log \left( 1 - D_i(G_{T \rightarrow S_i}(G_{S_j \rightarrow T}(\mathbf{x}_j))) \right)
$$

직관적으로는 “다른 source에서 target을 거쳐 다시 $S_i$ 로 돌아온 것”도 $S_i$ 와 일관된 표현이 되도록 강제하는 장치다. 이것은 adapted domains 간 간접적인 구조 정렬을 강화한다.

논문은 SAD와 CCD를 함께 사용해 여러 adapted domain을 더 통합된 하나의 intermediate domain으로 만들고자 한다.

### 3.6 Feature-aligned Task Learning (FTL)

이제 각 source가 번역·집계되어 aggregated domain $X'$ 를 형성하고, 원래 source label들을 그대로 사용할 수 있다고 가정한다. 즉, semantic consistency가 충분히 유지되었다면 adapted image에는 원래 label $Y_i$ 를 붙여도 된다는 것이다.

분류의 경우 task loss는 일반적인 cross-entropy다.

$$
\mathcal{L}_{cla}(F, X', Y) = - \mathbb{E}_{(\mathbf{x}', y) \sim (X', Y)} \sum_{l=1}^{L} \mathbf{1}[l = y] \log(\sigma(F(\mathbf{x}')))
$$

segmentation의 경우 픽셀 단위 cross-entropy를 사용한다.

$$
\mathcal{L}_{seg}(F, X', Y) = - \mathbb{E}_{(\mathbf{x}', \mathbf{y}) \sim (X', Y)} \sum_{l=1}^{L}\sum_{h=1}^{H}\sum_{w=1}^{W} \mathbf{1}[l = \mathbf{y}_{h,w}] \log(\sigma(F_{l,h,w}(\mathbf{x}')))
$$

여기에 더해, task network의 encoder 마지막 convolution layer feature $F_f(\cdot)$ 에 대해 aggregated domain과 target 사이의 **feature-level adversarial alignment** 를 수행한다. discriminator $D_{F_f}$ 가 이를 구분하려 하고, feature extractor는 구분 불가능하게 만든다.

$$
\mathcal{L}_{FLA}(F_f, D_{F_f}, X', X_T) = \mathbb{E}_{\mathbf{x}' \sim X'} \log D_{F_f}(F_f(\mathbf{x}')) + \mathbb{E}_{\mathbf{x}_T \sim X_T} \log \left(1 - D_{F_f}(F_f(\mathbf{x}_T))\right)
$$

즉, pixel-level alignment만으로 충분하다고 보지 않고, 최종 representation 공간에서도 target과 비슷하도록 한 번 더 정렬한다.

### 3.7 MADAN의 전체 학습 목표

논문은 위의 모든 손실을 합쳐 전체 objective를 정의한다. 각 source별로 pixel GAN loss, reverse GAN loss, cycle consistency, DSC, SAD, CCD를 더하고, 여기에 task loss와 feature-level alignment를 더한다.

개념적으로 쓰면 다음과 같다.

$$
\mathcal{L}_{MADAN} = \sum_{i=1}^{M} \left( \mathcal{L}_{GAN}^{S_i \rightarrow T} + \mathcal{L}_{GAN}^{T \rightarrow S_i} + \mathcal{L}_{cyc}^{S_i \leftrightarrow T} + \mathcal{L}_{DSC}^{S_i} + \mathcal{L}_{SAD}^{S_i} +
\mathcal{L}_{CCD}^{S_i} \right) + \mathcal{L}_{task} + \mathcal{L}_{FLA}
$$

최종적으로는 generator $G$ 는 손실을 최대화하는 방향, discriminator $D$ 와 task model $F$ 는 손실을 최소화하는 방향의 min-max 최적화로 표현한다.

$$
F^* = \arg\min_F \min_D \max_G \mathcal{L}_{MADAN}(G, D, F)
$$

수식 표기상 GAN 관례와 정확한 부호는 구현 세부에 따라 달라질 수 있으나, 논문이 말하는 의미는 분명하다. generator는 domain confusion과 target-like translation을 유도하고, discriminator는 구분하려 하며, task model은 라벨 예측과 target generalization을 동시에 개선한다.

### 3.8 MADAN+ : segmentation을 위한 확장

논문은 MADAN만으로도 segmentation adaptation이 가능하지만, segmentation에서는 더 정교한 처리가 필요하다고 본다. 그래서 MADAN+를 제안한다.

#### 3.8.1 Category-level Alignment (CLA)

글로벌 feature alignment는 클래스별 출현 빈도가 source와 target에서 비슷하다고 암묵적으로 가정한다. 그러나 실제 도로 장면에서는 sky, road, car, person 등의 빈도와 위치가 다르다. 따라서 클래스별로 local region을 따로 맞출 필요가 있다.

이를 위해 각 이미지의 grid 단위 region에서 특정 클래스 $l$ 이 얼마나 나타나는지를 계산한다. target에서는 ground truth가 없으므로 현재 모델의 prediction을 pseudo label로 사용한다.

$$
Y(\mathbf{x}_d) = \begin{cases} \mathbf{y}_d, & d \in {1, \dots, M} \\ F(\mathbf{x}_d), & d = T \end{cases}
$$

각 grid $n$ 과 class $l$ 에 대해 비율을 계산한 뒤,

$$
\aleph_n^l(\mathbf{x}_d) = \sum_{r \in \mathcal{R}(n)} \frac{|Y(\mathbf{x}_d^r) == l|}{|\mathcal{R}(n)|}
$$

이를 정규화하여

$$
\widetilde{\aleph}_n^l(\mathbf{x}_d) = \frac{\aleph_n^l(\mathbf{x}_d)} {\sum_{n=1}^{N} \aleph_n^l(\mathbf{x}_d)}
$$

class-specific discriminator $D_C^l$ 를 통해 adapted와 target의 해당 class region feature를 정렬한다.

$$
\mathcal{L}_{CLA} = \mathbb{E}_{\mathbf{x}' \sim X'} \sum_{l=1}^{L}\sum_{n=1}^{N} \widetilde{\aleph}_n^l(\mathbf{x}') \log D_C^l(F_f(\mathbf{x}')_n) + \mathbb{E}_{\mathbf{x}_T \sim X_T} \sum_{l=1}^{L}\sum_{n=1}^{N} \widetilde{\aleph}_n^l(\mathbf{x}_T) \log \left(1 - D_C^l(F_f(\mathbf{x}_T)_n)\right)
$$

쉽게 말하면, 글로벌하게 한꺼번에 맞추는 대신 “road끼리, sky끼리, car끼리” 비슷해지도록 정렬하는 것이다.

#### 3.8.2 Context-aware Generation (CAG)

segmentation에서는 전역 구조뿐 아니라 경계와 local detail도 중요하다. 논문은 CycleGAN 기반 translation이 한 가지 crop scale만 쓰면, 큰 crop에서는 local detail이 부족하고 작은 crop에서는 global semantic이 약해질 수 있다고 지적한다. 또한 source와 target에서 임의 위치를 독립적으로 crop하면 spatial misalignment도 생길 수 있다.

이를 해결하기 위해, adapted image와 target image를 같은 중심점을 기준으로 여러 크기 ${C_1, \dots, C_K}$ 로 **uniform cropping** 하고, 이를 고정 해상도로 resize한 multi-scale pyramid를 사용한다. 이렇게 하면 하늘은 위쪽, 도로는 아래쪽이라는 공간 구조를 어느 정도 보존하면서, global semantics와 local details를 동시에 반영할 수 있다.

논문은 이를 위한 CAG loss를 multi-scale 전반에 대한 GAN + cycle + DSC 손실의 합으로 쓴다.

$$
\mathcal{L}_{CAG} = \sum_{i=1}^{M}\sum_{k=1}^{K} \left[ \mathcal{L}_{GAN}^{S_i \rightarrow T} + \mathcal{L}_{GAN}^{T \rightarrow S_i} + \mathcal{L}_{cyc}^{S_i \leftrightarrow T} + \mathcal{L}_{DSC}^{S_i} \right]
$$

scale $k$ 별 cropped mini-batch에 대해 이를 적용하는 방식이다.

#### 3.8.3 MADAN+ 전체 목적함수

MADAN+는 MADAN의 손실에 CAG와 CLA를 더한 구조다.

$$
\mathcal{L}_{MADAN+} = \mathcal{L}_{CAG} + \sum_{i=1}^{M} \left( \mathcal{L}_{SAD}^{S_i} + \mathcal{L}_{CCD}^{S_i} \right) + \mathcal{L}_{task} + \mathcal{L}_{FLA} + \mathcal{L}_{CLA}
$$

즉, segmentation용으로는 translation 자체를 더 정교하게 하고, feature alignment도 클래스 수준으로 세분화한다.

### 3.9 학습 절차

논문은 이론적으로 end-to-end 학습이 가능하다고 하지만, 실제 구현에서는 **하드웨어 자원 제약** 때문에 3단계 학습을 수행했다.

첫째, 각 source-target 쌍에 대해 semantic consistency 없이 CycleGAN들을 먼저 학습하고, 생성된 adapted images에 대해 task model $F$ 를 학습한다.

둘째, 이렇게 업데이트된 $F$ 를 $F_A$ 로 사용해 DSC loss를 포함한 CycleGAN을 다시 학습하고, 동시에 SAD와 CCD로 adapted domain을 aggregation한다.

셋째, 새롭게 생성된 aggregated adapted domain에서 feature-level alignment를 포함하여 task model $F$ 를 다시 학습한다.

이 과정을 iterative하게 반복한다. 따라서 논문 제목과 표에서는 end-to-end 특성을 강조하지만, 구현 상세에서는 **실제 실험은 완전한 end-to-end가 아니라 staged training** 이다. 이 점은 해석할 때 중요하다.

## 4. 실험 및 결과

### 4.1 데이터셋과 작업

논문은 세 종류의 작업에서 실험했다.

첫째는 **digit recognition** 이다. Digits-five를 사용하며, MNIST, MNIST-M, SVHN, Synthetic Digits, USPS의 다섯 도메인 중 하나를 target으로, 나머지를 source로 둔다.

둘째는 **object classification** 이다. Office-31, Office+Caltech-10, Office-Home을 사용한다.

셋째는 **semantic segmentation** 이다. synthetic source인 GTA와 SYNTHIA를 사용하고, real target으로 Cityscapes와 BDDS를 사용한다. segmentation에서는 16개 공통 클래스의 mIoU와 class-wise IoU를 사용한다.

### 4.2 비교 기준과 평가 지표

분류에서는 평균 classification accuracy를 사용한다. segmentation에서는 class-wise IoU와 mean IoU를 사용한다.

$$
cwIoU_l = \frac{|\mathcal{P}_l \cap \mathcal{G}_l|}{|\mathcal{P}_l \cup \mathcal{G}_l|}
$$

$$
mIoU = \frac{1}{L}\sum_{l=1}^{L} cwIoU_l
$$

비교군은 source-only, single-source DA, source-combined DA, multi-source DA로 나뉜다. single-source 및 multi-source에서 다양한 기존 방법들과 비교한다.

### 4.3 분류 결과

#### Digits-five

Table II에서 MADAN은 평균 **90.9%** 를 기록했다. 이는 기존 multi-source 최고였던 MDDA의 **88.1%** 보다 높고, M3SDA의 **87.7%** 보다도 높다. source-combined CyCADA는 **85.1%** 로 MADAN보다 크게 낮다.

특히 mm target에서 MADAN은 **82.9%** 로 매우 강한 성능을 보였고, sy target에서는 **95.2%** 로 다른 방법 대비 큰 향상을 보인다. 저자들은 이를 여러 source의 상보적 정보를 활용하면서도 source 간 간섭을 aggregation으로 줄였기 때문이라고 해석한다.

#### Office-31

Table III에서 MADAN은 평균 **87.2%** 로, 기존 최고 MDDA의 **84.2%** 보다 높다. 특히 A target에서 **63.9%** 를 기록했는데, 이는 기존 대비 큰 개선폭이다. D와 W는 원래도 높았지만, 어려운 A domain에서 improvement가 크다는 점이 중요하다.

#### Office+Caltech-10

Table IV에서 MADAN은 평균 **98.6%** 로 최고 성능을 달성한다. M3SDA의 **96.4%** 를 넘는다. C target에서 **97.2%**, A target에서 **97.9%** 는 매우 높다.

#### Office-Home

Table V에서 MADAN은 평균 **70.4%** 로, source-combined CyCADA의 **66.3%**, MDAN의 **65.0%** 를 앞선다. Office-Home은 카테고리 수가 많고 도메인 차이도 커서 더 어려운 데이터셋인데, 여기서도 개선이 유지된다는 점은 방법의 일반성을 뒷받침한다.

### 4.4 segmentation 결과

#### FCN-VGG16 backbone, GTA+SYNTHIA → Cityscapes

Table VI에서 MADAN은 **41.4 mIoU**, MADAN+는 **42.8 mIoU** 를 기록한다. source-combined CyCADA는 **37.3**, MDAN은 **29.4** 이므로 상당한 차이가 있다. single-source best인 GTA-only CyCADA **38.7** 보다도 높다.

MADAN+는 road 87.9, sidewalk 41.0, sky 80.0, person 54.9, rider 21.5, car 80.1, bus 29.7 등에서 좋은 결과를 낸다. 다만 fence class가 1.3으로 매우 낮게 나온 부분처럼, 모든 클래스가 일관되게 개선되는 것은 아니다. 논문도 class-wise 편차는 존재함을 표를 통해 보여준다.

#### FCN-VGG16 backbone, GTA+SYNTHIA → BDDS

Table VII에서 MADAN은 **36.3 mIoU**, MADAN+는 **41.6 mIoU** 다. source-combined CyCADA의 **33.7** 과 비교하면 꽤 크고, MDAN의 **25.0** 과는 격차가 더 크다. 특히 MADAN+는 building 83.3, wall 27.2, pole 37.8, traffic light 23.2, car 80.2 등에서 향상이 눈에 띈다.

#### DeepLabV2-ResNet101 backbone, GTA+SYNTHIA → Cityscapes

Table VIII에서 MADAN은 **45.4 mIoU**, MADAN+는 **48.5 mIoU** 를 얻는다. source-combined CyCADA **39.3**, MDAN **36.0** 보다 높다. GTA-only CyCADA **47.9** 와 비교하면 MADAN+가 근소하게 우세하다.

#### DeepLabV2-ResNet101 backbone, GTA+SYNTHIA → BDDS

Table IX에서 MADAN은 **40.4**, MADAN+는 **42.7 mIoU** 다. source-combined CyCADA의 **37.2**, MDAN의 **29.4** 보다 높다.

이 전체 결과는, 분류뿐 아니라 구조적 예측인 segmentation에서도 multi-source setting에서 pixel-level adaptation과 domain aggregation이 실질적으로 도움이 됨을 보여준다.

### 4.5 정성적 결과와 해석 가능성

Figure 3의 segmentation visualization에서 MADAN+ 결과가 source-only와 단순 CycleGAN보다 ground truth에 더 가깝다고 설명한다. 예시로 보행자와 자전거 이용자의 contour가 더 선명해졌다고 한다.

Figure 5와 Figure 6은 image translation 결과를 보여준다. classification 데이터에서는 source 스타일이 target 스타일로 비교적 일관되게 바뀌고, segmentation에서는 최종 adapted image가 target의 hue와 brightness와 더 비슷해진다고 주장한다. 이는 pixel-level alignment의 해석 가능성을 높여주는 요소다.

Figure 7의 Grad-CAM 시각화에서는 adaptation 이후 모델 attention이 배경보다 실제 객체의 구분 가능한 부분에 더 집중한다고 한다. 이는 단순 성능 향상 외에도 feature가 더 transferable하고 discriminative해졌다는 해석을 제공한다.

### 4.6 DSC 비교 실험

Table X와 XI는 기존 semantic consistency (SC)와 제안한 DSC를 비교한다. Cityscapes에서 예를 들어 GTA→Cityscapes 기준으로 CycleGAN+SC는 **32.7 mIoU**, CycleGAN+DSC는 **38.1** 이다. CyCADA with SC는 **38.7**, CyCADA with DSC는 **40.0** 이다. SYNTHIA→Cityscapes에서도 DSC가 더 좋다.

BDDS에서도 유사하다. GTA 기준 CyCADA with SC는 **32.0**, CyCADA with DSC는 **35.5** 다. 이는 DSC가 실제로 semantic preservation과 target-oriented translation의 균형을 더 잘 맞춘다는 논문 주장을 뒷받침한다.

### 4.7 Ablation study

Table XII, XIII는 MADAN+의 각 구성요소를 incremental하게 더한 실험이다.

Cityscapes에서 baseline은 **30.0 mIoU** 이고, +SAD는 **36.4**, +CCD는 **33.1**, +SAD+CCD는 **37.5** 다. 여기에 +DSC가 들어가면 **40.2**, +FLA로 **41.4**, +CLA로 **41.6**, +CAG까지 포함하면 **42.8** 이 된다.

BDDS에서는 baseline **24.6**, +SAD **31.2**, +CCD **29.3**, +SAD+CCD **31.8**, +SAD+CCD+DSC **35.3**, +FLA **36.3**, +CLA **37.8**, +CAG 최종 **41.6** 이다.

이 실험은 몇 가지 메시지를 준다. SAD와 CCD 모두 유효하지만, 표상으로는 SAD가 좀 더 안정적으로 큰 이득을 주는 경향이 있다. DSC는 거의 항상 추가 이득을 준다. FLA는 pixel-level adaptation 이후에도 여전히 도움이 된다. CLA와 CAG는 segmentation 특화 확장으로서 MADAN을 MADAN+로 끌어올리는 핵심이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 multi-source domain adaptation을 단순 feature alignment 문제로 보지 않고, **pixel-level translation + source 간 aggregation + feature alignment** 라는 다층 구조로 재정의했다는 점이다. 특히 서로 다른 source들 사이의 misalignment를 명시적으로 다루는 SAD와 CCD는 논문의 독창성이 가장 강하게 드러나는 부분이다.

두 번째 강점은 **semantic segmentation에 대한 multi-source adaptation** 을 본격적으로 다뤘다는 점이다. 논문 스스로 “first work on multi-source structured domain adaptation”이라고 주장하며, 제공된 텍스트 범위에서는 이 주장을 뒷받침하는 비교와 실험이 제시되어 있다. segmentation은 classification보다 훨씬 까다로운 structured prediction 문제이므로, 여기서 효과를 보였다는 점은 의미가 있다.

세 번째 강점은 DSC다. 기존 semantic consistency를 그대로 쓰지 않고, target-like translated image에 더 적합한 기준 모델이 필요하다는 문제의식을 제시한 뒤, 이를 동적으로 해결한다. 이는 단순 heuristic이 아니라 adaptation 과정의 본질과 연결된 설계다.

네 번째 강점은 실험 범위가 넓다는 점이다. digit recognition, object classification, semantic segmentation까지 포함하고, synthetic-to-real setting에서도 두 개의 backbone과 두 개의 target dataset을 사용했다. 또한 정량 결과뿐 아니라 visualization, attention map, ablation, DSC 비교 실험까지 포함해 설득력을 높였다.

반면 한계도 분명하다.

첫째, 모델 구조가 매우 복잡하다. source마다 source→target, target→source generator와 다수의 discriminator가 필요하고, 여기에 SAD, CCD, feature-level discriminator, segmentation에서는 class-wise discriminator까지 추가된다. multi-source 개수가 늘수록 계산량과 메모리 사용량이 빠르게 증가할 가능성이 크다. 실제로 논문도 자원 제약 때문에 end-to-end 대신 3단계 학습을 사용했다고 밝힌다.

둘째, 이 논문은 이론적으로는 end-to-end 프레임워크를 제시하지만, 실제 실험은 staged training에 의존한다. 따라서 표에 있는 “end-to-end” 속성은 설계 수준의 특성이지, 실험 구현이 완전히 end-to-end였다는 의미로 읽으면 안 된다.

셋째, segmentation 결과에서 클래스별 성능 편차가 꽤 크다. 예를 들어 Cityscapes FCN-VGG16 결과에서 MADAN+의 fence가 1.3으로 극단적으로 낮다. 즉, 평균 mIoU는 좋아졌지만 모든 class에서 균형 있게 좋아진 것은 아니다. 이는 class imbalance나 translation artifact, 혹은 CLA/CAG의 클래스별 작동 차이와 관련될 수 있으나, 논문은 이에 대한 상세 원인 분석을 충분히 제공하지는 않는다.

넷째, 문제 설정이 **homogeneous closed-set unsupervised MDA** 로 제한된다. heterogeneous DA, open-set DA, category-shift DA는 미래 과제로 남겼다. 따라서 방법의 적용 범위는 넓지만, 더 현실적인 label mismatch 상황까지 다루는 것은 아니다.

다섯째, 여러 손실 항의 상대적 중요도를 어떻게 조정하는지가 중요한데, 논문은 sophisticated weighting이나 prior domain knowledge incorporation이 향후 과제라고 명시한다. 즉, 현재는 손실 조합이 강력하지만 다소 수작업적일 가능성이 있다.

비판적으로 보면, 이 논문은 성능 향상과 설계의 정교함 면에서 강하지만, 계산 효율성과 단순성 측면에서는 무겁다. 또한 많은 구성요소가 들어간 만큼 “어느 요소가 어떤 상황에서 반드시 필요한가”에 대한 일반화된 결론은 제한적이다. 그럼에도 ablation 결과상 각 모듈이 대체로 일관된 이득을 주므로, 단순히 과도한 engineering만이라고 보기는 어렵다.

## 6. 결론

이 논문은 multi-source domain adaptation을 위해 **MADAN** 이라는 새로운 프레임워크를 제안했다. 핵심은 각 source를 target처럼 변환하는 **pixel-level alignment**, 여러 adapted domain을 하나로 모으는 **domain aggregation**, 그리고 aggregated domain과 target 사이를 맞추는 **feature-level alignment** 를 함께 사용한 것이다. 여기에 semantic consistency를 정적으로 두지 않고 **Dynamic Semantic Consistency** 로 바꾸어, translation과 task learning이 더 자연스럽게 연결되도록 했다.

또한 semantic segmentation을 위해 **MADAN+** 를 제안하여, **category-level alignment** 와 **context-aware generation** 을 추가함으로써 structured prediction 문제에서 성능을 더 끌어올렸다. 실험 결과는 Digits-five, Office-31, Office+Caltech-10, Office-Home, Cityscapes, BDDS 전반에서 강력하며, 특히 segmentation에서 기존 multi-source 및 single-source baselines를 일관되게 앞선다.

실제 적용 측면에서 이 연구는 여러 synthetic 혹은 서로 다른 수집 조건의 labeled dataset을 함께 활용해야 하는 상황, 특히 자율주행·로봇 비전·산업 영상 등에서 중요한 의미를 가진다. 향후에는 계산량을 줄이는 방향, source별 중요도를 자동으로 조절하는 방향, 그리고 multi-modal adaptation으로 확장하는 방향이 유망해 보인다. 제공된 텍스트 기준으로 볼 때, 이 논문은 multi-source DA를 고수준 feature 정렬에 머물게 하지 않고, 이미지 생성과 구조적 aggregation까지 확장한 대표적인 작업으로 평가할 수 있다.
