# Source-Free Domain Adaptation for Semantic Segmentation

* **저자**: Yuang Liu, Wei Zhang, Jun Wang
* **발표연도**: 2021
* **arXiv**: [https://arxiv.org/abs/2103.16372](https://arxiv.org/abs/2103.16372)

## 1. 논문 개요

이 논문은 semantic segmentation에서의 **source-free domain adaptation** 문제를 정식으로 다룬다. 기존의 unsupervised domain adaptation(UDA) 방법들은 대체로 두 가지를 동시에 가정한다. 하나는 source domain에서 학습된 모델이 존재한다는 것이고, 다른 하나는 adaptation 단계에서도 원본 source dataset 전체에 접근할 수 있다는 것이다. 그러나 실제 환경에서는 source 데이터가 사내 자산이거나 개인정보, 상업적 이유로 외부에 공개될 수 없어서, 배포 가능한 것은 학습된 source model뿐인 경우가 많다. 논문은 바로 이 제약을 전제로, **source data 없이 source model과 unlabeled target data만으로 segmentation 모델을 적응시키는 방법**을 제안한다.

연구 문제는 단순히 “source 데이터 없이 adaptation이 가능한가”에 그치지 않는다. semantic segmentation은 image classification과 달리 픽셀 단위로 클래스를 예측해야 하므로, 한 이미지 안에 여러 클래스가 혼재하고 각 클래스의 위치와 문맥 관계가 중요하다. 따라서 classification용 source-free UDA에서 자주 쓰이는 feature clustering이나 image-level pseudo-label 전략을 그대로 적용하기 어렵다. 특히 source 데이터가 없으면 adaptation 도중 source domain knowledge를 유지하는 supervised anchor가 사라지기 때문에, target pseudo-label의 오류가 누적되면서 모델이 쉽게 drift할 수 있다.

이 문제가 중요한 이유는 분명하다. semantic segmentation은 자율주행, scene understanding, visual grounding 같은 실제 시스템의 핵심 기술인데, dense annotation 비용이 매우 크다. 논문에서도 Cityscapes 한 장을 수작업으로 레이블링하는 데 약 90분이 든다고 설명한다. 결국 현실적으로는 합성 데이터나 외부 데이터로 source model을 미리 학습해 두고, 현장 target domain에 맞게 적응시키는 방식이 유력하다. 이때 source data를 함께 가져올 수 없다면, source-free adaptation은 매우 실용적인 문제 설정이 된다.

이 논문은 이런 배경에서 **SFDA**라는 프레임워크를 제안한다. 핵심은 두 단계이다. 첫째, generator를 이용해 source-like fake samples를 합성하고, 이를 통해 source model의 지식을 target model로 전달하는 **knowledge transfer stage**를 수행한다. 둘째, target domain 내부에서 비교적 신뢰할 수 있는 patch를 골라 자기지도 형태로 활용하는 **model adaptation stage**를 수행한다. 저자들은 이를 통해 source data를 전혀 쓰지 않으면서도 기존 source-driven UDA와 경쟁 가능한 성능을 낼 수 있다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음과 같다. source 데이터는 없지만, **source domain에 대한 지식은 이미 source model의 파라미터 안에 저장되어 있다**. 따라서 adaptation의 핵심은 source 데이터를 다시 직접 쓰는 것이 아니라, 모델 내부에 저장된 source knowledge를 어떻게 복원하고 전달할 것인가이다. 저자들은 이 문제를 데이터 없는 knowledge distillation(data-free KD) 관점에서 해석한다.

이를 위해 논문은 generator를 도입해 source domain을 직접 관측하는 대신 **source-like fake images**를 합성한다. 이 fake images는 사람 눈에 자연스럽게 보일 필요는 없고, source model의 batch normalization statistics(BNS)와 semantic/contextual structure를 만족하면서, source model이 담고 있는 표현을 잘 끌어낼 수 있으면 된다. 이렇게 생성된 샘플을 바탕으로 source model의 출력을 보존하고, target model이 source knowledge를 잃지 않도록 distillation을 수행한다. 즉, adaptation 중에 source supervision이 사라지는 문제를 “가짜 source 데이터 생성 + distillation”으로 우회한다.

이 논문의 중요한 차별점은 segmentation에 맞는 distillation 설계를 했다는 점이다. classification에서는 주로 logits나 feature matching 정도로 충분한 경우가 많지만, segmentation은 픽셀 간 관계와 채널 간 문맥 정보가 중요하다. 이를 반영해 논문은 **Dual Attention Distillation (DAD)** 를 제안한다. DAD는 spatial attention과 channel attention을 모두 계산하여, source model과 student 쪽 모델이 단지 최종 예측만 비슷한 것이 아니라, **문맥적 관계(contextual relationships)** 자체도 비슷하도록 유도한다.

또 다른 핵심 아이디어는 target domain 내부의 자기지도 학습 방식이다. source-free 환경에서는 target pseudo-label이 유일한 학습 신호인데, 전체 이미지 기준으로는 noise가 크다. 저자들은 street-view segmentation에서는 동일한 위치의 patch들이 서로 유사한 구조를 가진다는 점에 착안해, 이미지를 $K \times K$ patch로 나눈 뒤 entropy 기준으로 쉬운 패치와 어려운 패치를 나눈다. 그리고 쉬운 패치의 예측을 기준 삼아 어려운 패치를 맞추도록 adversarial learning을 수행하는 **IPSM (Intra-domain Patch-level Self-supervision Module)** 을 도입한다. 이 설계는 전체 pseudo-label의 불확실성을 줄이면서도, target domain 자체에서 더 세밀한 supervision을 끌어내려는 시도다.

정리하면, 기존 접근과의 차별점은 세 가지다. 첫째, source-free segmentation adaptation을 정면으로 다뤘다. 둘째, segmentation에 특화된 attention-based knowledge transfer를 설계했다. 셋째, target 내 patch 수준 자기지도로 adaptation을 보강했다. 즉 이 논문은 “source data 없음”이라는 제약과 “segmentation의 픽셀 수준 구조”라는 난점을 동시에 겨냥한 방법이라고 볼 수 있다.

## 3. 상세 방법 설명

전체 프레임워크는 크게 **Knowledge Transfer stage** 와 **Model Adaptation stage** 로 나뉜다. 사용 가능한 것은 고정된 source model과 unlabeled target dataset뿐이다. 여기서 논문은 source model을 두 개의 역할로 사용한다. 하나는 파라미터를 고정한 teacher 역할의 $\tilde{\mathcal{S}}$이고, 다른 하나는 target model과 weight를 공유하며 업데이트되는 $\mathcal{S}$ 혹은 $\mathcal{T}$이다. generator $\mathcal{G}$는 noise $z$를 입력받아 synthetic source-like image $\tilde{x}_s$를 만든다.

먼저 일반적인 source-driven UDA는 다음과 같이 쓸 수 있다.

$$
\mathcal{L}_{DA}=\mathcal{L}_{SEG}(x_s,y_s)+\mathcal{L}_{TAR}(x_t)
$$

여기서 $\mathcal{L}_{SEG}$는 source label을 이용한 supervised segmentation loss이고, $\mathcal{L}_{TAR}$는 target pseudo-label 기반 self-supervision loss이다. 하지만 source-free 설정에서는 $(x_s,y_s)$가 없으므로 $\mathcal{L}_{SEG}$를 직접 사용할 수 없다. 이 때문에 논문은 source knowledge를 보존하는 새로운 우회 경로를 설계한다.

### 3.1 Target self-supervision의 기본 손실

저자들은 target adaptation의 보조 항으로 MaxSquare loss를 사용한다. 식은 다음과 같다.

$$
\mathcal{L}_{TAR}(x_t)= -\frac{1}{HW}\sum_{h,w}^{HW}\sum_{c}^{C}(p_t^{h,w,c})^2
$$

여기서 $p_t^{h,w,c}$는 target 이미지의 픽셀 $(h,w)$가 클래스 $c$일 확률이다. 이 loss는 예측 분포를 더 뾰족하게 만들어 entropy를 낮추는 효과가 있다. 즉, target pseudo-label을 더 confident하게 만들려는 목적이다. 다만 이것만 쓰면 잘못된 pseudo-label도 더 강하게 믿게 될 수 있으므로, source knowledge transfer와 patch-level self-supervision이 함께 필요하다.

### 3.2 Source domain estimation: generator와 BNS

source-free setting에서 가장 먼저 필요한 것은 “source를 닮은 입력”이다. 이를 위해 generator는 Gaussian noise에서 synthetic sample을 생성한다.

$$
\tilde{x}_{s}=\mathcal{G}(z),; z\sim\mathcal{N}(\mathbf{0},\mathbf{1})
$$

이 synthetic sample이 source domain을 어느 정도 복원하려면, source model 내부 BN 레이어가 저장하고 있는 통계와 맞아야 한다. 그래서 논문은 BNS loss를 둔다.

$$
\mathcal{L}_{BNS}=\sum_{l}|\mu_{l}(\tilde{x}_{s})-\bar{\mu}_{l}|^{2}_{2}
+\sum_{l}|\sigma_{l}^{2}(\tilde{x}_{s})-\bar{\sigma}_{l}^{2}|^{2}_{2}
$$

여기서 $\mu_l(\tilde{x}_s)$와 $\sigma_l^2(\tilde{x}_s)$는 synthetic sample을 source model의 $l$번째 BN layer에 통과시켰을 때의 batch mean, variance이고, $\bar{\mu}_l$, $\bar{\sigma}_l^2$는 source model이 원래 학습하며 저장한 BN 통계다. 즉 generator는 “source model이 익숙한 feature distribution”을 만들도록 유도된다.

이 손실은 fake image를 source-like하게 만드는 기초 장치다. 그러나 저자들은 이것만으로는 segmentation에 충분하지 않다고 본다. BN 통계는 분포의 저차 요약일 뿐이고, segmentation에 중요한 spatial context나 category relationship까지 직접 보장하진 않기 때문이다.

### 3.3 Output-level distillation: MAE loss

generator가 만든 fake sample에 대해, 고정 teacher $\tilde{\mathcal{S}}$의 예측 $\tilde{y}_s=\tilde{\mathcal{S}}(\tilde{x}_s)$를 pseudo target처럼 삼고, 학습 중인 모델 $\mathcal{S}$의 출력과 맞춘다.

$$
\mathcal{L}_{MAE}=\mathbb{E}_{\tilde{x}_{s}}\left(\frac{1}{C}|\mathcal{S}(\tilde{x}_{s})-\tilde{y}_{s}|_{1}\right)
$$

이 식은 synthetic source-like input 위에서 teacher와 student가 비슷한 segmentation 출력을 내도록 하는 output-level distillation이다. 직관적으로는 “가짜 source 이미지에서라도 source teacher처럼 예측하라”는 의미다. source supervision이 직접 없으므로, 이 항이 source model의 class-level decision boundary를 유지하는 데 도움을 준다.

### 3.4 Dual Attention Distillation (DAD)

논문의 핵심은 여기서부터다. 저자들은 segmentation에서 단순 output matching만으로는 부족하다고 본다. semantic segmentation은 픽셀 간 상대적 위치 관계, 장면 구조, 클래스 간 문맥적 연관성이 중요하기 때문이다. 예를 들어 road 아래쪽에 car가 있고 sky가 위에 오는 식의 구조적 관계는 segmentation 품질에 직접 연결된다.

이를 위해 논문은 feature map $F=\mathcal{F}(x)$에서 **spatial attention**과 **channel attention**을 동시에 계산하는 Dual Attention Module(DAM)을 정의한다. 먼저 backbone feature를 $F \in \mathbb{R}^{N_1 \times C_1}$ 형태로 reshape한다. 여기서 $N_1 = H_1W_1$는 spatial position의 개수다.

Spatial attention map $S \in \mathbb{R}^{N_1 \times N_1}$는 다음과 같이 계산된다.

$$
s_{ji}=\frac{\exp(F_{[i:]}\cdot F^{\top}_{[:j]})}{\sum^{N_{1}}_{i}\exp(F_{[i:]}\cdot F^{\top}_{[:j]})}
$$

이 값은 $i$번째 위치가 $j$번째 위치에 얼마나 영향을 주는지를 나타낸다. 즉, 픽셀 간 장거리 의존성을 표현한다.

마찬가지로 channel attention map $R \in \mathbb{R}^{C_1 \times C_1}$는 다음과 같다.

$$
r_{ji}=\frac{\exp(F^{\top}_{[i:]}\cdot F_{[:j]})}{\sum^{C_{1}}_{i}\exp(F^{\top}_{[i:]}\cdot F_{[:j]})}
$$

이는 $i$번째 channel이 $j$번째 channel에 주는 영향을 나타낸다. 즉, 클래스 구분이나 texture/shape detector 같은 feature channel 간 상호작용을 반영한다.

그 다음 dual attention map은 spatial과 channel 정보를 합친 표현으로 만든다.

$$
\mathcal{A}(x)=\mathtt{concat}(F\cdot S|R\cdot F)
$$

논문 설명에 따르면 attention map끼리 직접 concat하는 것이 아니라, 원래 feature $F$와 곱한 뒤 concat하여 shape를 맞춘다. 이렇게 얻은 representation은 segmentation에서 중요한 구조적 정보를 담는다고 본다.

이제 fake source sample 위에서 고정 source teacher와 학습 중인 source/student 모델의 dual attention representation 차이를 줄이는 손실을 정의한다.

$$
\mathcal{L}_{DAD}^{ss}=\mathbb{E}_{\tilde{x}_{s}}
\left(\frac{1}{M}
|\mathcal{A}(\tilde{\mathcal{F}}^{s}(\tilde{x}_{s}))
-\mathcal{A}(\mathcal{F}^{s}(\tilde{x}_{s}))|_{1}\right)
$$

이는 synthetic source-like sample에 대해 teacher와 student가 비슷한 contextual relationship을 갖도록 만드는 항이다.

추가로 논문은 source-like fake sample과 실제 target sample 사이의 attention distribution도 연결한다. 이를 위해 spatial attention과 channel attention 각각에 대해 KL divergence를 사용한다.

$$
\mathcal{L}_{DAD}^{st} = \mathbb{E}_{\tilde{x}_{s}} \left[ D_{KL}\left(S(\tilde{\mathcal{F}}^{s}(\tilde{x}_{s})),S(\mathcal{F}^{t}(x_{t}))\right) \right] +
\mathbb{E}_{\tilde{x}_{s}} \left[ D_{KL}\left(R(\tilde{\mathcal{F}}^{s}(\tilde{x}_{s})),R(\mathcal{F}^{t}(x_{t}))\right) \right]
$$

이 항의 의미는, generator가 만드는 fake source sample이 단지 BN 통계만 맞추는 것이 아니라, target domain이 가진 domain-agnostic semantic structure도 참고하게 하는 것이다. 저자들은 generator가 아무 prior 없이 source를 완전히 복원하기는 어렵다고 보고, target 이미지에 존재하는 일반적인 semantic structure를 attention 차원에서 흡수하게 하려는 것이다.

### 3.5 Knowledge transfer stage의 최적화

generator의 목적함수는 다음과 같다.

$$
\min_{\mathcal{G}}
\mathcal{L}_{BNS}
-\alpha\mathcal{L}_{MAE}
-\beta\mathcal{L}_{DAD}^{ss}
+\tau\mathcal{L}_{DAD}^{st}
$$

이 식을 해석하면 다음과 같다. generator는 먼저 BNS를 만족하는 샘플을 만들고, 동시에 teacher와 student 사이의 discrepancy를 크게 만들도록 $\mathcal{L}_{MAE}$와 $\mathcal{L}_{DAD}^{ss}$ 앞에 음수를 둔다. 이는 generator가 “두 모델이 아직 잘 맞지 않는 어려운 synthetic sample”을 찾도록 유도하는 역할로 볼 수 있다. 반면 $\mathcal{L}_{DAD}^{st}$는 최소화하여 target attention 구조와 어느 정도 맞는 fake sample을 생성하게 한다. 즉, source teacher를 잘 자극하면서도 target과 전혀 무관하지 않은 중간 영역의 샘플을 찾으려는 설계다.

반대로 target/student 모델 쪽은 teacher와의 차이를 줄이는 방향으로 학습된다.

$$
\min_{\mathcal{T},\mathcal{S}}
\alpha\mathcal{L}_{MAE}
+\beta\mathcal{L}_{DAD}^{ss}
$$

즉 generator와 student 사이에는 일종의 adversarial한 관계가 있고, student는 fake source sample 위에서 source teacher의 output과 contextual relation을 따라가도록 훈련된다. 이것이 논문이 말하는 **Source-Free Knowledge Transfer (SFKT)** 의 핵심이다.

### 3.6 IPSM: patch-level self-supervision

knowledge transfer만으로는 target adaptation이 충분하지 않을 수 있다. 저자들은 generator가 source domain을 완벽히 복원하지 못한다는 점을 인정하고, target domain 자체에서도 supervision을 더 뽑아내려 한다.

이를 위해 각 target image를 $K \times K$ patch로 나눈다. 각 patch $x_{t,k}$의 예측 probability map $p_{t,k}$에 대해 평균 entropy를 계산한다.

$$
E(x_{t,k})=
-\frac{1}{H_{2}W_{2}}
\sum_{h,w}^{H_{2}W_{2}}\sum_{c}^{C}
p_{t,k}^{h,w,c}\log(p_{t,k}^{h,w,c})
$$

entropy가 낮은 patch는 예측 confidence가 높고 정확할 가능성이 크다고 가정한다. 한 배치에서 같은 위치 $k$에 해당하는 patch들끼리 entropy ranking을 수행해, 절반은 easy group, 나머지 절반은 hard group으로 나눈다.

$$
I_{t,k}^{\bullet},I_{t,k}^{\circ}
\leftarrow
\mathtt{Rank}({E(x_{t,k}^{b})|b\in\mathbb{R}^{B}}),,k\in\mathbb{R}^{K^{2}}
$$

여기서 표기상 $I_{t,k}^{\circ}$는 easy patches, $I_{t,k}^{\bullet}$는 hard patches를 의미한다.

그 다음 discriminator $\mathcal{D}$는 easy patch와 hard patch를 구분하도록 학습되고, target model은 hard patch가 easy처럼 보이도록 속이게 된다. adversarial loss는 다음과 같다.

$$
\mathcal{L}_{ADV}(I_{t}^{\bullet},I_{t}^{\circ}) = -\sum^{K^{2}}_{k}\sum^{B/2}_{d,e} \log\left(1-\mathcal{D}(k,i_{t,k}^{e})\right) + \log\left(\mathcal{D}(k,i_{t,k}^{d})\right)
$$

논문 텍스트만 보면 기호 배치가 다소 어색한 부분이 있으나, 의도는 명확하다. discriminator는 easy/hard patch를 구분하고, target model은 hard patch representation이 easy patch distribution에 가까워지도록 만든다. 이는 target 내부에서 confidence가 높은 patch의 구조를 confidence가 낮은 patch로 전달하는 self-supervision이다.

### 3.7 최종 학습 목표

최종적으로 model adaptation stage에서 타깃/공유 모델과 discriminator는 다음 min-max 문제를 푼다.

$$
\min_{\mathcal{T},\mathcal{S}}\max_{\mathcal{D}} \mathcal{L}_{TAR} +\alpha\mathcal{L}_{MAE} +\beta\mathcal{L}_{DAD}^{ss} +\gamma\mathcal{L}_{ADV}
$$

여기서 $\gamma$는 adversarial self-supervision의 세기를 조절하는 하이퍼파라미터다. 즉 최종 학습은 세 가지 축을 동시에 가진다. 첫째, target prediction을 confident하게 만드는 $\mathcal{L}_{TAR}$. 둘째, source teacher knowledge를 유지하는 $\mathcal{L}_{MAE}$와 $\mathcal{L}_{DAD}^{ss}$. 셋째, target 내부 easy/hard patch gap을 줄이는 $\mathcal{L}_{ADV}$다.

전체적으로 보면 이 방법은 “source model로부터 지식을 잃지 않도록 붙들어 주는 축”과 “target 데이터 내부에서 신뢰할 수 있는 구조를 증폭하는 축”을 결합한 형태다.

## 4. 실험 및 결과

### 4.1 실험 설정

논문은 두 가지 큰 시나리오에서 평가한다. 하나는 synthetic-to-real adaptation이고, 다른 하나는 cross-city adaptation이다. synthetic-to-real에서는 GTA5 또는 SYNTHIA를 source로, Cityscapes를 target으로 사용한다. cross-city에서는 Cityscapes를 source로, NTHU의 Rio, Rome, Tokyo, Taipei를 target으로 사용한다.

평가 지표는 class-wise IoU, Pixel Accuracy(PA), 그리고 전체 평균인 mIoU와 mPA이다. 표의 핵심 비교는 mIoU다.

네트워크는 두 종류를 사용한다. 하나는 ResNet-50 backbone을 가진 DeepLabV3이고, 다른 하나는 VGG-16 backbone의 SegNet이다. DAM은 SegNet에서는 encoder 뒤에 붙는다. generator와 discriminator는 DCGAN 스타일 구조와 유사한 아키텍처를 사용하고, discriminator는 conditional version으로 확장했다. generator의 latent dimension과 discriminator의 label embedding dimension은 모두 256이다.

학습은 PyTorch로 수행되며, segmentation network는 SGD with Nesterov, momentum 0.9, weight decay $10^{-4}$, 초기 learning rate $2.5\times10^{-4}$를 사용한다. generator와 discriminator는 Adam, learning rate 0.1을 사용한다. target 이미지는 $512\times256$, synthetic fake sample은 $256\times128$로 사용한다. source pretraining은 Cityscapes에서 30 epoch, GTA5/SYNTHIA에서 20 epoch, source-free adaptation은 120 epoch, batch size 8이다.

하이퍼파라미터는 기본적으로 $\alpha=1.0$, $\beta=0.5$, $\tau=\beta$, $\gamma=0.01$이다. IPSM의 patch 수 $K$는 보통 3, 4, 5 중에서 선택한다.

### 4.2 GTA5 $\rightarrow$ Cityscapes

Table 1은 GTA5에서 Cityscapes로 적응하는 대표 실험이다. DeepLabV3 기준으로 source-only는 34.09 mIoU이고, MinEnt는 40.17, AdaptSegNet은 40.49, CBST는 42.94, MaxSquare는 43.12를 기록한다. 이들 비교 기법은 모두 source-driven UDA라서 source data를 사용한다. 반면 제안한 SFDA는 source-free임에도 불구하고, IPSM 없이 41.35, IPSM 포함 시 43.16 mIoU를 달성한다.

이 결과는 두 가지 점에서 중요하다. 첫째, source-free 제약을 두고도 MaxSquare의 43.12와 거의 같은 수준, 정확히는 약간 더 높은 43.16을 얻었다는 점이다. 둘째, IPSM이 실제로 성능 향상에 기여한다는 점이다. SFDA without IPSM이 41.35이고 full SFDA가 43.16이므로, 약 1.81 mIoU 개선이 있다.

SegNet에서도 비슷한 경향이 나온다. source-only는 27.13, MinEnt 33.97, AdaptSegNet 34.16, MaxSquare 36.11인데, SFDA without IPSM은 34.43, full SFDA는 35.86이다. DeepLabV3에서만큼 최고점을 넘지는 못했지만, source-free라는 조건을 고려하면 여전히 강한 결과다. 즉 제안 방법은 특정 backbone에만 맞는 것이 아니라, 비교적 다른 segmentation network에도 적용 가능하다는 점을 보인다.

클래스별 결과를 보면, full SFDA는 sidewalk, wall, pole, sign, terrain, sky, person, bike 등 여러 클래스에서 전반적으로 개선된다. 특히 sky는 85.3으로 매우 높고, rider, bike 같은 상대적으로 어려운 클래스도 source-only보다 많이 향상된다. 다만 train 클래스는 3.6으로 매우 낮아 여전히 어려운 범주임을 알 수 있다.

### 4.3 SYNTHIA $\rightarrow$ Cityscapes

Table 2는 SYNTHIA to Cityscapes 결과다. 여기서는 16-class mIoU와 13-class mIoU*를 함께 보고한다. source-only는 29.31 / 34.36이고, MinEnt는 36.30 / 42.31, AdaptSegNet은 37.08 / 43.19, CBST는 38.88 / 45.23, MaxSquare는 39.12 / 45.65다.

제안한 SFDA는 without IPSM에서 37.50 / 44.47, full version에서 39.20 / 45.89를 달성한다. 즉 16-class mIoU 기준으로 MaxSquare 39.12보다 조금 높고, 13-class mIoU* 기준으로도 45.89로 가장 높다. 이 역시 source-free인데 source-driven SOTA급과 경쟁한다는 논문의 핵심 주장에 부합한다.

저자들은 qualitative result와 함께 small object segmentation에서의 강점을 언급한다. traffic light, traffic sign, motorbike 같은 작은 객체 클래스에서 경쟁력 있는 수치를 보인다고 주장한다. 실제 표를 보면 light, sign, motor, bike 같은 클래스는 절대값이 높지는 않지만, 여러 baseline 대비 뒤처지지 않거나 개선되는 패턴이 일부 보인다. 다만 small object 전반에서 일관되게 압도적이라고 말하기에는 클래스별 편차가 있다. 예를 들어 traffic light는 3.3으로 MaxSquare의 6.6보다 낮다. 따라서 “작은 객체 전반에서 우수하다”기보다는, 일부 작은 객체에서 경쟁력이 있다고 보는 편이 더 논문 내용에 충실하다.

### 4.4 Cross-City adaptation

Table 3은 Cityscapes를 source로 하고 NTHU의 네 도시를 target으로 사용하는 실험이다. source-only는 Rome 46.44, Rio 45.06, Tokyo 44.05, Taipei 44.07이다. MaxSquare는 각각 48.48, 48.74, 47.10, 47.16이다.

제안한 SFDA는 without IPSM에서 47.38, 47.75, 45.18, 45.38이고, IPSM 포함 시 48.33, 49.03, 46.36, 47.20이다. 즉 Rio와 Taipei에서는 MaxSquare보다 약간 높고, Rome과 Tokyo에서는 비슷하거나 약간 낮다. 저자들이 “competitive”라고 표현한 이유가 여기에 있다. 모든 도시에서 일관되게 최고는 아니지만, source-free 조건에서 source-driven 강기법과 비슷한 성능을 낸다.

또 하나 흥미로운 항목은 “transfer only”다. 이것은 SFKT를 통해 source knowledge만 새 모델에 옮기고 target adaptation은 하지 않은 경우로 보인다. 점수는 Rome 45.87, Rio 44.03, Tokyo 43.96, Taipei 43.55다. source-only보다 약간 낮지만 큰 차이가 나지는 않는다. 이는 knowledge transfer 자체가 어느 정도는 유효하지만, 실제 target adaptation과 IPSM이 결합되어야 성능이 더 오른다는 점을 보여준다.

### 4.5 Ablation study

Table 4는 SFKT 내부 구성요소의 기여를 분석한다. 각 source model 데이터셋에 대해 BNS만 사용, DAD만 사용, BNS+DAD를 비교한다.

GTA5에서는 source model이 61.8이고, BNS만 쓰면 49.8, DAD만 쓰면 55.4, BNS+DAD는 58.3이다. Cityscapes에서는 73.6 대비 BNS 60.6, DAD 65.5, BNS+DAD 70.8이다. SYNTHIA에서도 source model 62.3, BNS 51.4, DAD 54.7, BNS+DAD 59.0이다.

이 결과는 두 가지를 시사한다. 첫째, 단순 BN 통계만 맞추는 것보다 DAD가 훨씬 효과적이다. 즉 segmentation에서 contextual relation distillation이 실제로 중요하다는 논문의 주장을 뒷받침한다. 둘째, BNS와 DAD는 상보적이다. DAD만으로도 좋지만, BNS를 함께 쓰면 source-like distribution anchoring 역할이 더해져 성능이 추가 개선된다.

Figure 7에 대한 설명도 이 해석을 보강한다. 저자들은 DAD가 없으면 fake sample의 semantic map이 거칠고 작은 객체나 정교한 구조를 잘 잡지 못한다고 말한다. 반대로 BNS가 없으면 생성기가 source의 원래 semantic distribution을 잘 보존하지 못한다고 한다. 즉 BNS는 분포 복원, DAD는 구조와 문맥 보존에 더 가깝게 작용한다고 해석할 수 있다.

### 4.6 하이퍼파라미터 분석

Table 5는 $\beta=0.5$로 고정하고 $\alpha$를 바꾼 결과다. $\alpha=0.1$일 때 41.33, 0.5일 때 42.70, 1.0일 때 43.16, 2.0일 때 42.47이다. 최적은 1.0이다. 저자들은 $\mathcal{L}_{MAE}$가 target segmentation loss와 유사한 수준의 supervision 역할을 하므로, $\alpha$는 1.0 근처가 적절하다고 설명한다. 너무 작으면 source output matching이 약해지고, 너무 크면 adaptation을 지나치게 제약할 수 있다는 뜻이다.

Table 6은 $\alpha=1.0$로 고정하고 $\beta$를 바꾼 결과다. $\beta=0.01$일 때 41.54, 0.1일 때 43.09, 0.5일 때 43.16, 1.0일 때 42.47이다. 최적은 0.5다. 저자들 설명대로 DAD는 intermediate-layer constraint이므로 너무 크게 주면 intermediate representation의 자유로운 adaptation을 막을 수 있다.

Figure 8은 IPSM의 patch 개수 $K$에 대한 민감도 분석이다. 논문은 $K=1$이면 IPSM이 없는 경우라고 설명한다. 결과적으로 $K$가 너무 작아도, 너무 커도 좋지 않고, 3~5 정도가 적절하다고 결론내린다. 이는 patch가 너무 크면 easy/hard 분리가 거칠고, 너무 작으면 patch별 semantic consistency가 약해질 수 있음을 시사한다. 다만 Figure 8의 정확한 수치 자체는 제공된 텍스트에 포함되지 않았으므로, 여기서는 정성적 결론만 말할 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정 자체의 실용성이다. 기존 UDA segmentation 연구가 거의 당연하게 가정하던 “source data 접근 가능” 조건을 버리고도 의미 있는 성능을 달성했다는 점은 분명한 기여다. 특히 실제 산업 환경에서는 source model은 배포 가능하지만 source dataset은 배포 불가능한 경우가 많기 때문에, 이 문제 설정은 학문적 새로움뿐 아니라 응용적 가치도 크다.

두 번째 강점은 방법론 설계가 segmentation의 특성을 잘 반영했다는 점이다. 저자들은 classification에서 쓰던 source-free KD를 그대로 가져오지 않고, spatial attention과 channel attention을 이용해 contextual relationship을 보존하는 DAD를 만들었다. ablation 결과에서도 DAD가 BNS보다 더 큰 효과를 보인다는 점이 확인된다. 이는 segmentation adaptation에서 intermediate context alignment가 중요하다는 설계를 설득력 있게 만든다.

세 번째 강점은 target domain 활용 방식이 세밀하다는 것이다. 전체 이미지 pseudo-label은 noise가 많은데, IPSM은 patch 단위로 entropy를 보고 easy/hard를 구분해 intra-domain self-supervision을 수행한다. 단순 entropy minimization보다 더 구조적인 활용이다. 실험에서도 IPSM이 GTA5→Cityscapes에서 약 1.81 mIoU, SYNTHIA→Cityscapes에서 약 1.70 mIoU 향상을 주어 실효성이 확인된다.

반면 한계도 분명하다. 가장 명시적인 한계는 논문 결론에서 직접 밝히듯, **high-resolution image segmentation을 지원하지 못한다**는 점이다. generator가 고해상도 fake sample을 안정적으로 생성하기 어렵기 때문이다. 이 한계는 단지 구현 디테일이 아니라, 방법의 핵심이 synthetic sample generation에 크게 의존한다는 사실을 드러낸다. segmentation이 원래 고해상도 정보에 민감한 작업임을 생각하면, 실제 대규모 고해상도 응용으로 확장하는 데 제약이 있다.

또 다른 한계는 generator가 만드는 fake sample의 해석 가능성과 안정성이다. 논문은 사람이 보기에는 fake sample을 인식하기 어렵더라도 CNN 표현 공간에서는 source-like하다고 주장한다. 이는 가능한 설명이지만, synthetic sample의 품질과 다양성이 실제로 어느 정도 source domain coverage를 보장하는지는 충분히 엄밀히 검증되지는 않는다. 저자들 스스로도 “generator가 source domain을 지속적으로 정확히 복원한다고 보장하기 어렵다”고 인정하며, 그래서 IPSM을 추가했다고 설명한다. 즉 knowledge transfer stage만으로는 완결적이지 않다.

IPSM에도 가정이 있다. 이 모듈은 street-view 장면에서 비슷한 위치에 비슷한 구조가 반복된다는 점에 기대고 있다. 이는 도시 주행 영상에는 꽤 타당하지만, 모든 segmentation 문제에 일반화된다고 보기는 어렵다. 예를 들어 의료영상이나 실내 장면, aerial imagery처럼 공간적 위치 편향이 다른 도메인에서는 같은 patch 위치끼리의 유사성이 약할 수 있다. 따라서 IPSM의 일반성은 제한적일 수 있다.

실험적 비교에도 해석상 주의가 필요하다. 논문은 source-free setting인데도 source-driven SOTA와 competitive하다는 점을 강조한다. 실제로 상당히 인상적이지만, 경쟁력의 정도는 데이터셋과 backbone에 따라 다르다. DeepLabV3에서는 최고 수준이지만, SegNet이나 cross-city에서는 “압도적 우세”보다는 “비슷한 수준”에 가깝다. 따라서 이 방법을 모든 상황에서 source-driven UDA를 대체하는 일반 해법으로 보기는 어렵고, **source 데이터 접근 불가라는 강한 제약 아래에서 매우 강한 대안**으로 보는 것이 적절하다.

또 한 가지는 수식과 서술의 명확성이다. 제공된 텍스트 기준으로 Equation (15)의 표기와 일부 설명은 다소 혼란스럽다. easy patch와 hard patch가 loss 식에서 어떤 방향으로 들어가는지, discriminator와 target model의 업데이트 방향이 한눈에 명확하지 않다. 보충 자료에 자세한 알고리즘이 있다고만 되어 있어, 본문만으로는 구현 세부가 충분히 투명하지 않다. 논문 텍스트에 명시되지 않은 정확한 optimizer schedule이나 alternating update 순서도 여기서는 완전히 복원할 수 없다.

## 6. 결론

이 논문은 semantic segmentation을 위한 **source-free unsupervised domain adaptation**이라는 새로운 문제를 제기하고, 이를 해결하기 위한 SFDA 프레임워크를 제안했다. 핵심 기여는 크게 세 가지다. 첫째, source data 없이도 source model의 지식을 활용해 target adaptation을 수행하는 실용적 문제 설정을 정립했다. 둘째, segmentation의 문맥 구조를 반영한 Dual Attention Distillation으로 source knowledge transfer를 설계했다. 셋째, entropy 기반 patch-level self-supervision(IPSM)으로 target domain 내부의 유용한 신호를 추가로 활용했다.

방법론적으로 이 연구는 source model 내부의 지식을 “가짜 source-like 입력을 통해 다시 꺼내어” 전달한다는 점에서 흥미롭고, segmentation에서 attention relation을 distillation 대상으로 삼았다는 점도 의미 있다. 실험적으로는 GTA5→Cityscapes, SYNTHIA→Cityscapes, Cityscapes→NTHU 같은 표준 벤치마크에서 source-free 조건임에도 source-driven UDA와 경쟁 가능한 성능을 보였다. 이는 privacy나 상업적 제약 때문에 source data를 쓸 수 없는 실제 응용 환경에서 특히 중요하다.

향후 연구 측면에서는 고해상도 fake sample 생성 문제, IPSM의 더 일반적인 도메인에 대한 확장, generator 의존도를 줄이는 source-free transfer 방식, 그리고 보다 안정적인 pseudo-label refinement 기법과의 결합 등이 유망해 보인다. 논문 자체도 고해상도 segmentation 한계를 후속 과제로 명시하고 있다. 종합하면, 이 논문은 segmentation domain adaptation 연구에서 “source data를 반드시 들고 있어야 한다”는 전제를 깨는 출발점으로서 의미가 크며, 실제 적용 가능성과 연구 확장성 모두를 갖춘 작업이라고 평가할 수 있다.
