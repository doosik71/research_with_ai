# Fully Convolutional Adaptation Networks for Semantic Segmentation

* **저자**: Yiheng Zhang, Zhaofan Qiu, Ting Yao, Dong Liu, Tao Mei
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1804.08286](https://arxiv.org/abs/1804.08286)

## 1. 논문 개요

이 논문은 semantic segmentation에서 synthetic source domain과 real target domain 사이의 domain shift를 줄이기 위한 unsupervised domain adaptation 방법을 제안한다. 문제의 배경은 분명하다. semantic segmentation은 pixel-level annotation이 필요하기 때문에 실제 데이터에 정밀한 라벨을 붙이는 비용이 매우 크다. 반면 GTA5 같은 게임 환경에서는 대량의 synthetic image와 자동 생성된 pixel-level ground truth를 얻기 쉽다. 하지만 synthetic data로 학습한 모델을 real image에 그대로 적용하면 성능이 크게 떨어진다.

논문은 이 문제를 단순히 feature alignment 하나로만 보지 않고, 두 가지 수준에서 동시에 해결하려고 한다. 첫째는 image appearance 자체를 target domain처럼 보이게 바꾸는 appearance-level adaptation이고, 둘째는 segmentation 모델 내부 표현을 두 도메인에서 구분하기 어렵게 만드는 representation-level adaptation이다. 저자들은 이 두 축을 하나의 구조로 결합한 Fully Convolutional Adaptation Networks, 즉 FCAN을 제안한다.

이 연구의 중요성은 semantic segmentation처럼 dense prediction이 필요한 문제에서 domain adaptation을 실제로 어떻게 설계해야 하는지를 구체적으로 보여준다는 점에 있다. classification과 달리 segmentation은 이미지 전체가 아니라 각 spatial location마다 정확한 semantic prediction이 필요하므로, domain discrepancy를 image-level 하나로만 다루는 방식은 충분하지 않을 수 있다. 논문은 바로 이 점을 파고들어, appearance와 representation을 함께 적응시키는 것이 더 효과적이라고 주장한다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 domain invariance를 하나의 방식으로만 만들지 않고, “보이는 모습”과 “내부 표현” 두 층위에서 동시에 구축하자는 것이다.

첫 번째 직관은 appearance-level adaptation이다. source image의 semantic content는 유지하면서, target domain의 low-level style, 예를 들어 texture, lighting, shading, color tone 같은 특성을 입히면 source image가 target domain에서 온 것처럼 보이게 만들 수 있다. 그러면 segmentation 모델이 source label을 그대로 활용하면서도 target-like visual statistics를 경험하게 되어 domain gap이 줄어든다.

두 번째 직관은 representation-level adaptation이다. source와 target의 feature representation이 domain discriminator 입장에서 구분되지 않도록 adversarial learning을 수행하면, segmentation에 필요한 semantic information은 남기면서 도메인 특유의 차이는 줄일 수 있다. 논문은 이때 discriminator를 image-level이 아니라 region-level, 즉 feature map의 각 spatial unit에 대응하는 receptive field 단위로 동작하게 설계한다. 이는 segmentation 문제의 구조에 더 잘 맞는다.

기존 접근과의 차별점은 명확하다. 논문에서 가장 가까운 관련 연구로 언급한 FCNWild는 fully convolutional adversarial training을 사용하지만, representation-level adaptation에 집중한다. 반면 FCAN은 appearance adaptation을 먼저 도입하고, 그 위에 representation adaptation을 결합한다. 즉, domain shift를 한 단계가 아니라 두 단계에서 줄이려는 설계가 차별점이다. 또한 discriminator 쪽에 ASPP를 도입하여 여러 scale의 문맥을 이용하도록 한 것도 중요한 설계 요소다.

## 3. 상세 방법 설명

FCAN은 크게 두 구성요소로 이루어진다. 하나는 Appearance Adaptation Networks (AAN), 다른 하나는 Representation Adaptation Networks (RAN)이다. 전체 흐름은 다음과 같다. 먼저 AAN이 source image를 target style에 맞게 변환하거나, 실험 설정에 따라 다른 방향의 변환을 수행한다. 그 다음 RAN이 source와 target, 또는 adaptive image를 shared FCN에 넣어 feature representation을 추출하고, segmentation loss와 adversarial loss를 함께 최적화한다.

### AAN: source content + target style

AAN의 목적은 서로 다른 domain의 이미지가 시각적으로 비슷하게 보이도록 만드는 것이다. 논문은 image style transfer의 아이디어를 차용한다. 여기서 source image의 high-level content는 보존하고, target domain 전체의 low-level statistics를 style로 간주해 adaptive image를 생성한다.

구체적으로 target domain의 이미지 집합을 $\mathcal{X}_t = {x_t^i \mid i=1,\dots,m}$, source image 하나를 $x_s$라고 두고, 출력 이미지 $x_o$를 white noise에서 시작해 반복적으로 업데이트한다. pre-trained CNN의 각 convolutional layer $l$에서 얻는 feature map을 $M^l \in \mathbb{R}^{N_l \times H_l \times W_l}$로 놓는다. 여기서 $N_l$은 channel 수이고, $H_l$, $W_l$는 feature map의 공간 크기다.

먼저 source image의 semantic content를 유지하기 위한 목적함수는 다음과 같다.

$$
\min_{x_o}\sum_{l\in L} w_s^l , Dist(M_o^l, M_s^l)
$$

여기서 $L$은 사용할 layer 집합이고, $w_s^l$는 각 layer의 가중치다. $M_o^l$와 $M_s^l$는 각각 출력 이미지 $x_o$와 source image $x_s$의 layer $l$ feature map이다. 이 항은 출력 이미지가 source image의 high-level semantics를 유지하게 만든다.

다음으로 target domain의 style을 반영해야 한다. 논문은 한 이미지의 style을 feature map 사이 correlation으로 정의한다. layer $l$에서 style statistic은 다음처럼 계산된다.

$$
G^{l,ij} = M^{l,i} \odot M^{l,j}
$$

여기서 $M^{l,i}$와 $M^{l,j}$는 각각 $i$번째, $j$번째 response map을 vectorized한 것이며, $\odot$는 inner product로 설명된다. 즉, Gram matrix와 유사한 형태로 채널 간 상관관계를 style로 본다. 그리고 target domain 전체의 style은 모든 target image의 $G^l$를 평균낸 $\bar{G}_t^l$로 정의한다.

target style을 출력 이미지에 주입하기 위한 목적함수는 다음과 같다.

$$
\min_{x_o}\sum_{l\in L} w_t^l , Dist(G_o^l, \bar{G}_t^l)
$$

최종적으로 AAN의 전체 손실은 두 항을 합친다.

$$
\mathcal{L}_{AAN}(x_o)=\sum_{l\in L} w_s^l , Dist(M_o^l, M_s^l)+\alpha \sum_{l\in L} w_t^l , Dist(G_o^l, \bar{G}_t^l)
$$

여기서 $\alpha$는 content와 style 사이의 tradeoff를 조절하는 파라미터다. 논문은 semantic segmentation이 목적이므로 content preservation이 특히 중요하다고 보고, style은 외형을 살짝 조정하는 역할로 간주한다. 그래서 $\alpha=10^{-14}$라는 매우 작은 값을 사용했다. 이 선택은 style을 강하게 바꾸기보다 semantic structure를 거의 그대로 두는 쪽에 무게를 둔 설정이라고 해석할 수 있다.

학습이라고 하기보다는 image optimization에 가깝게, AAN은 출력 이미지 $x_o$에 대해 gradient descent를 수행한다. 논문에 따르면 최대 iteration 수는 $I=1000$이고, 각 단계에서

$$
x_o^i = x_o^{i-1} - w^{i-1}\frac{g^{i-1}}{|g^{i-1}|_1}
$$

로 업데이트한다. 여기서 $g^{i-1}=\frac{\partial \mathcal{L}_{app}(x_o^{i-1})}{\partial x_o^{i-1}}$이고, $w^{i-1}=\beta \frac{I-i}{I}$, $\beta=10$이다. 즉, 초반에는 더 크게 움직이고 후반에는 점점 작은 step으로 refinement하는 방식이다.

### RAN: adversarial representation learning

AAN이 appearance를 맞춘 다음에도 domain shift가 완전히 사라지는 것은 아니므로, 논문은 feature representation 수준에서도 adaptation을 수행한다. 이를 위해 RAN을 설계했다. RAN은 shared FCN $F$와 domain discriminator $D$로 구성된다.

$F$는 source와 target image 모두에 대해 feature representation을 추출한다. 이후 segmentation branch는 pixel-level semantic prediction을 수행하고, adversarial branch는 feature map의 각 spatial unit이 source domain인지 target domain인지 판별하려 한다. 이때 discriminator의 출력은 image-level 하나가 아니라 spatial map 전체다. 즉, 각 위치마다 domain probability를 예측한다.

adversarial loss는 다음과 같다.

$$
\mathcal{L}_{adv}(\mathcal{X}_s,\mathcal{X}_t) = -E_{x_t\sim \mathcal{X}_t}\left[\frac{1}{Z}\sum_{i=1}^{Z}\log(D_i(F(x_t)))\right] -E_{x_s\sim \mathcal{X}_s}\left[\frac{1}{Z}\sum_{i=1}^{Z}\log(1-D_i(F(x_s)))\right]
$$

여기서 $Z$는 discriminator output의 spatial unit 개수이고, $D_i(F(x))$는 $i$번째 spatial location이 target domain일 확률이다. target feature는 target로 맞추고, source feature는 source로 맞추도록 discriminator를 학습시키는 일반적인 binary classification 형태다.

하지만 FCAN의 목표는 discriminator가 잘 맞히도록 하는 것이 아니라, feature extractor $F$가 discriminator를 속이도록 하는 것이다. 따라서 minimax objective는 다음과 같다.

$$
\max_F \min_D \mathcal{L}_{adv}(\mathcal{X}_s,\mathcal{X}_t)
$$

즉, discriminator $D$는 두 도메인을 잘 구분하려 하고, feature extractor $F$는 그 구분을 어렵게 만든다. 그 결과 source와 target representation이 더 비슷해진다.

### ASPP 기반 discriminator

논문은 단순 adversarial learning에서 한 걸음 더 나아가, segmentation의 특성을 고려해 multi-scale context를 discriminator에 넣는다. 여러 객체가 다양한 크기로 등장하므로, single-scale receptive field만으로는 domain discrepancy를 충분히 포착하기 어렵다고 본 것이다. 이를 위해 DeepLab의 Atrous Spatial Pyramid Pooling (ASPP)을 확장한 구조를 discriminator에 사용한다.

구체적으로 FCN output 위에 서로 다른 dilation rate를 가진 $k$개의 dilated convolution layer를 병렬로 둔다. 각 branch는 $c$개 channel을 출력하고, 모든 branch 출력을 쌓아 $ck$ 채널 feature map을 만든 뒤, $1\times1$ convolution과 sigmoid를 적용해 최종 domain score map을 얻는다. 이 논문 구현에서는 $k=4$, $c=128$이며 sampling rate는 1, 2, 3, 4다.

이 설계의 의미는 discriminator가 서로 다른 크기의 문맥 정보를 동시에 보게 해주어, 단순한 texture 차이뿐 아니라 region-level 구조 차이까지 반영할 수 있게 한다는 점이다.

### Segmentation loss와 전체 목적함수

RAN은 adversarial loss만 사용하는 것이 아니라, source domain의 label을 활용하는 supervised segmentation loss $\mathcal{L}_{seg}$를 동시에 최적화한다. 최종 목적함수는 다음과 같다.

$$
\max_F \min_D \left{\mathcal{L}_{adv}(\mathcal{X}_s,\mathcal{X}_t)-\lambda \mathcal{L}_{seg}(\mathcal{X}_s)\right}
$$

여기서 $\lambda$는 segmentation loss와 adversarial loss 사이의 tradeoff parameter다. 이 식은 사실상 $F$가 segmentation은 잘 하면서도 domain discriminator를 속이는 representation을 배우도록 만드는 구조다. test stage에서는 target image를 학습된 FCN에 넣어 pixel-level classification을 수행한다.

### 구현 세부사항

AAN에서는 pre-trained ResNet-50을 사용하고, layer 집합으로 $L={\text{conv1}, \text{res2c}, \text{res3d}, \text{res4f}, \text{res5c}}$를 선택한다. 이는 서로 다른 semantic scale의 정보를 담는다고 본 것이다.

RAN에서는 ResNet-101 기반 dilated FCN을 backbone으로 사용한다. 마지막 convolutional layer인 res5c의 feature map을 segmentation branch와 adversarial branch에 동시에 넣는다. segmentation branch에는 Pyramid Pooling을 추가하여 contextual prior를 통합한다.

학습 전략은 두 단계다. 먼저 source domain에서 segmentation loss만 사용해 RAN을 pre-train한다. 이때 초기 learning rate는 0.0025이고, “poly” learning rate policy를 쓰며 power는 0.9다. momentum은 0.9, weight decay는 0.0005, batch size는 6, 최대 iteration은 30k다. 이후 segmentation loss와 adversarial loss를 함께 사용해 fine-tuning한다. 이때 $\lambda=5$, 초기 learning rate는 0.0001, batch size는 8, 최대 iteration은 10k다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

주요 실험은 GTA5에서 Cityscapes로의 adaptation이다. GTA5는 24,966장의 game frame과 pixel-level ground truth를 포함하며, 해상도는 $1914 \times 1052$다. Cityscapes는 urban street scene benchmark로, 5,000장의 고품질 annotated image를 포함하고 해상도는 $2048 \times 1024$다. 평가에는 19개 semantic class를 사용하며, 논문은 기존 설정을 따라 Cityscapes validation set 500장을 unsupervised semantic segmentation 평가에 사용한다.

추가 실험으로 BDDS도 target domain으로 사용한다. 이 데이터셋은 dashcam frame들로 구성되며, label space가 Cityscapes와 호환된다. 해상도는 $1280 \times 720$이고, 평가에 1,500 frame을 사용한다.

평가 지표는 category별 IoU와 전체 평균인 mIoU다.

### AAN의 효과

논문은 먼저 AAN을 어디에 적용하는 것이 좋은지 실험한다. source image를 target style로 바꾸는 경우를 Src_Ada, target image를 source style로 바꾸는 경우를 Tar_Ada라 부른다. 그리고 FCN만 사용하는 경우와 RAN까지 붙이는 경우를 비교한다.

결과를 보면, 아무 adaptation 없이 source로 학습한 FCN을 target에 바로 적용하면 mIoU가 29.15%이고, 여기에 RAN만 사용하면 44.81%다. AAN을 source image에 적용한 뒤 RAN을 결합하면 46.21%를 얻어 가장 좋다. target adaptation이나 양방향 adaptation도 baseline보다는 좋아지지만, source adaptation이 가장 효과적이다. 네 가지 설정의 score map을 late fusion하면 46.60%까지 올라간다.

이 결과는 appearance-level adaptation이 의미 있고, representation-level adaptation과 상호보완적이라는 점을 보여준다. 특히 target image를 adaptation할 때는 object boundary 근처에 noise가 합성될 수 있어 segmentation stability에 악영향을 줄 수 있다고 저자들은 해석한다. 이는 논문 본문에 명시된 설명이며, 정확한 원인 규명 실험까지 제시되지는 않았다.

### FCAN 구성 요소에 대한 ablation study

논문은 FCAN의 각 설계가 얼마나 기여하는지도 단계적으로 분석한다. baseline FCN은 29.15%다. 여기에 Adaptive Batch Normalization (ABN)을 적용하면 35.51%로 크게 오른다. 이는 단순한 normalization statistics 교체만으로도 domain shift 완화에 효과가 있음을 보여준다.

그 다음 Adversarial Domain Adaptation (ADA)을 더하면 41.29%가 된다. 여기서 discriminator는 image-level domain classifier다. 다시 이를 region-level 판별 구조로 확장한 Conv 설계를 넣으면 43.17%가 된다. 이후 ASPP를 추가한 discriminator로 44.81%가 된다. 마지막으로 appearance adaptation인 AAN까지 더한 최종 FCAN은 46.60%다.

즉, RAN에 해당하는 ADA, Conv, ASPP가 합쳐서 총 9.3%p의 큰 향상을 주고, AAN이 추가로 1.79%p를 더해 준다. 이 수치는 논문의 주장, 즉 representation adaptation이 큰 축이고 appearance adaptation이 추가적인 성능 향상을 주는 보완적 역할이라는 점을 뒷받침한다.

### 기존 방법과의 비교

Cityscapes에서 state-of-the-art unsupervised domain adaptation 방법과 비교한 결과, Domain Confusion은 37.64%, ADDA는 38.30%, FCNWild는 42.04%, FCAN은 46.60%를 기록했다. multi-scale testing 또는 scheme을 적용한 FCAN(MS)은 47.75%까지 올라간다.

이 비교는 몇 가지 시사점을 준다. 첫째, image-level discriminator를 쓰는 DC와 ADDA보다 region-level adversarial learning을 쓰는 FCNWild와 FCAN이 훨씬 낫다. 둘째, FCAN은 FCNWild 대비 추가적으로 AAN과 ASPP를 사용하여 더 높은 성능을 얻는다. 논문은 category별 결과에서도 19개 클래스 중 17개에서 최고 성능을 기록했다고 설명한다.

### Domain discriminator 시각화

논문은 domain discriminator prediction map도 시각화한다. 밝을수록 해당 region이 target domain일 확률이 높다는 뜻이다. adversarial learning이 잘 작동하면 target image조차 discriminator가 헷갈려야 하므로, 이상적으로는 map이 어두워져야 한다고 논문은 설명한다.

시각화 사례에서 sky 영역은 discriminator가 헷갈리는 편이며 segmentation도 잘 된다. 반면 bicycle 같은 영역은 discriminator가 여전히 정확히 domain을 구분하여, segmentation 성능도 떨어진다. 이 분석은 adversarial alignment가 클래스나 region에 따라 다르게 작동할 수 있음을 보여준다.

### Semi-supervised adaptation

논문은 target domain에 일부 labeled image가 있을 때의 semi-supervised setting도 실험한다. 이 경우 target labeled set $\mathcal{X}_t^l$에 대한 segmentation loss를 추가하여 전체 목적함수를

$$
\max_F \min_D \left{\mathcal{L}_{adv}(\mathcal{X}_s,\mathcal{X}_t)-\lambda_s \mathcal{L}_{seg}(\mathcal{X}_s)-\lambda_t \mathcal{L}_{seg}(\mathcal{X}_t^l)\right}
$$

로 확장한다.

결과를 보면, target labeled image 수가 50장일 때 supervised FCN은 47.57%인데 semi-supervised FCAN은 56.50%다. 100장일 때는 54.41% 대 59.95%, 200장일 때는 59.53% 대 63.82%다. labeled target data가 많아질수록 두 방법 모두 좋아지지만, FCAN의 이득은 소량 라벨 환경에서 특히 크다. 1000장까지 늘려도 FCAN이 69.17%, FCN이 68.05%로 여전히 약간 우세하다.

즉, 이 방법은 완전한 unsupervised adaptation뿐 아니라, 소량의 target annotation이 있는 현실적 상황에서도 유용하다는 점을 보여준다.

### BDDS 결과

BDDS를 target domain으로 사용할 때, FCNWild는 39.37%, FCAN은 43.35%다. multi-scale 버전인 FCAN(MS)은 45.47%, 여기에 ensemble까지 추가한 FCAN(MS+EN)은 47.53%를 기록한다.

논문 초록에서는 BDDS에서 unsupervised setting으로 47.5% mIoU의 새로운 기록을 얻었다고 말하는데, 본문 Table 5의 수치 47.53%와 일관된다. 즉, 이 수치는 single model FCAN이 아니라 multi-scale과 ensemble까지 포함한 최종 강한 설정의 결과다. 이런 점은 결과를 읽을 때 구분할 필요가 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation을 appearance-level과 representation-level로 나누어 구조적으로 정리하고, 이를 실제 semantic segmentation 시스템으로 구현했다는 점이다. 단순히 “GAN을 붙였다” 수준이 아니라, 왜 image-level만으로는 부족할 수 있는지, 왜 region-level adversarial learning이 segmentation에 더 맞는지, 왜 appearance transfer가 source label 활용에 도움이 되는지까지 비교적 명확한 설계 논리로 제시한다.

두 번째 강점은 실험 설계가 체계적이라는 점이다. AAN 적용 방향, 각 구성 요소의 기여도, Cityscapes와 BDDS 두 타깃 도메인, semi-supervised 확장까지 폭넓게 보여준다. 특히 ablation study가 잘 되어 있어서 어떤 성능 향상이 어디서 오는지 파악하기 쉽다.

세 번째 강점은 segmentation 문제의 구조를 잘 반영한 discriminator 설계다. image-level 대신 region-level domain discrimination을 수행하고, ASPP를 통해 multi-scale context를 본다는 점은 dense prediction task에 더 적합한 접근으로 보인다.

반면 한계도 분명하다. 첫째, AAN은 white noise에서 시작해 반복적으로 adaptive image를 만드는 optimization 기반 방식이다. 이는 계산 비용이 클 가능성이 높고, 대규모 데이터에 대해 얼마나 실용적인지는 본문에서 깊게 논의되지 않는다. 실제 inference-time overhead가 아니라 training preparation 단계 비용일 수 있지만, 효율성 측면의 정량 비교는 제시되지 않았다.

둘째, AAN의 tradeoff parameter $\alpha$가 $10^{-14}$처럼 매우 작게 설정되어 있다. 이는 content preservation을 강하게 우선시한 설정이지만, 왜 이 값이 적절한지에 대한 체계적인 민감도 분석은 본문에 없다. 즉, style transfer가 실질적으로 어느 정도 강하게 적용되는지 정량적으로 파악하기는 어렵다.

셋째, domain alignment가 모든 category에 균일하게 잘 되는 것은 아니다. 논문이 직접 보여주듯 sky 같은 영역은 잘 적응되지만 bicycle처럼 더 어려운 클래스에서는 discriminator가 여전히 domain-dependent representation을 구분해낸다. 이는 adversarial alignment가 class imbalance나 fine-structure object에 취약할 수 있음을 시사한다.

넷째, BDDS 최고 성능은 multi-scale과 ensemble을 결합한 결과다. 따라서 FCAN 핵심 아이디어 자체의 순수 효과와 추가적인 engineering boost를 구분해서 봐야 한다. 물론 본문은 이를 표에서 구분해서 제시하므로 과장했다고 보기는 어렵지만, headline 숫자를 해석할 때 주의가 필요하다.

다섯째, 논문은 source-to-target adaptation을 주로 다루지만, adaptation이 왜 특정 방향에서 더 유리한지에 대한 이론적 설명은 제한적이다. 예를 들어 왜 source adaptation이 target adaptation보다 더 잘 작동하는지에 대해 boundary noise 가설을 제시하지만, 이를 검증하는 별도의 분석은 없다.

종합하면, 이 논문은 설계와 실험 모두 탄탄하지만, appearance adaptation의 효율성과 안정성, 그리고 클래스별 alignment 한계에 대한 후속 연구 여지는 분명히 남겨 둔다.

## 6. 결론

이 논문은 semantic segmentation을 위한 unsupervised domain adaptation 문제를 다루며, appearance-level adaptation과 representation-level adaptation을 결합한 FCAN을 제안했다. AAN은 source image의 semantic content를 유지하면서 target domain style을 입혀 시각적 격차를 줄이고, RAN은 adversarial learning을 통해 source와 target representation을 구분하기 어렵게 만든다. 또한 discriminator를 region-level로 설계하고 ASPP를 도입하여 segmentation task에 맞는 multi-scale adversarial learning을 구현했다.

실험적으로 GTA5에서 Cityscapes, 그리고 BDDS로의 adaptation에서 강한 성능을 보였고, ablation study를 통해 각 설계의 기여도도 설득력 있게 보여주었다. 특히 source-only FCN 대비 매우 큰 mIoU 향상을 달성했고, semi-supervised setting에서도 유의미한 이점을 유지했다.

이 연구의 의미는 semantic segmentation 같은 dense prediction 문제에서 domain adaptation을 어떻게 설계해야 하는지에 대해 실질적인 방향을 제시했다는 데 있다. 이후 연구에서는 이 논문의 아이디어를 더 발전시켜, 더 효율적인 image translation, 더 안정적인 adversarial alignment, class-aware adaptation, self-training과의 결합 같은 방향으로 확장할 수 있다. 실제 응용 측면에서도 synthetic data를 적극 활용해야 하는 자율주행, 로보틱스, 시뮬레이션 기반 학습 환경에서 중요한 기반 아이디어로 작동할 가능성이 크다.
