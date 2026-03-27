# Learning to Adapt Structured Output Space for Semantic Segmentation

* **저자**: Yi-Hsuan Tsai, Wei-Chih Hung, Samuel Schulter, Kihyuk Sohn, Ming-Hsuan Yang, Manmohan Chandraker
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1802.10349](https://arxiv.org/abs/1802.10349)

## 1. 논문 개요

이 논문은 **semantic segmentation에서의 unsupervised domain adaptation** 문제를 다룬다. 구체적으로는, 픽셀 단위 정답 라벨이 존재하는 source domain에서 학습한 segmentation 모델을, 정답 라벨이 없는 target domain에서도 잘 동작하도록 적응시키는 것이 목표다. 논문이 특히 주목하는 설정은 GTA5, SYNTHIA 같은 synthetic dataset에서 학습한 뒤, Cityscapes 같은 real-world dataset으로 일반화하는 경우다.

문제의 핵심은 **domain gap**이다. segmentation 모델은 source 이미지의 분포에는 잘 맞지만, 조명, 날씨, 도시 구조, 렌더링 스타일, texture 등이 달라지는 target domain에서는 성능이 크게 떨어진다. segmentation은 픽셀마다 정답이 필요하므로, target domain에 대해 다시 dense annotation을 만드는 비용이 매우 크다. 따라서 annotation 없이도 source에서 target으로 성능을 옮기는 방법이 실용적으로 중요하다.

기존 domain adaptation 연구는 주로 **feature space alignment**에 집중해 왔다. 그러나 저자들은 semantic segmentation에서는 feature가 appearance, shape, context 등 다양한 시각 정보를 동시에 담아야 하므로, 고차원 feature를 직접 맞추는 일이 어렵다고 본다. 대신 이 논문은 segmentation 결과 자체가 갖는 **structured output**의 성질에 주목한다. 이미지 외형은 달라도, 도로 장면이라면 하늘은 위쪽, 도로는 아래쪽, 건물은 좌우, 자동차는 도로 위에 위치하는 식의 **spatial layout**과 **local context**는 source와 target에서 상당히 비슷하다는 것이다. 이 관찰을 바탕으로, 저자들은 feature가 아니라 **output space에서 adversarial adaptation**을 수행하는 방법을 제안한다.

즉, 이 논문의 목표는 단순히 domain adaptation 성능을 조금 높이는 데 있지 않다. 보다 본질적으로는, semantic segmentation과 같은 pixel-level prediction task에서는 **출력 분포 자체를 정렬하는 것이 feature 분포를 정렬하는 것보다 더 자연스럽고 효과적일 수 있다**는 점을 보여주는 데 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 명확하다. **source와 target 이미지의 appearance는 크게 달라도, segmentation output은 구조적으로 유사하다**는 점이다. 예를 들어, 도로 장면에서 road, sidewalk, building, sky, car, person 같은 클래스의 상대적 배치와 인접 관계는 두 도메인 사이에서 어느 정도 공통적이다. 따라서 target 이미지의 segmentation prediction이 source 이미지의 prediction과 비슷한 구조적 분포를 갖도록 만들면, 별도 target annotation 없이도 adaptation이 가능하다는 것이 저자들의 직관이다.

이를 위해 저자들은 GAN 스타일의 adversarial learning을 사용한다. segmentation network가 낸 예측 결과를 discriminator에 넣고, discriminator는 이 출력이 source에서 온 것인지 target에서 온 것인지 구분한다. 그러면 segmentation network는 target 이미지에 대해서도 source처럼 보이는 segmentation distribution을 생성하도록 학습된다. 중요한 점은, 이 과정이 feature에 직접 제약을 거는 것이 아니라 **softmax prediction map 자체에 제약을 건다**는 것이다.

기존 접근과의 차별점은 두 가지다.

첫째, **adaptation의 위치**가 다르다. 기존 방법은 주로 intermediate feature를 맞추려 했지만, 이 논문은 최종 prediction 공간에서 정렬한다. segmentation에서는 output이 저차원이면서도 장면 구조를 많이 담고 있기 때문에, 오히려 더 안정적인 adaptation 신호가 될 수 있다는 주장이다.

둘째, 단일 출력만 맞추는 데서 끝나지 않고, **multi-level adversarial learning**을 도입한다. 저자들은 최상위 output에만 adversarial loss를 걸면 하위 feature까지 충분히 적응되지 않을 수 있다고 본다. 그래서 conv4와 conv5처럼 서로 다른 feature level에서 auxiliary segmentation output을 만들고, 각 output에 별도의 discriminator를 붙여 여러 단계에서 output-space adaptation을 수행한다. 이 설계는 deep supervision과 유사한 발상으로, 상위 출력뿐 아니라 더 낮은 수준의 표현까지 적응 효과가 전달되도록 한다.

정리하면, 이 논문의 핵심은 다음과 같다. **semantic segmentation의 domain adaptation은 feature를 직접 맞추기보다, 구조화된 segmentation output을 source처럼 보이게 만드는 방식이 더 효과적이며, 이를 여러 feature level에서 동시에 수행하면 더 좋아진다.**

## 3. 상세 방법 설명

### 전체 파이프라인

모델은 크게 두 부분으로 이루어진다.

첫째는 **segmentation network** $G$이고, 둘째는 **discriminator** $D_i$이다. 여기서 $i$는 multi-level adaptation에서 어느 출력 레벨에 붙는 discriminator인지를 나타낸다.

입력은 source domain 이미지 집합 ${\mathcal{I}_S}$와 target domain 이미지 집합 ${\mathcal{I}_T}$이다. source 이미지는 픽셀 정답 라벨이 있고, target 이미지는 라벨이 없다. source 이미지 $I_s$를 segmentation network에 통과시키면 source prediction $P_s = G(I_s)$가 나오고, target 이미지 $I_t$를 넣으면 target prediction $P_t = G(I_t)$가 나온다. 이 $P_s$, $P_t$는 각 픽셀마다 클래스별 softmax 확률을 갖는 map이다.

source prediction은 정답 라벨과 비교하여 supervised segmentation loss를 계산한다. 반면 target prediction은 discriminator에 넣어 source prediction과 구별되지 않도록 adversarial loss를 계산한다. 이렇게 하면 target prediction의 분포가 source prediction 분포에 가까워지도록 학습된다.

논문 Figure 2의 설명을 말로 풀면 다음과 같다. source와 target 이미지를 같은 segmentation network에 넣고, source에서는 segmentation supervision을 받고, target에서는 output map이 source처럼 보이도록 adversarial signal을 받는다. multi-level 설정에서는 이런 adaptation module을 여러 출력 단계에 중첩하여 붙인다.

### 기본 목적함수

단일 레벨 adaptation에서 전체 손실은 다음과 같다.

$$
\mathcal{L}(I_s, I_t) = \mathcal{L}_{seg}(I_s) + \lambda_{adv}\mathcal{L}_{adv}(I_t)
$$

여기서 $\mathcal{L}_{seg}$는 source 정답을 이용한 supervised segmentation loss이고, $\mathcal{L}_{adv}$는 target prediction을 source처럼 보이게 만드는 adversarial loss다. $\lambda_{adv}$는 두 손실의 균형을 조절하는 하이퍼파라미터다.

이 식은 매우 중요하다. segmentation 성능 자체를 유지하려면 source supervision이 필요하고, domain adaptation을 하려면 target output 분포를 source 쪽으로 끌어와야 한다. 따라서 논문은 supervised term과 adversarial term을 동시에 사용한다.

### Discriminator 학습

segmentation network의 출력은

$$
P = G(I) \in \mathbb{R}^{H \times W \times C}
$$

로 표현된다. 여기서 $H$, $W$는 이미지 높이와 너비, $C$는 클래스 수다. 즉, 각 픽셀 위치마다 $C$차원 softmax 확률벡터가 있다.

discriminator는 이 $P$를 입력으로 받아, 각 공간 위치 $(h,w)$에 대해 source인지 target인지 이진 분류한다. discriminator 학습 손실은 다음과 같다.

$$
\mathcal{L}_{d}(P)= -\sum_{h,w}(1-z)\log(D(P)^{(h,w,0)}) + z\log(D(P)^{(h,w,1)})
$$

논문 텍스트의 표기상 source와 target의 label 정의는 다소 혼동될 수 있으나, 의도는 분명하다. discriminator는 **source output이면 source로, target output이면 target으로 맞게 분류**하도록 학습된다. 여기서 $z$는 도메인 라벨이다.

중요한 해석은, discriminator가 segmentation 결과 맵의 **공간적 구조를 보고 도메인을 판별**한다는 점이다. 즉 단순한 global vector가 아니라 fully-convolutional discriminator를 사용해서 픽셀 주변 문맥과 공간 배치를 보존한 채 구분한다.

### Segmentation loss

source domain에서는 정답 라벨 $Y_s$가 있으므로 일반적인 픽셀 단위 cross-entropy를 쓴다.

$$
\mathcal{L}_{seg}(I_s)= -\sum_{h,w}\sum_{c\in C} Y_s^{(h,w,c)} \log(P_s^{(h,w,c)})
$$

여기서 $Y_s^{(h,w,c)}$는 source 정답의 one-hot 표현이고, $P_s = G(I_s)$는 source image에 대한 softmax output이다. 이 손실은 segmentation network가 source domain에서 올바른 semantic label을 예측하도록 만든다.

### Target에 대한 adversarial loss

target domain은 정답이 없으므로 segmentation loss를 직접 줄 수 없다. 대신 target prediction $P_t = G(I_t)$를 discriminator가 **source처럼 보이는 출력**으로 인식하게 만드는 손실을 준다.

$$
\mathcal{L}_{adv}(I_t)= -\sum_{h,w}\log(D(P_t)^{(h,w,1)})
$$

논문의 설명에 따르면 이 손실은 target prediction이 discriminator에게 source prediction으로 분류되도록 유도한다. 즉 segmentation network는 target 이미지를 넣었을 때도 source와 구조적으로 유사한 prediction map을 출력하려고 한다.

이 식의 의미를 쉬운 말로 설명하면, target 이미지에 대한 segmentation 결과가 "이건 target 특유의 이상한 구조다"라고 판별되지 않도록 만드는 것이다. 결국 네트워크는 target 도메인에서도 source에서 배운 scene layout과 local context를 재현하게 된다.

### 왜 output space adaptation이 유리한가

저자들의 논리는 다음과 같다.

feature space는 고차원이고, semantic segmentation을 위해 매우 복잡한 시각 정보를 담는다. 그래서 discriminator 입장에서는 source와 target feature를 쉽게 구분할 수 있고, segmentation network는 이를 충분히 속이기 어려울 수 있다. 반면 output space는 클래스 수 $C$에 해당하는 상대적으로 저차원 공간이지만, segmentation map 형태로 배치되어 있기 때문에 장면 구조에 대한 유용한 정보는 유지된다. 따라서 **도메인을 구분하기엔 충분하고, 동시에 정렬하기에는 feature보다 더 다루기 쉬운 공간**이라는 것이 논문의 주장이다.

### Multi-level adversarial learning

단일 output만 맞추면, segmentation network의 하위 계층까지 adaptation 신호가 약하게 전달될 수 있다. 저자들은 이를 보완하기 위해 auxiliary branch를 추가한다. conv4 feature에서 중간 segmentation output을 만들고, 여기에 별도의 discriminator를 연결한다. 최종 output에도 다른 discriminator를 연결한다. 그러면 여러 단계의 출력에 대해 각각 segmentation loss와 adversarial loss를 계산할 수 있다.

전체 손실은 다음처럼 확장된다.

$$
\mathcal{L}(I_s, I_t)=\sum_i \lambda_{seg}^{i}\mathcal{L}_{seg}^{i}(I_s)+\sum_i \lambda_{adv}^{i}\mathcal{L}_{adv}^{i}(I_t)
$$

여기서 $i$는 출력 레벨을 의미한다. 각 레벨에서 source는 supervised segmentation loss를 받고, target은 adversarial loss를 받는다. 결국 최적화 목표는 다음 min-max 문제다.

$$
\max_D \min_G \mathcal{L}(I_s, I_t)
$$

즉 discriminator는 source와 target output을 잘 구분하려 하고, segmentation network는 source segmentation을 잘 수행하는 동시에 target output이 source처럼 보이게 하려 한다.

### 네트워크 구조

#### Discriminator

discriminator는 DCGAN류 구조를 참고하지만 fully-convolutional하게 설계된다. 총 5개의 convolution layer를 사용하고, 각 layer의 kernel 크기는 $4 \times 4$, stride는 2이다. 채널 수는 순서대로 ${64, 128, 256, 512, 1}$이다. 마지막 층을 제외한 각 층 뒤에는 slope 0.2의 leaky ReLU를 사용한다. batch normalization은 작은 batch size와 joint training 환경 때문에 사용하지 않는다.

이 설계는 output map의 **공간 정보 유지**를 목표로 한다. segmentation map은 단지 클래스 histogram이 아니라 위치별 패턴이 중요하기 때문에, fully connected discriminator보다 fully-convolutional discriminator가 더 자연스럽다.

#### Segmentation network

기본 segmentation backbone은 **DeepLab-v2 + ResNet-101**이다. ImageNet으로 pretrain된 ResNet-101을 기반으로 하며, 마지막 classification layer를 제거하고 마지막 두 convolution layer의 stride를 2에서 1로 바꿔 feature map resolution을 높인다. 또한 conv4와 conv5에는 dilated convolution을 적용하여 receptive field를 키운다. 마지막 classifier로는 **ASPP (Atrous Spatial Pyramid Pooling)** 를 사용하고, 최종적으로 upsampling 후 softmax output을 입력 이미지 크기에 맞춘다.

논문은 이 baseline만으로도 Cityscapes train에서 학습하고 Cityscapes val에서 테스트했을 때 **65.1% mIoU**를 달성한다고 밝힌다. 저자들이 강한 baseline을 강조하는 이유는, adaptation 모듈의 진짜 효과를 보기 위해서다. baseline이 너무 약하면 adaptation이 좋아 보이더라도 실제 실용성은 떨어질 수 있기 때문이다.

### 학습 절차

학습은 **one-stage end-to-end joint training**으로 수행된다. 각 배치에서 다음이 반복된다.

먼저 source image $I_s$를 넣어 segmentation loss $\mathcal{L}_{seg}$를 계산하고 segmentation network를 업데이트한다. 동시에 source prediction $P_s$를 저장한다. 다음으로 target image $I_t$를 넣어 target prediction $P_t$를 얻는다. 이제 $P_s$와 $P_t$를 discriminator에 넣어 discriminator loss $\mathcal{L}_d$를 계산한다. 그리고 target prediction $P_t$에 대해 adversarial loss $\mathcal{L}_{adv}$를 계산하여 이 gradient를 segmentation network 쪽으로 역전파한다.

multi-level인 경우에는 이 절차를 각 adaptation module에 대해 반복하면 된다. 테스트 시에는 discriminator를 모두 버리고 segmentation network만 사용하므로 추가 연산 비용이 없다는 점도 장점으로 제시된다.

### 최적화 세부사항

segmentation network는 SGD with Nesterov momentum을 사용한다. momentum은 0.9, weight decay는 $5\times 10^{-4}$, 초기 learning rate는 $2.5\times 10^{-4}$이며 polynomial decay를 사용한다. discriminator는 Adam optimizer를 사용하고 learning rate는 $10^{-4}$, momentum은 0.9와 0.99를 사용한다. 구현은 PyTorch, 단일 Titan X GPU 12GB 메모리에서 수행되었다.

### LS-GAN 부록

Appendix A에서는 기존의 vanilla GAN loss 대신 **least-squares GAN (LS-GAN)** 을 실험한다. discriminator loss는

$$
\mathcal{L}_{d}^{LS}(P)=\sum_{h,w} z\left(D(P)^{(h,w,1)}-1\right)^2 + (1-z)\left(D(P)^{(h,w,0)}\right)^2
$$

이고, target adversarial loss는

$$
\mathcal{L}_{adv}^{LS}(I_t)=\sum_{h,w}\left(D(P_t)^{(h,w,1)}-1\right)^2
$$

이다.

저자들은 이 목적함수가 더 안정적인 GAN 학습과 더 좋은 결과를 줄 수 있는지 살펴본다. 본문 핵심 기여는 output space adaptation 자체이지만, 부록은 **어떤 GAN objective를 쓰느냐도 성능에 영향을 줄 수 있음**을 보여준다.

## 4. 실험 및 결과

### 실험 설정 전반

논문은 세 가지 큰 실험 축을 제시한다. 첫째는 **GTA5 $\rightarrow$ Cityscapes**, 둘째는 **SYNTHIA $\rightarrow$ Cityscapes**, 셋째는 **Cityscapes $\rightarrow$ Cross-City**다. 모두 target domain에는 학습용 annotation이 없고, 평가는 target validation 또는 annotated evaluation set에서 수행된다.

평가 지표는 IoU이며, 핵심 비교값은 **mIoU**다. 논문은 단순히 기존 방법과 수치 비교만 하지 않고, feature adaptation과 output space adaptation의 차이, single-level과 multi-level의 차이, $\lambda_{adv}$ 민감도, oracle 대비 성능 격차까지 함께 분석한다.

### GTA5 to Cityscapes

GTA5는 24,966장의 synthetic 이미지로 구성되며, Cityscapes와 호환되는 19개 클래스를 가진다. target인 Cityscapes는 train 2,975장으로 adaptation에 사용하고, validation 500장으로 평가한다.

VGG-16 backbone 기준으로 기존 방법들과 비교했을 때, 제안한 **single-level output adaptation**은 **35.0 mIoU**를 기록한다. 비교 대상 성능은 FCNs in the Wild 27.1, CDA 28.9, CyCADA(feature) 29.2, CyCADA(pixel) 34.8이다. 즉, 이 논문은 최소한 VGG 기반 공정 비교에서는 기존 방법보다 약간 더 높은 성능을 달성한다.

하지만 논문의 더 중요한 결과는 ResNet-101 strong baseline 위에서의 ablation이다. source만으로 학습한 baseline은 **36.6 mIoU**이고, feature adaptation은 **39.3**, output single-level은 **41.4**, output multi-level은 **42.4**를 기록한다. 따라서 이 논문의 핵심 주장인 **output space adaptation > feature adaptation**, 그리고 **multi-level > single-level**이 이 실험에서 분명히 확인된다.

클래스별로 보면 road, sidewalk, building, vegetation, sky, person, car 등 주요 도시 장면 클래스에서 개선이 보이며, 특히 전체적인 구조를 반영하는 클래스에서 output adaptation의 장점이 드러난다. 반면 pole, traffic sign 같은 작은 객체는 적응이 상대적으로 어렵다고 저자들이 직접 언급한다. 이는 작은 물체는 segmentation map에서 차지하는 비율이 작고, 배경과 섞이기 쉬워 adversarial alignment만으로는 충분히 회복되지 않기 때문으로 해석할 수 있다.

oracle과의 격차 분석도 중요하다. Table 2에서 ResNet-101 기반 제안 방법의 adapted 성능은 **42.4**, oracle은 **65.1**로, gap은 **-22.7**이다. VGG 기반 기존 방법들보다 gap이 작다. 저자들은 이를 근거로, adaptation만 잘 설계하는 것보다 **강한 backbone을 쓰는 것 자체가 domain gap을 줄이는 실용적 방법**이라고 주장한다.

### 파라미터 민감도 분석

$\lambda_{adv}$에 대한 분석은 논문의 설득력을 높이는 부분이다. GTA5 $\rightarrow$ Cityscapes에서 단일 레벨 설정으로 실험한 결과, feature adaptation은 $\lambda_{adv}$ 값에 매우 민감했다. 0.0005에서 35.3, 0.001에서 39.3, 0.002에서 35.9, 0.004에서 32.8로 크게 흔들린다. 반면 output space adaptation은 40.2, 41.4, 40.4, 40.1로 비교적 안정적이다.

이 결과는 저자들의 주장을 실험적으로 뒷받침한다. 즉, output space adaptation은 **학습이 더 안정적이고 하이퍼파라미터에 덜 민감**하다. segmentation처럼 복잡한 pixel-level task에서는 이것이 실제 적용에서 큰 장점이 된다.

### SYNTHIA to Cityscapes

SYNTHIA-RAND-CITYSCAPES는 9,400장의 synthetic 이미지로 구성되며, 실험은 Cityscapes validation set의 13개 클래스 기준으로 수행된다.

VGG 기반 비교에서는 제안한 single-level 방법이 **37.6 mIoU**로, FCNs in the Wild 22.9, CDA 34.8, Cross-City 35.7보다 높다. ResNet-101 기반 강한 baseline 위에서는 baseline 38.6, feature 40.8, single-level 45.9, multi-level 46.7이다. 여기서도 **feature보다 output**, **single보다 multi-level**이라는 패턴이 유지된다.

oracle gap 분석에서도 제안법이 강점을 보인다. ResNet-101 기준 adapted 46.7, oracle 71.7로 gap은 **-25.0**이다. 논문은 제안법이 oracle gap을 30% 이하로 줄인 유일한 방법이라고 강조한다. 이는 synthetic-to-real처럼 domain gap이 큰 상황에서도 structured output alignment가 꽤 강력하다는 의미다.

### Cross-City adaptation

이 실험은 synthetic-to-real보다 domain gap이 작은 실제 도시 간 적응을 본다. source는 Cityscapes train set이고, target은 Rome, Rio, Tokyo, Taipei 네 도시다. 각 도시는 annotation 없는 3,200장으로 adaptation하고, annotation 있는 100장으로 평가한다.

이 설정에서는 domain gap이 더 작기 때문에 adversarial loss weight를 더 작은 값 $\lambda_{adv}^{i}=0.0005$로 사용한다. 결과적으로 Rome에서는 baseline 50.9에서 output space adaptation 53.8, Rio는 48.2에서 51.6, Tokyo는 47.7에서 49.9, Taipei는 46.5에서 49.1로 향상된다. feature adaptation도 baseline보다 좋아지지만, 대부분의 도시에서 output space adaptation이 더 높다.

이 결과는 제안법이 synthetic-to-real처럼 큰 domain gap뿐 아니라, **실제 도시 간 미세한 분포 차이**에도 일관되게 효과가 있음을 보여준다. 즉 이 방법은 특정 데이터셋에만 맞춘 트릭이 아니라, 비교적 일반적인 adaptation 원리로 작동할 가능성이 있다.

### LS-GAN 실험

부록의 GTA5 $\rightarrow$ Cityscapes 실험에서 vanilla GAN은 **41.4 mIoU**, LS-GAN은 **44.1**이다. SYNTHIA $\rightarrow$ Cityscapes에서는 vanilla GAN이 **45.9 mIoU***, LS-GAN이 **47.6 mIoU***다. 따라서 동일한 output space adaptation 프레임워크 안에서도 GAN objective를 바꾸면 추가 성능 향상이 가능함을 보여준다.

이는 본문의 핵심 메시지와는 별개로, **output space adaptation이라는 아이디어가 GAN 손실 선택과 결합되어 더 발전할 수 있는 여지**를 보여주는 결과다.

### Synscapes 실험

Appendix B에서는 Synscapes $\rightarrow$ Cityscapes도 실험한다. Synscapes는 photorealistic synthetic dataset이라 domain gap이 GTA5나 SYNTHIA보다 작다. 그래서 without adaptation도 **45.3 mIoU**로 이미 높다. 그러나 vanilla GAN을 적용하면 **52.7**, LS-GAN은 **53.1**까지 향상된다.

이는 중요한 보조 메시지를 준다. 즉 domain gap이 아주 크지 않은 경우에도 output space adaptation은 여전히 유효하며, photorealistic synthetic data 위에서도 추가 이득을 줄 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법이 매우 잘 맞아떨어진다는 점이다. semantic segmentation은 본질적으로 structured prediction 문제이기 때문에, feature보다 output map의 구조를 정렬하겠다는 발상이 자연스럽다. 논문은 이 직관을 단지 개념적으로 제시하는 데 그치지 않고, 수식화된 adversarial loss, fully-convolutional discriminator, multi-level extension으로 구체화했다.

또 하나의 강점은 **실험 설계가 설득력 있다**는 점이다. 단순히 "우리 방법이 기존보다 좋다" 수준이 아니라, VGG 기준 공정 비교, ResNet strong baseline 위에서의 ablation, feature vs output, single vs multi-level, $\lambda_{adv}$ 민감도, oracle gap 분석까지 포함한다. 특히 output space adaptation이 feature adaptation보다 안정적이라는 메시지가 Table 3에서 잘 드러난다.

세 번째 강점은 실용성이다. 학습 시에는 discriminator가 필요하지만, 테스트 시에는 segmentation network만 사용하므로 추가 추론 비용이 없다. 또한 target domain의 라벨이나 사전지식이 필요 없다는 점도 장점이다.

그럼에도 한계는 분명하다.

먼저, 이 방법은 **source와 target의 구조적 layout이 어느 정도 공유된다는 가정** 위에 서 있다. 도로 장면처럼 레이아웃이 비교적 안정적인 문제에서는 잘 맞지만, 장면 구성이 크게 달라지는 일반적인 segmentation 문제에서도 같은 효과가 날지는 논문만으로는 확실하지 않다. 즉, 이 방법의 강점은 urban scene segmentation이라는 문제 특성에 어느 정도 의존한다.

둘째, 출력 분포를 source 쪽으로 맞춘다고 해서 항상 더 좋은 semantic correctness가 보장되는 것은 아니다. 예를 들어 target 도메인에 source에 없던 구조가 있거나, class prior가 다를 경우에는 오히려 source 스타일에 과도하게 끌려갈 수 있다. 논문은 이런 failure mode를 깊게 분석하지 않는다.

셋째, 작은 객체에 대한 적응은 여전히 어렵다. 저자들도 pole, traffic sign 같은 작은 클래스를 언급하며 background에 쉽게 섞인다고 말한다. output map 기반 alignment는 큰 구조에는 강하지만, 얇고 작은 객체의 세밀한 경계 복원에는 한계가 있을 수 있다.

넷째, multi-level adaptation은 성능을 높이지만, 왜 특정 레벨 조합이 가장 좋은지에 대한 이론적 분석은 충분하지 않다. 논문은 conv4와 최종 출력 두 단계만 사용하며 효율성과 정확성의 균형 때문이라고 설명하지만, 더 깊거나 얕은 레벨을 포함했을 때의 체계적 비교는 본문에 없다.

다섯째, adversarial learning 자체의 일반적 문제인 학습 불안정성은 완전히 해소된 것은 아니다. 다만 이 논문은 output space 쪽이 feature space보다 안정적이라고 보여준다. 부록에서 LS-GAN이 더 좋은 결과를 보인 것도, 결국 objective choice가 여전히 중요하다는 뜻이다.

비판적으로 해석하면, 이 논문은 "semantic segmentation에서 무엇을 align해야 하는가?"라는 질문에 대해 매우 강한 답을 제시한 작품이다. 다만 이 답은 universal한 정답이라기보다는, **장면 구조의 공통성이 강한 pixel-level task에서 특히 강력한 원리**라고 보는 것이 더 정확하다.

## 6. 결론

이 논문은 semantic segmentation의 unsupervised domain adaptation 문제에서, 기존의 feature alignment 중심 접근과 달리 **output space adaptation**이라는 새로운 방향을 제시했다. 핵심 주장은 segmentation output이 저차원이면서도 scene layout과 local context를 풍부하게 담고 있기 때문에, source와 target의 예측 분포를 이 공간에서 정렬하는 것이 더 효과적이라는 것이다.

방법적으로는 source supervision을 위한 segmentation loss와 target 정렬을 위한 adversarial loss를 결합했고, 이를 single-level뿐 아니라 multi-level output에 확장했다. 실험적으로는 GTA5 $\rightarrow$ Cityscapes, SYNTHIA $\rightarrow$ Cityscapes, Cross-City adaptation 등 다양한 설정에서 일관된 성능 향상을 보여 주었다. 특히 strong baseline과의 비교, oracle gap 분석, 하이퍼파라미터 민감도 분석을 통해 제안 방법의 의미를 비교적 탄탄하게 입증했다.

이 연구의 중요한 의의는 단순히 한 가지 benchmark에서 성능을 높였다는 데 있지 않다. 더 크게 보면, **structured prediction task에서는 representation보다 prediction structure 자체를 adaptation의 중심 대상으로 삼을 수 있다**는 관점을 제시했다는 데 있다. 이후 semantic segmentation domain adaptation 연구에서 output-level alignment, entropy regularization, pseudo-labeling, image translation과의 결합 등이 활발히 연구된 흐름을 생각하면, 이 논문은 그 출발점 중 하나로 볼 수 있다.

실제 적용 측면에서도, synthetic data로 학습한 모델을 real driving scene에 옮기는 문제는 자율주행과 도시 장면 이해에서 매우 중요하다. 그런 점에서 이 논문은 이론적 아이디어와 실용적 문제를 잘 연결한 연구다. 향후에는 이 접근을 더 정교한 discriminator 설계, self-training, class-wise alignment, 혹은 diffusion 기반 translation과 결합하는 방향으로 확장할 수 있을 것이다.
