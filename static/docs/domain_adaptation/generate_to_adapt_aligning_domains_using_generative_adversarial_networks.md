# Generate To Adapt: Aligning Domains using Generative Adversarial Networks

* **저자**: Swami Sankaranarayanan, Yogesh Balaji, Carlos D. Castillo, Rama Chellappa
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1704.01705](https://arxiv.org/abs/1704.01705)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 다룬다. 즉, 라벨이 있는 source domain 데이터와 라벨이 없는 target domain 데이터를 함께 사용해, target domain에서도 잘 작동하는 분류기를 학습하는 것이 목표다. 문제의 핵심은 source와 target의 데이터 분포가 다르기 때문에, source에서 잘 학습한 모델이 target에서는 성능이 크게 떨어진다는 점이다.

저자들은 이 문제를 해결하기 위해, 단순히 feature space에서 domain discrepancy를 줄이거나, 혹은 GAN으로 source-to-target 이미지를 생성한 뒤 그 이미지를 재학습에 사용하는 기존 방식과는 다른 접근을 제안한다. 이 논문은 **feature embedding 자체를 GAN의 생성-판별 학습과 결합하여, source와 target이 공유하는 joint feature space를 직접 학습**하려고 한다.

논문의 중요성은 다음과 같다. 실제 응용에서는 synthetic 데이터는 많이 만들 수 있지만, real 데이터에 라벨을 달기는 어렵다. 예를 들어 CAD로 만든 객체 이미지나 시뮬레이션 영상은 풍부하지만, 실제 사진에 대한 라벨 데이터는 부족할 수 있다. 이런 상황에서 unlabeled target data를 활용해 domain gap을 줄이는 방법은 실용성이 매우 높다. 저자들은 digits, OFFICE, synthetic-to-real이라는 서로 다른 난도의 세 가지 설정에서 실험하여, 제안한 방식이 단순한 toy setting이 아니라 더 복잡한 데이터셋에서도 잘 동작함을 보이려 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **분류를 위한 embedding network $F$와 GAN의 generator-discriminator 쌍 $G$-$D$를 서로 분리된 모듈이 아니라, 상호 보완적인 구조로 함께 학습시키는 것**이다. source 데이터는 supervised classification loss로 학습하고, source와 target 모두는 adversarial image generation 과정에 참여한다. 이때 target 데이터는 라벨이 없지만, generator와 discriminator가 만드는 gradient를 통해 embedding이 업데이트된다.

보다 직관적으로 말하면, 저자들은 embedding $F(x)$가 “좋은 분류 feature”일 뿐 아니라, generator가 source-like 이미지를 만들 수 있을 정도로 **class-consistent하고 domain-aligned된 표현**이 되기를 원한다. source 샘플의 embedding은 당연히 해당 class의 source 스타일 이미지를 만들어야 한다. target 샘플의 embedding도 generator를 통과했을 때 source-like 이미지로 이어지도록 강제하면, 결국 target embedding이 source embedding이 사는 feature manifold 쪽으로 이동하게 된다.

기존 adversarial domain adaptation 방법인 RevGrad나 ADDA는 주로 embedding space에서 domain classifier를 속이도록 학습한다. 반면 이 논문은 **pixel/image generation 경로를 통해 더 풍부한 gradient를 embedding에 전달**하려고 한다. 또 기존 GAN 기반 domain adaptation 중 일부는 “이미지를 target 스타일로 변환한 뒤 그걸 새 학습 데이터로 사용”하는 데이터 증강 관점이 강한데, 이 논문은 GAN을 데이터 증강 도구라기보다 **embedding을 정렬시키는 학습 신호 발생기**로 사용한다는 점을 차별점으로 내세운다.

특히 저자들은 OFFICE처럼 샘플 수가 적고 이미지 생성이 쉽지 않은 상황에서는, 완벽한 이미지 생성이 아니어도 충분히 유의미한 gradient를 줄 수 있다고 주장한다. 즉, 생성 품질이 다소 낮아도, generator-discriminator 경로가 embedding 정렬에 도움을 줄 수 있다는 것이 이 접근의 핵심 철학이다.

## 3. 상세 방법 설명

전체 시스템은 네 개의 네트워크로 구성된다. $F$는 입력 이미지 $x$를 $d$차원 feature embedding으로 매핑하는 encoder이고, $C$는 그 embedding을 받아 class를 예측하는 classifier다. 여기에 generator $G$와 discriminator $D$가 붙는다. 학습 시에는 두 개의 스트림이 동시에 존재한다.

첫 번째 스트림은 일반적인 supervised classification branch이다. source 이미지 $x_s$를 $F$에 넣어 embedding을 얻고, $C$가 class를 예측한다. 이 경로는 source label을 이용한 cross entropy로 학습된다.

두 번째 스트림은 adversarial branch이다. source 또는 target 이미지 $x$를 $F$에 통과시켜 얻은 embedding $F(x)$에 random noise $z$와 label encoding $l$을 concat하여 generator의 입력으로 사용한다. 논문에서 generator 입력은 다음과 같이 정의된다.

$$
x_g = [F(x), z, l]
$$

여기서 $z \in \mathbb{R}^d$는 Gaussian noise이고, $l$은 one-hot label encoding이다. source 샘플의 경우에는 실제 class label을 one-hot으로 넣는다. target 샘플은 라벨이 없기 때문에, 저자들은 $(N_c+1)$번째 **fake class**에 해당하는 one-hot encoding을 넣는다. 즉, target은 “실제 class가 무엇인지 모른다”는 사실을 반영한 특수 입력으로 generator에 들어간다.

이 논문은 conditional GAN의 변형인 **Auxiliary Classifier GAN (AC-GAN)** 구조를 사용한다. discriminator $D$는 입력 이미지가 real인지 fake인지를 판별하는 binary head $D_{data}(x)$와, 입력 이미지의 class를 예측하는 multiclass head $D_{cls}(x)$를 동시에 가진다. source 데이터에 대해서는 real/fake 판별과 class 예측을 모두 사용하고, target 데이터에 대해서는 라벨이 없으므로 real/fake 판별만 이용한다.

논문은 먼저 일반 GAN objective와 conditional GAN objective를 소개한다. 일반 GAN의 기본 목적함수는 다음과 같다.

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_{noise}}[\log(1 - D(G(z)))]
$$

conditional GAN에서는 $y$ 같은 조건 정보가 들어가며, 목적함수는 다음과 같이 바뀐다.

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x|y)] + \mathbb{E}_{z \sim p_{noise}}[\log(1 - D(G(z|y)))]
$$

하지만 제안법은 입력에 직접 class를 condition으로 넣는 고전적 conditional GAN 대신, discriminator가 auxiliary classifier를 가지는 AC-GAN 변형을 사용한다.

### 판별기 업데이트

학습은 alternating optimization으로 진행된다. 먼저 source batch와 target batch를 각각 샘플링하고, 각 이미지의 embedding을 계산한다. 그 다음 source embedding과 target embedding을 generator에 넣어 생성 이미지를 만든다.

discriminator의 전체 목적함수는 다음과 같다.

$$
L_D = L_{data,src} + L_{cls,src} + L_{adv,tgt}
$$

각 항의 의미는 다음과 같다.

첫째, $L_{data,src}$는 source real image는 real로, source embedding으로 생성한 image는 fake로 맞히도록 하는 손실이다.

$$
L_{data,src} = \max_D \frac{1}{k}\sum_{i=1}^{k} \log D_{data}(s_i) + \log(1 - D_{data}(G(f_{g_i})))
$$

여기서 $s_i$는 source image이고, $f_{g_i}$는 source embedding, noise, label을 concat한 generator 입력이다.

둘째, $L_{cls,src}$는 source real image가 올바른 class로 분류되도록 하는 손실이다.

$$
L_{cls,src} = \max_D \frac{1}{k}\sum_{i=1}^{k} \log (D_{cls}(s_i)_{y_i})
$$

즉 discriminator의 auxiliary classifier가 source class 구조를 잘 학습하도록 한다.

셋째, $L_{adv,tgt}$는 target embedding으로 생성된 이미지를 discriminator가 fake로 판별하도록 하는 항이다.

$$
L_{adv,tgt} = \max_D \frac{1}{k}\sum_{i=1}^{k} \log(1 - D_{data}(G(h_{g_i})))
$$

여기서 $h_{g_i}$는 target embedding, noise, fake-label encoding을 concat한 generator 입력이다.

### 생성기 업데이트

generator는 source 데이터에 대해서만 class-consistent하고 realistic한 이미지를 만들도록 학습된다. generator 손실은 다음과 같다.

$$
L_G = \min_G \frac{1}{k}\sum_{i=1}^{k} \left[ -\log(D_{cls}(G(f_{g_i}))_{y_i}) + \log(1 - D_{data}(G(f_{g_i}))) \right]
$$

첫 번째 항은 생성 이미지가 올바른 class로 판별되도록 강제한다. 두 번째 항은 생성 이미지가 fake가 아니라 real처럼 보이도록 유도한다. 따라서 generator는 단순히 아무 이미지나 만드는 것이 아니라, **해당 source class에 맞는 source-like 이미지**를 생성하게 된다.

### 임베딩 네트워크와 분류기 업데이트

분류기 $C$와 encoder $F$는 source classification loss로 학습된다. 동시에 $F$는 adversarial 경로를 통해 source와 target 모두로부터 추가 gradient를 받는다. 전체적으로 $F$의 목적함수는 다음과 같다.

$$
L_F = L_C + \alpha L_{cls,src} + \beta L_{F_{adv}}
$$

여기서 $L_C$는 source supervised classification loss다.

$$
L_C = \min_C \min_F \frac{1}{k}\sum_{i=1}^{k} -\log(C(f_i)_{y_i})
$$

이 항은 source 분류 성능을 직접 높인다.

그 다음 source adversarial classification 항은 다음과 같다.

$$
L_{cls,src} = \min_F \frac{1}{k}\sum_{i=1}^{k} -\log(D_{cls}(G(f_{g_i}))_{y_i})
$$

이 항은 source embedding이 generator를 거친 뒤에도 올바른 class 정보를 보존하도록 돕는다. 즉, embedding이 generator-discriminator 구조와 호환되게 정렬되도록 만든다.

마지막으로 target adversarial loss는 다음과 같다.

$$
L_{F_{adv}} = \min_F \frac{1}{k}\sum_{i=1}^{k} \log(1 - D_{data}(G(h_{g_i})))
$$

논문의 설명에 따르면 이 항은 target embedding이 generator를 통해 만든 이미지가 discriminator에게 더 real-like하게 보이도록 embedding을 업데이트한다. 직관적으로는 **target embedding을 source manifold 쪽으로 끌어당기는 역할**을 한다. target 라벨이 없기 때문에 class loss는 직접 줄 수 없지만, source에서 이미 학습된 generator의 class-conditioning 능력이 target embedding에도 간접적으로 작용하게 된다고 저자들은 설명한다.

### 왜 auxiliary classifier가 중요한가

논문은 ablation study를 통해 discriminator 안의 두 구성요소, 즉 real/fake classifier $C_1$과 auxiliary classifier $C_2$의 역할을 따로 본다. real/fake만 써도 domain alignment에 일부 도움이 되지만, auxiliary classifier가 없으면 mode collapse와 class mismatch가 심해져 성능이 떨어진다. 즉, target embedding이 source 스타일로만 맞춰질 뿐, **class-consistent하게 정렬되지 못할 위험**이 있다. auxiliary classifier는 “이 embedding이 어떤 class에 속해야 하는가”라는 구조를 generator와 discriminator 경로에 주입해 주는 핵심 장치다.

### 추론 단계

테스트 시에는 adversarial branch를 제거하고, 오직 $F$와 $C$만 사용해 분류를 수행한다. 즉 학습 시에는 GAN이 domain adaptation을 돕는 학습 장치이고, 추론 시에는 표준 분류기처럼 동작한다. 이는 실전 배치 관점에서도 단순한 장점이 있다.

## 4. 실험 및 결과

이 논문은 세 가지 큰 실험 축을 갖는다. 첫째는 단순하고 domain shift가 비교적 작은 digits adaptation, 둘째는 더 복잡하고 샘플 수가 적은 OFFICE, 셋째는 synthetic-to-real adaptation이다. 추가로 VISDA benchmark와 ablation study도 제시한다.

### Digits 실험

사용 데이터셋은 MNIST, USPS, SVHN이다. 모든 설정은 unsupervised adaptation protocol을 따른다. 즉 source만 라벨이 있고 target은 unlabeled이다. backbone으로는 modified LeNet을 사용하고, generator/discriminator는 DCGAN 스타일 구조를 사용했다.

#### MNIST ↔ USPS

MNIST와 USPS 사이 적응은 비교적 쉬운 설정이다. 논문은 두 가지 프로토콜을 쓴다. 하나는 일부 샘플만 사용하는 protocol 설정, 다른 하나는 전체 training set을 사용하는 full 설정이다.

Table 1에 따르면 제안법은 다음 성능을 보인다.

* MNIST $\rightarrow$ USPS (protocol): **92.8 $\pm$ 0.9**
* MNIST $\rightarrow$ USPS (full): **95.3 $\pm$ 0.7**
* USPS $\rightarrow$ MNIST: **90.8 $\pm$ 1.3**
* SVHN $\rightarrow$ MNIST: **92.4 $\pm$ 0.9**

비교 방법으로 RevGrad, DRCN, CoGAN, ADDA, PixelDA 등이 제시된다. 제안법은 대부분의 digits 설정에서 최고 성능을 달성하며, MNIST $\rightarrow$ USPS full에서는 PixelDA 95.9에 약간 못 미치지만 매우 근접하다. 따라서 단순 도메인뿐 아니라 다양한 digit adaptation 설정에 강하다는 점을 보여준다.

#### SVHN $\rightarrow$ MNIST

이 설정은 domain gap이 훨씬 크다. SVHN은 자연 장면에서 잘라낸 숫자 이미지라 배경, 조명, 형태 다양성이 크고, MNIST는 단순한 흑백 손글씨 숫자이기 때문이다. source-only가 60.3%인데, 제안법은 이를 **92.4%**로 끌어올린다. 논문은 이를 **32.1%p 향상**으로 설명한다. 또한 다른 비교 방법 대비 최소 10.4%p 이상 개선되었다고 주장한다.

논문은 t-SNE 시각화도 제시한다. 비적응 모델에서는 source는 어느 정도 class별 군집을 이루지만, target은 feature space에서 혼재되어 있다. 반면 적응 후에는 source와 target이 class-consistent하게 더 가까이 모인다. 이 정성 결과는 “임베딩 공간 정렬”이라는 논문의 주장을 시각적으로 뒷받침한다.

### OFFICE 실험

OFFICE는 Amazon, Webcam, DSLR 세 도메인으로 구성된 31-class 객체 분류 데이터셋이다. 샘플 수가 적고 데이터 복잡도가 높아 GAN 기반 방법에 불리하다. 저자들도 이 점을 분명히 인정하며, 이 데이터셋에서는 generator가 매우 사실적인 이미지를 잘 만들지 못한다고 말한다. 그럼에도 불구하고 이 실험을 한 이유는, **생성 품질이 완벽하지 않아도 gradient signal로서 GAN이 유용할 수 있음**을 보이기 위해서다.

feature extractor $F$는 ImageNet pretrained ResNet-50으로 초기화했고, 마지막 layer를 제거해 2048차원 embedding을 사용했다. generator는 224×224 입력을 받아도 64×64 downsampled 이미지를 생성하도록 설계되었다. 이는 고해상도 생성의 어려움을 줄이기 위한 실용적 선택으로 보인다.

Table 2 결과는 다음과 같다.

* A $\rightarrow$ W: **89.5**
* D $\rightarrow$ W: **97.9**
* W $\rightarrow$ D: **99.8**
* A $\rightarrow$ D: **87.7**
* D $\rightarrow$ A: **72.8**
* W $\rightarrow$ A: **71.4**
* 평균: **86.5**

비교 대상인 JAN의 평균이 84.3, RevGrad가 82.2, RTN이 81.6이다. 제안법은 평균적으로 최고 성능이다. 특히 어려운 transfer로 알려진 A $\rightarrow$ W, A $\rightarrow$ D, D $\rightarrow$ A, W $\rightarrow$ A에서 일관된 개선을 보인다. 이 결과는 저자들이 강조하는 핵심 주장, 즉 **이미지 생성이 완벽하지 않아도 embedding alignment에는 효과가 있다**는 점을 실험적으로 지지한다.

### Synthetic to Real 실험

이 실험은 CAD synthetic dataset을 source로, PASCAL VOC 일부를 target으로 사용하는 더 어려운 설정이다. synthetic 이미지에는 realistic background와 texture가 부족하므로, natural image manifold와의 차이가 매우 크다. 이 때문에 domain adaptation이 특히 어렵다.

이 설정에서는 pretrained VGG16의 마지막 fully connected layer를 제거한 네트워크를 $F$로 사용한다. 논문 Table 3에 따르면,

* Source only: **38.1 $\pm$ 0.4**
* RevGrad: **48.3 $\pm$ 0.7**
* RTN: **43.2 $\pm$ 0.5**
* JAN: **46.4 $\pm$ 0.8**
* Ours: **50.4 $\pm$ 0.6**

즉 baseline 대비 12.3%p 향상하며, 비교 방법보다도 높다. 이 결과는 digits나 OFFICE보다 훨씬 더 큰 domain gap에서도 방법이 일정 수준 유효함을 보여준다.

보충 실험에서는 pretrained ResNet-50을 사용한 synthetic-to-real setting도 제시한다. 여기서는 source-only 30.2, RevGrad 41.7, Ours 46.5로 보고된다. VGG16 기반 결과보다 절대 성능은 낮지만, 여전히 baseline 대비 큰 개선을 보인다. 다만 왜 backbone에 따라 이렇게 차이가 나는지에 대한 깊은 분석은 제공되지 않는다.

### VISDA challenge

VISDA는 synthetic-to-real adaptation을 위한 대규모 benchmark다. 논문은 이전 실험과 유사한 hyperparameter와 augmentation을 사용했다고 적고, classification challenge 성능을 보고한다.

* ResNet-18: 35.3 $\rightarrow$ 63.1
* ResNet-50: 40.2 $\rightarrow$ 69.5
* ResNet-152 (Val): 44.5 $\rightarrow$ 77.1
* ResNet-152 (Test): 40.9 $\rightarrow$ 72.3

상대적 gain도 70% 이상으로 매우 크다. 다만 여기서는 타 방법과의 직접 비교표가 아니라 source-only 대비 적응 후 개선 중심으로 제시된다. 따라서 “benchmark 전체에서 SOTA인지”는 이 표만으로는 단정할 수 없고, 논문도 여기서는 주로 baseline 대비 큰 성능 향상을 강조한다.

### Ablation Study

OFFICE A $\rightarrow$ W 설정에서 각 구성요소의 효과를 분석한다.

* Stream 1만 사용한 source-only: **68.4**
* Stream 1 + Stream 2의 real/fake classifier만 사용: **80.5**
* Stream 1 + Stream 2 전체 사용(real/fake + auxiliary classifier): **89.5**

이 결과는 두 가지를 보여준다. 첫째, adversarial stream 자체가 domain adaptation에 큰 기여를 한다. 둘째, auxiliary classifier가 없으면 성능이 상당히 낮아진다. 이는 class-conditional 구조가 단순 alignment보다 더 중요한 역할을 한다는 논문의 주장을 강하게 뒷받침한다.

### 생성 결과 및 noise 분석

논문은 noise dimension $d$에 대한 민감도도 분석한다. SVHN $\rightarrow$ MNIST에서 $d \in {32,64,128,256,512}$를 비교했을 때, 모든 설정이 평균 90.5% 이상을 달성한다. 즉 noise dimension에 지나치게 민감하지 않다. 다만 너무 작거나(32) 너무 큰 값(512)은 약간 비효율적이다.

생성 이미지 시각화에서는 digits에서는 생성 품질이 좋지만, OFFICE에서는 mode collapse가 있고 생성 품질이 떨어진다고 명시한다. 이는 오히려 저자들의 주장과 연결된다. **이미지 생성 품질이 낮아도 adaptation 성능은 여전히 좋을 수 있다**는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **GAN을 데이터 생성 도구가 아니라 feature alignment를 위한 학습 신호 발생기로 재해석했다는 점**이다. 이는 당시 adversarial domain adaptation 흐름 안에서 분명한 차별점이 있다. 단순히 domain classifier를 속이는 것보다, class-consistent image generation 경로를 통해 embedding에 더 구조적인 제약을 준다는 발상이 설득력 있다.

또 다른 강점은 **실험 범위의 다양성**이다. digits처럼 쉬운 설정부터, OFFICE처럼 데이터가 적고 복잡한 설정, synthetic-to-real처럼 매우 어려운 설정까지 포함한다. 그리고 대부분의 설정에서 강한 성능을 보인다. 특히 OFFICE에서 잘 작동했다는 점은, “GAN은 복잡한 실제 이미지에서는 불안정하다”는 일반적인 우려를 어느 정도 정면으로 다룬 사례로 볼 수 있다.

ablation study도 비교적 명확하다. auxiliary classifier가 실제로 중요한지, real/fake branch만으로 충분한지를 수치로 보여준다. 이런 분석은 제안한 구조의 타당성을 강화한다.

하지만 한계도 분명하다. 첫째, 방법의 핵심 설명은 직관적으로는 설득력이 있으나, **왜 target embedding이 class-consistent하게 정렬되는가**에 대한 이론적 분석은 충분하지 않다. target은 라벨이 없고 fake-class encoding으로 generator에 들어가는데, source에서 학습된 class-conditioning이 target에도 일반화될 것이라고 설명할 뿐, 그것이 언제 성립하고 언제 깨지는지는 명확히 규명하지 않는다.

둘째, 손실 함수의 부호와 최적화 서술은 일부 구간에서 다소 혼란스럽다. 예를 들어 GAN loss는 일반적으로 generator가 $\log(1-D(\cdot))$를 최소화하는 형태보다 다른 대체형을 쓰는 경우가 많은데, 본문 서술은 구현 세부나 안정화 기법을 충분히 설명하지 않는다. 논문의 핵심 아이디어를 이해하는 데는 큰 문제는 없지만, 재현 관점에서는 명료성이 다소 부족하다.

셋째, 생성 품질이 낮아도 잘 된다고 주장하지만, 그 경우 어떤 종류의 gradient가 alignment에 실제로 유효했는지에 대한 더 세밀한 분석은 없다. 예를 들어 mode collapse가 심한데도 왜 분류 성능은 좋아지는지, discriminator의 어떤 representation이 중요한지 등은 후속 연구 과제로 남는다.

넷째, 계산량과 학습 안정성 문제도 잠재적 한계다. 분류기만 학습하는 것이 아니라 $F, C, G, D$를 번갈아 최적화해야 하므로 시스템이 복잡하다. 논문은 성능을 강조하지만, training difficulty나 resource cost에 대한 비교는 거의 제공하지 않는다.

다섯째, 본문은 여러 설정에서 state-of-the-art라고 주장하지만, 일부 비교 방법들은 당시 매우 최신 방법들과 완전히 동일 조건에서 재검증된 것은 아닐 수 있다. 또한 VISDA에서는 baseline 대비 향상만 제시되고 타 최신 방법과의 직접 비교는 제한적이다. 따라서 모든 벤치마크에서 절대적 SOTA라고 일반화해서 읽기보다는, **광범위한 설정에서 강한 경쟁력을 보인 방법**으로 이해하는 것이 더 정확하다.

## 6. 결론

이 논문은 unsupervised visual domain adaptation을 위해, **분류용 embedding 학습과 GAN 기반 생성-판별 학습을 결합한 joint adversarial-discriminative framework**를 제안한다. 핵심은 source와 target을 같은 feature space로 끌어오되, 그 과정에서 generator-discriminator 쌍이 embedding에 풍부한 gradient를 제공하도록 만드는 것이다. 특히 discriminator의 auxiliary classifier를 통해 class-consistent alignment를 유도한 점이 중요한 기여다.

실험적으로는 digits, OFFICE, synthetic-to-real, VISDA 등 다양한 환경에서 좋은 성능을 보이며, 특히 생성 품질이 완벽하지 않은 경우에도 adaptation이 가능하다는 점을 강조한다. 이는 GAN을 “그럴듯한 이미지를 만드는 모델”로만 보지 않고, representation learning을 돕는 도구로 활용할 수 있음을 보여준다.

향후 연구 측면에서는 더 강한 encoder backbone, 더 안정적인 GAN 훈련 기법, 그리고 의료영상이나 RGB-D 인식처럼 더 어려운 도메인 차이 문제로 확장할 여지가 있다. 또한 후속 관점에서 보면, 이 논문은 이후의 class-conditional adversarial adaptation, feature-pixel joint adaptation, generative alignment 계열 연구로 이어지는 중요한 연결고리 역할을 한다고 볼 수 있다.
