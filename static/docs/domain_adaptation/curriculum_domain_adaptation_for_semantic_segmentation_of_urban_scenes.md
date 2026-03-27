# Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes

* **저자**: Yang Zhang, Philip David, Boqing Gong
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1707.09465](https://arxiv.org/abs/1707.09465)

## 1. 논문 개요

이 논문은 synthetic urban scene 데이터로 학습한 semantic segmentation 모델을 real urban scene에 더 잘 일반화시키기 위한 unsupervised domain adaptation 문제를 다룬다. 구체적으로는 SYNTHIA 또는 GTA 같은 합성 도시 장면 데이터에는 풍부한 pixel-wise annotation이 존재하지만, 실제 도시 주행 영상 데이터인 Cityscapes와는 texture, viewpoint, lighting, object appearance 등에서 큰 domain gap이 존재한다. 이 때문에 source domain에서만 학습한 segmentation network는 target domain에서 성능이 크게 저하된다.

저자들이 주목한 핵심 문제는, 기존 domain adaptation이 주로 classification이나 regression처럼 비교적 단순한 예측 문제에서 잘 작동해 온 반면, semantic segmentation은 픽셀 단위의 highly structured prediction이기 때문에 같은 방식으로 풀기 어렵다는 점이다. 기존 방법은 보통 source와 target이 어떤 공통의 feature space에서 비슷한 분포를 갖도록 만들고, 그 위에서 같은 prediction function $P(Y \mid Z)$가 작동한다고 가정한다. 그러나 저자들은 segmentation처럼 출력 구조가 복잡한 문제에서는 이 가정이 약하며, feature alignment만으로는 오히려 구조적 단서를 망가뜨릴 수 있다고 본다.

이 논문의 목표는 이 어려운 문제를 정면으로 풀기보다, 먼저 더 쉬운 target-domain 추론 과제를 해결하고, 그 결과를 이용해 segmentation network를 간접적으로 regularize하는 것이다. 저자들은 이를 curriculum domain adaptation이라고 부른다. 먼저 target image 전체의 label distribution과 landmark superpixel의 local label distribution 같은 상대적으로 쉬운 속성을 예측하고, 이후 segmentation network가 target image에서 내는 예측이 이 속성과 일치하도록 학습시킨다. 즉, pixel-wise label이 없는 target domain에 대해, “정답 마스크 자체”는 모르더라도 “이 이미지에는 road, building, sky가 대략 어느 비율로 존재해야 하는지”, “이 특정 superpixel은 building일 가능성이 매우 높다” 같은 약한 구조적 지식을 먼저 얻고, 이를 hard task인 segmentation에 활용한다.

이 문제는 자율주행을 포함한 실제 응용에서 중요하다. semantic segmentation은 scene understanding의 핵심 요소이며, 현실 데이터에 대한 dense annotation 비용은 매우 높다. 따라서 합성 데이터를 최대한 활용하면서 real-world 성능을 끌어올리는 방법은 실용적 가치가 크다.

## 2. 핵심 아이디어

논문의 중심 직관은 “semantic segmentation 자체는 어렵지만, 그보다 더 거친 수준의 target-domain 속성은 상대적으로 쉽고 domain gap에도 덜 민감하다”는 것이다. 도시 주행 장면은 임의 자연 이미지와 달리 강한 구조적 규칙성을 가진다. 예를 들어 road는 보통 아래쪽 큰 영역을 차지하고, building은 양옆과 위쪽에 많으며, sky는 상단에 위치한다. car, sidewalk, vegetation의 상대적 크기와 위치에도 일정한 패턴이 있다. 저자들은 이 장면 구조의 규칙성이 domain을 넘어 어느 정도 유지된다고 본다.

따라서 이 논문은 먼저 두 가지 쉬운 과제를 푼다. 첫째는 이미지 전체 수준의 global label distribution을 추정하는 것이다. 이는 각 클래스가 이미지 전체 픽셀 중 몇 퍼센트를 차지하는지를 나타낸다. 둘째는 신뢰도 높은 일부 superpixel에 대해 local label distribution을 추정하는 것이다. 여기서는 실제로 superpixel 하나에 대해 dominant class를 one-hot distribution처럼 사용한다. 이렇게 얻은 분포들은 segmentation 정답 전체를 복원해 주지는 않지만, 최소한 모델이 target domain에서 너무 비정상적인 출력을 내지 않도록 제약을 준다.

기존 접근과의 차별점은 크게 두 가지다. 첫째, intermediate feature를 source와 target 사이에서 직접 맞추는 대신, output-side regularization을 이용한다. 즉, hidden feature가 domain-invariant해야 한다는 강한 가정 대신, target prediction이 가져야 할 필요한 성질을 만족하도록 한다. 둘째, structured prediction 문제의 성격에 맞게 posterior regularization 형태를 택했다. 이 방식은 segmentation network의 구조를 바꾸지 않고 loss만 수정하면 되므로 비교적 간단하게 적용할 수 있다.

또 하나 중요한 점은, image-level distribution과 superpixel-level distribution이 서로 보완적이라는 것이다. 이미지 전체 분포는 “무엇이 얼마나 있어야 하는가”를 알려 주고, landmark superpixel은 “어디를 어떻게 수정해야 하는가”를 알려 준다. 저자들은 이 두 종류의 쉬운 신호를 함께 사용하면, target domain에서 더 균형 잡히고 공간적으로 타당한 segmentation 결과를 얻을 수 있다고 주장한다.

## 3. 상세 방법 설명

논문의 방법은 source의 dense supervision과 target의 inferred property를 함께 사용하는 학습 프레임워크로 정리할 수 있다.

먼저 표기부터 정리하면, target image $I_t$의 ground-truth pixel label은 $Y_t \in \mathbb{R}^{W \times H \times C}$이고, 각 픽셀 $(i,j)$에 대해 클래스 $c$에 해당하면 $Y_t(i,j,c)=1$인 one-hot encoding을 사용한다. segmentation network의 예측은 $\widehat{Y}_t(i,j,c) \in [0,1]$이며, 픽셀별 softmax 출력이다.

저자들이 사용한 핵심 target property는 클래스 분포 $p_t \in \Delta$이다. 여기서 $p_t(c)$는 클래스 $c$가 전체 이미지 또는 특정 superpixel 내에서 차지하는 비율을 의미한다. 이미지 전체에 대한 global label distribution은 다음과 같이 정의된다.

$$
p_t(c)=\frac{1}{WH}\sum_{i=1}^{W}\sum_{j=1}^{H}Y_t(i,j,c), \quad \forall c
$$

이 식은 매우 직관적이다. 이미지 안에서 클래스 $c$에 속하는 픽셀 수를 전체 픽셀 수 $WH$로 나눈 것이다. 마찬가지로 network prediction으로부터도 예측 분포 $\widehat{p}_t$를 계산할 수 있다.

저자들의 기본 아이디어는, target에서 사람이 만든 정답을 직접 모르더라도, target prediction의 분포 $\widehat{p}_t$가 추정된 target property $p_t$와 비슷해지도록 만드는 것이다. 이를 위해 cross-entropy 형태의 항을 최소화한다.

$$
\mathcal{C}(p_t,\widehat{p}_t)=H(p_t)+KL(p_t,\widehat{p}_t)
$$

여기서 $H(p_t)$는 엔트로피이고, $KL(p_t,\widehat{p}_t)$는 KL divergence이다. 학습 시에는 $p_t$가 고정된 target-side teacher signal처럼 작동하므로, 실제 최적화 관점에서는 $p_t$와 $\widehat{p}_t$ 사이의 cross-entropy를 줄이는 효과를 낸다. 쉽게 말해, network가 target image에 대해 예측하는 클래스 비율이 추정된 분포와 맞아떨어지도록 유도하는 것이다.

최종 학습 목적함수는 source supervised loss와 target property matching loss를 함께 포함한다.

$$
\min ;\dfrac{\gamma}{|S|}\sum_{s\in S}\mathcal{L}(Y_s,\widehat{Y}_s)
+\frac{1-\gamma}{|T|}\sum_{t\in T}\sum_{k}\mathcal{C}(p_t^k,\widehat{p}_t^k)
$$

여기서 첫 번째 항 $\mathcal{L}$은 labeled source image에 대한 pixel-wise cross-entropy loss이다. 즉, segmentation 네트워크의 기본적인 픽셀 분류 능력은 source에서 배운다. 두 번째 항은 unlabeled target image에 대해 여러 종류의 label distribution $k$를 맞추는 regularization이다. $\gamma \in [0,1]$는 두 손실의 상대적 비중을 조절한다. $k$는 이 논문에서는 크게 두 가지, 즉 global image distribution과 local landmark superpixel distribution을 의미한다.

이제 중요한 것은 target property $p_t^k$를 어떻게 얻느냐이다. target에는 정답 label이 없으므로 직접 계산할 수 없고, source에서 학습한 별도의 방법으로 추정해야 한다.

### 3.1 Global label distribution 추정

첫 번째 쉬운 과제는 target image 전체에 대한 global label distribution을 예측하는 것이다. 저자들은 segmentation network가 target에서 흔히 road를 sidewalk나 car로 과대분류하는 등 disproportional prediction을 낸다고 관찰했다. 이를 바로잡기 위해, 이미지 전체에서 어떤 클래스가 어느 정도 비율로 있어야 하는지를 예측한다.

이를 위해 Inception-ResNet-v2 feature를 이미지 표현으로 사용한 뒤, 다음과 같은 방법을 비교한다.

첫째는 multinomial logistic regression이다. 일반적인 one-hot 분류 대신, source image의 ground-truth label distribution $p_s$를 target value로 사용해 학습한다. 즉, LR의 출력이 곧 클래스 분포 예측이 되도록 만든다.

둘째는 nearest neighbors 방식이다. target image의 feature와 가장 가까운 source image들을 찾고, 그들의 label distribution 평균을 target의 분포로 전이한다. 거리로는 $\ell_2$를 사용한다.

비교를 위해 모든 source 이미지 분포의 평균을 언제나 출력하는 source mean, 그리고 uniform distribution도 control 실험으로 넣었다.

결과적으로 logistic regression이 가장 정확한 global distribution predictor로 선택되었다. 이는 pixel-wise segmentation보다 훨씬 간단한 문제이면서도 target에 의미 있는 전역 제약을 제공한다는 점을 보여 준다.

### 3.2 Landmark superpixel의 local label distribution 추정

이미지 전체 분포는 전체 클래스 비율은 맞춰 주지만, 공간적 정보를 충분히 제공하지 못한다. 이를 보완하기 위해 저자들은 superpixel 수준의 local property를 도입한다. 다만 모든 superpixel을 다 사용하는 것은 위험하다고 본다. 이유는 추정이 부정확한 superpixel까지 모두 강하게 regularize하면 source supervision에서 배운 픽셀 수준 식별력을 해칠 수 있기 때문이다.

그래서 각 이미지를 linear spectral clustering으로 100개의 superpixel로 나눈 뒤, source의 superpixel마다 dominant label을 부여하고, 이를 기반으로 multi-class linear SVM을 학습한다. target superpixel을 입력하면 SVM은 예측 클래스와 decision value를 출력한다. 이 점수를 confidence로 해석하여 상위 60%만 골라 landmark superpixel로 사용한다.

landmark superpixel에 대해서는 예측된 단일 클래스 라벨을 one-hot vector로 바꾸어 local label distribution처럼 사용한다. 즉, 어떤 landmark superpixel이 building으로 분류되었다면, 그 superpixel의 분포는 building 클래스에 확률 1을 둔 것으로 본다.

superpixel feature는 시각적 정보와 문맥 정보를 같이 담도록 설계된다. 먼저 PASCAL CONTEXT로 사전학습된 FCN-8s를 사용해 각 픽셀의 59차원 detection score를 얻고, 이를 superpixel 내부에서 평균한다. 그리고 현재 superpixel뿐 아니라 왼쪽, 오른쪽, 위쪽 두 개, 아래쪽 두 개의 인접 superpixel 표현을 이어 붙여 최종 feature를 만든다. 즉, 단일 패치 모양뿐 아니라 주변 맥락도 함께 SVM이 활용하도록 설계했다.

### 3.3 전체 학습 흐름

전체 파이프라인은 다음과 같이 이해할 수 있다.

먼저 source domain의 이미지와 dense label로 segmentation network를 학습할 수 있는 기반을 만든다. 동시에 source 정보를 이용해 두 종류의 쉬운 교사 모델을 준비한다. 하나는 image feature로부터 global label distribution을 예측하는 모델이고, 다른 하나는 superpixel feature로부터 dominant class를 예측하는 SVM이다.

그 다음 unlabeled target image가 들어오면, global distribution predictor가 이미지 전체 클래스 비율을 추정하고, superpixel classifier가 신뢰도 높은 landmark superpixel의 클래스를 추정한다. segmentation network는 target image에 대한 픽셀별 예측을 내고, 이 예측으로부터 image-level 및 superpixel-level 분포를 계산한다. 그리고 이 값들이 위에서 추정한 target property와 일치하도록 추가 손실을 받는다.

저자들은 이 과정을 model compression의 teacher-student 관계에 비유한다. 쉬운 과제를 푸는 모델들이 teacher처럼 target domain의 약한 지식을 제공하고, segmentation network가 student로서 이를 따르며 harder task를 푼다는 의미다. 또한 posterior regularization의 관점에서도, target property가 network output posterior에 부과되는 구조적 제약이라고 해석한다.

이 방법의 실용적 장점은 네트워크 구조를 바꾸지 않고 output loss만 수정하면 된다는 점이다. 논문에서도 intermediate feature regularization을 추가하는 기존 deep adaptation과 달리, 자신들의 방법은 다른 segmentation backbone에도 쉽게 적용 가능하다고 강조한다.

## 4. 실험 및 결과

실험은 주로 SYNTHIA $\rightarrow$ Cityscapes 설정에서 수행되며, 논문 말미에는 GTA $\rightarrow$ Cityscapes 추가 실험도 보고된다. segmentation backbone은 FCN-8s이고, convolution layer는 VGG-19로 초기화했다. optimizer는 AdaDelta를 사용했다. 기본 학습 시 mini-batch는 source 5장, target 5장으로 구성한다. 구현은 Keras와 Theano, 하드웨어는 Tesla K40 GPU 단일 장비이다.

### 데이터셋과 평가 설정

Cityscapes는 실제 차량 주행 시점(real-world, vehicle-egocentric)의 도시 장면 데이터셋이며, fine annotation이 있는 train, val, test split과 coarse annotation이 있는 auxiliary set을 포함한다. SYNTHIA는 synthetic urban scene 데이터셋이고, 이 논문에서는 특히 SYNTHIA-RAND-CITYSCAPES subset을 사용한다. 이는 Cityscapes와 대응되는 도시 장면을 synthetic하게 생성한 데이터다.

실험 목적은 real-world urban scene segmentation이므로 source는 SYNTHIA, target은 Cityscapes이다. 테스트는 Cityscapes validation set에서 수행했고, Cityscapes training set 중 500장을 별도 validation 용도로 분리하여 학습 중 모니터링에 사용했다. 두 데이터셋 사이에는 texture 반복, 카메라 시점, 조명 조건 등 뚜렷한 domain mismatch가 존재한다.

클래스는 두 데이터셋에 공통으로 존재하는 16개를 수동으로 맞추어 사용했다. 예를 들어 sky, building, road, sidewalk, fence, vegetation, pole, car, traffic sign, person, bicycle, motorcycle, traffic light, bus, wall, rider가 포함된다.

평가 지표는 PASCAL VOC 스타일의 IoU이다.

$$
\text{IoU}=\frac{\text{TP}}{\text{TP}+\text{FP}+\text{FN}}
$$

여기서 TP, FP, FN은 전체 테스트셋에서의 true positive, false positive, false negative 픽셀 수이다.

### 4.1 Global label distribution 추정 성능

논문은 먼저 쉬운 과제 자체가 잘 풀리는지 검증한다. Cityscapes validation image에서 예측 분포와 실제 분포 사이의 $\chi^2$ distance를 측정했다. 결과는 다음과 같다.

Uniform은 1.13으로 가장 나빴고, source만으로 학습한 baseline segmentation network의 출력을 이용한 NoAdapt는 0.65였다. source mean은 0.44, nearest neighbors는 0.33, logistic regression은 0.27로 가장 좋았다.

이 결과는 몇 가지 의미를 갖는다. 첫째, source-only segmentation model의 target output은 실제로 전역 분포 관점에서도 많이 왜곡되어 있다. 둘째, global distribution 자체는 segmentation보다 훨씬 더 쉽게 예측 가능하다. 셋째, 저자들이 이후 실험에서 logistic regression 기반 image-level property를 사용하는 선택이 실험적으로 정당화된다.

### 4.2 SYNTHIA $\rightarrow$ Cityscapes 본 실험

가장 중요한 비교는 Table 2에 제시된 semantic segmentation 결과이다. 평균 IoU 기준으로 보면, Hoffman et al.의 FCNs in the Wild가 20.2였고, 저자들이 직접 재구현한 source-only baseline NoAdapt는 22.0이었다. 저자들 방법 중 image-level만 쓴 Ours (I)는 25.5, landmark superpixel만 쓴 Ours (SP)는 28.1, 둘을 모두 쓴 Ours (I+SP)는 29.0을 기록했다.

즉, baseline 22.0에서 최종 29.0으로 약 7.0 포인트 상승했다. 논문은 이 개선폭이 기존 경쟁 방법인 FCNs in the Wild가 보고한 개선폭보다 더 크다고 강조한다. 또한 자신들의 baseline 자체가 더 높음에도 추가 성능 향상을 더 크게 얻었다는 점을 언급한다.

세부적으로 보면 image-level distribution만 써도 의미 있는 개선이 나타난다. 특히 road와 sidewalk의 비정상적 비율 예측이 교정되는 효과가 컸다고 설명한다. 실제 표에서도 Ours (I)는 road와 sidewalk 관련 값이 baseline 대비 크게 개선된다. 이는 전역 분포 제약이 픽셀 단위 지도 없이도 비율 불균형을 바로잡는 데 유용함을 보여 준다.

superpixel 기반 분포의 효과도 크다. 단순 superpixel classification 결과인 SP나 SP Lndmk 자체도 baseline보다 높은 경우가 많고, 이를 segmentation regularization에 활용한 Ours (SP)는 mean IoU 28.1로 더 높은 결과를 낸다. 특히 landmark superpixel만 사용하는 설계가 중요하다. 논문은 모든 superpixel을 다 regularize하면 오히려 거의 개선이 없었다고 보고한다. 이는 잘못 추정된 local signal까지 강하게 넣으면 학습이 망가질 수 있음을 보여 준다.

흥미로운 점은 두 signal의 특성이 다르다는 것이다. superpixel 기반 방법은 sky, road, building처럼 큰 면적을 차지하는 클래스에 강하지만, fence, traffic light, traffic sign 같은 작은 객체에는 약하다. 반대로 image-level distribution 기반 방법은 작은 객체에서 상대적으로 더 낫다. 저자들은 이 때문에 두 방법이 상호보완적이라고 해석하며, 실제로 Ours (I+SP)가 최고 성능을 낸다.

또한 superpixel classifier 자체의 정확도도 보고한다. target domain의 모든 superpixel에 대한 분류 정확도는 71%이고, confidence 상위 60%로 선택한 landmark superpixel만 보면 88% 이상이다. 이는 “모든 superpixel이 아니라 신뢰도 높은 일부만 쓰겠다”는 설계가 데이터적으로 타당함을 뒷받침한다.

### 4.3 기존 feature alignment 방식과의 비교

저자들은 논문 말미에 자신들이 처음에는 classification용 deep domain adaptation처럼 feature alignment를 시도했다고 밝힌다. 예를 들어 output 이전 feature layer에 maximum mean discrepancy를 거는 방식이나, gradient reversal로 domain classifier를 붙이는 방식 등을 실험했다. 그러나 FCN-8s의 어느 feature layer에 적용하더라도 baseline 대비 눈에 띄는 성능 향상을 얻지 못했다고 적고, 그 결과는 본문에 생략했다.

이 부분은 논문의 핵심 주장과 정확히 연결된다. 즉, segmentation 같은 structured prediction에서는 feature-level domain invariance를 강제하는 접근이 충분하지 않거나, 심지어 덜 적합할 수 있다는 것이다. 저자들의 curriculum/property-based regularization이 단순한 대안이 아니라 구조적으로 더 맞는 접근이라는 논리적 근거가 된다.

### 4.4 GTA $\rightarrow$ Cityscapes 추가 실험

논문 본문 채택 이후 추가한 보완 실험에서는 source를 GTA로 바꾸어 Cityscapes에 적응시킨다. GTA는 Grand Theft Auto V 게임에서 수집한 synthetic dataset으로, 총 24,996장의 image를 포함하고 Cityscapes의 19개 공식 training class와 fully compatible한 annotation을 제공한다.

이 설정에서도 유사한 경향이 반복된다. NoAdapt는 22.3, Ours (I)는 23.1, Ours (SP)는 27.8, Ours (I+SP)는 28.9였다. FCNs in the Wild는 27.1이었다. 즉, 여기서도 두 property를 결합한 Ours (I+SP)가 최고 성능이다.

논문은 GTA가 SYNTHIA보다 더 photo-realistic하기 때문에 전체 성능이 더 좋아지는 것이 놀랍지 않다고 해석한다. 즉, source quality가 좋아지면 curriculum adaptation도 더 잘 작동하는 경향이 있음을 보여 준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation for semantic segmentation 문제를 feature alignment 중심 사고에서 벗어나 다시 정의했다는 점이다. 저자들은 segmentation이 structured prediction이라는 사실을 진지하게 받아들이고, “공통 feature space + 동일 prediction rule”이라는 강한 가정을 피한다. 대신 target output이 만족해야 할 필요한 속성에 직접 regularization을 건다. 이는 아이디어 차원에서 명확하고, 실제로 당시 거의 유일한 직접 경쟁 방법보다도 좋은 성능을 보였다.

또 다른 강점은 방법이 단순하고 모듈식이라는 점이다. segmentation backbone을 바꾸지 않고 loss 항만 추가하면 되며, target property 추정기 역시 logistic regression과 linear SVM 같은 비교적 단순한 모델로 구현된다. 그럼에도 불구하고 baseline 대비 뚜렷한 성능 향상을 달성했다. 이는 아이디어의 실효성을 잘 보여 준다.

실험 분석도 설득력이 있다. 저자들은 global distribution 자체의 추정 품질을 먼저 검증하고, landmark superpixel 선택의 타당성을 accuracy로 보이며, image-level과 superpixel-level 신호의 상보성을 class-wise 결과로 설명한다. 단순히 “좋아졌다”가 아니라 왜 좋아지는지 구조적으로 해석하려고 한 점이 좋다.

반면 한계도 분명하다. 첫째, global image distribution은 전체 비율만 제공할 뿐 공간 배치를 충분히 제약하지 못한다. 저자들도 이를 인정하고 superpixel property를 추가한다. 그러나 superpixel도 결국 coarse한 단위이며, 작은 객체나 경계 정밀도에는 약하다. 실제로 fence, traffic sign, traffic light 같은 작은 클래스에서 성능이 매우 낮다.

둘째, superpixel property는 dominant class를 one-hot으로 두는 매우 단순한 근사다. 실제 superpixel 내부에는 여러 클래스가 섞여 있을 수 있는데, 이 복잡성은 반영되지 않는다. 또 SVM confidence 상위 60%라는 threshold 선택은 경험적 설계이며, 이 비율이 왜 최적인지는 본문에서 깊게 분석되지 않는다.

셋째, target property 추정 정확도 자체가 adaptation의 상한을 결정한다. 논문은 쉬운 과제가 segmentation보다 덜 어렵다고 주장하지만, 여전히 domain gap의 영향을 받는다. 예를 들어 image-level logistic regression이나 superpixel SVM이 심하게 틀리면 regularization이 잘못된 방향으로 작용할 수 있다. 논문은 landmark만 선택해 이 문제를 완화하지만, 추정 오차의 불확실성을 loss에 어떻게 반영할지까지는 다루지 않는다.

넷째, 논문은 structured prediction에서는 feature alignment가 잘 안 된다고 주장하지만, 이 주장은 본 논문 설정에서는 타당해 보여도 더 강력한 backbone이나 다른 alignment 기법, 혹은 이후 세대의 adversarial/pixel translation 방법과 비교한 것은 아니다. 즉, “feature alignment 전체가 부적절하다”기보다는 “이 문제에서는 output property regularization이 더 효과적이었다” 정도로 이해하는 것이 정확하다.

다섯째, 본문 설명상 Figure 1, Figure 2를 직접 보면 더 분명했겠지만, 제공된 텍스트만으로는 시각적 비교의 세부 사항까지 완전히 확인할 수는 없다. 따라서 정성 결과에 대한 해석은 논문 텍스트에 명시된 범위 내에서만 가능하다.

종합하면, 이 논문은 당시 기준으로 매우 좋은 문제 재구성과 실용적인 해법을 제시했지만, 사용한 property가 여전히 거칠고 작은 객체 처리에 약하며, 추정 품질에 강하게 의존한다는 한계가 있다.

## 6. 결론

이 논문은 synthetic-to-real semantic segmentation adaptation 문제를 위해 curriculum domain adaptation이라는 새로운 관점을 제안했다. 핵심은 target domain에서 직접 픽셀 정답을 모르는 상황에서도, 이미지 전체의 global label distribution과 신뢰도 높은 landmark superpixel의 local label distribution 같은 쉬운 속성은 비교적 잘 추정할 수 있고, 이를 이용해 segmentation network를 regularize할 수 있다는 점이다.

방법적으로는 source domain의 pixel-wise supervised loss와 target domain의 property-matching loss를 함께 최적화한다. 이때 target property는 logistic regression과 superpixel SVM 같은 별도 teacher 모델이 추정한다. 결과적으로 network는 source에서 세밀한 분류 능력을 배우고, target에서는 비율과 위치 측면에서 더 그럴듯한 출력을 내도록 유도된다.

실험에서는 SYNTHIA $\rightarrow$ Cityscapes와 GTA $\rightarrow$ Cityscapes 모두에서 일관된 성능 향상을 보였다. 특히 image-level과 superpixel-level property를 함께 사용하는 조합이 가장 좋았고, 이는 두 종류의 약한 신호가 서로 보완된다는 점을 보여 준다.

이 연구의 중요성은 단순히 당시 성능 개선에만 있지 않다. semantic segmentation처럼 구조적 출력이 중요한 문제에서는, 도메인 간 feature를 무작정 맞추기보다 output이 가져야 할 구조적 성질을 명시적으로 활용하는 것이 유효할 수 있음을 보여 주었다. 이후 연구 관점에서도 이는 self-training, pseudo-labeling, output-space adaptation, structured regularization 같은 방향과 맞닿아 있다. 실제 적용 측면에서는 dense annotation 비용이 매우 높은 자율주행 환경에서 synthetic data의 활용도를 높이는 실질적 단서를 제공한 연구라고 볼 수 있다.
