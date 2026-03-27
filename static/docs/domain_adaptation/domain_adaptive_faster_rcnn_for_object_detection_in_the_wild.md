# Domain Adaptive Faster R-CNN for Object Detection in the Wild

* **저자**: Yuhua Chen, Wen Li, Christos Sakaridis, Dengxin Dai, Luc Van Gool
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1803.03243](https://arxiv.org/abs/1803.03243)

## 1. 논문 개요

이 논문은 **object detection model이 source domain에서 학습되고 target domain에서는 라벨 없이 적용되는 상황**에서, domain shift 때문에 발생하는 성능 저하를 줄이는 방법을 다룬다. 저자들은 특히 Faster R-CNN을 기반으로, **cross-domain object detection**을 위한 end-to-end 적응 구조를 제안한다. 논문의 문제 설정은 전형적인 **unsupervised domain adaptation**이다. 즉, source domain에는 bounding box와 category label이 모두 있고, target domain에는 이미지 자체만 있으며 annotation은 없다.

논문이 다루는 연구 문제는 명확하다. 기존 object detector는 보통 학습 데이터와 테스트 데이터가 같은 분포에서 왔다고 가정한다. 하지만 실제 환경에서는 카메라가 다르고, 도시가 다르고, 날씨가 다르고, synthetic image와 real image 사이에도 시각적 차이가 존재한다. 이런 차이는 detector의 일반화 성능을 크게 떨어뜨린다. 특히 object detection은 단순 분류보다 더 어렵다. 어떤 물체가 있는지 맞혀야 할 뿐 아니라, **어디에 있는지도 localization**해야 하기 때문이다. 따라서 classification용 domain adaptation 기법을 그대로 쓰기 어렵다.

이 문제가 중요한 이유는 분명하다. 실제 응용, 특히 자율주행에서는 수많은 환경 변화가 발생하며, 모든 target domain에 대해 bounding box annotation을 새로 만드는 것은 비용이 매우 크다. 논문은 이 점에서 의미가 있다. **추가 target label 없이도 새로운 도메인에서 robust한 detector를 만들 수 있는지**를 다루기 때문이다. 또한 당시 기준으로는 end-to-end object detector 자체에 domain adaptation을 체계적으로 결합한 초기 작업이라는 점도 중요하다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 domain shift가 한 군데에서만 생기지 않는다는 관찰이다. 저자들은 domain discrepancy를 두 수준으로 나눈다. 하나는 **image-level shift**이고, 다른 하나는 **instance-level shift**이다. image-level shift는 illumination, style, scale, weather 같은 전역적인 변화에 대응한다. instance-level shift는 object appearance, size, viewpoint 같은 지역적이고 객체 중심의 변화에 대응한다.

이 논문은 Faster R-CNN 내부에 **두 개의 domain classifier**를 삽입한다. 하나는 backbone feature map 위에서 작동하는 **image-level domain classifier**이고, 다른 하나는 ROI feature 위에서 작동하는 **instance-level domain classifier**이다. 두 classifier는 모두 adversarial learning으로 학습된다. 즉, domain classifier는 source와 target을 잘 구분하려 하고, feature extractor는 그 구분을 어렵게 만들어 **domain-invariant representation**을 학습한다.

여기서 끝나지 않고, 저자들은 두 수준의 adaptation이 서로 연결되어야 한다고 주장한다. 단지 image-level alignment만 하거나 instance-level alignment만 하면 충분하지 않을 수 있다. 그래서 두 domain classifier의 출력을 일관되게 만드는 **consistency regularization**을 도입한다. 이 regularization의 목적은 Faster R-CNN의 RPN이 domain-dependent proposal generator가 되지 않도록 압력을 주는 것이다. 즉, proposal 생성 자체도 domain-invariant하게 만들려는 시도다.

기존 접근법과의 차별점은 세 가지로 볼 수 있다. 첫째, classification이 아니라 **object detection의 구조 안으로 직접 domain adaptation을 넣었다**는 점이다. 둘째, global discrepancy와 local discrepancy를 각각 image-level, instance-level로 나누어 처리했다는 점이다. 셋째, 두 정렬 수준 사이의 관계를 확률적으로 해석하고, 이를 바탕으로 **RPN에 대한 consistency 제약**까지 포함했다는 점이다.

## 3. 상세 방법 설명

전체 구조는 Faster R-CNN 위에 세 가지 추가 요소를 붙인 형태다. 원래 Faster R-CNN은 shared convolutional layers, RPN, ROI-based classifier로 구성된다. 논문은 여기에 다음을 더한다. 첫째, 마지막 convolutional feature map 뒤에 image-level domain classifier를 둔다. 둘째, ROI pooling 후 fully connected layer를 거친 instance feature 뒤에 instance-level domain classifier를 둔다. 셋째, 두 classifier 사이에 consistency loss를 둔다.

기본 detection loss는 Faster R-CNN과 동일하게 다음과 같다.

$$
L_{det} = L_{rpn} + L_{roi}
$$

여기서 $L_{rpn}$은 proposal 생성용 classification 및 box regression loss를 포함하고, $L_{roi}$는 ROI classifier의 classification 및 box regression loss를 포함한다. 이 부분은 기존 Faster R-CNN 설정을 그대로 따른다.

### 확률적 관점에서 본 문제 설정

논문은 detection을 $P(C, B \mid I)$를 학습하는 문제로 본다. 여기서 $I$는 image representation, $B$는 bounding box, $C$는 class label이다. source와 target의 joint distribution이 서로 다르기 때문에 성능 저하가 생긴다고 본다.

먼저 joint distribution은 다음처럼 분해할 수 있다.

$$
P(C, B, I) = P(C, B \mid I) P(I)
$$

저자들은 covariate shift assumption을 둔다. 즉, $P(C, B \mid I)$는 source와 target에서 같고, 차이는 $P(I)$에 있다고 가정한다. 이때 필요한 것이 **image-level adaptation**이다. Faster R-CNN의 backbone이 만든 feature map이 사실상 $I$에 해당하므로, source와 target의 image representation 분포를 맞추는 것이 목표가 된다.

또 다른 분해는 다음과 같다.

$$
P(C, B, I) = P(C \mid B, I) P(B, I)
$$

여기서도 $P(C \mid B, I)$가 두 도메인에서 같다고 가정하면, 차이는 $P(B, I)$에 있다. 이것이 **instance-level adaptation**의 근거다. ROI feature는 객체 수준의 표현에 대응하므로, source와 target의 instance representation 분포를 맞추어야 한다고 해석한다.

논문은 이상적으로는 한 수준의 정렬만 완벽해도 다른 수준도 맞을 수 있다고 설명한다. 하지만 실제로는 proposal predictor인 $P(B \mid I)$를 정확하게, 그리고 domain-invariant하게 학습하기가 어렵다. 그 이유는 첫째, image-level alignment가 완벽하지 않을 수 있고, 둘째, bounding box supervision이 source에만 있기 때문이다. 이 때문에 저자들은 두 수준을 동시에 정렬하고, 추가로 consistency regularization을 넣는다.

### $\mathcal{H}$-divergence와 adversarial adaptation

논문은 domain discrepancy를 $\mathcal{H}$-divergence로 설명한다. source sample $\mathbf{x}_{\mathcal{S}}$와 target sample $\mathbf{x}_{\mathcal{T}}$를 domain classifier $h$가 구분한다고 할 때, divergence는 다음처럼 정의된다.

$$
d_{\mathcal{H}}(\mathcal{S}, \mathcal{T}) =
2 \left(1 - \min_{h \in \mathcal{H}} \Big( err_{\mathcal{S}}(h(\mathbf{x})) + err_{\mathcal{T}}(h(\mathbf{x})) \Big) \right)
$$

직관은 단순하다. 최적의 domain classifier도 source와 target을 잘 구분하지 못하면, 두 도메인은 가깝다고 볼 수 있다. 따라서 feature extractor $f$는 domain classifier가 구분하기 어렵도록 feature를 만들어야 한다. 이는 다음 adversarial objective로 연결된다.

$$
\min_f d_{\mathcal{H}}(\mathcal{S}, \mathcal{T})
\Leftrightarrow
\max_f \min_{h \in \mathcal{H}}
{ err_{\mathcal{S}}(h(\mathbf{x})) + err_{\mathcal{T}}(h(\mathbf{x})) }
$$

구현에서는 **Gradient Reversal Layer (GRL)**를 사용한다. domain classifier는 domain classification loss를 줄이도록 학습되고, backbone 또는 ROI feature extractor 쪽은 GRL을 통해 그 loss를 키우는 방향으로 gradient를 받는다. 그 결과 domain-invariant feature가 학습된다.

### Image-level adaptation

image-level adaptation에서는 backbone의 feature map 각 spatial activation에 대해 domain classifier를 적용한다. 즉, feature map의 위치 $(u,v)$마다 그 activation이 source에서 왔는지 target에서 왔는지를 예측한다. 논문은 이를 **patch-based domain classifier**라고 설명한다. 각 activation의 receptive field가 입력 이미지의 한 patch에 대응하기 때문이다.

이 설계의 장점은 두 가지라고 논문은 말한다. 첫째, image style, illumination, global scale 같은 전역적 차이를 줄이는 데 도움이 된다. 둘째, object detection은 입력 해상도가 커서 batch size가 작기 쉬운데, patch 단위로 domain prediction을 하면 학습 샘플 수를 늘리는 효과가 있다.

source/target domain label을 $D_i \in {0,1}$라고 하고, image $i$의 feature map 위치 $(u,v)$에서 domain classifier의 출력을 $p_i^{(u,v)}$라 하면, image-level loss는 다음과 같다.

$$
\mathcal{L}_{img} = -\sum_{i,u,v} \Big[ D_i \log p_i^{(u,v)} + (1-D_i)\log(1-p_i^{(u,v)}) \Big]
$$

이 loss 자체는 일반적인 binary cross-entropy이며, adversarial하게 쓰일 때 backbone이 domain-invariant image representation을 만들도록 유도한다.

### Instance-level adaptation

instance-level adaptation은 ROI pooling 및 FC layer 이후의 ROI feature에 대해 적용된다. 여기서는 각 proposal 또는 ROI feature vector가 source인지 target인지 구분되도록 domain classifier를 학습한다. 목적은 object appearance, size, viewpoint 같은 instance-level discrepancy를 줄이는 것이다.

$i$번째 이미지의 $j$번째 proposal에 대한 instance-level domain classifier 출력이 $p_{i,j}$일 때 loss는 다음과 같다.

$$
\mathcal{L}_{ins} = -\sum_{i,j} \Big[ D_i \log p_{i,j} + (1-D_i)\log(1-p_{i,j}) \Big]
$$

이 역시 GRL을 통해 adversarial하게 최적화된다. 결과적으로 ROI feature가 domain-specific하지 않고 더 domain-invariant해지도록 만든다.

### Consistency regularization

논문에서 가장 흥미로운 부분 중 하나가 consistency regularization이다. 저자들은 image-level classifier가 추정하는 도메인 확률 $P(D \mid I)$와 instance-level classifier가 추정하는 도메인 확률 $P(D \mid B, I)$가 서로 일관되어야 한다고 본다. 이 아이디어는 다음 식에 기반한다.

$$
P(D \mid B, I) P(B \mid I) = P(B \mid D, I) P(D \mid I)
$$

논문은 실제 학습에서는 target bounding box label이 없기 때문에 $P(B \mid I)$ 대신 domain-dependent한 predictor $P(B \mid D, I)$를 배우기 쉽다고 지적한다. 따라서 $P(D \mid B, I)$와 $P(D \mid I)$를 비슷하게 강제하면, RPN이 domain-dependent predictor가 되는 경향을 완화할 수 있다고 해석한다.

실제 regularizer는 image-level domain classifier의 spatial output을 평균 내어 이미지 하나의 도메인 확률로 만들고, 이를 각 ROI의 instance-level 도메인 확률과 가깝게 한다.

$$
L_{cst} = \sum_{i,j} \left| \frac{1}{|I|} \sum_{u,v} p_i^{(u,v)} - p_{i,j} \right|_2
$$

여기서 $|I|$는 feature map activation 수이다. 이 loss는 결국 image-level prediction과 instance-level prediction의 평균적 일관성을 맞추는 역할을 한다.

### 최종 학습 목적함수와 학습 절차

최종 전체 loss는 다음과 같다.

$$
L = L_{det} + \lambda (L_{img} + L_{ins} + L_{cst})
$$

여기서 $\lambda$는 detection objective와 domain adaptation objective 사이의 trade-off parameter다. 논문에서는 실험 전반에 대해 $\lambda = 0.1$을 사용한다.

학습은 end-to-end SGD로 수행된다. 배치에는 source 이미지 1장과 target 이미지 1장이 들어간다. source 이미지는 detection loss와 adaptation loss 둘 다 기여하고, target 이미지는 adaptation loss에만 기여한다. 추론 시에는 domain classifier와 consistency 관련 구성 요소를 제거하고, **adapted Faster R-CNN 본체만 사용**한다. 즉, 학습 시에만 adaptation module이 필요하다.

## 4. 실험 및 결과

### 실험 설정

논문은 세 가지 domain shift scenario에서 실험한다. 첫째는 **synthetic to real adaptation**, 둘째는 **clear weather to foggy weather adaptation**, 셋째는 **cross-camera adaptation**이다. 평가 지표는 기본적으로 IoU threshold 0.5에서의 **mAP** 또는 단일 class에 대한 **AP**이다.

별도 언급이 없으면 모든 이미지의 짧은 변 길이를 500 pixels로 맞춘다. backbone은 ImageNet pretrained weights로 초기화한다. 학습은 learning rate 0.001로 50k iteration, 이후 0.0001로 20k iteration 진행한다. momentum은 0.9, weight decay는 0.0005다. 배치는 source 1장과 target 1장으로 구성된다.

논문은 ablation을 상당히 명확히 제시한다. baseline은 adaptation이 없는 Faster R-CNN이다. 이후 image-level만, instance-level만, 둘 다, 둘 다에 consistency까지 더한 전체 모델을 비교한다. 이 구성 덕분에 각 요소가 어떤 역할을 하는지 비교적 설득력 있게 보여 준다.

### 4.1 Learning from Synthetic Data: SIM10K $\rightarrow$ Cityscapes

이 실험에서는 source domain으로 **SIM10K**, target domain으로 **Cityscapes**를 사용한다. SIM10K에는 car class만 bounding box annotation이 있으므로, 이 실험 역시 car detection만 평가한다. Cityscapes training set의 unlabeled image로 adaptation을 수행하고, validation set에서 성능을 평가한다.

결과는 매우 직접적이다. baseline Faster R-CNN의 Car AP는 **30.12**다. image-level adaptation만 넣으면 **33.03**으로 올라가고, instance-level adaptation만 넣으면 **35.79**가 된다. 둘을 함께 쓰면 **37.86**이 되며, 여기에 consistency regularization까지 더하면 **38.97**이 된다.

즉, baseline 대비 전체 모델은 **+8.8 AP** 향상이다. 이 결과는 논문의 핵심 주장을 잘 뒷받침한다. image-level과 instance-level 정렬이 각각 효과가 있고, 둘을 결합하면 더 좋으며, consistency regularization은 추가 이득을 준다. 특히 instance-level adaptation 단독 성능이 image-level adaptation 단독보다 더 높게 나온 점은 synthetic-real 간 차이에서 객체 appearance discrepancy가 크게 작용했을 가능성을 시사한다. 다만 이것은 논문 수치에 기반한 관찰이지, 저자들이 원인을 정밀하게 분해해 입증한 것은 아니다.

### 4.2 Driving in Adverse Weather: Cityscapes $\rightarrow$ Foggy Cityscapes

이 실험은 source로 **Cityscapes**, target으로 **Foggy Cityscapes**를 사용한다. Foggy Cityscapes는 Cityscapes를 기반으로 synthetic fog를 입힌 데이터셋이다. 평가 클래스는 person, rider, car, truck, bus, train, motorcycle, bicycle이다.

baseline Faster R-CNN의 mAP는 **18.8**이다. image-level adaptation만 사용하면 **25.7**, instance-level adaptation만 사용하면 **26.3**, 둘을 함께 쓰면 **26.6**, consistency까지 포함한 전체 모델은 **27.6**이다. 전체 개선폭은 **+8.6 mAP**다.

클래스별로 보면 car는 27.1에서 40.5로, truck은 11.9에서 22.1로, train은 9.1에서 20.2로 좋아진다. bicycle도 22.8에서 27.1로 개선된다. 모든 클래스가 균일하게 크게 좋아지는 것은 아니지만, 전반적으로 여러 category에 걸쳐 성능 향상이 나타난다. 이 점은 제안 방법이 특정 class 하나에만 맞는 것이 아니라, 여러 객체 class에서 weather-induced domain shift를 줄일 수 있음을 보여 준다.

이 실험에서 특히 image-level adaptation의 중요성을 해석해 볼 수 있다. 날씨 변화는 주로 전역적 시각 변화이므로 image-level alignment가 중요할 것이라고 예상할 수 있다. 실제로 image-only와 instance-only의 mAP 차이는 크지 않지만, 둘을 함께 쓸 때 가장 좋고, consistency까지 넣으면 추가 개선이 있다. 즉, weather shift 역시 전역 변화만으로 환원되지 않고, 객체 visibility와 local appearance 변화도 함께 작용한다는 해석이 가능하다.

### 4.3 Cross Camera Adaptation: KITTI $\leftrightarrow$ Cityscapes

세 번째는 서로 다른 real dataset 사이 적응이다. 논문은 **KITTI $\rightarrow$ Cityscapes**와 **Cityscapes $\rightarrow$ KITTI** 두 방향 모두를 실험한다. 평가는 Car AP 기준이다.

KITTI $\rightarrow$ Cityscapes에서는 baseline이 **30.2**, 전체 모델이 **38.5**이다. Cityscapes $\rightarrow$ KITTI에서는 baseline이 **53.5**, 전체 모델이 **64.1**이다. 각각 약 **+8.3 AP**, **+10.6 AP**에 해당한다. image-level, instance-level, 두 개 결합, consistency 포함 전체 모델 순으로 성능이 올라가는 경향도 유지된다.

이 결과는 domain adaptation이 synthetic-real에서만 필요한 것이 아니라, real-real 데이터셋 사이에도 여전히 중요함을 보여 준다. 카메라 setup, 해상도, scene bias, object scale distribution 차이만으로도 detector 성능이 충분히 흔들린다는 것이다.

### 4.4 Error Analysis on Top Ranked Detections

논문은 KITTI $\rightarrow$ Cityscapes를 사례로 top-20000 confident detections에 대한 error analysis를 수행한다. error type은 correct, mis-localized, background 세 가지다. 결과 figure에서 두 adaptation component 모두 correct detection 수를 늘리고 false positive를 크게 줄인다고 보고한다.

또한 instance-level alignment만 사용한 모델은 image-level alignment만 사용한 모델보다 **background error가 더 높다**고 관찰한다. 논문은 그 이유로 image-level alignment가 RPN을 더 직접적으로 개선하여, localization이 더 좋은 proposal을 생성하기 때문일 수 있다고 설명한다. 이 해석은 consistency regularization이 왜 필요한지와도 자연스럽게 연결된다. object detection에서는 proposal quality가 매우 중요하기 때문이다.

다만 여기서는 figure의 정확한 수치가 텍스트에 포함되어 있지 않으므로, 정량값까지는 복원할 수 없다. 따라서 그래프 기반 세부 수치를 제시하는 것은 부정확할 수 있어 생략하는 것이 맞다.

### 4.5 Image-level vs. Instance-level Alignment

논문은 scale mismatch를 인위적으로 만들며 두 adaptation의 역할 차이를 추가 분석한다. KITTI $\rightarrow$ Cityscapes 설정에서 source scale은 500으로 고정하고, target image scale을 200부터 1000까지 변화시킨다. 이 실험에서는 효율을 위해 더 작은 **VGG-M backbone**을 사용했다고 명시한다.

결과는 image scale mismatch가 커질수록 baseline Faster R-CNN 성능이 크게 떨어진다는 점을 보여 준다. 그리고 image-level adaptation 모델이 instance-level adaptation 모델보다 **scale change에 더 robust**하다고 설명한다. 이는 scale change가 본질적으로 전역적 transformation이기 때문이다. 반면 instance-level alignment는 proposal quality에 의존하는데, 전역 shift가 심하면 proposal localization 자체가 흔들려 alignment 정확성도 영향을 받는다.

하지만 가장 중요한 결론은 여전히 **둘을 함께 쓰는 것이 가장 좋다**는 점이다. 즉, image-level과 instance-level alignment는 대체 관계가 아니라 상보적 관계로 해석된다.

여기서도 figure의 정확한 좌표값은 텍스트로 주어지지 않았으므로, 정밀 수치까지는 말할 수 없다. 다만 실험의 방향성과 저자 해석은 분명하다.

### 4.6 Consistency Regularization 분석

논문은 consistency regularization의 효과를 보기 위해 KITTI $\rightarrow$ Cityscapes에서 RPN proposal quality를 비교한다. 지표는 top-300 proposal이 ground truth와 가질 수 있는 **maximum achievable mean overlap**, 표에는 mIoU로 표현되어 있다.

baseline Faster R-CNN은 **18.8**, image-level과 instance-level adaptation만 적용한 Ours(w/o)은 **28.5**, consistency까지 포함한 최종 Ours는 **30.3**이다. 즉, consistency regularization은 RPN의 proposal quality를 추가로 높인다. 이 결과는 저자들이 제시한 확률적 논리, 즉 두 domain classifier 간 consistency가 RPN의 domain dependence를 줄여 준다는 가설을 실험적으로 지지한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 정의와 방법 설계가 detector 구조와 잘 맞물려 있다**는 점이다. classification에서 쓰던 domain adversarial learning을 단순히 가져온 것이 아니라, object detection에서 실제로 중요한 두 수준인 image-level과 instance-level로 분리해 설계했다. 또한 Faster R-CNN 내부의 어느 지점에서 adaptation을 걸어야 하는지가 명확하다. backbone feature map과 ROI feature라는 구조적 대응이 뚜렷하다.

두 번째 강점은 **ablation이 비교적 설득력 있다**는 점이다. image-only, instance-only, joint, joint+consistency를 모두 비교했고, 세 가지 서로 다른 domain shift scenario에서 일관된 향상을 보였다. synthetic-to-real, weather adaptation, real-to-real camera adaptation이라는 다양한 상황을 포함한 것도 장점이다.

세 번째 강점은 **RPN 문제를 인식했다는 점**이다. detection에서는 classification feature만 맞추면 끝나는 것이 아니라 proposal 생성이 도메인 차이에 영향을 받을 수 있다. 논문은 이를 consistency regularization으로 다루며, RPN quality 개선 표까지 제시한다. 이 점은 detection task의 특수성을 이해한 설계라고 볼 수 있다.

반면 한계도 분명하다. 우선 논문은 **covariate shift assumption**을 중심 전제로 둔다. 즉, $P(C,B \mid I)$ 또는 $P(C \mid B,I)$는 source와 target에서 같고, 차이는 입력 분포 쪽에 있다고 본다. 하지만 실제 환경에서는 label distribution이나 conditional distribution 자체가 변할 수 있다. 예를 들어 특정 객체가 다른 도메인에서 전혀 다른 형태로 나타나거나, annotation policy가 다를 수도 있다. 이 경우 제안 방식의 이론적 정당성은 약해질 수 있다.

또 다른 한계는 instance-level adaptation이 **proposal quality에 의존한다**는 점이다. 논문 스스로도 전역 shift가 심하면 proposal localization이 흔들리고, 그러면 instance-level alignment의 정확성이 손상될 수 있다고 설명한다. 즉, 이 방법은 완전히 detector-independent한 일반 기법이라기보다, proposal quality가 어느 정도 유지되어야 잘 작동하는 구조다.

또한 실험은 당시 기준으로 의미 있지만, 대부분 **한정된 detector family와 제한된 adaptation setting** 위에서 수행되었다. 예를 들어 one-stage detector나 anchor-free detector에 대해 이 논문이 직접 입증하는 바는 없다. backbone도 당시 Faster R-CNN 계열 중심이다. 따라서 방법의 일반성을 지금 관점에서 과대평가하면 안 된다.

비판적으로 보면, consistency regularization은 좋은 아이디어이지만 그 자체가 정확히 어떤 failure mode를 얼마나 줄이는지에 대한 분석은 제한적이다. 예컨대 어떤 종류의 proposal error가 줄었는지, 혹은 어떤 도메인 shift에서 consistency가 특히 중요한지까지는 충분히 세분화되어 있지 않다. 그래도 표 4는 최소한 proposal quality 향상이라는 직접적 근거를 준다.

## 6. 결론

이 논문은 Faster R-CNN 기반 object detector를 새로운 target domain에 적응시키기 위해, **image-level adaptation**, **instance-level adaptation**, 그리고 **consistency regularization**을 결합한 **Domain Adaptive Faster R-CNN**을 제안한다. 핵심 기여는 단순하다. domain shift를 한 수준이 아니라 두 수준에서 다루고, adversarial learning으로 feature 분포를 맞추며, proposal generator인 RPN까지 더 robust하게 만들도록 설계했다는 점이다.

실험적으로도 이 방법은 SIM10K $\rightarrow$ Cityscapes, Cityscapes $\rightarrow$ Foggy Cityscapes, KITTI $\leftrightarrow$ Cityscapes 등 여러 시나리오에서 baseline Faster R-CNN을 뚜렷하게 능가한다. 특히 synthetic-real, weather, camera difference처럼 서로 성격이 다른 domain shift에서 일관된 향상을 보인다는 점은 방법의 실용성을 높여 준다.

실제 적용 측면에서 이 연구는 중요한 의미를 가진다. 자율주행처럼 환경 변화가 큰 분야에서는 새로운 target domain마다 많은 detection label을 다시 만들기 어렵다. 이 논문은 그런 상황에서 **라벨 없는 target image만으로 detector를 적응시키는 초기이자 대표적인 방향**을 제시한다. 향후 연구 관점에서도 이 작업은 domain adaptive detector 설계의 출발점 역할을 한다. 이후 더 강력한 backbone, one-stage detector, transformer detector, pixel-level adaptation, self-training 등이 발전했지만, **“어떤 수준의 representation을 정렬해야 하는가”**라는 이 논문의 질문은 지금도 여전히 중요하다.
