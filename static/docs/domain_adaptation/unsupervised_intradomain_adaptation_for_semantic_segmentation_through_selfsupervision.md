# Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision

* **저자**: Fei Pan, Inkyu Shin, Francois Rameau, Seokju Lee, In So Kweon
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2004.07703](https://arxiv.org/abs/2004.07703)

## 1. 논문 개요

이 논문은 semantic segmentation을 위한 unsupervised domain adaptation(UDA) 문제를 다룬다. 구체적으로는, label이 있는 synthetic source domain에서 학습한 segmentation 모델을 label이 없는 real target domain으로 옮길 때 생기는 성능 저하를 줄이는 것이 목표다. 기존 UDA 연구는 대체로 source와 target 사이의 분포 차이, 즉 inter-domain gap을 줄이는 데 집중했다. 그러나 이 논문은 거기서 한 걸음 더 나아가, target domain 내부에서도 이미지 난이도와 장면 특성에 따라 예측 품질 차이가 크게 벌어진다는 점, 즉 intra-domain gap이 존재한다는 문제를 전면에 내세운다.

논문의 핵심 문제의식은 다음과 같다. 같은 target domain이라도 어떤 이미지는 이미 잘 적응되어 prediction entropy가 낮고 segmentation map이 매끄럽게 나오지만, 다른 이미지는 움직이는 객체, 복잡한 배경, 조명 변화, 날씨 등으로 인해 매우 불확실한 예측을 만든다. 저자들은 이런 내부 격차를 무시하면 source-to-target 정렬만으로는 target 전체를 충분히 잘 다루기 어렵다고 본다. 따라서 먼저 source와 target 사이를 맞춘 뒤, 다시 target 내부에서 쉬운 샘플과 어려운 샘플 사이를 맞추는 2단계 self-supervised adaptation 전략을 제안한다.

이 문제는 실제 자율주행과 같은 응용에서 중요하다. synthetic data는 annotation 비용을 크게 줄여 주지만, 실제 deployment에서는 real-world target domain의 복잡한 장면이 성능 병목이 되기 쉽다. 따라서 단순히 “합성에서 실제로 옮긴다”를 넘어서, 실제 데이터 내부의 난이도 차이까지 고려하는 접근은 practical value가 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 target domain을 하나의 균일한 집합으로 보지 않고, 현재 모델이 상대적으로 잘 다루는 easy split과 아직 잘 다루지 못하는 hard split으로 나눈 뒤, easy split으로부터 hard split으로 추가 적응을 수행하는 것이다. 이때 easy/hard 분리는 별도의 annotation 없이, 모델 출력의 entropy를 이용해 자동으로 정한다.

직관은 명확하다. inter-domain adaptation을 거친 모델은 target 이미지들에 대해 서로 다른 수준의 confidence를 보일 것이다. 예측이 확신적이고 부드러운 이미지일수록 entropy가 낮고, 예측이 흔들리거나 noisy한 이미지일수록 entropy가 높다. 저자들은 평균 entropy를 이용해 target 이미지를 순위화하고, 낮은 entropy 이미지들을 easy split, 높은 entropy 이미지들을 hard split으로 둔다. 그런 다음 easy split의 prediction을 pseudo label로 사용하고, easy와 hard 사이의 entropy distribution을 adversarial하게 정렬하여 hard split의 성능을 끌어올린다.

기존 접근과의 차별점은 두 가지다. 첫째, 기존 semantic segmentation UDA가 주로 source-target 간 alignment만 다뤘다면, 이 논문은 target 내부의 분포 불균형을 별도의 adaptation 대상으로 본다. 둘째, curriculum domain adaptation 계열처럼 외부 intermediate domain이나 추가적인 구조적 정보에 의존하지 않고, 모델이 스스로 만든 entropy만으로 easy/hard 분할을 수행한다. 즉, 단순하면서도 data-driven한 curriculum을 설계했다는 점이 특징이다.

## 3. 상세 방법 설명

전체 파이프라인은 세 단계로 이해하면 쉽다. 첫 번째 단계는 inter-domain adaptation이다. 여기서는 label이 있는 source image와 label이 없는 target image를 함께 사용해 $G_{inter}$와 $D_{inter}$를 학습한다. 두 번째 단계는 entropy-based ranking이다. 학습된 $G_{inter}$로 모든 target image의 prediction과 entropy map을 만들고, 이미지별 평균 entropy를 계산해 easy split과 hard split으로 나눈다. 세 번째 단계는 intra-domain adaptation이다. easy split의 pseudo label을 이용해 $G_{intra}$를 supervised하게 학습하고, 동시에 easy와 hard의 entropy map이 구분되지 않도록 adversarial alignment를 수행한다.

먼저 inter-domain adaptation을 보자. source 이미지 $X_s$와 정답 segmentation map $Y_s$가 주어졌을 때, generator $G_{inter}$는 soft segmentation map $P_s = G_{inter}(X_s)$를 출력한다. 각 픽셀 $(h,w)$에서 $P_s^{(h,w,c)}$는 클래스 $c$에 대한 확률이다. source에 대해서는 일반적인 pixel-wise cross-entropy loss를 쓴다.

$$
\mathcal{L}_{inter}^{seg}(X_s, Y_s) = - \sum_{h,w} \sum_c Y_s^{(h,w,c)} \log P_s^{(h,w,c)}
$$

이 식은 source label supervision 자체를 담당한다. 즉, segmentation 모델이 최소한 source domain에서는 제대로 semantic class를 예측하도록 만든다.

그다음 target image $X_t$에 대해서는 $P_t = G_{inter}(X_t)$를 얻고, 픽셀별 entropy map $I_t$를 계산한다.

$$
I_t^{(h,w)} = \sum_c - P_t^{(h,w,c)} \log P_t^{(h,w,c)}
$$

entropy가 낮다는 것은 한 클래스 확률이 높고 나머지는 낮아 prediction이 confident하다는 뜻이고, entropy가 높다는 것은 여러 클래스에 확률이 분산되어 prediction uncertainty가 높다는 뜻이다. 저자들은 AdvEnt 계열 접근을 따라, 이 entropy map을 discriminator $D_{inter}$에 넣어 source와 target을 구분하게 하고, generator는 target entropy map이 source처럼 보이도록 학습한다. 논문에 제시된 adversarial loss는 다음과 같다.

$$
\mathcal{L}_{inter}^{adv}(X_s, X_t) = \sum_{h,w} \log\bigl(1 - D_{inter}(I_t^{(h,w)})\bigr) + \log\bigl(D_{inter}(I_s^{(h,w)})\bigr)
$$

여기서 $I_s$는 source image의 entropy map이다. 의미적으로는, discriminator는 source entropy map과 target entropy map을 구분하려 하고, generator는 target 쪽 entropy 구조를 source와 비슷하게 만들어 inter-domain gap을 줄이려 한다.

이제 논문의 핵심인 entropy-based ranking 단계로 넘어간다. 각 target image에 대해 entropy map 전체의 평균을 계산해 ranking score $R(X_t)$를 만든다.

$$
R(X_t) = \frac{1}{HW} \sum_{h,w} I_t^{(h,w)}
$$

즉, 이미지 전체 평균 entropy가 낮을수록 easy, 높을수록 hard라고 본다. 이때 절대 임곗값을 쓰지 않고, 전체 target 중 easy split에 들어갈 비율을 나타내는 hyperparameter $\lambda$를 사용한다. 정의는 $\lambda = \frac{|X_{te}|}{|X_t|}$이며, $|X_{te}|$는 easy split의 이미지 수, $|X_t|$는 전체 target image 수다. 이 설계는 특정 entropy threshold가 데이터셋마다 달라지는 문제를 피하기 위한 것이다.

이후 intra-domain adaptation 단계에서는 easy split 이미지 $X_{te}$에 대해 $G_{inter}$가 낸 segmentation prediction $P_{te}$를 one-hot pseudo label $\mathcal{P}_{te}$로 바꿔 사용한다. 그리고 $G_{intra}$는 easy split에 대해 다음 segmentation loss를 최소화한다.

$$
\mathcal{L}_{intra}^{seg}(X_{te}) = - \sum_{h,w} \sum_c \mathcal{P}_{te}^{(h,w,c)} \log\left(G_{intra}(X_{te})^{(h,w,c)}\right)
$$

이 식은 self-training의 역할을 한다. 다만 모든 target을 pseudo label로 쓰는 것이 아니라, 상대적으로 믿을 수 있는 easy split만 사용한다는 점이 중요하다. 이는 noisy pseudo label로 인한 학습 붕괴를 줄이려는 의도다.

동시에 easy split과 hard split 사이의 intra-domain gap도 adversarial하게 줄인다. easy image와 hard image의 entropy map을 각각 $I_{te}$, $I_{th}$라고 하면, discriminator $D_{intra}$는 둘을 구분하고, generator $G_{intra}$는 hard split의 entropy map도 easy split과 비슷하게 보이도록 만든다. 논문 식은 다음과 같다.

$$
\mathcal{L}_{intra}^{adv}(X_{te}, X_{th}) = \sum_{h,w} \log\bigl(1 - D_{intra}(I_{th}^{(h,w)})\bigr) + \log\bigl(D_{intra}(I_{te}^{(h,w)})\bigr)
$$

쉽게 말해, easy split은 pseudo label supervision으로 semantic signal을 제공하고, hard split은 adversarial alignment를 통해 easy split의 prediction 특성 쪽으로 끌려간다. 결과적으로 hard image에서도 더 confident하고 구조적인 segmentation map을 얻는 것이 목적이다.

전체 손실은 다음과 같이 네 항의 합으로 구성된다.

$$
\mathcal{L} = \mathcal{L}_{inter}^{seg} + \mathcal{L}_{inter}^{adv} + \mathcal{L}_{intra}^{seg} + \mathcal{L}_{intra}^{adv}
$$

하지만 저자들은 이 전체 목적함수를 한 번에 end-to-end로 최적화하지 않는다. 대신 3-stage training을 쓴다. 첫째, $G_{inter}$와 $D_{inter}$를 이용해 inter-domain adaptation을 수행한다. 둘째, 학습된 $G_{inter}$로 target pseudo label과 entropy ranking을 생성한다. 셋째, $G_{intra}$와 $D_{intra}$를 이용해 intra-domain adaptation을 수행한다. 논문 설명상 이것은 two-step self-supervised adaptation을 안정적으로 구현하기 위한 선택이다.

구현 측면에서는 GTA5→Cityscapes와 SYNTHIA→Cityscapes에서는 AdvEnt를 inter-domain 및 intra-domain adaptation의 기본 프레임워크로 사용했고, Synscapes→Cityscapes에서는 AdaptSegNet을 사용했다. backbone은 ResNet-101 기반 DeepLab-v2이며, multi-level output인 conv4와 conv5를 활용한다. $G_{inter}$는 70,000 iteration 학습 후 target 전체 2,975장의 Cityscapes train image에 대한 pseudo label과 entropy를 생성한다. 이후 $G_{intra}$는 ImageNet pretrained parameter로 초기화되고, $D_{intra}$는 scratch에서 학습된다.

## 4. 실험 및 결과

실험은 synthetic-to-real semantic segmentation 설정에서 수행되었다. source domain으로는 GTA5, SYNTHIA, Synscapes를 사용하고, target domain은 Cityscapes다. 평가 역시 Cityscapes validation set에서 진행한다. GTA5는 24,966장의 synthetic image, SYNTHIA-RAND-CITYSCAPES는 9,400장, Synscapes는 25,000장을 사용한다. target 쪽에서는 Cityscapes train 2,975장을 adaptation training에, validation 500장을 평가에 사용한다. 평가지표는 class별 IoU와 평균인 mIoU다. IoU는 $\text{IoU} = \frac{TP}{TP + FP + FN}$으로 정의된다.

가장 중요한 결과는 GTA5→Cityscapes에서 나타난다. baseline 없는 adaptation은 36.6 mIoU, ROAD는 39.4, AdaptSegNet은 42.4, MinEnt는 43.1, AdvEnt는 43.8을 기록했다. 제안 방법은 46.3 mIoU를 달성했다. 즉, 동일한 AdvEnt 계열 inter-domain adaptation 위에 intra-domain adaptation을 추가해 2.5 mIoU를 더 얻었다. 클래스별로 보면 road 90.6, building 82.6, vegetation 85.2, sky 80.2, car 86.4, bus 53.9, bike 37.6 등 여러 항목에서 개선이 있었다. 특히 car, bus, bike 같은 이동 객체 계열에서 향상이 두드러진다. 반면 train은 0.0으로 매우 낮아, 드문 클래스나 pseudo label 품질이 낮은 범주에 대해서는 여전히 취약함을 드러낸다.

SYNTHIA→Cityscapes에서도 성능 향상이 확인된다. 16-class 기준으로 AdaptSegNet 39.6, MinEnt 38.1, AdvEnt 40.8에 비해 제안 방법은 41.7 mIoU를 달성했다. 13-class 기준 mIoU*는 48.9로, AdvEnt의 47.6보다 높다. 저자들은 특히 car와 motorbike에서 큰 개선이 있다고 설명한다. 실제 표에서도 car 78.0, motorbike 20.3, bike 36.5 등 이동 객체 쪽 이득이 눈에 띈다.

Synscapes→Cityscapes에서는 기존에 비교 가능한 작업으로 AdaptSegNet을 baseline으로 두었고, AdaptSegNet의 52.7 mIoU에 비해 제안 방법은 54.2를 달성했다. 즉, backbone이나 기본 adaptation framework가 달라져도 intra-domain adaptation의 이점이 유지된다는 점을 보이려 한다.

논문은 ablation도 제시한다. 가장 중요한 것은 easy split 비율 $\lambda$에 대한 분석이다. GTA5→Cityscapes에서 $\lambda=0.0$은 사실상 easy split이 없는 경우로 이해할 수 있으며 43.8 mIoU를 기록해 AdvEnt baseline과 동일하다. $\lambda=0.5$에서는 45.2, $\lambda=0.6$에서는 46.0, $\lambda=0.67$에서는 46.3으로 최고 성능을 보인다. 하지만 $\lambda=0.7$에서는 45.6, $\lambda=1.0$에서는 45.5로 다시 떨어진다. 이는 easy split이 너무 작아도, 반대로 모든 target을 easy로 간주해 self-training만 하게 되어도 최적이 아님을 보여 준다. 즉, 적당한 비율로 reliable pseudo label과 hard split 정렬을 동시에 활용하는 것이 핵심이다.

또 다른 ablation에서는 구성 요소별 기여를 분리한다. AdvEnt baseline은 43.8, intra-domain adversarial adaptation만 추가한 경우 45.1, self-training만 한 경우($\lambda=1.0$, 즉 모든 pseudo label 사용) 45.5, 둘을 함께 쓴 제안 방법은 46.3이다. 이 결과는 두 요소가 모두 유효하며 결합 시 추가 이득이 있음을 보여 준다. 다시 말해, easy pseudo label supervision과 easy-hard alignment는 서로 대체 관계가 아니라 보완 관계에 가깝다.

흥미로운 추가 결과로 entropy normalization도 제시된다. 복잡한 장면은 물체가 많아서 entropy 평균이 높아질 수 있으므로, 희귀 클래스를 많이 포함한 이미지를 unfair하게 hard로 보내는 문제가 있다고 저자들은 인정한다. 이를 완화하기 위해 rare class 개수로 entropy 평균을 나누는 정규화를 시도했고, GTA5→Cityscapes에서 mIoU를 47.0까지 높였다고 보고한다. 다만 본문에서 이 정규화된 점수가 Table 3에만 간략히 제시되고, 방법 정의가 아주 자세히 설명되지는 않는다.

정성적 결과도 논문의 주장을 보강한다. Figure 3에서는 inter-domain adaptation만 사용한 경우보다 제안 방법이 더 정확하고 구조적인 segmentation map을 생성한다고 보이고, Figure 4에서는 hard split 이미지들에 대해 intra-domain alignment 이후 noisy prediction이 개선되었다고 제시한다. 다만 제공된 추출 텍스트에는 원본 그림 자체가 포함되어 있지 않으므로, 여기서는 저자 설명 수준에서만 해석할 수 있다.

마지막으로 저자들은 digit classification에도 같은 아이디어를 적용했다. MNIST→USPS에서 95.8±0.1, USPS→MNIST에서 97.8±0.1, SVHN→MNIST에서 95.1±0.3을 기록해 ADDA와 CyCADA보다 높다. 이는 제안 방식이 semantic segmentation에만 국한된 것이 아니라, entropy 기반 easy/hard 분할과 intra-domain alignment라는 큰 아이디어 자체는 다른 adaptation 문제에도 적용 가능하다는 점을 보여 주려는 실험이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정의 재정의가 명확하다는 점이다. 기존 UDA가 inter-domain gap에 치중했다면, 이 논문은 target 내부의 difficulty mismatch를 별도 적응 대상으로 삼아 실제 target distribution의 이질성을 포착했다. 이는 단순한 engineering trick이 아니라, 왜 기존 adaptation이 일부 target sample에서 충분히 작동하지 않는지에 대한 설득력 있는 관찰에서 출발한다.

두 번째 강점은 방법이 단순하고 범용적이라는 점이다. entropy map은 이미 AdvEnt류 방법에서 사용되던 신호이므로, 저자 방식은 완전히 새로운 supervision을 요구하지 않는다. 실제로 논문도 AdvEnt와 AdaptSegNet 위에 올려서 성능 향상을 보였다. 즉, 이 방법은 “기존 inter-domain adaptation 이후에 붙일 수 있는 추가 단계”로 이해할 수 있고, 그만큼 실용성이 높다.

세 번째 강점은 ablation이 비교적 명확하다는 점이다. $\lambda$ 변화, intra-domain adversarial term 단독, self-training 단독, 결합 모델을 나눠서 보여 주므로, 성능 향상의 원인이 단순한 pseudo label self-training인지, easy-hard alignment 때문인지, 혹은 둘의 결합인지 구분이 가능하다.

반면 한계도 분명하다. 가장 직접적인 한계는 easy split의 pseudo label 품질이 전체 방법의 출발점이라는 점이다. 초기 inter-domain adaptation이 충분히 잘되지 않으면 entropy ranking 자체가 신뢰할 수 없고, easy split이 실제로는 쉬운 샘플이 아닐 수 있다. 저자들도 이론 분석에서 source-target divergence $d_{\mathcal{H}}(S,T)$가 크면 방법이 덜 효과적이라고 인정한다. 즉, 이 방법은 첫 단계 모델이 어느 정도 target에 적응되어 있다는 가정 위에 놓여 있다.

또한 entropy 평균만으로 이미지 난이도를 재는 기준은 직관적이지만 완전하지 않다. 저자 스스로도 complex scene이 물체가 많다는 이유만으로 hard로 분류될 수 있다고 인정했고, 이를 보완하려고 entropy normalization을 추가했다. 이는 곧 원래의 ranking metric이 장면 복잡도와 예측 불확실성을 완전히 분리하지 못한다는 뜻이다. 게다가 rare class 정의도 Cityscapes에 맞춘 heuristic이라 다른 데이터셋에 그대로 일반화될지는 본문만으로 확신하기 어렵다.

또 다른 한계는 학습 절차가 3-stage라는 점이다. end-to-end가 아니라 먼저 inter-domain adaptation을 학습하고, 그다음 pseudo label 생성과 ranking을 수행한 후, 다시 intra-domain adaptation을 따로 학습한다. 이는 구현과 실험 관리 측면에서 다소 번거롭고, 중간 단계의 오류가 다음 단계로 전파될 수 있음을 뜻한다.

비판적으로 보면, 이 논문은 “target을 easy와 hard로 나누면 왜 그것이 진짜 subdomain structure를 반영하는가”를 엄밀히 증명하지는 않는다. entropy가 uncertainty proxy라는 점은 널리 쓰이지만, uncertainty가 항상 domain difficulty나 subdomain discrepancy와 일치하는 것은 아니다. 그럼에도 불구하고 실험적으로는 충분한 개선을 보여 주므로, 이 논문의 설득력은 이론적 완결성보다는 practical effectiveness에 더 가깝다.

## 6. 결론

이 논문은 semantic segmentation UDA에서 source-target 간 inter-domain gap만이 아니라 target 내부의 intra-domain gap도 중요한 문제라고 주장하고, 이를 줄이기 위한 2단계 self-supervised adaptation 프레임워크를 제안했다. 먼저 기존 방식으로 source와 target을 정렬하고, 그 결과로 얻은 entropy를 이용해 target을 easy/hard split으로 나눈 다음, easy split의 pseudo label과 easy-hard adversarial alignment를 통해 hard split까지 적응시키는 구조다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, target 내부 분포 차이를 명시적으로 문제화했다. 둘째, entropy 기반 ranking으로 label 없이 easy/hard 분할을 수행했다. 셋째, self-training과 adversarial alignment를 결합한 intra-domain adaptation을 통해 GTA5, SYNTHIA, Synscapes에서 모두 일관된 성능 향상을 보였다.

실제 적용 관점에서 보면, 이 연구는 synthetic-to-real segmentation처럼 target domain 내부 난이도 편차가 큰 환경에서 특히 유용할 가능성이 크다. 또한 아이디어 자체가 segmentation에만 고정되지 않고 digit classification에도 적용된 점을 보면, confidence-based curriculum과 intra-target alignment의 결합은 broader domain adaptation 문제로 확장될 여지도 있다. 다만 효과는 초기 inter-domain adaptation 품질과 ranking의 신뢰도에 크게 의존하므로, 이후 연구에서는 더 정교한 uncertainty estimation이나 adaptive split 전략으로 발전할 수 있을 것이다.
