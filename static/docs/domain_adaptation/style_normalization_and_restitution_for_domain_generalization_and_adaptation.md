# Style Normalization and Restitution for Domain Generalization and Adaptation

* **저자**: Xin Jin, Cuiling Lan, Wenjun Zeng, Zhibo Chen
* **발표연도**: 2021
* **arXiv**: [https://arxiv.org/abs/2101.00588](https://arxiv.org/abs/2101.00588)

## 1. 논문 개요

이 논문은 domain generalization(DG)과 unsupervised domain adaptation(UDA)에서 공통적으로 중요한 문제, 즉 학습 데이터와 테스트 데이터 사이의 **style discrepancy**를 줄이면서도 분류에 필요한 **discriminative information**은 유지하는 방법을 다룬다. 저자들은 실제 환경에서 카메라, 조명, 장소, 날씨, 색 대비, 이미지 품질 등의 차이 때문에 같은 객체라도 서로 다른 domain처럼 보이게 되고, 이로 인해 source domain에서 잘 학습된 모델이 unseen target domain에서 성능이 크게 떨어진다고 지적한다.

기존의 많은 DG/UDA 방법은 domain-invariant feature를 학습하려고 feature alignment나 adversarial learning, moment matching 등을 사용한다. 그러나 논문은 이런 정렬 또는 normalization 기반 접근이 domain-specific variation을 줄이는 대신, 동시에 task-relevant한 discriminative signal까지 제거할 수 있다는 점을 핵심 문제로 설정한다. 특히 Instance Normalization(IN)은 style variation을 줄이는 데 효과적이지만 task-ignorant한 연산이므로, 성능에 필요한 정보까지 함께 없앨 수 있다는 것이 출발점이다.

이 문제의 중요성은 매우 실용적이다. 예를 들어 object classification, semantic segmentation, object detection과 같은 다양한 vision task에서 모델은 보통 특정 데이터셋 분포에 과적합되기 쉽다. 하지만 실제 서비스 환경에서는 학습 시 보지 못한 장면, 다른 카메라, 다른 날씨, 다른 렌즈 품질 등을 마주하게 된다. 따라서 style variation을 줄여 generalization을 높이면서도, task discrimination을 유지하는 표현 학습이 중요하다. 이 논문은 바로 그 균형을 맞추기 위해 **Style Normalization and Restitution (SNR)** 모듈을 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 간단하지만 구조적으로 분명하다. 먼저 IN을 사용해 feature map에서 instance-specific style을 줄여 domain discrepancy를 완화한다. 그러나 IN이 제거한 정보 중에는 task와 관련된 discriminative information도 일부 포함될 수 있다. 그래서 저자들은 IN 이후 사라진 정보 전체를 버리지 않고, 원래 feature와 normalized feature의 차이인 residual에서 **다시 task-relevant한 부분만 골라 복원(restitution)** 하자고 제안한다.

즉, SNR의 직관은 다음과 같다. 첫째, style normalization은 필요하다. 둘째, 하지만 normalization만으로는 부족하다. 셋째, normalization 과정에서 빠져나간 residual 안에는 쓸모없는 style noise만 있는 것이 아니라 유용한 signal도 있으므로, 이를 disentangle해서 유용한 부분만 되돌려야 한다. 이 “제거 후 복원”의 2단 구조가 논문의 본질이다.

기존 접근과의 차별점은 크게 두 가지다. 하나는 IN을 쓰되, IN의 부작용인 discriminative information loss를 정면으로 다룬다는 점이다. IBN-Net이나 BIN처럼 BN과 IN을 섞거나 선택하는 방법은 일부 discrimination을 보존하려 하지만, 이 논문은 아예 residual을 분석해서 task-relevant 부분을 적응적으로 복구한다. 다른 하나는 dual restitution loss를 통해 residual을 **task-relevant feature**와 **task-irrelevant feature**로 더 잘 분리하도록 학습시킨다는 점이다. 단순히 attention으로 feature를 나누는 것이 아니라, 복원 후 entropy는 낮아져야 하고 오염(contamination) 후 entropy는 높아져야 한다는 방향성을 loss로 강제한다.

이 아이디어는 분류뿐 아니라 segmentation과 detection에도 적용되며, 저자들은 SNR을 backbone에 삽입하는 plug-and-play 모듈로 설계했다. 그래서 특정 task 전용 기법이라기보다, “style variation을 줄이되 discriminative information은 다시 살린다”는 일반 원리를 다양한 vision task에 이식한 형태라고 볼 수 있다.

## 3. 상세 방법 설명

SNR 모듈은 입력 feature map $F \in \mathbb{R}^{h \times w \times c}$를 받아, 최종적으로 더 generalizable하면서도 discriminative한 출력 feature $\widetilde{F}^{+}$를 만든다. 전체 흐름은 크게 세 단계로 구성된다. 첫 단계는 style normalization, 두 번째는 residual disentanglement와 restitution, 세 번째는 dual restitution loss를 통한 분리 강화이다.

### 3.1 Style normalization

입력 feature $F$에 대해 먼저 Instance Normalization을 적용하여 style-normalized feature $\widetilde{F}$를 만든다. 논문 식은 다음과 같다.

$$
\widetilde{F} = \mathrm{IN}(F) = \gamma \left(\frac{F - \mu(F)}{\sigma(F)}\right) + \beta
$$

여기서 $\mu(F)$와 $\sigma(F)$는 각 sample과 각 channel에 대해 spatial dimension 위에서 계산한 평균과 표준편차이다. $\gamma, \beta \in \mathbb{R}^{c}$는 학습 가능한 파라미터다. 이 연산은 spatial structure는 유지하면서도 instance-specific한 contrast, illumination, saturation 같은 style 요인을 약화시킨다. 결과적으로 서로 다른 domain이나 sample 사이의 discrepancy가 줄어든다.

하지만 저자들이 강조하듯, 이 단계는 task-ignorant하다. 다시 말해, 어떤 정보가 style인지, 어떤 정보가 classification이나 detection에 필요한지 구분하지 않고 정규화한다. 따라서 discriminative information도 일부 손실될 수 있다.

### 3.2 Residual feature와 restitution

IN이 제거한 정보를 명시적으로 다루기 위해, 저자들은 residual feature를 다음처럼 정의한다.

$$
R = F - \widetilde{F}
$$

이 $R$은 원래 feature에서 normalized feature를 뺀 값이므로, IN 과정에서 제거되었거나 약화된 정보의 집합으로 볼 수 있다. 중요한 점은 이 residual 전체가 불필요한 style noise는 아니라는 것이다. 일부는 task-relevant information일 수 있다.

그래서 논문은 $R$을 두 부분으로 나눈다. 하나는 task-relevant feature $R^{+}$, 다른 하나는 task-irrelevant feature $R^{-}$이다. 이 분해는 channel attention을 통해 수행된다. 우선 residual $R$로부터 channel attention vector $\mathbf{a} = [a_1, a_2, \dots, a_c] \in \mathbb{R}^c$를 계산한다.

$$
\mathbf{a} = g(R) = \sigma\left(W_2 , \delta\left(W_1 , \mathrm{pool}(R)\right)\right)
$$

여기서 $\mathrm{pool}(R)$는 spatial global average pooling이고, $W_1$, $W_2$는 두 개의 fully connected layer, $\delta$는 ReLU, $\sigma$는 sigmoid이다. reduction ratio $r$는 16으로 둔다. 이 구조는 SE-like channel attention이다.

그 다음 각 channel마다 attention weight를 곱해 residual을 둘로 나눈다.

$$
R^{+}(:,:,k) = a_k , R(:,:,k)
$$

$$
R^{-}(:,:,k) = (1-a_k) , R(:,:,k)
$$

즉 $a_k$가 큰 channel은 task-relevant하다고 보고 $R^{+}$로 더 많이 보내고, 반대로 작은 channel은 $R^{-}$ 쪽으로 더 많이 보낸다. 논문은 두 개의 독립 attention을 쓰지 않고, $\mathbf{a}$와 $1-\mathbf{a}$를 보완적으로 사용한다. 이렇게 하면 $R^{+}$와 $R^{-}$가 합쳐서 원래 residual $R$을 구성하게 되어 disentanglement가 더 구조적으로 이루어진다고 본다.

이제 실제 복원은 단순하다. style-normalized feature $\widetilde{F}$에 task-relevant residual만 다시 더해 최종 출력 feature를 만든다.

$$
\widetilde{F}^{+} = \widetilde{F} + R^{+}
$$

반대로 task-irrelevant residual을 더한 contaminated feature도 정의한다.

$$
\widetilde{F}^{-} = \widetilde{F} + R^{-}
$$

$\widetilde{F}^{+}$는 실제 추론에 쓰이는 feature이고, $\widetilde{F}^{-}$는 disentanglement를 학습시키기 위한 loss 계산에 활용된다. 논문에 따르면 inference 시에는 점선의 auxiliary branch는 버려진다.

### 3.3 Dual restitution loss

이 논문의 핵심 학습 장치는 dual restitution loss다. 저자들의 의도는 분명하다. task-relevant residual을 더한 $\widetilde{F}^{+}$는 원래 normalized feature $\widetilde{F}$보다 더 discriminative해야 한다. 반대로 task-irrelevant residual을 더한 $\widetilde{F}^{-}$는 오히려 더 덜 discriminative해야 한다. 저자들은 discrimination 정도를 predicted class likelihood의 entropy로 측정한다. 낮은 entropy는 더 sharp한 예측, 즉 더 분명한 class discrimination을 의미한다.

분류 문제에서는 feature map을 spatial average pooling해서 vector를 만든 뒤 FC와 softmax를 통과시켜 entropy를 계산한다. 표기상

$$
\begin{aligned}
\widetilde{\mathbf{f}} &= \mathrm{pool}(\widetilde{F}) \\
\widetilde{\mathbf{f}}^{+} &= \mathrm{pool}(\widetilde{F} + R^{+}) \\
\widetilde{\mathbf{f}}^{-} &= \mathrm{pool}(\widetilde{F} + R^{-})
\end{aligned}
$$

이고, classifier와 softmax를 $\phi(\cdot)$로 쓴다. 엔트로피 함수는 $H(\cdot) = -p(\cdot)\log p(\cdot)$로 둔다.

그럼 positive restitution loss는

$$
\mathcal{L}_{SNR}^{+} = \mathrm{Softplus}\left(H(\phi(\widetilde{\mathbf{f}}^{+})) - H(\phi(\widetilde{\mathbf{f}}))\right)
$$

이다. 이 식은 $\widetilde{\mathbf{f}}^{+}$의 entropy가 $\widetilde{\mathbf{f}}$보다 작아지도록 유도한다. 즉 restitution 이후 예측이 더 sharp해져야 한다.

negative restitution loss는

$$
\mathcal{L}_{SNR}^{-} = \mathrm{Softplus}\left(H(\phi(\widetilde{\mathbf{f}})) - H(\phi(\widetilde{\mathbf{f}}^{-}))\right)
$$

이다. 이 식은 contaminated feature의 entropy가 normalized feature보다 커지도록 유도한다. 즉 task-irrelevant part를 더하면 예측이 더 모호해져야 한다.

최종적으로

$$
\mathcal{L}_{SNR} = \mathcal{L}_{SNR}^{+} + \mathcal{L}_{SNR}^{-}
$$

이다. Softplus는 음수 loss 문제를 피하고 최적화를 더 안정적으로 만들기 위해 사용된다.

### 3.4 Task별 적용 방식

논문은 SNR이 classification, segmentation, detection 모두에 적용 가능하다고 주장한다. 구조 자체는 비슷하지만 restitution loss를 계산하는 단위가 task마다 다르다.

분류에서는 image-level classification이므로 spatial average pooled vector로 entropy를 계산한다.

segmentation에서는 각 spatial position이 하나의 pixel classifier 역할을 하므로, 각 픽셀 위치 $(i,j)$의 feature vector에 대해 entropy를 계산하고 이를 평균한다. 식은 본문에 따르면 다음 형태다.

$$
\mathcal{L}_{SNR}^{+} = \mathrm{Softplus} \left( \frac{1}{h \times w}\sum_{i=1}^{h}\sum_{j=1}^{w} H(\phi(\widetilde{F}^{+}(i,j,:))) - \frac{1}{h \times w}\sum_{i=1}^{h}\sum_{j=1}^{w} H(\phi(\widetilde{F}(i,j,:))) \right)
$$

negative loss도 동일한 방식으로 $\widetilde{F}^{-}$와 비교한다. 저자들은 각 픽셀별로 개별 loss를 쓰는 것보다 평균 entropy를 쓰는 방식이 계산량도 적고 성능도 약간 더 좋다고 말한다.

detection에서는 object detection이 region-wise classification으로 볼 수 있으므로, ground-truth bounding box 영역 내 feature를 spatial average pooling해서 region feature vector를 만들고, 각 region의 entropy를 평균해 restitution loss를 계산한다.

### 3.5 네트워크 삽입 위치와 학습 절차

논문은 ResNet-50 예시에서 각 convolutional block 뒤에 SNR을 넣는 흐름도를 제시한다. 분류 실험에서는 ResNet18/50, segmentation에서는 DRN-D-105와 DeepLabV2(ResNet-101 backbone), detection에서는 Faster R-CNN backbone에 SNR을 삽입했다. 기본적으로 backbone feature extractor 중 앞쪽 여러 stage에 plug-and-play 방식으로 넣는 설계다.

추론 시에는 실제로 $\widetilde{F}^{+}$만 사용하며, contaminated branch는 학습 중 loss를 위한 보조 역할만 한다. 따라서 실질적 목적은 IN이 만든 style-robustness 위에, residual 기반 restitution으로 discrimination을 복원하는 것이다.

## 4. 실험 및 결과

논문은 object classification, semantic segmentation, object detection의 세 가지 task에서 DG와 UDA를 모두 평가한다. 이 점은 논문의 강한 장점 중 하나다. 특정 태스크나 특정 데이터셋에만 맞춘 것이 아니라, feature-level 모듈이 여러 문제에 공통적으로 이득이 있는지를 실험적으로 확인하려 한다.

### 4.1 Object classification

분류 실험에서는 DG용으로 PACS와 Office-Home, UDA용으로 Digit-Five와 DomainNet을 사용한다. DG는 leave-one-domain-out 방식으로 평가하며, source domain만 사용해 학습하고 unseen target domain에서 테스트한다. UDA는 target unlabeled data를 활용한다.

DG 결과에서 SNR은 PACS에서 평균 정확도 81.8%, Office-Home에서 66.1%를 기록했다. baseline AGG는 각각 79.5%, 64.7%였으므로 SNR은 평균적으로 PACS에서 2.3%p, Office-Home에서 1.4%p 향상되었다. PACS에서는 L2A-OT가 평균 82.8%로 더 높았지만, 논문은 L2A-OT가 data generation을 통해 source diversity를 늘리는 접근이므로 SNR과 개념적으로 상보적이라고 해석한다. 즉 SNR은 입력 다양성을 늘리는 대신 feature normalization과 restitution으로 generalization을 높인다.

정규화 기반 비교 실험도 중요하다. AGG-All-IN, AGG-IN, IBN-a, IBN-b, BIN, adaptive BIN 변형과 비교했을 때 SNR이 가장 좋은 평균 성능을 보였다. PACS 기준 AGG-IN은 80.1%, AGG-All-BIN*는 80.6%인데 SNR은 81.8%다. 이는 단순히 IN을 쓰거나 BN/IN을 섞는 것만으로는 부족하고, 제거된 discriminative signal을 restitution하는 설계가 추가로 필요함을 뒷받침한다.

UDA 결과는 더 인상적이다. M3SDA를 baseline으로 사용할 때 Digit-Five에서 baseline은 평균 86.13%이고 SNR-M3SDA는 94.12%로 7.99%p 향상되었다. mini-DomainNet에서도 55.03%에서 58.07%로 3.04%p 향상되었고, full DomainNet에서는 42.67%에서 46.67%로 4.0%p 향상되었다. 논문은 이를 통해 SNR이 단순 DG 모듈에 그치지 않고, 기존 UDA alignment 방법과 결합했을 때 더 큰 효과를 낼 수 있다고 주장한다.

### 4.2 Classification ablation

ablation은 이 논문의 설계 타당성을 보여주는 핵심 부분이다. 먼저 dual restitution loss를 제거하면 PACS 평균이 80.8%, Office-Home이 65.3%로 떨어진다. 최종 SNR은 각각 81.8%, 66.1%이므로 dual restitution loss가 실제로 disentanglement를 돕고 있음을 보여준다. 또한 $\mathcal{L}_{SNR}^{+}$ 또는 $\mathcal{L}_{SNR}^{-}$ 중 하나만 제거해도 성능이 낮아진다. 이는 positive와 negative constraint가 모두 필요하다는 뜻이다.

또한 “비교 없이” 단순히 enhanced entropy는 줄이고 contaminated entropy는 늘리는 방식보다, normalized feature를 기준점으로 하여 비교하는 현재 설계가 더 좋다. PACS에서 0.6%p, Office-Home에서 0.7%p 이득이 있었다. 이는 절대 entropy를 강제하는 것보다 “복원 전후의 상대 비교”가 더 안정적이고 의미 있는 supervision이라는 해석을 가능하게 한다.

SNR을 어느 stage에 넣을지도 실험했다. ResNet18의 stage-1, stage-2, stage-3, stage-4 어느 위치에 하나만 넣어도 baseline보다 향상되었고, 모든 stage에 넣었을 때 PACS 평균 81.8%로 가장 좋았다. 이는 style discrepancy가 네트워크 여러 깊이에서 문제를 일으키며, 다단계 보정이 유리함을 시사한다.

disentanglement design 비교도 흥미롭다. 1x1 conv로 residual을 나누는 SNRconv나, 서로 독립적인 두 attention gate를 쓰는 방식보다, 보완적 채널 attention 하나를 사용하는 현재 설계가 더 좋았다. spatial attention만 쓰는 SNR-S는 SNR보다 약간 낮았고, spatial+channel attention을 병렬로 쓴 SNR-SC가 소폭 더 높은 평균 82.1%를 보였지만, 논문은 단순성과 IN과의 정합성을 이유로 channel attention만 기본 선택으로 유지한다.

### 4.3 Semantic segmentation

semantic segmentation에서는 Cityscapes, GTA5, Synthia를 사용했다. DG에서는 source만으로 학습하고 Cityscapes에서 테스트한다. UDA에서는 target unlabeled data도 학습에 사용한다. metric은 class IoU와 mIoU다.

DG에서 GTA5→Cityscapes 설정의 DRN-D-105 backbone 기준 baseline mIoU는 29.84, Baseline-IN은 32.64, SNR은 36.16이다. 즉 baseline 대비 6.32%p, Baseline-IN 대비 3.52%p 개선이다. DeepLabV2 backbone에서도 baseline 36.94, Baseline-IN 39.46, SNR 42.68로 꾸준히 향상된다. Synthia→Cityscapes에서도 DRN-D-105는 23.56→26.30, DeepLabV2는 31.12→34.36으로 개선된다.

UDA에서는 MCD와 MaxSquare(MS)에 SNR을 붙였다. GTA5→Cityscapes에서 DRN-105 기반 MCD는 35.0 mIoU인데 SNR-MCD는 40.3으로 5.3%p 높다. DeepLabV2 기반 MaxSquare는 44.3이고 SNR-MS는 46.5다. Synthia→Cityscapes에서도 MCD는 36.6, SNR-MCD는 39.6이며, MS는 39.3, SNR-MS는 45.1이다. 특히 마지막 설정에서는 5.8%p의 큰 개선이 나타난다.

정성적 결과에서도 SNR을 넣은 모델이 baseline 대비 segmentation map 품질이 더 좋고, target adaptation을 함께 쓸 때 성능이 더 좋아진다고 논문은 설명한다.

### 4.4 Object detection

detection에서는 Cityscapes, Foggy Cityscapes, KITTI를 사용한다. DG baseline은 Faster R-CNN, UDA baseline은 DA Faster R-CNN이다. 평가 지표는 IoU threshold 0.5 기준 mAP다.

Cityscapes→Foggy Cityscapes에서 DG baseline Faster R-CNN은 mAP 18.8이고, SNR-Faster R-CNN은 22.3이다. 3.5%p 향상이다. UDA baseline DA Faster R-CNN은 27.6이고 SNR-DA Faster R-CNN은 30.6으로 3.0%p 향상된다.

KITTI와 Cityscapes 사이 cross-dataset detection에서도 개선이 확인된다. Car class AP 기준 K→C에서 DG baseline은 30.24, SNR은 35.92이고, C→K에서는 53.52에서 57.94로 상승한다. UDA에서도 각각 38.52→43.51, 64.15→69.17로 향상된다. 논문은 qualitative result를 통해 SNR이 false positive를 줄이고 baseline이 놓친 car를 추가로 검출했다고 설명한다.

### 4.5 Complexity analysis

SNR은 plug-and-play이지만 비용이 얼마나 드는지도 따로 분석한다. ResNet-18에 SNR을 넣으면 FLOPs는 1.83G에서 2.03G로 9.8% 증가하고, parameter는 11.74M에서 12.30M로 4.5% 증가한다. ResNet-50에서는 FLOPs가 3.87G에서 4.08G로 5.1%, parameter가 24.56M에서 25.12M로 2.2% 늘어난다. 즉 성능 개선 대비 추가 비용이 아주 크지는 않다는 것이 논문의 주장이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의가 명확하고 설계가 논리적으로 닫혀 있다는 점이다. IN이 style discrepancy를 줄이는 데 효과적이지만 discriminative information도 잃는다는 관찰은 직관적이며, residual을 통해 그 손실을 보상한다는 발상도 자연스럽다. 특히 residual 전체를 다시 더하는 것이 아니라 channel attention을 이용해 task-relevant part만 선택적으로 복원하는 구조는 단순한 shortcut이 아니라 분해와 복원을 결합한 설계다.

또 다른 강점은 dual restitution loss의 설계다. 많은 논문이 attention이나 disentanglement를 이야기하지만, 실제로 무엇을 기준으로 disentangle가 잘 되었는지 supervision이 모호한 경우가 많다. 이 논문은 “복원하면 entropy가 줄어야 하고, 오염시키면 entropy가 늘어나야 한다”는 비교 기반 제약을 둠으로써, task relevance를 classifier confidence의 관점에서 구체화했다. 이 때문에 disentanglement가 단순한 구조적 가정에 그치지 않고 학습 목표로 구현된다.

실험 범위도 강점이다. 분류, 분할, 검출 모두를 다루고, DG와 UDA 양쪽을 함께 평가한다. 또한 normalization 변형, loss 구성요소, 삽입 위치, disentanglement 설계까지 ablation을 폭넓게 수행했다. 이 때문에 단순한 성능 보고를 넘어, 제안 요소 각각이 왜 필요한지 비교적 설득력 있게 보여준다.

하지만 한계도 있다. 첫째, 논문은 style discrepancy를 주된 domain gap 원인으로 본다. 실제로 많은 상황에서 style은 중요한 요인이지만, domain shift는 label distribution 차이, geometry 차이, background bias, object co-occurrence, annotation policy 차이 등으로도 발생한다. SNR은 이 중 style-like discrepancy에 특히 잘 맞는 접근이며, 모든 종류의 domain shift를 동일하게 해결한다고 보기는 어렵다. 논문도 이를 명시적으로 일반 이론 수준으로 증명하지는 않는다.

둘째, task relevance를 entropy 기반으로 정의한 것은 실용적이지만 완전한 정의는 아니다. 예측 entropy가 낮다고 해서 반드시 더 “올바른” discriminative feature만 복원되었다고 단정할 수는 없다. 잘못된 overconfident prediction이 생길 가능성도 이론적으로는 있다. 논문은 실험적으로 성능 향상을 보여주지만, entropy surrogate 자체의 한계에 대한 깊은 분석은 제공하지 않는다.

셋째, segmentation과 detection에 대해서는 classification에 비해 방법 설명이 상대적으로 간단하다. 물론 loss 계산 단위가 pixel 또는 region으로 바뀌는 구조는 설명되어 있지만, 실제 detector나 segmentor 내부에서 SNR이 어떤 feature level에서 가장 효과적인지, proposal 단계나 decoder 단계에 미치는 영향 등은 자세히 다루지 않는다. 또한 Supplementary에 구현 세부사항이 더 있다고 하지만, 제공된 본문만으로는 일부 훈련 세부값이나 데이터 처리 방식이 축약되어 있다.

넷째, SNR-SC가 channel-only SNR보다 평균적으로 약간 더 좋았음에도 불구하고 기본 설계로 채택하지 않은 부분은 실용적 선택이긴 하지만, 최종 설계가 절대 최적이라는 의미는 아니다. 저자들은 단순성과 IN과의 정합성을 이유로 들지만, 실제 응용에서는 spatial+channel 설계가 더 유리할 가능성을 남겨둔다.

종합하면, 이 논문은 “style normalization만으로는 부족하며 restitution이 필요하다”는 핵심 메시지를 실험적으로 잘 입증했지만, domain gap 전체를 포괄하는 보편 해법이라기보다는 **style-heavy한 domain shift에 매우 잘 맞는 일반 모듈**로 해석하는 것이 더 정확하다.

## 6. 결론

이 논문은 DG와 UDA에서 중요한 두 요구, 즉 **generalization**과 **discrimination**을 동시에 만족시키기 위해 SNR이라는 모듈을 제안했다. SNR은 먼저 IN을 통해 style discrepancy를 줄여 generalization을 높이고, 그 과정에서 제거된 residual로부터 task-relevant feature를 채널 attention 기반으로 골라 다시 더함으로써 discrimination을 복원한다. 여기에 dual restitution loss를 도입해 residual을 task-relevant part와 task-irrelevant part로 더 잘 분리하도록 유도했다.

논문의 주요 기여는 세 가지로 정리할 수 있다. 첫째, IN의 장점과 한계를 동시에 고려한 normalization-plus-restitution 구조를 제시했다. 둘째, entropy comparison 기반 dual restitution loss로 disentanglement를 구체적으로 학습시켰다. 셋째, 이 모듈이 classification, segmentation, detection에서 모두 효과적이며, DG뿐 아니라 기존 UDA 방법과 결합할 때도 성능 향상을 준다는 점을 보여주었다.

실제 적용 측면에서 이 연구는 backbone feature가 style variation에 민감한 다양한 vision system에 유용할 가능성이 크다. 특히 다른 domain adaptation 기법과 경쟁하기보다 상보적으로 결합될 수 있다는 점이 실용적이다. 향후 연구에서는 spatial attention과의 결합, 더 다양한 유형의 domain shift에 대한 분석, 혹은 self-supervised representation learning과의 결합 등을 통해 이 아이디어를 더욱 확장할 수 있을 것으로 보인다. 적어도 이 논문이 전달하는 핵심 메시지는 분명하다. **도메인 차이를 줄이는 것만으로는 충분하지 않으며, 그 과정에서 잃어버린 task-relevant signal을 어떻게 되살릴 것인가가 성능의 관건**이라는 점이다.
