# CenterMask: single shot instance segmentation with point representation

## 1. Paper Overview

이 논문은 **one-stage, anchor-box free instance segmentation**을 단순하면서도 빠르고 정확하게 구현하는 문제를 다룹니다. 저자들은 one-stage instance segmentation의 핵심 난제를 두 가지로 정리합니다. 첫째는 **겹치는 인스턴스들을 어떻게 구분할 것인가**, 둘째는 **pixel-wise alignment를 어떻게 유지할 것인가**입니다. 기존 global-area 기반 방법은 픽셀 정렬은 잘 되지만 overlap 상황에서 인스턴스 분리가 약하고, local-area 기반 방법은 인스턴스 분리는 잘하지만 경계가 거칠거나 정렬 문제가 생기기 쉽다고 봅니다. 이를 해결하기 위해 논문은 instance mask를 두 병렬 구성요소로 분해합니다. 하나는 인스턴스 분리를 담당하는 **Local Shape**, 다른 하나는 정밀한 픽셀 정렬을 담당하는 **Global Saliency Map**입니다. 두 출력을 결합해 최종 마스크를 만드는 것이 CenterMask의 핵심입니다.

이 문제가 중요한 이유는 분명합니다. instance segmentation은 detection과 semantic segmentation의 성격을 동시에 가지며, 실제 응용에서는 정확도뿐 아니라 **속도와 구조적 단순성**도 중요합니다. 당시 strong baseline들은 주로 two-stage 방식이었고, one-stage 방식들은 아직 품질이나 정렬 측면에서 아쉬움이 있었습니다. CenterMask는 이 간극을 메우기 위해 제안된 방법입니다. 논문은 COCO에서 Hourglass-104 backbone으로 **34.5 mask AP, 12.3 FPS**를 달성했으며, 이는 TensorMask를 제외한 다른 one-stage 방법들보다 높은 정확도라고 주장합니다.  

## 2. Core Idea

CenterMask의 핵심 아이디어는 다음처럼 요약할 수 있습니다.

**“instance differentiation과 pixel alignment를 하나의 표현으로 동시에 해결하려 하지 말고, 서로 다른 성질의 두 branch로 분리한 뒤 마지막에 조립하자.”**

논문은 이를 다음 두 branch로 구현합니다.

* **Local Shape branch**: 객체 중심점(center point)의 표현으로부터 각 인스턴스의 거친 mask를 예측합니다. 이 branch는 **instance-aware**하며, 겹치는 객체를 서로 구분하는 데 강합니다.
* **Global Saliency branch**: 이미지 전체에서 foreground saliency를 예측합니다. 이 branch는 **pixel-aligned**하고 세밀하지만, 그 자체로는 어떤 픽셀이 어느 인스턴스의 것인지는 모호할 수 있습니다.

최종 마스크는 “coarse but instance-aware”한 Local Shape와 “precise but instance-unaware”한 Global Saliency를 결합해 얻습니다. 이 분해는 기존 one-stage 방법들이 각각 가지던 약점을 상호 보완하려는 설계입니다.  

이 아이디어가 흥미로운 이유는, Mask R-CNN처럼 RoI 내부에서 정렬을 따로 맞추는 것도 아니고, TensorMask처럼 복잡한 align operation을 쓰는 것도 아니며, YOLACT처럼 전역 prototype 조합만으로 instance separation을 기대하지도 않는다는 점입니다. CenterMask는 **center-point representation을 이용한 local instance cue**와 **semantic-segmentation 스타일의 dense saliency**를 결합하는 보다 직관적인 절충안을 제시합니다.  

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

CenterMask의 전체 구조는 detection과 segmentation을 한 번에 수행하는 one-stage 구조입니다. backbone 뒤에 총 다섯 개의 head가 붙습니다.

* Heatmap head
* Offset head
* Shape head
* Size head
* Saliency head

Heatmap head는 카테고리별 center point heatmap을 예측하고, Offset head는 output stride로 생기는 양자화 오차를 보정합니다. Shape head와 Size head는 Local Shape를 만들고, Saliency head는 Global Saliency Map을 만듭니다. 마지막에는 Local Shape와 cropped Saliency Map을 곱해 instance별 final mask를 구성합니다.

### 3.2 Local Shape Prediction

Local Shape branch의 출발점은 “객체는 center point로 대표될 수 있다”는 가정입니다. 하지만 단순히 중심 위치의 고정 크기 feature만으로는 서로 다른 크기의 mask를 표현하기 어렵습니다. 그래서 논문은 mask를 두 요소로 분해합니다.

* **mask size**: 객체의 높이와 너비
* **mask shape**: 고정 크기 $S \times S$의 2D binary array

Shape head 출력은 다음과 같이 표현됩니다.

$$
F_{shape} \in \mathbb{R}^{H \times W \times S^2}
$$

그리고 Size head 출력은

$$
F_{size} \in \mathbb{R}^{H \times W \times 2}
$$

입니다. 어떤 center point $(x, y)$에 대해, 그 위치의 shape feature $F_{shape}(x,y)$는 길이 $S^2$인 벡터이며, 이를 $S \times S$ 배열로 reshape합니다. 동시에 $F_{size}(x,y)$로부터 예측된 높이 $h$와 너비 $w$를 얻고, 이 coarse shape array를 $h \times w$로 resize하여 최종 local shape를 만듭니다.

이 branch의 역할은 매우 명확합니다. 픽셀 단위의 정밀도보다는, 각 center point 주변에 대해 **어디까지가 해당 인스턴스의 대략적 영역인가**를 결정하는 것입니다. 그래서 겹치는 객체를 나누는 데 유리합니다. 논문은 실제로 Shape-only 모델이 복잡한 overlap 상황에서도 인스턴스를 잘 분리한다고 보여줍니다.  

### 3.3 Global Saliency Generation

Local Shape만으로는 정밀한 segmentation이 어렵습니다. 고정 크기 shape vector를 object size로 resize하는 과정에서 spatial detail이 손실되기 때문입니다. 이를 보완하기 위해 CenterMask는 **Global Saliency Map**을 예측합니다.

이 Saliency branch는 semantic segmentation과 유사하게 이미지 전체에 대해 픽셀별 예측을 수행하지만, multi-class softmax가 아니라 **sigmoid 기반 binary classification**을 사용합니다. 즉 각 픽셀이 객체 영역인지 아닌지를 판단합니다. 이 saliency map은 class-agnostic 또는 class-specific 둘 다 가능하며, class-specific 설정에서는 카테고리별 binary map을 출력합니다.  

이 branch의 핵심 장점은 **pixel-wise alignment를 자연스럽게 보존한다**는 점입니다. 기존 instance segmentation에서 복잡한 align/crop/warp 연산을 추가로 설계하던 문제를, global dense prediction으로 우회한 셈입니다. 따라서 Local Shape가 coarse instance prior를 제공하고, Global Saliency가 fine detail을 복원하는 구조가 됩니다.  

### 3.4 Mask Assembly

최종 instance mask는 Local Shape와 Global Saliency를 결합해 만듭니다. 논문은 한 객체에 대한 Local Shape를 $L_k \in \mathbb{R}^{h \times w}$, 대응하는 cropped Saliency Map을 $G_k \in \mathbb{R}^{h \times w}$라 두고, sigmoid를 적용한 뒤 Hadamard product를 취합니다.

$$
M_k = \sigma(L_k) \odot \sigma(G_k)
$$

즉, Local Shape는 인스턴스의 거친 공간 범위를 제한하고, 그 범위 안에서 Global Saliency가 세밀한 foreground 영역을 결정합니다. 결과적으로 두 branch는 다음과 같이 역할이 나뉩니다.

* Local Shape: **instance separation**
* Global Saliency: **precise segmentation**

이 조합은 CenterMask 전체 설계의 본질입니다.  

### 3.5 Loss Function

CenterMask의 overall loss는 네 항으로 구성됩니다.

* center point loss
* offset loss
* size loss
* mask loss

mask loss는 조립된 final mask에 대해서만 정의됩니다. 즉 Local Shape branch와 Global Saliency branch 각각에 독립적인 loss를 두는 것이 아니라, **assembled mask supervision을 통해 두 branch를 함께 학습**합니다. 이는 branch별 역할 분리가 명확하지만, 최종 목적은 하나의 좋은 mask를 만드는 것이라는 점을 반영합니다. 다만 class-specific saliency 설정에서는 saliency branch에 직접 BCE supervision을 추가하면 성능이 더 좋아진다고 보고합니다.  

## 4. Experiments and Findings

### 4.1 핵심 성능

논문이 가장 강조하는 결과는 COCO test-dev 기준 성능입니다. Hourglass-104 backbone 사용 시 CenterMask는 **34.5 AP, 12.3 FPS**를 달성합니다. 저자들은 이 수치가 TensorMask를 제외한 다른 one-stage instance segmentation보다 높고, TensorMask는 약 5배 느리다고 설명합니다. 또한 DLA-34 backbone 버전은 **32.5 mAP, 25.2 FPS**로 더 좋은 speed-accuracy trade-off를 보인다고 보고합니다.

이 결과는 CenterMask가 “정확도는 높지만 느린 모델”도, “빠르지만 부정확한 모델”도 아니라는 점을 보여주려는 것입니다. 논문 제목의 “single shot”이 단순 구조만 의미하는 것이 아니라, 실제로 경쟁력 있는 효율을 갖춘다는 메시지입니다.

### 4.2 Local Shape와 Global Saliency의 기여

Ablation에서 가장 중요한 메시지는 두 branch가 서로 다른 상황에서 다른 강점을 보인다는 점입니다.

* Shape branch만 있는 경우: **26.5 AP**, 겹치는 객체를 잘 분리하지만 mask가 거칠다.
* Saliency branch를 추가하면: Shape-only 대비 **약 5 AP 향상**
* Shape branch의 도입 효과: Saliency 중심 설정에서 **약 10 AP 향상**

논문은 시각화 결과를 통해 Shape-only는 overlap 상황에서 분리는 잘하지만 coarse하고, Saliency-only는 non-overlap에서는 괜찮지만 겹치는 상황에서 artifact가 심하다고 설명합니다. 둘을 결합하면 인스턴스 분리와 정밀 segmentation을 동시에 달성합니다.

이 결과는 CenterMask의 아이디어가 단순한 engineering tweak가 아니라, 실제로 **문제를 적절히 분해한 설계**였음을 뒷받침합니다.

### 4.3 Global Saliency의 설정

Global Saliency는 class-agnostic과 class-specific 두 버전이 가능합니다. 논문은 **class-specific 설정이 class-agnostic보다 2.4 point 높다**고 보고합니다. 이는 서로 다른 카테고리 간 분리에도 saliency branch가 도움이 된다는 뜻입니다. 또한 class-specific saliency에 BCE direct supervision을 추가하면 **추가로 0.5 point 개선**됩니다.

즉, saliency branch는 단순 보조 모듈이 아니라 설계 선택에 따라 성능에 실질적인 영향을 주는 핵심 요소입니다.

### 4.4 Backbone과 shape size

논문은 shape representation의 크기 $S \times S$에 대한 민감도도 분석합니다. larger shape size가 약간의 이득을 주지만 큰 차이는 아니며, 이는 Local Shape 표현이 비교적 robust하다는 의미로 해석합니다. 또 backbone 측면에서는 큰 Hourglass가 작은 DLA-34보다 **약 1.4 AP 높다**고 보고합니다.  

이런 결과는 CenterMask의 핵심 성능이 단지 backbone 크기에만 의존하는 것이 아니라, representation decomposition 자체에서 온다는 점을 보완적으로 보여줍니다.

### 4.5 다른 방법들과 비교

논문은 CenterMask가 overlap 상황에서 YOLACT보다 인스턴스 분리가 낫고, PolarMask보다 마스크 정밀도가 높다고 시각적으로 설명합니다. 또한 Hourglass-104 backbone 기준으로 PolarMask보다 **1.6 point 높고 속도도 더 빠르다**고 언급합니다.

즉, CenterMask는 기존 one-stage 계열에서

* YOLACT의 global-mask 조합 방식보다 overlap handling이 좋고
* PolarMask의 contour-based mask보다 boundary precision이 좋으며
* TensorMask보다 훨씬 단순하고 빠른

중간 지점을 차지합니다.  

### 4.6 FCOS로의 일반화

저자들은 CenterMask의 Local Shape와 Global Saliency branch가 CenterNet뿐 아니라 다른 detector에도 쉽게 붙는다고 주장하며, 실제로 **FCOS**와 결합한 실험도 제시합니다. 논문 조각에서는 LVIS에서 **CenterMask-FCOS가 ResNet-101-FPN 기준 40.0 AP**를 기록하고, 비교 대상으로 인용된 Mask R-CNN의 36.0 AP보다 높다고 제시합니다.

이 결과는 CenterMask가 CenterNet 전용의 특수한 트릭이 아니라, **일반적인 one-stage detector 위에 얹을 수 있는 mask formulation**이라는 점을 강조합니다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

가장 큰 강점은 문제를 아주 명확하게 분해했다는 점입니다.
instance differentiation과 pixel alignment를 각각 Local Shape와 Global Saliency로 나눈 것은 직관적이면서도 실험적으로 잘 뒷받침됩니다.  

두 번째 강점은 구조의 단순성입니다.
TensorMask처럼 복잡한 alignment 연산 없이도 높은 성능을 얻으려 했고, 실제로 좋은 speed-accuracy trade-off를 보여줍니다.  

세 번째는 일반화 가능성입니다.
CenterNet 기반으로 시작했지만 FCOS 같은 다른 one-stage detector에도 embedding 가능하다는 점은 방법의 범용성을 보여줍니다.  

### Limitations

첫째, Local Shape는 본질적으로 coarse mask입니다.
논문도 직접 인정하듯 fixed-size shape vector를 resize하는 과정에서 spatial detail 손실이 발생합니다. 그래서 결국 Global Saliency가 필수적입니다. 즉 Local Shape만으로는 고품질 segmentation이 어렵습니다.

둘째, Global Saliency는 그 자체로는 instance-unaware합니다.
정확한 pixel segmentation은 가능하지만, overlap 상황에서 어느 픽셀이 어느 인스턴스에 속하는지 구분하지 못합니다. 이 역시 branch 조합이 필수적이라는 뜻입니다.  

셋째, center-point representation 기반 접근의 한계도 있습니다.
복잡한 비정형 물체나 중심 표현이 충분히 안정적이지 않은 경우에는, coarse local shape의 표현력이 제한될 가능성이 있습니다. 이 부분은 논문이 크게 비판적으로 다루지는 않지만, 구조상 자연스러운 제약입니다.

### Interpretation

비판적으로 보면 CenterMask의 가장 중요한 기여는 “새로운 강력한 mask head” 자체보다는, **mask representation을 두 개의 서로 다른 성질의 표현으로 factorize했다는 점**입니다.

* Local Shape = 인스턴스 구분용 local prior
* Global Saliency = 정밀 픽셀 정렬용 dense prior

이 factorization이 매우 잘 작동한 것입니다. 따라서 이 논문은 단순히 CenterNet 기반 인스턴스 segmentation 모델이라기보다, **one-stage segmentation에서 어떤 정보는 local하게, 어떤 정보는 global하게 표현해야 하는가**에 대한 설계 원칙을 제시한 논문으로 보는 편이 더 정확합니다.

## 6. Conclusion

CenterMask는 **single-shot, anchor-box free one-stage instance segmentation**을 위해, mask prediction을 **Local Shape**와 **Global Saliency**라는 두 병렬 branch로 분해하고 이를 조립하는 방식을 제안한 논문입니다. Local Shape는 center-point representation에서 coarse instance mask를 예측해 overlap 상황에서도 인스턴스를 구분하고, Global Saliency는 이미지 전체의 dense saliency를 예측해 precise pixel alignment를 제공합니다. 최종적으로 두 출력을 곱해 instance mask를 만들며, 이 단순한 설계로 COCO에서 **34.5 AP, 12.3 FPS**라는 강한 결과를 달성합니다.  

실무적으로는 빠르고 단순한 instance segmentation이 필요한 환경에서 의미가 있고, 연구적으로는 one-stage mask representation 설계에서 **instance-aware local cue와 pixel-aligned global cue의 결합**이 얼마나 중요한지 보여준 사례라고 볼 수 있습니다.
