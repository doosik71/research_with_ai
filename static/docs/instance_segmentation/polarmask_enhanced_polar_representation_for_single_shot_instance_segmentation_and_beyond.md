# PolarMask++: Enhanced Polar Representation for Single-Shot Instance Segmentation and Beyond

이 논문은 instance segmentation의 파이프라인을 더 단순하고 빠르게 만들기 위해, 기존의 **“detect then segment”** 방식 대신 **polar representation**으로 객체 마스크를 직접 예측하는 single-shot 프레임워크를 제안합니다. 핵심 아이디어는 객체를 bounding box나 dense binary mask로 다루지 않고, **객체 중심점과 여러 각도에서의 ray length**로 표현하는 것입니다. 이렇게 하면 instance segmentation과 object detection을 하나의 통합된 표현으로 다룰 수 있고, box branch 없이도 마스크를 예측할 수 있습니다. 저자들은 이 기본 아이디어를 **PolarMask**라 부르고, 여기에 **soft polar centerness**, **polar IoU loss**, 그리고 **Refined Feature Pyramid**를 추가해 성능을 높인 버전을 **PolarMask++**로 제시합니다. 논문은 COCO instance segmentation뿐 아니라 ICDAR2015 rotated text detection, DSB2018 cell segmentation에서도 강한 성능을 보고합니다.  

## 1. Paper Overview

이 논문이 해결하려는 문제는 명확합니다. instance segmentation은 보통 먼저 bounding box를 찾고, 그 안에서 다시 mask를 예측하는 two-stage 구조를 따릅니다. Mask R-CNN이 대표적입니다. 이런 방식은 성능은 좋지만 파이프라인이 복잡하고, box detection 결과에 mask 품질이 강하게 의존하며, 실시간 응용에는 계산 부담이 큽니다. 저자들은 이런 구조적 복잡성이 real-world application에 불리하다고 보고, **bounding-box-free이면서 single-shot인 instance segmentation**을 목표로 삼습니다.

이 문제가 중요한 이유는 instance segmentation이 다양한 downstream task에서 핵심 역할을 하기 때문입니다. 논문은 텍스트 검출과 인식, 세포 분할, 제조 결함 위치 탐지 등에서 object mask가 bounding box보다 더 정확한 경계 정보를 제공한다고 설명합니다. 따라서 더 단순하면서도 정확한 instance segmentation 파이프라인은 실용적 가치가 큽니다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음과 같습니다.

> **객체 마스크를 pixel grid로 직접 예측하지 말고, 중심점에서 여러 방향으로 뻗는 ray들의 길이로 표현하자.**

즉 하나의 객체를 $(x_c, y_c)$ 중심점과 여러 각도에서의 contour distance로 표현합니다. 논문은 이를 **polar representation**이라 부릅니다. 이 표현은 세 가지 장점이 있습니다.

첫째, **객체 중심점**을 자연스럽게 정의할 수 있습니다.
둘째, 각 contour point는 중심으로부터의 **거리 하나**만 회귀하면 됩니다. Cartesian contour처럼 $(x,y)$ 두 좌표를 모두 회귀할 필요가 없습니다.
셋째, 각도 자체가 방향성을 제공하므로 contour point들을 순서대로 연결하기 쉽습니다.

저자들이 강조하는 novelty는 polar representation이 **bounding box와 mask를 하나의 통일된 표현으로 다룬다**는 점입니다. 논문 표현대로, bounding box는 단지 4개의 ray만 가진 가장 단순한 polar mask로 볼 수 있습니다. 따라서 PolarMask는 box branch를 따로 둘 필요 없이 mask prediction 자체로 detection까지 포괄할 수 있습니다.

또한 PolarMask++는 원 논문 PolarMask에서 더 나아가 두 가지 핵심 개선을 추가합니다.

* **soft polar centerness**: 좋은 center sample에 더 높은 점수를 주도록 설계
* **polar IoU loss**: ray regression을 개별 좌표가 아니라 mask 품질 관점에서 최적화
* **Refined Feature Pyramid**: 특히 작은 객체와 scale variation 대응을 위한 feature fusion 강화  

## 3. Detailed Method Explanation

### 3.1 Polar representation으로 마스크를 다시 정의하기

논문은 객체 인스턴스 mask가 주어졌을 때, 먼저 객체 중심점 $(x_c, y_c)$를 잡고 contour 위의 점들을 polar coordinate로 표현합니다. 중심에서 시작해 동일한 각도 간격 $\Delta \theta$로 $n$개의 ray를 쏘고, 각 ray가 contour와 만나는 지점까지의 거리를 회귀합니다. 예를 들어 논문은 보통 **36개의 ray**, 즉 $\Delta \theta = 10^\circ$ 설정을 사용합니다. 이렇게 하면 최종적으로 배워야 하는 것은 각 ray의 길이 ${d_1, d_2, \dots, d_n}$입니다.  

즉 instance segmentation 문제는 다음처럼 바뀝니다.

* **instance center classification**
* **polar ray length regression**

그리고 예측된 contour point들을 각도 순서대로 연결하면 최종 mask를 조립할 수 있습니다.

### 3.2 Center choice: 왜 mass center인가

중심점을 어떻게 정의할지는 이 방법의 핵심입니다. 논문은 bounding box center와 **mass center**를 비교하고, mass center가 더 유리하다고 결론내립니다. 이유는 mass center가 실제 객체 내부에 위치할 확률이 더 높기 때문입니다. 도넛처럼 비정상적인 예외는 있지만, 대부분의 일반 객체에서는 mass center가 더 좋은 center prior를 제공합니다.

또 양성 샘플은 mass center 주변의 일정 영역에서 선택합니다. 논문은 FCOS 스타일로, feature stride를 기준으로 mass center 주변 약 9~16 pixel 정도를 positive center sample로 취급한다고 설명합니다. 이 설계는 positive/negative imbalance를 줄이고, center representation을 더 안정화하는 역할을 합니다.

### 3.3 Ray regression의 세부 처리

각 positive center sample에 대해 네트워크는 ray 길이들을 예측합니다. 흥미로운 것은 논문이 concave shape와 center-outside-mask 같은 예외도 다룬다는 점입니다.

* 어떤 ray가 contour와 여러 번 만나는 경우: **가장 먼 교점**을 선택
* center가 mask 밖에 있어 해당 방향에서 contour와 만나지 않는 경우: 작은 상수 $\epsilon$을 사용

즉 이 방법은 star-convex shape에 특히 자연스럽지만, 실제 객체의 복잡한 모양에도 어느 정도 robust하도록 설계되어 있습니다.

### 3.4 Soft polar centerness

PolarMask 계열의 또 다른 핵심은 **centerness**입니다. FCOS의 centerness를 그대로 쓰면, ray 길이들 간 차이가 큰 샘플에서 점수가 지나치게 낮아지는 문제가 있습니다. 특히 복잡한 객체 모양에서는 $d_{\min} / d_{\max} \to 0$에 가까워져 classification score가 과하게 낮아질 수 있습니다. 논문은 이를 original polar centerness가 너무 “aggressive”하다고 해석합니다.

이를 완화하기 위해 제안한 것이 **soft polar centerness**입니다. ray들을 4개 subset으로 나누고, subset 수준에서 균형을 보게 하여 extreme length imbalance의 영향을 줄입니다. 구현상으로는 classification branch와 병렬인 단일 추가 branch로 예측되며, 최종적으로 classification score와 곱해져 low-quality mask를 억제합니다. 논문은 이 모듈이 특히 **AP75 같은 strict localization metric**에서 성능을 높인다고 보고합니다.  

### 3.5 Polar IoU loss

일반적인 Smooth-L1 loss로 ray를 각각 독립적으로 회귀하면, dense distance prediction 문제에서는 regression loss가 classification loss보다 지나치게 커지고, 전체 contour 품질을 직접 반영하지 못하는 문제가 있습니다. 이를 해결하기 위해 논문은 **polar IoU loss**를 제안합니다. 핵심 식은 다음과 같습니다.

$$
\mathrm{Polar\ IoU\ Loss} = \log \frac{\sum_{i=1}^{n} d_i^{\max}}{\sum_{i=1}^{n} d_i^{\min}}
$$

여기서 $d_i^{\max}$와 $d_i^{\min}$는 예측 ray와 GT ray의 각 방향에서의 큰 값과 작은 값입니다. 이 손실은 ray들을 개별적으로 보지 않고 **전체 contour를 하나의 구조로 최적화**하려는 목적을 가집니다. 저자들은 이것이 실제로 polar space에서 mask IoU를 최대화하는 방향이라고 설명합니다.

논문이 주장하는 polar IoU loss의 장점은 세 가지입니다.

1. 미분 가능하고 병렬 계산이 쉬워 학습이 빠름
2. Smooth-L1보다 전체 성능을 크게 개선
3. dense ray regression에서 classification/regression 간 균형을 자동으로 맞추는 데 도움

즉 이 논문은 representation만 바꾼 것이 아니라, 그 표현에 맞는 **task-specific optimization objective**도 함께 설계했다는 점이 중요합니다.

### 3.6 Network architecture와 Refined Feature Pyramid

PolarMask++는 FCOS를 기반으로 한 one-stage fully convolutional framework입니다. backbone과 FPN 위에 세 개의 head가 올라갑니다.

* classification branch
* polar centerness branch
* mask regression branch

PolarMask++의 추가 개선점은 **Refined Feature Pyramid**입니다. 논문은 기존 FPN만으로는 특히 작은 객체에서 feature fusion이 충분하지 않다고 보고, scale 간 feature representation을 더 잘 섞는 refinement 모듈을 추가합니다. 저자들은 이것이 작은 객체 성능을 개선하는 데 중요하다고 설명합니다.

또 중요한 설계는 **bounding box branch를 제거했다**는 점입니다. 논문은 polar representation에서는 bounding box도 사실 4-ray mask의 특수 경우이므로, 별도의 box detection head가 mask prediction에 거의 기여하지 않는다고 분석합니다. 실제 ablation에서도 box branch의 기여가 작다고 보고하며, 따라서 PolarMask++는 box branch를 두지 않아 더 단순하고 빠릅니다.

## 4. Experiments and Findings

### 4.1 주요 성능

논문은 single-model, single-scale setting에서도 강한 성능을 보고합니다. 대표적으로:

* **COCO**: 38.7% mask mAP
* **ICDAR2015**: 85.4% F-measure
* **DSB2018**: 74.2% mAP

즉 이 방법은 단순히 COCO용 instance segmentation에만 맞춘 것이 아니라, **rotated text detection**과 **cell segmentation** 같은 다양한 instance-level 문제로 확장 가능합니다. 이는 polar representation이 단순한 마스크 기법이 아니라, 보다 일반적인 instance shape representation으로 동작함을 보여줍니다.

### 4.2 Polar representation의 직접적 효과

논문은 “without bells and whistles” 설정에서도 PolarMask가 기존 방식 대비 mask accuracy를 약 **25% 상대 향상**시킨다고 말합니다. 특히 strict localization metric에서 이득이 크다고 강조합니다. 이는 polar contour representation이 coarse bounding box 중심 접근보다 경계 품질에 더 직접적으로 맞닿아 있기 때문으로 해석할 수 있습니다.

### 4.3 Ablation: ray 개수

ray 수는 중요한 설계 변수입니다. 논문에 따르면:

* 18 → 24 rays: 약 1.1% AP 향상
* 24 → 36 rays: 추가 0.3% AP 향상
* 72 rays는 오히려 36 rays보다 0.1% 낮음

저자들은 ray가 너무 많으면 이론적 upper bound는 조금 더 올라가더라도, CNN이 학습해야 할 정보량이 늘어 optimization이 어려워진다고 해석합니다. 따라서 **36 rays**가 성능과 복잡도의 좋은 균형점이라고 결론냅니다.

### 4.4 Ablation: center strategy

Figure 10 관련 분석에서 논문은 **mass center가 box center보다 더 효과적**이라고 보고합니다. 또 ray 수를 90까지 늘리면 72보다 약 0.4% upper bound 개선이 있지만, 성능은 120 근처에서 포화된다고 설명합니다. 이는 representation capacity와 practical learnability 사이에 trade-off가 있음을 보여줍니다.

### 4.5 Ablation: Polar IoU vs. Smooth-L1

논문은 Polar IoU loss가 Smooth-L1보다 더 적합하다고 강하게 주장합니다. 특히 dense distance prediction에서는 Smooth-L1의 regression loss가 지나치게 커지고, ray들을 독립적으로 취급하기 때문에 전체 contour 구조를 반영하지 못합니다. 반면 Polar IoU loss는 ray들의 집합을 하나로 보면서 mask IoU를 더 직접적으로 최적화합니다.  

### 4.6 Box branch는 필요한가

흥미로운 ablation으로, 논문은 box branch가 mask prediction에 거의 기여하지 않는다고 보고합니다. 이는 이 방법의 철학과 잘 맞습니다. box는 polar representation 안에서 이미 mask의 특수 경우이기 때문에, 별도의 box head는 중복성이 큽니다. 결과적으로 PolarMask++는 box-free 설계를 유지하는 편이 더 단순하고 빠르며, 성능 손해도 크지 않습니다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **representation 설계가 매우 깔끔하다**는 점입니다. instance segmentation과 detection을 polar contour prediction으로 통합해, 기존의 “detect then segment” 이중 구조를 단순화했습니다. 이 점은 구조적 아름다움뿐 아니라 실제 속도와 구현 복잡도 측면에서도 이점입니다.

둘째, **task-specific loss와 scoring design이 잘 맞물려 있다**는 점입니다. 단순히 ray를 회귀하는 데서 끝나지 않고, soft polar centerness와 polar IoU loss를 통해 center quality와 contour quality를 함께 다룹니다. 이 때문에 representation 변경이 실제 성능 향상으로 이어집니다.  

셋째, **범용성**도 큽니다. COCO뿐 아니라 rotated text detection, cell segmentation까지 같은 관점으로 확장되었다는 점은 polar representation이 특정 benchmark trick이 아니라 보다 일반적인 instance-level geometry 표현일 수 있음을 보여줍니다.

### 한계

첫째, 이 방법은 본질적으로 객체를 **중심점 + 방사형 contour**로 근사합니다. 따라서 star-convex에 가까운 형상에는 자연스럽지만, 매우 복잡한 concave shape나 중심이 객체 바깥에 놓일 수 있는 특이 형태에서는 표현 한계가 있을 수 있습니다. 논문도 이런 경우를 완화하기 위한 예외 처리($\epsilon$ 사용, 가장 먼 교점 선택)를 넣고 있습니다. 이는 곧 representation 자체의 제약을 보여줍니다.

둘째, ray 수를 늘리면 upper bound는 좋아져도 실제 네트워크 학습은 오히려 어려워질 수 있습니다. 즉 더 정밀한 representation이 항상 더 좋은 최종 성능을 보장하지는 않습니다. 이 점은 72 rays가 36 rays보다 낫지 않았다는 결과에서 드러납니다.

셋째, 오늘 관점에서 보면 이 방법은 query-based segmentation이나 dynamic mask modeling 같은 후속 흐름에 비해 contour parameterization에 더 강하게 의존합니다. 즉 단순성과 효율성은 뛰어나지만, very high-fidelity dense mask modeling의 유연성 면에서는 후속 방식들보다 제한적일 수 있습니다. 이 평가는 논문이 보여준 구조적 특성에 기반한 해석입니다.

### 비판적 해석

제 해석으로 이 논문의 진짜 공헌은 “single-shot instance segmentation을 하려면 꼭 dense mask branch나 box-first pipeline이 필요한가?”라는 질문에 대해, **representation을 바꾸면 훨씬 단순한 해법이 가능하다**고 보여준 데 있습니다. PolarMask++는 mask를 contour regression 문제로 재정의함으로써, detection과 segmentation의 경계를 흐리게 만듭니다.

또 하나 중요한 점은 이 논문이 단순히 새로운 loss 하나를 제안한 것이 아니라, **표현–샘플링–손실–피처 피라미드**를 하나의 일관된 프레임으로 설계했다는 것입니다. 즉, polar representation을 쓰겠다면 centerness와 IoU도 polar space에 맞게 다시 설계해야 한다는 점을 잘 보여줍니다. 이 부분이 이 논문의 가장 설득력 있는 연구적 미덕입니다.

## 6. Conclusion

이 논문은 instance segmentation을 **polar coordinate 기반 contour prediction**으로 다시 정의한 **PolarMask / PolarMask++**를 제안합니다. 핵심은 객체를 중심점과 여러 방향의 ray length로 표현해, bounding box 없이도 mask와 detection을 통합된 방식으로 예측하는 것입니다. 이를 위해 **soft polar centerness**와 **polar IoU loss**를 도입해 center sample quality와 contour regression을 안정화했고, **Refined Feature Pyramid**로 scale-aware feature representation도 강화했습니다. 결과적으로 COCO, ICDAR2015, DSB2018 등에서 단순하면서도 강한 성능을 보였습니다.  

실무적으로는 파이프라인 단순화와 속도 측면에서 가치가 크고, 연구적으로는 instance segmentation을 **dense mask prediction이 아니라 structured contour regression**으로 볼 수 있다는 강한 관점을 제시한 논문입니다. 이후의 contour-based, polygon-based, shape-parameterized instance modeling을 이해할 때도 중요한 연결고리 역할을 하는 작업이라고 볼 수 있습니다.
