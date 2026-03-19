# Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation

이 논문은 instance segmentation의 성능 자체보다도 특히 **mask boundary 품질이 낮다**는 문제를 정면으로 다룹니다. 저자들은 기존 모델의 마스크가 객체 내부는 대체로 맞추더라도, 경계에서는 저해상도 출력과 boundary pixel의 극심한 불균형 때문에 거칠고 부정확해진다고 지적합니다. 이를 해결하기 위해 어떤 특정 segmentation model을 새로 만드는 대신, **기존 instance segmentation 결과 위에 붙일 수 있는 post-processing refinement framework인 BPR (Boundary Patch Refinement)**를 제안합니다. 핵심은 전체 인스턴스를 다시 세분화하지 않고, **예측된 경계 주변의 작은 patch들만 고해상도로 잘라 따로 refinement**한 뒤 다시 조립하는 것입니다. 논문은 이 방식이 Mask R-CNN 결과를 Cityscapes에서 크게 개선하고, 다른 방법의 결과에도 전이 가능하며, “PolyTransform + SegFix”와 결합해 당시 Cityscapes leaderboard 1위를 달성했다고 보고합니다.  

## 1. Paper Overview

이 논문이 해결하려는 문제는 명확합니다. instance segmentation의 예측 마스크는 객체의 대략적 영역은 잡아도 **실제 object boundary와 정확히 정렬되지 않는 경우가 많다**는 것입니다. 논문은 특히 두 가지 원인을 듭니다. 첫째, Mask R-CNN의 $28\times 28$ mask 같은 저해상도 출력이나 일부 one-stage 모델의 낮은 출력 해상도 때문에 경계 근처의 세밀한 형태가 사라집니다. 둘째, boundary pixel은 전체 픽셀의 1% 미만으로 매우 적고 원래도 분류가 어려운데, 일반적인 학습은 이들을 충분히 강조하지 못해 내부 영역 위주로 최적화되는 편향이 생깁니다.  

이 문제가 중요한 이유는 instance segmentation이 자율주행, 로보틱스 같은 실제 응용에서 쓰이기 때문입니다. 객체 mask가 조금만 삐져나오거나 깎여도, 특히 얇은 구조나 작은 물체에서 품질 저하가 큽니다. 논문은 upper-bound 분석으로 이를 정량화하는데, 경계에서 1px/2px/3px 이내 오류 픽셀만 ground truth로 고쳐도 AP가 각각 **9.4 / 14.2 / 17.8** 상승한다고 보입니다. 즉 성능 병목이 경계에 집중되어 있다는 점을 매우 설득력 있게 보여줍니다.  

## 2. Core Idea

논문의 핵심 직관은 사람의 annotation 행동을 모방하는 것입니다. 사람도 처음에는 객체를 대략 분리한 뒤, 최종적으로는 **경계 부분을 확대해서(local zoom-in) 더 정교하게 수정**합니다. 저자들은 이 과정을 모델화해, 전체 mask를 다시 예측하는 대신 **boundary region만 “look closer” 해서 refine**하자고 제안합니다.

이 아이디어는 몇 가지 점에서 기존 방법과 다릅니다. BMask R-CNN, Gated-SCNN 같은 방식은 boundary-aware branch를 내부에 추가하지만 여전히 저해상도 표현 한계를 크게 벗어나기 어렵습니다. PolyTransform과 SegFix는 post-processing 계열이지만, 전자는 계산량이 크고 후자는 정확한 boundary prediction에 의존합니다. 반면 BPR은 **예측 경계 주변의 작은 patch만 잘라서**, RGB patch와 coarse mask patch를 함께 입력으로 받아 **이진 segmentation 문제**로 경계만 다시 푸는 구조입니다. 즉, 전체 문제를 “boundary-local binary refinement”로 단순화했다는 것이 새롭습니다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

BPR는 어떤 instance segmentation model의 coarse mask든 입력으로 받을 수 있는 post-processing framework입니다. 전체 흐름은 다음과 같습니다.

1. 기존 모델이 예측한 coarse instance mask를 받는다.
2. 예측된 boundary를 따라 작은 정사각형 patch 후보들을 조밀하게 배치한다.
3. 중복이 많은 후보들 중 일부를 NMS로 골라낸다.
4. 해당 위치의 **image patch + binary mask patch**를 추출한다.
5. refinement network가 각 patch를 독립적으로 이진 segmentation한다.
6. refined patch들을 원래 위치에 다시 조립해 최종 instance mask를 만든다.

이 과정의 중요한 점은 “전체 인스턴스를 다시 자르지 않는다”는 것입니다. 경계 주변만 보기 때문에 같은 연산량으로도 훨씬 높은 해상도로 처리할 수 있고, patch 내부에서는 boundary pixel 비율이 자연스럽게 커져 최적화 편향도 줄어듭니다. 논문이 이를 crop-then-refine 전략이라고 부르는 이유입니다.

### 3.2 Boundary Patch Extraction

저자들은 boundary patch 추출을 위해 **sliding-window 스타일** 방식을 사용합니다. 핵심 제약은 “patch 중심부가 boundary pixel을 포함해야 한다”는 점입니다. 이렇게 해야 추출된 patch가 실제로 refinement가 필요한 부위를 중심에 담게 됩니다. 이후 너무 많은 중복을 줄이기 위해 NMS를 사용하고, threshold를 조절해 speed/accuracy trade-off를 맞춥니다.

왜 이런 방식이 중요한지는 비교 실험에서 잘 드러납니다. 단순 pre-defined grid 방식은 foreground/background 비율이 지나치게 치우친 patch를 많이 만들고, whole instance patch 방식은 다시 내부 픽셀이 지배해 boundary refinement의 장점이 사라집니다. 반면 dense sampling + NMS는 boundary가 중심에 오도록 patch를 고르기 때문에 local ambiguity를 줄이고 학습을 더 잘 유도합니다. Table 4에서도 이 방식이 가장 좋습니다.  

### 3.3 Mask Patch의 역할

이 논문의 가장 중요한 설계 중 하나는 **RGB image patch만 쓰지 않고, coarse binary mask patch를 함께 넣는 것**입니다. 저자들에 따르면 mask patch는 두 가지 역할을 합니다.

* 학습 수렴을 빠르게 한다.
* “이 patch 안에서 어느 인스턴스를 refine해야 하는가”에 대한 위치 가이드를 제공한다.  

이 설계가 왜 중요한지 논문 설명이 매우 설득력 있습니다. 경계 patch에는 여러 인스턴스가 동시에 들어올 수 있는데, RGB patch만 보면 어떤 경계를 따라 foreground/background를 다시 나눠야 하는지 목표가 모호합니다. coarse mask patch를 함께 주면 refinement network는 인스턴스-level semantics를 새로 학습할 필요 없이, **기존 결정 경계 부근의 어려운 픽셀을 올바른 쪽으로 밀어내는 작업**에 집중할 수 있습니다. 이 직관은 ablation에서도 검증되며, mask patch를 제거하면 성능이 오히려 baseline보다 심각하게 떨어집니다.  

### 3.4 Boundary Patch Refinement Network

각 patch refinement는 사실상 작은 **binary semantic segmentation** 문제입니다. 논문은 입력 채널을 4개(3 RGB + 1 mask)로 만들고 출력 클래스를 2개(foreground/background)로 두면 어떤 semantic segmentation network도 쓸 수 있다고 설명합니다. 구현에서는 고해상도 표현 유지에 강한 **HRNetV2**를 refinement network로 사용합니다.

중요한 점은 이 네트워크가 전체 인스턴스 category를 분류하는 것이 아니라, 주어진 coarse mask를 바탕으로 **경계 근처의 hard pixel만 재판단**한다는 것입니다. 그래서 저자는 low-level cue, 예를 들어 색 일관성이나 contrast 같은 정보가 특히 중요하다고 해석합니다. 이는 일반 instance segmentation model이 고수준 semantics로 거친 mask를 만들고, BPR이 그 위에 저수준 경계 디테일을 보완하는 역할 분담으로 볼 수 있습니다.

### 3.5 Reassembling, Training, Inference

Refined patch들은 원래 mask에 다시 삽입됩니다. refinement되지 않은 픽셀은 기존 prediction을 그대로 유지하고, 겹치는 patch 영역은 **logit 평균 후 threshold 0.5**로 foreground/background를 결정합니다.

학습 시에는 predicted mask와 ground truth 간 IoU가 0.5보다 큰 인스턴스에서만 boundary patch를 추출하고, supervision은 해당 ground truth mask patch에 대한 pixel-wise BCE loss입니다. 중요한 것은 이 과정이 **기존 instance segmentation model을 직접 fine-tuning하지 않는다**는 점입니다. 즉, BPR는 완전히 별도 모듈로 학습되고 적용됩니다.

## 4. Experiments and Findings

### 4.1 데이터셋과 평가 지표

주요 실험은 **Cityscapes fine** 데이터셋에서 수행됩니다. train/val/test는 각각 2,975 / 500 / 1,525장이며, 해상도는 $1024\times 2048$입니다. 카테고리는 bicycle, bus, person, train, truck, motorcycle, car, rider의 8개입니다. 평가는 COCO-style mask AP와 함께, boundary 품질을 보기 위한 **boundary F-score (AF)**도 사용합니다. 이 AF는 이 논문의 주제와 매우 잘 맞는 보조 지표입니다.

### 4.2 Upper-bound 분석

논문 도입부의 upper-bound 실험은 이 연구의 강한 동기입니다. baseline Mask R-CNN이 AP 36.4인데, 예측 경계로부터 1px/2px/3px 이내의 오류 픽셀만 GT로 교체해도 AP가 각각 **45.8 / 50.6 / 54.2**로 급상승합니다. 특히 작은 객체에서 gain이 더 큽니다. 이는 “경계 refinement만 잘해도 전체 AP가 크게 오른다”는 논문 가설을 강하게 뒷받침합니다.

### 4.3 Mask patch의 효과

가장 인상적인 ablation은 mask patch 유무입니다. baseline Mask R-CNN은 AP 36.4 / AF 54.9인데, image patch만 넣고 mask patch를 빼면 AP가 **20.1**로 크게 붕괴합니다. 반면 mask patch를 포함하면 AP가 **39.8**, AF가 **66.8**로 올라갑니다. 저자들은 다중 인스턴스가 겹친 patch에서 target instance를 식별하는 데 coarse mask patch가 결정적이라고 해석합니다. 이 결과는 논문의 핵심 설계가 단순한 보조 장치가 아니라 필수 요소임을 보여줍니다.

### 4.4 Patch size, extraction scheme, input resolution

Patch size 실험에서는 **$64\times 64$ without padding**이 전반적으로 가장 좋습니다. 너무 작으면 문맥이 부족하고, 너무 크면 boundary에 덜 집중하게 됩니다.

Patch extraction scheme 비교에서는 제안한 **dense sampling + NMS**가 pre-defined grid나 instance-level patch보다 우수합니다. 논문은 그 이유를 foreground/background 비율 불균형과 boundary cue 부족에서 찾습니다. 즉, “어디를 볼 것인가” 자체가 refinement 품질을 크게 좌우합니다.  

Refinement network의 input size는 커질수록 좋아지다가 $256\times 256$ 부근에서 가장 좋습니다. Table 5에 따르면 AP는 64에서 39.1, 128에서 39.8, 256에서 **40.0**으로 올라가지만 512에서는 다시 39.7로 소폭 하락합니다. 속도는 반대로 17.5 FPS → 9.4 FPS → 4.1 FPS → 2 FPS 미만으로 줄어듭니다. 즉, 이 논문은 성능과 계산비의 trade-off를 꽤 명확히 제시합니다.

### 4.5 최종 성능과 전이성

논문은 Cityscapes에서 Mask R-CNN baseline 대비 **+4.3 AP** 향상을 강조합니다. 또한 PointRend, SegFix 같은 다른 boundary-improving 방법의 결과에도 전이 가능하다고 보고합니다. 즉 BPR이 특정 backbone이나 detector에 종속된 것이 아니라, **boundary correction ability 자체를 학습**한 모듈이라는 주장입니다.  

속도 면에서는 default setting 전체 파이프라인이 Cityscapes 한 장당 약 **211ms**이며, patch extraction 52ms, refinement 81ms, reassembling 78ms로 나뉩니다. 이는 PolyTransform보다 빠르다고 저자들은 주장합니다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 문제를 매우 정확히 짚었다는 점입니다. 이 논문은 “instance segmentation 전체를 더 복잡하게 만들기”보다, **정말 부족한 부분인 boundary만 고해상도로 다시 보자**는 직관적 해법을 제시합니다. upper-bound 실험과 ablation 모두 그 정당성을 잘 뒷받침합니다.

둘째, **model-agnostic post-processing**이라는 점이 강합니다. 어떤 two-stage, one-stage 모델에도 붙일 수 있고, 기존 모델을 다시 학습할 필요가 없습니다. 연구적 참신함뿐 아니라 실용성도 큽니다.  

셋째, 설계가 단순합니다. patch extraction, mask-guided binary refinement, reassemble이라는 구조는 이해하기 쉽고, semantic segmentation 발전을 refinement network에 곧바로 가져올 수 있습니다. 실제로 저자도 “semantic segmentation advances를 그대로 활용 가능”하다고 봅니다.

### 한계

논문도 간접적으로 몇 가지 한계를 드러냅니다. 첫째, COCO에서는 Cityscapes만큼 큰 개선이 나오지 않습니다. 부록에서 저자들은 **COCO의 polygon-based coarse annotation**이 실제 경계와 어긋나는 경우가 많아, local boundary refinement의 최적화 목표를 흐리게 만든다고 설명합니다. 즉 이 방법은 boundary annotation 품질에 민감합니다.

둘째, post-processing이므로 구조적으로 **완전한 end-to-end instance segmentation**은 아닙니다. 분명 실용적이지만, 여전히 원본 segmentation model과 refinement model이 분리되어 있습니다. 따라서 나중 세대의 integrated boundary refinement와 비교하면 과도기적 성격도 있습니다. 이 평가는 논문 구조에 근거한 해석입니다.

셋째, 계산량이 공짜는 아닙니다. patch extraction과 reassembling까지 포함한 전체 시간은 211ms 수준이며, 입력 해상도와 patch 수가 커지면 속도 저하가 분명합니다. 물론 PolyTransform보다는 빠르지만, 실시간성이 절대적으로 필요한 환경에서는 여전히 부담이 될 수 있습니다.  

### 비판적 해석

제 해석으로 이 논문의 핵심 가치는 “boundary 문제를 **local refinement task**로 재정의했다”는 데 있습니다. 많은 segmentation 논문이 backbone이나 head 전체를 더 복잡하게 만드는 반면, 이 논문은 error가 집중된 곳만 고해상도로 다시 보자는 선택을 했고, 그 결과가 매우 납득 가능합니다.

동시에 이 논문은 “좋은 instance segmentation = 좋은 coarse mask + 좋은 boundary correction”이라는 분해를 제안한 셈입니다. 이는 후속 연구에서 point-based refinement, boundary-aware loss, iterative correction 같은 흐름과도 잘 이어집니다. 절대적으로 새로운 end-to-end 패러다임이라기보다는, **실제 오류 패턴을 정확히 겨냥한 engineering-rich but insightful paper**로 보는 것이 적절합니다. 이 평가는 논문 본문과 실험 결과에 근거한 해석입니다.

## 6. Conclusion

이 논문은 instance segmentation에서 boundary quality가 왜 나쁜지 분석하고, 이를 해결하기 위해 **Boundary Patch Refinement (BPR)**라는 단순하고 효과적인 post-processing 프레임워크를 제안합니다. 핵심은 coarse instance mask의 경계 주변만 작은 patch로 추출해, RGB와 mask patch를 함께 이용해 고해상도 binary refinement를 수행한 뒤 다시 mask로 조립하는 것입니다. 이 접근은 저해상도 출력 문제와 boundary pixel imbalance 문제를 동시에 완화하며, Mask R-CNN은 물론 다른 segmentation 방법의 결과에도 전이 가능합니다.  

실무적으로는 기존 시스템을 크게 뜯어고치지 않고 boundary 품질을 개선할 수 있다는 점에서 가치가 크고, 연구적으로는 “전체를 다시 풀지 말고 boundary를 더 가깝게 보자”는 관점을 성공적으로 보여준 논문입니다. 특히 고품질 annotation을 갖는 데이터셋에서는 매우 효과적이며, boundary-centric refinement 연구의 좋은 기준점으로 남을 만합니다.
