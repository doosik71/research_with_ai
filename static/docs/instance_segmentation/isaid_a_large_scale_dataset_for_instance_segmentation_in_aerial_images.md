# iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images

이 논문은 항공 영상(aerial imagery)에서 **instance segmentation**을 위한 첫 대규모 벤치마크 데이터셋 **iSAID**를 제안합니다. 기존 Earth Vision 데이터셋들은 대체로 semantic segmentation 또는 object detection 중 하나에만 적합했고, 픽셀 단위 instance mask를 충분한 규모와 다양성으로 제공하지 못했습니다. 저자들은 이 공백을 메우기 위해 2,806장의 고해상도 영상에 대해 15개 카테고리, 총 **655,451개 instance**를 정밀하게 주석한 데이터셋을 구축하고, Mask R-CNN과 PANet을 벤치마크로 평가합니다. 논문의 메시지는 분명합니다. **자연 영상용 off-the-shelf instance segmentation 모델은 항공 영상에서 그대로는 충분하지 않으며, 이를 위한 전용 방법론이 필요하다**는 것입니다.  

## 1. Paper Overview

이 논문이 해결하려는 핵심 문제는 **항공 영상에서의 정밀 instance segmentation 연구를 위한 적절한 대규모 데이터셋 부재**입니다. 자연 영상 분야에서는 MSCOCO, Cityscapes, ADE20K 같은 데이터셋이 detection/segmentation 연구를 크게 견인했지만, 항공 영상 쪽은 대체로 bounding box 중심이거나 단일 클래스 데이터셋에 머물러 있었습니다. 저자들은 이 차이가 단순한 “데이터 수 부족”이 아니라, 항공 영상 자체가 가지는 구조적 어려움 때문이라고 설명합니다. 항공 영상에는 한 장의 이미지 안에 객체가 매우 많이 들어가고, 방향이 제각각이며, aspect ratio와 scale variation이 극단적이고, 아주 작은 객체가 풍부하게 등장합니다. 이런 조건은 자연 영상용 알고리즘을 그대로 가져와 쓰기 어렵게 만듭니다.  

따라서 논문의 1차 기여는 새 알고리즘 자체보다, **연구 문제를 제대로 드러내는 데이터셋을 만든 것**에 있습니다. 그리고 2차 기여는 그 데이터셋 위에서 기존 대표 모델을 평가해, 항공 영상 instance segmentation이 실제로 얼마나 어려운지 정량적으로 보여 준 데 있습니다. 즉, 이 논문은 “새로운 방법을 제안하는 모델 논문”이라기보다, **문제를 정의하고 기준점을 세운 benchmark/dataset paper**로 읽는 것이 맞습니다.  

## 2. Core Idea

이 논문의 중심 아이디어는 크게 세 가지입니다.

첫째, **instance segmentation용 항공 영상 데이터셋을 detection/semantic segmentation의 단순 확장이 아니라 독립 문제로 다룬다**는 점입니다. 기존 항공 영상 데이터셋은 대개 oriented box, horizontal box, center point, 또는 단일 카테고리 polygon annotation 수준에 머물렀습니다. 그러나 instance segmentation은 각 객체의 정확한 경계를 픽셀 수준으로 알아야 하므로, 이런 coarse annotation만으로는 충분하지 않습니다. iSAID는 이를 위해 모든 객체를 polygon mask 수준에서 다시 주석했습니다.

둘째, **DOTA를 기반으로 하되 DOTA를 단순 변환하지 않고 scratch부터 재주석**했다는 점입니다. 저자들은 원본 DOTA에 잘못된 라벨, 누락된 instance, 부정확한 bounding box가 존재한다고 보고, iSAID를 독립적으로 처음부터 다시 만들었다고 명시합니다. 그 결과 원래 DOTA의 188,282 instance 대비 iSAID는 655,451 instance로 증가했고, 이는 상대적으로 약 **250% 증가**입니다. 즉, iSAID의 핵심은 “기존 detection dataset 위에 mask를 얹은 것”이 아니라 **annotation completeness 자체를 대폭 개선한 재구성**입니다.

셋째, 데이터셋의 설계 철학이 “어려운 실제 항공 장면”을 반영한다는 점입니다. 논문은 높은 객체 밀도, 큰 scale variation, 큰 aspect ratio, 여러 class의 동시 존재, 작은 객체 다수, class imbalance를 iSAID의 본질적 특징으로 내세웁니다. 이는 단순히 규모가 큰 데이터셋이 아니라, **현실적인 aerial perception 난제를 노출하도록 설계된 데이터셋**이라는 뜻입니다.  

## 3. Detailed Method Explanation

이 논문은 알고리즘 제안 논문이 아니므로 “방법”은 주로 **데이터셋 구축 절차와 벤치마크 설계**를 의미합니다.

### 3.1 데이터셋 구성

iSAID는 총 **2,806장 이미지**, **15개 카테고리**, **655,451개 instance**로 구성됩니다. 논문은 기존 소규모 aerial instance segmentation 데이터셋과 비교해 iSAID가 **15배 더 많은 object category**, **5배 더 많은 instance**를 갖는다고 설명합니다. 또한 항공 영상 width가 약 **800~13,000 pixels** 범위에 걸쳐 있어, 일반 자연 영상보다 훨씬 큰 해상도를 다룹니다.  

카테고리는 다음 15개입니다.
plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, swimming pool, soccer ball field.
이들은 overhead imagery interpretation에서 자주 쓰이고 실용성이 높은 범주들로 선택됐습니다.

데이터 분할은 원본 이미지 기준으로 **train = 1/2**, **validation = 1/6**, **test = 1/3**입니다. train/validation은 이미지와 GT를 공개하고, test는 이미지 공개 + annotation 비공개 평가 서버 방식으로 공정 비교를 유도합니다.

### 3.2 Annotation pipeline

논문에서 가장 중요한 부분 중 하나는 annotation quality control입니다. 저자들은 단순 crowd-sourcing이 아니라, **명확한 annotation guideline + annotator training + 다단계 검수** 체계를 만들었다고 설명합니다. annotation은 in-house 툴 **Haibei**를 사용해 polygon mask를 직접 그리는 방식으로 수행됩니다.

가이드라인의 핵심은 다음과 같습니다.

* 15개 카테고리에 속하는 **명확히 보이는 모든 객체**를 annotate한다.
* mask는 객체의 시각적 경계와 맞아야 한다.
* 필요한 경우 zoom in/out으로 경계를 정교하게 다듬는다.
* 애매한 경우 supervisor와 논의해 고신뢰 annotation을 만든다.
* 동일 시설, 동일 소프트웨어에서 일관되게 작업한다.

annotator는 사전 훈련과 평가를 거쳐 선발되며, 실제 데이터 작업 전 약 **4시간 훈련**을 받습니다. 한 장의 이미지 annotation에는 평균 **약 3.5시간**이 걸렸고, 전체 2,806장을 기준으로 약 **409 man-hours**가 소요됐다고 보고합니다. 이 수치는 항공 영상 instance mask annotation이 얼마나 노동집약적인지를 잘 보여 줍니다.

품질 관리도 매우 강합니다. self-review, peer review, supervisory sampling(70%), expert sampling(20%), 통계 기반 outlier double-check까지 포함하는 **5단계 품질 통제**를 수행합니다. 이 점은 iSAID가 단순히 “큰 데이터셋”이 아니라 **annotation completeness와 consistency를 매우 강하게 의식한 데이터셋**임을 의미합니다.

### 3.3 데이터 통계와 난이도 분석

논문은 iSAID의 난이도를 여러 통계로 설명합니다.

우선 image당 instance 수가 매우 많습니다. 평균 **약 239 instance per image**, 최대 **8,000 instance per image**까지 도달합니다. 이는 MSCOCO 7.1, Cityscapes 2.6, PASCAL-VOC 10.3, ADE20K 19.5, NYU Depth V2 23.5보다 훨씬 큽니다. 항공 영상의 넓은 시야(field of view) 때문에 parking lot, marina 같은 장면에서 객체가 매우 밀집됩니다.  

둘째, 객체 크기 분포가 극단적으로 불균형합니다. 논문은 object area 기준으로 small(10~144 pixels), medium(144~1024), large(1024+)를 정의하고, 비율을 각각 **52.0%, 33.7%, 9.7%**로 제시합니다. 즉, 작은 객체가 과반수입니다. 또 같은 class 안에서도 크기 범위가 매우 크고, 이미지 내 largest/smallest object area ratio가 **최대 20,000**까지 갈 수 있다고 설명합니다. 이는 멀티스케일 처리의 어려움을 극단적으로 보여 줍니다.

셋째, aspect ratio도 매우 큽니다. 평균 aspect ratio는 **2.4**, 최대는 **90**까지 도달한다고 합니다. 이는 자연 영상 대비 훨씬 이례적이며, bridge·harbor 같은 길쭉한 구조물에서 특히 문제를 일으킬 수 있습니다.

넷째, class imbalance가 존재합니다. small vehicle이 가장 빈번하고 ground track field가 가장 드뭅니다. 이런 class imbalance는 실제 응용에서는 자연스럽지만, 학습과 평가에서는 baseline 모델 성능을 크게 흔드는 요인이 됩니다.

### 3.4 벤치마크 실험 설계

저자들은 대표적인 natural-image instance segmentation 모델인 **Mask R-CNN**과 **PANet**을 baseline으로 채택합니다. Mask R-CNN은 meta algorithm으로 널리 쓰이고, PANet은 당시 강한 성능을 보였기 때문입니다. 그리고 원본 모델뿐 아니라 항공 영상에 맞춘 간단한 수정 버전도 함께 평가합니다. 평가 지표는 COCO-style instance segmentation AP, 즉 AP, $AP_{50}$, $AP_{75}$, $AP_S$, $AP_M$, $AP_L$를 사용합니다.

논문이 말하는 “simple modifications”의 핵심 메시지는, 복잡한 새 구조를 도입하지 않아도 **extreme-sized object** 대응을 조금만 강화해도 baseline 성능이 의미 있게 오른다는 것입니다. 즉, iSAID는 단순히 자연 영상 benchmark를 복제한 게 아니라, aerial domain에 맞는 특수 처리가 성능에 직접적인 영향을 주는 benchmark임을 보여 줍니다.  

## 4. Experiments and Findings

### 4.1 기존 방법의 직접 적용은 성능이 충분하지 않다

논문의 가장 중요한 실험적 결론은, **off-the-shelf Mask R-CNN과 PANet을 그대로 적용하면 성능이 suboptimal**이라는 점입니다. 이는 abstract에서도 직접 강조됩니다. 자연 영상에서 강한 모델이라고 해서 aerial imagery의 dense, tiny, high-resolution setting에서 잘 작동하는 것은 아니라는 뜻입니다.

### 4.2 PANet 계열이 Mask R-CNN보다 더 강하다

논문은 instance segmentation과 object detection 모두에서 **PANet 및 그 변형이 Mask R-CNN 및 그 변형보다 전반적으로 우수**하다고 보고합니다. qualitative result에서도 원본 Mask R-CNN은 누락 instance가 많고 가장 부정확한 편이며, **PANet++**가 가장 convincing한 mask를 제공한다고 평가합니다.  

또한 PANet++는 baseball diamond, basketball court, harbour 같은 일부 카테고리에서 **$AP_{50}$ 기준 약 5포인트 이상 향상**을 보였다고 합니다. 이는 aerial imagery에서 단순 backbone 성능보다도, 큰 객체와 작은 객체를 함께 다루는 멀티스케일/feature aggregation 설계가 중요하다는 해석으로 이어집니다.

### 4.3 데이터셋이 기존 detection benchmark보다 더 어렵다

논문은 detection 쪽 결과가 원래 DOTA에 보고된 수치보다 낮다고 언급하는데, 그 이유 중 하나로 **iSAID가 DOTA보다 훨씬 많은 instance를 포함하고 annotation이 더 완전하기 때문**이라고 해석합니다. 즉, iSAID의 성능 수치는 단순히 “모델이 나빠서” 낮은 것이 아니라, **benchmark 자체가 더 엄격하고 더 현실적이기 때문**입니다.

### 4.4 실험이 실제로 보여 주는 것

이 실험들이 보여 주는 것은 세 가지입니다.

첫째, 항공 영상 instance segmentation은 자연 영상보다 구조적으로 어렵습니다.
둘째, high-resolution, tiny-object-heavy, dense-scene 조건에 맞춘 specialized solution이 필요합니다.
셋째, iSAID는 그런 specialized solution의 효과를 드러낼 만큼 충분히 어렵고 큰 benchmark입니다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **benchmark로서의 설득력**입니다. 단순히 “큰 데이터셋”이 아니라, aerial instance segmentation이 왜 별도 연구 주제여야 하는지를 데이터 통계와 baseline 성능으로 잘 입증합니다. 객체 밀도, scale variation, 해상도, class imbalance, aspect ratio 모두가 극단적이라는 점이 강점입니다.  

두 번째 강점은 **annotation quality에 대한 강한 집착**입니다. scratch re-annotation, 5단계 검수, outlier double-check는 이 데이터셋의 신뢰성을 높입니다. dataset paper에서 흔히 문제되는 “양은 많지만 annotation noise가 큰 데이터셋”과는 다른 방향입니다.

세 번째 강점은 **기준선 제시의 실용성**입니다. Mask R-CNN과 PANet 같은 익숙한 모델로 baseline을 제공하기 때문에, 후속 연구자가 비교 기준을 잡기 쉽습니다.

### 한계

이 논문은 새 알고리즘 논문이 아니므로, 성능 향상 메커니즘에 대한 이론적 깊이는 제한적입니다. benchmark의 필요성을 잘 보이지만, “그 어려움을 어떻게 풀어야 하는가”에 대해서는 비교적 열린 문제로 남겨 둡니다. 이건 dataset paper로서는 자연스럽지만, 독자가 해결책까지 기대하면 아쉬울 수 있습니다.

또한 벤치마크 모델이 주로 Mask R-CNN/PANet 계열이라, proposal-free, center-based, transformer-style 방법에 대한 비교는 없습니다. 물론 논문 발표 시점의 맥락상 자연스럽지만, 지금 기준으로 보면 방법 다양성이 제한적입니다.

마지막으로, iSAID는 강력한 benchmark이지만 여전히 특정 source dataset(DOTA) 기반의 항공 장면을 출발점으로 합니다. 센서 다양성을 줄이려는 노력이 있었지만, 모든 remote-sensing 환경을 대표한다고 보기는 어렵습니다. 이 역시 데이터셋 설계의 현실적 한계입니다.

### 해석

비평적으로 보면, 이 논문은 “항공 영상 instance segmentation”을 자연 영상의 부차 과제가 아니라 **독립적인 난제**로 분리해 낸 논문입니다. 이후 aerial panoptic segmentation, rotated instance modeling, tiny object instance segmentation 같은 후속 연구들의 출발점으로 읽을 수 있습니다. 특히 “자연 영상용 좋은 모델이 aerial domain에서는 곧바로 통하지 않는다”는 메시지는 지금도 유효합니다.  

## 6. Conclusion

이 논문은 항공 영상 instance segmentation을 위한 대규모 데이터셋 **iSAID**를 제안하고, 그 필요성을 통계와 baseline 실험으로 설득력 있게 보여 줍니다. iSAID는 **2,806장 고해상도 이미지**, **15개 카테고리**, **655,451개 instance**를 포함하며, 기존 항공 영상 instance segmentation 데이터셋보다 훨씬 크고 다양하고 어렵습니다. 또한 DOTA 기반이지만 scratch부터 다시 주석해 annotation completeness를 대폭 높였습니다. 실험 결과는 Mask R-CNN과 PANet 같은 강한 자연 영상 모델도 항공 영상에서는 충분하지 않음을 보여 주며, specialized solution의 필요성을 분명히 합니다.

연구적으로 이 논문의 가치는 “더 좋은 모델”보다 “더 정당한 문제 정의와 benchmark”에 있습니다. 실무적으로도 항공 영상에서 객체를 정밀하게 분리해야 하는 감시, 도시 계획, 재난 대응, 원격 탐사 분석 등에 중요한 기반이 됩니다. 후속 연구를 이해할 때는, 이 논문을 **aerial instance segmentation의 MSCOCO/Cityscapes에 해당하는 출발점**으로 보면 가장 적절합니다.  
