# Camouflaged Instance Segmentation In-The-Wild: Dataset, Method, and Benchmark Suite

이 논문은 **camouflaged object segmentation**을 넘어, 한 장면 속 위장된 영역을 **개별 인스턴스 단위**로 분해하는 새로운 문제인 **camouflaged instance segmentation in-the-wild**를 제안한다. 저자들은 기존 camouflage 연구가 대체로 “위장된 영역이 어디인가”를 region level에서만 다뤘고, 더 나아가 **항상 위장 객체가 존재한다는 가정** 위에서 데이터셋과 모델을 설계해 왔다고 비판한다. 이에 대해 본 논문은 두 가지를 함께 제시한다. 첫째, camouflage 이미지와 non-camouflage 이미지를 모두 포함하고, instance mask까지 제공하는 대규모 **CAMO++** 데이터셋이다. 둘째, 서로 다른 instance segmentation 모델들의 장점을 장면별로 결합하는 **Camouflage Fusion Learning(CFL)** 프레임워크다. 논문은 이 데이터셋이 기존 CAMO, COD, MoCA보다 더 크고 더 다양하며 더 어려운 benchmark라고 주장하고, CFL이 기존 단일 모델보다 더 높은 AP를 달성한다고 보고한다.

## 1. Paper Overview

이 논문이 해결하려는 문제는 단순한 camouflage detection이 아니다. 저자들이 새로 정의하는 문제는 **camouflaged instance segmentation**으로, 각 픽셀을 위장/비위장으로 구분하는 것만이 아니라, **어느 픽셀이 어떤 위장 인스턴스에 속하는지까지 식별**해야 한다. 논문은 이를 기존 camouflaged object segmentation보다 더 어려운 문제로 설명한다. 왜냐하면 기존 task는 위장된 전체 foreground 영역만 찾으면 되지만, 새 task는 그 안에서 **객체 수와 경계, 인스턴스 identity**까지 분리해야 하기 때문이다. 또한 기존 방법들은 위장 객체가 항상 존재한다고 가정했지만, 실제 자연환경에서는 그렇지 않으므로, 논문은 **camouflaged instances are not always present**인 unrestricted setting을 더 현실적인 목표로 둔다.

이 문제가 중요한 이유는 위장 객체 인식이 단순 시각적 흥미를 넘어서 여러 응용과 연결되기 때문이다. 논문은 search-and-rescue, 야생 동물 탐지, 의료 진단, media forensics 같은 응용을 언급하며, 특히 region-level camouflage segmentation만으로는 장면 속 객체 개수나 개별 객체의 의미를 파악할 수 없다고 설명한다. 결국 이 논문은 camouflage 연구를 “숨은 영역 찾기”에서 “숨은 객체들을 개별적으로 이해하기”로 확장하려는 시도다.  

## 2. Core Idea

논문의 핵심 아이디어는 두 축으로 나뉜다.

첫째, **문제를 가능하게 만드는 데이터셋을 새로 만들자**는 것이다. 기존 CAMO, COD, MoCA 같은 데이터셋들은 규모가 작거나, non-camouflage 이미지의 GT가 없거나, object-level mask만 있고 instance-level benchmark로 쓰기 어려웠다. 저자들은 이를 해결하기 위해 **CAMO++**를 구축했다. 이 데이터셋은 **5,500장 이미지**, **93 categories**, **32,756 instances**, 그리고 camouflage / non-camouflage가 약 50:50으로 섞인 구조를 가진다. 특히 기존 COD는 camouflage instance GT는 제공하지만 in-the-wild setting을 충분히 반영하지 못한다고 논문은 주장한다. CAMO++는 바로 이 점을 보완한다.

둘째, **단일 instance segmentation 모델로는 camouflage 장면의 다양한 난점을 모두 커버하기 어렵기 때문에, 장면별로 더 잘 맞는 모델을 선택·융합하자**는 것이다. 이것이 CFL이다. 논문은 일반 instance segmentation 방법들을 CAMO++에 fine-tuning하면 어느 정도 작동하지만, 각 방법이 특정 장면에서는 강하고 다른 장면에서는 약하다고 본다. 그래서 image context를 학습해 **각 이미지에 대해 가장 적합한 모델 결과를 선택**하거나 결합하는 scene-driven fusion 전략을 제안한다. 즉 이 논문의 novelty는 “새로운 camouflage 전용 backbone”보다, **더 좋은 benchmark + 장면 문맥 기반 fusion framework**에 있다.  

## 3. Detailed Method Explanation

### 3.1 문제 정의: camouflaged object vs camouflaged instance

논문은 먼저 두 개념을 구분한다.

* **Camouflaged object**: 이미지 안의 위장된 foreground 픽셀 전체 집합
* **Camouflaged instance**: 개별 객체 인스턴스를 덮는 의미 있는 픽셀 집합

즉 기존 camouflaged object segmentation은 여러 위장 객체가 한 덩어리 region처럼 다뤄질 수 있지만, camouflaged instance segmentation은 그 안에서 각각의 객체를 분리해야 한다. 이 문제 정의는 논문 전체의 출발점이다. 저자들은 이것이 최초의 camouflaged instance segmentation 연구라고 주장한다.  

### 3.2 CAMO++ 데이터셋 설계

CAMO++는 논문의 중심 공헌이다. 데이터 구성은 다음과 같다.

* 총 **5,500 images**
* 그중 **2,700 camouflage images**
* **2,800 non-camouflage images**
* 총 **32,756 instances**
* train/test split:

  * camouflage: **1700 / 1000**
  * non-camouflage: **1800 / 1000**  

camouflage 이미지는 인터넷에서 수집한 4,000장 후보에서 중복과 저해상도를 제거하고, 기존 CAMO의 1,250장과 합쳐 최종 **2,700장**으로 만들었다. annotation은 10명의 annotator가 custom interactive segmentation tool로 수행했고, 이미지당 5–20분이 걸렸다고 설명한다. non-camouflage 이미지는 **LVIS**에서 사람이나 동물이 있는 2,800장을 골라 수집했고, camouflaged instance가 없도록 수동으로 선별했다. 이 점이 중요하다. 기존 camouflage 데이터셋은 보통 camouflage 이미지에만 집중했지만, CAMO++는 **negative context**까지 명시적으로 포함해 real-world setting을 시뮬레이션한다.

### 3.3 CAMO++가 왜 더 어려운가

논문은 CAMO++의 난도를 여러 통계로 설명한다.

첫째, **category diversity**가 크다. CAMO++는 **13 biological meta-categories**와 **93 categories**를 포함하며, 각 category당 평균 352 instances가 있다고 설명한다. 시각적 관점에서 다시 묶은 **8 vision meta-categories**도 제시한다. 이는 기존 COD의 69 categories / 5 meta-categories보다 넓은 범위다.

둘째, **instance density**가 높다. CAMO++는 이미지당 평균 **6.0 instances**를 가지며, COD는 평균 **1.2 instances**에 불과하다고 한다. 또한 이미지의 **51%**가 multiple instance를 포함하고, 그중 **38%**는 2–10개, **10%**는 11–30개, **3%**는 30개 초과 인스턴스를 포함한다. 즉 connected / overlapping camouflage instances가 많아 instance-level parsing이 훨씬 어렵다.

셋째, **small and tiny instances**가 많다. 논문은 CAMO++에 small and medium instance가 많이 포함되며, small instances가 **69.6%**를 차지한다고 설명한다. 추가로 CAMO++는 tiny부터 large까지 mask size diversity가 가장 크다고 주장한다. 이는 proposal-based instance segmentation 모델들에게 상당히 불리한 조건이다.  

넷째, **center bias가 약하다**. 기존 camouflage datasets는 위장 객체가 이미지 중앙에 오도록 crop된 경향이 강하지만, CAMO++에서는 인스턴스가 이미지 전역에 더 넓게 분포한다고 설명한다. 이는 detector가 중앙 prior에 덜 의존하도록 만든다.

### 3.4 CFL: Camouflage Fusion Learning

CFL은 논문의 방법론 공헌이다. 논문이 제시하는 기본 전제는 이렇다. 일반 instance segmentation 방법들, 예를 들어 Mask R-CNN 계열이나 single-stage 방법들은 CAMO++에 fine-tuning될 수 있지만, **각 방법이 장면 조건에 따라 장단점이 다르다**. 어떤 방법은 tiny instances에 강하고, 어떤 방법은 medium / large에 강하며, 어떤 방법은 cluttered background에서 더 robust하다. 그러므로 하나의 모델만 쓰기보다, 이미지 문맥을 보고 **해당 이미지에서 가장 잘 맞는 모델을 선택**하는 편이 낫다는 것이다.

논문은 CFL을 **scene-driven framework**라고 설명한다. 먼저 여러 instance segmentation 방법을 CAMO++에 독립적으로 학습시킨다. 이후 이미지의 visual deep feature를 이용해 어떤 모델 결과가 가장 적절한지를 adaptively 고르는 메타 선택기를 학습한다. 검색 결과에 Algorithm 1이 “Search results of the best model for each image”로 나타나는 점도, CFL이 본질적으로 **per-image model selection/fusion** 문제라는 해석과 맞는다. 즉 CFL은 mask를 직접 더 잘 만드는 구조라기보다, **복수의 강한 baseline을 image context로 조합하는 ensemble-on-demand**에 가깝다.  

### 3.5 Benchmark 설정

논문은 benchmark를 두 가지 설정으로 나눠 본다.

* **Setting 1**: camouflaged instances are **not always present**
* **Setting 2**: camouflaged instances are **always present**

이 구분은 중요하다. 기존 camouflage literature는 대체로 Setting 2에 가까웠고, 이미지에 위장 객체가 있다는 전제를 둔 경우가 많았다. 하지만 본 논문은 real-world in-the-wild를 반영하려고 Setting 1도 별도로 평가한다. 이것이 dataset 설계와 benchmark 철학의 일관성을 보여준다.  

## 4. Experiments and Findings

논문의 실험 메시지는 세 가지로 요약할 수 있다.

첫째, **CAMO++는 실제로 기존 데이터셋보다 더 어렵다.** cross-dataset generalization 결과 설명에서, 논문은 CAMO++ training images가 상대적으로 편향이 적고, CAMO++ testing images는 tiny objects, extreme background resemblance, distraction, occlusion, overlap 같은 어려운 사례를 많이 포함하기 때문에 가장 까다롭다고 말한다. 즉 dataset contribution이 단순히 큰 규모가 아니라, 실제 난도와 다양성 측면에서도 의미가 있다는 주장이다.  

둘째, **단일 state-of-the-art instance segmentation 방법들만으로는 camouflage instance segmentation이 아직 충분히 풀리지 않았다.** 논문은 benchmark section에서 방법들이 metric별로 조금씩 강점이 달랐으며, 전체적으로 dominant한 단일 방법은 없었다고 정리한다. 예를 들어 RetinaMask가 일부 small-object 관련 지표에서 낫고, MS R-CNN이 AP75나 medium/large 관련 지표에서 더 강한 식의 trade-off가 있었다고 설명한다. 이 분석은 CFL의 필요성을 뒷받침한다.

셋째, **CFL이 모든 metric에서 가장 좋은 성능을 보였다.** 논문은 CFL이 scene-driven adaptive selection을 통해 component model들의 장점을 활용하여 SOTA를 달성했다고 보고한다. 특히 benchmark 설명에서 CFL이 ResNet50-FPN, ResNet101-FPN, ResNeXt101-FPN backbone 기준 각각 **AP 19.2, 21.9, 25.1**을 달성했다고 밝힌다. 또한 AR 측면에서도 일관되게 더 좋았다고 한다. Setting 2에서도 CFL이 다시 전 metric 최고 성능을 보였다고 설명한다.  

하지만 동시에 논문은 현 수준이 충분하지 않다고도 인정한다. failure discussion 인용에서, benchmark 최고 수준조차도 여전히 **AP ≤ 25** 수준에 머무르며, accurate camouflaged instance segmentation of in-the-wild images is still far from being achieved라고 명시한다. 이 문장은 이 논문이 단순히 benchmark 승리를 보고하는 데서 끝나지 않고, 분야의 난도를 솔직하게 드러낸다는 점에서 중요하다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **문제 정의와 데이터셋 설계가 매우 설득력 있다**는 점이다. camouflage segmentation을 region-level에서 instance-level로 끌어올렸고, 동시에 이미지에 위장 객체가 없는 경우까지 포함해 in-the-wild setting을 반영했다. 이는 기존 camouflage literature의 중요한 가정을 깨는 변화다.

두 번째 강점은 **CAMO++의 질적·양적 확장성**이다. 기존 CAMO, COD, MoCA와 비교했을 때 이미지 수, 카테고리 수, 인스턴스 수, annotation richness에서 우위가 분명하다. 특히 **5,500 images / 32,756 instances / 93 categories**라는 수치는 camouflage 연구를 instance segmentation 수준으로 끌어올릴 수 있는 최소한의 기반을 마련한다.

세 번째 강점은 **CFL의 실용성**이다. 논문은 camouflage 전용 완전히 새로운 segmentation architecture를 만들기보다, 이미 강한 general instance segmenter들을 context-aware하게 융합한다. 이 접근은 engineering 관점에서 현실적이며, benchmark에서 실제 이득을 보였다.  

한계도 명확하다. 첫째, 방법론적으로 CFL은 매우 실용적이지만, 본질적으로는 **fusion/selection framework**이기 때문에 camouflage difficulty 자체를 근본적으로 해결한 것은 아니다. 둘째, benchmark 최고 수준도 AP가 25 전후에 머무르므로, 문제는 여전히 매우 어렵다. 셋째, 논문 스스로 contextual information과 motion information 같은 요소를 future work로 남기고 있어, 단일 이미지 appearance만으로는 camouflage instance segmentation에 본질적 한계가 있음을 인정한다.  

비판적으로 해석하면, 이 논문의 진짜 기여는 CFL보다도 **CAMO++와 task formulation**에 더 가깝다. 이후 연구가 더 좋은 transformer, multimodal fusion, temporal reasoning을 쓰더라도, 그 출발점은 “camouflage object segmentation만으로는 부족하다”는 이 논문의 문제 설정에 있다.

## 6. Conclusion

이 논문은 **camouflaged instance segmentation in-the-wild**라는 새로운 문제를 제안하고, 이를 뒷받침하는 대규모 데이터셋 **CAMO++**와 benchmark suite, 그리고 장면 문맥 기반 융합 방법인 **CFL**을 함께 제시했다. CAMO++는 **5,500장 이미지**, **2,700 camouflage / 2,800 non-camouflage**, **32,756 instances**, **93 categories**를 포함하며, 기존 데이터셋보다 더 다양하고 더 어려운 instance-level benchmark를 제공한다. 실험적으로는 일반 instance segmentation 모델들 사이에 뚜렷한 상보성이 존재하고, CFL이 이를 이용해 **AP 19.2 / 21.9 / 25.1** 수준의 최고 성능을 달성한다고 보고한다. 동시에 저자들은 현재 최고 성능도 여전히 낮아, accurate camouflaged instance segmentation remains far from solved라고 평가한다.
