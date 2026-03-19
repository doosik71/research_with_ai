# Learning Equivariant Segmentation with Instance-Unique Querying

이 논문은 최근 instance segmentation의 주류가 된 **query-based segmentation**의 학습 방식 자체를 다시 설계한다. 기존 CondInst, SOLOv2, SOTR, Mask2Former 같은 방법들은 공통적으로 instance-aware query embedding으로 dense image feature를 조회해 instance mask를 만든다. 하지만 저자들은 이런 방법들이 사실상 **한 장면 내부(intra-scene)에서만 인스턴스를 구분하도록 학습**되기 때문에, query embedding의 판별력이 충분히 커지지 못한다고 본다. 이를 해결하기 위해 논문은 두 가지 학습 원리, 즉 **dataset-level uniqueness**와 **transformation equivariance**를 query-instance 관계의 핵심 속성으로 정의하고, 이를 기존 query-based 모델 위에 얹을 수 있는 일반적인 training framework로 제안한다. 핵심은 query가 자기 이미지 안의 대상만 맞히는 데 그치지 않고, **전체 학습 데이터셋 차원에서 고유한 인스턴스를 식별**하도록 강제하는 것, 그리고 crop/flip 같은 기하학 변환에 대해 **feature와 query가 함께 equivariant**하도록 학습하는 것이다. 논문은 이 방법이 구조 변경이나 추론 속도 저하 없이 COCO와 LVIS에서 일관된 AP 향상을 만든다고 주장한다.

## 1. Paper Overview

이 논문의 문제의식은 간단하지만 중요하다. query-based instance segmentation은 매우 강력하지만, 기존 학습은 기본적으로 **현재 이미지 안에서만 query와 instance를 매칭**한다. 예를 들어 한 장면 안에 고양이 두 마리가 있으면 두 마리를 분리하는 법은 배우지만, 데이터셋 전체에 있는 수많은 유사 인스턴스들 사이에서 query embedding이 얼마나 고유해야 하는지는 직접적으로 요구하지 않는다. 저자들은 이것이 learned query의 discrimination potential을 제한한다고 본다. 즉, 현재 방식은 “scene-local separation”에는 충분할 수 있지만, 더 강한 query embedding 학습에는 부족하다는 것이다.  

또 하나의 문제는 **기하학 변환에 대한 일관성 부족**이다. instance segmentation은 본질적으로 equivariant한 과제다. 이미지를 crop하거나 flip하면 instance 위치와 모양도 같이 바뀌어야 하며, query와 feature representation도 그 변화에 맞춰 변해야 한다. 그런데 일반적인 데이터 augmentation은 변환된 이미지를 그냥 새로운 샘플로 추가할 뿐, 원본과 변환본 사이 representation/query의 대응 관계를 직접 제약하지 않는다. 논문은 이 점을 두 번째 병목으로 본다.

이 문제가 중요한 이유는 query-based segmenter의 핵심이 결국 **query-instance 일대일 대응 관계**에 있기 때문이다. query가 더 고유하고 더 robust해지면 instance separation 자체가 좋아진다. 따라서 이 논문은 backbone이나 decoder를 바꾸지 않고도, **학습 목표 자체를 강화하면 성능을 의미 있게 끌어올릴 수 있다**는 관점을 제시한다.

## 2. Core Idea

논문의 핵심 아이디어는 다음과 같다.

> **좋은 query-based segmenter라면, query는 현재 이미지 내부에서만이 아니라 데이터셋 전체의 인스턴스들 사이에서도 충분히 고유해야 하고, 동시에 기하학 변환에 대해 일관되게 대응해야 한다.**

이 아이디어는 두 개의 축으로 구성된다.

첫째는 **dataset-level uniqueness**다. 기존 query-based 학습은 query가 자기 이미지의 정답 instance를 맞히면 된다. 하지만 이 논문은 query가 **다른 학습 이미지의 인스턴스 픽셀에는 mismatch**되도록까지 요구한다. 즉, query가 dataset-wide하게 더 배타적이고 discriminative한 embedding을 갖도록 만든다. 이게 논문 제목의 **Instance-Unique Querying**에 해당한다.  

둘째는 **transformation equivariance**다. 입력 이미지를 crop하거나 flip했을 때, image feature, instance representation, query embedding, 최종 mask prediction이 모두 그 변환에 맞춰 함께 바뀌어야 한다는 제약을 둔다. 저자들은 instance segmentation에는 invariance가 아니라 **equivariance**가 맞다고 강조한다. 왜냐하면 mask는 변환 후에도 동일해야 하는 것이 아니라, **같이 이동/반전되어야** 하기 때문이다.

중요한 점은 이 두 아이디어가 새로운 segmentation architecture가 아니라, **기존 query-based 모델에 추가되는 training framework**라는 점이다. 논문은 이 framework가 CondInst, SOLOv2, SOTR, Mask2Former 같은 서로 다른 계열 모델에 모두 들어갈 수 있다고 주장한다. 즉 novelty는 “새 head”가 아니라, **query learning principle의 재설계**에 있다.  

## 3. Detailed Method Explanation

### 3.1 기본 문제 정식화

논문은 query-based instance segmentation을 고전적인 관점에서 보면, 결국 **mask prediction + classification** 문제로 본다. 각 query embedding은 특정 인스턴스의 위치, appearance 같은 특징을 암묵적으로 담고 있으며, dense image feature를 조회해 대응 instance의 mask를 만든다. 기존 방법은 retrieved mask와 GT mask의 차이를 줄이는 방향으로 학습한다. 그런데 이 방식은 본질적으로 **같은 장면 안에서만 instance discrimination**을 학습하게 만든다. 저자들의 문제 제기는 여기서 출발한다.

### 3.2 Dataset-level uniqueness learning

논문이 제안하는 첫 번째 축은 **intra-scene + inter-scene instance disambiguation**이다. 현재 이미지 안의 정답 인스턴스에 query가 반응하는 것뿐 아니라, 다른 학습 이미지의 인스턴스 픽셀에 대해서는 query가 **mismatch**되도록 한다. 저자들의 표현대로라면, 더 이상 “single scene 내부 인스턴스만 구분”하는 것이 아니라, **whole training dataset의 모든 인스턴스를 구분할 수 있어야 한다**는 요구를 넣는 것이다.  

직관은 매우 강하다. 한 이미지 안의 객체 수와 배경 복잡도는 제한적이므로, intra-scene만으로는 query 학습 난도가 낮다. 반면 cross-scene retrieval을 요구하면, 비슷한 appearance를 가진 훨씬 더 많은 인스턴스들 사이에서 자기 대상을 구분해야 하므로 query embedding이 더 날카로워질 수밖에 없다. 논문은 이를 “paradigm shift in training query-based segmenters”라고 표현한다.

### 3.3 External memory와 샘플링

문제는 데이터셋 전체를 상대로 query-instance matching을 직접 계산하면 비용이 크다는 점이다. 이를 위해 논문은 **external memory**를 사용한다. 이 메모리는 large-scale query-instance matching을 위한 저장소 역할을 하며, 학습 중에만 사용된다. 중요한 실용적 포인트는 **external memory가 training 후 مباشرة 버려진다**는 점이다. 따라서 추론 시 추가 연산이 없다.

또한 논문은 픽셀 샘플링 방식도 조정한다. 인스턴스 전체 픽셀을 그대로 쓰면 큰 객체가 더 자주 샘플링되어 작은 객체 학습이 불리해질 수 있다. 그래서 **각 인스턴스 영역에서 고정된 수의 픽셀을 랜덤 샘플링**하는 전략을 택하고, 이것이 small instance 성능을 개선한다고 설명한다.

### 3.4 Transformation equivariance learning

두 번째 축은 **equivariance constraint**다. 논문은 crop/flip 같은 기하학 변환을 적용했을 때, 원본과 변환본 사이에서

* image feature,
* instance representation,
* query embedding,
* predicted mask

가 모두 대응되게 바뀌어야 한다고 본다. 즉 원본에서 객체를 찾는 query가 flip된 이미지에서도 같은 객체의 flip된 위치/모양을 가리켜야 한다.

저자들은 여기서 중요한 개념 구분을 한다. **Invariance는 instance segmentation에 부적절하다.** 만약 변환 후에도 representation이 바뀌지 않게 만들면, segmentation mask도 변환에 무감각해져야 하는 셈인데, 이는 픽셀 수준 예측과 맞지 않는다. 반면 equivariance는 입력 변환에 따라 출력도 함께 변하도록 강제하므로 segmentation task와 더 자연스럽다.

또한 이 제약은 일반적 data augmentation과도 다르다. augmentation은 transformed image를 독립된 샘플로만 쓰지만, 이 논문은 **원본과 변환본의 representation/query 간 관계 자체를 loss로 묶는다**. 논문은 이 equivariance constraint가 단순 transformation-based augmentation보다 더 큰 성능 향상을 준다고 주장한다.  

### 3.5 전체 training objective

논문이 제안하는 framework는 기존 모델의 원래 training objective에 **보완적으로 추가**된다. 즉 기존 segmentation loss를 버리는 것이 아니라,

* 원래의 mask/classification loss
* inter-scene uniqueness를 위한 추가 loss
* transformation equivariance regularization

을 함께 최적화한다. 이런 구조 덕분에 방법이 모델-agnostic하게 적용될 수 있다. 저자들은 실제로 네 가지 대표 query-based 방법에 동일한 철학을 적용한다.  

정리하면, 이 논문의 method는 새 architecture보다도 **query-instance 관계를 dataset-wide uniqueness와 geometric robustness의 관점에서 재정의한 학습 프레임워크**라고 보는 것이 정확하다.

## 4. Experiments and Findings

### 4.1 데이터셋과 평가 대상

논문은 주된 실험을 **COCO**에서 수행하고, 추가로 **LVISv1**에서도 평가한다. 적용 대상 모델은 다음 네 가지 대표 query-based 인스턴스 분할기다.

* **CondInst**
* **SOLOv2**
* **SOTR**
* **Mask2Former**

또한 backbone도 ResNet과 Swin 같은 서로 다른 설정에서 검증해, 특정 아키텍처에만 맞는 trick이 아니라는 점을 보이려 한다.  

### 4.2 메인 결과

논문이 가장 강조하는 결과는 구조 변경 없이도 일관된 AP 향상이 난다는 점이다. 초록과 서론에 따르면 COCO에서 네 대표 모델에 대해 대략 **+1.6에서 +3.2 AP** 범위의 향상을 보인다. 더 구체적으로 서론 요약에서는 COCO에서

* **CondInst**: +2.8 ~ +3.1 AP
* **SOLOv2**: +2.9 ~ +3.2 AP
* **SOTR**: +2.4 ~ +2.6 AP
* **Mask2Former**: +1.6 ~ +2.4 AP

의 개선을 보고한다. 또한 **LVISv1에서 SOLOv2 기준 +2.7 AP** 향상도 제시한다.  

이 결과의 의미는 분명하다. query-based segmentation의 성능 한계가 꼭 architecture 부족만은 아니며, **query embedding을 더 discriminative하고 equivariant하게 학습시키는 것만으로도** 여러 강한 baseline 위에서 성능을 끌어올릴 수 있다는 것이다.

### 4.3 Diagnostic experiment가 보여주는 것

논문은 진단 실험에서 세부 설계를 따로 검증한다. 검색 결과상 특히 다음 포인트가 강조된다.

* **inter-scene uniqueness loss의 설계**가 중요하다.
* **transformation equivariance loss**는 일반적인 data augmentation보다 더 큰 효과를 낸다.
* **pixel sampling strategy**는 small instance 성능에 영향을 준다.

특히 저자들은 equivariance constraint가 단순히 flip/crop augmentation을 더 많이 넣은 효과가 아니라고 주장한다. 즉 transformed samples를 독립적으로 쓰는 것보다, **원본-변환본 사이의 대응성을 직접 학습**하는 것이 더 중요하다는 것이다.  

### 4.4 Qualitative 결과

질적 결과에서도 논문은 baseline보다 더 나은 인스턴스 분리를 보인다고 설명한다. 예를 들어 appendix 설명에서는 baseline이 버스를 두 개로 잘못 쪼개거나, 두 마리 기린을 제대로 분리하지 못하는 사례에서 제안 방법이 더 정확한 예측을 만든다고 한다. 이는 더 discriminative한 query embedding이 실제로 **challenging scene에서 instance separation**을 개선했다는 정성적 근거다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제 설정이 날카롭다**는 점이다. 대부분의 논문이 backbone, decoder, head를 바꾸는 데 집중하는 반면, 이 논문은 성능 병목을 **query embedding learning 자체**로 본다. 그리고 dataset-level uniqueness와 transformation equivariance라는 두 개의 개념으로 이를 매우 명확하게 정리한다.

두 번째 강점은 **범용성**이다. 이 방법은 CondInst, SOLOv2, SOTR, Mask2Former처럼 서로 다른 계열의 query-based 모델에 모두 적용된다. 즉 특정 architecture에 종속된 기법이 아니라, query-based segmentation عامة에 적용 가능한 training principle이라는 점이 강하다.  

세 번째 강점은 **추론 비용 증가가 없다는 점**이다. external memory와 추가 손실은 학습 중에만 쓰이고, training 후 제거된다. 따라서 deployment에서 architecture 변경이나 speed delay가 없다는 실용적 장점이 있다.  

### 한계

첫째, 이 논문은 training framework 논문이므로, 성능 향상의 상당 부분은 **기존 baseline이 이미 충분히 query-based 구조를 잘 갖고 있다**는 전제 위에 있다. 즉 non-query-based 방법에 바로 같은 효과를 기대하긴 어렵다.

둘째, ar5iv HTML 조각만으로는 inter-scene matching loss와 equivariance loss의 **정확한 수식 형태를 완전하게 복원하기 어렵다**. 논문의 핵심 논리는 명확하지만, 구현 수준의 coefficient와 식 전개를 추적하려면 원문 PDF나 코드 저장소를 병행하는 편이 더 안전하다. 이는 이 답변의 정보 한계이기도 하다.

셋째, cross-scene uniqueness는 직관적으로 매우 좋지만, dataset-wide discrimination을 지나치게 강하게 요구하면 유사 인스턴스가 많은 데이터셋에서 class-level 공유 구조를 약화시킬 위험도 생각해볼 수 있다. 논문은 이를 잘 다루지만, 이 균형은 future work 여지가 있는 부분이다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “+2~3 AP 개선” 자체보다 더 크다. 더 중요한 것은 query-based instance segmentation을 **scene-local matching 문제에서 dataset-global matching 문제로 확장**했다는 점이다. 또 segmentation에서 transformation learning은 흔히 augmentation 수준으로만 생각되는데, 이 논문은 이를 **representation/query correspondence constraint**로 끌어올렸다. 이런 관점은 향후 query-based detection, panoptic segmentation, 더 넓게는 dense prediction 전반에도 연결될 가능성이 있다. 실제로 저자들도 conclusion에서 query-based detection과 panoptic segmentation으로의 확장 가능성을 언급한다.

## 6. Conclusion

이 논문은 query-based instance segmentation의 성능을 높이기 위해, query-instance 관계의 두 핵심 속성인 **dataset-level uniqueness**와 **transformation equivariance**를 학습 목표로 삼는 새로운 training framework를 제안한다. 핵심은 query가 현재 이미지 안에서만 인스턴스를 구분하는 것이 아니라 **전체 데이터셋 수준에서 고유한 인스턴스 식별자**가 되도록 만들고, 동시에 기하학 변환에 대해 **feature와 query가 함께 equivariant**하도록 하는 것이다. 이 방법은 architecture를 바꾸지 않고도 CondInst, SOLOv2, SOTR, Mask2Former에서 COCO 기준 일관된 AP 향상을 보였고, LVIS에서도 개선을 달성했다.  

실무적으로 보면 이 논문은 “좋은 segmentation model은 좋은 architecture만으로 완성되지 않는다”는 점을 잘 보여준다. 특히 query-based 방법에서는 **어떻게 query를 학습시키느냐**가 구조 못지않게 중요하며, 그 학습을 dataset-wide discrimination과 geometric robustness 관점에서 강화할 수 있다는 점이 이 논문의 핵심 메시지다.
