# Nuclei instance segmentation and classification in histopathology images with StarDist

이 논문은 형광 현미경용으로 개발된 **StarDist**를 **histopathology nuclei instance segmentation and classification** 문제로 확장한 작업이다. 저자들은 단순히 핵의 경계를 분할하는 데서 그치지 않고, 각 핵을 **세포 타입별로 분류**하는 기능까지 추가한 StarDist 변형을 제안한다. 핵심 질문은 세 가지다. 첫째, StarDist에 instance-level classification을 어떻게 자연스럽게 추가할 수 있는가. 둘째, 형광 영상 중심으로 설계된 이 방법이 H&E 염색 기반 병리 이미지에도 잘 작동하는가. 셋째, 실제 benchmark와 challenge 환경에서 HoverNet 같은 기존 histopathology 방법들과 비교해 어느 정도 경쟁력이 있는가. 논문은 Lizard 데이터셋과 **CoNIC 2022 challenge**를 통해 이를 검증하며, segmentation+classification task에서 preliminary와 final test phase 모두 1위를 달성했다고 보고한다.

## 1. Paper Overview

핵 인스턴스 분할과 분류는 computational pathology의 핵심 문제다. 병리 영상에서는 핵들이 촘촘히 붙어 있고 서로 접촉하는 경우가 많아, 단순 bounding-box detection이나 픽셀 그룹화 방식이 어려움을 겪는다. StarDist는 이런 상황에서 각 객체를 bounding box 대신 **star-convex polygon**으로 표현해, 둥글고 조밀한 핵 객체를 다루는 데 유리한 방법으로 이미 알려져 있었다. 이 논문은 이 StarDist를 병리 영상으로 옮기고, 여기에 핵 타입 분류까지 추가하는 것을 목표로 한다.

이 문제가 중요한 이유는 병리 영상 분석에서 “핵이 어디 있는가”만으로는 충분하지 않기 때문이다. 실제 응용에서는 epithelial, lymphocyte, plasma, eosinophil 등 **핵의 종류**도 알아야 하며, challenge metric 역시 단순 segmentation 품질이 아니라 **instance segmentation과 classification을 함께 평가**한다. 따라서 이 논문은 segmentation model을 classification-aware하게 확장하면서, domain shift, stain variability, severe class imbalance 같은 histopathology 특유의 문제를 함께 다룬다.

## 2. Core Idea

논문의 핵심 아이디어는 비교적 단순하지만 매우 실용적이다.

> **기존 StarDist의 “object probability + radial distance” 예측 구조에, 픽셀 단위 class probability를 예측하는 semantic head를 추가하고, 최종 인스턴스별 클래스는 해당 인스턴스 내부 픽셀들의 class probability를 집계해서 정하자.**

즉 StarDist의 장점인 **별모양 polygon 기반 instance representation**은 그대로 유지하면서, 네트워크가 동시에 **핵 클래스 분포**도 예측하게 만든다. 그 결과 하나의 모델이 instance segmentation과 classification을 함께 수행할 수 있게 된다.

하지만 논문의 진짜 기여는 이 확장 자체보다, **histopathology에서 실제로 잘 동작하게 만든 여러 엔지니어링 결정**에 있다. 저자들은 다음 네 가지를 특히 중요하게 보고한다.

* **H&E stain variability를 다루는 color augmentation**
* **심한 class imbalance를 다루는 balancing strategy**
* **test-time augmentation과 model ensemble**
* **StarDist polygon 결과를 더 안정적으로 만드는 shape refinement**

즉 이 논문은 “새로운 복잡한 아키텍처”보다, **StarDist를 병리영상 도메인에 맞춰 정교하게 적응시킨 실전형 프레임워크**에 가깝다.

## 3. Detailed Method Explanation

### 3.1 기본 StarDist 복습

원래 StarDist는 각 픽셀에 대해 두 가지를 예측한다.

1. **object probability**
   해당 픽셀이 객체의 일부인지 여부

2. **radial distances**
   그 픽셀 위치에서 여러 방향으로 객체 경계까지의 거리

이렇게 하면 각 픽셀은 자신이 속한 객체를 **star-convex polygon**으로 “투표”하게 되고, 이후 NMS를 통해 중복 polygon을 제거해 최종 객체 인스턴스를 얻는다. 핵처럼 비교적 둥글고 응집된 객체에는 이 표현이 잘 맞는다.

### 3.2 StarDist를 classification으로 확장

논문은 여기서 한 개의 head를 더 추가한다.

3. **class probabilities**
   각 픽셀에 대해 cell type class probability를 예측

즉 backbone CNN은 이제

* object probability
* radial distances
* class probabilities

를 동시에 출력한다. 이후 instance segmentation이 끝나면, 각 instance mask에 포함된 픽셀들의 class probability를 모아 **인스턴스 단위 클래스**를 결정한다. 이 방식은 detection head를 따로 두지 않고, segmentation 결과와 semantic class map을 자연스럽게 결합하는 구조다.

이 설계의 장점은 명확하다. 병리영상의 핵은 크기가 작고 밀집되어 있어 box 기반 classify-then-segment 구조보다, **정확한 instance mask를 먼저 잡고 그 내부의 semantic evidence를 집계**하는 편이 더 자연스럽다.

### 3.3 데이터와 클래스 불균형

저자들은 CoNIC 주최 측이 제공한 **Lizard dataset patch**만 사용해 학습한다. 데이터는 총 **4981장**, 각 이미지 크기는 **256 × 256 × 3**이며, 여섯 가지 핵 타입이 라벨링되어 있다.

* neutrophil
* epithelial
* lymphocyte
* plasma
* eosinophil
* connective tissue cell nuclei

문제는 클래스 분포가 매우 불균형하다는 점이다. epithelial nuclei가 전체의 **60% 이상**을 차지하는 반면, neutrophil과 eosinophil은 각각 **1% 미만**이다. 저자들은 전체의 **90%를 train, 10%를 internal validation**으로 사용한다.

이 논문에서 class imbalance는 매우 중요한 실험 포인트다. 저자들은 세 가지 전략을 비교한다.

* semantic head에 **class weights** 적용
* semantic head에 **focal loss** 적용
* minority class가 많이 들어 있는 patch를 더 자주 뽑는 **oversampling**

놀랍게도 가장 단순한 **oversampling이 압도적으로 가장 좋았다**고 보고한다. internal validation 결과에서 oversampling은 mPQ **0.5885**로, class weights **0.5099**, focal loss **0.4541**, no balancing **0.3900**보다 확실히 높다. 이건 꽤 중요한 메시지다. 병리영상 핵 분류처럼 작은 객체의 long-tail 문제에서는 loss engineering보다 **샘플 분포 자체를 바꾸는 것**이 더 효과적일 수 있다는 뜻이다.

### 3.4 Augmentation 전략

논문은 augmentation도 상당히 공들여 다룬다. 기본적으로는

* flips
* 90도 회전
* elastic deformation

같은 기하학적 augmentation을 on-the-fly로 적용한다. 여기에 더해 색상 계열 augmentation을 비교한다.

* brightness only
* brightness + hue
* brightness + H&E staining

internal validation에서는 **brightness only**가 가장 좋았고, mPQ **0.6034**를 기록했다. 반면 brightness + H&E는 **0.5884**, brightness + hue는 **0.5495**였다. 하지만 저자들은 challenge hidden test에서는 관찰이 달랐다고 말한다. 그쪽에서는 **staining augmentation이 더 큰 이득**을 줬다. 이 차이는 internal validation과 external challenge set 사이의 **domain shift** 때문이라고 해석한다. 즉 내부 분할에서는 단순 밝기 변화가 충분했지만, 외부 데이터에서는 stain variation 대응이 중요해졌다는 것이다.

이 부분은 병리영상에서 매우 현실적이다. 내부 validation만 보면 augmentation 선택을 잘못할 수 있고, 실제 generalization에는 **stain variability 대응**이 더 중요할 수 있다는 점을 보여준다.

### 3.5 학습 세부사항

모델은 다음 설정을 사용한다.

* **64 rays**
* **U-Net backbone depth 4**

손실 함수는 head별로 다르다.

* object probabilities: **binary cross-entropy**
* radial distances: **mean absolute error**
* class probabilities: **categorical cross-entropy + Tversky loss**

학습은 랜덤 초기화에서 시작해 **1000 epochs**, batch size **4**, Adam optimizer, 초기 learning rate **0.0003**으로 진행한다. validation loss가 개선되지 않으면 80 epoch 후 learning rate를 절반으로 줄인다. 최종 모델은 가장 낮은 validation loss를 기록한 checkpoint를 사용한다.

이 구성은 매우 “challenge practical”하다. backbone이나 loss가 과도하게 복잡하지 않고, 대신 data strategy와 inference strategy에 더 비중을 둔다.

### 3.6 Test-Time Augmentation (TTA)

저자들은 training-time augmentation을 많이 쓰더라도, **TTA가 여전히 유효하다**고 말한다. 구체적으로는

* 0°, 90°, 180°, 270° 회전
* 각 회전에 horizontal flip 적용 여부

를 조합한 총 **8개의 기하학적 TTA**를 사용한다. 각 변환에 대해 object probability, radial distances, class probabilities를 예측한 뒤, 이를 다시 원래 좌표계로 되돌려 element-wise averaging한다.

StarDist에서는 radial distances가 polar coordinate direction에 대응하므로, 단순히 이미지를 되돌리는 것만으로는 안 되고 **ray ordering permutation**도 함께 보정해야 한다는 점이 중요하다. 즉 TTA 구현조차 StarDist representation에 맞춘 세심한 처리가 필요하다.

### 3.7 Shape Refinement

논문이 제안한 또 하나의 실용적 아이디어는 **shape refinement**다. 기본 StarDist는 NMS에서 winner polygon 하나만 남기지만, 저자들은 winner가 억제한 polygon들까지 같이 모아 하나의 그룹으로 만든다. 이후 이 polygon들을 모두 binary mask로 rasterize한 뒤 **majority vote**로 최종 mask를 얻는다.

이 방법의 직관은 좋다. StarDist의 여러 픽셀이 같은 객체를 조금씩 다른 polygon으로 예측하는데, winner 하나만 남기면 shape 정보 일부를 버리게 된다. 반면 suppressed polygons까지 모아 다수결을 하면 더 매끈하고 안정적인 객체 경계를 얻을 수 있다.

### 3.8 Model Ensemble

논문은 여러 StarDist 모델의 CNN prediction 자체를 모아 ensemble하는 전략도 사용한다. 중요한 점은 TTA와 ensemble이 완전히 같은 방식으로 통합된다는 것이다. 즉,

* 모델 여러 개
* 각 모델마다 여러 TTA

의 prediction tensor를 모두 모아 averaging한 뒤, **NMS와 shape refinement는 마지막에 한 번만 수행**한다. 이 구조는 구현이 단순하면서도 효과적이다.

## 4. Experiments and Findings

### 4.1 평가 지표

논문은 CoNIC challenge의 지표를 그대로 사용한다. 핵심은 **panoptic quality(PQ)** 계열이다.

* **DQ (Detection Quality)**: F1 기반 detection quality
* **SQ (Segmentation Quality)**: matched instance들의 평균 IoU
* **PQ = DQ × SQ**
* **mPQ**: 클래스별 PQ 평균

즉 이 논문은 단순 segmentation IoU가 아니라, **검출, 분할, 분류를 함께 반영하는 multi-class panoptic quality**를 최종 성능으로 본다. 이는 이 논문의 구조적 선택, 특히 class imbalance와 classification head 개선에 민감한 평가 체계다.

### 4.2 Ablation 결과

Ablation Table 1은 꽤 설득력이 있다.

#### Class balancing

* none: mPQ **0.3900**
* focal loss: **0.4541**
* class weights: **0.5099**
* oversampling: **0.5885**

oversampling이 매우 큰 차이로 최고다. 특히 DQ가 **0.7186**으로 크게 높아지는 점을 보면, minority class를 더 잘 보게 만들면서 detection/classification matching 자체가 개선된 것으로 해석할 수 있다.

#### Color augmentation

* brightness: mPQ **0.6034**
* brightness + hue: **0.5495**
* brightness + H&E: **0.5884**

internal validation에서는 brightness-only가 가장 좋다. 하지만 저자들은 hidden challenge data에서는 staining augmentation이 더 강했다고 해석한다. 이 차이는 논문에서 매우 중요한 practical lesson이다. validation split이 external domain shift를 충분히 반영하지 못하면 augmentation 선택이 바뀔 수 있다.

#### Test-time strategy

* none: mPQ **0.5884**
* shape refinement: **0.5832**
* TTA: **0.5913**
* TTA + shape refinement: **0.5984**

test-time 전략의 이득은 class balancing만큼 크진 않지만, 꾸준히 도움이 된다. 특히 **TTA + shape refinement** 조합이 가장 좋다. SQ도 **0.8276**으로 가장 높아, refinement가 segmentation boundary를 다듬는 데 기여함을 시사한다.

### 4.3 Challenge 결과

논문 초록과 서론은 제안 방법이 **CoNIC 2022 segmentation and classification task에서 preliminary와 final test phase 모두 1위**를 차지했다고 명시한다. 또한 본문은 preliminary leaderboard 결과를 바탕으로, stain augmentation, ensemble, TTA가 외부 데이터에서 중요했다고 설명한다. 즉 이 방법은 internal validation 최적화에만 머문 것이 아니라, **hidden test generalization**에서도 강했다는 것이 저자들의 핵심 주장이다.

이 결과가 중요한 이유는 논문의 novelty가 거대한 새 architecture가 아니라, StarDist를 병리 도메인에 맞추는 일련의 설계 조합에 있기 때문이다. challenge 1위는 바로 이런 조합이 실제 benchmark 환경에서도 유효했음을 뒷받침한다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제와 방법의 정합성**이다. 핵은 대체로 둥글고 밀집되어 있어 star-convex polygon representation과 잘 맞는다. 따라서 StarDist를 병리 영상에 가져오는 선택 자체가 매우 타당하다. 여기에 classification head를 픽셀 기반으로 추가한 방식도 nuclei-like object에 잘 어울린다.

두 번째 강점은 **실전성**이다. 이 논문은 복잡한 새 backbone보다, 데이터 불균형, stain variability, TTA, ensemble, postprocessing 같은 실제 challenge 성능을 좌우하는 요소를 집요하게 분석한다. 특히 oversampling이 focal loss보다 훨씬 강했다는 점, hidden data에서는 H&E augmentation이 더 중요했다는 점은 현업에서도 바로 참고할 만한 결론이다.

세 번째 강점은 **segmentation과 classification의 자연스러운 결합**이다. 인스턴스를 먼저 star-convex polygon으로 분리하고, 그 내부 픽셀의 class probability를 집계하는 방식은 box classifier를 따로 두는 것보다 핵 문제에 더 잘 맞는다.

### 한계

첫째, 이 논문은 StarDist 기반이라 객체가 **star-convex에 가깝다**는 가정의 혜택을 본다. 핵에는 잘 맞지만, 매우 비정형적인 세포 구조나 복잡한 세포질 형태까지 일반화되지는 않을 수 있다. 이는 방법의 강점이자 동시에 적용 범위의 제약이다.

둘째, 성능 향상의 상당 부분이 **TTA와 ensemble**에 의존한다. challenge setting에서는 매우 효과적이지만, 실시간성이나 계산 예산이 제한된 환경에서는 동일한 이득을 그대로 얻기 어렵다. 실제로 저자들도 time limit 때문에 ensemble당 사용할 augmentations 수를 줄였다고 언급한다.

셋째, internal validation과 external hidden test에서 augmentation 선호가 달랐다는 점은, 모델 선택이 validation split에 민감할 수 있음을 보여준다. 즉 일반화 성능을 제대로 보려면 더 다양한 stain/domain 분포를 반영한 validation 설계가 필요하다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “StarDist에 classification head 하나 더 붙였다”가 아니다. 더 본질적으로는, **형광 현미경용 instance representation이 histopathology nuclei에도 충분히 유효하며**, 실제 성능은 architecture보다 **데이터 불균형과 domain shift를 어떻게 다루느냐**에 크게 좌우된다는 점을 보여줬다. 즉 computational pathology에서는 종종 더 복잡한 모델보다 **문제 구조에 맞는 representation + 강한 데이터 전략**이 더 중요하다는 메시지를 준다.

## 6. Conclusion

이 논문은 StarDist를 병리 영상 핵 분석으로 확장해, **nuclei instance segmentation과 classification을 동시에 수행하는 실용적 프레임워크**를 제안했다. 방법 자체는 간결하다. 기존 StarDist의 object probability와 radial distance 예측에 **semantic class head**를 추가하고, 인스턴스별 클래스는 해당 인스턴스 내부 픽셀의 class probability를 집계해 결정한다. 하지만 실제 성능을 끌어올린 핵심은 **oversampling 기반 class balancing**, **stain-aware augmentation**, **TTA**, **shape refinement**, **ensemble** 같은 실전적 설계다. 결과적으로 CoNIC 2022 segmentation and classification task에서 preliminary와 final phase 모두 1위를 차지했으며, 이는 StarDist가 histopathology nuclei 문제에도 매우 경쟁력 있음을 보여준다.
