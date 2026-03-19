# Capsules for Biomedical Image Segmentation

## 1. Paper Overview

이 논문은 **capsule network를 biomedical image segmentation에 본격적으로 확장한 최초의 연구**를 목표로 합니다. 기존 capsule network는 주로 MNIST 같은 작은 해상도의 분류 문제에서 part-to-whole 관계와 pose 정보를 잘 다룰 수 있다는 장점이 있었지만, segmentation처럼 큰 입력 해상도와 픽셀 단위 출력을 요구하는 문제에는 메모리와 파라미터 비용이 너무 커서 사실상 적용이 어려웠습니다. 이 논문은 바로 그 병목을 해결하기 위해 **SegCaps**라는 convolutional-deconvolutional capsule architecture를 제안합니다. 핵심은 capsule의 표현력은 유지하되, routing 범위를 지역적으로 제한하고 transformation matrix를 공유해 계산량을 크게 줄이는 것입니다. 저자들은 이를 통해 최대 $512 \times 512$ 해상도 영상에서도 capsule 기반 segmentation을 실현했다고 주장합니다.  

이 연구가 중요한 이유는 biomedical image segmentation이 단순한 시각적 객체 분할이 아니라, 병변 위치 파악, 질병 진행 추적, 해부학적 구조 분석, CAD 시스템의 전처리 등 실제 임상과 분석 파이프라인의 핵심 단계이기 때문입니다. 특히 CT와 MRI에서는 병변에 따른 높은 intra-class variation, noise, scanner artifact, 비정상적인 형태 변화가 많아 일반 CNN 기반 segmentation이 형태 일반화 측면에서 약점을 가질 수 있습니다. 저자들은 capsule이 pose, 위치, 변형 정보를 벡터 형태로 표현할 수 있다는 점이 이런 문제에 더 적합하다고 봅니다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음 한 문장으로 요약할 수 있습니다.

**“capsule의 part-whole agreement와 pose-aware representation을 segmentation에 쓰고 싶다면, 전역 fully-connected routing을 그대로 유지할 것이 아니라, 이를 지역 제약(local constraint)과 parameter sharing으로 재설계해야 한다.”**

저자들이 제안한 핵심 기술적 아이디어는 세 가지입니다.

첫째, **locally-constrained routing**입니다.
원래 capsule routing은 모든 child capsule이 가능한 모든 parent capsule로 routing되는 구조라서 고해상도 입력에서는 연산량이 폭발합니다. 이를 해결하기 위해, 각 child capsule은 전체 공간이 아니라 **정의된 spatially local window 안의 parent들로만 routing**되도록 제한합니다. 이렇게 하면 part-to-whole 관계를 지역 영역 안에서 형성하게 되어 계산량이 크게 감소합니다.  

둘째, **transformation matrix sharing**입니다.
동일 capsule type 내에서 격자 위치별로 별도 transformation matrix를 두지 않고 공유함으로써 파라미터 수를 줄입니다. 다만 capsule type 간에는 공유하지 않아 각 타입의 의미적 다양성은 유지합니다. 이 선택은 CNN의 convolution weight sharing과 유사한 효율성을 capsule routing에 도입한 것이라고 볼 수 있습니다.

셋째, **deconvolutional capsules**입니다.
local routing만 사용하면 receptive field가 줄어들고 전역 문맥 손실이 생길 수 있습니다. 이를 보완하기 위해 저자들은 transposed convolution 방식으로 동작하는 deconvolutional capsules를 도입해, 전체 네트워크를 **U-Net 유사 encoder-decoder 구조**로 만듭니다. 즉, downsampling 과정에서 고수준 capsule 표현을 만들고, upsampling 과정에서 이를 다시 dense segmentation mask로 복원합니다. skip connection도 함께 사용합니다.  

이 세 가지의 조합이 바로 SegCaps의 핵심 novelty입니다. 논문은 이를 통해 capsule의 장점은 유지하면서도 실제 segmentation task에 필요한 입력 해상도와 출력 밀도를 감당할 수 있게 만들었다고 주장합니다.

## 3. Detailed Method Explanation

### 3.1 왜 기존 CapsNet은 segmentation에 부적합한가

논문은 먼저 기존 CapsNet이 왜 segmentation에 부적합한지를 매우 수치적으로 설명합니다. 예시로, $6 \times 6$ spatial grid에 32 capsule types, type당 8D capsule이 10개의 16D parent capsule로 routing되는 한 레이어만 해도 파라미터 수가

$$
10 \times (6 \times 6 \times 32) \times 16 \times 8 = 1,474,560
$$

개에 달합니다. 저자들은 이 한 레이어만으로도 자신들이 제안한 전체 SegCaps 네트워크와 비슷한 수준의 파라미터 수가 된다고 말합니다. 더 나아가 원래의 fully connected routing을 $512 \times 512$ 입력으로 확장하면 이론상 파라미터 수가 사실상 감당 불가능한 수준까지 증가한다고 설명합니다. 이 부분은 논문의 출발점이 단순한 “새 아키텍처 제안”이 아니라, **capsule segmentation의 계산 가능성 자체를 다시 설계하는 문제**였음을 보여줍니다.

### 3.2 SegCaps의 전체 구조

SegCaps는 deep **encoder-decoder capsule network**입니다. 입력은 예시로 $512 \times 512$ 크기의 영상이며, 첫 2D convolution layer가 16개의 feature map을 만듭니다. 이 출력은 첫 번째 capsule 집합이 되며, 이후 convolutional capsule layer와 deconvolutional capsule layer가 순차적으로 이어집니다. skip connection은 같은 spatial dimension을 가지는 earlier capsule type을 decoder 쪽과 concatenation하여 U-Net 스타일의 정보 전달을 수행합니다.

즉 구조적으로는 U-Net과 유사하지만, 각 spatial location의 표현이 scalar feature가 아니라 **vector capsule**이라는 점이 다릅니다. 각 capsule vector는 feature의 존재 여부뿐 아니라 위치, 방향, 변형 같은 속성을 담습니다. 저자들은 이것이 CNN보다 더 풍부한 representation을 제공한다고 봅니다.

### 3.3 Locally-Constrained Dynamic Routing

이 논문의 가장 중요한 기여는 routing 수정입니다. 기존 dynamic routing에서는 모든 child capsule이 모든 parent capsule에 vote를 보내는 구조였지만, SegCaps에서는 **정의된 kernel/window 내부의 parent capsule에만 routing**합니다. 논문의 설명에 따르면, 이렇게 하면 local neighborhood 안에서만 agreement를 계산하게 되어 메모리와 계산 부담이 현저히 감소합니다. 동시에 transformation matrices는 grid 위치별로 따로 두지 않고, capsule type 내부에서 공유합니다.

직관적으로 해석하면 이렇습니다.

* CNN의 convolution은 local receptive field를 통해 고수준 feature를 쌓아 올립니다.
* SegCaps의 locally-constrained routing도 비슷하게 **국소 영역 안에서만 part-to-whole composition**을 수행합니다.
* 다만 단순 convolution과 달리, capsule 간 agreement를 이용해 어떤 child가 어떤 parent에 얼마나 기여할지를 동적으로 조절합니다.

따라서 이 방법은 CNN의 효율성과 capsule의 구조적 표현력을 절충한 셈입니다.

논문 본문 조각에서 최종 segmentation mask는 **마지막 layer capsule vector의 길이(length)**를 계산하고, threshold를 넘어가면 positive class로 분류하는 방식으로 얻는다고 설명합니다. 즉 segmentation output 자체도 capsule vector magnitude에 기반합니다.

### 3.4 Deconvolutional Capsules

local routing만으로는 receptive field와 global context가 부족할 수 있기 때문에, 저자들은 이를 보완하기 위해 **deconvolutional capsules**를 제안합니다. 이는 transposed convolution 형태로 prediction vector를 만들고, routing 자체는 동일한 locally-constrained dynamic routing을 사용합니다. 쉽게 말해 convolutional capsule이 encoder라면, deconvolutional capsule은 decoder 역할을 합니다.

이 아이디어의 중요성은 capsule을 단순 분류용 shallow model에서 벗어나, 실제 segmentation에 필요한 **dense prediction network**로 확장했다는 데 있습니다. 기존 CapsNet은 기본적으로 얕고 작은 입력에 맞춰져 있었는데, SegCaps는 이를 deep encoder-decoder 구조로 일반화했습니다. 저자들도 이를 주요 공헌으로 직접 요약합니다.

### 3.5 Reconstruction Regularization의 확장

논문은 classification capsule network에서 쓰이던 **masked reconstruction regularization**을 segmentation으로 확장합니다. 즉, 정답 클래스 또는 positive segmentation과 관련된 capsule representation을 사용해 입력을 재구성하도록 하여 feature learning을 regularize합니다. 제공된 본문 조각에 따르면 total loss는 reconstruction loss와 weighted BCE segmentation loss의 합으로 구성됩니다. reconstruction loss의 weight는 대략 $0.0001$에서 $0.001$ 사이가 적절하며, 이보다 낮거나 높으면 성능이 저하된다고 설명합니다.

이 regularization은 단순 auxiliary loss라기보다, capsule vector가 실제 입력 구조의 의미 있는 속성을 담도록 유도하는 역할로 해석할 수 있습니다. 논문 후반 qualitative 결과는 capsule vector의 서로 다른 차원이 실제로 다른 해부학적 속성을 학습하고 있음을 보여준다고 주장합니다.

### 3.6 학습 설정과 출력 해석

실험 설정 조각에 따르면, 모델은 validation 2D Dice score를 기준으로 최대 **250,000 iterations**까지 학습하며, 테스트 시 segmentation score map의 threshold를 validation set에서 동적으로 선택합니다. 이는 biomedical segmentation에서 데이터셋마다 foreground/background imbalance와 intensity 특성이 다르기 때문에 합리적인 선택입니다. 또한 loss로 weighted BCE를 사용하는 점은 class imbalance 문제를 고려한 설계입니다.

## 4. Experiments and Findings

### 4.1 실험 대상과 검증 범위

이 논문은 단일 benchmark에 한정되지 않고 두 축으로 실험을 설계합니다.

* **Pathological lung segmentation from CT**
* **Muscle and adipose tissue segmentation from thigh MRI**

특히 폐 분할 실험은 저자들이 “문헌상 가장 큰 규모의 pathological lung segmentation study”라고 주장할 만큼 대규모입니다. clinical 및 pre-clinical 데이터를 포함한 5개 데이터셋, 총 **약 2000 CT scans**를 사용합니다. 또한 thigh MRI 실험은 **50명 피험자, 3개 contrast, 총 150 scans**를 포함합니다. 이 구성은 단순 성능 자랑보다는, 제안한 capsule segmentation이 modality와 anatomy를 넘어 일반화될 수 있음을 보이려는 의도로 읽힙니다.  

### 4.2 Lung Segmentation 결과

논문은 폐 분할 실험을 여러 데이터셋에 대해 보고합니다. 검색 결과 조각에 따르면 주요 clinical dataset인 **LIDC-IDRI는 885 CT scans**로 구성되며, 성능 평가는 3D Dice Similarity Coefficient와 Hausdorff Distance(HD)로 수행됩니다. 다른 데이터셋으로는 UHG, JHU-TBS, JHU-TB 등이 포함되며, 특히 JHU-TBS에 대해서는 pre-clinical subjects의 fully automated deep learning lung segmentation 결과를 처음 제시했다고 주장합니다.

정량 테이블 전체 숫자는 조각에 모두 드러나 있지는 않지만, 저자들의 서술은 일관됩니다. SegCaps는 **U-Net, Tiramisu, P-HNN**과 비교해 Dice와 HD 측면에서 더 나은 결과를 보였고, 동시에 파라미터 수는 훨씬 적었다고 말합니다. 특히 SegCaps는 U-Net 파라미터의 **4.6%**, P-HNN의 **9.5%**, Tiramisu의 **14.9%** 수준만 사용한다고 강조합니다. 이는 이 논문의 가장 강한 실험적 메시지입니다.

즉, 이 논문의 실험은 “capsule도 segmentation이 된다” 수준이 아니라, **매우 적은 파라미터로도 strong CNN baseline을 넘을 수 있다**는 점을 보여주려는 구조입니다.

### 4.3 MRI Muscle/Fat Segmentation 결과

MRI thigh segmentation 실험에서는 water and fat, water-only, fat-only의 세 contrast를 사용한 총 150 scans가 활용됩니다. 이 실험에서는 U-Net 및 기존 SOTA와 SegCaps를 **Dice coefficient** 기준으로 비교합니다. 저자들은 이 실험을 통해 CT 폐 분할과는 다른 modality와 anatomy에서도 SegCaps가 효과적임을 보이려 합니다.  

즉, SegCaps의 장점이 특정 데이터셋이나 특정 해부학 구조에만 국한되지 않는다는 점을 실험적으로 뒷받침하려는 것입니다.

### 4.4 Ablation과 해석

논문은 각 요소별로 ablation study를 수행했다고 명시합니다. 검색 결과 조각상 최소한 다음 요소들이 ablation 대상입니다.

* locally-constrained routing
* transformation matrix sharing
* reconstruction regularization
* baseline 3-layer capsule vs deep SegCaps

특히 reconstruction regularization weight는 성능에 민감하며, 적절한 범위는 $0.0001$–$0.001$라고 보고합니다. 이보다 지나치게 작거나 크면 성능이 떨어진다고 합니다.

또한 capsule vector의 여러 차원을 조작하거나 관찰한 qualitative 결과를 통해, 서로 다른 vector dimension이 실제로 다른 속성을 학습하고 있음을 보인다고 설명합니다. 이런 결과는 capsule representation이 단순 scalar map보다 해석 가능하고 속성 분해적(disentangled)일 수 있다는 저자들의 주장을 강화합니다.

### 4.5 추가 일반화 실험

논문은 부록 수준에서 capsule-based segmentation network가 **unseen rotations/reflections**에 대해서도 더 잘 일반화할 수 있음을 보여준다고 말합니다. 이는 capsule을 segmentation에 쓰는 철학적 동기와 직결됩니다. 즉, 의료영상에서도 병변 형태나 해부학 구조가 orientation, deformation, pathology에 따라 크게 달라지기 때문에, pose-aware representation이 일반 CNN보다 더 안정적으로 작동할 수 있다는 것입니다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 capsule network를 segmentation으로 확장할 때의 **핵심 계산 병목을 정면으로 해결했다는 점**입니다. capsule의 장점만 강조한 개념 논문이 아니라, local routing과 matrix sharing이라는 구체적 설계를 통해 실제 해상도에서 작동하는 시스템을 만들었습니다.

두 번째 강점은 **parameter efficiency**입니다.
SegCaps는 U-Net 대비 5% 이하의 파라미터로 강한 성능을 낸다고 주장합니다. biomedical segmentation에서는 데이터 규모가 한정적이고 과적합 위험이 크기 때문에, 적은 파라미터로 높은 성능을 내는 것은 실질적 가치가 큽니다.

세 번째 강점은 **representation의 철학적 일관성**입니다.
저자들은 왜 capsule이 segmentation에 적합한지, 왜 local routing이 필요한지, 왜 deconvolutional capsule이 필요한지 모두 명확한 논리로 연결합니다. 단지 CNN을 대체한 것이 아니라, segmentation에 맞는 capsule 연산을 설계했다는 점이 좋습니다.  

### Limitations

첫째, 논문은 capsule의 구조적 장점을 잘 제시하지만, 실제로 **왜 특정 데이터셋에서 CNN보다 우수한지에 대한 정량적 원인 분석**은 제한적입니다. 예를 들어 pathology severity, shape variation, data scarcity 각각에 대해 capsule이 어느 요인에서 가장 큰 이점을 주는지는 실험적으로 완전히 분해되어 있지 않습니다.

둘째, SegCaps는 기존 CapsNet의 비현실적인 비용을 크게 줄였지만, 여전히 CNN보다 구현과 이해가 단순하다고 보기는 어렵습니다. routing 자체가 iterative하고 capsule vector를 다루므로, 실무 배포 측면에서는 여전히 복잡성이 있습니다. 논문은 parameter 수 감소를 잘 보여주지만, runtime 효율성 전반이 언제나 CNN보다 우월하다고까지는 이 조각만으로 단정하기 어렵습니다.

셋째, 제공된 첨부 HTML 조각에서는 Section 6의 discussion/conclusion 전체 문장과 모든 정량 테이블 수치가 완전히 드러나지 않아, 일부 세부 수치는 논문이 주장하는 방향 중심으로만 해석할 수 있습니다. 따라서 특정 데이터셋별 세부 숫자 비교를 완전히 재현하는 수준의 분석은 이 첨부본 조각만으로는 제한이 있습니다.

### Interpretation

비판적으로 해석하면, 이 논문의 진짜 공헌은 “capsule이 segmentation에서도 CNN보다 좋다”를 증명했다기보다, **capsule을 dense prediction에 맞게 어떻게 재구성해야 하는가**에 대한 설계 원칙을 제시했다는 데 있습니다.

그 설계 원칙은 다음과 같습니다.

* global all-to-all routing은 버리고 local routing으로 바꾼다.
* type-specific semantics는 유지하되 위치별 matrix는 공유한다.
* encoder-decoder 구조와 skip connection을 도입해 dense output을 만든다.
* reconstruction regularization으로 capsule representation을 안정화한다.

이런 원칙은 이후 capsule을 detection, segmentation, 3D medical imaging에 확장하려는 연구들에게 일종의 blueprint 역할을 했다고 볼 수 있습니다.

## 6. Conclusion

이 논문은 **SegCaps**라는 deep convolutional-deconvolutional capsule network를 제안해, capsule network를 대규모 biomedical image segmentation에 처음으로 실질 적용한 연구입니다. 핵심 기술은 **locally-constrained routing**, **transformation matrix sharing**, **deconvolutional capsules**, 그리고 **segmentation용 reconstruction regularization**입니다. 이 설계를 통해 기존 capsule의 메모리/파라미터 폭증 문제를 해결하고, 최대 $512 \times 512$ 입력과 dense segmentation output을 처리할 수 있게 만들었습니다. 저자들은 pathological lung CT와 thigh MRI segmentation에서 SegCaps가 strong CNN baselines보다 더 좋은 결과를 내거나 최소한 경쟁력 있는 성능을 보이면서도, 훨씬 적은 파라미터만 사용한다고 주장합니다.

실무적으로는 파라미터 효율이 중요한 medical imaging 환경에서 의미가 있고, 연구적으로는 capsule representation을 dense prediction 문제로 확장하는 출발점으로 큰 의의가 있습니다. 다만 이 접근의 장점이 항상 CNN을 압도한다고 일반화하기보다는, **형태 변화와 pose 정보가 중요한 segmentation 문제에서 capsule의 잠재력을 보여준 논문**으로 이해하는 것이 가장 정확합니다.
