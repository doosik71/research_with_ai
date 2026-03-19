# Instance-Specific Feature Propagation for Referring Segmentation

이 논문은 **referring segmentation**에서 자주 쓰이는 두 계열, 즉 **one-stage 방식**과 **two-stage 방식**의 장점을 동시에 취하려는 시도다. one-stage 방법은 이미지와 문장을 바로 융합해 픽셀 단위 마스크를 예측하므로 단순하고 효율적이지만, 이미지 안의 여러 후보 인스턴스 사이 관계를 명시적으로 모델링하지 못한다. 반대로 two-stage 방법은 인스턴스 proposal을 먼저 만들고 언어와 매칭하므로 객체 간 비교는 잘하지만, 언어 정보가 segmentation 단계에 직접 개입하지 못해 후보 품질에 성능이 묶인다. 이 논문은 이 간극을 메우기 위해, **각 인스턴스를 명시적으로 표현하는 Instance-Specific Feature(ISF)**를 만들고, 이들 사이 정보를 주고받는 **Feature Propagation Module(FPM)**로 목표 인스턴스를 식별하면서 동시에 마스크까지 생성하는 통합 프레임워크를 제안한다. 저자들은 이 방식이 RefCOCO, RefCOCO+, RefCOCOg에서 모두 기존 SOTA를 넘어선다고 주장한다.  

## 1. Paper Overview

이 논문이 다루는 핵심 문제는 다음과 같다. **자연어 표현이 가리키는 특정 객체를 이미지 안에서 정확히 찾아내고, 그 객체의 segmentation mask까지 생성하려면 후보 객체들 사이의 관계를 어떻게 모델링할 것인가?** 예를 들어 “the rightmost person” 같은 표현은 단순히 사람을 찾는 것이 아니라, 여러 사람을 서로 비교해야 풀린다. 기존 one-stage 방법은 이런 비교를 픽셀 공간의 융합 feature 안에 암묵적으로 맡기기 때문에 tangled layout이나 복잡한 관계 표현에서 약하다. 반면 two-stage 방법은 비교는 잘하지만, 언어가 segmentation 생성 과정에 직접 관여하지 못한다는 한계가 있다.  

이 문제가 중요한 이유는 referring segmentation이 vision-language 융합의 대표적 과제이기 때문이다. 단순 detection보다 더 정밀한 mask가 필요하고, semantic segmentation보다 더 강한 instance-level reasoning이 필요하다. 따라서 객체 간 비교, 언어 조건, 경계 품질을 동시에 만족시키는 구조가 핵심이다. 이 논문은 바로 그 세 요소를 함께 풀려 한다.

## 2. Core Idea

논문의 핵심 아이디어는 한 문장으로 요약할 수 있다.

> **이미지의 각 인스턴스를 grid 기반의 명시적 표현으로 만들고, 인스턴스들끼리 feature propagation을 수행해 목표 대상을 강조한 뒤, 그와 동시에 각 grid별 mask를 생성하자.**

구체적으로 저자들은 이미지를 $S \times S$ grid로 나누고, 각 인스턴스의 중심이 속한 grid가 그 인스턴스를 대표하게 만든다. 각 grid는 해당 인스턴스의 크기, 질감, 모양 등을 담은 **Instance-Specific Feature (ISF)**를 가진다. 이렇게 하면 이미지 안의 인스턴스들이 feature map 위에 공간적으로 정렬되어 표현되므로, 서로 비교하고 관계를 학습할 수 있다.  

이 위에서 저자들은 **bi-directional propagation**을 수행하는 FPM을 제안한다. 각 인스턴스의 ISF가 다른 모든 인스턴스와 정보를 교환하면서, 문장이 가리키는 대상을 점점 더 두드러지게 만드는 구조다. 동시에 **Segmentation Branch**는 각 grid에 대한 마스크를 생성하고, 최종적으로 Identification Branch가 선택한 grid의 마스크를 출력한다. 즉, 기존 two-stage처럼 “마스크를 먼저 다 만든 후 선택”하는 것이 아니라, **식별과 분할이 동시에, 협력적으로 일어난다**는 점이 이 논문의 novelty다.  

## 3. Detailed Method Explanation

### 3.1 전체 구조

논문 프레임워크는 크게 네 부분으로 요약된다.

* **Backbone**
* **Instance Extraction Module**
* **Feature Propagation Module (FPM)**
* **Mask Refinement Module**

Backbone은 이미지와 문장을 각각 인코딩한 뒤 vision-language fused feature를 만든다. 그다음 Instance Extraction Module이 이 fused feature로부터 각 grid에 대응하는 **ISF map**과 각 grid별 **coarse mask**를 만든다. FPM은 ISF들 사이 상호작용을 통해 어느 grid가 목표 인스턴스인지 결정한다. 마지막으로 선택된 coarse mask는 Refinement Module을 거쳐 더 정밀한 경계로 보정된다.  

### 3.2 Backbone

Backbone에서는 이미지와 문장을 각각 처리한다.

이미지 쪽은 **FPN**을 써서 세 개의 멀티스케일 feature $F_{vl}, F_{vm}, F_{vs}$를 추출한다. 문장 쪽은 **GloVe embedding**과 **bi-directional GRU**를 통해 hidden states를 만들고, self-attention으로 각 단어 중요도를 구한 뒤 최종 문장 표현 $F_t$를 얻는다. 이후 논문은 이 언어 표현과 시각 표현을 **dense multiplication** 방식으로 융합한다.

또한 흥미로운 점은 FPN feature를 단순히 독립적으로 쓰지 않고, **두 방향으로 cross-scale fusion**한다는 것이다. 논문 Figure 3 설명에 따르면 upsampling pathway의 출력은 Segmentation Branch로, downsampling pathway의 출력은 Identification Branch로 들어간다. 저자들의 의도는 분명하다. segmentation에는 더 높은 해상도 정보가, identification에는 더 응축된 semantic 정보가 중요하기 때문이다.

### 3.3 Instance Extraction Module

이 모듈은 두 개의 branch로 나뉜다.

* **Identification Branch**
* **Segmentation Branch**

Identification Branch는 이미지의 각 grid를 특정 인스턴스에 대응시키고, 그 인스턴스를 대표하는 **ISF**를 생성한다. 이 ISF는 size, texture, shape 같은 정보를 담는다. 중요한 것은 인스턴스별 feature가 공간적으로 배열된다는 점이며, 덕분에 이후 FPM이 모든 객체 간 관계를 다룰 수 있다.  

Segmentation Branch는 각 grid에 대해 하나의 마스크를 동시에 생성한다. 여기에도 언어 feature가 주입되어 목표 객체와 관련된 visual response를 강화한다. 그리고 최종적으로 Identification Branch가 선택한 grid 위치의 마스크를 결과로 사용한다. 논문은 이 segmentation part만 단독으로 보면 one-stage referring segmentation처럼 동작하지만, 인스턴스 awareness와 관계 모델링이 추가될 때 성능이 크게 오른다고 설명한다.  

### 3.4 Feature Propagation Module (FPM)

FPM이 이 논문의 진짜 핵심이다. 저자들은 기존 two-stage 방법 중 일부가 같은 semantic category의 소수 후보만 뽑아 관계를 본다고 지적한다. 하지만 이는 표현이 요구하는 비교 범위와 항상 일치하지 않는다. 그래서 이 논문은 **전체 인스턴스를 전역적으로 비교**하는 방식을 택한다.

FPM은 각 인스턴스의 ISF를 다른 모든 인스턴스와 교환하게 만들며, 이를 통해 “rightmost”, “closest”, “man next to the car” 같은 **비교적 표현(comparative relations)**을 더 잘 다룬다. 검색 결과에 따르면 propagation은 여러 방향(예: up, down, left, right 및 대각선)으로 이루어지는 bi-directional 구조로 설명된다. 중요한 것은 이 모듈이 객체 간 관계를 local하지 않고 **명시적이고 전역적인 비교 과정**으로 바꾼다는 점이다.  

### 3.5 Mask Refinement Module

Segmentation Branch가 직접 내놓는 마스크는 coarse하다. 저자들은 그 이유를 두 가지로 든다.

* computational resource 제약 때문에 output spatial size가 제한됨
* backbone의 상대적으로 high-level vision feature를 사용함

이를 보완하기 위해 논문은 **원본 이미지와 resize된 predicted mask를 함께 입력**으로 쓰는 refinement module을 둔다. 이 모듈은 세 개의 $3 \times 3$ convolution과 중간 upsampling으로 구성되며, 최종적으로 1-channel refined prediction map을 생성한다. 즉, coarse semantic localization 위에 low-level spatial detail을 다시 얹는 역할이다.

### 3.6 학습 관점

논문 전체를 보면 이 구조는 식별과 분할을 완전히 따로 풀지 않는다. 오히려

* grid 기반 explicit instance representation
* propagation에 의한 target identification
* language-conditioned mask generation
* low-level detail refinement

이 하나의 학습 체계로 묶여 있다. 이런 점에서 기존 one-stage보다 instance-aware하고, 기존 two-stage보다 segmentation 단계에 언어를 더 깊게 주입한다고 볼 수 있다.

## 4. Experiments and Findings

### 4.1 데이터셋과 평가

논문은 세 개의 대표 referring segmentation benchmark에서 실험한다.

* **RefCOCO**
* **RefCOCO+**
* **RefCOCOg**

그리고 주된 평가는 **IoU** 기반이다. 추가로 저자들은 targeting performance를 보기 위해 precision 계열 비교도 제시한다.  

### 4.2 메인 결과

논문은 세 benchmark 전부에서 이전 SOTA를 넘었다고 주장한다. 특히 benchmark 결과 설명에서, 저자들은 자기 방법이 기존 방법보다 **최대 1.5% 정도 superior**하다고 서술한다. survey 논문이 아니라 제안 논문 맥락에서 이 수치는 작아 보일 수 있지만, RefCOCO 계열처럼 이미 강한 베이스라인이 많은 영역에서는 의미 있는 개선으로 해석할 수 있다.  

또한 precision 비교에서는 MCN과 targeting 성능이 비슷한 수준이면서도, MCN이 mask+bbox를 함께 쓰는 데 비해 이 논문은 mask supervision만으로 그 수준에 도달했다고 해석한다. 그리고 Pr@0.6, 0.7, 0.8에서는 가장 높은 점수를 보였다고 보고한다.

### 4.3 Branch analysis

논문은 각 branch가 성능에 얼마나 기여하는지 보기 위해, branch output을 ground truth로 대체하는 분석을 한다.

* **Identification Branch: Prediction vs GT**
* **Segmentation Branch: Prediction vs GT**

Identification Branch의 출력을 GT target grid로 바꾸면, 모델의 식별 상한선을 볼 수 있다. 반대로 Segmentation Branch의 마스크를 GT mask로 바꾸면, 목표 grid 선택은 유지한 채 mask 품질의 상한선을 본다. 논문은 두 실험 모두 성능이 증가한다고 보고하며, 특히 **Segmentation Branch를 GT mask로 대체했을 때 더 큰 성능 상승**이 나타났다고 설명한다. 이는 현재 프레임워크에서 identification도 중요하지만, coarse mask 생성과 refinement가 여전히 더 큰 개선 여지를 갖는다는 뜻이다.  

### 4.4 정성적/실패 사례

질적 결과에서는 복잡한 관계 표현에 대해 꽤 설득력 있는 성능을 보였다고 한다. 다만 실패 사례도 명확히 제시한다. 논문은 특히 **두 오토바이가 겹치고 색도 비슷한 장면**처럼, 인스턴스 간 외관이 매우 유사하고 상호 얽혀 있는 경우 구분이 어렵다고 설명한다. 저자들은 이런 경우 정확도가 **40%–60% 수준**으로 떨어질 수 있다고 언급한다. 이는 FPM이 전역 비교를 도입했더라도, 시각적으로 매우 기만적인 상황에서는 여전히 한계가 있음을 보여준다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제 설정을 잘 재구성했다는 점**이다. 기존 one-stage와 two-stage를 대립적으로 보지 않고, 둘의 장점을 결합할 수 있는 구조를 만들었다. 즉, explicit instance modeling은 유지하면서도 segmentation 생성 과정에 언어가 직접 작용하게 만든다.  

두 번째 강점은 **ISF와 FPM이라는 명시적 관계 모델링**이다. referring segmentation의 어려움은 결국 “어느 객체가 문장을 만족하는가”인데, 이 논문은 그 문제를 픽셀 분류가 아니라 **인스턴스 간 비교 문제**로 정면에서 다룬다. rightmost, closest 같은 표현에 강해질 수밖에 없는 구조다.

세 번째 강점은 **식별과 분할을 하나의 체계로 묶은 점**이다. two-stage처럼 segmentation을 language-agnostic proposal에 맡기지 않고, segmentation branch에도 언어를 넣는다. 이 점에서 referring task에 더 맞는 구조라고 볼 수 있다.

### 한계

첫째, 논문 스스로도 segmentation branch의 output이 **coarse**하다고 인정한다. 그래서 refinement module이 필요하다. 이는 구조상 identification 논리는 강하지만, fine mask detail은 후단 보정에 많이 의존한다는 뜻이다.

둘째, 복잡하게 얽힌 유사 객체 장면에서는 성능이 크게 떨어질 수 있다. 즉 FPM이 관계를 잘 모델링하더라도, 시각적 구분 단서 자체가 약하면 여전히 어렵다.

셋째, grid-based assignment는 인스턴스 중심이 어느 cell에 속하느냐에 영향을 받으므로, 매우 조밀하거나 작은 객체가 많은 장면에서는 구조적 제약이 있을 가능성이 있다. 이 부분은 논문이 직접 크게 비판하진 않지만, 방법의 설계상 자연스럽게 따라오는 한계로 읽힌다.

### 해석

비판적으로 보면 이 논문의 진짜 기여는 “새 backbone”이 아니라, **referring segmentation을 explicit instance reasoning 문제로 다시 세운 것**이다. 이후 등장한 query-based vision-language segmentation이나 DETR류 인스턴스 표현과도 통하는 관점이다. 즉 이 논문은 단순히 성능만 올린 것이 아니라, one-stage referring segmentation의 약점이 무엇인지 구조적으로 보여준 사례로 읽을 수 있다.

## 6. Conclusion

이 논문은 referring segmentation에서 one-stage의 단순함과 two-stage의 instance awareness를 결합하기 위해, **Instance-Specific Feature(ISF)**와 **Feature Propagation Module(FPM)** 기반 통합 프레임워크를 제안했다. 각 인스턴스를 grid 기반 명시적 표현으로 만들고, 이들 사이 feature propagation을 통해 목표 객체를 식별하면서, 동시에 Segmentation Branch가 마스크를 생성한다. coarse prediction은 Refinement Module로 보완된다. 실험적으로는 RefCOCO, RefCOCO+, RefCOCOg 전부에서 기존 방법을 능가했고, 정량 비교와 branch 분석 모두 FPM과 refinement의 효과를 뒷받침한다.

실무적으로 이 논문은 “referring segmentation에서 핵심은 언어-시각 융합 그 자체가 아니라, **객체 후보들 사이 관계를 어떻게 명시적으로 모델링하느냐**”라는 점을 잘 보여준다. 이 메시지는 이후의 instance-aware vision-language segmentation 연구를 이해하는 데도 여전히 중요하다.
