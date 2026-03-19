# Instance Segmentation in the Dark

이 논문은 **극저조도(extremely low-light) 환경에서의 instance segmentation**을 본격적으로 다룬다. 기존 instance segmentation 모델들은 정상 조도에서는 잘 작동하지만, 매우 어두운 환경에서는 photon count 부족과 sensor noise 때문에 이미지 세부 정보가 사라지고 feature map 내부에 고주파성 disturbance가 생겨 성능이 크게 무너진다고 지적한다. 이를 해결하기 위해 저자들은 입력 영상을 먼저 enhancement/denoising하는 대신, **네트워크 내부 feature 자체를 더 robust하게 만드는 방향**을 택한다. 구체적으로 **Adaptive Weighted Downsampling (AWD)**, **Smooth-oriented Convolutional Block (SCB)**, **Disturbance Suppression Learning (DSL)**을 도입하고, 더 나아가 8-bit sRGB 대신 **14-bit RAW 입력**과 **low-light RAW synthetic pipeline**을 활용한다. 또한 실제 연구를 위해 **LIS(Low-light Instance Segmentation)** 데이터셋도 새로 구축했다. 논문은 이러한 설계로 별도 image preprocessing 없이도 SOTA 대비 약 **4% AP** 향상을 달성했다고 주장한다.  

## 1. Paper Overview

이 논문의 핵심 문제는 간단하다. **왜 기존 instance segmentation 방법은 어두운 환경에서 급격히 성능이 나빠지는가?** 저자들은 단순히 “이미지가 어두워서”가 아니라, 저조도 노이즈가 네트워크 feature map을 오염시켜 shallow layer에서는 edge 같은 저수준 정보가 깨지고, deep layer에서는 object semantic response가 약해지기 때문이라고 설명한다. 즉, low-light 문제를 픽셀 공간의 brightness 부족이 아니라 **feature space disturbance 문제**로 재해석한 것이다.  

이 문제가 중요한 이유는 자율주행, 로보틱스, 야간 감시 같은 실제 응용이 어두운 환경을 피할 수 없기 때문이다. 기존에는 low-light enhancement나 denoising을 preprocessing으로 붙인 뒤 segmentation을 수행하는 파이프라인이 흔했지만, 논문은 이런 방식이 latency와 계산량을 늘리며, 특히 8-bit sRGB 영상에서는 이미 잃어버린 정보를 복원하는 데 한계가 있다고 비판한다. Figure 1의 예시도 enhancement+denoising보다 RAW 기반 제안 방법이 더 나은 segmentation 결과를 낸다는 점을 시각적으로 보여준다.  

## 2. Core Idea

이 논문의 가장 중요한 아이디어는 다음 두 축으로 요약할 수 있다.

첫째, **이미지를 복원하지 말고 feature disturbance를 직접 억제하자.**
저자들은 저조도 이미지 노이즈가 CNN 내부 feature map에 high-frequency disturbance를 유발한다고 보고, 이를 줄이기 위한 경량 모듈 AWD, SCB와 학습 전략 DSL을 제안한다. 핵심은 segmentation backbone이 noisy input에서도 stable semantic response를 유지하도록 만드는 것이다.

둘째, **8-bit sRGB 대신 high-bit-depth RAW를 쓰자.**
저조도에서는 sRGB camera output이 scene information을 너무 많이 잃어버린다. 논문은 RAW가 더 높은 bit depth를 통해 어두운 환경에서도 더 풍부한 정보를 보존한다고 보고, RAW-input instance segmentation을 제안한다. 하지만 RAW segmentation dataset이 부족하기 때문에, 기존 sRGB dataset으로부터 realistic low-light RAW를 합성하는 pipeline을 함께 제시한다.  

즉 이 논문의 novelty는 단순한 backbone 변경이 아니라, **feature denoising-oriented architecture design + RAW-input training/data strategy + new benchmark dataset**의 결합이다.

## 3. Detailed Method Explanation

### 3.1 전체 프레임워크

논문의 전체 구조는 Figure 3으로 요약된다. 제안 방법은 기존 instance segmentation 모델 위에 다음 요소들을 더한다.

* **AWD**: feature downsampling 과정에서 노이즈를 줄임
* **SCB**: convolution 단계에서 smoothing-oriented branch를 추가해 robustness 강화
* **DSL**: 학습 중 disturbance-invariant feature를 유도
* **Low-light RAW synthetic pipeline**: RAW 기반 end-to-end 학습을 가능하게 함

중요한 점은 이 요소들이 **model-agnostic**하다는 것이다. 즉 특정 segmentation 아키텍처에만 묶이지 않고, 기존 instance segmentation 모델을 보강하는 방식으로 설계되었다. 저자들은 AWD만 소폭의 추가 계산량이 있고, SCB와 DSL은 학습 단계에만 관여하거나 re-parameterization이 가능해 inference cost 증가가 거의 없다고 강조한다.

### 3.2 문제 분석: feature noise

논문이 제안법으로 들어가기 전에 먼저 관찰하는 현상은 매우 중요하다. 저조도 노이즈는 shallow feature를 noisy하게 만들고, deep feature에서는 object semantic response를 약하게 만든다. 결과적으로 object recall이 낮아지고 segmentation 성능이 떨어진다. 저자들은 이 현상을 adversarial robustness 문헌과도 연결해 해석한다. 즉, noisy sample에 대해 feature를 안정화하는 것이 모델 robustness에 핵심이라는 관점이다.

이 해석은 논문의 방법 전체를 관통한다. 일반 low-light vision 논문이 pixel restoration에 더 집중하는 반면, 이 논문은 **feature restoration**을 중심에 둔다.

### 3.3 AWD: Adaptive Weighted Downsampling

AWD는 논문의 대표 모듈이다. 저자들은 vanilla ResNet의 stride 2 downsampling이 사실상 nearest-neighbor downsampling처럼 동작하여, feature noise suppression에 거의 도움이 안 된다고 본다. 반면 mean filter 같은 low-pass filter는 noise를 줄일 수 있다. 따라서 AWD는 **content-aware low-pass filter**를 생성해 feature map downsampling 시 주변 정보를 적응적으로 aggregation한다. 목적은 고주파 disturbance를 줄이면서도 중요한 scene detail은 유지하는 것이다.

Ablation에서도 이 관찰이 뒷받침된다. Table 1에서 baseline(None)은 **AP 38.0**인데, Gaussian 38.3, Bilateral 38.1, Mean 38.5, Spatial-variant 39.0으로 점진적 개선을 보이고, 최종 **AWD는 AP 39.3 / AP50 61.4 / AP75 40.2**로 가장 좋다. 동시에 disturbance metric도 **1.5292 → 1.3715**로 낮아져 feature disturbance 억제 효과를 보여준다. 계산량은 **109.95 GFLOPs → 110.25 GFLOPs**로 소폭 증가하는 수준이다. 즉 AWD는 적은 비용으로 가장 큰 denoising benefit을 주는 설계다.

### 3.4 SCB: Smooth-oriented Convolutional Block

SCB는 ordinary convolution에 **smooth-oriented convolution branch**를 추가한 블록이다. 이 branch는 training 중 여러 경로의 선형 결합으로 robustness를 높이고, inference 시에는 **일반 3x3 convolution으로 re-parameterize**할 수 있다. 즉 training-time robustness boost는 얻되 inference overhead는 남기지 않겠다는 설계다.

이 블록의 의미는 AWD와 잘 연결된다. AWD가 downsampling 시점의 노이즈 억제라면, SCB는 convolution 연산 전체가 noise-resistant feature를 형성하도록 돕는다. 따라서 논문 전체는 특정 한 지점만 보정하는 것이 아니라, **feature propagation 경로 전반에서 disturbance를 줄이는 구조**라고 볼 수 있다.

### 3.5 DSL: Disturbance Suppression Learning

DSL은 모델이 noisy low-light image에서도 clean input과 유사한 semantic response를 갖도록 유도하는 학습 전략이다. 논문이 강조하는 바는, 단순히 feature를 smoothing하는 것만으로는 충분하지 않고, **학습 과정 자체가 disturbance-invariant representation을 만들도록 설계되어야 한다**는 점이다. 즉 DSL은 AWD와 SCB가 만든 구조적 이점을 학습 수준에서 강화하는 역할을 한다.

업로드된 ar5iv HTML에서는 DSL의 모든 수식이 완전히 읽기 쉽지는 않지만, 논문의 서술상 목적은 분명하다. noisy low-light image에 대해 stable semantic feature를 유지하도록 supervision을 주어 final prediction robustness를 높이는 것이다.

### 3.6 Low-light RAW Synthetic Pipeline

RAW pipeline은 이 논문의 또 다른 핵심이다. 구성은 두 단계다.

* **Unprocessing**: 기존 sRGB image를 ISP 역변환해 RAW 형태로 되돌림
* **Noise injection**: physics-based noise model을 이용해 realistic low-light RAW noise를 주입

저자들은 기존 Poisson-Gaussian noise보다 더 현실적인 noise source를 반영하는 모델을 사용하며, photon shot noise, read noise, banding pattern noise, quantization noise 등을 고려한다고 설명한다. 이 덕분에 기존 label이 있는 sRGB dataset으로부터 **training 가능한 synthetic low-light RAW dataset**을 만들 수 있다.

이 부분은 실용적으로 매우 중요하다. 논문이 단순히 “RAW가 좋다”고 주장하는 데 그치지 않고, **RAW dataset scarcity**를 해결하는 학습용 data generation pipeline까지 제안했기 때문이다.

### 3.7 LIS 데이터셋

논문은 실제 저조도 instance segmentation 연구를 위해 **LIS (Low-light Instance Segmentation)** 데이터셋도 구축했다. 이 데이터셋은 **2230쌍의 low/normal-light image pair**로 구성되며, indoor/outdoor의 다양한 real-world scene을 포함한다. 또 **sRGB-dark, sRGB-normal, RAW-dark, RAW-normal**의 paired sample 구조를 제공한다. train/test split은 **1561 / 669 pair**다.  

이 데이터셋의 의미는 단순 benchmark 제공을 넘는다. 기존 nighttime detection/segmentation dataset들은 극저조도가 아니거나 instance-level mask annotation이 부족했는데, LIS는 정확한 pixel-wise instance annotation을 제공해 이 분야를 독립된 문제로 정립한다는 점에서 기여가 있다.

## 4. Experiments and Findings

### 4.1 무엇을 보여주려는 실험인가

실험의 목적은 세 가지다.

* feature denoising 모듈들이 실제로 low-light segmentation에 도움이 되는가
* RAW input이 sRGB보다 유리한가
* 제안 방법이 실제 low-light benchmark인 LIS에서 기존 방법보다 우수한가

즉 단순히 한 모델의 AP 개선을 보이는 것이 아니라, **feature disturbance suppression**, **RAW data utility**, **dataset-level generalization**을 함께 검증하는 구조다.

### 4.2 AWD 관련 ablation

Table 1은 AWD 설계 정당성을 잘 보여준다. 단순 low-pass filter들도 baseline보다 낫지만, content-aware한 **AWD가 가장 우수**하다.
성능은 baseline **38.0 AP**에서 AWD **39.3 AP**로 상승하고, AP50은 **59.9 → 61.4**, AP75는 **39.1 → 40.2**로 개선된다. disturbance 값도 가장 낮다. 이는 논문의 핵심 가설인 “downsampling 과정에서 feature noise suppression이 중요하다”를 정량적으로 지지한다.  

### 4.3 메인 결과

논문은 전체적으로 제안 방법이 **state-of-the-art competitors 대비 약 4% AP 높다**고 요약한다. 또한 LIS에서 accuracy뿐 아니라 inference speed 측면에서도 큰 폭의 우위를 가진다고 주장한다. 추상과 서론 단계에서 이 메시지가 반복되며, 이는 논문이 단순 accuracy paper가 아니라 **practical low-light instance segmentation framework**를 목표로 했다는 점과 맞닿는다.  

### 4.4 실험이 시사하는 바

정리하면 실험은 다음을 보여준다.

* 단순 enhancement+segmentation보다 **feature-space denoising이 더 효과적**일 수 있다.
* **RAW input**은 extreme low-light에서 scene information 보존 측면에서 유리하다.
* model-agnostic lightweight design으로도 의미 있는 성능 향상이 가능하다.
* LIS 같은 실제 데이터셋에서 improvement가 관찰되어, 단순 synthetic setting에 그치지 않는다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 저조도 문제를 **pixel restoration이 아니라 feature robustness 문제**로 재정의한 점이다. 이 덕분에 AWD, SCB, DSL이 개념적으로 일관되게 연결된다. 또 RAW input과 synthetic pipeline, LIS dataset까지 포함해 방법론과 데이터 측면을 동시에 해결했다는 점도 강하다.

두 번째 강점은 **실용성**이다. AWD만 약간의 계산량 증가가 있고, SCB와 DSL은 inference cost를 거의 늘리지 않도록 설계됐다. 이는 야간 비전처럼 latency가 중요한 환경에 적합하다.

세 번째 강점은 **dataset contribution**이다. LIS는 이 문제를 독립적인 benchmark로 다룰 수 있게 한다. low-light instance segmentation 연구 기반을 만든 점은 방법 성능만큼 중요하다.

### 한계

첫째, 논문의 성능 향상 논리는 매우 설득력 있지만, 업로드된 HTML만으로는 DSL의 정확한 수식적 정의와 일부 실험 테이블 전체를 완벽히 재현하기 어렵다. 따라서 구현 수준 재현을 하려면 원문 PDF나 코드 저장소를 병행하는 편이 좋다.

둘째, RAW 기반 접근은 강력하지만 실제 deployment에서는 **sensor/ISP 의존성** 문제가 남을 수 있다. 모든 시스템이 RAW를 쉽게 활용할 수 있는 것은 아니므로, sRGB-only 환경에서는 논문의 full benefit이 제한될 수 있다. 이 점은 논문이 직접 강하게 비판하는 기존 sRGB 한계와도 연결된다.

셋째, LIS가 중요한 benchmark이긴 하지만 데이터 수는 2230 pairs로 아주 거대한 수준은 아니다. 따라서 더 다양한 야외 조건, motion blur, extreme weather까지 포함한 확장은 미래 과제로 남는다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “저조도 segmentation용 모듈 3개”가 아니다. 더 근본적으로는 **low-light high-level vision을 feature disturbance suppression과 RAW information preservation의 결합 문제로 정식화했다는 점**이 핵심이다. 이 관점은 instance segmentation뿐 아니라 detection, panoptic segmentation, tracking에도 확장될 수 있다. 다만 그 확장은 논문이 직접 검증한 범위를 넘어서는 해석이므로, 가능성 수준에서 보는 것이 적절하다.

## 6. Conclusion

이 논문은 극저조도 환경에서 instance segmentation이 왜 어려운지를 feature-level 관점에서 분석하고, 이를 해결하기 위해 **AWD, SCB, DSL**을 제안한다. 동시에 **14-bit RAW 입력**, **low-light RAW synthetic pipeline**, 그리고 **LIS dataset**을 통해 데이터와 학습 문제도 함께 해결한다. 결과적으로 별도 image preprocessing 없이도 기존 방법보다 더 나은 정확도와 효율을 보이며, low-light instance segmentation을 하나의 독립적 연구 주제로 끌어올린 논문이라고 볼 수 있다.  
