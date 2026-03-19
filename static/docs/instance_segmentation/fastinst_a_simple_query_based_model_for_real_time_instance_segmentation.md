# FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation

FastInst는 **실시간 instance segmentation**을 목표로 한 **간단한 query-based 모델**이다. 논문의 문제의식은 분명하다. DETR/Mask2Former 계열의 query-based 방법은 NMS-free, end-to-end라는 장점이 있지만, 실제 **real-time benchmark**에서는 convolution 기반 방법들보다 속도-정확도 균형이 충분히 입증되지 않았다는 점이다. 저자들은 이 간극을 메우기 위해, Mask2Former의 큰 틀은 유지하되 query 초기화, decoder 업데이트 방식, masked attention 학습 방식을 단순하고 효율적으로 재설계한 **FastInst**를 제안한다. 논문은 COCO `test-dev`에서 **32.5 FPS / 40.5 AP**를 달성했고, 더 빠른 설정에서는 **53.8 FPS / 35.6 AP**도 가능하다고 보고한다.

## 1. Paper Overview

이 논문이 다루는 핵심 문제는 다음과 같다. **query-based instance segmentation 모델이 정말로 실시간 환경에서도 경쟁력이 있는가?** 기존의 region-based 방법은 proposal 중복 때문에 비효율적이고, FCN 기반 single-stage 방법은 빠르지만 여전히 NMS 같은 post-processing에 의존하는 경우가 많다. 반면 query-based 모델은 end-to-end set prediction이라는 미학적 장점이 있으나, attention 계산 비용, 무거운 pixel decoder, 많은 decoder layer 의존성 때문에 “빠르고 정확한” 실시간 모델로는 다소 불리했다. 이 논문은 그 한계를 구조적으로 분석하고, query-based 계열도 적절히 설계하면 실시간 인스턴스 분할에서 매우 강력할 수 있음을 보여주려 한다.

이 문제가 중요한 이유는 자율주행, 로보틱스, 비전 기반 edge 시스템처럼 **latency가 직접 성능 조건이 되는 응용**이 많기 때문이다. 저자들은 단순히 AP를 올리는 것이 아니라, **속도와 정확도를 동시에 끌어올리는 설계 원리**를 제시하는 것을 논문의 실질적 기여로 본다.

## 2. Core Idea

FastInst의 중심 아이디어는 “query-based framework를 유지하되, 비효율을 유발하는 세 지점을 정확히 손보자”이다. 논문은 이를 세 가지 핵심 설계로 정리한다.

첫째는 **instance activation-guided queries**이다. DETR식 zero query나 Mask2Former식 learned static query 대신, 이미지에서 이미 객체 가능성이 높은 pixel embedding을 뽑아 초기 query로 쓴다. 즉 query를 이미지-독립적 벡터로 두지 않고, **이미 객체 semantic을 담은 위치에서 시작**하게 만든다. 이렇게 하면 decoder가 오랜 refinement를 할 필요가 줄어든다.  

둘째는 **dual-path update strategy**이다. 기존에는 query만 주로 갱신되는 반면, FastInst는 **query feature와 pixel feature를 번갈아 업데이트**한다. 이 설계는 pixel feature 자체의 표현력을 높여 주고, heavy pixel decoder에 대한 의존을 줄인다. 동시에 query와 pixel 사이 직접적인 정보 교환을 강화해 적은 decoder layer로도 수렴이 가능하게 한다.  

셋째는 **ground truth mask-guided learning**이다. Masked attention은 query마다 attention 영역을 좁혀줘 효율적이지만, 잘못된 mask에 의해 suboptimal update로 빠질 위험이 있다. 이를 막기 위해 학습 시에는 마지막 layer의 bipartite matching으로 대응된 **ground truth mask를 사용해 decoder를 다시 통과**시키고, 고정된 matching assignment로 supervision을 준다. 이로써 query가 학습 초기에 더 올바른 foreground 전체를 보게 만든다.  

요약하면 FastInst의 novelty는 transformer를 더 복잡하게 만든 것이 아니라, **query initialization**, **query-pixel interaction**, **masked attention supervision**을 실시간 목적에 맞게 재설계한 데 있다.

## 3. Detailed Method Explanation

### 3.1 전체 구조

FastInst는 세 모듈로 구성된다.

* **backbone**
* **pixel decoder**
* **Transformer decoder**

입력 이미지 $\mathbf{I}\in\mathbb{R}^{H\times W\times 3}$를 backbone에 넣어 $C_3$, $C_4$, $C_5$ feature map을 얻고, 이를 256채널로 projection한 후 pixel decoder로 보낸다. pixel decoder는 향상된 multi-scale feature $E_3$, $E_4$, $E_5$를 출력한다. 이후 $E_4$에서 **IA-guided queries**를 뽑고, auxiliary learnable queries와 결합해 초기 query 집합 $\mathbf{Q}\in\mathbb{R}^{N\times 256}$를 만든다. Transformer decoder는 고해상도 pixel feature $E_3$를 펼친 표현과 이 query들을 함께 받아 class와 mask를 예측한다.

논문 그림 설명에 따르면 $N_a$개의 IA-guided query와 $N_b$개의 auxiliary learnable query를 합쳐 총 $N=N_a+N_b$개의 query를 사용한다. 기본 설정은 **100개의 IA-guided query와 8개의 auxiliary learnable query**다.  

### 3.2 Lightweight pixel decoder

FastInst는 multi-scale contextual feature가 segmentation에 중요하다는 점은 인정하지만, 그것을 위해 MSDeformAttn 같은 무거운 decoder를 쓰는 것은 실시간 목적에 맞지 않는다고 본다. 그래서 pixel decoder는 가볍게 두고, 대신 transformer 내부에서 pixel feature를 계속 refinement하는 쪽을 택한다. 논문은 이 설계 덕분에 pixel decoder가 mask prediction을 직접 책임질 필요가 없어져, 더 단순한 모듈로도 충분하다고 설명한다. 실제 구현에서는 **PPM-FPN**을 speed-accuracy trade-off가 좋은 선택으로 사용한다.  

이 선택은 단순한 엔지니어링 타협이 아니라 FastInst 철학의 일부다. 즉 **decoder를 밖에서 무겁게 하지 말고, query와 pixel의 상호작용 안에서 feature quality를 키우자**는 방향이다.

### 3.3 Instance activation-guided queries

FastInst가 가장 강조하는 부분이다. 기존 query-based 모델은 query를 zero-initialized 혹은 learned vector로 시작한다. 이 경우 query가 이미지별 객체 정보를 거의 갖지 않은 상태에서 decoder 여러 층을 거쳐 refinement되어야 하므로 비효율적이다. FastInst는 이를 바꾸기 위해 pixel decoder 출력 $E_4$ 위에 auxiliary classification head를 붙여 각 pixel의 class probability를 예측하고, foreground semantic이 높은 위치를 선택해 query로 사용한다.

직관적으로 말하면, “객체일 가능성이 높은 픽셀을 대표자로 뽑아 query로 쓰자”는 것이다. 이는 DETR류 query를 더 data-dependent하게 만드는 전략이다. 논문 실험에서는 이 방식이 zero query나 learned query보다 더 좋은 성능을 보이며, 특히 **decoder layer 수가 적을 때 더 효과적**이라고 보고한다. 이는 IA-guided query가 처음부터 더 좋은 object embedding을 갖고 시작함을 시사한다.

또한 query 수를 늘리면 object recall이 좋아져 성능이 올라가지만, 속도는 다소 희생된다. 흥미롭게도 **10개 query만으로도 31.2 AP**를 얻었다고 보고하는데, 이는 선택된 query 자체의 품질이 높다는 증거로 해석할 수 있다.

### 3.4 Dual-path Transformer decoder

기존 Mask2Former식 decoder는 주로 query 업데이트에 초점이 있다. FastInst는 여기에 **pixel feature 업데이트 경로**를 추가한다. 즉, query만 refinement하는 것이 아니라 query와 pixel feature를 번갈아 갱신한다. 논문은 이를 **dual-path update strategy**라고 부른다.

이 설계의 효과는 세 가지로 이해할 수 있다.

첫째, pixel feature의 표현력이 향상된다.
둘째, query와 pixel이 직접적으로 정보를 교환하므로 decoder 수렴이 빨라진다.
셋째, 더 적은 decoder layer로도 충분한 성능을 얻을 수 있다.  

논문은 실제로 dual-path 방식이 conventional single-query update보다 더 잘 동작하며, update 순서는 큰 영향을 주지 않았다고 보고한다. 이는 핵심이 update order보다 **query-pixel co-optimization** 자체에 있음을 보여준다.

또한 FastInst는 decoder layer를 아예 쓰지 않아도 **30.5 AP**를 얻는다고 설명하는데, 이는 IA-guided query 자체와 dual-path 구조가 매우 강한 초기 표현을 제공함을 시사한다. 그리고 성능은 대략 **6층 부근에서 포화**된다고 보고한다.

### 3.5 Ground Truth Mask-Guided Learning

Masked attention은 계산량을 줄이고 convergence를 돕지만, 각 query의 receptive field를 너무 일찍 잘못 제한할 위험이 있다. FastInst는 이를 보완하기 위해 **GT mask-guided learning**을 도입한다. 학습 시 standard masked attention에 사용되는 predicted mask 대신, 마지막 layer bipartite matching으로 대응된 ground-truth mask를 사용해 decoder를 다시 한 번 forward하고, 동일한 matching assignment로 supervision을 준다.  

이 방법의 의도는 분명하다. **query가 target object의 전체 foreground를 보게 함으로써**, attention이 잘못된 작은 영역에 갇히지 않게 한다. 논문은 이 기법이 최대 **0.5 AP** 성능 향상을 가져왔다고 보고하며, 여러 backbone에서도 효과가 유지된다고 설명한다.

### 3.6 Prediction과 score

FastInst에서는 refined pixel feature에 linear projection을 적용해 mask feature를 만들고, query에서 생성된 mask embedding과 곱해 각 query의 segmentation mask를 산출한다. 각 query는 class probability를 예측하며, 최종 confidence는 previous work와 마찬가지로 **class score와 mask score의 곱**으로 계산한다. 여기서 mask score는 foreground 영역에서의 mask probability 평균이다.

이 설계는 DETR류 set prediction의 깔끔함을 유지하면서도 segmentation 평가에 필요한 confidence scoring을 자연스럽게 구성한 것이다.

## 4. Experiments and Findings

### 4.1 실험 설정

주요 실험은 **MS COCO**에서 수행되며, 특히 real-time 비교를 위해 `test-dev` benchmark를 사용한다. FLOPs는 validation image 100장 평균, FPS는 **single V100 GPU**, batch size 1 기준으로 측정했다. 기본 입력 크기는 짧은 변 640, 긴 변 최대 864다.  

### 4.2 메인 결과

논문이 가장 강하게 내세우는 결과는 다음 두 점이다.

* **가장 빠른 모델**: ResNet-50 backbone에서 **35.6 AP / 53.8 FPS**
* **최적 trade-off 모델**: **40.5 AP / 32.5 FPS**

저자들은 이 결과가 이전 real-time instance segmentation 방법들보다 전반적으로 더 나은 speed-accuracy trade-off를 보인다고 주장한다. 특히 ResNet-50 기반 **FastInst-D1**은 strong convolutional baseline인 SparseInst보다 **0.9 AP 높고**, 학습 epoch도 적고 추론 시간도 짧다고 설명한다.  

또한 공정 비교를 위해 Mask2Former의 heavy pixel decoder를 PPM-FPN으로 바꾼 light version도 비교했는데, 이 설정에서도 FastInst가 **정확도와 속도 모두에서 더 낫다**고 보고한다. 이는 FastInst의 장점이 단순히 backbone이나 decoder 축소의 효과가 아니라, **구조 설계 자체**에 있음을 뒷받침한다.  

### 4.3 Ablation study가 보여주는 것

Ablation 결과는 논문의 기여를 상당히 설득력 있게 정리해 준다.

* **IA-guided queries**는 zero query나 learnable query보다 우수하다.

* 특히 decoder layer가 적을수록 이득이 커서, lightweight model 설계에 유리하다.

* **Dual-path update**는 single-path query-only update보다 더 좋다.

* 이는 lightweight pixel decoder 설정에서 query와 pixel의 공동 최적화가 중요함을 의미한다.  

* **GT mask-guided learning**은 최대 0.5 AP 개선을 보이며, masked attention의 학습 안정성과 object embedding quality를 높인다.

* **Pixel decoder 선택**에서는 강한 decoder일수록 성능은 오르지만 연산량도 증가한다.

* real-time 목적에서는 **PPM-FPN이 가장 좋은 균형점**이라고 결론짓는다.

* **Transformer decoder layer 수**는 늘릴수록 성능이 오르지만 약 6층 전후에서 포화된다.

* 이는 FastInst가 “많은 decoder layer가 필수”인 구조가 아님을 보여준다.

### 4.4 정성적 분석과 failure case

논문은 FastInst가 instance-level segmentation에서 Mask2Former보다 더 나은 qualitative 결과를 보인다고 말한다. 다만 failure case도 존재하며, 대표적으로

* duplicate prediction
* over-segmentation
* false positive / false negative

가 나타난다고 부록에서 분석한다. 이는 FastInst가 실용적으로 매우 강하더라도, set prediction 기반 모델이 가진 중복/분할 과잉 문제를 완전히 없애지는 못했음을 보여준다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제를 정확히 쪼개고, 각각에 대해 간단하지만 효과적인 해법을 준 점**이다. 많은 논문이 성능 향상을 위해 구조를 복잡하게 만들지만, FastInst는 오히려

* query를 더 잘 시작시키고
* query와 pixel을 함께 갱신하고
* attention 학습을 더 바르게 유도하는

세 가지 설계만으로 실질적 개선을 만든다. 이는 매우 좋은 engineering research다.

둘째, FastInst는 Mask2Former의 meta-architecture를 유지하면서도, **real-time setting에서 query-based 모델의 경쟁력을 처음으로 강하게 입증한 사례**라는 의미가 있다. 논문 메시지는 “transformer 기반 segmentation이 느리다”가 아니라 “잘 설계되지 않았을 뿐”에 가깝다.

셋째, ablation이 명확하다. 각 핵심 아이디어의 효과가 따로 검증되어 있어 기여 분리가 비교적 잘 된다.  

### 한계

첫째, 실험 중심은 거의 **COCO real-time benchmark**에 있다. 따라서 다른 도메인, 특히 객체 밀도나 shape prior가 크게 다른 문제에서 같은 우위를 유지할지는 추가 검증이 필요하다.

둘째, FastInst는 lightweight setting에서 강하지만, 절대적인 최고 성능을 추구하는 대형 모델 문맥에서는 heavy decoder나 더 복잡한 multi-scale attention이 다시 유리할 수도 있다. 즉 이 논문은 “최고 AP”보다 “실시간 효율”에 최적화된 논문이다.

셋째, 부록의 failure case가 보여주듯 duplicate prediction과 over-segmentation은 여전히 남아 있다. NMS-free 구조가 깔끔한 대신, 일부 edge case에서는 예측 중복이나 boundary over-splitting이 발생할 수 있다.

### 해석

FastInst를 비판적으로 해석하면, 이 논문의 핵심 기여는 “새로운 거대한 segmentation framework”가 아니라 **query-based segmentation을 실용화하는 세 가지 설계 원리**를 제시한 데 있다. 특히 IA-guided query는 이후 많은 vision transformer 계열에서 반복적으로 등장하는 “더 나은 initialization” 문제와 맞닿아 있고, dual-path update는 token과 dense feature의 상호작용을 효율적으로 설계하는 방향성과도 연결된다. 따라서 FastInst는 단일 모델 이상의 의미를 가지며, **실시간 transformer vision model 설계의 하나의 교과서적 사례**로 볼 수 있다.

## 6. Conclusion

FastInst는 query-based instance segmentation이 real-time 환경에서도 충분히 강력할 수 있음을 보여준 논문이다. Mask2Former의 틀을 유지하되, **instance activation-guided queries**, **dual-path update strategy**, **ground truth mask-guided learning**이라는 세 가지 설계를 통해 더 가벼운 pixel decoder와 적은 decoder layer로도 높은 성능을 달성했다. COCO에서 **32.5 FPS / 40.5 AP**라는 결과는 이 주장을 강하게 뒷받침하며, convolution 기반 real-time baseline뿐 아니라 light Mask2Former 대비에서도 우수함을 보인다.

실무적으로 보면 이 논문은 “transformer는 느리다”는 편견에 대한 반례이자, **효율적인 segmentation 모델 설계에서 initialization, feature interaction, supervision strategy가 얼마나 중요한지**를 잘 보여준다. 앞으로 real-time panoptic/semantic segmentation이나 edge deployment용 vision transformer 설계에도 좋은 출발점이 될 가능성이 높다.
