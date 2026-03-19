# Box-supervised Instance Segmentation with Level Set Evolution

이 논문은 **pixel-wise mask annotation 없이 bounding box만으로 instance segmentation을 얼마나 정확하게 할 수 있는가**라는 문제를 다룬다. 기존 box-supervised instance segmentation은 대체로 pseudo mask를 따로 만들거나, 인접 픽셀 간 affinity를 강제하는 방식에 의존했는데, 저자들은 이런 접근이 배경 노이즈나 유사한 물체로부터 쉽게 오염된다고 본다. 이를 해결하기 위해 논문은 고전적인 **level set evolution**, 특히 **Chan-Vese energy** 기반 곡선 진화를 딥러닝 네트워크 안에 통합한 단일 단계(single-shot) box-supervised instance segmentation 방법을 제안한다. 핵심은 SOLOv2가 예측한 instance-aware mask map을 각 인스턴스의 level set function으로 보고, 입력 이미지와 deep structural feature를 함께 이용해 box 내부에서 에너지를 반복적으로 최소화하며 경계를 점진적으로 정교화하는 것이다. 논문은 COCO, Pascal VOC, iSAID, LiTS의 네 벤치마크에서 이 방법이 강한 성능을 보이며, 특히 remote sensing과 medical image처럼 배경 혼동이 심한 환경에서 큰 이점을 보인다고 주장한다.  

## 1. Paper Overview

이 논문의 연구 문제는 명확하다. fully supervised instance segmentation은 고품질 mask annotation에 크게 의존하는데, 이 주석 비용이 매우 크다. 따라서 box annotation만으로 instance mask를 학습하는 box-supervised setting이 중요해졌지만, 기존 방법들은 크게 두 부류로 나뉜다. 하나는 pseudo mask를 별도 네트워크나 post-processing으로 생성하는 방식이고, 다른 하나는 pixel-pair affinity를 이용해 end-to-end 학습하는 방식이다. 하지만 전자는 학습 파이프라인이 복잡하고, 후자는 “가까운 픽셀은 같은 라벨일 것이다”라는 단순 가정 때문에 주변 배경이나 유사한 객체의 노이즈를 쉽게 흡수한다. 논문은 바로 이 한계를 출발점으로 삼는다.  

이 문제가 중요한 이유는 instance segmentation이 자율주행, 로봇 조작, 원격탐사, 의료영상 등 다양한 응용의 핵심이기 때문이다. 특히 라벨링 비용이 큰 분야일수록 box-level supervision만으로 정밀한 경계를 복원할 수 있다면 실용적 가치가 매우 크다. 논문은 이 문제를 단순한 weak supervision 기법 문제가 아니라, **객체 경계를 box 내부에서 어떻게 더 안정적으로 수렴시킬 것인가**의 문제로 재정의한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음과 같다.

> **box 안에서 object boundary를 직접 pseudo label로 찍어내려 하지 말고, level set energy minimization을 통해 곡선을 점진적으로 진화시켜 경계에 맞추자.**

저자들은 classical variational segmentation의 level set formulation을 box-supervised instance segmentation에 가져온다. 이때 단순히 전통적인 energy minimization을 후처리로 붙인 것이 아니라, **fully differentiable energy function**으로 구성해 네트워크 안에서 end-to-end로 학습되게 만든 것이 핵심이다. 즉, 네트워크가 바로 mask를 정답처럼 맞히는 것이 아니라, **level set function의 연속적인 진화 과정을 학습**한다는 점이 novelty다.  

또 다른 중요한 아이디어는 입력 데이터 항(data term)에 **원본 이미지와 deep structural feature를 함께 사용**한다는 점이다. 전통적인 Chan-Vese는 주로 저수준 intensity uniformity를 기반으로 경계를 찾는데, 저자들은 여기에 long-range dependency를 담은 deep feature를 추가해 더 robust한 curve evolution을 유도한다. 그리고 각 단계마다 **box projection function**을 사용해 초기 boundary를 자동으로 생성함으로써, weak supervision 환경에서도 안정적인 초기화를 가능하게 한다.  

정리하면 이 논문의 novelty는 “새로운 segmentation backbone”이 아니라, **box supervision + SOLOv2 + differentiable level set evolution + automatic box-based initialization**을 하나의 단일 학습 프레임으로 묶었다는 데 있다. 저자들은 이것이 box-supervised instance segmentation을 다루는 최초의 deep level set 기반 방법이라고 주장한다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

전체 프레임워크는 **SOLOv2 기반**이다. SOLOv2가 full-image size의 instance-aware mask map을 예측하면, 논문은 이 mask map 자체를 각 객체의 level set function $\phi$로 해석한다. 이후 box 내부에서 energy minimization을 반복 수행해 level set curve가 객체 경계 쪽으로 진화하도록 만든다. Figure 1 설명에서도 positive mask maps가 bounding box region 안에서 level set evolution으로 얻어지며, iterative energy minimization을 통해 box annotation만으로 정확한 instance segmentation이 가능하다고 설명한다.

즉 전체 흐름은 다음처럼 이해할 수 있다.

1. SOLOv2가 instance-aware mask map을 예측한다.
2. 이 mask map을 level set function으로 간주한다.
3. input image와 deep feature를 함께 사용해 Chan-Vese형 energy를 계산한다.
4. box projection으로 초기 경계를 정한다.
5. box 내부에서 반복적으로 level set을 업데이트한다.
6. 최종적으로 refined instance mask를 얻는다.

### 3.2 왜 level set인가

논문은 기존 affinity 기반 방법이 local pairwise relation에 너무 크게 의존한다고 본다. 반면 level set은 **implicit curve를 energy function으로 표현하고 gradient descent로 반복 최적화**하는 방식이기 때문에, 객체-배경 경계 전체를 보다 구조적으로 맞출 수 있다. 특히 이 논문은 Chan-Vese functional을 채택하는데, 이는 object 안팎의 region statistics를 활용해 contour를 진화시키는 region-based level set 방법이다. 따라서 단순 edge cue보다 더 안정적으로 동작할 수 있다.

저자들은 또한 기존 deep level set 계열인 Levelset R-CNN, DVIS 등이 대부분 **mask supervision이 있는 fully supervised setting**이었다는 점을 지적한다. 반면 이 논문은 bounding box만으로 같은 철학을 weak supervision에 옮긴다. 이 차이가 논문의 포지셔닝에서 중요하다.

### 3.3 Chan-Vese energy의 역할

업로드된 ar5iv HTML에서 수식 렌더링은 다소 복잡하지만, 논리 구조는 비교적 분명하다. Chan-Vese형 energy는 크게 다음 요소들로 읽을 수 있다.

* 객체 내부와 외부 영역이 각각 일정한 통계적 성질을 갖도록 하는 **data fitting term**
* contour가 지나치게 복잡해지지 않도록 하는 **regularization term**
* box 제약 안에서 진화하도록 하는 **local optimization structure**

논문 설명에 따르면 첫 두 항은 예측된 $\phi(x,y)$가 객체 안팎의 uniformity를 따르도록 강제하고, contour length를 조절하는 비음수 regularization weight $\gamma$가 추가된다. 실험에서는 이 $\gamma$를 $10^{-4}$로 둔다고 설명한다. 즉 방법의 본질은 “마스크를 바로 supervise하는 것”이 아니라, **예측된 level set이 region-based variational principle을 만족하도록 학습하는 것**이다.  

### 3.4 Box projection initialization

weak supervision에서 초기화는 매우 중요하다. 논문은 **box projection function**을 사용해 각 step에서 초기 level set $\phi_0$를 자동 생성한다. 이 초기화 덕분에 level set function이 bounding box 내부에서 경계를 향해 더 안정적으로 수렴한다. ablation에서도 단순 box projection만 넣은 경우와 energy term을 추가한 경우를 비교하며, 초기화가 실제로 학습에 의미 있는 도움을 준다고 해석한다.  

이 부분은 기존 pseudo-mask 생성 방식과 다르다. 기존에는 box로부터 mask proxy를 직접 만들거나 외부 module로 보강하는 경우가 많았는데, 이 논문은 box를 **초기 boundary generator**로 활용하고 이후 refinement는 energy minimization에 맡긴다.

### 3.5 Image + deep feature 결합

저자들은 input data term으로 원본 이미지 $I_{img}$와 deep feature $I_{feat}$를 함께 사용한다. 원본 이미지는 low-level appearance cue를 주고, deep feature는 long-range dependency와 구조 정보를 담는다. 논문은 이 deep structural feature가 box 내부에서 곡선이 더 robust하게 객체 경계로 이동하도록 돕는다고 설명한다. 특히 tree filter를 통해 long-range dependency를 보존한 high-level deep feature를 사용하는 것이 중요하다고 본다.  

Ablation에서도 원본 이미지 하나만 넣은 경우보다 deep feature를 넣은 경우가 더 낫고, tree filter를 적용한 deep feature는 추가로 **+1.9% AP** 개선을 만든다고 보고한다. 이 결과는 이 논문의 level set이 단순 classical variational model이 아니라, **deep representation에 의해 구조적으로 강화된 variational model**임을 보여준다.  

### 3.6 End-to-end single-shot learning

논문은 자신들의 방법이 pseudo-mask 생성용 별도 네트워크나 multi-stage pipeline을 쓰지 않는 **single-shot, end-to-end** 구조라고 강조한다. 이는 BBAM, DiscoBox 등 proxy mask 생성 기반 방법과 대비된다. 다시 말해, 이 논문은 weak supervision을 “좋은 pseudo label 만들기”로 보지 않고, **energy-driven boundary refinement process를 직접 학습하는 문제**로 본다. 이 인식 전환이 핵심이다.

## 4. Experiments and Findings

### 4.1 실험 설정과 비교 범위

논문은 네 가지 데이터셋에서 실험한다.

* **COCO**
* **Pascal VOC**
* **iSAID** (remote sensing)
* **LiTS** (medical)

즉 일반 장면, 원격탐사, 의료영상처럼 서로 다른 난이도의 시나리오를 모두 포함한다. 이는 방법이 단순 general-scene benchmark에만 맞춘 것이 아니라, 배경/객체 관계가 까다로운 도메인으로도 일반화되는지를 보이려는 의도다.

### 4.2 Pascal VOC와 COCO에서의 결과

논문은 Pascal VOC에서 box-supervised 기존 방법들보다 더 좋은 성능을 보였다고 보고한다. 구체적으로 Pascal VOC에서 **BoxInst 대비 ResNet-50 기준 2.0% AP, ResNet-101 기준 1.8% AP** 향상이 있다고 설명한다. 또한 COCO와 Pascal VOC 모두에서 SOTA를 달성했다고 주장한다.  

저자들은 이 결과를 통해 mask-supervised와 box-supervised 사이의 성능 격차를 좁혔다고 해석한다. 특히 fully supervised variational method와 비교했을 때도 성능이 꽤 근접하며, 일부는 능가한다고 말한다.

### 4.3 Fully supervised variational methods와의 비교

흥미로운 점은 이 논문이 box-supervised 방법만 비교한 것이 아니라, **DeepSnake**, **Levelset R-CNN**, **DVIS-700** 같은 fully supervised variational-based instance segmentation과도 비교했다는 점이다. 논문은 Table 3 기준으로 자사 방법이 fully supervised variational methods와 comparable하며, **DeepSnake와 Levelset R-CNN은 오히려 능가**한다고 설명한다. 이는 weak supervision만 사용하고도 variational segmentation 계열에서 상당히 강한 결과를 낸다는 의미다.  

이 비교는 논문의 주장을 강화한다. 즉 box-supervision이라는 제약이 있어도, boundary modeling이 충분히 강하면 단순 fully supervised baseline 못지않은 결과를 얻을 수 있다는 것이다.

### 4.4 iSAID와 LiTS에서의 강점

이 논문의 실험에서 가장 설득력 있는 부분은 remote sensing과 medical image 결과다. iSAID는 같은 클래스 객체가 매우 밀집되어 있고, LiTS는 foreground-background가 매우 유사하다. 저자들은 이런 환경에서 pixel-pair relation 기반 box-supervised 방법은 noisy context에 취약한 반면, 자신들의 level set-driven curve fitting은 더 robust하다고 해석한다. 실제로 **iSAID에서 BoxInst보다 2.3% AP, LiTS에서 3.8% AP** 더 높다고 보고한다.

LiTS 결과 표 snippet에서는 자사 방법이 **44.5 AP / 78.6 AP50 / 45.6 AP75**를 기록한 것으로 보인다. 이 수치는 medical setting에서 box supervision만으로 꽤 강한 boundary quality를 얻었음을 보여준다.

### 4.5 Ablation study

Ablation은 Pascal VOC에서 수행된다. 핵심 관찰은 다음과 같다.

* **Box projection initialization만** 사용해도 일정 성능이 나오며, 초기 경계 생성이 실제로 효과적이다.
* 원본 이미지 $I_u$를 data term으로 쓰면 **22.2% AP**를 얻는다.
* deep high-level feature $I_f$를 사용하면 **24.7% AP**로 더 좋아진다.
* tree filter를 통한 structural feature 사용은 추가로 **+1.9% AP** 개선을 준다.
* 더 긴 학습 스케줄은 **24.7% → 34.4%**처럼 큰 향상을 보이며, 이는 level set evolution이 충분한 수렴 시간을 필요로 한다는 점을 시사한다.  

이 ablation은 꽤 중요하다. 논문의 성능 향상이 단순 backbone 차이가 아니라,

* initialization,
* input data term 설계,
* deep structural feature,
* optimization time

의 결합 효과라는 것을 보여주기 때문이다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제 재정의가 좋다**는 점이다. 기존 box-supervised segmentation이 pseudo-mask 생성 또는 pairwise affinity에 집중했다면, 이 논문은 이를 **energy minimization 기반 boundary evolution** 문제로 다시 본다. 이 관점 덕분에 weak supervision에서도 경계 정렬(boundary alignment)을 보다 직접적으로 다룰 수 있다.

두 번째 강점은 classical vision과 deep learning의 결합이 자연스럽다는 점이다. Chan-Vese level set이라는 고전적 variational model을 SOLOv2, deep structural feature, differentiable training과 결합해 현대적인 단일 파이프라인으로 재구성했다. 이는 단순히 “옛 기법을 붙였다”가 아니라, 고전적 inductive bias를 modern deep segmentation에 구조적으로 통합한 사례다.  

세 번째 강점은 일반 장면뿐 아니라 remote sensing과 medical image에서 더 큰 이점을 보인다는 점이다. 이는 경계 ambiguity와 noisy context가 심한 문제일수록 이 방식이 더 유리하다는 해석을 가능하게 한다.

### 한계

첫째, ar5iv HTML 기준으로 수식 렌더링이 상당히 복잡해 exact energy formulation과 update rule을 구현 수준으로 추적하기는 쉽지 않다. 논리 구조는 분명하지만, coefficient 설정이나 최적화 식을 완전히 재현하려면 원문 PDF나 코드 저장소를 함께 보는 편이 좋다. 이는 답변의 정보 한계이기도 하다.

둘째, 논문 자체가 level set evolution을 반복적으로 수행하므로, 직관적으로는 일반적인 direct mask supervision보다 학습 수렴 시간이 더 민감할 수 있다. 실제 ablation에서도 longer schedule이 큰 차이를 만든다. 이는 장점이자 실무적 부담이 될 수 있다.

셋째, 방법이 SOLOv2 기반이므로 성능의 일부는 base architecture 선택에도 의존한다. 즉 이 논문의 핵심 기여는 level set formulation이지만, 실제 성능은 strong mask-supervised backbone adaptation과 함께 봐야 한다.

### 해석

비판적으로 보면, 이 논문의 진짜 기여는 “box-supervised SOTA 하나”보다 더 크다. 더 본질적으로는 **weak supervision에서 explicit pseudo mask 대신 implicit curve evolution을 학습할 수 있다**는 점을 보여줬다. 이 철학은 향후 box-supervised segmentation뿐 아니라 scribble/point supervision, medical contour extraction, remote sensing boundary refinement 같은 문제로도 확장될 수 있다. 다만 이 확장은 논문이 직접 실험한 범위를 넘는 해석이므로 가능성 수준으로 보는 것이 적절하다.

## 6. Conclusion

이 논문은 box-supervised instance segmentation에서 기존 pseudo-mask 생성이나 pairwise affinity 기반 접근의 한계를 지적하고, 이를 해결하기 위해 **Level Set Evolution 기반 단일 단계 end-to-end 프레임워크**를 제안했다. SOLOv2가 예측한 mask map을 level set function으로 해석하고, **Chan-Vese energy**, **box projection initialization**, **input image + deep structural feature**를 결합해 bounding box 내부에서 경계를 점진적으로 정교화한다. 실험적으로는 COCO, Pascal VOC, iSAID, LiTS에서 강한 성능을 보였고, 특히 noisy context가 심한 remote sensing과 medical image에서 큰 개선을 나타냈다.  

실무적으로 이 논문은 “weak supervision에서는 proxy label이 꼭 필요하다”는 통념에 대한 좋은 반례다. 경계 진화 자체를 differentiable하게 학습하는 방식이 충분히 강력할 수 있음을 보여주며, 고전 variational segmentation과 modern instance segmentation의 결합 가능성을 잘 드러낸다.
