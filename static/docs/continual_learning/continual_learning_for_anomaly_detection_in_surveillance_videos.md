# Continual Learning for Anomaly Detection in Surveillance Videos

* **저자**: Keval Doshi, Yasin Yilmaz
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2004.07941](https://arxiv.org/abs/2004.07941)

## 1. 논문 개요

이 논문은 **감시 비디오(surveillance video)에서의 anomaly detection** 문제를 **continual learning** 관점에서 다룬다. 기존의 비디오 이상 탐지 연구들은 주로 고정된 데이터셋에서 높은 성능을 내는 데 집중해 왔고, 대체로 대규모 딥러닝 모델을 학습한 뒤 테스트하는 방식이다. 그러나 실제 감시 환경에서는 정상 패턴과 이상 패턴의 정의가 시간에 따라 바뀌고, 새로운 정상 행동이 계속 나타난다. 따라서 한 번 학습한 모델을 그대로 유지하는 방식은 현실적인 배치 환경에서는 한계가 있다.

이 논문이 해결하려는 핵심 문제는 두 가지다. 첫째, **새로운 정상 패턴을 소량의 최근 데이터로 빠르게 반영할 수 있는가**이다. 둘째, **실시간 온라인 의사결정**이 가능한가이다. 저자들은 기존 state-of-the-art 비디오 이상 탐지 방법들이 정적 데이터셋에서는 잘 작동해도 continual learning 환경에서는 재학습 비용과 저장 비용 때문에 실용적이지 않다고 지적한다. 특히 딥러닝 기반 결정 규칙은 새 데이터가 들어올 때마다 전체 데이터를 다시 학습해야 하거나, 그렇지 않으면 catastrophic forgetting 문제가 생긴다.

이를 해결하기 위해 저자들은 **transfer learning 기반 feature extraction**과 **statistical sequential anomaly detection**을 결합한 하이브리드 프레임워크를 제안한다. 즉, 영상에서 motion, location, appearance 특징은 사전학습된 딥러닝 모듈이 추출하고, 최종 이상 판단과 온라인 업데이트는 kNN 기반 통계적 검출기가 담당한다. 이 설계의 목적은 딥러닝의 표현력은 활용하면서도, 새 데이터가 들어왔을 때 전체 네트워크를 다시 학습하지 않고도 지속적으로 적응하는 것이다.

문제의 중요성은 매우 크다. 실제 감시 시스템은 방대한 스트리밍 데이터를 생성하며, 이벤트가 발생했을 때 빠른 탐지와 즉각적 대응이 필요하다. 또한 “자전거를 타는 사람”처럼 어떤 장면에서는 이상이지만 다른 장면에서는 정상인 경우도 많아, 이상 여부는 환경과 문맥에 따라 달라진다. 따라서 실용적인 시스템은 단순한 오프라인 분류기가 아니라, **시간이 지나며 정상의 정의를 계속 업데이트할 수 있는 온라인 시스템**이어야 한다. 이 논문은 հենց 그 지점을 겨냥한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 **이상 탐지의 feature learning과 decision making을 분리**하는 데 있다. 기존 많은 방법은 anomaly score를 딥러닝 모델 내부에서 직접 학습하도록 설계한다. 반면 이 논문은 딥러닝을 “좋은 특징을 추출하는 도구”로만 쓰고, 실제 이상 판단은 통계적 순차 검출기가 담당하게 만든다. 이렇게 하면 딥러닝 모델을 새 데이터마다 다시 학습할 필요가 없고, 최근 정상 샘플을 decision set에 추가하는 방식으로 continual learning을 구현할 수 있다.

구체적으로 저자들은 anomaly를 세 종류의 정보로 포착하려 한다. 첫째는 **motion**이다. 예를 들어 자전거는 보행자보다 훨씬 빠르게 움직이므로 optical flow 분포가 달라질 수 있다. 둘째는 **location**이다. 같은 사람이라도 도로 한가운데를 걷는 것은 이상할 수 있다. 셋째는 **appearance**이다. 예를 들어 총을 든 사람이나 이전에 보지 못한 객체 클래스는 외형 측면에서 이상일 수 있다. 저자들은 기존 연구가 optical flow만 보거나 appearance만 보는 식으로 특정 측면에 치우친 경우가 많다고 비판하고, 이 세 정보를 하나의 feature vector로 결합한다.

또 하나의 핵심 직관은, 비디오 이상은 대개 **순간적인 single-shot event라기보다 일정 시간 지속되는 변화**라는 점이다. 그래서 한 프레임의 점수만 thresholding하는 방식보다, 시간에 따라 anomaly evidence를 누적하는 **CUSUM-like sequential statistic**이 더 적절하다고 본다. 즉, 현재 프레임 하나만 보고 판단하지 않고, 최근 프레임들에서 이상 근거가 연속적으로 쌓이는지를 본다. 이는 false alarm을 줄이고 detection delay를 통제하는 데 유리하다.

기존 접근과의 가장 큰 차별점은, 이 방법이 **딥러닝 모델 자체의 continual learning**을 시도하지 않는다는 것이다. 대신 사전학습된 YOLO와 FlowNet2를 feature extractor로 사용하고, continual update는 kNN 기반 nominal set에 새 정상 샘플을 추가하는 방식으로 한다. 즉, “representation은 transfer learning으로 가져오고, continual adaptation은 통계적 메모리 업데이트로 해결한다”는 구조다. 이 점이 논문의 가장 명확한 설계 철학이다.

## 3. 상세 방법 설명

### 전체 시스템 구조

논문의 프레임워크는 크게 두 부분으로 구성된다. 첫 번째는 **neural network-based feature extraction module**이고, 두 번째는 **statistical anomaly detection module**이다. 매 시각 $t$에서 입력 프레임 $X_t$가 들어오면, feature extraction 모듈은 각 객체에 대해 motion, location, appearance 정보를 뽑아 feature vector를 만든다. 이후 anomaly detection 모듈은 이 벡터들이 nominal data manifold에서 얼마나 벗어나는지를 계산하고, 순차적으로 anomaly evidence를 누적해 이상 여부를 결정한다.

이 구조의 장점은 분명하다. feature extractor는 사전학습된 모델을 그대로 활용하므로 학습 비용이 적다. 반면, decision module은 통계적 규칙에 따라 업데이트되므로 새 정상 샘플이 들어왔을 때 빠르게 적응할 수 있다. 즉, representation learning과 continual update를 분리함으로써 현실적인 online system을 구현한다.

### 특징 선택과 feature vector 구성

저자들은 이상 사건이 appearance, motion, location 중 어느 측면에서 나타날지 사전에 알 수 없다고 본다. 그래서 각 객체 $i$에 대해 시각 $t$에서 다음과 같은 형태의 feature vector를 만든다.

$F_t^i = [w_1 F_{motion},; w_2 F_{location},; w_3 F_{appearance}]$

여기서 $w_1, w_2, w_3$는 각 feature group의 상대적 중요도를 조절하는 가중치다. 이 식을 더 구체적으로 풀면 최종 feature vector는 다음과 같다.

$$
F_t^i=
\begin{bmatrix}
w_1\text{Mean}\\\\
w_1\text{Variance}\\\\
w_1\text{Skewness}\\\\
w_1\text{Kurtosis}\\\\
w_2 C_x\\\\
w_2 C_y\\\\
w_2 Area\\\\
w_3 p(C_1)\\\\
w_3 p(C_2)\\\\
\vdots\\\\
w_3 p(C_n)
\end{bmatrix}
$$

여기서 Mean, Variance, Skewness, Kurtosis는 optical flow에서 얻은 통계량이고, $C_x, C_y, Area$는 bounding box의 중심 좌표와 면적이다. $p(C_1), \dots, p(C_n)$은 object detector가 출력한 클래스 확률이다. 따라서 클래스 수가 $n$이면 전체 feature 차원은 $m = n+7$이 된다.

이 설계는 비교적 단순하지만 해석 가능성이 높다. 예를 들어 optical flow의 왜도(skewness)와 첨도(kurtosis)가 달라지면 움직임의 분포가 바뀌었다는 의미이고, 중심 위치나 bounding box area가 바뀌면 객체의 위치나 크기 패턴이 달라졌다는 의미다. appearance 확률은 이전에 보지 못한 객체나 비정상 클래스 조합을 감지하는 데 쓰인다.

### Transfer Learning 모듈

#### Object Detection

location과 appearance feature는 사전학습된 object detector에서 가져온다. 논문은 대표적으로 **YOLO**를 사용한다. 이유는 속도와 정확도의 균형 때문이다. 온라인 anomaly detection에서는 fps가 중요하므로, SSD나 ResNet 계열보다 빠른 YOLOv3를 선호한다고 설명한다.

YOLO는 각 프레임에서 객체를 검출하고 bounding box와 클래스 확률을 반환한다. 저자들은 bounding box 전체를 그대로 쓰지 않고, **박스 중심 좌표와 면적**을 location feature로 사용한다. 이 정보만으로도 “사람이 있어야 할 위치가 아닌 곳에 등장했는가”와 같은 location anomaly를 잡으려는 것이다.

#### Optical Flow

motion 정보는 사전학습된 optical flow 모델인 **FlowNet 2**에서 추출한다. 저자들의 가정은 명확하다. 어떤 motion anomaly가 발생하면 optical flow 분포 자체가 변할 것이므로, flow map 전체를 쓰기보다는 그 분포를 요약하는 통계량을 쓰는 것이 효과적일 수 있다는 것이다.

그래서 optical flow로부터 평균, 분산, 왜도, 첨도를 뽑는다. 평균과 분산은 움직임의 크기와 퍼짐 정도를 반영하고, 왜도와 첨도는 분포의 비대칭성과 뾰족함을 나타낸다. 예를 들어 보행자만 있는 장면에 자전거가 들어오면 속도 분포가 바뀌고, 이 변화가 특히 skewness와 kurtosis에서 잘 드러날 수 있다고 논문은 실험적으로 보인다.

### Statistical Sequential Anomaly Detection

이 논문의 중심은 사실 이 부분이다. 딥러닝은 feature만 만들고, 실제 anomaly decision은 **nonparametric sequential detector**가 담당한다.

#### 훈련 단계

훈련 데이터는 anomaly가 없는 nominal videos만으로 구성된다고 가정한다. 모든 훈련 비디오에서 객체별 feature vector를 추출하여 전체 nominal feature set $\mathcal{F}^M = {F^i}$를 만든다. 총 $M$개의 객체 feature가 있다고 하자.

이때 목표는 nominal data distribution을 모수적으로 모델링하는 것이 아니라, **k-nearest neighbor distance**를 이용해 nominal manifold를 기술하는 것이다. 저자들의 핵심 가정은 anomaly feature는 nominal manifold에서 멀리 떨어질 가능성이 높다는 것이다.

훈련 절차는 다음과 같다.

첫째, nominal feature set을 두 부분 $\mathcal{F}^{M_1}$과 $\mathcal{F}^{M_2}$로 무작위 분할한다.
둘째, $\mathcal{F}^{M_1}$의 각 점 $F^i$에 대해 $\mathcal{F}^{M_2}$를 기준으로 kNN distance $d_i$를 계산한다.
셋째, significance level $\alpha$에 대해 이 거리들의 $(1-\alpha)$ percentile인 $d_\alpha$를 baseline statistic으로 정한다.

이 $d_\alpha$는 나중에 test point가 nominal 범위 안에 있는지 판단하는 기준 역할을 한다.

#### 테스트 단계

테스트 시점에서 시각 $t$에 프레임이 들어오면, 각 객체 feature $F_t^i$에 대해 training nominal set $\mathcal{F}^{M_2}$와의 kNN distance $d_t^i$를 계산한다. 그리고 프레임 수준 anomaly evidence를 다음과 같이 정의한다.

$$
\delta_t = \left(\max_i {d_t^i}\right)^m - d_\alpha^m
$$

여기서 $m$은 feature vector 차원이다. 이 식은 해당 프레임에서 가장 이상한 객체 하나를 대표 evidence로 사용한다는 의미다. 만약 어떤 객체 하나라도 nominal manifold에서 크게 벗어나면 프레임 전체를 suspicious하게 보는 셈이다.

이후 CUSUM과 유사한 방식으로 running statistic을 업데이트한다.

$$
s_t = \max{s_{t-1} + \delta_t,\\; 0}, \qquad s_0=0
$$

이 식의 의미는 직관적이다. nominal frame이면 보통 $\delta_t$가 음수라서 $s_t$가 다시 0 근처로 내려간다. 반면 anomalous frame이 연속되면 $\delta_t$가 양수가 되어 $s_t$가 점점 누적 상승한다. 최종적으로 $s_t > h$가 되면 threshold $h$를 넘었다고 보고 anomaly alarm을 낸다.

논문은 detection 이후 후처리도 제안한다. $s_t$가 상승하기 시작한 마지막 시점 $\tau_{start}$와, 이후 감소가 일정 프레임 수 동안 지속되는 시점 $\tau_{end}$를 찾아 그 구간 전체를 anomalous frames로 라벨링한다. 그리고 다음 anomaly를 찾기 위해 $s_{\tau_{end}}=0$으로 reset한다. 이는 frame-level segmentation을 좀 더 자연스럽게 하기 위한 장치다.

### Continual Learning 절차

continual learning은 surprisingly simple하게 구현된다. 테스트 중에 어떤 시점 $t$에서 $s_t=0$이면, 즉 현재 feature가 nominal하다고 판단되면 그 feature vector를 nominal training set $\mathcal{F}^{M_2}$에 추가한다. 즉, **모델이 정상이라고 판단한 최근 데이터는 바로 정상 기준 집합에 포함**된다.

반면 $s_t$가 threshold $h$를 넘어 alarm이 발생하면, 해당 구간은 “훈련 시 보지 못한 패턴”으로 간주된다. 이때 논문은 **human-in-the-loop**를 가정한다. 사람이 이 알람을 검토해 false alarm이라고 판단하면, $\tau_{start}$부터 $t$까지의 feature vectors를 nominal set에 추가한다. 이렇게 하면 이후 유사한 이벤트가 다시 발생해도 더 이상 false alarm이 나지 않게 된다.

이 방식의 장점은, 딥러닝 detector처럼 전체 네트워크를 retraining하지 않고도 최근 정상 패턴을 즉시 반영할 수 있다는 점이다. 즉, continual learning이 “parameter update”가 아니라 **reference set update**로 구현된다.

### 계산 복잡도

논문은 sequential anomaly detector의 시간·공간 복잡도를 명시한다. 훈련 단계는 $\mathcal{F}^{M_1}$의 각 샘플에 대해 $\mathcal{F}^{M_2}$와의 kNN 거리를 계산하므로 시간 복잡도는 $\mathcal{O}(M_1 M_2 m)$이다. 저장해야 하는 것은 $\mathcal{F}^{M_2}$뿐이라 공간 복잡도는 $\mathcal{O}(M_2 m)$이다.

테스트 시에는 새 점 하나와 $\mathcal{F}^{M_2}$ 전체 사이의 거리를 계산하므로 시간 복잡도는 $\mathcal{O}(M_2 m)$이다. 여기서 중요한 비교는, 딥러닝 기반 continual update는 보통 고해상도 원본 비디오를 다시 저장하고 재학습해야 하므로 공간 복잡도가 대략 $\mathcal{O}(ab M_2)$라고 논문이 설명한다. 여기서 $a \times b$는 영상 해상도다. 보통 $ab \gg m$이므로 제안법이 훨씬 가볍다는 주장이다.

실행 속도 측면에서도 YOLO는 약 12 ms per image, FlowNet2는 약 40 fps이고, 전체 시스템은 약 32 fps라고 보고한다. 이는 실시간 surveillance stream 처리 가능성을 보여주기 위한 결과다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

논문은 세 개의 공개 benchmark dataset에서 먼저 평가한다. **UCSD Ped2**, **CUHK Avenue**, **ShanghaiTech Campus**가 사용된다. 세 데이터셋 모두 training set은 nominal events만 포함한다.

UCSD Ped2는 보행자 구역에서 자전거, 스케이트보드, 휠체어 등이 등장하는 경우를 anomaly로 본다.
Avenue는 running, loitering, object throwing 같은 행동 이상이 포함된다.
ShanghaiTech은 13개 scene, 330 train videos, 107 test videos로 구성된 더 크고 어려운 데이터셋이다.

비교 방법은 MPPCA, Conv-AE, ConvLSTM-AE, Stacked RNN, GAN 계열, Liu et al., Sultani et al. 등 다양한 hand-crafted / deep learning 계열 state-of-the-art를 포함한다. 평가지표는 기존 anomaly detection 문헌을 따라 **frame-level AUC**를 사용한다.

### 일반 anomaly detection 성능

Table 1 결과에 따르면 제안법은 Avenue에서 **86.4**, UCSD Ped2에서 **97.8**, ShanghaiTech에서 **71.62**의 frame-level AUC를 얻는다. 이는 Avenue와 UCSD Ped2에서는 기존 방법들보다 높은 수치이며, ShanghaiTech에서는 competitive한 수준이다.

구체적으로 Avenue에서는 Liu et al.의 85.1보다 높고, UCSD Ped2에서는 Liu et al.의 95.4보다 높다. ShanghaiTech에서는 Liu et al.의 72.8, Sultani et al.의 71.5와 비교할 때 약간 낮거나 비슷한 수준이다. 저자들은 특히 ShanghaiTech 결과가 **future frames를 보지 않는 온라인 방식**으로 나온 점을 강조한다. 반면 일부 기존 방법은 test video별 normalization에 미래 프레임 정보를 사용하므로, 엄밀한 실시간 설정과는 다르다고 비판한다.

즉, 이 논문은 단순히 continual learning만 잘하는 것이 아니라, 일반적인 anomaly detection benchmark에서도 충분히 경쟁력 있는 성능을 낸다고 주장한다.

### Sequential anomaly detection의 효과

저자들은 제안한 sequential statistic의 중요성을 보이기 위해, $\delta_t$ 자체에 threshold를 걸어 즉시 판단하는 nonsequential version과 비교한다. Figure 3에 따르면, instantaneous anomaly evidence에만 의존하면 noise에 매우 민감해 false alarm이 많이 난다. 반면 제안한 $s_t$는 최근 evidence를 누적하므로, 일시적 잡음은 상쇄되고 지속적인 이상만 크게 반영된다.

이 결과는 논문의 문제 정의와 잘 맞는다. 감시 비디오의 이상은 보통 한 프레임짜리 충격이 아니라 몇 초 동안 이어지는 사건이므로, single-shot detection보다 sequential detection이 더 적합하다는 주장이다.

### Optical flow의 효과

Figure 4는 UCSD 첫 번째 테스트 비디오에서 optical flow 통계가 어떻게 변하는지를 보여준다. 이 예시는 보행자 구역에 자전거가 등장하는 anomaly다. 논문은 특히 skewness와 kurtosis 변화가 두드러진다고 설명한다. 이는 자전거가 보행자보다 빠르게 움직여 flow distribution shape 자체를 바꾸기 때문이다.

이 실험은 motion feature가 단순 평균 속도만 보는 것이 아니라, 분포의 고차 통계량까지 포함해야 하는 이유를 뒷받침한다. 즉, 논문이 optical flow를 raw feature 대신 통계적으로 요약한 설계가 단순 heuristic만은 아니라는 점을 보여준다.

### Continual learning 성능

이 논문의 가장 핵심적인 실험은 여기다. 기존 benchmark에는 video surveillance continual learning용 표준 데이터셋이 없기 때문에, 저자들은 UCSD 설정을 조금 바꿔서 “자전거를 anomaly가 아니라 nominal behavior로 인정해야 하는 상황”을 만든다. 즉, 테스트 중 새 정상 패턴이 나타났을 때 시스템이 얼마나 빨리 적응하는지를 본다.

처음에는 제안법도 자전거를 anomaly로 탐지한다. 하지만 human supervision으로 해당 구간을 nominal로 라벨링하고 training nominal set에 추가하면, 이후에는 자전거를 정상으로 받아들이도록 업데이트된다. Figure 5에 따르면 이 과정에서 제안법은 Liu et al. [23]와 Ionescu et al. [16]보다 continual learning 성능이 훨씬 좋다.

특히 Table 2가 매우 인상적이다. 제안법의 update time은 **10초**인데, Liu et al.은 **4.8시간**, Ionescu et al.은 **2.5시간**이 걸린다. 이는 제안법이 full retraining 없이 새 nominal sample만 추가해 업데이트하기 때문이다. 저자들은 또한 소수 샘플만으로도 비교적 높은 AUC를 얻는다고 하여 **few-shot learning ability**도 주장한다.

이 결과는 논문의 주장과 정확히 맞물린다. 즉, 기존 state-of-the-art video anomaly detector는 성능은 좋더라도 continual update 측면에서는 사실상 비실용적이고, 제안법은 정확도뿐 아니라 update latency 측면에서도 훨씬 현실적이라는 것이다.

### 실제 CCTV 스트림 실험

논문은 공개 benchmark 외에도 8시간 23분 길이의 실제 CCTV street feed에서 실험한다. 여기서는 처음 10분 데이터만으로 훈련하고, 이후 streaming data가 들어오면서 모델을 계속 업데이트한다. 이 실험은 정확도 자체보다는 **지속적 적응 능력과 false alarm 감소**를 보는 데 초점이 있다.

Figure 6은 false alarm의 원인을 보여준다. 예를 들어 도로 중앙에 사람이 오래 서 있는 경우, 바람이나 날씨 변화로 표지판이 움직이는 경우, 여러 대의 차가 동시에 지나가 optical flow 분포가 바뀌는 경우, 처음 등장한 자전거 등이 모두 false alarm 원인이 된다. 이 예시들은 실제 환경에서는 anomaly와 novel nominal의 경계가 모호하다는 사실을 잘 보여준다.

Figure 7에서는 각 20,000프레임 구간 뒤에 사람이 false positive 일부를 nominal로 라벨링해 업데이트할 때, false alarm 수가 점진적으로 감소하는 모습을 보여준다. 저자들은 업데이트에 전체 false positive가 아니라 20%만 사용해도 효과가 있다고 말한다. 그리고 이 양이 프레임 수로는 커 보여도 시간으로는 약 10초 분량이라 few-shot으로 볼 수 있다고 주장한다.

이 실험의 의미는 명확하다. 공개 benchmark처럼 training nominal set이 완전하지 않은 실제 환경에서도, 제안법은 최근 정상 패턴을 빠르게 반영해 false alarm rate를 줄일 수 있다. 이는 논문의 practical relevance를 강화한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정이 매우 현실적**이라는 점이다. 비디오 anomaly detection 연구는 종종 정적 데이터셋에서 AUC 경쟁으로 흘러가는데, 이 논문은 실제 감시 시스템에서 더 중요한 online detection, continual adaptation, update latency를 전면에 내세운다. 특히 “기존 이상 패턴만 잘 찾는가”보다 “새로운 정상 패턴을 빠르게 학습할 수 있는가”를 묻는 점이 실용적이다.

둘째 강점은 **설계가 단순하고 해석 가능하다**는 것이다. YOLO와 FlowNet2로 feature를 뽑고, kNN distance와 CUSUM-like statistic으로 판단한다는 구조는 비교적 명확하다. 왜 anomaly가 발생했는지도 motion/location/appearance 중 어느 요소 때문인지 어느 정도 해석이 가능하다. 이는 end-to-end deep model보다 analyzability가 높다.

셋째, **continual learning 비용이 매우 낮다**는 점이 강점이다. 새 샘플이 들어올 때 딥러닝 모델을 다시 학습하지 않고 nominal set만 갱신하므로 update time이 짧다. Table 2의 10초 대 2.5~4.8시간 비교는 이 장점을 아주 잘 보여 준다.

넷째, **실시간 처리 가능성**도 장점이다. 전체 파이프라인이 약 32 fps 수준으로 동작한다고 보고하며, 이는 surveillance stream의 online use case에 적합하다.

하지만 한계도 분명하다. 첫째, 이 논문에서 말하는 continual learning은 **모델 파라미터를 진짜로 continual optimization하는 형태가 아니라, reference nominal set을 업데이트하는 형태**다. 따라서 일반적인 deep continual learning 문헌에서 말하는 representation adaptation과는 결이 다르다. 장점이기도 하지만, “딥러닝 기반 continual learning을 해결했다”고 보기는 어렵다.

둘째, 방법의 성능은 **사전학습된 feature extractor의 품질에 크게 의존**한다. YOLO가 잘 못 잡는 객체, FlowNet2가 잘 표현하지 못하는 motion, 혹은 object detector의 class vocabulary 밖에 있는 appearance anomaly에는 한계가 있을 수 있다. 예를 들어 미세한 행위 이상이나 복잡한 상호작용은 이 feature set만으로 충분히 표현되지 않을 가능성이 있다.

셋째, 제안법은 anomaly를 kNN distance 기반 outlier로 본다. 이는 해석 가능하고 단순하지만, **고차원 feature space에서 density structure가 복잡할 때 nearest neighbor distance가 항상 최선이라고 보기 어렵다**. feature weighting $w_1, w_2, w_3$의 선택도 성능에 영향을 줄 수 있지만, 본문 텍스트에서는 이 가중치를 어떻게 정했는지 충분히 자세히 설명하지 않는다.

넷째, continual update는 human-in-the-loop에 의존한다. 즉, alarm이 false alarm인지 아닌지를 사람이 간헐적으로 확인해 주어야 한다. 실제 감시 시스템에서는 합리적인 가정일 수 있지만, **완전 자동 continual adaptation은 아니다**. 또한 사람이 잘못 라벨링하면 nominal set이 오염될 위험도 있다.

다섯째, benchmark comparison에서 저자들은 기존 방법들의 online unfairness를 지적하지만, 반대로 제안법의 comparison도 완전히 동일 조건이라고 보기는 어렵다. 일부 방법은 원래 continual learning을 위해 설계되지 않았고, 일부는 future normalization을 쓰므로 직접 비교에는 설정 차이가 존재한다. 저자도 이 점을 어느 정도 인정하고 있다.

종합적으로 보면, 이 논문은 end-to-end deep anomaly detector보다 **현실적인 운영 관점의 장점**이 크다. 그러나 feature representation의 한계와 human supervision 의존성은 앞으로 보완될 필요가 있다.

## 6. 결론

이 논문은 감시 비디오 이상 탐지에서 **transfer learning 기반 feature extraction**과 **statistical sequential anomaly detection**을 결합한 continual learning 프레임워크를 제안한다. 핵심은 motion, location, appearance 특징을 사전학습된 모델에서 효율적으로 뽑고, kNN distance와 CUSUM-like 누적 통계량으로 이상 여부를 온라인으로 판단하는 것이다. 이를 통해 기존 딥러닝 기반 anomaly detector가 갖는 재학습 비용, 저장 비용, catastrophic forgetting 문제를 우회한다.

실험 결과를 보면, 제안법은 일반 benchmark anomaly detection에서도 Avenue와 UCSD Ped2에서 state-of-the-art 수준 혹은 그 이상의 성능을 보였고, ShanghaiTech에서도 경쟁력 있는 결과를 냈다. 더 중요한 것은 continual learning 실험에서 새 nominal pattern을 few-shot으로 빠르게 반영할 수 있었고, update time이 수 초 수준으로 매우 짧았다는 점이다. 실제 CCTV stream에서도 false alarm을 지속적으로 줄이는 모습을 보였다.

이 연구의 실제적 의의는 매우 분명하다. 현실의 감시 환경에서는 정상/이상 개념이 고정되지 않으며, 새로운 정상 행동이나 환경 변화가 계속 발생한다. 따라서 시스템은 단순히 높은 AUC를 내는 정적 분류기가 아니라, **시간에 따라 스스로 정상 기준을 업데이트할 수 있는 온라인 시스템**이어야 한다. 이 논문은 그 방향에 대한 실용적이고 계산 효율적인 해법을 제시했다는 점에서 가치가 있다.

향후 연구로는 논문이 언급하듯 dynamic weather, rotating cameras, 더 복잡한 temporal relationship 같은 어려운 조건에 대한 확장이 중요할 것이다. 또한 현재는 새로운 nominal label을 점진적으로 반영하는 데 초점이 맞춰져 있는데, 앞으로는 **새로운 anomaly label까지 포함하는 richer continual learning framework**로 확장될 수 있다. 이런 점에서 이 논문은 surveillance anomaly detection을 “고정된 benchmark 문제”에서 “계속 변하는 운영 문제”로 옮겨 놓은 의미 있는 연구라고 볼 수 있다.
