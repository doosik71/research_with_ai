# A Survey on Anomaly Detection for Technical Systems using LSTM Networks

- **저자**: Benjamin Lindemann, Benjamin Maschler, Nada Sahlab, Michael Weyrich
- **발표연도**: 2021
- **arXiv**: <https://arxiv.org/abs/2105.13810>

## 1. 논문 개요

이 논문은 technical systems에서 발생하는 anomaly detection 문제를 다루는 survey 논문이다. 특히 일반적인 통계 기반 기법이나 정적 모델 중심의 방법이 아니라, 시간 의존성과 문맥 정보를 다룰 수 있는 LSTM(Long Short-Term Memory) 기반 접근법에 초점을 맞춘다. 저자들은 최근 anomaly detection 연구가 deep neural network로 빠르게 이동하고 있음에도, 기존 survey들이 LSTM 기반 방법을 주변적으로만 다루거나 충분히 구조화하지 못했다고 지적한다. 이 논문은 바로 그 공백을 메우기 위해, LSTM 기반 anomaly detection 방법들을 체계적으로 정리하고 비교하려는 목적을 가진다.

연구 문제는 비교적 명확하다. 기술 시스템에서 발생하는 이상은 단순한 outlier 한 점으로 끝나지 않는 경우가 많고, 시간에 따라 변화하며, 주변 맥락에 따라 정상인지 이상인지가 달라질 수 있다. 그런데 기존 PCA, SVM, k-NN, correlation analysis 같은 대표적 anomaly detection 방법은 기본적으로 정적이거나 time-invariant한 가정 위에 놓여 있어, 실제 시스템의 동적이고 시변적인 이상을 충분히 표현하지 못한다. 이를 보완하기 위해 sliding window 같은 보조 기법을 붙이더라도, 시간적 문맥 자체를 학습하는 모델은 아니라는 한계가 있다.

이 문제가 중요한 이유는 산업 설비, 제조 시스템, 차량 통신, 로봇, 네트워크 보안 등에서 anomaly가 효율 저하, 품질 문제, 안전 문제, 심지어 시스템 실패로 이어질 수 있기 때문이다. 특히 anomaly의 원인이 명확히 알려져 있지 않은 복잡한 시스템에서는, 사람이 규칙을 다 정의하기 어렵기 때문에 데이터 기반 모델이 필요하다. 저자들은 이러한 상황에서 LSTM이 short-term dependency와 long-term dependency를 모두 포착할 수 있어 contextual anomaly나 collective anomaly처럼 시간 구조를 가진 이상 탐지에 적합하다고 본다.

또한 이 논문은 단지 LSTM 논문 몇 편을 나열하는 데 그치지 않고, regular LSTM, encoder-decoder 기반 LSTM, hybrid approach로 분류하여 설명하며, 더 나아가 graph-based approach와 transfer learning도 최근 흐름으로 포함한다. 즉, 현재 가능한 방법을 정리하는 데서 멈추지 않고 앞으로 anomaly detection이 어떤 방향으로 발전할지를 함께 논의한다는 점에서 survey의 범위를 넓히고 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 anomaly detection을 단순한 정적 분류 문제가 아니라, 시간과 문맥을 포함한 동적 시스템 이해 문제로 바라봐야 한다는 것이다. 저자들은 특히 technical systems에서 anomaly가 point anomaly로만 나타나는 것이 아니라, 데이터 개별 값은 정상 범위 안에 있어도 시퀀스 전체 패턴이 이상한 collective anomaly, 혹은 주변 맥락 때문에 이상이 되는 contextual anomaly가 중요하다고 본다. 따라서 이런 이상을 탐지하려면 단순한 거리 계산이나 고정 임계치보다, 시간적 관계를 내재적으로 모델링하는 구조가 필요하다는 것이 논문의 기본 직관이다.

이 관점에서 LSTM은 중요한 역할을 한다. 일반 RNN은 vanishing gradient problem 때문에 긴 시간 간격의 의존성을 학습하기 어렵지만, LSTM은 memory cell과 gating mechanism을 통해 과거 정보를 선택적으로 유지하거나 버릴 수 있다. 그래서 시계열 안의 short-term pattern과 long-term pattern을 동시에 다루기 쉽고, 그 결과 contextual anomaly detection에 특히 유리하다. 논문은 이 점을 survey 전체의 분류 기준으로 삼는다.

기존 survey와의 차별점도 비교적 분명하다. 기존 survey들은 anomaly detection 기법을 statistical, classification-based, clustering-based, information-theoretic approach 등 넓은 범주로 다루면서 LSTM을 부차적으로만 언급하거나, deep learning 기반 방법도 Autoencoder나 일반 RNN 수준에서 넓게 다루는 경우가 많았다. 반면 이 논문은 LSTM 자체를 중심에 놓고, 어떤 architecture가 어떤 anomaly type에 강한지, 어떤 데이터 형태와 시나리오에서 사용되는지, supervised인지 unsupervised인지, adaptiveness가 있는지, 어떤 metric으로 평가되었는지를 함께 정리한다.

또 하나의 핵심 아이디어는 anomaly detection을 더 강하게 만들기 위해 LSTM 단독 모델을 넘어서야 한다는 점이다. 논문은 두 가지 미래 방향을 강조한다. 첫째는 graph-based approach이다. 현실 시스템에서는 데이터가 센서, 액추에이터, 환경 변수, 시스템 구성 요소 등 이질적인 관계를 가지므로, 이를 graph로 표현하면 문맥 정보와 관계 구조를 더 잘 담을 수 있다. 둘째는 transfer learning이다. 산업 분야에서는 대규모 labeled anomaly dataset을 수집하기 어렵기 때문에, 다른 시스템이나 태스크에서 학습한 표현을 이전해 데이터 부족 문제를 줄여야 한다. 즉, LSTM의 장점은 인정하되, 실제 산업 적용을 위해서는 graph와 transfer learning이 결합된 확장된 프레임워크가 필요하다는 것이 저자들의 판단이다.

## 3. 상세 방법 설명

이 논문은 새로운 알고리즘을 제안하는 논문이 아니라 survey이므로, 하나의 통합된 학습 알고리즘이나 단일 손실 함수를 제시하지는 않는다. 대신 anomaly type의 정의와 LSTM 계열 모델들의 구조적 차이를 설명하고, 각 접근법이 anomaly를 어떤 방식으로 검출하는지 정리한다. 따라서 여기서는 논문이 설명하는 개념적 파이프라인과 주요 모델군을 중심으로 상세히 정리하는 것이 적절하다.

먼저 저자들은 anomaly를 세 가지로 구분한다. point anomaly는 개별 데이터 포인트가 정상 범위를 벗어난 경우이다. collective anomaly는 개별 포인트는 정상 범위일 수 있으나, 여러 포인트가 이루는 시퀀스 구조 전체가 이상한 경우이다. contextual anomaly는 개별 포인트나 그룹 자체만 보면 정상처럼 보이지만, 주변 문맥과 비교했을 때 비정상인 경우이다. 이 분류는 survey 전체에서 어떤 모델이 무엇을 잘 탐지하는지 설명하는 기준이 된다.

문맥의 의미도 중요하게 정의된다. multivariate time series에서 특정 데이터 벡터 또는 데이터 벡터 집합의 context는 일정 시간 구간 내 주변 데이터들의 집합으로 이해된다. 기존 방법은 sliding window를 사용해 현재 윈도우와 이전 윈도우 간 거리 차이를 계산하고 동적 threshold를 넘으면 contextual anomaly로 보는 식이었다. 하지만 저자들은 이런 방식이 문맥을 직접 학습하는 것이 아니라 수동적으로 비교하는 수준이라고 본다. LSTM은 이 문맥을 hidden state와 cell state 안에 압축된 상태 표현으로 유지할 수 있기 때문에 더 강력하다고 해석한다.

논문은 LSTM cell의 기본 구조도 짧게 설명한다. 핵심은 출력 $h(t)$가 단순히 현재 입력 $x(t)$와 이전 출력 $h(t-1)$의 조합이 아니라, cell state $z(t)$를 거쳐 계산된다는 점이다. 그리고 cell state는 forget gate와 add gate에 의해 반복적으로 업데이트된다. 즉, 매 시점마다 과거 정보 중 일부는 잊고, 일부는 새로 추가하면서 memory behavior를 제어한다. 그 결과 중요한 과거 사건은 오래 유지하고, 정보량이 적은 최근 입력은 상대적으로 덜 반영하는 것이 가능해진다. 이 설명은 공식 중심이라기보다 개념 중심이며, 논문 본문에도 구체적인 수식 전개는 제공되지 않는다.

### 3.1 Regular LSTM 기반 접근

가장 기본적인 형태는 LSTM을 예측기(predictor)로 사용하는 방식이다. 정상 시계열을 학습한 뒤 다음 시점 또는 미래 시퀀스를 예측하고, 실제 값과 예측 값의 차이를 anomaly score로 사용한다. 예를 들어 Malhotra et al. [21]의 stacked LSTM은 차원 축소 feature 없이 원 시계열을 입력받아 시계열 이상을 탐지한다. 이때 핵심 아이디어는 reconstruction error가 아니라 prediction error를 본다는 점이다. variance analysis 등을 통해 예측 오차의 크기나 패턴이 크면 anomaly로 간주한다.

이 계열의 장점은 구조가 비교적 단순하면서도 contextual anomaly를 잘 잡는다는 점이다. LSTM이 정상적인 temporal pattern을 배운다면, 정상 문맥에서는 예측이 잘 맞고 이상 문맥에서는 residual이 커지기 때문이다. 차량 버스 통신을 다룬 연구 [22]에서는 정상 통신 패턴을 예측하고, 실제 통신이 dynamic threshold를 넘는 정도로 벗어나면 cyber-attack 성격의 anomaly를 탐지한다.

collective anomaly를 위해서는 한 시점의 오차만 보는 것으로 부족할 수 있다. Bontemps et al. [24]는 여러 step의 one-step-ahead prediction error를 묶어서 평가한다. 즉, 각 시간점별 독립 판단이 아니라, 일정 구간에 걸쳐 누적된 예측 실패 패턴이 collective anomaly를 나타낸다고 본다. 이것은 survey가 강조하는 중요한 포인트다. anomaly가 시간 구조를 갖는다면, score 역시 시간적으로 통합되어야 한다는 것이다.

또 다른 변형으로 dual LSTM [25]은 하나의 LSTM으로 short-term characteristic을 모델링하고, 다른 하나로 long-term threshold control을 수행한다. 이는 사실상 국소적인 이상과 장기 drift를 분리해 다루려는 시도다. survey 논문은 이를 real-time capable approach로 소개한다.

### 3.2 Encoder-Decoder 기반 접근

이 범주는 산업 현장에서 특히 중요하다. 실제 데이터에는 label이 부족하고, 정상/이상 상태를 정확히 구분한 정보 모델도 없는 경우가 많기 때문이다. 이런 상황에서는 unsupervised learning이 필요하고, 그 대표 구조가 Autoencoder(AE)와 Seq2Seq이다.

Autoencoder 기반 접근에서는 encoder가 입력 시계열을 저차원 latent representation으로 압축하고, decoder가 이를 다시 복원한다. 정상 데이터로 학습하면 정상 패턴은 잘 복원되지만, 이상 데이터는 latent space에 잘 맞지 않아 reconstruction error가 커진다. 따라서 anomaly score는 대체로 reconstruction error 또는 likelihood 기반 지표가 된다.

LSTM AE의 강점은 단순 AE보다 temporal dependency를 latent representation 안에 담을 수 있다는 점이다. 즉, 입력의 시간 순서를 무시하지 않고 short-term 및 long-term temporal relation을 함께 압축한다. 이 때문에 단순 outlier뿐 아니라 collective anomaly, contextual anomaly에도 더 적합하다.

논문은 여러 변형을 소개한다. robust deep AE [27]는 PCA와 regularization layer를 함께 사용해 입력 잡음을 줄이고 더 강건한 탐지를 수행한다. contractive LSTM AE [28]는 anomaly separation과 relation discovery를 모두 고려하는 reconstruction metric을 사용한다고 설명된다. 다만 논문 본문에는 그 metric의 정확한 수식은 나오지 않는다. 따라서 이 survey를 기반으로는 손실 함수의 구체식을 복원할 수 없다. denoising LSTM AE 역시 손상된 입력에서 underlying relation을 회복하도록 학습하여 detection accuracy를 높이려는 목적을 가진다.

variational LSTM AE [29]는 더 확률적인 구조를 사용한다. encoder와 decoder가 deterministic vector가 아니라 분포를 다루며, 입력 시퀀스를 latent feature distribution으로 사상하고 다시 샘플 또는 대표값을 이용해 복원한다. anomaly detection은 real output과 reconstructed output에 대한 log-likelihood score 계산으로 수행된다. 즉, 단순한 $L_1$ 또는 $L_2$ 재구성 오차 대신, 정상 분포 하에서 얼마나 가능도(likelihood)가 낮은가를 score로 삼는다. 이 방법은 latent representation을 차원 축소 용도로도 활용할 수 있다는 장점이 있다.

observer-based LSTM AE [30]는 다소 독특하다. 정상 제조 공정의 동작을 LSTM AE로 모델링한 뒤, decoder를 inverse process model처럼 사용한다. 그리고 실제 actuating variable과 재구성된 actuating variable을 disturbance observer 관점에서 비교하여 anomaly를 잡는다. 즉, 단순히 데이터 자체를 복원하는 것을 넘어서, 시스템 입력-출력 관계를 역으로 추정하는 모델로 쓰는 접근이다.

Seq2Seq LSTM도 encoder-decoder 구조의 또 다른 형태다. [32]에서는 encoder에서 decoder로 전달되는 cell state와 copying vector의 이상 여부를 바탕으로 anomaly를 판단하고, 이후 clustering으로 후처리한다. [33]은 여러 속성을 모델링하고 예측하여 다양한 anomaly type을 탐지한다. [34]는 recurrent autoencoder ensemble에 skip connection을 넣고, 여러 encoder가 하나의 copying layer를 공유하는 구조를 제안한다. 이 접근의 목적은 generalization과 extrapolation을 높여 과적합을 줄이는 것이다. 저자들에 따르면 이 구조의 cost function은 전체 reconstruction error를 최소화하면서, joint copying layer의 정보 흐름을 제어하는 penalty term을 포함한다. 이를 수식으로 쓰면 개념적으로는

$$
\mathcal{L} = \sum_i \text{ReconstructionError}_i + \lambda \cdot \text{Penalty}
$$

와 같은 형태로 이해할 수 있다. 다만 survey 본문은 정확한 함수 형태를 제시하지 않으므로, 위 식은 구조적 해석 수준의 표현일 뿐 원 논문의 정확한 식이라고 단정할 수는 없다.

### 3.3 Hybrid 접근

Hybrid approach는 LSTM 하나만으로 예측과 탐지를 모두 해결하기보다, 서로 다른 네트워크나 기법을 결합해 역할을 분담하는 방식이다. 논문은 이를 predictor와 detector의 조합으로 설명한다. 일반적으로 한 구성 요소는 정상 동역학을 예측하거나 표현을 만들고, 다른 구성 요소는 그 표현 공간이나 residual에서 anomaly를 분리한다.

예를 들어 stacked AE와 LSTM의 조합 [35]에서는 encoder가 여러 시퀀스를 처리하며 feature를 추출하고, LSTM이 reconstructed feature space에서 deviation characteristic을 식별한다. 여기서 핵심은 AE가 표현 공간을 정리하고, LSTM이 시간적 deviation을 담당한다는 역할 분담이다.

또 다른 사례 [36]는 LSTM AE 뒤에 clustering algorithm을 붙여, 재구성된 시스템 동역학을 state space representation으로 본다. 그러면 anomaly는 갑작스러운 state transition, drift하는 transition, 또는 새로운 state의 생성으로 해석될 수 있다. 이것은 anomaly detection을 단순 오차 기반이 아니라 상태 전이 해석 문제로 확장하는 예다.

GAN + LSTM [37]에서는 generator가 정상 시계열을 복원하거나 생성하고, discriminator가 이 결과가 정상 입력에서 온 것인지 이상 입력에서 온 것인지 판별한다. 즉, reconstruction difficulty와 adversarial discrimination을 동시에 이용한다. 이때 generator는 일종의 decoder처럼 작동하며, discriminator가 정상/이상 경계를 학습한다.

CNN + LSTM [38]은 고차원, 다차원 데이터에서 유용하다. CNN이 공간적 또는 국소적 feature를 추출하고, LSTM이 그 feature의 시간 변화를 추적한다. 따라서 spatial dimension과 temporal dimension이 동시에 중요한 웹 트래픽이나 복합 센서 데이터에서 contextual anomaly를 탐지하는 데 적합하다. 분류는 cross entropy 기반으로 이루어진다고 정리되어 있다.

LSTM + EWMA [39]는 예측 residual을 부드럽게 누적해서 anomaly를 보는 방식이다. LSTM이 multivariate time series의 정상 패턴을 예측하고, EWMA와 dynamic thresholding이 residual sequence를 분석한다. EWMA를 쓰는 이유는 순간적인 noise보다 지속적 deviation을 더 잘 드러내기 위해서로 이해할 수 있다. 이 구조의 장점은 하나의 detection process 안에서 새로운 시퀀스 전체의 contextual anomaly를 빠르게 찾을 수 있다는 점이다.

### 3.4 Graph-based approach와 Transfer Learning

논문의 4장은 strict sense의 LSTM architecture 설명은 아니지만, 앞으로의 anomaly detection 확장을 이해하는 데 중요하다.

Graph-based approach는 heterogeneous data와 contextual relation을 노드와 엣지로 표현하는 사전 단계로 볼 수 있다. 데이터가 이미 graph 형태이거나, 여러 데이터베이스와 포맷의 정보를 graph로 구성한 뒤, graph database에 저장하고 시간에 따라 versioning한다. 그 다음 structural feature나 temporal feature를 기반으로 subgraph clustering 또는 partitioning을 수행하고, cluster 안에서 anomalous node/edge를 찾는다. 이후 graph embedding을 통해 vector representation을 만든 뒤, 이를 다시 LSTM 입력으로 사용할 수 있다. 즉, graph는 context structuring layer이고, LSTM은 dynamic behavior modeling layer가 될 수 있다.

Transfer learning은 데이터 부족 문제를 해결하는 방향이다. 논문은 parameter transfer와 relational knowledge transfer를 소개한다. 전자는 미리 학습한 network의 일부 또는 전체 파라미터를 target task에 재사용하는 방식이고, 후자는 여기에 domain adaptation을 더한 개념이다. 예를 들어 LSTM-based AE의 마지막 encoding/decoding layer만 target 데이터셋에 맞게 재학습하거나, ConvLSTM에 one-shot learning을 결합해 새로운 intrusion type을 빨리 학습하도록 만들 수 있다. 이 논문은 산업 anomaly detection에서 large and diverse dataset 확보가 어렵기 때문에, 이런 접근이 현실적으로 중요하다고 본다.

## 4. 실험 및 결과

이 논문은 survey이기 때문에 자체적인 단일 실험을 수행하지는 않는다. 대신 각 논문을 application scenario, data type, labels 유무, feature extraction 유무, anomaly type, architecture, adaptiveness, evaluation metric, performance의 관점에서 비교하는 표를 제공한다. 따라서 여기서의 “실험 및 결과”는 개별 연구들에 대한 저자들의 정리와 해석을 종합적으로 설명하는 것이 맞다.

먼저 regular LSTM 계열을 보면, Malhotra et al. [21]의 stacked LSTM은 multivariate time series에서 contextual anomaly를 다루고, precision, recall, F1-score 기준으로 일반 RNN보다 높은 성능을 보였다고 정리된다. Ergen et al. [23]의 stacked LSTM-SVM은 collective anomaly와 contextual anomaly를 다루며, 여러 시나리오 예를 들어 HTTP request 데이터에서 AUC, ROC 관점에서 SVM과 SVDD보다 높다고 되어 있다. Bontemps et al. [24]의 LSTM 기반 collective anomaly 탐지는 PC network intrusion detection에 적용되었지만, 다른 방법과의 비교는 수행되지 않았다고 명시된다. 이는 survey 저자들이 비교 가능성과 실험 완성도를 중요하게 본다는 신호다. Lee et al. [25]의 dual LSTM은 multiple domain streaming data에서 precision, recall, F1-score 기준 경쟁 방법보다 높고 real-time capable하다고 정리된다.

encoder-decoder 기반 방법에서는 robust deep LSTM AE [27]가 image sequence에서 point anomaly를 다루며 benchmark dataset에서 높은 성능을 보였다고 요약된다. contractive LSTM AE [28]는 network intrusion detection용 multivariate time series에서 outlier, collective, contextual anomaly를 모두 다루며, ELM, k-NN, RF, SVM 같은 conventional ML method보다 AUC, ROC, precision, recall, accuracy에서 더 높았다고 정리된다. variational LSTM AE [29]는 robot system의 environmental anomaly 탐지에서 HMM, SVM, AE보다 높다고 보고된다. observer-based LSTM AE [30]는 discrete manufacturing에서 다양한 anomaly type을 다루지만, 타 방법과 직접 비교는 없었다고 명확히 적는다. 이것은 실제 산업 적용 사례가 있어도 benchmark와 비교 부재가 여전히 큰 문제라는 뜻이다.

Seq2Seq 및 recurrent autoencoder ensemble 계열에서는 [33]이 stacked LSTM [21]보다 benchmark dataset에서 더 좋은 결과를 냈다고 논문이 소개한다. [34]는 generalization과 extrapolation을 높이기 위한 구조를 제안하지만, 본 survey가 제공한 표 안에는 직접적인 수치 비교가 자세히 실려 있지 않다.

hybrid 방법에서는 GAN + LSTM [37]이 여러 시나리오에서 VAE와 SVM보다 precision, recall, F1-score 측면에서 높다고 정리된다. CNN + LSTM [38]은 web traffic anomaly detection에서 RF, MLP, KNN보다 좋았다고 소개된다. 반면 LSTM + EWMA [39]는 industrial robotic manipulators 시나리오에서 precision, recall을 사용했지만, 경쟁 기법과의 직접 비교는 없었다고 표에 적혀 있다.

graph-based approach에 대한 표는 세 편 정도만 다루며, 아직 연구 수가 적고 benchmark가 부족함을 드러낸다. Prado-Romero et al. [40]의 graph clustering + outlier score 함수는 contextual collective anomaly를 다루고, Amazon purchased products 데이터에서 clustered attributed graph 기반 outlier ranking보다 약간 높았다고 요약된다. Haque et al. [41]의 minimum spanning tree clustering + voting scheme은 sensor data의 contextual outlier detection에 사용되며, 별도 label 없이 동작한다. Zheng et al. [42]의 GCN with GRU는 dynamic graph에서 anomalous edges를 탐지하며, 세 가지 graph-outlier algorithm보다 AUC가 높았다고 정리된다.

transfer learning 접근도 흥미롭지만, survey의 결론은 아직 직접 비교가 어렵다는 쪽이다. Liang et al. [47]은 electricity consumption anomaly detection에 denoising AE transfer learning을 적용했지만 경쟁 기법과 비교가 없다고 적혀 있다. Canizo et al. [48]의 CNN + 여러 RNN 변형은 service elevator operation에서 높은 accuracy를 보였으나 경쟁 기법과 직접 비교는 부족하다. Hsieh et al. [49]의 LSTM + AE transfer는 discrete manufacturing에서 non-transfer approach보다 accuracy가 높았다고 정리된다. Tariq et al. [50]의 CANTransfer는 CAN intrusion detection에서 SVM, isolation forest, ensemble RNN보다 unknown anomaly task에서 더 좋고, known task에서는 최고 경쟁자보다 약간 떨어진다고 설명된다. 또한 CAN data rate 대비 real-time 탐지가 가능하다고 한다. Maschler et al. [52]의 continual learning 기반 stacked LSTM은 product change가 잦은 metal forming process에서 elastic weight consolidation을 사용해 old task와 new task 모두에서 conventional approach보다 의미 있는 향상을 보였다고 요약된다.

이 survey가 실험 결과를 통해 실제로 전달하는 가장 중요한 메시지는 다음과 같다. 첫째, LSTM 계열 방법은 point anomaly만이 아니라 collective anomaly와 contextual anomaly에 강점을 보인다. 둘째, encoder-decoder 구조는 high-dimensional data에서 특히 유리하다. 셋째, graph-based와 transfer learning은 유망하지만 아직 공개 benchmark와 비교 실험이 부족하여 명확한 승패를 말하기 어렵다. 즉, 성능 향상 사례는 많지만, 연구 분야 전체가 아직 비교 가능성과 표준화 면에서는 미성숙하다는 점도 이 survey의 중요한 실험적 결론이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 LSTM 기반 anomaly detection을 구조적으로 분류하고, anomaly type과 application scenario를 연결해서 설명한다는 점이다. 단순히 “LSTM이 좋다”라고 말하는 것이 아니라, regular LSTM은 temporal prediction에 기반한 anomaly detection에 적합하고, LSTM AE나 variational LSTM AE는 unlabeled setting과 고차원 데이터에 유리하며, hybrid approach는 predictor와 detector의 역할 분리를 통해 더 정교한 탐지가 가능하다고 체계화한다. 이런 분류는 실제 연구자나 엔지니어가 자신의 데이터와 문제 조건에 맞는 계열을 선택하는 데 도움이 된다.

두 번째 강점은 contextual anomaly의 중요성을 강조한다는 점이다. 많은 anomaly detection survey는 여전히 point anomaly 중심 설명에 머무르기 쉬운데, 이 논문은 technical system에서 정말 어려운 문제는 문맥 의존적 이상이라는 점을 분명히 한다. LSTM을 선택하는 이유도 단순한 sequence model이어서가 아니라, context를 hidden state에 압축해 담을 수 있기 때문이라고 설명한다. 이는 논리적으로 설득력이 있다.

세 번째 강점은 graph-based approach와 transfer learning을 별도의 최근 동향으로 포함했다는 점이다. 이로 인해 survey가 단순 정리형 문서가 아니라, 연구 로드맵을 제공하는 문서 역할도 한다. 특히 산업 현장에서 데이터 부족, heterogeneous data, interacting systems 문제를 함께 보고 있다는 점이 실용적이다.

하지만 한계도 뚜렷하다. 첫째, survey임에도 정량 비교의 일관성이 충분하지 않다. 표는 잘 정리되어 있지만, 각 논문이 서로 다른 데이터셋, 서로 다른 metric, 서로 다른 anomaly type에서 평가되었기 때문에 “어떤 구조가 가장 좋다”는 식의 직접 결론은 어렵다. 저자들도 이를 어느 정도 인정하고 있으며, 공개 benchmark 부재를 반복적으로 문제 삼는다. 따라서 이 survey는 landscape를 보여주는 데는 강하지만, 객관적 ranking을 제공하는 데는 제한적이다.

둘째, 일부 설명은 개념 수준에 머무른다. 예를 들어 contractive LSTM AE, variational LSTM AE, ensemble recurrent autoencoder의 구체적 손실 함수나 수식 구조는 survey 본문에서 자세히 전개되지 않는다. 이는 survey 논문이라는 장르상 자연스럽지만, 독자가 이 문서만으로 실제 구현 수준까지 이해하기에는 부족할 수 있다.

셋째, graph-based approach 부분은 흥미롭지만 LSTM과의 실제 결합 사례가 충분히 제시되지는 않는다. 논문도 “theoretically promising”라고 표현하며, 실제 구현 사례가 거의 없다고 인정한다. 즉, 이 장은 현황 보고라기보다는 전망 제시에 가깝다.

넷째, transfer learning 부분도 아직 방법 수가 적고 benchmark가 부족하다. 저자들은 promising하다고 평가하지만, 명확한 trend를 식별할 수 없다고 스스로 말한다. 따라서 독자가 이 부분을 읽을 때는 “강한 결론”보다 “향후 가능성”으로 받아들이는 것이 맞다.

비판적으로 보면, 이 논문은 anomaly detection의 어려움을 잘 설명하지만, 실제 산업 배치 관점에서 false positive cost, latency, explainability, maintenance burden 같은 운영 문제는 상대적으로 덜 다룬다. 물론 survey의 범위를 생각하면 과도한 요구일 수 있으나, technical system 적용을 강조했다면 deployment 관점의 논의가 조금 더 있었으면 더 강한 survey가 되었을 것이다.

## 6. 결론

이 논문은 technical systems에서의 anomaly detection을 위해 LSTM 기반 접근을 중심으로 기존 연구를 정리한 survey이다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, anomaly를 point, collective, contextual anomaly로 구분하고, 특히 시간적 문맥이 중요한 contextual anomaly 탐지에 LSTM이 적합하다는 점을 분명히 했다. 둘째, LSTM 기반 방법을 regular LSTM, encoder-decoder 기반, hybrid approach로 나누어 각각의 구조와 탐지 메커니즘, 적용 시나리오를 비교했다. 셋째, 향후 중요한 확장 방향으로 graph-based learning과 transfer learning을 제시함으로써, anomaly detection이 단순 시계열 예측 문제를 넘어 관계 구조와 지식 이전 문제로 발전해야 함을 강조했다.

실제 적용 측면에서 이 연구의 의미는 크다. 제조, 로보틱스, 차량 통신, 네트워크 보안 등에서는 이상 현상이 시간 구조를 가지며, 데이터 레이블이 부족하고, 시스템 간 차이가 크다. 이런 상황에서 LSTM은 temporal modeling의 좋은 출발점이 되고, AE나 hybrid 구조는 unlabeled setting과 복잡한 데이터 표현을 다루는 데 도움이 된다. 또한 graph-based representation은 heterogeneous context를 통합하는 수단이 될 수 있고, transfer learning은 데이터 부족이라는 산업 현장의 핵심 제약을 줄일 가능성이 있다.

향후 연구 방향에 대해서도 논문은 비교적 설득력 있는 메시지를 준다. 이상 탐지는 앞으로 단일 장비나 단일 시계열을 넘어서, interacting systems의 network 수준에서 다뤄져야 한다. 이를 위해서는 graph 구조 안에서 context를 표현하고, 그 위에서 LSTM 또는 관련 sequence model이 dynamic behavior를 모델링하는 방식이 유망하다. 동시에 transfer learning이나 continual learning을 통해 시스템 간, 시간 간 knowledge transfer를 가능하게 해야 한다. 저자들의 결론대로, 실제 산업 커뮤니티가 직면한 큰 문제는 “새롭고 복잡한 정상 행동”과 “진짜 이상 행동”을 구별하는 것이다. 이 점에서 LSTM, graph, transfer learning의 결합은 충분히 중요한 연구 방향으로 보인다.

다만 이 논문이 제시하는 결론은 어디까지나 survey에 기반한 종합적 해석이며, 특정 아키텍처의 절대적 우위를 증명하는 것은 아니다. 따라서 실무나 후속 연구에서는 데이터 특성, 라벨 유무, anomaly의 시간적 성질, 설명 가능성 요구, 실시간 제약 등을 함께 고려하여 모델군을 선택하는 것이 필요하다. 그럼에도 불구하고 이 논문은 LSTM 기반 anomaly detection의 전체 지형을 이해하는 데 매우 유용한 입문이자 정리 문서라고 평가할 수 있다.
