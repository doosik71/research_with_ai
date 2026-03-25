# Universal Domain Adaptation for Semantic Segmentation

* **저자**: Seun-An Choe, Keon-Hee Park, Jinwoo Choi, Gyeong-Moon Park
* **발표연도**: 2025
* **arXiv**:

## 1. 논문 개요

이 논문은 semantic segmentation에서의 **Universal Domain Adaptation** 문제를 새롭게 정의하고, 이를 해결하기 위한 프레임워크 **UniMAP**을 제안한다. 기존의 Unsupervised Domain Adaptation for Semantic Segmentation(UDA-SS)는 source domain에는 라벨이 있고 target domain에는 라벨이 없다는 설정에서, 두 도메인이 어떤 클래스를 공유하는지 미리 알고 있다고 가정한다. 그러나 실제 환경에서는 target 쪽 클래스 구성이 사전에 알려져 있지 않으며, source에만 있는 클래스(source-private)나 target에만 있는 클래스(target-private)가 동시에 존재할 수 있다. 저자들은 바로 이 점이 기존 방법의 실사용성을 크게 떨어뜨린다고 본다.

논문이 다루는 핵심 문제는 다음과 같다. semantic segmentation 모델이 unlabeled target domain에 적응할 때, 공통 클래스(common classes)와 private classes가 섞여 있으면 pseudo-label의 confidence가 크게 흔들린다. 특히 source-private classes가 존재하면, target의 common class 픽셀이 source-private class와 헷갈리면서 confidence가 떨어지고, 이 때문에 실제 common class가 unknown 또는 target-private로 잘못 처리될 수 있다. 결과적으로 common class도 제대로 학습되지 못하고, private class 탐지도 함께 악화된다.

이 문제는 중요하다. semantic segmentation은 자율주행, 의료영상, 인간-기계 상호작용 같은 실제 응용에서 쓰이는데, 이런 환경에서는 데이터 분포 변화(domain shift)뿐 아니라 클래스 구성 자체의 변화(category shift)도 빈번하다. 따라서 source와 target의 클래스 겹침을 미리 안다는 가정은 지나치게 이상적이다. 이 논문은 이런 현실적 제약을 반영해 **UniDA-SS(Universal Domain Adaptation for Semantic Segmentation)**라는 새로운 문제 설정을 제시하고, common class를 안정적으로 학습하면서 target-private class를 unknown으로 분류할 수 있는 방법을 제안한다는 점에서 의의가 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **target domain에서 common classes의 pseudo-label confidence를 높이는 것**이 UniDA-SS 문제 해결의 핵심이라는 관찰이다. 저자들은 기존 UDA-SS 및 open-set 계열 방법들이 pseudo-label confidence에 크게 의존하는데, source-private class가 존재하면 common class의 confidence가 낮아져 잘못된 unknown 할당이 빈번해진다고 분석한다. 즉, UniDA-SS의 본질적 어려움은 단순히 unknown을 검출하는 것이 아니라, **공통 클래스가 private class와 혼동되지 않도록 표현을 더 정교하게 만드는 것**이라고 본다.

이를 위해 제안된 UniMAP은 두 가지 핵심 구성요소를 결합한다. 첫 번째는 **DSPD(Domain-Specific Prototype-based Distinction)**이고, 두 번째는 **TIM(Target-based Image Matching)**이다.

DSPD의 직관은 이렇다. 기존 self-training 기반 UDA-SS는 같은 semantic class라면 source와 target에서 하나의 동일한 표현으로 맞추려고 하는 경향이 있다. 하지만 실제로는 같은 클래스라도 도메인이 다르면 appearance나 texture가 달라진다. 예를 들어 road 클래스라도 유럽과 인도의 도로는 시각적으로 다를 수 있다. 따라서 source와 target의 common class를 완전히 같은 representation으로 취급하는 것은 오히려 구분을 흐리게 만든다. 그래서 저자들은 클래스마다 하나의 prototype이 아니라 **source용 prototype과 target용 prototype을 각각 둔다**. 이렇게 하면 같은 클래스이되 도메인 특이성(domain-specificity)을 유지하면서도, 공통 클래스는 양쪽 prototype 모두와 어느 정도 잘 맞는 특성을 갖게 된다. 반대로 private class는 둘 중 하나에만 더 가깝게 나타날 가능성이 높다. 이 차이를 이용해 common/private 구분을 더 잘하게 만들겠다는 발상이다.

TIM의 직관은 학습 배치 구성 차원에서 common class 학습을 강화하는 것이다. source-private class가 많은 source 이미지를 무작위로 뽑으면, common class보다 private class 중심의 학습이 이루어질 수 있다. 그러면 domain-invariant representation 학습이 어려워진다. 그래서 target pseudo-label을 보고 현재 target 이미지에 어떤 클래스가 있는지 추정한 뒤, **그 클래스들과 가장 많이 겹치는 source image를 골라 같은 batch로 묶는다**. 이렇게 하면 common classes가 더 풍부하게 포함된 source-target 쌍을 학습하게 되어, 공통 클래스 중심의 적응이 더 효과적으로 일어난다. 또한 target에서 드문 클래스에는 더 높은 가중치를 줘 class imbalance도 완화하려고 한다.

기존 접근과의 차별점은 명확하다. 이 논문은 단순히 unknown head를 붙이거나 confidence threshold만 조정하는 수준이 아니라, **표현 공간에서 common/private를 구조적으로 구분하는 prototype 설계**와 **배치 샘플링 단계에서 common class 중심 학습을 유도하는 image matching 전략**을 함께 제안한다. 즉, prediction 단계와 representation learning 단계, 그리고 data pairing 단계까지 동시에 손을 댄다는 점이 차별적이다.

## 3. 상세 방법 설명

전체 프레임워크는 source의 supervised segmentation loss, target의 pseudo-label 기반 segmentation loss, 그리고 prototype-based loss를 함께 최적화하는 구조다. 논문에 따르면 기본 모델은 BUS를 바탕으로 구성되며, 여기에 DSPD와 TIM이 추가된다. 학습은 source labeled image와 target unlabeled image를 함께 사용한다.

### 3.1 문제 설정

source domain은 $D_s={X_s, Y_s}$, target domain은 $D_t={X_t}$이다. source에는 pixel-wise label이 있고, target에는 정답 라벨이 없다. source 클래스 집합을 $C_s$, target 클래스 집합을 $C_t$라고 할 때, 공통 클래스는 $C_c=C_s \cap C_t$이다. source-private class는 $C_s \setminus C_c$, target-private class는 $C_t \setminus C_c$이다. target-private class는 모델이 구체적 이름을 모른 채 **unknown**으로 처리해야 한다.

따라서 UniDA-SS에서는 두 문제가 동시에 중요하다. 하나는 common class를 올바르게 분류하는 것이고, 다른 하나는 target-private class를 unknown으로 탐지하는 것이다.

### 3.2 베이스라인

논문은 BUS의 open-set self-training 구조를 바탕으로 baseline을 정의한다. classifier head 수는 $(C_s+1)$개이며, 마지막 하나는 unknown class를 위한 것이다. source에 대해서는 일반적인 categorical cross-entropy를 사용한다.

$$
\mathcal{L}_{seg}^{s}=-\sum_{j=1}^{H\cdot W}\sum_{c=1}^{C_s+1} y_s^{(j,c)} \log f_{\theta}(x_s)^{(j,c)}
$$

여기서 $j$는 pixel index이고, $f_\theta$는 segmentation network이다.

target pseudo-label은 teacher network $g_\phi$가 생성한다. 이 teacher는 student인 $f_\theta$의 EMA(exponential moving average) 버전이다. 각 pixel에 대해 가장 높은 known-class confidence가 threshold $\tau_p$ 이상이면 그 클래스를 pseudo-label로 쓰고, 아니면 unknown으로 둔다.

$$
\hat{y}_{tp}^{(j)}=
\begin{cases}
c', & \text{if } \max_{c'} g_\phi(x_t)^{(j,c')} \ge \tau_p \
C_s+1, & \text{otherwise}
\end{cases}
$$

또한 이미지 단위 신뢰도 $q_t$를 계산한다.

$$
q_t=\frac{1}{H\cdot W}\sum_{j=1}^{H\cdot W}\left[\max_{c'}g_\phi(x_t)^{(j,c')} \ge \tau_t\right]
$$

즉, 전체 픽셀 중 confidence가 일정 threshold 이상인 픽셀 비율을 이미지-level reliability로 사용한다. 이 값을 이용해 target loss를 가중한다.

$$
\mathcal{L}_{seg}^{t}=-\sum_{j=1}^{H\cdot W}\sum_{c=1}^{C_s+1} q_t \cdot \hat{y}_{tp}^{(j,c)} \log f_{\theta}(x_t)^{(j,c)}
$$

이 baseline 자체도 open-set segmentation 구조를 갖고 있지만, 논문은 이것만으로는 source-private class가 들어오는 상황에서 common class confidence 하락을 막기 어렵다고 본다.

### 3.3 DSPD: Domain-Specific Prototype-based Distinction

DSPD는 이 논문의 핵심적인 표현 학습 장치다. 아이디어는 클래스마다 source prototype과 target prototype을 각각 두는 것이다. 저자들은 ProtoSeg의 prototype-based learning 개념을 확장하되, domain-specific distinction을 위해 각 클래스에 두 개의 prototype을 할당한다.

prototype들은 **Simplex Equiangular Tight Frame(ETF)** 공간에 고정된 형태로 배치된다. 논문은 다음 식으로 prototype 집합을 정의한다.

$$
{p_k}_{k=1}^{2C+1} = \sqrt{\frac{2C+1}{2C}} U\left(I_{2C+1}-\frac{1}{2C+1}1_{[2C+1]}1_{[2C+1]}^\intercal\right)
$$

여기서 각 클래스 $c$는 $p_s^c$와 $p_t^c$라는 source/target prototype 쌍을 가진다. 추가로 unknown class를 위한 prototype도 target 측에 하나 더 둔다고 설명한다. ETF를 쓰는 이유는 prototype들 사이의 cosine similarity와 norm을 균등하게 유지해, 공간상 분리가 안정적으로 이루어지도록 하기 위함이다.

이 prototype 구조 위에서 논문은 세 가지 loss를 사용한다.

첫 번째는 prototype 기준 cross-entropy loss다.

$$
\mathcal{L}_{CE}^{D} = -\log \frac{\exp(i^\intercal p_D^c)} {\exp(i^\intercal p_D^c)+\sum_{c' \ne c}\exp(i^\intercal p_D^{c'})}
$$

여기서 $i$는 $L_2$ normalize된 pixel embedding이고, $D \in {s,t}$는 source 또는 target domain이다. source에서는 ground-truth label, target에서는 pseudo-label을 사용해 해당 클래스 prototype 쪽으로 embedding을 끌어당긴다.

두 번째는 pixel-prototype contrastive loss다.

$$
\mathcal{L}_{PPC}^{D} = -\log \frac{\sum_{p \in p^c}\exp(i^\intercal p^c/\tau)} {\sum_{p \in p^c}\exp(i^\intercal p^c/\tau)+\sum_{p^- \in P^-}\exp(i^\intercal p^-/\tau)}
$$

이 식은 정답 클래스 prototype 집합에는 가깝게, 나머지 prototype들에는 멀어지도록 만드는 contrastive 성격을 가진다. 여기서 $P^-$는 해당 클래스 prototype을 제외한 나머지 prototype 집합이다.

세 번째는 pixel-prototype distance optimization loss다.

$$
\mathcal{L}_{PPD}^{D}=(1-i^\intercal p_D^c)^2
$$

이 식은 pixel embedding과 해당 prototype의 유사도를 직접 높이도록 만든다.

최종 prototype loss는 다음과 같다.

$$
\mathcal{L}_{proto} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{PPC} + \lambda_2 \mathcal{L}_{PPD}
$$

이 구조를 통해 모델은 클래스 단위로는 통합된 semantic을 유지하되, source와 target의 도메인별 차이를 별도 prototype에 담을 수 있다.

### 3.4 Prototype-based Weight Scaling

DSPD는 prototype을 단지 representation 학습에만 쓰지 않고, pseudo-label 생성과 target loss weighting에도 활용한다. 저자들의 논리는 다음과 같다. common class pixel embedding은 학습이 진행될수록 source prototype과 target prototype 모두에 비교적 가깝게 된다. 반면 private class는 둘 중 하나에만 더 가까워질 것이다. 따라서 embedding이 source/target prototype 둘 다와 유사하면 common class일 가능성이 높다.

이를 바탕으로 pixel-wise weight scaling factor $w$를 정의한다.

$$
w=\frac{2(d_s+1)(d_t+1)}{(d_s+1)+(d_t+1)}
$$

여기서 $d_s$, $d_t$는 각각 pixel embedding과 source/target prototype 사이의 cosine similarity다. 식 형태를 보면 일종의 조화 평균 비슷한 구조이며, 두 유사도가 모두 높을 때 $w$가 커지고 한쪽만 높으면 상대적으로 작아진다. 즉, common class처럼 양쪽 prototype과 모두 잘 맞는 픽셀에 더 높은 가중치를 주도록 설계되어 있다.

이 $w$는 target segmentation loss에 직접 들어간다.

$$
\mathcal{L}_{seg}^{t} = -\sum_{j=1}^{H\cdot W}\sum_{c=1}^{C+1} w \cdot q_t \cdot \hat{y}_{tp}^{(j,c)} \log f_\theta(x_t)^{(j,c)}
$$

또한 pseudo-label 생성 threshold에도 곱해진다.

$$
\hat{y}_{tp}^{(j)} =
\begin{cases}
c', & \text{if } \max_{c'} g_\phi(x_t)^{(j,c')} \cdot w \ge \tau_p \\
C+1, & \text{otherwise}
\end{cases}
$$

즉, common class일 가능성이 높은 픽셀은 더 쉽게 known class로 유지되고, 그렇지 않은 픽셀은 unknown 쪽으로 가게 된다. 이로써 common class가 target-private로 잘못 떨어지는 문제를 완화하려는 것이다.

### 3.5 TIM: Target-based Image Matching

TIM은 batch 구성 전략이다. 저자들은 common class confidence를 높이려면 공통 클래스를 더 자주, 더 많이 학습해야 한다고 본다. 문제는 source-private class가 많은 source 이미지를 함께 쓰면 common class 비중이 줄어들어 domain-invariant learning이 약해진다는 점이다.

먼저 target pseudo-label에서 클래스별 픽셀 비율을 계산한다.

$$
f_c=\frac{n_c}{\sum_k n_k}
$$

여기서 $n_c$는 target pseudo-label에서 클래스 $c$의 픽셀 수다.

다음으로 rare class에 더 높은 중요도를 주기 위해 다음 가중치를 만든다.

$$
\hat{f_c}=\text{softmax}\left(\frac{1-f_c}{T}\right)
$$

빈도가 낮을수록 $1-f_c$가 커지므로, rare class가 더 높은 weight를 받게 된다.

그 다음 각 source image에 대해, 현재 target pseudo-label과 겹치는 클래스 집합 $c^*$를 기준으로 score $S_s$를 계산한다.

$$
S_s=\sum_{c \in c^*} n_c^s \hat{f_c}
$$

여기서 $n_c^s$는 source ground truth에서 클래스 $c$의 픽셀 수다. 즉, target에 나타난 클래스들, 특히 rare common class를 많이 포함하는 source image일수록 높은 점수를 받는다. 최종적으로 가장 높은 $S_s$를 가진 source image를 선택해 그 target image와 같은 batch로 묶는다.

이 전략의 목적은 명확하다. target에 현재 존재하는 common class와 가장 잘 대응되는 source 이미지를 골라, common class에 대한 domain-invariant representation을 더 잘 배우도록 유도하는 것이다. 동시에 rare common classes도 놓치지 않도록 weighting을 넣었다.

### 3.6 전체 학습 관점에서 본 UniMAP

정리하면 UniMAP은 다음 세 축으로 동작한다. 첫째, source supervised loss로 기본 segmentation 성능을 유지한다. 둘째, target pseudo-label 기반 self-training을 수행하되, prototype 기반 weight scaling으로 common/private 혼동을 줄인다. 셋째, prototype loss로 domain-specific prototype 구조를 학습하고, TIM으로 common-class 중심 source-target pairing을 유도한다. 결국 이 프레임워크는 **confidence calibration, representation structuring, data pairing**을 동시에 개선하는 방식이라고 볼 수 있다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 설정

논문은 두 개의 새 OPDA-SS benchmark를 사용한다.

첫 번째는 **Pascal-Context $\rightarrow$ Cityscapes**이다. 이는 real-to-real setting이며, Pascal-Context는 indoor와 outdoor를 모두 포함하는 반면 Cityscapes는 driving scene 중심이라 source-private class가 많이 생긴다. 논문은 12개 클래스를 common classes로 두고, 나머지 7개 클래스인 “pole”, “light”, “sign”, “terrain”, “person”, “rider”, “train”을 target-private로 둔다고 설명한다.

두 번째는 **GTA5 $\rightarrow$ IDD**이다. 이는 synthetic-to-real setting이다. GTA5는 도시 주행 장면의 synthetic 데이터이고, IDD는 인도 도로의 실제 driving dataset이다. 여기서는 17개 common classes, 2개의 source-private class인 “terrain”, “train”, 그리고 1개의 target-private class인 “auto-rickshaw”를 사용한다.

평가 지표는 **H-score**다. 이는 common mIoU와 target-private IoU의 harmonic mean이다. 이 설정에서는 common class 성능만 좋아도 안 되고, unknown 또는 private class 탐지도 좋아야 하므로 H-score가 타당한 지표로 제시된다.

### 4.2 구현 세부사항

논문은 BUS를 기반으로 하되, MIC의 multi-resolution self-training 전략과 training parameter를 사용했다고 적고 있다. backbone은 MiT-B5 encoder이며, ImageNet-1k pretrained initialization을 사용했다. optimizer는 AdamW이고, backbone learning rate는 $6\times 10^{-5}$, decoder head는 $6\times 10^{-4}$, weight decay는 0.01이다. 1.5k step warm-up이 적용된다. EMA factor는 $\alpha=0.999$다.

또한 ImageNet feature distance, DACS augmentation, MIC module, Dilation-Erosion-based Contrastive Loss를 사용했다고 한다. BUS의 일부 구성도 OPDA setting에 맞게 수정했다. 예를 들어 OpenReMix에서는 Resizing Object만 적용하고, Attaching Private는 제거했으며, MobileSAM을 이용한 refinement도 사용하지 않았다. rare class sampling 역시 기존 source 분포 기반이 아니라 target pseudo-label 분포 기반 target sampling으로 바꿨다.

학습은 batch size 2, $512 \times 512$ random crop, 40k iteration으로 진행된다. 하이퍼파라미터는 $\tau_p=0.5$, $\tau_t=0.968$, $\lambda_1=0.01$, $\lambda_2=0.01$, $\tau=0.1$, $T=0.01$이다.

### 4.3 주요 정량 결과

#### Pascal-Context $\rightarrow$ Cityscapes

OPDA-SS benchmark에서 UniMAP은 Common 60.94, Private 31.27, H-score 41.33을 기록했다. 비교 대상 중 가장 강한 baseline인 BUS는 Common 57.64, Private 20.38, H-score 30.11이다. 따라서 UniMAP은 BUS 대비 Common에서 약 3.3, Private에서 약 10.89, H-score에서 약 11.22 향상되었다. 이 수치는 논문이 주장한 핵심, 즉 common class 학습을 강화하면서도 private class 구분까지 개선했다는 점을 뒷받침한다.

특히 Private 성능 상승폭이 크다는 점이 눈에 띈다. 제안 방법은 common class confidence 향상을 주목적으로 하지만, 결과적으로 private class detection에도 큰 이득을 준다. 이는 common/private 경계가 더 구조적으로 정리되면 unknown 판정도 더 안정화될 수 있음을 시사한다.

#### GTA5 $\rightarrow$ IDD

이 benchmark에서 UniMAP은 Common 64.08, Private 34.78, H-score 45.51을 기록했다. BUS는 Common 65.47, Private 29.70, H-score 41.26이다. 여기서는 Common 평균은 BUS보다 조금 낮지만, Private 성능이 약 5.08 더 높고 H-score가 약 4.25 더 높다. 논문 본문에서는 여러 이전 방법 대비 Private와 H-score에서 큰 향상이 있다고 설명한다.

이 결과는 매우 중요하다. UniMAP은 모든 상황에서 common accuracy만 극단적으로 끌어올리는 방법이 아니라, **common과 private 사이의 균형을 더 잘 맞춘다**는 점을 보여준다. 특히 Universal / Open Partial setting에서는 이런 균형이 실제 성능의 핵심이다.

### 4.4 정성적 결과

논문은 Cityscapes 예시에서 HRDA, MIC, BUS와 UniMAP의 예측 결과를 비교한다. 설명에 따르면 기존 baseline들은 common class를 target-private로 오분류하거나, target-private를 맞추려다 common class 성능을 희생하는 경향이 있다. 반면 UniMAP은 common과 target-private를 모두 비교적 잘 구분한다. 특히 “sidewalk” 같은 클래스가 다른 방법보다 더 정확하게 분할된다고 서술한다.

정성 결과는 정량 결과와 잘 연결된다. 즉, 제안 방법은 단순히 평균 수치만 좋아진 것이 아니라, 실제 예측 맵에서 common/private 구분이 더 자연스럽고 일관되게 나타난다는 점을 보여주려 한다.

### 4.5 Ablation Study

#### UniMAP 전체 구성 요소의 기여

Pascal-Context $\rightarrow$ Cityscapes에서 baseline은 Common 53.79, Private 26.54, H-score 36.03이다. 여기에 DSPD만 추가하면 Common 59.46, Private 27.97, H-score 38.04가 된다. TIM만 추가하면 Common 56.22, Private 29.14, H-score 38.39가 된다. 둘 다 추가하면 Common 60.94, Private 31.27, H-score 41.33으로 가장 좋다.

이 결과는 DSPD와 TIM이 서로 다른 측면에서 기여함을 보여준다. DSPD는 domain-specific feature modeling과 common/private distinction 측면에서 기여하고, TIM은 batch-level alignment와 class imbalance 완화에 기여한다. 두 방법을 함께 쓸 때 상승폭이 더 크다는 점에서 상호보완성이 있다고 해석할 수 있다.

#### DSPD 내부 구성 분석

DSPD의 두 요소인 $w$와 $\mathcal{L}_{proto}$를 따로 분석한 결과도 제공된다. baseline에서 $w$만 쓰면 Common 54.38, Private 21.75, H-score 31.08로 오히려 나빠진다. 반면 $\mathcal{L}*{proto}$만 쓰면 Common 59.71, Private 26.76, H-score 36.96으로 baseline보다 개선된다. 둘을 함께 쓰면 Common 59.46, Private 27.97, H-score 38.04로 가장 좋다.

이 결과는 논문의 중요한 메시지 하나를 보여준다. **weight scaling $w$는 standalone으로는 충분히 동작하지 않는다**. 즉, prototype 구조가 먼저 제대로 형성되어야 $w$가 common/private distinction 신호로서 의미 있게 작동한다. 다시 말해, $w$는 prototype learning이 뒷받침될 때만 효과적이다. 이는 방법 설계의 논리적 일관성을 뒷받침한다.

### 4.6 다양한 category setting에서의 비교

Pascal-Context $\rightarrow$ Cityscapes에 대해 Open Partial Set DA, Open Set DA, Partial Set DA, Closed Set DA를 모두 비교한 결과도 제시된다. 여기서 UniMAP은 일부 특화 방법보다 Closed Set 또는 Open Set 한정 성능은 아주 약간 뒤질 수 있지만, 다양한 설정을 통틀어 평균적으로 가장 강인한 성능을 보인다. 논문은 Common Average 60.86, H-score Average 37.90을 기록했다고 강조한다.

이 부분은 제안 방법의 실용성을 뒷받침한다. 특정 가정을 만족하는 좁은 setting에만 특화된 방법보다, category setting이 불확실한 현실 환경에서 더 안정적으로 동작하는 것이 UniMAP의 장점이라는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation 분야에서 **Universal Domain Adaptation** 문제를 명확히 제기했다는 점이다. classification에서는 UniDA가 연구되어 왔지만, segmentation은 픽셀 단위 예측이 필요하기 때문에 훨씬 더 어렵다. 이 논문은 այդ 문제를 정식으로 설정하고 benchmark까지 제안했다는 점에서 문제 정의 자체의 기여가 있다.

두 번째 강점은 방법론이 문제 분석과 잘 연결되어 있다는 점이다. 저자들은 단순히 성능 향상 기법을 덧붙인 것이 아니라, 왜 성능이 무너지는지를 “common class confidence 저하”라는 형태로 진단하고, 이를 representation 측면의 DSPD와 data pairing 측면의 TIM으로 해결하려 한다. 즉, 문제 진단과 해법이 비교적 일관되다.

세 번째 강점은 ablation 결과가 설계 논리를 잘 뒷받침한다는 것이다. $\mathcal{L}_{proto}$ 없이 $w$만 쓰면 성능이 떨어지는 결과는, prototype 구조 학습이 선행되어야 weighting도 의미를 갖는다는 저자들의 주장을 잘 지지한다. 또한 DSPD와 TIM을 함께 썼을 때 가장 좋은 성능이 나오는 점도 두 구성요소의 보완 관계를 설득력 있게 보여준다.

네 번째 강점은 Private 성능 개선이다. 많은 adaptation 방법은 common classes 성능에 집중하다가 unknown detection을 희생하거나, 반대로 unknown을 잘 잡으려다 common 성능을 해칠 수 있다. UniMAP은 특히 Pascal-Context $\rightarrow$ Cityscapes와 GTA5 $\rightarrow$ IDD에서 H-score를 크게 높이며 균형 잡힌 개선을 보인다.

한편 한계도 있다. 첫째, 논문은 방법이 여러 기존 모듈에 의존한다. 예를 들어 BUS, MIC, DACS, feature distance, contrastive loss 등 다양한 요소가 함께 들어가 있어, UniMAP 자체의 순수 기여가 얼마나 간결한지에 대해서는 다소 복합적인 인상을 준다. 물론 ablation이 이를 어느 정도 보완하지만, 전체 시스템은 꽤 많은 기존 기법 위에 구축되어 있다.

둘째, prototype 설계와 weighting의 안정성은 데이터셋 특성에 따라 달라질 수 있다. 논문은 common class가 source/target 두 prototype에 모두 가깝고 private class는 한쪽에 더 가깝다는 가정을 사용한다. 이는 직관적으로 타당하지만, 실제로 domain gap이 매우 크거나 pseudo-label 자체가 불안정한 초기 학습 구간에서는 이 구조가 얼마나 안정적으로 형성되는지 더 깊은 분석이 있으면 좋았을 것이다.

셋째, target-private classes를 모두 하나의 unknown으로 묶는 설정은 practical하지만, 서로 다른 target-private semantic을 구분하는 문제까지는 다루지 않는다. 따라서 실제 응용에서 unknown 이후의 세분화가 필요한 경우에는 추가 연구가 필요하다.

넷째, 본문에 제시된 benchmark는 2개이며, 모두 driving 또는 street-scene에 비교적 가까운 성격을 가진다. Pascal-Context가 일부 indoor/outdoor 다양성을 포함하긴 하지만, 의료영상이나 실내 scene parsing 등 전혀 다른 segmentation 영역으로도 일반화되는지는 이 논문만으로는 판단하기 어렵다. 논문도 그런 범용성까지 직접 실험으로 입증하지는 않는다.

다섯째, 논문은 qualitative/quantitative 결과를 통해 성능 향상을 보여주지만, 계산 비용이나 학습 복잡도 증가에 대한 자세한 논의는 제공된 텍스트 기준으로 충분히 상세하지 않다. prototype loss, matching 과정, 추가 샘플링 전략이 실제 training overhead를 얼마나 유발하는지는 명확히 드러나지 않는다.

## 6. 결론

이 논문은 semantic segmentation에서 source와 target의 클래스 구성이 미리 알려져 있지 않은 현실적 상황을 다루기 위해 **UniDA-SS**라는 새로운 문제를 제시하고, 이를 위한 프레임워크 **UniMAP**을 제안했다. 핵심 기여는 크게 세 가지로 요약할 수 있다. 첫째, Universal Domain Adaptation을 segmentation으로 확장한 문제 정의와 benchmark 제안이다. 둘째, source/target별 prototype을 이용해 common/private distinction을 강화하는 **DSPD**다. 셋째, target pseudo-label에 맞추어 common-class가 풍부한 source image를 선택하는 **TIM**이다.

실험 결과를 보면 UniMAP은 특히 Open Partial Domain Adaptation setting에서 strong baseline들을 일관되게 능가하며, common class segmentation과 target-private detection 사이의 균형을 잘 맞춘다. 이는 실제 환경에서 category setting이 미리 정해지지 않은 경우에 특히 중요하다.

실제 적용 측면에서도 이 연구는 의미가 있다. 자율주행처럼 새로운 환경으로 모델을 옮겨야 하는 상황에서는 domain shift뿐 아니라 label space mismatch도 흔하다. UniMAP은 바로 그런 상황에서 더 견고한 segmentation adaptation의 방향을 제시한다. 앞으로 이 연구는 semantic segmentation뿐 아니라 panoptic segmentation, instance segmentation, 또는 continual adaptation과 결합된 universal setting 연구로도 확장될 가능성이 있다. 제공된 텍스트 기준으로 볼 때, 이 논문은 segmentation adaptation을 더 현실적인 문제 설정으로 끌고 가는 초기이면서도 꽤 설득력 있는 시도라고 평가할 수 있다.
