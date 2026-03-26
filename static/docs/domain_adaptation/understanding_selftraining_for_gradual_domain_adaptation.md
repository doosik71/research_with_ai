# Understanding Self-Training for Gradual Domain Adaptation

- **저자**: Ananya Kumar, Tengyu Ma, Percy Liang
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2002.11361

## 1. 논문 개요

이 논문은 시간이 지나면서 데이터 분포가 조금씩 변하는 환경에서, 라벨이 없는 중간 도메인들만을 활용해 초기 분류기를 어떻게 안정적으로 적응시킬 수 있는지를 다룬다. 저자들은 이 문제를 **gradual domain adaptation**이라고 부른다. 출발점은 labeled source domain이고, 최종 목표는 unlabeled target domain에서 높은 정확도를 얻는 것이다. 다만 source와 target 사이에는 직접 점프하는 대신, 분포가 서서히 변하는 intermediate domains가 존재한다고 가정한다.

연구 문제는 명확하다. 일반적인 unsupervised domain adaptation은 source에서 target으로 한 번에 적응하려고 하는데, source와 target의 support가 거의 겹치지 않는 현대의 고차원 문제에서는 이 접근이 쉽게 실패할 수 있다. 반면 실제 응용에서는 변화가 대개 급격하지 않고 점진적이다. 예를 들어 센서 노화, 자율주행 환경 변화, brain-machine interface 신호 변화, 연도에 따른 인물 사진 스타일 변화처럼 분포 이동은 누적되지만 각 시간 간격의 변화량은 작을 수 있다. 이 논문은 바로 이 “점진성”이 이론적으로도, 실험적으로도 활용 가능한 구조인지 묻는다.

논문의 핵심 기여는 세 가지로 정리할 수 있다. 첫째, gradual shift를 이용한 self-training에 대해 처음으로 **non-vacuous upper bound**를 제시한다. 즉, 직접 target에 적응하면 오차가 무한정 커질 수 있는 상황에서도, 중간 도메인을 순차적으로 따라가면 오차를 제어할 수 있음을 보인다. 둘째, 이론 분석을 통해 단순한 구현 선택처럼 보였던 요소들, 예를 들어 **regularization**, **label sharpening**, 그리고 **ramp loss**가 사실상 필수적임을 설명한다. 셋째, 이러한 통찰이 실제 데이터셋에서도 유효함을 보여 주며, Rotating MNIST와 Portraits 데이터셋에서 direct target adaptation보다 유의미하게 높은 정확도를 달성한다.

이 논문이 중요한 이유는, gradual domain adaptation을 단순한 경험적 트릭이 아니라 이론적으로 해석 가능한 문제로 끌어올렸기 때문이다. 특히 기존 domain adaptation 이론이 잘 다루지 못했던 “source와 target의 지지집합이 멀리 떨어진 경우”에 대해, intermediate domains라는 현실적 구조를 활용하는 방법을 제시했다는 점이 의미 있다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음과 같다. source에서 target으로 한 번에 이동하면 현재 분류기가 target 데이터를 거의 전부 틀리게 pseudo-labeling할 수 있다. 이 경우 self-training은 잘못된 pseudo-label을 강화하는 악순환에 빠진다. 하지만 분포가 충분히 천천히 변한다면, 현재 분류기는 다음 시점의 데이터에 대해서는 아직 대부분 맞출 수 있다. 그러면 그 pseudo-label로 다시 학습한 새 분류기는 새 도메인에 적응한 더 좋은 분류기가 되고, 이 과정을 반복하면 최종 target까지 따라갈 수 있다.

즉, self-training의 성패는 “한 번에 얼마나 큰 shift를 건너뛰느냐”에 달려 있고, gradual domain adaptation은 이 큰 이동을 여러 개의 작은 이동으로 쪼개어 pseudo-labeling 오류가 폭발하지 않게 한다. 논문 Figure 2의 직관도 바로 이것이다. source classifier는 source에서는 완벽할 수 있지만, 몇 단계 뒤 target에서는 완전히 뒤집힌 예측을 할 수 있다. 그러나 중간 단계에서는 아직 꽤 정확하므로, 순차적인 self-training이 가능하다.

기존 접근과의 차별점은 두 층위에서 드러난다. 문제 설정 차원에서는, 많은 기존 domain adaptation 방법이 source와 target 사이의 direct adaptation만 고려하거나, density ratio가 잘 정의된다는 식의 겹침 가정을 둔다. 이 논문은 intermediate unlabeled domains가 있다는 정보를 활용한다. 알고리즘 차원에서는, 새로운 복잡한 적응 모듈을 설계하기보다 **가장 단순한 self-training**을 점진적 구조 위에 올려놓고 그 작동 조건을 분석한다. 즉, “왜 self-training이 gradual shift에서 작동할 수 있는가”를 설명하는 논문이다.

또한 저자들은 gradual shift가 모든 거리 개념에서 유효한 것은 아니라고 주장한다. 특히 이론과 실험 모두에서, 중요한 것은 total variation이나 KL divergence가 아니라 **class-conditional Wasserstein-infinity distance**가 작다는 점이다. 다시 말해 분포 전체 통계량이 조금 달라지는 것보다, 각 클래스의 샘플이 다음 시점으로 이동할 때 개별 포인트 수준에서 너무 멀리 이동하지 않는 것이 중요하다는 해석을 제공한다.

## 3. 상세 방법 설명

### 3.1 문제 설정

입력은 $x \in \mathbb{R}^d$, 레이블은 $y \in \{-1,1\}$인 이진 분류 문제를 다룬다. 시간에 따라 달라지는 joint distribution을 $P_0, P_1, \dots, P_T$로 두며, $P_0$는 source domain, $P_T$는 target domain, 나머지는 intermediate domains이다. source에서는 labeled data $S_0 = \{(x_i^{(0)}, y_i^{(0)})\}_{i=1}^{n_0}$가 주어지고, 각 $t \ge 1$에 대해서는 unlabeled data $S_t = \{x_i^{(t)}\}_{i=1}^n$만 주어진다.

모델 $M_\theta: \mathbb{R}^d \to \mathbb{R}$는 score를 출력하고, 최종 예측은 $\operatorname{sign}(M_\theta(x))$로 정의한다. 분류 오차는 다음과 같이 0-1 loss 기반으로 정의된다.

$$
\operatorname{Err}(\theta, P) = \mathbb{E}_{X,Y \sim P}\left[\operatorname{sign}(M_\theta(X)) \neq Y\right]
$$

목표는 target distribution $P_T$에서 낮은 오차를 가지는 classifier를 얻는 것이다.

### 3.2 source training과 self-training 정의

먼저 source labeled data에서 supervised learning으로 초기 모델 $\theta_0$를 학습한다.

$$
\theta_0 = \arg\min_{\theta' \in \Theta} \frac{1}{n_0} \sum_{(x_i,y_i) \in S_0} \ell(M_{\theta'}(x_i), y_i)
$$

여기서 $\ell$은 일반적인 supervised loss이다. 그 다음 self-training은 unlabeled data $S$에 대해 현재 모델 $\theta$가 생성한 pseudo-label $\operatorname{sign}(M_\theta(x_i))$를 정답처럼 사용해 새 모델을 학습한다.

$$
\operatorname{ST}(\theta, S) = \arg\min_{\theta' \in \Theta} \frac{1}{|S|} \sum_{x_i \in S} \ell(M_{\theta'}(x_i), \operatorname{sign}(M_\theta(x_i)))
$$

이 논문은 pseudo-label을 확률값으로 두지 않고, $-1$ 또는 $1$의 **hard label**로 두는 방식을 사용한다. 논문은 이를 **label sharpening**이라고 부르며, 이 선택이 단순한 구현 옵션이 아니라 이론적으로 필수에 가깝다고 주장한다.

직접 adaptation baseline은 $\theta_0$에서 시작해 target unlabeled set $S_T$만으로 self-training하는 것이다. 반면 gradual self-training은 각 intermediate domain을 순서대로 따라간다.

$$
\theta_i = \operatorname{ST}(\theta_{i-1}, S_i), \quad i \ge 1
$$

최종 출력은 $\theta_T$이다. 알고리즘 자체는 매우 단순하지만, 핵심은 각 step에서 shift가 작아 pseudo-label noise가 통제된다는 점이다.

### 3.3 margin setting의 이론적 분석

논문은 첫 번째 이론 분석으로 distribution-free에 가까운 **margin setting**을 다룬다. 여기서 모델 클래스는 norm이 제한된 regularized linear model이다.

$$
\Theta_R = \{(w,b): w \in \mathbb{R}^d, b \in \mathbb{R}, \|w\|_2 \le R\}
$$

모델 출력은 $M_{w,b}(x) = w^\top x + b$이다. 이때 regularization은 단순한 일반화 목적이 아니라, classifier가 일정한 **geometric margin** $\gamma = 1/R$를 갖게 해 다음 도메인에서도 예측을 유지하게 하는 핵심 장치다.

손실 함수로는 hinge loss와 ramp loss를 정의하지만, 실제 이론 보장은 ramp loss를 중심으로 전개된다. 힌지 함수와 ramp 함수는 다음과 같다.

$$
h(m) = \max(1-m, 0)
$$

$$
r(m) = \min(h(m), 1)
$$

ramp loss는 큰 오차를 낸 outlier가 손실을 무한정 키우지 못하도록 상한 1을 둔 손실이다. 논문은 self-training 이론에서 이 boundedness가 결정적으로 중요하다고 설명한다.

도메인 간 거리로는 class-conditional **Wasserstein-infinity distance**를 사용한다. 단순히 전체 분포 사이의 KL divergence나 total variation을 보는 것이 아니라, 각 클래스 조건부 분포가 얼마나 멀리 이동했는지를 측정한다.

$$
\rho(P,Q) = \max\Big(W_\infty(P_{X\mid Y=1}, Q_{X\mid Y=1}), \; W_\infty(P_{X\mid Y=-1}, Q_{X\mid Y=-1})\Big)
$$

이 정의의 직관은, 각 클래스의 포인트들이 한 step 동안 최대 얼마만큼 이동했는지를 본다는 것이다. shift가 작다는 조건은 $\rho(P_t, P_{t+1}) \le \rho < 1/R$ 형태로 둔다. 즉, 한 step의 이동량이 현재 모델이 확보한 margin보다 작아야 한다.

추가 가정으로는 각 도메인마다 낮은 ramp loss를 갖는 어떤 선형 분류기가 존재한다는 **$\alpha^*$-separation assumption**, 데이터의 second moment가 제한된다는 bounded data assumption, 그리고 시간에 따라 클래스 비율이 바뀌지 않는 no label shift assumption을 둔다.

### 3.4 왜 baseline은 실패하는가

논문은 먼저 source classifier가 source에서는 100% 정확하더라도 target에서는 0% 정확도가 될 수 있음을 예제로 보인다. 더 중요한 점은, 그런 상황에서는 target에 직접 self-training해도 pseudo-label이 전부 틀리기 때문에 전혀 복구되지 않는다는 것이다. Example 3.1은 intermediate shift는 작지만 source에서 target까지 누적 shift는 커져서, direct adaptation이 실패할 수 있음을 구성적으로 보여 준다.

이 부분은 논문의 메시지를 선명하게 만든다. gradual shift가 없으면 self-training은 단지 현재 classifier의 실수를 재생산할 뿐이지만, gradual structure가 있으면 그 실수율을 각 단계에서 제어할 수 있다.

### 3.5 핵심 정리: gradual self-training의 오차 상계

핵심 결과인 Theorem 3.2는, 한 step의 shift가 작고 현재 모델의 ramp loss가 작다면, 다음 도메인에 대해 self-training한 모델의 ramp loss도 통제 가능하다고 말한다. 정리의 형태는 다음과 같다.

$$
L_r(\theta', Q) \le \frac{2}{1-\rho R} L_r(\theta, P) + \alpha^* + \frac{4BR + \sqrt{2\log(2/\delta)}}{\sqrt{n}}
$$

여기서 $\theta' = \operatorname{ST}(\theta, S)$이고, $S$는 $Q$에서 샘플된 unlabeled sample이다. 이 식은 세 부분으로 해석할 수 있다. 첫 번째 항은 이전 도메인에서의 오차가 shift를 거쳐 증폭되는 양이고, 두 번째 항 $\alpha^*$는 새로운 도메인 자체의 난이도, 세 번째 항은 finite sample로 인한 통계 오차다.

증명 아이디어는 세 단계다. 첫째, shift가 margin보다 작으면 현재 classifier는 새 도메인에서도 많은 샘플을 여전히 올바르게 pseudo-labeling한다. 둘째, pseudo-label 오류율이 작다면, 새 도메인에서 낮은 ramp loss를 갖는 좋은 classifier도 pseudo-labeled distribution에 대해 여전히 낮은 손실을 가진다. 셋째, self-training은 바로 그 pseudo-labeled distribution 위에서 empirical risk minimization을 수행하므로, 결국 새 모델도 낮은 ramp loss를 갖게 된다.

이 정리를 반복 적용하면 Corollary 3.3이 나온다. source 모델의 초기 손실이 $\alpha_0$일 때, $T$ step 후 target에서의 ramp loss는 대략 다음처럼 상계된다.

$$
L_r(\theta, P_T) \le \beta^{T+1} \left( \alpha_0 + \frac{4BR + \sqrt{2\log(2T/\delta)}}{\sqrt{n}} \right), \quad \beta = \frac{2}{1-\rho R}
$$

즉, 오차가 step마다 제어되기는 하지만 일반적으로는 **지수적으로 증가할 수 있다**. 이 점은 논문의 태도가 매우 정직한 부분이다. gradual self-training이 항상 완벽히 안정적이라고 주장하지 않고, direct adaptation이 아예 망가질 수 있는 상황에서라도 적어도 의미 있는 bound를 줄 수 있다고 말한다.

### 3.6 bound의 tightness와 필수 요소들

저자들은 Example 3.4를 통해 위 지수 bound가 단순한 proof artifact가 아니라 어느 정도 tight할 수 있다고 보인다. 즉, 초기 오차 $\alpha_0$가 있으면, gradual self-training을 해도 오차가 step마다 상수배씩 누적되어 실제로 지수적으로 증가할 수 있다. 따라서 더 강한 bound를 얻으려면 데이터 분포에 대한 추가 구조가 필요하다.

이 때문에 논문은 Section 4에서 더 강한 Gaussian setting을 도입한다.

그전에 Section 3.4는 왜 **regularization**, **label sharpening**, **ramp loss**가 필수인지 설명한다.

첫째, regularization이 없으면 self-training이 모델을 바꿀 유인이 없어진다. 현재 모델이 만든 pseudo-label은 현재 모델이 이미 완벽히 맞히므로, 같은 decision boundary를 유지한 채 scale만 키운 모델이 최적해가 될 수 있다. 즉, parameter update가 사실상 일어나지 않는다. 이 논점은 “왜 self-training에 regularization이 필요한가”를 매우 설득력 있게 보여 준다.

둘째, soft label을 쓰면 현재 모델이 자기 자신의 soft prediction distribution을 이미 가장 잘 설명하므로, 역시 업데이트가 일어나지 않을 수 있다. 논문은 logistic loss 기반 soft pseudo-label objective에서 원래 모델 자체가 항상 minimizer가 될 수 있다는 Example 3.6을 제시한다. 이것이 label sharpening이 필요한 이유다.

셋째, hinge loss는 실제 최적화에서는 편하지만, 이론 분석에는 충분히 robust하지 않다. Example 3.7은 hinge loss를 쓰면 gradual shift 조건이 있어도 self-training 결과가 완전히 틀어질 수 있음을 보인다. 따라서 bounded loss인 ramp loss가 이론상 핵심 도구가 된다.

### 3.7 no shift setting과 Gaussian setting

논문은 분포 이동이 아예 없는 경우에는 self-training의 오차 증가가 지수가 아니라 선형적으로만 증가한다고 보인다. 이는 분포 이동이 실제로 오차 폭증의 핵심 원인임을 뒷받침한다.

그다음 Gaussian setting에서는 더 강한 구조를 가정한다. 각 클래스 조건부 분포가 isotropic Gaussian이라고 두고,

$$
P_t(X \mid Y=y) = \mathcal{N}(y\mu_t, \sigma_t^2 I)
$$

형태를 가정한다. 이 setting에서는 unlabeled objective를

$$
U(w,P) = \mathbb{E}_{X \sim P}[\phi(|w^\top X|)]
$$

로 두고, 이전 step의 해 $w_{t-1}$ 근처에서 이 objective를 최소화하는 constrained update를 분석한다. 핵심은 unlabeled objective가 전역적으로는 non-convex여도, 좋은 해 주변의 국소 영역에서는 올바른 Bayes-optimal 방향 $w^*(\mu_t)$가 유일한 local minimizer라는 점이다.

Theorem 4.1은 초기 분류기 $w_0$가 source의 Bayes-optimal direction $w^*(\mu_0)$에 충분히 가까우면, gradual self-training이 각 step에서 최적 방향을 계속 따라가 결국 $w_T = w^*(\mu_T)$를 회복한다고 말한다. 이는 margin setting에서의 지수 bound를 넘어, 더 강한 분포 가정 아래에서는 self-training이 거의 이상적으로 작동할 수 있음을 보여 주는 결과다.

## 4. 실험 및 결과

### 4.1 실험 설정

저자들은 세 가지 데이터셋에서 gradual self-training을 평가한다.

첫 번째는 **Gaussian synthetic dataset**이다. 차원 $d=100$에서 두 클래스의 mean과 covariance가 시간에 따라 변한다. source에서는 500개의 labeled sample, intermediate domains 전체에 걸쳐 5000개의 unlabeled sample을 사용한다. 모델은 $\ell_2$ regularization을 둔 logistic regression이다.

두 번째는 **Rotating MNIST**이다. MNIST 숫자 이미지를 0도에서 60도 사이로 회전시켜 도메인 이동을 인위적으로 만든다. source는 0도에서 5도 사이, target은 55도에서 60도 사이, intermediate domains는 그 사이 각도를 갖는다. 이 데이터셋은 gradual shift가 시각적으로 매우 직관적이다.

세 번째는 **Portraits** 데이터셋이다. 미국 고등학교 졸업사진을 연도순으로 나열한 실제 데이터셋이며, gender classification을 수행한다. source는 초기 2000장, intermediate는 그다음 14000장, target은 다음 2000장이다. 이 데이터셋은 실제로 시간이 지남에 따라 사진 스타일, 헤어스타일, 조명, 구도 등이 바뀌는 현실적인 gradual shift를 담는다.

MNIST와 Portraits에는 dropout 0.5와 마지막 layer batch normalization을 포함한 3-layer convolutional network를 사용했고, source held-out examples에서는 약 97%에서 98% 정도의 정확도를 얻었다. self-training 시에는 각 step마다 confidence가 가장 낮은 10% 샘플을 버리고 나머지에 대해서 pseudo-label을 사용한다.

비교 baseline은 네 가지다. **Source**는 adaptation 없이 source classifier를 그대로 target에 적용한다. **Target self-train**은 intermediate domains를 무시하고 target에서만 반복 self-training한다. **All ST**는 intermediate와 target의 unlabeled 데이터를 한데 모아 pooled self-training을 한다. **Gradual ST**는 intermediate domains를 순서대로 따라가며 self-training한다.

### 4.2 gradual shift 구조가 실제로 도움을 주는가

가장 핵심적인 결과는 Table 1이다. 각 데이터셋에서 target 정확도는 다음과 같다.

Gaussian에서는 Source가 $47.7 \pm 0.3\%$, Target ST가 $49.6 \pm 0.0\%$, All ST가 $92.5 \pm 0.1\%$, Gradual ST가 $98.8 \pm 0.0\%$이다. 즉 gradual structure를 순차적으로 활용하면 거의 완벽한 성능에 도달한다.

Rotating MNIST에서는 Source가 $31.9 \pm 1.7\%$, Target ST가 $33.0 \pm 2.2\%$, All ST가 $38.0 \pm 1.6\%$, Gradual ST가 $87.9 \pm 1.2\%$이다. 이 결과는 매우 극적이다. intermediate unlabeled data를 단순히 모으는 것만으로는 충분하지 않고, 그 **순서 정보**를 활용해야 함을 보여 준다.

Portraits에서는 Source가 $75.3 \pm 1.6\%$, Target ST가 $76.9 \pm 2.1\%$, All ST가 $78.9 \pm 3.0\%$, Gradual ST가 $83.8 \pm 0.8\%$이다. 실제 데이터셋에서도 gradual self-training이 가장 좋으며, source 대비 약 8.5%p 정도 개선된다.

이 결과의 의미는 분명하다. 단순히 더 많은 unlabeled data를 쓰는 것이 아니라, **도메인 변화의 순차적 구조 자체**가 성능 개선에 핵심이라는 것이다. 특히 Rotating MNIST처럼 source와 target이 멀리 떨어진 경우, target에 직접 self-training하는 것은 거의 도움이 되지 않지만 gradual ST는 강력하게 작동한다.

### 4.3 regularization과 hard labeling의 중요성

Table 2는 이론에서 강조한 요소들이 실제 deep network에서도 중요한지를 검증한다.

Soft Labels의 경우 Gaussian $90.5 \pm 1.9\%$, Rotating MNIST $44.1 \pm 2.3\%$, Portraits $80.1 \pm 1.8\%$다. No Reg의 경우 Gaussian $84.6 \pm 1.1\%$, Rotating MNIST $45.8 \pm 2.5\%$, Portraits $76.5 \pm 1.0\%$다. 반면 정규화와 hard labels를 모두 사용하는 Gradual ST는 Gaussian $99.3 \pm 0.0\%$, Rotating MNIST $83.8 \pm 2.5\%$, Portraits $82.6 \pm 0.8\%$를 얻는다.

해석은 명확하다. implicit regularization만으로는 부족하며, explicit regularization과 hard pseudo-labeling이 함께 있어야 점진적 adaptation이 크게 향상된다. 특히 Rotating MNIST에서 No Reg와 Gradual ST 사이 격차가 매우 커서, 이론적 주장과 실험적 관찰이 잘 맞는다.

### 4.4 데이터가 많아져도 regularization은 여전히 중요한가

논문은 supervised learning과 다른 점을 강조하기 위해 sample size를 늘린 Rotating MNIST 실험도 수행한다. 같은 이미지들을 여러 각도로 회전시키므로, 이 실험은 unseen sample generalization보다 “same sample의 shifted view 적응”에 가깝다.

Table 3에 따르면, $N=2000$일 때 Source $28.3 \pm 1.4\%$, No Reg $55.7 \pm 3.9\%$, Reg $93.1 \pm 0.8\%$이고, $N=5000$일 때 Source $29.9 \pm 2.5\%$, No Reg $53.6 \pm 4.0\%$, Reg $91.7 \pm 2.4\%$이며, $N=20{,}000$일 때 Source $33.9 \pm 2.6\%$, No Reg $55.1 \pm 3.9\%$, Reg $87.4 \pm 3.1\%$다.

즉 데이터가 훨씬 많아져도 regularized gradual self-training과 unregularized version 사이 성능 격차가 거의 줄지 않는다. 이는 논문의 중요한 메시지다. supervised learning에서는 데이터가 많아질수록 explicit regularization의 상대적 중요성이 줄어드는 경우가 많지만, gradual domain adaptation에서는 regularization이 “generalization을 위한 장식”이 아니라, **shift를 견디는 margin 유지 메커니즘**이기 때문에 여전히 중요하다.

### 4.5 언제 gradual shift가 도움이 되지 않는가

Section 5.3은 논문의 균형감을 보여 준다. 저자들은 Rotating MNIST에서 intermediate domain을 다르게 구성해 본다. source와 target은 그대로 두되, intermediate domains를 “조금씩 더 많은 target-style 이미지가 섞이게 하는 방식”으로 만든다. 이 경우 successive domains 사이 total variation은 작지만, 각 이미지가 실제로는 약 55도 정도의 큰 회전을 겪는 셈이라 Wasserstein-infinity는 크다.

결과적으로 gradual self-training은 target 직접 self-training보다 거의 낫지 않다. Gradual ST가 $33.5 \pm 1.5\%$, direct target adaptation이 $33.0 \pm 2.2\%$다. 이 결과는 논문의 이론적 주장, 즉 gradual structure가 유용하려면 각 step의 이동이 “pointwise하게” 작아야 한다는 해석을 실험적으로 뒷받침한다.

### 4.6 추가 ablation

confidence thresholding을 제거한 Table 4에서도 gradual ST는 여전히 가장 좋다. Rotating MNIST에서 Source $30.5 \pm 1.0\%$, Target ST $31.1 \pm 1.4\%$, All ST $32.6 \pm 1.3\%$, Gradual ST $80.3 \pm 1.4\%$다. Portraits에서도 Source $76.2 \pm 0.5\%$, Target ST $76.9 \pm 1.3\%$, All ST $77.1 \pm 0.5\%$, Gradual ST $81.7 \pm 1.3\%$다.

window size를 줄인 Table 5에서도 gradual ST의 우위는 유지된다. Rotating MNIST에서 $90.4 \pm 2.0\%$, Portraits에서 $83.8 \pm 0.5\%$로 강한 성능을 보인다. 추가 epoch를 더 주거나 Portraits에서 더 먼 미래까지 extrapolation한 실험에서도 gradual ST가 상대적으로는 가장 낫지만, extrapolation이 너무 멀어지면 모든 방법이 크게 어려워진다. 이는 gradual self-training의 유효 범위가 무한하지 않음을 보여 준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은, gradual domain adaptation에 대해 단순히 “잘 된다”는 경험적 주장에 머무르지 않고, self-training이 왜 작동하고 언제 실패하는지에 대한 정교한 이론을 제시했다는 점이다. 특히 source와 target이 멀리 떨어져 direct adaptation 이론이 취약한 상황에서도, intermediate domains라는 현실적 구조를 통해 오차를 통제할 수 있음을 보인 점이 새롭다.

두 번째 강점은 이론과 실험의 연결이 좋다는 점이다. regularization, label sharpening, ramp loss의 필요성이 단순한 수학적 편의가 아니라 실제 성능 차이로 이어진다는 것을 실험으로 확인했다. 이론 결과가 모델 선택과 pseudo-label 설계 원칙으로 이어진다는 점에서 실용적 가치가 크다.

세 번째 강점은 “gradual shift”의 의미를 단순한 직관이 아니라 구체적인 거리 개념으로 해석했다는 데 있다. 특히 Wasserstein-infinity가 중요하다는 해석은, 어떤 intermediate curriculum이 유효하고 어떤 것은 그렇지 않은지를 가르는 개념적 기준을 제공한다.

반면 한계도 분명하다. 먼저 margin setting의 주요 bound는 본질적으로 지수적이다. 저자들도 이를 숨기지 않고 tight example까지 제시한다. 따라서 많은 step이 누적되는 장기 적응 상황에서는 여전히 error accumulation 문제가 남는다. 이 논문은 gradual self-training이 direct adaptation보다 낫다는 것을 보여 주지만, 완전히 안정적인 해법을 제시한 것은 아니다.

또한 이론 가정은 현실을 단순화한다. no label shift assumption을 두지만, 실제 Portraits 데이터셋에서는 label proportion이 변한다고 논문 자체가 인정한다. 그럼에도 실험은 잘 되지만, 이론과 실제 데이터 조건 사이에는 틈이 있다. 마찬가지로 Gaussian setting은 매우 이상화된 구조이며, 실제 deep network의 비선형 표현 학습을 그대로 설명하지는 못한다.

self-training의 optimization dynamics에 대한 분석도 완결되어 있지 않다. Gaussian setting에서 저자들은 self-training이 특정 unlabeled objective를 감소시키는 방향으로 움직인다는 prior work를 활용하지만, 실제로 제안한 constrained minimization 해에 수렴하는지까지는 보이지 못한다. 즉, 통계적 분석은 강하지만 optimization 분석은 아직 열려 있다.

또 하나의 실질적 한계는 intermediate domains가 필요하다는 점이다. 실제 응용에서는 시간 순서로 unlabeled data가 축적되는 경우가 많지만, 항상 domain boundaries가 명확하거나 균일한 window로 나뉘는 것은 아니다. intermediate segmentation을 어떻게 정할지, 얼마나 촘촘해야 하는지, shift magnitude를 사전에 어떻게 추정할지는 여전히 실무적 난제다.

비판적으로 보면, 이 논문은 self-training이라는 매우 단순한 알고리즘을 분석 대상으로 삼았기 때문에 강점과 동시에 제한도 갖는다. representation alignment, feature invariance, memory, temporal smoothing 같은 더 고급 기법과의 결합 가능성은 거의 다루지 않는다. 따라서 이 논문은 “최종 해법”이라기보다 gradual domain adaptation 연구의 이론적 출발점에 가깝다.

## 6. 결론

이 논문은 gradual domain adaptation이라는 문제를 명확히 정식화하고, 순차적 self-training이 direct target adaptation보다 본질적으로 유리할 수 있음을 처음으로 이론적으로 보였다. 특히 작은 class-conditional Wasserstein shift와 margin 구조가 있을 때 self-training의 오류를 단계적으로 제어할 수 있으며, 더 강한 Gaussian 가정 아래에서는 Bayes-optimal classifier를 추적할 수 있다는 결과를 제시했다.

실험적으로도 gradual self-training은 Gaussian synthetic data, Rotating MNIST, Portraits에서 일관되게 strongest baseline이었다. 특히 target에 직접 self-training하거나 unlabeled 데이터를 그냥 합쳐 쓰는 것보다, intermediate domains를 순서대로 따라가는 것이 훨씬 효과적이라는 점이 반복적으로 확인되었다.

실제 적용 측면에서 이 연구는, 시간이 따라 변하는 데이터 스트림에서 라벨을 계속 수집하기 어려운 상황에 유용한 가이드라인을 제공한다. intermediate unlabeled data가 존재하고 각 step의 shift가 충분히 작다면, regularized hard-label self-training만으로도 상당한 적응 효과를 낼 수 있다. 향후 연구로는 label shift를 포함한 더 현실적인 이론, deep nonlinear model에 대한 직접 분석, 더 안정적인 alternative algorithm, intermediate domain scheduling 전략 등이 중요할 것이다.

## 출처

이 보고서는 사용자가 제공한 arXiv 논문 추출 텍스트를 바탕으로 작성되었다. 논문 정보는 본문 상단의 제목, 저자, arXiv 식별자 `[2002.11361]`와 논문 본문에 포함된 내용을 기준으로 정리했다.
