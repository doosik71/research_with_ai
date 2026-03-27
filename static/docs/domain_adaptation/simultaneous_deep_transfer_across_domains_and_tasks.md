# Simultaneous Deep Transfer Across Domains and Tasks

* **저자**: Eric Tzeng, Judy Hoffman, Trevor Darrell, Kate Saenko
* **발표연도**: 2015
* **arXiv**: [https://arxiv.org/abs/1510.02192](https://arxiv.org/abs/1510.02192)

## 1. 논문 개요

이 논문은 **domain adaptation** 문제를 다룬다. 구체적으로는, 잘 라벨링된 source domain에서 학습한 CNN 분류 모델을, 라벨이 거의 없거나 아예 없는 target domain에 효과적으로 적응시키는 방법을 제안한다. 저자들은 단순히 source 데이터만으로 학습한 모델이 target 환경에서 성능이 떨어지는 이유를 **dataset bias** 또는 **domain shift**로 설명한다. 즉, 같은 클래스라도 source와 target의 이미지 분포가 다르면, source에서 잘 작동한 분류기가 target에서는 오동작할 수 있다.

논문의 핵심 문제는 두 가지다. 첫째, target domain에는 충분한 labeled data가 없기 때문에 일반적인 fine-tuning이 어렵다. 둘째, target domain에 일부 클래스만 소량으로 라벨이 존재하는 **semi-supervised adaptation** 상황에서는, 라벨이 없는 클래스까지 잘 일반화해야 한다. 기존 방법들은 주로 domain 간 분포 차이를 줄이는 데 집중했지만, source에서 학습된 **클래스 간 관계 구조**까지 target으로 전달하는 데는 한계가 있었다.

이 문제는 실제 응용에서 매우 중요하다. 논문 서두의 예시처럼, 제조사가 학습시킨 로봇이 각 가정의 서로 다른 환경에 배치되면 조명, 배경, 촬영 각도, 물체 외형 등의 차이로 인해 성능이 저하될 수 있다. 그런데 새로운 환경마다 대량의 라벨을 수집하는 것은 비현실적이다. 따라서 **적은 target 라벨과 많은 unlabeled target 데이터만으로도 적응 가능한 방법**은 매우 실용적인 가치를 가진다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **domain transfer**와 **task transfer**를 하나의 CNN 안에서 동시에 수행하자는 것이다. 저자들은 target adaptation을 단순히 분포 정렬 문제로만 보지 않고, source에서 학습된 클래스 간 의미적 관계도 함께 옮겨야 한다고 본다. 이를 위해 두 개의 보조 목적을 classification loss에 추가한다.

첫 번째 아이디어는 **domain confusion**이다. 이는 source와 target의 feature representation이 너무 구별되지 않도록 학습해서, feature space 상에서 두 도메인의 **marginal distribution**을 가깝게 만드는 것이다. 직관적으로는, 어떤 representation이 정말 domain-invariant하다면 그 representation만 보고는 샘플이 source에서 왔는지 target에서 왔는지 맞히기 어려워야 한다. 이를 위해 논문은 별도의 domain classifier를 두고, representation은 그 classifier를 헷갈리게 만들도록 학습한다.

두 번째 아이디어는 **soft label transfer**이다. domain confusion만으로는 두 도메인의 전체 분포는 비슷하게 만들 수 있어도, 클래스 정렬이 보장되지는 않는다. 예를 들어 source의 bottle feature가 target의 mug 근처로 가는 식의 잘못된 정렬이 일어날 수 있다. 이를 막기 위해 저자들은 source에서 각 클래스의 평균 softmax 분포를 계산하여 **soft label**로 사용한다. 그러면 target의 소량 라벨 샘플은 단순한 one-hot hard label이 아니라, “이 클래스는 다른 어떤 클래스와 얼마나 비슷한가”라는 분포적 정보를 함께 가지게 된다. 예를 들어 bottle은 mug와는 비슷하지만 keyboard와는 덜 비슷하다는 관계가 soft label에 담긴다.

기존 접근과의 차별점은 바로 이 결합이다. 기존 CNN 기반 adaptation 방법도 domain-invariant representation을 시도했지만, **source semantic structure**, 즉 클래스 간 유사도 구조를 target에 직접 전달하는 방식은 이 논문이 강조하는 차별점이다. 특히 target에 일부 클래스만 라벨이 있는 상황에서, soft label은 라벨이 없는 클래스의 classifier 파라미터도 간접적으로 업데이트되도록 만든다는 점에서 의미가 있다.

## 3. 상세 방법 설명

논문의 전체 모델은 기본적으로 Krizhevsky의 CaffeNet/AlexNet 계열 구조를 사용한다. 구체적으로 conv1–conv5와 fc6–fc8로 구성되며, 저자들은 conv1–fc7을 representation $f(x; \theta_{\text{repr}})$로 보고, 마지막 fc8을 분류기 $\theta_C$로 둔다. 여기에 추가로 domain classifier 층 $\theta_D$를 붙여 source/target 구분을 수행한다.

기본 분류 목적은 일반적인 softmax cross-entropy이다. 클래스 수를 $K$라고 하면, 분류 loss는 다음과 같다.

$$
\mathcal{L}_C(x,y;\theta_{\text{repr}},\theta_C) = -\sum_k \mathbb{1}[y=k]\log p_k
$$

여기서 $p$는 분류기 출력의 softmax이며,

$$
p = \text{softmax}(\theta_C^T f(x;\theta_{\text{repr}}))
$$

이다. 이것만 사용하면 source 데이터에 잘 맞는 representation은 얻을 수 있지만, target에서의 domain shift를 직접 해결하지 못해 source overfitting이 생길 수 있다.

### 3.1 Domain confusion loss

저자들은 representation이 domain-invariant하려면, 그 representation만으로는 source와 target을 잘 구분할 수 없어야 한다고 본다. 이를 위해 domain classifier를 따로 두고, domain label $y_D$에 대해 다음의 domain classification loss를 정의한다.

$$
\mathcal{L}_D(x_S, x_T, \theta_{\text{repr}};\theta_D) = -\sum_d \mathbb{1}[y_D=d]\log q_d
$$

여기서 $q$는 domain classifier의 softmax 출력이며,

$$
q = \text{softmax}(\theta_D^T f(x;\theta_{\text{repr}}))
$$

이다. 이 식은 domain classifier 입장에서는 source와 target을 최대한 잘 구분하도록 학습시키는 목적이다.

반면 representation 쪽은 그 반대로, domain classifier가 무엇을 보든 **균등분포처럼** 예측하도록 만들고 싶다. 그래서 저자들은 다음의 domain confusion loss를 사용한다.

$$
\mathcal{L}_{\text{conf}}(x_S, x_T, \theta_D;\theta_{\text{repr}}) = -\sum_d \frac{1}{D}\log q_d
$$

여기서 $D$는 domain 개수이며, 이 논문에서는 source와 target 두 개이므로 사실상 domain 예측이 50:50에 가깝게 되게 만드는 목적이다. 쉽게 말하면, domain classifier가 확신을 가지지 못하도록 representation을 조정하는 것이다.

흥미로운 점은 $\mathcal{L}_D$와 $\mathcal{L}_{\text{conf}}$가 서로 반대 방향을 가진다는 것이다. domain classifier는 구분을 잘하고 싶고, representation은 구분을 어렵게 만들고 싶다. 논문은 이를 한 번에 푸는 대신, 다음 두 단계를 번갈아 업데이트한다.

첫째, representation을 고정하고 domain classifier 파라미터 $\theta_D$를 업데이트해서 $\mathcal{L}_D$를 최소화한다.

$$
\min_{\theta_D} \mathcal{L}_D(x_S, x_T, \theta_{\text{repr}}; \theta_D)
$$

둘째, domain classifier를 고정하고 representation 파라미터 $\theta_{\text{repr}}$를 업데이트해서 $\mathcal{L}_{\text{conf}}$를 최소화한다.

$$
\min_{\theta_{\text{repr}}} \mathcal{L}_{\text{conf}}(x_S, x_T, \theta_D; \theta_{\text{repr}})
$$

이 구조는 후대의 adversarial domain adaptation과 매우 유사한 발상으로 볼 수 있다. 다만 이 논문은 gradient reversal layer 대신, 명시적이고 반복적인 두 단계 최적화를 설명한다.

### 3.2 Soft label loss

domain confusion은 두 도메인의 **주변 분포**를 맞추지만, 클래스 수준 정렬을 보장하지는 않는다. 저자들은 이를 보완하기 위해 source에서 클래스별 **평균 출력 분포**를 만든다. 각 클래스 $k$에 대해 source training example들의 softmax 출력을 평균낸 벡터를 $l^{(k)}$라고 둔다. 이 벡터는 단순한 one-hot이 아니라, 그 클래스가 다른 클래스들과 얼마나 유사한지를 담은 분포다.

하지만 보통 softmax는 너무 뾰족해서 정보가 잘 드러나지 않는다. 그래서 저자들은 **temperature** $\tau$를 높인 softmax를 사용한다. temperature를 높이면 확률 분포가 더 부드러워져, 유사한 클래스들에도 확률 질량이 분산된다. 이 부분은 model distillation에서 차용한 아이디어다.

그 다음 target의 labeled sample $(x_T, y_T)$에 대해, 현재 모델이 낸 soft activation이 source에서 계산한 해당 클래스의 soft label과 일치하도록 학습한다. loss는 다음과 같다.

$$
\mathcal{L}_{\text{soft}}(x_T, y_T; \theta_{\text{repr}}, \theta_C) = -\sum_i l_i^{(y_T)} \log p_i
$$

여기서 target sample의 soft activation $p$는

$$
p = \text{softmax}(\theta_C^T f(x_T; \theta_{\text{repr}})/\tau)
$$

이다. 즉, target sample이 정답 클래스만 맞히는 것이 아니라, source에서 그 클래스가 보였던 **전체 분포 패턴**을 재현하도록 하는 것이다.

이 loss의 중요한 장점은, target 라벨이 없는 클래스에 대해서도 모델 출력이 완전히 죽지 않게 만든다는 점이다. 예를 들어 target에서 monitor 클래스 라벨이 하나도 없더라도, laptop 클래스의 soft label이 monitor에 어느 정도 질량을 주고 있으면, laptop의 labeled target sample을 통해 monitor 관련 파라미터도 간접적으로 갱신된다. 논문은 이를 통해 **across tasks** 전이가 가능하다고 주장한다.

### 3.3 최종 목적 함수와 학습 절차

최종적으로 논문은 세 개의 loss를 합친 joint objective를 사용한다.

$$
\begin{aligned}
\mathcal{L}(x_S,y_S,x_T,y_T,\theta_D;\theta_{\text{repr}},\theta_C) = &\ \mathcal{L}_C(x_S,y_S,x_T,y_T;\theta_{\text{repr}},\theta_C) \\
&+ \lambda \mathcal{L}_{\text{conf}}(x_S,x_T,\theta_D;\theta_{\text{repr}}) \\
&+ \nu \mathcal{L}_{\text{soft}}(x_T,y_T;\theta_{\text{repr}},\theta_C)
\end{aligned}
$$

여기서 $\lambda$와 $\nu$는 각각 domain confusion과 soft label loss의 중요도를 조절하는 하이퍼파라미터다. 논문 실험에서는 $\lambda = 0.01$, $\nu = 0.1$을 사용했다.

학습 절차를 정리하면 다음과 같다. 먼저 ImageNet으로 사전학습된 CaffeNet을 기반으로 source labeled data로 추가 fine-tuning하여 source CNN을 만든다. 이 source CNN으로부터 클래스별 soft label을 계산한다. 그 다음 이 가중치를 초기값으로 사용하여, source/target classification loss, domain confusion loss, soft label loss를 함께 최적화한다. target 데이터는 labeled와 unlabeled를 모두 활용하지만, soft label loss는 labeled target 샘플에만 적용된다.

## 4. 실험 및 결과

논문은 두 가지 벤치마크에서 실험한다. 첫째는 고전적인 **Office dataset**, 둘째는 보다 큰 domain gap을 가진 **cross-dataset testbed**이다.

### 4.1 Office dataset

Office dataset은 Amazon, DSLR, Webcam 세 도메인으로 구성되며, 총 31개 office object category를 포함한다. 저자들은 두 가지 설정을 실험한다.

첫 번째는 **supervised adaptation**이다. 이 경우 source에는 모든 클래스의 labeled data가 있고, target에도 각 클래스마다 아주 적은 labeled sample이 존재한다. 논문에서는 target domain의 각 클래스당 3개의 labeled example을 사용한다.

두 번째는 **semi-supervised adaptation** 또는 논문 표현대로 **task adaptation**이다. 이 경우 source는 모든 클래스가 라벨되어 있지만, target은 31개 클래스 중 15개 클래스에 대해서만 각 10개의 labeled example이 있고, 나머지 16개 클래스는 target labeled data가 전혀 없다. 평가는 이 16개 held-out class에 대해서만 수행한다. 즉, 진짜로 라벨 없는 target class에 얼마나 잘 일반화되는지를 보는 설정이다.

#### Supervised adaptation 결과

Table 1에 따르면, baseline인 Source CNN의 평균 multi-class accuracy는 66.22이고, Target CNN은 74.05, Source+Target CNN은 81.50이다. 제안 방법은 다음과 같은 평균 성능을 보인다.

* domain confusion only: 82.13
* soft labels only: 82.17
* domain confusion + soft labels: 82.22

즉, supervised setting에서는 Source+Target CNN도 이미 강한 baseline이지만, 저자들의 추가 regularization이 평균적으로 소폭 더 좋은 성능을 낸다. 특히 6개 domain shift 중 5개에서 soft label 또는 domain confusion이 hard label fine-tuning보다 일관된 개선을 보였다고 서술한다. 다만 수치 차이는 전반적으로 크지 않다. 이는 target의 모든 클래스에 최소한의 labeled sample이 존재하는 상황에서는, 단순 joint fine-tuning도 상당히 효과적이기 때문으로 해석할 수 있다.

#### Semi-supervised adaptation 결과

Table 2가 이 논문의 진짜 강점을 보여준다. Source CNN의 평균 성능은 62.0이고, 제안 방법은 다음과 같다.

* domain confusion only: 64.8
* soft labels only: 64.8
* domain confusion + soft labels: 66.4

여기서는 hard-label fine-tuning baseline보다, 저자들이 제안한 두 메커니즘이 훨씬 더 중요해진다. 특히 held-out category에 대해 평가하므로, 모델은 분류 loss만으로는 unlabeled target class를 직접 맞추도록 학습할 수 없다. 이런 상황에서 domain confusion은 도메인 간 representation alignment를, soft label은 클래스 간 관계 전이를 담당하면서 실질적 성능 향상을 만든다.

논문은 Amazon$\rightarrow$Webcam 예시에서, baseline이 notebooks를 letter trays로, black mug를 black mouse로 잘못 분류하던 사례를 제안 방법이 바로잡았다고 설명한다. 이는 feature distribution 자체를 맞추는 것뿐 아니라, 클래스 간 유사도 구조가 보존되었기 때문이라는 해석과 연결된다.

### 4.2 Cross-dataset adaptation

저자들은 더 큰 규모의 domain gap을 보기 위해 cross-dataset analysis testbed도 사용한다. 여기서는 40개 공통 클래스를 공유하는 ImageNet, Caltech-256, SUN, Bing 중에서 **ImageNet을 source**, **Caltech-256을 target**으로 설정한다.

프로토콜은 각 split마다 ImageNet 5534장, Caltech-256 4366장을 사용하고, target training set에서 클래스당 labeled example 수를 1개, 3개, 5개로 줄여가며 평가한다. 논문은 Figure 6에서 제안 방법이 source only와 source+target fine-tuning baseline보다 더 우수하다고 보고한다. 특히 label이 거의 없을수록 개선폭이 커지고, target labeled sample 수가 늘어나면 일반 fine-tuning의 성능이 점차 따라오는 경향을 보인다고 한다.

흥미로운 점은 target examples만으로 fine-tuning한 경우의 정확도가 클래스당 1, 3, 5개일 때 각각 $36.6 \pm 0.6$, $60.9 \pm 0.5$, $67.7 \pm 0.5$로, 오히려 source only 모델보다도 낮았다는 것이다. 이는 적은 target sample만으로는 독립적인 target 학습이 불안정하며, source knowledge를 활용한 adaptation이 필수적이라는 점을 잘 보여준다.

또한 저자들은 기존 cross-dataset testbed 논문 [30]의 24.8%보다 훨씬 높은 결과를 얻었다고 언급한다. 다만 이는 adaptation method 차이뿐 아니라, 기존 연구가 SURF BoW feature를 쓴 반면 본 논문은 강력한 CNN feature를 사용했다는 점도 큰 이유라고 명시한다.

### 4.3 추가 분석

논문은 method의 작동 원리를 확인하기 위해 두 가지 분석도 제공한다.

첫째, **domain confusion이 정말 domain invariance를 만드는가**를 보기 위해, Amazon과 Webcam 이미지를 구분하는 SVM을 학습했다. baseline CaffeNet fc7 feature에서는 domain classifier가 99% 정확도로 두 도메인을 구분했지만, domain confusion으로 학습한 fc7 representation에서는 정확도가 56%에 불과했다. 이는 거의 random guess에 가까운 수준으로, feature가 실제로 domain-invariant해졌다는 주장을 뒷받침한다.

둘째, **soft label이 task transfer를 수행하는가**를 보기 위해, Amazon$\rightarrow$Webcam semi-supervised 설정에서 monitor 예시를 분석한다. target에는 monitor labeled data가 없었지만, laptop soft label이 monitor에 상대적으로 높은 확률을 두고 있었고, 그 결과 모델이 monitor를 올바르게 예측할 수 있었다고 설명한다. 이는 soft label이 클래스 간 관계를 통해 unlabeled target category로 정보를 전달한다는 논문의 핵심 주장을 잘 보여주는 사례다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **domain alignment와 class-relationship transfer를 동시에 최적화한다는 설계의 명확성**이다. 단순히 source와 target의 분포를 맞추는 것만으로는 충분하지 않다는 점을 짚고, 클래스 간 구조까지 target으로 넘기려는 발상이 매우 설득력 있다. 특히 semi-supervised adaptation처럼 실제로 중요한데 어려운 설정에서 뚜렷한 개선을 보인 점은 논문의 기여를 강화한다.

또 다른 강점은 방법이 비교적 구현 가능성이 높다는 점이다. 논문은 CaffeNet 기반 구조에 domain classifier와 soft label loss를 추가하는 형태로 제안하며, 표준 backpropagation으로 학습 가능하다고 설명한다. 즉, 완전히 새로운 생성모델이나 복잡한 최적화 체계를 요구하지 않고, 기존 fine-tuning 파이프라인의 확장처럼 사용할 수 있다.

실험 설계도 장점이다. Office dataset의 supervised / semi-supervised 두 설정을 모두 평가했고, cross-dataset이라는 더 어려운 벤치마크까지 포함했다. 또한 단순 성능표뿐 아니라 domain classifier accuracy, qualitative examples, soft label transfer 사례를 통해 왜 성능이 좋아지는지 해석하려 한다.

반면 한계도 분명하다. 먼저, domain confusion은 **marginal distribution 정렬**에 초점을 두며, 클래스 조건부 분포까지 명시적으로 맞추는 것은 아니다. 저자들도 이를 인지하고 soft label을 보완책으로 넣었지만, 여전히 클래스 정렬이 완벽히 보장된다고 보기 어렵다. 특히 target에 labeled examples가 매우 적거나 특정 클래스와 source 클래스 관계가 약한 경우에는 soft label만으로 충분한 alignment가 되지 않을 수 있다.

또한 soft label transfer는 source model이 학습한 클래스 관계가 target에서도 유효하다는 가정에 의존한다. 예를 들어 source에서는 bottle과 mug가 비슷하지만, target 도메인에서는 배경이나 촬영 방식 때문에 전혀 다른 혼동 구조가 나타날 수도 있다. 이런 경우 source soft label이 잘못된 inductive bias를 줄 가능성이 있다. 논문은 이러한 실패 사례를 체계적으로 분석하지는 않는다.

실험 면에서도 몇 가지 제약이 있다. Table 1에서 supervised setting의 성능 향상은 평균적으로 크지 않다. 즉, target의 각 클래스에 약간이라도 라벨이 주어진 상황에서는 제안 기법의 이점이 제한적일 수 있다. 반대로 말하면, 이 논문의 강점은 “일반적인 supervised fine-tuning 대체”라기보다 “라벨이 매우 부족하거나 일부 클래스만 라벨이 있는 상황”에서 더 두드러진다.

또 하나의 비판적 해석은, cross-dataset 결과의 향상이 adaptation 아이디어 자체의 효과와 강력한 CNN representation의 효과가 혼합되어 있다는 점이다. 저자들도 기존 24.8% 결과와의 차이가 주로 SURF BoW 대비 CNN feature의 우수성 때문이라고 인정한다. 따라서 이 논문의 순수한 adaptation 기여를 해석할 때는, feature backbone의 세대 차이를 분리해서 볼 필요가 있다.

마지막으로, 제공된 텍스트 기준으로는 temperature $\tau$의 구체적 설정값이나 soft label 계산의 세부 구현, class subset 선택 방식의 세밀한 통계적 분석은 충분히 제시되지 않았다. 이런 요소들은 재현성과 민감도 분석 측면에서 더 설명이 있었으면 좋았을 것이다. 논문 텍스트에는 이 부분이 명확히 나오지 않으므로 추정할 수 없다.

## 6. 결론

이 논문은 domain adaptation을 위해, **domain confusion loss**와 **soft label loss**를 결합한 CNN 학습 전략을 제안한다. domain confusion은 source와 target의 representation을 구별하기 어렵게 만들어 domain-invariant feature를 유도하고, soft label loss는 source에서 학습된 클래스 간 관계를 target으로 전달하여 일부 또는 전혀 라벨이 없는 target class에도 정보를 전이한다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, deep CNN 기반 adaptation에서 **분포 정렬과 task transfer를 동시에** 수행하는 unified architecture를 제안했다. 둘째, soft labels를 domain adaptation에 도입하여, 클래스 간 관계를 이용한 **across-task transfer**를 실현했다. 셋째, Office dataset과 cross-dataset benchmark에서 supervised 및 semi-supervised adaptation 성능 향상을 실증했다.

실제 적용 측면에서 이 연구는 target domain 라벨이 부족한 다양한 비전 시스템에 중요한 시사점을 준다. 예를 들어 로봇 비전, 산업 현장 카메라, 의료 영상 장비, 자율주행 센서 등에서는 deployment 환경이 training 환경과 자주 달라진다. 이때 완전한 target annotation 없이도 적응이 가능하다는 것은 큰 장점이다. 또한 이 논문의 아이디어는 이후의 adversarial domain adaptation, knowledge distillation 기반 transfer, semi-supervised transfer learning 연구로 자연스럽게 이어질 수 있는 성격을 가진다.

전체적으로 보면, 이 논문은 단순한 “도메인만 맞추는” 접근에서 한 걸음 더 나아가, **어떤 클래스가 어떤 클래스와 닮았는지**라는 semantic structure까지 함께 전달해야 한다는 점을 강조한 의미 있는 작업이다. 특히 labeled target data가 극히 적은 현실적인 조건에서 더 큰 가치를 가지는 연구라고 평가할 수 있다.
