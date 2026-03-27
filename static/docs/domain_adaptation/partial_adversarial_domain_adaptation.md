# Partial Adversarial Domain Adaptation

* **저자**: Zhangjie Cao, Lijia Ma, Mingsheng Long, Jianmin Wang
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1808.04205](https://arxiv.org/abs/1808.04205)

## 1. 논문 개요

이 논문은 **partial domain adaptation**이라는 새로운 문제 설정을 다룬다. 기존의 unsupervised domain adaptation은 보통 source domain과 target domain이 **같은 label space**를 가진다고 가정한다. 하지만 실제 환경에서는 이 가정이 자주 깨진다. 예를 들어, ImageNet처럼 매우 큰 데이터셋으로 학습한 모델을 더 작은 실제 문제에 옮기고 싶을 때, source의 클래스 집합이 target의 클래스 집합보다 더 큰 경우가 자연스럽다. 즉, target label space가 source label space의 부분집합인 상황이 많다.

논문이 겨냥하는 핵심 문제는 바로 여기서 발생하는 **negative transfer**이다. 기존 방법들은 source와 target의 feature distribution 전체를 맞추려 하기 때문에, 실제로는 target에 존재하지 않는 source의 **outlier classes**까지 target에 정렬하려고 한다. 그 결과, target 분류 성능이 오히려 나빠질 수 있다. 저자들은 이 상황이 단순한 domain shift보다 더 어렵다고 본다. 왜냐하면 target의 클래스가 무엇인지 학습 중에 알 수 없고, 따라서 어떤 source 클래스가 공유 클래스이고 어떤 클래스가 outlier인지도 직접 주어지지 않기 때문이다.

이 논문의 목표는 크게 두 가지이다. 첫째, target과 무관한 outlier source classes의 영향을 줄여서 negative transfer를 완화하는 것이다. 둘째, target과 실제로 공유되는 source classes에 대해서만 feature alignment를 강화하여 positive transfer를 유도하는 것이다. 이를 위해 저자들은 **PADA (Partial Adversarial Domain Adaptation)**라는 end-to-end adversarial adaptation 프레임워크를 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **target 데이터가 source의 어떤 클래스들과 닮았는지 source classifier의 예측 분포를 이용해 추정하고, 그 정보를 class-level weight로 바꾸어 outlier source classes의 영향을 줄인다**는 것이다.

기존의 DANN 계열 방법은 source와 target을 구분하는 domain discriminator를 속이도록 feature extractor를 학습한다. 이 방식은 source와 target의 label space가 같을 때는 효과적이지만, partial domain adaptation에서는 문제가 된다. target에 없는 source class들까지 함께 정렬되기 때문이다. 저자들은 이 점이 partial setting에서 adversarial adaptation이 쉽게 실패하는 핵심 원인이라고 본다.

PADA의 차별점은 **“무조건 전체 source를 target에 맞추지 않는다”**는 데 있다. 대신 target 샘플들을 source classifier에 통과시켜, 각 source class에 대한 평균 예측 확률을 구한다. 이 평균값이 큰 클래스는 target과 관련 있을 가능성이 크고, 평균값이 작은 클래스는 outlier일 가능성이 크다. 그런 다음 이 class weight를 source classifier의 학습과 domain adversary의 학습에 모두 반영한다. 결과적으로, target과 관련 있는 source 데이터는 더 많이 반영되고, 관련 없는 source 데이터는 자동으로 약화된다.

이 접근의 중요한 장점은, 별도의 target label 없이도 shared classes와 outlier classes를 **간접적으로 식별**한다는 점이다. 즉, target label space를 직접 알지 못해도, target 데이터가 source classifier에서 보이는 평균 반응을 이용해 어느 클래스가 target과 연결되는지 추론한다. 이 아이디어는 단순하지만 partial setting의 본질적 난점을 잘 겨냥하고 있다.

## 3. 상세 방법 설명

논문은 먼저 기존 **DANN (Domain Adversarial Neural Network)**의 목적함수를 설명한 뒤, 이를 partial setting에 맞게 수정한다. 기본 구성은 세 부분으로 이루어진다. feature extractor $G_f$, source classifier $G_y$, domain discriminator $G_d$이다. 입력 $\mathbf{x}$는 feature $ \mathbf{f} = G_f(\mathbf{x}) $로 변환되고, classifier는 이를 기반으로 class prediction을, discriminator는 domain prediction을 수행한다.

기존 DANN의 목적은 source classification loss를 줄이면서, 동시에 domain discriminator가 source와 target을 구분하지 못하도록 만드는 것이다. 논문에 제시된 목적함수는 다음과 같다.

$$
C_0(\theta_f, \theta_y, \theta_d) = \frac{1}{n_s}\sum_{\mathbf{x}_i \in \mathcal{D}_s} L_y\left(G_y(G_f(\mathbf{x}_i)), y_i\right) - \frac{\lambda}{n_s+n_t} \sum_{\mathbf{x}_i \in \mathcal{D}_s \cup \mathcal{D}_t} L_d\left(G_d(G_f(\mathbf{x}_i)), d_i\right)
$$

여기서 $L_y$는 source classification loss이고, $L_d$는 domain discrimination loss이다. $\lambda$는 두 목적 사이의 trade-off를 조절하는 하이퍼파라미터이다. 최적화는 minimax 구조를 따른다. 즉, $(\theta_f, \theta_y)$는 전체 목적을 최소화하고, $\theta_d$는 domain classification을 잘 하도록 최대화한다. 보통 이 구조는 **Gradient Reversal Layer (GRL)**로 구현된다.

문제는 partial adaptation에서는 이 방식이 전체 source distribution과 전체 target distribution을 정렬하려 하기 때문에, source의 outlier classes가 target 쪽 feature 구조를 오염시킨다는 점이다. 이를 해결하기 위해 PADA는 먼저 target 데이터의 class prediction 평균을 계산한다. 각 target 샘플 $\mathbf{x}_i^t$에 대해 classifier 출력 $\hat{\mathbf{y}}_i = G_y(\mathbf{x}_i^t)$를 얻고, 이를 전체 target에 대해 평균낸다.

$$
\mathbf{\gamma} = \frac{1}{n_t}\sum_{i=1}^{n_t}\hat{\mathbf{y}}_i
$$

이때 $\mathbf{\gamma}$는 source label space의 각 클래스에 대한 weight vector이다. 직관적으로, 어떤 source class가 target과 공유된다면 target 샘플들이 그 클래스에 비교적 높은 확률을 줄 것이므로 해당 weight는 커진다. 반대로 outlier source class는 target 샘플들이 거의 선택하지 않으므로 weight가 작아진다. 논문은 이 weight를 최대값으로 나누어 정규화한다. 즉, 가장 큰 weight가 1이 되도록 맞춘다.

이제 이 class weight를 source classifier와 domain adversary 모두에 적용한다. 논문이 제안한 PADA의 목적함수는 다음과 같다.

$$
\begin{aligned}
C(\theta_f, \theta_y, \theta_d) = & \frac{1}{n_s}\sum_{\mathbf{x}_i \in \mathcal{D}_s} \gamma_{y_i} L_y\left(G_y(G_f(\mathbf{x}_i)), y_i\right) \\
& - \frac{\lambda}{n_s}\sum_{\mathbf{x}_i \in \mathcal{D}_s} \gamma_{y_i} L_d\left(G_d(G_f(\mathbf{x}_i)), d_i\right) \\
& - \frac{\lambda}{n_t}\sum_{\mathbf{x}_i \in \mathcal{D}_t} L_d\left(G_d(G_f(\mathbf{x}_i)), d_i\right)
\end{aligned}
$$

여기서 핵심은 source 샘플 $\mathbf{x}_i$의 label $y_i$에 대응하는 class weight $\gamma_{y_i}$가 classifier loss와 source-side domain loss 모두에 곱해진다는 점이다. 의미를 풀어 쓰면 다음과 같다.

첫째, **source classifier**는 outlier class에 속한 샘플의 영향을 덜 받게 된다. 따라서 분류기 자체가 target과 관련된 source classes에 더 집중하도록 유도된다.

둘째, **domain discriminator**는 source 쪽에서 outlier class 샘플들을 덜 중요하게 보게 된다. 따라서 adversarial alignment가 전체 source를 target에 맞추는 것이 아니라, weight가 큰 shared classes 중심으로 일어나게 된다.

셋째, target 쪽 domain loss에는 별도의 class weight가 직접 곱해지지 않는다. target은 label이 없기 때문에 class별 가중치를 샘플 수준으로 직접 배정하기 어렵기 때문이다. 대신 source 측을 down-weighting하여 결과적으로 alignment의 중심을 shared classes 쪽으로 이동시킨다.

최적화 구조는 DANN과 동일한 minimax 형태이다.

$$
(\hat{\theta}_f, \hat{\theta}_y) = \arg\min_{\theta_f,\theta_y} C(\theta_f,\theta_y,\theta_d), \quad \hat{\theta}_d = \arg\max_{\theta_d} C(\theta_f,\theta_y,\theta_d)
$$

아키텍처 관점에서 보면, PADA는 완전히 새로운 네트워크라기보다 **DANN 위에 class-weighting 메커니즘을 추가한 구조**이다. 따라서 구현은 비교적 단순하지만, partial setting의 핵심 실패 원인을 직접 다룬다는 점에서 설계가 정교하다.

다만 논문은 이 방법이 잘 작동하려면 source classifier가 target 데이터에 대해 완전히 엉뚱한 예측만 내놓지 않아야 한다는 암묵적 전제를 갖는다. 이를 완화하기 위해 저자들은 단일 샘플 예측 대신 **전체 target에 대한 평균 예측**을 사용한다고 설명한다. 즉, 일부 target 샘플의 잘못된 예측은 평균 과정에서 희석된다고 본다.

## 4. 실험 및 결과

논문은 네 가지 데이터셋 계열에서 partial domain adaptation 성능을 평가한다. Office-31, Office-Home, ImageNet-Caltech, 그리고 VisDA2017이다. 모든 실험은 unsupervised domain adaptation 설정으로 수행되며, source는 라벨이 있고 target은 라벨이 없다.

Office-31에서는 원래 31개 클래스가 있는 각 domain을 source로 쓰고, target은 Office-31과 Caltech-256이 공유하는 10개 클래스만 사용한다. 따라서 source는 31 classes, target은 10 classes가 된다. 평가 task는 $A \rightarrow W$, $D \rightarrow W$, $W \rightarrow D$, $A \rightarrow D$, $D \rightarrow A$, $W \rightarrow A$의 6개이다.

Office-Home은 더 어려운 데이터셋으로, 4개 domain과 65개 object categories를 가진다. source는 65개 전체 클래스를 사용하고, target은 알파벳 순서상 앞의 25개 클래스를 사용한다. 따라서 총 12개의 partial transfer task를 구성한다. 이 데이터셋은 domain gap이 크기 때문에 partial adaptation의 난도를 더 잘 보여준다.

ImageNet-Caltech에서는 ImageNet-1K와 Caltech-256이 공유하는 84개 클래스를 기반으로 두 개의 task를 만든다. 특히 ImageNet-1K를 source로 쓰는 설정은 source 클래스 수가 매우 크기 때문에 partial adaptation의 실제 가치를 잘 보여준다.

VisDA2017에서는 두 domain이 원래 12 classes를 공유하지만, target domain은 그중 처음 6개만 사용하여 partial setting을 만든다. 이 실험은 대규모 이미지 수 환경에서의 효율성을 보여주기 위한 것이다.

비교 대상은 ResNet-50, DAN, RTN, DANN, ADDA, JAN 등 대표적인 transfer learning 및 deep domain adaptation 방법들이다. 또한 제안 방식의 효과를 분해해서 보기 위해 두 가지 ablation variant도 포함한다. **PADA-classifier**는 source classifier 쪽 weighting을 제거한 버전이고, **PADA-adversarial**은 domain adversary 쪽 weighting을 제거한 버전이다.

실험 결과는 매우 일관적이다. Office-Home에서 평균 정확도는 ResNet이 53.71, DAN이 54.48, DANN이 47.39, RTN이 59.25인 반면, PADA는 **62.06**을 기록한다. 특히 $Ar \rightarrow Pr$에서는 67.00, $Pr \rightarrow Rw$에서는 78.79, $Rw \rightarrow Pr$에서는 77.09를 달성하며 기존 방법보다 높다. domain gap이 큰 task에서도 improvement가 유지된다는 점이 중요하다.

Office-31에서는 향상이 더욱 극적이다. 평균 정확도 기준으로 ResNet은 75.64, RTN은 84.81, LEL은 84.79인데, PADA는 **92.69**를 달성한다. 예를 들어 $A \rightarrow W$에서 86.54, $D \rightarrow A$에서 92.69, $W \rightarrow A$에서 95.41을 기록했다. 특히 $W \rightarrow D$에서는 100의 정확도를 얻는다. 저자들은 이 결과를 통해 small domain gap 상황에서도, label-space mismatch가 있으면 기존 adversarial adaptation이 심하게 무너질 수 있음을 강조한다.

ImageNet-Caltech와 VisDA2017에서도 PADA가 가장 높은 평균 성능을 보인다. ImageNet-Caltech에서는 ResNet 평균이 68.90, RTN이 70.29인데 PADA는 **72.76**이다. VisDA2017에서는 ResNet 평균 54.77, DANN 62.43, RTN 61.49에 비해 PADA가 **65.01**을 기록한다. 특히 large-scale source dataset에서 효과가 유지된다는 점은 논문의 동기를 강하게 뒷받침한다.

실험 결과에서 특히 눈에 띄는 점은 **기존 domain adaptation 방법이 종종 source-only baseline인 ResNet보다도 못하다**는 것이다. 이는 partial setting에서 negative transfer가 매우 심각할 수 있음을 직접 보여준다. DANN이나 DAN이 전체 source와 target을 맞추려 하기 때문에, 오히려 target 성능을 해치는 것이다.

Ablation study도 설득력이 있다. Office-31 평균 성능은 PADA-classifier가 90.85, PADA-adversarial이 85.37, PADA가 92.69이다. 즉, 두 weighting이 모두 중요하지만 특히 **adversarial network 쪽 weighting**이 더 큰 기여를 한다고 볼 수 있다. Office-Home에서도 PADA-classifier는 55.55, PADA-adversarial은 52.13, PADA는 62.06이다. 이는 outlier source classes를 domain alignment 단계에서 억제하는 것이 partial adaptation에서 매우 중요하다는 해석과 잘 맞는다.

논문은 추가적인 empirical analysis도 제시한다. 첫째, class weight histogram 분석에서 PADA는 shared classes에 큰 weight를, outlier classes에 작은 weight를 부여한다. 일부 outlier class는 거의 0에 가까운 weight를 받는다. 반면 DANN은 outlier classes에도 큰 weight가 남아 있어서 harmful alignment를 막지 못한다.

둘째, target class 수를 줄여가며 partialness를 더 강하게 만들었을 때, DANN은 빠르게 성능이 붕괴한다. 반면 PADA는 더 안정적이며, outlier class 비율이 커질수록 DANN 대비 상대적 우위가 커진다. 저자들은 이를 negative transfer의 심화와 연결해 설명한다.

셋째, 학습 과정에서의 test error를 보면 ResNet, DAN, DANN은 negative transfer 때문에 test error가 증가하거나 불안정해지는 반면, PADA는 더 빠르고 안정적으로 낮은 error로 수렴한다.

넷째, t-SNE 시각화에서는 DAN, DANN, RTN이 target data를 잘못된 source class까지 포함한 전체 source 구조 쪽으로 끌어당기는 반면, PADA는 shared classes와 target을 더 올바르게 가깝게 두고 outlier classes의 영향을 줄이는 것으로 해석된다. 이 부분은 정성적 분석이지만, 제안한 weighting 메커니즘의 직관을 잘 시각화한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정 자체가 현실적이고 중요하다**는 점이다. 많은 실제 전이 학습 상황에서 source는 크고 target은 작기 때문에, label space가 완전히 일치하지 않는 경우가 흔하다. 논문은 이 상황을 명시적으로 정의하고, 기존 방법들이 왜 실패하는지 분명하게 분석한다.

두 번째 강점은 **방법이 단순하면서도 정확히 문제를 찌른다**는 점이다. PADA는 새로운 복잡한 generative mechanism이나 pseudo-labeling 체계를 도입하지 않는다. 대신 source classifier의 target 평균 예측이라는 간단한 신호를 이용해 class-level weighting을 만들고, 이것을 classifier와 adversary에 동시에 적용한다. 구조적으로는 DANN의 확장처럼 보이지만, partial adaptation에서 필요한 inductive bias를 잘 반영한다.

세 번째 강점은 **실험이 매우 광범위하고 일관적**이라는 점이다. 작은 데이터셋, 큰 데이터셋, 작은 domain gap, 큰 domain gap, 그리고 large-class setting까지 포함한다. 또한 ablation, convergence, class weight statistics, t-SNE 분석까지 포함해 제안 방법의 작동 원리를 다양한 관점에서 뒷받침한다.

네 번째 강점은 **negative transfer를 직접 다룬다**는 점이다. 많은 transfer learning 연구가 성능 향상 자체를 보여주는 데 집중하는 반면, 이 논문은 왜 성능이 나빠지는지, 특히 label-space mismatch 때문에 생기는 구조적 실패를 중심 문제로 놓는다. 이는 후속 연구에 중요한 기준점을 제공한다.

반면 한계도 분명하다. 첫째, class weight $\mathbf{\gamma}$는 source classifier의 target prediction 평균에 의존한다. 따라서 초기에 classifier가 target에 대해 매우 부정확한 예측을 하면 weight 추정도 왜곡될 수 있다. 논문은 평균을 통해 일부 오류를 상쇄한다고 설명하지만, 이 추정의 안정성과 초기화 민감도에 대한 더 깊은 이론 분석은 제공하지 않는다.

둘째, weighting이 **class-level**이다. 즉, 같은 클래스 안의 샘플이라면 모두 같은 weight를 받는다. 하지만 실제로는 shared class 내부에도 domain shift 정도가 다를 수 있고, outlier class처럼 보이는 샘플이나 어려운 경계 샘플도 있을 수 있다. sample-level 또는 instance-level 정교함은 이 논문 범위를 벗어난다.

셋째, 논문은 target label space가 source label space의 부분집합이라는 가정 위에 서 있다. 즉, **target-private classes**가 있는 open-set 또는 universal domain adaptation 문제는 다루지 않는다. 따라서 적용 범위는 partial setting으로 한정된다.

넷째, 실험은 매우 강하지만, 방법의 이론적 보장은 제한적이다. 예를 들어 class weight 추정 오차가 최종 target risk에 어떤 영향을 주는지, 혹은 weight가 잘못 추정될 때의 failure mode가 무엇인지는 이 논문에서 엄밀히 분석하지 않는다.

비판적으로 보면, 이 논문은 partial adaptation의 핵심 문제를 잘 포착하고 실용적 해법을 제시했지만, weight estimation 자체를 더 robust하게 만드는 방향은 후속 연구의 과제로 남겨둔다. 또한 target distribution이 매우 복잡하거나 source classifier가 초기에 심하게 편향되어 있을 때 얼마나 안정적인지는 본문만으로는 충분히 판단하기 어렵다.

## 6. 결론

이 논문은 source label space가 target label space를 포함하는 **partial domain adaptation** 문제를 명확히 정의하고, 이를 해결하기 위한 **PADA**를 제안했다. 핵심 기여는 target prediction 평균으로부터 source class별 중요도를 추정하고, 이 가중치를 source classifier와 domain adversarial discriminator에 함께 적용함으로써, outlier source classes의 영향을 줄이고 shared classes 중심의 alignment를 유도한 것이다.

실험적으로 PADA는 Office-31, Office-Home, ImageNet-Caltech, VisDA2017 등 다양한 partial adaptation benchmark에서 기존 방법들을 일관되게 능가했다. 특히 기존 adversarial adaptation이나 MMD 기반 방법들이 partial setting에서는 오히려 성능을 해칠 수 있음을 보여주고, 제안한 weighting 메커니즘이 negative transfer를 완화하는 데 핵심적이라는 점을 설득력 있게 입증했다.

이 연구의 실제적 의미는 크다. 대규모 source dataset에서 학습한 모델을 더 작고 라벨이 없는 target dataset에 전이하는 문제는 매우 흔하며, 이때 label-space mismatch는 피하기 어렵다. PADA는 그런 상황에서 단순한 feature reuse를 넘어 **classifier layer까지 포함한 더 적극적인 transfer**를 가능하게 하는 방향을 제시한다. 따라서 이 논문은 이후 partial, open-set, universal domain adaptation 연구로 이어지는 중요한 출발점으로 볼 수 있다. 또한 실제 응용에서도, source가 더 크고 target이 더 좁은 문제 구조를 가질 때 유용한 설계 원칙을 제공한다.
