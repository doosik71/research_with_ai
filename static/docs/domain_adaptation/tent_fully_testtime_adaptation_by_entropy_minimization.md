# Tent: Fully Test-Time Adaptation by Entropy Minimization

* **저자**: Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2006.10726](https://arxiv.org/abs/2006.10726)

## 1. 논문 개요

이 논문은 **fully test-time adaptation**이라는 문제를 정면으로 다룬다. 이는 모델이 테스트 시점에 **source data도 없고, target label도 없으며, 오직 현재 들어오는 target data와 이미 학습된 자기 자신의 파라미터만으로 적응해야 하는 상황**을 의미한다. 저자들은 이런 제약이 실제 배포 환경에서 매우 중요하다고 본다. 예를 들어 개인정보, 대역폭, 상업적 이유로 source data를 함께 배포할 수 없고, 테스트 중에 source data를 다시 처리할 계산 여유도 없으며, 분포 변화가 생기면 적응 없이는 정확도가 실제 사용 가능 수준에 미치지 못할 수 있기 때문이다.

논문의 핵심 목표는 이런 극단적으로 제한된 조건에서도 모델이 스스로 target distribution에 적응하도록 만드는 것이다. 이를 위해 저자들은 **prediction entropy를 테스트 시점에 직접 최소화하는 방법**을 제안하고, 이 방법을 **Tent**라고 부른다. 핵심 발상은 간단하다. 모델의 예측이 불확실할수록 entropy가 높고, distribution shift가 심할수록 일반적으로 entropy도 증가한다. 따라서 테스트 데이터에서 entropy를 낮추도록 모델을 조정하면, 보다 확신 있는 예측을 하도록 유도할 수 있고, 그 결과 generalization error도 줄일 수 있다는 것이다.

이 문제가 중요한 이유는, 기존의 fine-tuning, domain adaptation, test-time training 같은 적응 방식들이 대체로 source data, target data, 또는 별도의 self-supervised proxy task에 의존하기 때문이다. 반면 이 논문은 **학습 과정을 바꾸지 않고**, 그리고 **추가 데이터나 레이블 없이**, 테스트 중에 바로 적응하는 방법을 제시한다. 즉, 이 연구는 “모델이 배포된 이후 실제 환경 변화에 맞춰 스스로 적응할 수 있는가?”라는 매우 실용적인 질문에 대한 직접적인 해답을 제시한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **테스트 시점에 모델의 prediction entropy를 최소화하도록, 전체 모델이 아니라 normalization layer의 channel-wise affine parameters만 업데이트하자**는 것이다. 다시 말해 Tent는 테스트 배치가 들어올 때마다 해당 배치에서 normalization statistics를 다시 추정하고, 동시에 각 채널의 scale과 shift 파라미터인 $\gamma, \beta$를 entropy gradient로 업데이트한다.

이 직관은 두 가지 관찰에 기반한다. 첫째, 논문은 corrupted data에서 **낮은 entropy가 낮은 error와 연결**된다는 점을 보인다. 둘째, corruption severity가 커질수록 entropy와 loss가 함께 증가하는 경향이 있다는 점도 보인다. 따라서 테스트 중에 entropy를 낮추는 것은 단순히 confidence를 키우는 것이 아니라, 실제로 error reduction과 연결될 가능성이 높다는 논리다.

기존 접근과의 차별점도 분명하다. 일반적인 domain adaptation은 source와 target을 동시에 사용하며, test-time training은 학습 단계에서 self-supervised auxiliary task를 함께 학습하도록 training 자체를 바꿔야 한다. 반면 Tent는 **training을 전혀 바꾸지 않는다**. 또한 entropy minimization 자체는 이전에도 train-time regularization으로 쓰였지만, 이 논문은 그것을 **오직 test-time loss로만 사용해 fully test-time adaptation을 수행하는 최초의 시도**라고 주장한다.

또 하나의 중요한 차별점은 **어떤 파라미터를 업데이트할 것인가**에 있다. 전체 모델 파라미터 $\theta$를 직접 업데이트하면 표현력이 크지만, source knowledge가 저장된 파라미터 전체를 테스트 중에 바꾸는 것은 불안정하고 계산량도 크다. Tent는 이 문제를 피하기 위해 normalization layer의 affine modulation만 조정한다. 이 선택은 파라미터 수가 매우 적고, 채널 단위의 선형 modulation이기 때문에 효율적이며, 테스트 중 온라인 적응에도 적합하다는 장점이 있다.

## 3. 상세 방법 설명

Tent의 방법은 세 가지 요소로 정리할 수 있다. 첫째는 **entropy objective**, 둘째는 **feature modulation parameterization**, 셋째는 **test-time optimization procedure**이다.

### 3.1 문제 설정

적응 대상 모델은 $f_\theta(x)$로 표기된다. 이 모델은 이미 source domain에서 supervised task로 학습되어 있어야 하며, 테스트 시점에는 target input $x^t$만 주어진다. 이때 fully test-time adaptation은 source data $x^s, y^s$도 없고, target label $y^t$도 없는 상태에서 테스트 중 적응을 수행한다.

논문은 이 설정을 다른 적응 시나리오와 구분한다.

fine-tuning은 target label이 필요하고, domain adaptation은 source와 target data를 함께 써야 하며, test-time training은 학습 단계에서 self-supervised task를 같이 학습해야 한다. Tent는 이들보다 더 제약이 강한 조건, 즉 **오직 target unlabeled data만 있는 상황**을 전제로 한다.

### 3.2 Entropy objective

Tent의 테스트 시점 목적 함수는 예측 분포의 Shannon entropy를 최소화하는 것이다. 모델의 prediction을 $\hat{y} = f_\theta(x^t)$라고 하면, entropy는 다음과 같다.

$$
H(\hat{y}) = - \sum_c p(\hat{y}_c)\log p(\hat{y}_c)
$$

여기서 $\hat{y}_c$는 class $c$에 대한 예측 확률이다. 이 값이 크다는 것은 예측이 퍼져 있고 확신이 낮다는 뜻이고, 값이 작다는 것은 특정 클래스에 확률이 집중되어 있다는 뜻이다.

논문은 entropy를 unsupervised objective로 해석한다. label 없이도 예측 분포만으로 계산할 수 있기 때문이다. 동시에 이 loss는 supervised task와 직접 연결되어 있다. self-supervised proxy task처럼 본래 task와 간접적인 관계를 갖는 것이 아니라, **분류 결과 자체의 불확실성**을 줄이는 방향으로 최적화되기 때문이다.

다만 단일 샘플에 대해 entropy만 최소화하면 자명한 해, 즉 한 클래스에 확률을 몰아주는 trivial solution으로 흐를 수 있다. 이를 막기 위해 저자들은 **배치 단위의 shared parameter update**를 사용한다. 즉, 개별 예측이 아니라 배치 전체에 공통으로 작용하는 modulation parameter를 업데이트하여, 단순한 one-sample collapse를 피하려고 한다.

### 3.3 왜 전체 파라미터가 아니라 modulation만 업데이트하는가

가장 자연스러운 생각은 전체 모델 파라미터 $\theta$를 테스트 중 직접 최적화하는 것이다. 하지만 논문은 두 가지 이유로 이를 피한다.

첫째, fully test-time adaptation에서는 $\theta$가 사실상 source data에 대한 유일한 저장소다. 따라서 이를 크게 바꾸면 source domain에서 학습된 유용한 표현 자체가 무너질 수 있다.

둘째, 딥네트워크는 비선형이고 고차원이라 전체 파라미터를 테스트 중 안정적으로 최적화하기 어렵다. 특히 테스트 환경에서는 빠르고 가벼운 적응이 필요하므로, 전체 모델 업데이트는 계산량과 안정성 측면에서 부적절하다.

그래서 Tent는 **normalization layer의 channel-wise affine parameter**만 업데이트한다. 구체적으로 feature $x$에 대해 먼저 normalization을 적용한다.

$$
\bar{x} = \frac{x-\mu}{\sigma}
$$

그 다음 affine transform을 적용한다.

$$
x' = \gamma \bar{x} + \beta
$$

여기서 $\mu, \sigma$는 현재 target batch에서 추정되는 normalization statistics이고, $\gamma, \beta$는 학습 가능한 scale과 shift parameter이다. 논문은 테스트 중에 source에서 얻은 기존 normalization statistics는 버리고, target batch로부터 새롭게 $\mu, \sigma$를 추정한다. 동시에 $\gamma, \beta$는 entropy gradient를 따라 업데이트한다.

이 구조의 의미는 다음과 같다. normalization statistics의 재추정은 target domain의 feature distribution에 맞게 중심과 분산을 보정하는 역할을 한다. 반면 affine parameters의 업데이트는 단순한 통계 보정에 그치지 않고, 실제 task loss와 연결된 entropy를 직접 줄이도록 feature representation을 재조정하는 역할을 한다.

논문은 이때 적응되는 파라미터 수가 전체 모델의 1% 미만이라고 설명한다. 즉, 적은 수의 파라미터만 조정해도 상당한 적응 효과를 낼 수 있다는 것이다.

### 3.4 알고리즘 흐름

Tent 알고리즘은 initialization, iteration, termination의 세 단계로 설명된다.

초기화 단계에서는 각 normalization layer의 각 channel에 대한 affine parameters ${\gamma_{l,k}, \beta_{l,k}}$를 수집하여 optimizer의 업데이트 대상으로 설정한다. 그 외 나머지 파라미터는 고정된다. 또한 source data에서 저장된 normalization statistics ${\mu_{l,k}, \sigma_{l,k}}$는 버린다.

반복 단계에서는 각 target batch에 대해 forward pass 중 normalization statistics를 새로 추정한다. 그 다음 예측 entropy의 gradient $\nabla H(\hat{y})$를 계산하고, backward pass에서 $\gamma, \beta$를 업데이트한다. 중요한 점은 현재 batch에서 계산된 gradient로 parameter를 업데이트해도, 그 업데이트는 기본적으로 **다음 batch의 prediction에 반영**된다는 것이다. 즉, 기본 online setting에서는 현재 batch로 통계를 추정하고, 그 배치의 entropy로 다음 배치를 위한 파라미터를 개선한다.

종료 조건은 setting에 따라 다르다. online adaptation에서는 테스트 데이터가 계속 들어오는 동안 업데이트를 계속하면 된다. offline adaptation에서는 먼저 target set 전체 또는 일부에 대해 적응을 수행한 뒤, 다시 inference를 반복할 수 있다. 논문은 기본적으로 매우 효율적인 setting을 강조하기 위해 **test point당 one gradient step** 수준의 적응을 사용한다.

### 3.5 BN과 Tent의 차이

겉으로 보면 Tent는 batch normalization statistics를 target batch로 바꾸는 test-time normalization과 비슷해 보이지만, 실제로는 더 강한 방법이다. BN baseline은 테스트 시 $\mu, \sigma$만 target batch로 교체하고 affine parameter는 그대로 둔다. 반면 Tent는 **통계 추정 + affine parameter optimization**을 함께 수행한다.

논문의 분석에 따르면 BN은 corrupted feature를 원래 source-like feature 쪽으로 되돌리는 경향이 있지만, Tent는 꼭 원래 source feature로 돌아가지는 않는다. 오히려 target label을 알고 최적화한 oracle feature 변화와 더 가까운 방향으로 움직인다. 이는 Tent가 단순한 distribution alignment가 아니라, **task-aware한 adaptation**을 수행하고 있음을 시사한다.

## 4. 실험 및 결과

논문은 Tent를 크게 세 종류의 문제에서 평가한다. 첫째는 corruption robustness, 둘째는 source-free domain adaptation for digit recognition, 셋째는 더 큰 규모의 semantic segmentation과 VisDA-C benchmark이다. 전반적으로 공통 메시지는 “source data 없이, training 수정 없이, 테스트 중 짧은 최적화만으로도 의미 있는 적응이 가능하다”는 것이다.

### 4.1 실험 설정

이미지 분류 실험에서는 CIFAR-10, CIFAR-100, ImageNet을 사용한다. corruption robustness 평가는 CIFAR-10-C, CIFAR-100-C, ImageNet-C에서 수행되며, 이들은 원래 test/validation set에 15종의 corruption을 5개 severity level로 적용한 benchmark이다.

digit domain adaptation에서는 SVHN을 source로, MNIST, MNIST-M, USPS를 target으로 사용한다. semantic segmentation에서는 GTA를 source, Cityscapes를 target으로 사용하고, 모델은 HRNet-W18이다. VisDA-C에서는 synthetic-to-real object recognition benchmark를 사용한다.

모델은 분류 실험에서 ResNet 계열을 사용한다. CIFAR 실험에는 26-layer residual network, ImageNet에는 ResNet-50을 사용한다. 모든 네트워크는 batch normalization을 갖추고 있다. Tent의 optimization은 ImageNet에서 SGD with momentum, 나머지에서는 Adam을 사용한다. 배치 크기와 learning rate는 inference 메모리 제약을 고려해 조정된다.

비교 기준선으로는 source-only, adversarial domain adaptation (RG), self-supervised domain adaptation (UDA-SS), test-time training (TTT), test-time normalization (BN), pseudo-labeling (PL)이 사용된다. 이 중 BN, PL, Tent만이 fully test-time adaptation 계열이다.

### 4.2 CIFAR corruption benchmark

가장 severe한 corruption level에서의 평균 error는 다음과 같다.

CIFAR-10-C에서는 source가 40.8%, RG가 18.3%, UDA-SS가 16.7%, TTT가 17.5%, BN이 17.3%, PL이 15.7%, Tent가 14.3%이다.

CIFAR-100-C에서는 source가 67.2%, RG가 38.9%, UDA-SS가 47.0%, TTT가 45.0%, BN이 42.6%, PL이 41.2%, Tent가 37.3%이다.

이 결과는 몇 가지 점에서 중요하다. 첫째, Tent는 fully test-time baselines인 BN과 PL보다 consistently 더 낮은 error를 보인다. 둘째, source와 target을 함께 쓰는 domain adaptation이나, 학습 단계 자체를 수정하는 TTT보다도 더 좋은 결과를 내는 경우가 있다. 즉, 더 적은 정보와 더 적은 최적화로 더 좋은 성능을 낼 수 있다는 점이 강하게 드러난다.

### 4.3 ImageNet-C robustness

ImageNet-C는 규모가 훨씬 크므로 모든 baselines를 동일하게 돌리기 어렵고, 논문은 효율적인 방법들 중심으로 비교한다. Tent는 corruption type 대부분에서 source-only 및 BN보다 error를 더 낮춘다. 특히 원본 clean data의 error를 증가시키지 않으면서 대부분의 corruption 유형에서 개선을 보였다는 점이 강조된다.

정량적으로는, 저자들이 언급한 prior state-of-the-art robust training 방법 중 ANT가 50.2% error, AugMix가 51.7%, ANT+SIN이 47.4%를 기록한다. Tent는 **online adaptation으로 44.0%**, **offline adaptation으로 42.3%** error를 달성한다. 이는 학습을 바꾸지 않고, 테스트 시점 적응만으로 robust training 기반 방법들을 넘어섰다는 뜻이다.

특히 BN 대비 49.9%에서 44.0%로 떨어졌다는 점은 단순한 normalization statistics 업데이트만으로는 부족하고, entropy 기반 affine modulation이 실제로 추가적인 이득을 준다는 증거로 해석할 수 있다.

### 4.4 Digit source-free domain adaptation

SVHN에서 MNIST, MNIST-M, USPS로 가는 적응 결과도 제시된다.

source-only는 각각 18.2%, 39.7%, 19.3% error이다. BN은 15.7%, 39.7%, 18.0%이고, Tent는 1 epoch 기준 10.0%, 37.0%, 16.3%를 달성한다. 10 epoch까지 늘리면 8.2%, 36.8%, 14.4%까지 향상된다.

이 결과는 source-free adaptation이 가능할 뿐 아니라, BN보다 항상 낫고, 일부 경우에는 source data를 쓰는 RG와 UDA-SS보다도 더 좋은 결과를 낸다는 점을 보여준다. 다만 논문은 동시에 harder shift에서는 한계가 있음을 인정한다. 예를 들어 SVHN-to-MNIST와 달리 MNIST-to-SVHN 같은 더 어려운 shift에서는 Tent가 실패하여 오히려 error를 증가시키는 사례가 언급된다. 이는 fully test-time adaptation이 강력하지만 만능은 아니며, 여전히 일부 문제에서는 source-target joint optimization이 필요함을 시사한다.

### 4.5 Semantic segmentation과 VisDA-C

이 논문은 Tent가 작은 분류 문제에만 국한되지 않음을 보여주기 위해 semantic segmentation도 실험한다. GTA에서 Cityscapes로의 sim-to-real shift에서 HRNet-W18 기반 segmentation 모델의 target mIoU는 source-only가 28.8%, BN이 31.4%, Tent가 35.8%이다. 단일 이미지에 대해 episodic optimization을 수행하면 10 iterations 만에 36.4%까지 도달한다.

이 결과는 Tent가 batch-level classification뿐 아니라 **pixel-wise dense prediction**에도 적용될 수 있음을 보여준다. 논문은 qualitative example도 제시하며, Tent가 segmentation noise를 줄이고 누락된 클래스 예측을 복구하는 모습을 설명한다.

VisDA-C에서는 source model이 56.1% validation error를 보이는 반면, Tent는 45.6%까지 낮춘다. 더 나아가 마지막 classifier를 제외한 모든 layer를 업데이트하면 39.6%까지 개선된다. 다만 저자들은 이 경우 SHOT 같은 offline source-free adaptation 방법들이 더 낮은 error를 낼 수 있다고 솔직하게 인정한다. 대신 Tent의 장점은 **online test-time adaptation이 가능하다**는 점에 있다.

### 4.6 분석 실험

논문은 단순히 성능 표만 제시하지 않고, 왜 Tent가 작동하는지를 분석하려고 한다.

먼저 entropy 감소와 task loss 감소의 상관을 본다. CIFAR-100-C의 모든 corruption type/level 조합에 대해 entropy 변화 $\Delta H$와 loss 변화 $\Delta L$를 그려 보면, 대체로 entropy가 줄어든 경우 loss도 줄어든다. 즉, Tent의 최적화 목표가 실제 task error reduction과 무관한 것이 아니라는 점을 뒷받침한다.

다음으로 ablation study에서는 normalization과 transformation 둘 다 중요하다고 보인다. normalization을 업데이트하지 않으면 성능이 악화되고, transformation parameter를 업데이트하지 않으면 사실상 BN baseline과 동일해진다. 마지막 layer만 업데이트하는 것은 일부 개선은 있지만 계속 최적화하면 오히려 나빠지고, 전체 모델 파라미터 $\theta$를 업데이트하는 것은 source-only보다도 좋아지지 못했다고 보고한다. 이는 Tent의 parameterization choice가 핵심이라는 의미다.

또한 adaptation이 point-specific overfitting이 아닌지도 점검한다. target train에서 적응한 뒤 target test에서 평가했을 때도 error가 더 줄어든다. 예를 들어 CIFAR-100-C는 37.3%에서 34.2%로, SVHN-to-MNIST는 8.2%에서 6.5%로 감소한다. 따라서 Tent가 단순히 적응에 사용된 샘플에만 맞춘 것이 아니라, target distribution 전체에 어느 정도 일반화되는 modulation을 학습한다고 해석할 수 있다.

마지막으로 feature visualization에서는 BN이 corrupted feature를 원래 source reference feature 쪽으로 되돌리는 반면, Tent는 oracle과 더 비슷한 방향으로 feature를 움직인다. 이는 Tent가 단순한 statistics correction을 넘어서 task-specific adaptation을 수행하고 있다는 분석적 근거다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 매우 명확하고 실용적이라는 점이다. source data가 없고 training도 바꿀 수 없는 상황은 실제 배포 환경에서 충분히 현실적이다. 저자들은 이 setting을 단순한 이론적 변형이 아니라, availability, efficiency, accuracy라는 관점에서 설득력 있게 정당화한다.

두 번째 강점은 방법이 매우 간단하면서도 강력하다는 점이다. Tent는 복잡한 auxiliary network, generative model, memory bank, pseudo-label selection scheme 없이도 동작한다. loss는 entropy 하나뿐이고, 업데이트 대상도 normalization affine parameter로 제한되어 있다. 이런 단순성은 구현과 적용이 쉽고, 계산 효율도 좋다는 장점으로 이어진다.

세 번째 강점은 training을 수정하지 않는다는 것이다. 많은 test-time adaptation 방법은 training 단계에서 미리 adaptation-friendly representation을 만들어야 한다. 하지만 Tent는 pretrained model만 있으면 적용할 수 있으므로, 기존 모델에 사후적으로 붙이기 쉽다. 논문이 “If the model can be run, it can be adapted.”라고 말하는 이유가 여기에 있다.

네 번째 강점은 실험 범위가 넓다는 점이다. corruption robustness, digit adaptation, semantic segmentation, VisDA-C, alternative architectures까지 포함하여 방법의 일반성을 입증하려고 한다. 특히 convolutional model 외에 SAN, MDEQ 같은 다른 구조에도 적용해 error를 줄였다는 점은 parameterization의 보편성을 어느 정도 뒷받침한다.

하지만 한계도 분명하다. 가장 본질적인 한계는 entropy minimization이 항상 올바른 방향의 적응을 보장하지 않는다는 점이다. 모델이 잘못된 high-confidence prediction으로 수렴할 가능성은 이론적으로 존재한다. 논문은 배치 공통 파라미터를 업데이트하여 trivial collapse를 어느 정도 피하지만, 더 어려운 shift에서는 여전히 실패 사례가 나온다. MNIST-to-SVHN에서 error가 증가한 사례는 바로 이런 한계를 보여준다.

또한 Tent는 **배치 단위 최적화에 의존**한다. 논문 스스로도 entropy loss는 한 점씩 episodic하게 업데이트하기 어렵고, batch가 있어야 한다고 말한다. 이는 streaming 환경에서 batch 구성이 어렵거나, 샘플 하나씩 순차적으로 처리해야 하는 경우 제약이 될 수 있다. segmentation 예제에서는 한 이미지 안의 픽셀 집합을 batch처럼 해석해 이 문제를 우회하지만, 모든 문제에 일반적으로 통하는 해결책은 아니다.

또 다른 한계는 업데이트 가능한 파라미터 공간이 제한적이라는 점이다. Tent는 안정성과 효율을 위해 modulation만 업데이트하지만, 더 큰 shift에서는 이것만으로 충분하지 않을 수 있다. 실제로 VisDA-C에서는 더 많은 layer를 업데이트했을 때 성능이 더 좋아졌고, 저자들도 SHOT의 parameterization이 큰 shift에는 더 적합할 수 있다고 인정한다. 즉, Tent의 parameterization은 매우 실용적이지만, 표현력 측면에서는 제한이 있다.

비판적으로 보면, 이 논문은 entropy minimization이 효과적인 경우를 강하게 보여주지만, **언제 실패하는지에 대한 이론적 조건**은 충분히 정리하지 않는다. 또한 calibration과 entropy objective의 관계를 discussion에서 언급만 하고 깊게 다루지는 않는다. 예측 confidence가 잘 calibration되지 않은 모델에서는 entropy minimization이 오히려 위험할 수 있는데, 이 점은 후속 연구가 더 다뤄야 할 부분이다.

## 6. 결론

이 논문은 fully test-time adaptation이라는 중요한 설정을 명확히 제안하고, 이를 위한 간단하면서도 효과적인 방법인 Tent를 제시한다. Tent의 핵심은 테스트 시점에 target batch의 normalization statistics를 다시 추정하고, normalization layer의 channel-wise affine parameters를 prediction entropy minimization으로 업데이트하는 것이다. 이 방식은 source data 없이, training 변경 없이, 그리고 매우 적은 추가 계산만으로 distribution shift에 적응할 수 있게 한다.

실험적으로 Tent는 CIFAR-10/100-C, ImageNet-C, SVHN-to-MNIST류 적응, GTA-to-Cityscapes segmentation, VisDA-C 등 다양한 설정에서 strong baseline들을 능가하거나 경쟁력 있는 성능을 보였다. 특히 ImageNet-C에서 robust training 계열 state-of-the-art를 넘어서는 결과를, 오직 test-time adaptation만으로 달성한 점은 이 논문의 가장 인상적인 성과다.

이 연구의 의미는 단순히 하나의 기법을 제안한 데 그치지 않는다. 오히려 “배포 이후, 모델은 자기 예측만을 이용해 스스로를 계속 조정할 수 있는가?”라는 질문을 본격적으로 연구 가능한 형태로 제시했다는 점이 더 중요하다. 실제 응용에서는 자율주행, 로보틱스, 의료영상, 감시 시스템처럼 환경 변화가 잦고 source data 재접근이 어려운 분야에서 매우 큰 가치를 가질 수 있다.

향후 연구 방향도 자연스럽다. 더 어려운 shift에 대한 robustness, batch 의존성을 줄이는 더 정교한 loss 설계, calibration과 adaptation의 관계, modulation보다 더 강력하지만 여전히 안정적인 parameterization 탐색 등이 이어질 수 있다. 그런 의미에서 Tent는 단순한 기법을 넘어, **test-time adaptation 연구의 출발점을 잘 정의한 논문**이라고 평가할 수 있다.
