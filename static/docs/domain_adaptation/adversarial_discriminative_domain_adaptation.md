# Adversarial Discriminative Domain Adaptation

* **저자**: Eric Tzeng, Judy Hoffman, Kate Saenko, Trevor Darrell
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1702.05464](https://arxiv.org/abs/1702.05464)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 다룬다. 구체적으로는, 라벨이 있는 source domain 데이터로 학습한 분류기를 라벨이 없는 target domain에도 잘 동작하게 만드는 것이 목표다. 현실에서는 학습 데이터와 실제 적용 데이터의 분포가 다른 경우가 많고, 이 차이를 보통 **domain shift** 또는 **dataset bias**라고 부른다. 예를 들어 손글씨 숫자 데이터셋끼리도 이미지 해상도, 배경, 필기 스타일이 다를 수 있고, 더 나아가 RGB 이미지와 depth 이미지처럼 센서 자체가 다르면 분포 차이는 훨씬 커진다.

기존의 domain adaptation 연구는 source와 target의 feature distribution 차이를 줄이기 위해 MMD, CORAL, reconstruction, adversarial learning 등을 사용해 왔다. 그러나 논문은 기존 adversarial adaptation 방법들이 몇 가지 한계를 가진다고 본다. generative 방법은 시각적으로 흥미로운 결과를 만들 수 있지만 최종 목표가 분류라면 불필요하게 무거울 수 있고, discriminative 방법 중 일부는 source와 target에 **shared weights**를 강제하여 충분한 유연성을 확보하지 못한다. 또한 GAN 스타일의 loss를 직접 활용하지 않은 접근도 있었다.

이 논문의 핵심 문제의식은 다음과 같다. **분류가 목표라면 굳이 이미지 자체를 생성할 필요가 있는가?** 그리고 **source와 target의 저수준 특징이 많이 다를 때도 같은 encoder를 강제로 공유해야 하는가?** 저자들은 이 질문에 대해 부정적으로 답하며, 분류 중심의 표현 학습과 비대칭적 매핑을 결합한 새로운 방법을 제안한다. 그 결과물이 **ADDA (Adversarial Discriminative Domain Adaptation)** 이다.

이 문제는 중요하다. 라벨이 부족한 새로운 데이터셋, 실제 서비스 환경의 데이터, 다른 센서나 다른 수집 조건에서 들어오는 데이터에 대해 매번 충분한 라벨을 확보하는 것은 비용이 크기 때문이다. 따라서 source에서 배운 분류 능력을 target으로 잘 옮길 수 있다면 실제 적용 가치가 높다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 명확하다. 먼저 source domain에서 **분류에 유리한 discriminative feature space**를 학습한다. 그 다음 target domain의 encoder를 따로 두고, 이 encoder가 만든 target feature가 source feature처럼 보이도록 학습한다. 이때 target encoder는 domain discriminator를 속이도록 훈련된다. 즉, discriminator는 "이 feature가 source에서 왔는지 target에서 왔는지"를 구분하려 하고, target encoder는 자신의 출력이 source처럼 보이게 만들어 discriminator를 헷갈리게 한다.

중요한 점은 이 논문이 세 가지 설계 선택을 조합해 ADDA를 만든다는 것이다.

첫째, **generative model이 아니라 discriminative model**을 쓴다. 저자들은 domain adaptation의 최종 목적이 이미지 생성이 아니라 분류 성능 향상이라고 본다. 따라서 이미지 생성 능력을 학습하는 데 필요한 많은 파라미터와 학습 부담은 꼭 필요하지 않다고 주장한다.

둘째, **source와 target의 weights를 공유하지 않는다 (unshared / untied weights)**. 이는 source와 target의 저수준 특징이 꽤 다를 수 있다는 가정을 반영한다. 예를 들어 RGB와 depth는 입력의 통계와 시각 구조가 다르기 때문에, 같은 encoder를 강제로 공유하는 것이 오히려 불리할 수 있다. 대신 source encoder를 먼저 학습한 뒤, 그것을 초기값으로 사용하여 target encoder를 따로 적응시키는 편이 더 유연하다.

셋째, adversarial objective로 **GAN loss**를 사용한다. 기존의 minimax 방식은 discriminator가 너무 빨리 강해지면 gradient가 약해질 수 있는데, GAN의 inverted label loss는 target encoder에 더 강한 학습 신호를 준다.

이 논문의 차별점은 단순히 새로운 모델 하나를 던지는 데 있지 않다. 먼저 기존 adversarial adaptation 방법들을 하나의 **generalized framework** 안에서 정리한 뒤, 그 프레임워크에서 설계 요소를 분해해 비교하고, 그 조합으로 ADDA를 제안한다. 논문이 정리한 비교 축은 크게 다음 세 가지다.

* base model이 discriminative인지 generative인지
* source/target weights를 shared로 둘지 unshared로 둘지
* adversarial loss를 minimax, confusion, GAN 중 무엇으로 둘지

논문은 이 관점에서 기존 방법들을 재배치한다. 예를 들어 Gradient Reversal은 discriminative + shared + minimax, Domain Confusion은 discriminative + shared + confusion, CoGAN은 generative + unshared + GAN으로 볼 수 있다. ADDA는 discriminative + unshared + GAN이라는 조합이다. 이 조합은 당시로서는 명시적으로 탐구되지 않았던 설계라고 저자들은 주장한다.

## 3. 상세 방법 설명

ADDA를 이해하려면 먼저 논문이 제시한 일반화된 adversarial adaptation 프레임워크를 보는 것이 중요하다.

문제 설정은 다음과 같다. source domain에는 입력 이미지 $\mathbf{X}_s$와 라벨 $Y_s$가 있다. target domain에는 입력 이미지 $\mathbf{X}_t$만 있고 라벨은 없다. 목표는 source에서 배운 분류 능력을 target에도 적용할 수 있도록 target representation $M_t$를 학습하는 것이다. 논문은 source mapping을 $M_s$, source classifier를 $C_s$라고 두고, feature space가 잘 정렬되면 별도의 target classifier를 새로 학습하지 않아도 $C_s$를 그대로 target에 적용할 수 있다고 본다. 즉, 실질적으로 $C=C_s=C_t$처럼 사용한다.

먼저 source encoder와 classifier는 표준 supervised classification loss로 학습한다. 논문에 제시된 식을 정리하면 다음과 같다.

$$
\mathcal{L}_{\text{cls}}(\mathbf{X}_{s},Y_{s}) = \mathbb{E}_{(\mathbf{x}_{s},y_{s})\sim(\mathbf{X}_{s},Y_{s})} \left[ -\sum_{k=1}^{K}\mathbb{1}_{[k=y_{s}]}\log C(M_{s}(\mathbf{x}_{s})) \right]
$$

이 식은 일반적인 다중 클래스 cross-entropy다. 의미는 단순하다. source 이미지 $\mathbf{x}_s$를 encoder $M_s$로 feature로 바꾸고, classifier $C$가 정답 클래스에 높은 확률을 주도록 학습한다.

그 다음 adversarial adaptation 단계에서는 domain discriminator $D$를 둔다. $D$는 입력 feature가 source에서 온 것인지 target에서 온 것인지 예측한다. discriminator의 loss는 다음과 같다.

$$
\mathcal{L}_{\text{adv}_{D}}(\mathbf{X}_{s},\mathbf{X}_{t},M_{s},M_{t}) = -\mathbb{E}_{\mathbf{x}_{s}\sim\mathbf{X}_{s}}[\log D(M_{s}(\mathbf{x}_{s}))] -\mathbb{E}_{\mathbf{x}_{t}\sim\mathbf{X}_{t}}[\log(1-D(M_{t}(\mathbf{x}_{t})))]
$$

여기서 $D(M_s(\mathbf{x}_s))$는 source feature를 source라고 맞히는 확률이고, $D(M_t(\mathbf{x}_t))$는 target feature를 source라고 볼 확률로 해석할 수 있다. 따라서 discriminator는 source를 1, target을 0으로 잘 구분하도록 학습된다.

논문은 adversarial adaptation 전체를 다음처럼 쓴다.

$$
\min_{D}\mathcal{L}_{\text{adv}_{D}}(\mathbf{X}_{s},\mathbf{X}_{t},M_{s},M_{t})
$$

$$
\min_{M_s,M_t}\mathcal{L}_{\text{adv}_{M}}(\mathbf{X}_{s},\mathbf{X}_{t},D)
\quad
\text{s.t.}\ \psi(M_s,M_t)
$$

여기서 $\psi(M_s,M_t)$는 source와 target encoder 사이의 제약 조건이다. 예를 들어 모든 레이어를 공유하면 완전한 shared setting이고, 일부만 공유하면 partially shared, 아무것도 공유하지 않으면 unshared가 된다. 논문은 각 레이어 $\ell_i$에 대해 제약을 따로 둘 수 있다고 설명한다.

$$
\psi(M_s,M_t)\triangleq {\psi_{\ell_i}(M_s^{\ell_i}, M_t^{\ell_i})}_{i\in{1,\dots,n}}
$$

가장 흔한 제약은 레이어별 동일성이다.

$$
\psi_{\ell_i}(M_s^{\ell_i}, M_t^{\ell_i}) = (M_s^{\ell_i}=M_t^{\ell_i})
$$

이것은 곧 weight sharing을 의미한다. 기존 discriminative adversarial adaptation 방법들은 이런 shared mapping을 많이 썼다. 하지만 논문은 shared mapping이 파라미터 수는 줄여주고 target도 source처럼 discriminative하게 만들 가능성은 있지만, 두 도메인을 하나의 encoder가 동시에 처리해야 하므로 최적화가 잘 안 될 수 있다고 지적한다. 특히 두 도메인의 저수준 특징 차이가 크면 같은 encoder를 강제하는 것이 불리할 수 있다.

이제 adversarial mapping loss $\mathcal{L}_{\text{adv}_M}$를 본다. 기존 Gradient Reversal 스타일은 discriminator loss를 직접 뒤집는다.

$$
\mathcal{L}_{\text{adv}_{M}} = -\mathcal{L}_{\text{adv}_{D}}
$$

이것은 GAN의 원래 minimax objective와 비슷하지만, discriminator가 빨리 수렴하면 gradient가 약해질 수 있다. 그래서 GAN에서는 보통 generator를 inverted label loss로 학습한다. 논문은 이를 adaptation에 적용해 다음 loss를 사용한다.

$$
\mathcal{L}_{\text{adv}_{M}}(\mathbf{X}_{s},\mathbf{X}_{t},D)
=

-\mathbb{E}_{\mathbf{x}_{t}\sim\mathbf{X}_{t}}[\log D(M_t(\mathbf{x}_{t}))]
$$

이 식의 의미는 명확하다. target encoder $M_t$는 discriminator가 target feature를 보고도 그것을 source라고 믿게 만들도록 학습된다. 따라서 target feature distribution이 source feature distribution에 가까워진다.

논문은 또 다른 대안으로 domain confusion loss도 소개한다.

$$
\mathcal{L}_{\text{adv}_{M}}(\mathbf{X}_{s},\mathbf{X}_{t},D)
=

-\sum_{d\in{s,t}}
\mathbb{E}_{\mathbf{x}_{d}\sim\mathbf{X}_{d}}
\left[
\frac{1}{2}\log D(M_d(\mathbf{x}_d))
+
\frac{1}{2}\log (1-D(M_d(\mathbf{x}_d)))
\right]
$$

이 loss는 discriminator 출력이 source와 target 모두에 대해 $0.5$에 가깝도록 만든다. 다만 ADDA는 이것이 아니라 앞의 GAN loss를 선택한다. 이유는 ADDA가 source encoder를 고정하고 target encoder만 움직이는 구조이므로, 고정된 source distribution을 target이 따라가는 GAN 상황과 더 유사하기 때문이다.

정리하면 ADDA의 실제 학습 절차는 **2단계 sequential training**이다.

첫 번째 단계는 source pretraining이다. source encoder $M_s$와 classifier $C$를 source 라벨 데이터로 supervised learning 한다. 여기서 source에서 좋은 discriminative feature space를 만든다.

두 번째 단계는 adversarial adaptation이다. source encoder $M_s$는 **고정(fixed)** 하고, target encoder $M_t$를 source encoder의 파라미터로 초기화한 뒤 학습한다. 동시에 discriminator $D$를 학습한다. discriminator는 source feature와 target feature를 구분하고, target encoder는 discriminator를 속여 target feature가 source처럼 보이게 만든다.

테스트 시에는 target 이미지 $\mathbf{x}_t$를 target encoder $M_t$에 통과시켜 feature를 만들고, 그 feature를 source classifier $C$로 분류한다. 즉, target용 classifier를 별도로 학습하지 않는다.

이 구조에서 중요한 설계 선택의 의미를 쉽게 말하면 다음과 같다.

source encoder는 "이미 잘 훈련된 기준 feature space" 역할을 한다. discriminator는 "이 feature가 기준 space와 닮았는지 감시하는 장치" 역할을 한다. target encoder는 "target 이미지를 그 기준 space에 맞게 투영하는 장치"다. 결국 ADDA는 **분류에 유용한 source feature space를 기준 좌표계로 삼고, target을 그 좌표계에 맞추는 방법**이라고 이해할 수 있다.

## 4. 실험 및 결과

논문은 네 가지 domain shift에서 ADDA를 평가한다. 세 가지는 digits adaptation이고, 하나는 RGB-to-depth modality adaptation이다. 저자들이 특히 강조하는 점은, ADDA가 단순한 digit 데이터셋뿐 아니라 더 어려운 cross-modality 상황에서도 효과를 보인다는 것이다.

### 4.1 Digits adaptation: MNIST, USPS, SVHN

이 실험은 10개 숫자 클래스를 가진 세 데이터셋 사이의 adaptation이다. 평가 방향은 다음 세 가지다.

* MNIST $\rightarrow$ USPS
* USPS $\rightarrow$ MNIST
* SVHN $\rightarrow$ MNIST

MNIST와 USPS 사이의 실험에서는 기존 protocol을 따라 MNIST에서 2000장, USPS에서 1800장을 샘플링했다. SVHN $\rightarrow$ MNIST에서는 full training set을 사용했다. 모든 실험은 target label을 사용하지 않는 **unsupervised setting**이다.

기본 분류 backbone은 Caffe의 modified LeNet이다. ADDA의 discriminator는 fully connected 3층 구조이며, 앞의 두 층은 각각 hidden unit 500개와 ReLU를 사용하고 마지막에 domain output을 둔다.

정량 결과는 Table 2에 제시되어 있다.

* **Source only**

  * MNIST $\rightarrow$ USPS: $0.752 \pm 0.016$
  * USPS $\rightarrow$ MNIST: $0.571 \pm 0.017$
  * SVHN $\rightarrow$ MNIST: $0.601 \pm 0.011$

* **Gradient reversal**

  * MNIST $\rightarrow$ USPS: $0.771 \pm 0.018$
  * USPS $\rightarrow$ MNIST: $0.730 \pm 0.020$
  * SVHN $\rightarrow$ MNIST: $0.739$

* **Domain confusion**

  * MNIST $\rightarrow$ USPS: $0.791 \pm 0.005$
  * USPS $\rightarrow$ MNIST: $0.665 \pm 0.033$
  * SVHN $\rightarrow$ MNIST: $0.681 \pm 0.003$

* **CoGAN**

  * MNIST $\rightarrow$ USPS: $0.912 \pm 0.008$
  * USPS $\rightarrow$ MNIST: $0.891 \pm 0.008$
  * SVHN $\rightarrow$ MNIST: 수렴 실패

* **ADDA**

  * MNIST $\rightarrow$ USPS: $0.894 \pm 0.002$
  * USPS $\rightarrow$ MNIST: $0.901 \pm 0.008$
  * SVHN $\rightarrow$ MNIST: $0.760 \pm 0.018$

이 결과를 보면, 쉬운 adaptation인 MNIST와 USPS 사이에서는 ADDA가 CoGAN과 비슷한 수준의 높은 성능을 보인다. MNIST $\rightarrow$ USPS에서는 CoGAN이 약간 높지만, USPS $\rightarrow$ MNIST에서는 ADDA가 오히려 가장 높다. 더 어려운 SVHN $\rightarrow$ MNIST에서는 ADDA가 다른 discriminative adversarial 방법들보다 높고, CoGAN은 아예 수렴하지 못했다.

이 결과가 의미하는 바는 분명하다. 첫째, **이미지 생성기가 없어도** domain adaptation이 충분히 잘 될 수 있다. 둘째, 도메인 차이가 클수록 generative coupled modeling은 학습이 불안정할 수 있는데, ADDA처럼 discriminative하고 단순한 구조가 더 실용적일 수 있다. 셋째, source-only 대비 향상 폭이 크기 때문에 adversarial alignment 자체가 target recognition에 실질적인 도움이 됨을 보여준다.

### 4.2 Modality adaptation: NYUD RGB $\rightarrow$ Depth

두 번째 실험은 훨씬 어렵다. NYU depth dataset을 사용해 RGB 이미지를 source, depth 이미지를 target으로 두는 **cross-modality adaptation**을 수행한다. 데이터셋에는 실내 장면의 19개 object class bounding box annotation이 있다. 저자들은 bounding box를 crop하여 19-way object classification task를 만든다.

중요한 데이터 분할은 다음과 같다.

* source domain: train split의 RGB 이미지
* target domain: val split의 depth 이미지
* source labeled images: 2,186장
* target unlabeled images: 2,401장

이 분할은 동일 instance가 source와 target 양쪽에 동시에 등장하지 않도록 하기 위한 것이다. target 쪽 depth 이미지는 HHA encoded depth로 표현된다. 논문은 tight bounding box와 낮은 해상도 때문에 이 task 자체가 어렵고, bathtub나 toilet처럼 샘플 수가 매우 적은 클래스도 있어 성능이 낮을 수 있다고 설명한다.

이 실험의 backbone은 VGG-16이며, ImageNet pretrained weight로 초기화한다. source domain에서 batch size 128로 20,000 iteration fine-tuning한 뒤, ADDA adaptation을 batch size 128로 추가 20,000 iteration 수행한다. discriminator는 fully connected 3층 구조로 hidden dimension이 1024, 2048, 그리고 최종 출력이며, 중간 층은 ReLU를 사용한다.

논문은 class imbalance가 크기 때문에 per-class accuracy를 보고한다. 전체 평균 정확도는 다음과 같다.

* **Source only**: $13.9%$
* **ADDA**: $21.1%$
* **Train on target**: $46.8%$

즉 ADDA는 source-only 대비 평균 per-category accuracy를 $13.9%$에서 $21.1%$로 크게 올린다. 논문은 이를 **50% 이상 상대적 향상**이라고 표현한다. 실제로 $\frac{21.1 - 13.9}{13.9} \approx 0.518$이므로 약 51.8% 상대 향상이다.

세부 클래스별 결과를 보면 개선 폭이 큰 항목도 있다. 예를 들어 **counter** 클래스는 $2.9%$에서 $44.7%$로 크게 상승한다. chair, lamp, table, television 등도 개선된다. 하지만 모든 클래스가 좋아지는 것은 아니다. 논문이 명시적으로 말하듯, adaptation 전에도 정답을 하나도 맞히지 못한 클래스 중 일부는 adaptation 후에도 회복되지 못했다. 또한 **pillow**와 **night stand**는 오히려 성능이 감소했다.

논문은 confusion matrix 분석도 제공한다. source-only 모델은 대부분의 샘플을 **pillow**로 잘못 예측하는 경향이 있다. 그래서 pillow 클래스만 이상하게 정확도가 높고 나머지는 전반적으로 성능이 나쁘다. 반면 ADDA 이후에는 훨씬 다양한 클래스를 예측하게 되며, 이로 인해 pillow 정확도는 내려가지만 더 많은 클래스에서 성능이 올라간다. 또 oracle에 해당하는 “train on target” 모델과 비교하면, ADDA의 오분류는 chair와 table 사이의 혼동처럼 어느 정도 타당한 구조를 보인다고 설명한다. 이는 ADDA가 depth 이미지에 대해서도 의미 있는 표현을 학습하고 있음을 시사한다.

### 4.3 실험 결과의 해석

전체 실험은 ADDA의 설계 선택을 꽤 설득력 있게 뒷받침한다. digits처럼 비교적 단순한 도메인 변화뿐 아니라 RGB와 depth처럼 modality 차이가 큰 경우에도 효과가 나타난다. 특히 CoGAN이 큰 domain shift에서 수렴하지 못했다는 보고는, image generation 중심 접근이 항상 유리한 것은 아니라는 논문의 주장과 잘 맞는다.

또한 결과는 단순히 accuracy 숫자만 보여주는 것이 아니라, source-only representation이 특정 클래스 편향에 빠져 있다는 점, adaptation 후에는 더 균형 잡힌 예측 분포가 나온다는 점을 보여준다. 이는 ADDA가 단순한 score trick이 아니라 feature space 자체를 더 target-friendly하게 만든다는 논문의 메시지를 보강한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **개념적 정리와 실용적 방법 제안이 동시에 이루어진다**는 점이다. 많은 논문이 새로운 방법만 제시하거나, 반대로 survey처럼 정리만 하는데, 이 논문은 adversarial adaptation 분야를 세 가지 설계 축으로 재구성하고 그 위에서 ADDA를 제안한다. 그래서 독자는 기존 방법들 사이의 관계를 더 잘 이해할 수 있다.

또 다른 강점은 **방법이 단순하다**는 점이다. ADDA는 source supervised pretraining과 target adversarial adaptation이라는 2단계 절차로 요약된다. 별도의 image generator나 reconstruction branch가 필요 없고, 테스트 시에도 source classifier를 그대로 재사용한다. 실험 결과도 이 단순성이 성능 저하로 이어지지 않음을 보여준다. 오히려 몇몇 설정에서는 더 강력하다.

세 번째 강점은 **비대칭적 매핑의 타당성을 실험으로 보여준다**는 점이다. shared weights가 항상 정답이라는 암묵적 가정을 깨고, target encoder를 별도로 두는 것이 큰 domain shift에서 유리할 수 있음을 보여준다. 이는 이후 많은 domain adaptation 연구에서 encoder를 완전히 공유하지 않는 방향에 영향을 준 중요한 포인트다.

네 번째 강점은 **cross-modality adaptation** 실험이다. digits 데이터셋만으로는 방법의 일반성을 충분히 보여주기 어렵다. 그런데 RGB $\rightarrow$ depth 같은 훨씬 어려운 문제를 추가함으로써 ADDA가 단순 benchmark tuning이 아니라는 점을 강조한다.

하지만 한계도 분명하다. 첫째, 논문은 feature alignment가 왜 분류 경계를 잘 보존하는지에 대해 깊은 이론적 분석을 제공하지 않는다. 도메인 분포가 feature 공간에서 겹친다고 해서 클래스 조건부분포까지 자동으로 잘 맞는 것은 아니다. 실제로 일부 클래스는 성능이 좋아지지 않거나 악화된다. 즉, **domain invariance와 class discriminativeness 사이의 긴장 관계**가 완전히 해결된 것은 아니다.

둘째, target encoder를 source encoder로 초기화하고 source encoder를 고정하는 전략은 안정적이지만, 동시에 source feature space 자체가 최적인지 재검토하지 않는다. 만약 source에서 배운 feature space가 target에 충분히 적합하지 않다면, source를 완전히 고정하는 것은 적응 한계를 만들 수 있다. 논문도 이 선택을 주로 GAN setting과 유사하다는 직관으로 정당화하지, 대안과의 광범위한 ablation을 제시하지는 않는다.

셋째, 실험은 당시 기준으로 설득력 있지만, 세부적인 학습 안정성이나 하이퍼파라미터 민감도에 대한 분석은 많지 않다. 예를 들어 discriminator 구조, adaptation iteration 수, 어느 레이어까지 분리하는지가 성능에 얼마나 민감한지는 본문에 충분히 드러나지 않는다.

넷째, NYUD 실험에서 class imbalance가 매우 크고 일부 클래스는 샘플 수가 매우 적다. 논문은 이를 인지하고 per-class accuracy를 보고하지만, 이런 극단적 imbalance 환경에서 adversarial alignment가 소수 클래스에 미치는 영향은 충분히 해부되지 않았다. 실제로 일부 클래스는 개선되지 못한다.

다섯째, 논문은 ADDA가 source classifier를 target에 그대로 적용한다고 설명하지만, class-conditional alignment를 직접적으로 강제하지는 않는다. 따라서 도메인은 맞춰졌지만 클래스가 섞이는 현상이 있을 가능성을 배제하지 못한다. 이는 이후 conditional adversarial adaptation, class-aware alignment 같은 방향의 연구 필요성을 암시한다.

종합하면, 이 논문은 매우 강한 실용적 아이디어를 제시하지만, **무엇을 align해야 하는가**에 대한 질문에는 아직 feature distribution 수준의 답만 제시한다. 그럼에도 당시 adversarial domain adaptation의 중요한 전환점으로 볼 수 있다.

## 6. 결론

이 논문은 adversarial unsupervised domain adaptation 방법들을 하나의 통합된 틀로 정리하고, 그 위에서 새로운 방법인 **ADDA**를 제안한다. ADDA의 핵심은 source domain에서 discriminative feature space를 먼저 학습한 뒤, target encoder를 adversarial하게 학습해 그 feature space에 맞추는 것이다. 이 과정에서 **discriminative base model**, **unshared weights**, **GAN loss**라는 세 가지 선택을 결합한다.

실험적으로 ADDA는 digits adaptation에서 강력한 성능을 보였고, 특히 SVHN $\rightarrow$ MNIST 같은 더 어려운 shift에서도 경쟁력 있는 결과를 냈다. 또한 RGB에서 depth로 넘어가는 cross-modality object classification에서도 source-only 대비 큰 성능 향상을 보여, 단순한 도메인 차이뿐 아니라 modality gap에도 일정 부분 대응 가능함을 보여주었다.

이 연구의 중요성은 두 가지 차원에서 볼 수 있다. 하나는 실용적인 차원으로, 라벨이 없는 새로운 도메인에 대해 단순하고 효과적인 adaptation 절차를 제공했다는 점이다. 다른 하나는 연구적 차원으로, adversarial adaptation의 설계 공간을 명확히 분해하여 이후 연구들이 어떤 축을 더 발전시켜야 하는지 보여주었다는 점이다. 실제로 이후 연구들은 ADDA의 기본 아이디어를 바탕으로 class-conditional alignment, pixel-level adaptation, self-training, entropy minimization 등 더 정교한 방향으로 확장되었다.

결국 ADDA는 “분류가 목적이면 분류에 필요한 것만 남기고, target을 source의 discriminative space에 맞추자”는 단순하지만 강력한 철학을 잘 구현한 논문이다. 원문이 보여주는 바에 따르면, 이 방법은 특히 큰 domain shift 상황에서 generative 접근보다 더 단순하면서도 효과적인 대안이 될 수 있다.
