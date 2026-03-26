# Bridging Theory and Algorithm for Domain Adaptation

* **저자**: Yuchen Zhang, Tianle Liu, Mingsheng Long, Michael I. Jordan
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1904.05801](https://arxiv.org/abs/1904.05801)

이 논문은 ICML 2019에 발표된 domain adaptation 연구로, 비지도 domain adaptation에서 오랫동안 존재해 온 “이론은 discrepancy bound를 말하지만 실제 알고리즘은 adversarial learning으로 구현된다”는 간극을 줄이려는 시도다.

## 1. 논문 개요

이 논문의 목표는 **unsupervised domain adaptation**에서 이론적 일반화 보장과 실제 알고리즘 설계를 하나의 일관된 틀로 연결하는 것이다. 전통적인 supervised learning 이론은 학습 데이터와 테스트 데이터가 같은 분포에서 온다고 가정하지만, domain adaptation은 source domain과 target domain의 분포가 다르다는 점이 본질이다. 따라서 source에서 학습한 모델이 target에서 잘 작동하려면, source에서의 성능뿐 아니라 두 도메인 사이의 분포 차이를 어떻게 측정하고 줄일지에 대한 이론이 필요하다.

기존 이론, 특히 Ben-David 계열의 domain adaptation bound는 target error를 대체로 세 항의 합으로 다룬다. 하나는 source error, 다른 하나는 두 도메인 간 discrepancy, 마지막 하나는 두 도메인을 동시에 잘 설명하는 이상적인 hypothesis의 존재 여부를 나타내는 항이다. 문제는 이 이론이 주로 $0$-$1$ loss와 labeling function 중심으로 전개되어 왔다는 점이다. 반면 실제 딥러닝 기반 알고리즘은 대개 **scoring function**, **softmax**, **margin**, **cross-entropy**, 그리고 **adversarial optimization**을 사용한다. 논문은 바로 이 지점에서 “이론의 언어”와 “알고리즘의 언어”가 맞지 않는다고 본다.

저자들이 제기하는 핵심 문제는 두 가지다. 첫째, multiclass classification에서 실제로 널리 쓰이는 scoring function 기반 분류기와 margin loss를 domain adaptation 이론이 충분히 다루지 못했다는 점이다. 둘째, 기존 discrepancy는 최적화 관점에서 너무 강한 supremum을 포함해 실제 minimax adversarial training으로 옮기기 어렵다는 점이다. 이 논문은 이를 해결하기 위해 **Margin Disparity Discrepancy (MDD)** 라는 새 discrepancy를 제안하고, 이에 대한 일반화 bound를 제시한 뒤, 그것을 직접 adversarial learning 알고리즘으로 구현한다.

이 문제가 중요한 이유는 매우 분명하다. 실제 응용에서는 라벨이 풍부한 source domain과 라벨이 없는 target domain이 함께 주어지는 경우가 많으며, 예를 들어 synthetic-to-real, Amazon-to-Webcam, clip-art-to-real-world 같은 상황에서 domain shift가 성능을 크게 떨어뜨린다. 따라서 이론적으로 설명 가능하고 실제로 잘 학습되는 adaptation 방법은 학문적으로도, 실용적으로도 중요하다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 간단히 말해 다음과 같다. **source 분류 성능을 유지하면서, target에서 source와 “다르게 보이게 만드는” 분류 경향을 adversarial하게 억제하면, target error를 줄일 수 있다.** 이를 위해 저자들은 기존의 $\mathcal{H}\Delta\mathcal{H}$-divergence를 그대로 쓰지 않고, 특정 기준 분류기 $h$ 또는 scoring function $f$를 중심에 둔 비대칭적 discrepancy를 정의한다.

먼저 binary 또는 일반 labeling-function 수준에서는 **Disparity Discrepancy (DD)** 를 정의한다. 기존 $\mathcal{H}\Delta\mathcal{H}$-divergence는 $h, h'$ 두 가설 쌍 전체에 대해 supremum을 취하므로 최적화가 어렵다. 반면 DD는 현재 분류기 $h$를 고정하고, 다른 가설 $h'$ 하나만 adversary로 움직이게 한다. 즉 “현재 분류기와 다르게 예측하는 정도”가 source보다 target에서 더 크도록 만드는 adversary를 찾는다. 이렇게 하면 supremum의 범위가 줄어들어 실제 adversarial optimization과 더 잘 맞는다.

하지만 이 논문의 진짜 핵심은 multiclass scoring function에 맞춘 **MDD**다. 여기서는 disparity를 단순한 $0$-$1$ disagreement가 아니라 **margin-based disparity**로 바꾼다. 즉 기준 분류기 $f$가 예측한 라벨 $h_f(x)$를 기준으로, 다른 scoring function $f'$가 그 라벨에 대해 얼마나 작은 margin을 주는지 본다. source에서는 그 라벨을 잘 지지하도록 만들고, target에서는 그 라벨에 대해 불리한 margin을 만들도록 adversary가 움직인다. 이 차이를 최대로 만드는 값이 MDD다.

이 설계의 차별점은 세 가지다. 첫째, 실제 딥러닝 분류기가 사용하는 scoring function과 margin 개념을 이론에 직접 넣었다. 둘째, discrepancy가 비대칭적이지만 domain adaptation bound를 세우는 데 충분하며, 오히려 adversarial training과 자연스럽게 연결된다. 셋째, 최적화해야 하는 adversary가 “현재 classifier를 기준으로 얼마나 반대되는 판단을 만들 수 있는가”를 학습하므로, 기존 divergence보다 훨씬 구현 친화적이다.

결국 이 논문은 “좋은 discrepancy란 단지 분포 차이를 수학적으로 측정하는 양이 아니라, 실제 신경망 학습에서 adversary로 구현 가능한 양이어야 한다”는 입장을 취한다. 그 점에서 MDD는 이론과 알고리즘 사이의 접점을 의도적으로 설계한 개념이라고 볼 수 있다.

## 3. 상세 방법 설명

논문의 방법론은 세 단계로 이해할 수 있다. 첫째, multiclass scoring function과 margin loss를 정의한다. 둘째, 이를 기반으로 target error upper bound를 세운다. 셋째, 그 bound를 직접 최소화하는 adversarial learning 알고리즘으로 바꾼다.

### 3.1 기본 분류 설정과 margin loss

논문은 multiclass 분류기 $f:\mathcal{X}\to\mathbb{R}^k$를 사용한다. 각 클래스별 score를 출력하고, 최종 예측 라벨은 가장 큰 score를 갖는 클래스로 정한다. 즉,
$$h_f(x)=\arg\max_{y\in\mathcal{Y}} f(x,y)$$
이다.

한 샘플 $(x,y)$에 대한 margin은 “정답 클래스 score가 가장 높은 오답 클래스 score보다 얼마나 큰가”를 뜻하며,
$$\rho_f(x,y)=\frac{1}{2}\left(f(x,y)-\max_{y'\neq y} f(x,y')\right)$$
로 정의된다.

margin loss는 margin이 충분히 크면 $0$, 작으면 선형으로 커지고, 음수이면 $1$이 되는 함수 $\Phi_\rho$를 사용한다. 직관적으로는 “정답을 맞췄는지”만 보는 $0$-$1$ loss보다 “얼마나 여유 있게 맞췄는지”까지 반영한다. 이것이 중요한 이유는, 딥러닝 분류기에서는 단순 정오보다 decision boundary와의 거리, 즉 margin이 일반화 성능과 더 밀접하게 연결되기 때문이다.

### 3.2 Disparity와 MDD의 정의

두 labeling function $h, h'$ 사이의 disparity는 같은 입력에 대해 서로 다르게 예측하는 비율이다. 이를 source와 target 각각에서 계산할 수 있다. 저자들은 기준 분류기 $h$를 고정한 뒤,
$$d_{h,\mathcal{H}}(P,Q)=\sup_{h'\in\mathcal{H}}\left(\mathrm{disp}_Q(h',h)-\mathrm{disp}_P(h',h)\right)$$
를 정의한다. 이것이 DD다.

직관적으로는 “현재 분류기와 일부 adversarial classifier가 target에서 더 많이 충돌하고 source에서는 덜 충돌하게 만들 수 있는 정도”다. 만약 이런 값이 크다면, 현재 분류기가 source에서 학습한 판단 기준이 target에서는 안정적이지 않다는 뜻이다.

이제 scoring function과 margin loss에 맞추어 이를 확장한 것이 MDD다. 먼저 기준 함수 $f$와 adversarial 함수 $f'$에 대해 margin disparity를 정의한다. 여기서는 정답 라벨 대신 기준 분류기 $h_f(x)$가 준 라벨을 surrogate label처럼 사용한다. 즉 adversary $f'$가 그 라벨에 대해 얼마나 낮은 margin을 주는지를 본다. 그리고 source와 target에서의 margin disparity 차이를 최대로 만드는 adversary를 찾는다.

결과적으로 MDD는
$$d_{f,\mathcal{F}}^{(\rho)}(P,Q)=\sup_{f'\in\mathcal{F}}\Big(\mathrm{disp}^{(\rho)}_Q(f',f)-\mathrm{disp}^{(\rho)}_P(f',f)\Big)$$
로 정의된다.

이 정의는 비대칭적이다. $f$와 $f'$를 바꾸면 값이 달라진다. 하지만 저자들은 domain adaptation에 필요한 것은 반드시 대칭적 metric이 아니라, **현재 분류기 기준으로 target risk를 잘 upper-bound하는 quantity**라고 본다. 이 관점이 논문의 중요한 철학적 전환이다.

### 3.3 일반화 bound

논문이 제시한 핵심 이론 결과는 target error가 다음 세 항으로 제어된다는 것이다.

$$\mathrm{err}_Q(h_f)\le \mathrm{err}^{(\rho)}*P(f)+d^{(\rho)}*{f,\mathcal{F}}(P,Q)+\lambda$$

여기서 첫 항 $\mathrm{err}^{(\rho)}*P(f)$는 source에서의 empirical margin error다. 둘째 항 $d^{(\rho)}*{f,\mathcal{F}}(P,Q)$는 source와 target의 차이로 인해 발생하는 일반화 간극이다. 셋째 항 $\lambda$는 source와 target을 동시에 잘 설명하는 이상적인 분류기의 combined margin loss 최소값이다. 즉 adaptation 자체가 가능한 문제인지, hypothesis class가 충분히 풍부한지를 반영하는 problem-dependent constant다.

이 bound의 의미는 아주 직접적이다. 좋은 adaptation 모델이 되려면 source에서 정확해야 하고, source와 target 사이의 MDD가 작아야 하며, 두 도메인에 동시에 잘 맞는 공통 구조가 존재해야 한다. 논문은 여기에 더해 empirical MDD와 true MDD 사이의 차이를 **Rademacher complexity**로 제어한다. 따라서 최종적으로는 empirical source loss와 empirical MDD를 줄이면 target error를 줄일 수 있다는 학습 원리가 성립한다.

또한 bound에는 margin $\rho$가 직접 들어간다. 저자들은 이것이 **일반화와 최적화 사이의 trade-off**를 드러낸다고 말한다. 너무 작은 margin은 bound가 충분히 informative하지 않을 수 있고, 너무 큰 margin은 실제 최적화가 어려워질 수 있다. 즉 margin은 단순 하이퍼파라미터가 아니라 이론-실험을 동시에 관통하는 핵심 변수다.

### 3.4 알고리즘으로의 변환

이론을 알고리즘으로 옮기기 위해 저자들은 feature extractor $\psi$, main classifier $f$, auxiliary classifier $f'$를 사용한다. $\psi$는 입력을 feature space로 보내고, $f$는 실제 예측을 수행하며, $f'$는 adversary 역할을 한다.

이론적으로 최소화해야 할 것은 source margin error와 empirical MDD의 합이다. 하지만 MDD 안에는 supremum이 있으므로 minimax game이 된다. 저자들은 이를 다음과 같은 구조로 바꾼다. 먼저 $f'$는 target에서 $f$의 예측을 흔들고 source에서는 $f$의 예측을 유지하도록 학습된다. 반면 $f$와 $\psi$는 source에서 분류를 잘하면서, 동시에 $f'$가 source와 target을 구분하며 discrepancy를 키우지 못하도록 학습된다.

실제 구현에서는 margin loss를 그대로 쓰지 않고, 최적화가 쉬운 **combined cross-entropy loss**를 사용한다. source에서는 보통의 cross-entropy를 쓴다.

$$L(f(\psi(x^s)),y^s)=-\log \sigma_{y^s}(f(\psi(x^s)))$$

또한 source에서 $f'$는 $f$가 예측한 라벨을 맞히도록 학습된다.

$$L(f'(\psi(x^s)),f(\psi(x^s)))=-\log \sigma_{h_f(\psi(x^s))}(f'(\psi(x^s)))$$

반대로 target에서는 $f'$가 $f$의 예측 라벨에 낮은 확률을 주도록 수정된 adversarial loss를 쓴다.

$$L'(f'(\psi(x^t)),f(\psi(x^t)))=\log\left(1-\sigma_{h_f(\psi(x^t))}(f'(\psi(x^t)))\right)$$

그러면 discrepancy term은 대략 “target에서 기준 라벨을 무너뜨리는 정도”에서 “source에서 기준 라벨을 유지하는 정도”를 뺀 값이 된다. source 쪽 항에는 $\gamma$라는 가중치가 붙는다. 전체 objective는 $f'$에 대해서는 최대화, $f,\psi$에 대해서는 최소화된다.

이때 $\gamma=\exp(\rho)$로 두어 margin과 직접 연결한다. 저자들의 informal proposition에 따르면, 제약이 충분히 약하면 equilibrium에서 $\sigma_{h_f}(f'(\cdot))$는 $\gamma/(1+\gamma)$에 가까워지고, 그에 해당하는 margin은 $\log\gamma$가 된다. 즉 $\gamma$는 실질적으로 원하는 margin 수준을 제어하는 factor다.

실제 학습에서는 $f$에 대한 discrepancy loss의 직접 미분이 번거롭기 때문에, **gradient reversal layer (GRL)** 를 사용해 feature extractor $\psi$가 discrepancy를 줄이도록 만든다. 따라서 전체 구조는 DANN과 비슷한 adversarial network처럼 보이지만, 판별기가 source-vs-target domain classifier가 아니라 **현재 classifier의 결정 경계를 흔드는 auxiliary classifier**라는 점이 다르다.

## 4. 실험 및 결과

실험은 세 가지 대표적인 unsupervised domain adaptation 벤치마크에서 수행된다. **Office-31**, **Office-Home**, **VisDA-2017**이다. Office-31은 Amazon, Webcam, DSLR 세 도메인과 31개 클래스로 구성된 고전적 벤치마크다. Office-Home은 Artistic, Clip Art, Product, Real-world 네 도메인으로 더 어렵고 크다. VisDA-2017은 synthetic-to-real adaptation 문제로, 시뮬레이션과 실제 영상 사이의 큰 domain gap을 다룬다.

비교 대상은 DAN, DANN, ADDA, JAN, GTA, MCD, CDAN 등 당시의 주요 SOTA 방법들이다. backbone은 ResNet-50이고, feature extractor는 ImageNet 사전학습 가중치에서 fine-tuning한다. main classifier와 auxiliary classifier는 모두 폭 1024의 2-layer 네트워크다. 하이퍼파라미터는 importance-weighted cross-validation으로 고르고, $\eta$의 asymptotic 값은 0.1, $\gamma$는 데이터셋별로 ${2,3,4}$ 중 선택한다.

### 4.1 Office-31

Office-31에서 MDD는 평균 정확도 **88.9%**를 기록한다. 이는 ResNet-50 baseline의 76.1%, DAN의 80.4%, DANN의 82.2%, ADDA의 82.9%, JAN의 84.3%, GTA의 86.5%, MCD의 86.5%, CDAN의 87.7%보다 높다. 특히 A→W에서 94.5, A→D에서 93.5, D→A에서 74.6, W→A에서 72.2를 기록해 대부분의 task에서 최고 수준이다.

이 결과가 의미 있는 이유는 이전 방법들의 강점이 task 유형마다 달랐기 때문이다. 예를 들어 feature alignment 계열은 large-to-small transfer에서 강하고, pixel-level adaptation 계열은 small-to-large에서 더 유리한 경향이 있었다. 그런데 MDD는 이런 경향을 넘어 거의 모든 task에서 강한 성능을 보였다. 즉 특정 adaptation 스타일에 국한되지 않는 **범용성**을 보여준다.

### 4.2 Office-Home

Office-Home에서는 평균 정확도 **68.1%**를 기록해 CDAN의 65.8%보다 2.3%p 높다. 세부적으로도 거의 모든 방향에서 우세하다. 예를 들어 Ar→Cl 54.9, Ar→Pr 73.7, Cl→Ar 60.0, Pr→Ar 61.2, Rw→Ar 72.5 등이다. Office-Home은 도메인 간 시각적 차이가 더 크고 클래스 수도 많아 어려운 데이터셋인데, 여기서 큰 폭의 향상을 보였다는 점은 제안한 discrepancy가 복잡한 multiclass adaptation에서도 잘 작동함을 시사한다.

### 4.3 VisDA-2017

VisDA-2017에서는 **74.6%**를 기록해 JAN 61.6, MCD 69.2, GTA 69.5, CDAN 70.0을 모두 앞선다. synthetic-to-real은 실제 응용에서도 매우 중요한 설정인데, 여기는 도메인 갭이 매우 커서 단순 feature alignment만으로는 부족한 경우가 많다. MDD가 여기서도 뚜렷한 우위를 보였다는 점은, 이 논문이 단지 이론 제안에 그치지 않고 실제 큰 domain shift에서도 효과적인 학습 objective를 제공했다는 근거가 된다.

### 4.4 분석 실험

논문은 단순히 성능만 비교하지 않고, 자신들의 이론-알고리즘 연결이 실제로 성립하는지도 분석한다.

첫째, auxiliary classifier $f'$가 정말 empirical MDD를 크게 만드는 adversary 역할을 하는지 본다. minimization 없이 $f'$만 학습시키면 MDD가 빠르게 1에 가까워지는 결과를 보여, 사용한 surrogate loss가 실제 MDD 최대화와 유사하게 동작함을 시사한다.

둘째, minimax equilibrium에서 target 쪽 $\sigma_{h_f}\circ f'$ 값이 이론이 예측한 $\gamma/(1+\gamma)$에 가까워지는지 확인한다. 실험에서는 최종 학습 단계에서 target에서 이 값이 예측치에 가깝게 가며, 큰 margin이 형성됨을 보인다.

셋째, $\gamma$를 1, 2, 4 등으로 바꿔 DD, $\log 2$-MDD, $\log 4$-MDD를 비교한다. 결과는 대체로 더 큰 $\gamma$가 더 작은 MDD와 더 높은 test accuracy로 이어진다는 점을 보여준다. 예를 들어 Office-31 평균 정확도는 $\gamma=1$일 때 87.6, $\gamma=2$일 때 88.1, $\gamma=3$일 때 88.5, $\gamma=4$일 때 88.9로 증가하다가, 너무 커지면 약간 다시 떨어진다. 이는 논문이 이론적으로 말한 **margin의 이점과 최적화 불안정성 사이의 trade-off**와 잘 맞아떨어진다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **새 discrepancy를 제안했다**는 사실 자체보다, 그 discrepancy가 정확히 어떤 gap을 메우는지 분명하다는 점이다. 기존 이론은 $0$-$1$ loss와 labeling function 중심이었고, 기존 알고리즘은 scoring function, softmax, adversarial training 중심이었다. MDD는 그 둘을 이어주는 설계된 개념이다. 즉 이론의 확장도 있고, 실제 training objective로의 변환도 있다.

둘째 강점은 **multiclass scoring function에 대한 formal bound**를 제시했다는 점이다. domain adaptation 이론은 binary classification이나 symmetric loss에 머무는 경우가 많았는데, 이 논문은 margin loss와 Rademacher complexity를 통해 multiclass setting으로 분석을 확장했다. 특히 target error bound에 margin이 직접 들어가며, margin choice가 일반화와 최적화 모두에 영향을 준다고 해석한 점이 인상적이다.

셋째 강점은 **알고리즘적 구현 가능성**이다. DD와 MDD는 모두 기존 $\mathcal{H}\Delta\mathcal{H}$-divergence보다 adversarial optimization에 더 자연스럽다. 특히 supremum이 기준 classifier를 고정한 상태에서 단일 hypothesis space 위에서만 이뤄지므로, auxiliary classifier 하나로 구현하기 쉽다. 이 점은 “이론적으로 타당하지만 실제로 학습 불가능한 discrepancy”와 대비된다.

넷째는 실험적 설득력이다. Office-31, Office-Home, VisDA-2017에서 모두 당시 SOTA 또는 그 이상 수준의 성능을 보였고, 추가 분석을 통해 surrogate loss와 equilibrium behavior까지 확인했다. 즉 논문은 이론, 알고리즘, 실험이 비교적 균형 있게 맞물려 있다.

반면 한계도 있다. 먼저, bound 안의 $\lambda$는 여전히 problem-dependent constant이며 실제로 계산되거나 제어되지 않는다. 이는 domain adaptation 이론 전반의 고질적 문제이기도 하다. 즉 이 논문도 “언제 adaptation이 원천적으로 가능한가”를 완전히 해결하지는 못한다.

또한 MDD는 비대칭적 quantity다. 이 점은 논문의 목적에는 부합하지만, 전통적인 의미의 distance나 divergence로 보기에는 해석이 다소 까다롭다. 저자들도 pseudo-metric 또는 well-defined discrepancy의 성질을 논하지만, 완전한 대칭 metric은 아니다. 따라서 이 양이 언제 얼마나 안정적으로 분포 차이를 반영하는지는 추가 연구 여지가 있다.

알고리즘 측면에서는, 이론에서 출발했지만 실제 구현은 margin loss 대신 **combined cross-entropy**라는 surrogate objective를 사용한다. 저자들은 이것이 margin property를 잘 보존한다고 주장하고 분석 실험도 제시하지만, 엄밀히 말하면 “이론의 MDD”와 “실제 최적화한 loss” 사이에는 여전히 변환 단계가 존재한다. 즉 gap을 줄였지만 완전히 제거한 것은 아니다.

또 하나의 한계는 실험 범위가 주로 image classification adaptation에 한정된다는 점이다. 논문의 이론은 더 일반적인 multiclass setting을 다루지만, 실제 검증은 ResNet-50 기반 시각 도메인 적응에 집중되어 있다. 다른 구조, 다른 modality, 더 최근의 large-scale setting에서 동일한 장점이 유지되는지는 본문만으로는 알 수 없다.

비판적으로 보면, 이 논문은 “theory to algorithm”을 연결하는 데 성공했지만, 그 연결은 어디까지나 **잘 설계된 surrogate와 adversarial approximation을 통해서** 이루어진다. 그럼에도 불구하고 이는 domain adaptation 분야에서는 매우 실용적이고 의미 있는 진전이다. 완벽한 닫힌 형태의 이론적-실용적 일치는 아니더라도, 적어도 서로 다른 언어를 쓰던 두 흐름을 하나의 프레임으로 묶어냈다는 데 가치가 있다.

## 6. 결론

이 논문은 unsupervised domain adaptation에서 두 가지 중요한 공백을 메운다. 하나는 multiclass scoring function과 margin loss를 포함하는 새로운 일반화 이론을 제시했다는 점이고, 다른 하나는 그 이론이 실제 adversarial learning 알고리즘으로 자연스럽게 이어지도록 설계했다는 점이다. 핵심 개념인 **Margin Disparity Discrepancy (MDD)** 는 현재 classifier를 기준으로 source와 target 사이의 비대칭적 불일치 정도를 측정하며, 이를 줄이는 것이 target 일반화 향상으로 이어진다는 이론적 근거를 제공한다.

알고리즘적으로는 feature extractor, main classifier, auxiliary classifier의 minimax 학습 구조로 구현되며, gradient reversal과 결합된 형태로 실제 딥러닝 학습에 적용된다. 실험적으로도 Office-31, Office-Home, VisDA-2017에서 매우 강한 성능을 보였고, margin factor $\gamma$와 equilibrium 분석을 통해 이론과 실험의 정합성도 어느 정도 입증했다.

실제 적용 측면에서 이 연구는 단순히 하나의 강한 adaptation 방법을 제안한 것에 그치지 않는다. 더 중요한 점은, 앞으로 domain adaptation 알고리즘을 설계할 때 “어떤 discrepancy가 좋은가”를 묻는 기준을 바꾸었다는 것이다. 즉 discrepancy는 샘플에서 추정 가능해야 할 뿐 아니라, scoring function과 margin 구조를 반영하고, adversarial optimization으로 학습 가능해야 한다. 이런 관점은 이후 classifier discrepancy 기반 방법, conditional/adversarial alignment 방법, 더 넓게는 representation learning과 generalization theory를 잇는 연구에도 영향을 줄 수 있다.

종합하면, 이 논문은 domain adaptation 분야에서 이론과 알고리즘이 따로 발전하던 흐름을 하나의 정교한 설계 원리로 접속한 중요한 작업이다. 발표 당시 기준으로도 성능이 강했고, 지금 다시 읽어도 “왜 이 discrepancy를 이렇게 정의해야 하는가”에 대한 설명이 분명한 논문이다. ICML 2019에 발표되었으며 PMLR 97권에 수록되었다.
