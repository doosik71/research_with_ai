# Unsupervised Domain Adaptation by Backpropagation

* **저자**: Yaroslav Ganin, Victor L. Lempitsky
* **발표연도**: 2014
* **arXiv**: [https://arxiv.org/abs/1409.7495](https://arxiv.org/abs/1409.7495)

## 1. 논문 개요

이 논문은 **source domain에는 라벨이 충분히 있지만 target domain에는 라벨이 전혀 없는 상황**에서, deep neural network를 이용해 **unsupervised domain adaptation**을 수행하는 방법을 제안한다. 문제의 핵심은 source와 target의 데이터 분포가 다르기 때문에, source에서 잘 학습된 분류기가 target에서는 성능이 크게 떨어질 수 있다는 점이다. 예를 들어 synthetic image로는 라벨을 대량 확보할 수 있지만, 실제 이미지와는 배경, 질감, 조명, 노이즈 등이 달라 그대로 적용하면 일반화가 잘 되지 않는다.

논문이 해결하려는 연구 문제는 다음과 같다. **어떻게 하면 source 데이터에 대해서는 class-discriminative하면서도, 동시에 source와 target 사이의 domain shift에는 둔감한 feature를 학습할 수 있는가?** 기존의 많은 domain adaptation 방법은 고정된 feature 위에서 분포를 맞추거나, feature space를 사후적으로 변환하는 방식이었다. 반면 이 논문은 **representation learning 자체에 domain adaptation을 내장**하여, end-to-end 학습 과정에서 feature가 점차 domain-invariant하게 바뀌도록 만든다.

이 문제는 매우 중요하다. 실제 응용에서는 라벨 없는 target 데이터만 풍부한 경우가 흔하고, 특히 computer vision에서는 synthetic-to-real, simulator-to-real, web image-to-real image 같은 설정이 자주 등장한다. 따라서 source supervision만으로 target에서도 잘 작동하는 표현을 학습하는 것은 학문적으로도 중요하고 실용적으로도 가치가 크다. 이 논문은 이러한 목표를 위해 매우 단순하면서 구현이 쉬운 **Gradient Reversal Layer, GRL**을 제안했고, 이후 adversarial domain adaptation 계열 연구의 출발점이 된 대표적인 논문으로 평가된다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 명확하다. 좋은 target 성능을 얻으려면 feature가 두 조건을 동시에 만족해야 한다. 첫째, **source label prediction에는 유용해야 한다.** 둘째, **source와 target을 구분하기 어려울 정도로 domain-invariant해야 한다.** 이 두 조건을 동시에 만족시키기 위해 저자들은 하나의 feature extractor 위에 두 개의 head를 붙인다. 하나는 원래의 task를 위한 **label predictor**이고, 다른 하나는 source/target을 판별하는 **domain classifier**이다.

여기서 핵심은 feature extractor를 어떻게 학습시키느냐이다. label predictor는 당연히 source label loss를 줄이는 방향으로 학습되어야 한다. 반면 domain classifier는 source와 target을 잘 구분하도록 학습된다. 그런데 feature extractor 입장에서는 정반대로, domain classifier가 **헷갈리도록** feature를 만들어야 한다. 즉, domain classifier의 loss를 **크게 만드는 방향**으로 feature extractor가 업데이트되어야 한다. 이렇게 하면 domain classifier는 domain을 구분하려 하고, feature extractor는 그 구분을 무력화하려 하므로, 결과적으로 feature가 domain-invariant한 방향으로 이동한다.

이 논문의 차별점은 이 adversarial한 구조를 별도의 복잡한 min-max optimization 루프 없이, **표준 backpropagation만으로 구현**했다는 점이다. 그 비결이 바로 GRL이다. 이 층은 forward 때는 identity처럼 아무 일도 하지 않지만, backward 때는 gradient에 $-\lambda$를 곱해 방향을 뒤집는다. 따라서 domain classifier의 gradient가 feature extractor로 전달될 때, 원래는 domain classification loss를 줄이는 방향이 아니라 오히려 **늘리는 방향**으로 작동하게 된다. 이 한 가지 장치만으로 deep domain adaptation을 매우 간단하게 구현할 수 있다는 것이 논문의 가장 큰 공헌이다.

## 3. 상세 방법 설명

전체 시스템은 세 부분으로 나뉜다. 입력 $x$가 먼저 **feature extractor** $G_f$를 통과해 feature vector $f$가 된다. 그 다음 이 feature는 두 갈래로 분기된다. 하나는 **label predictor** $G_y$로 들어가 class label $y$를 예측하고, 다른 하나는 **domain classifier** $G_d$로 들어가 domain label $d \in {0,1}$, 즉 source인지 target인지 예측한다. 논문은 이 구조를 거의 모든 feed-forward network 위에 덧붙일 수 있다고 주장한다.

형식적으로, feature는 다음처럼 정의된다.

$$
f = G_f(x; \theta_f)
$$

여기서 $\theta_f$는 feature extractor의 파라미터이다. label predictor는 $G_y(f; \theta_y)$이고, domain classifier는 $G_d(f; \theta_d)$이다. source 샘플에는 class label이 있으므로 label predictor의 supervised loss를 계산할 수 있다. 반면 domain label은 source와 target 모두에서 알 수 있으므로 domain classifier loss는 모든 샘플에 대해 계산된다.

논문이 제안한 핵심 목적함수는 다음과 같다.

$$
E(\theta_f,\theta_y,\theta_d) = \sum_{i: d_i=0} L_y^i(\theta_f,\theta_y) \lambda \sum_{i=1}^{N} L_d^i(\theta_f,\theta_d)
$$

여기서 $L_y$는 label prediction loss이고, $L_d$는 domain classification loss이다. 중요한 점은 domain loss 앞에 **마이너스 부호**가 있다는 것이다. 이는 $\theta_d$는 domain 분류를 잘하도록 $L_d$를 줄여야 하지만, $\theta_f$는 반대로 domain 분류가 잘 안 되도록 $L_d$를 키워야 함을 뜻한다. 그래서 최적화 목표는 일반적인 최소화 문제가 아니라 **saddle point** 탐색 문제가 된다.

논문은 이를 다음과 같이 표현한다.

$$
\begin{aligned}
(\hat{\theta}_f,\hat{\theta}_y) &= \arg\min_{\theta_f,\theta_y} E(\theta_f,\theta_y,\hat{\theta}_d) \\
\hat{\theta}_d &= \arg\max_{\theta_d} E(\hat{\theta}_f,\hat{\theta}_y,\theta_d)
\end{aligned}
$$

즉, label predictor와 feature extractor는 label loss를 줄이되, feature extractor는 동시에 domain classifier를 방해해야 하고, domain classifier는 domain discrimination을 최대한 잘해야 한다.

이 목적을 직접 구현하면 복잡해 보이지만, 논문은 샘플 단위 stochastic update를 다음처럼 정리한다.

$$
\begin{aligned}
\theta_f \leftarrow & \theta_f - \mu \left( \frac{\partial L_y^i}{\partial \theta_f} - \lambda \frac{\partial L_d^i}{\partial \theta_f} \right) \\
\theta_y \leftarrow & \theta_y - \mu \frac{\partial L_y^i}{\partial \theta_y} \\
\theta_d \leftarrow & \theta_d - \mu \frac{\partial L_d^i}{\partial \theta_d}
\end{aligned}
$$

여기서 핵심은 첫 번째 식이다. feature extractor는 label loss gradient는 일반적인 방향으로 받지만, domain loss gradient는 $-\lambda$가 곱해진 방향으로 받는다. 즉, domain classifier가 잘하도록 feature를 바꾸는 것이 아니라, domain classifier가 못 하도록 feature를 바꾼다.

이제 GRL이 등장한다. 저자들은 GRL을 다음과 같은 “pseudo-function”으로 정의한다.

$$
\begin{aligned}
R_{\lambda}(x) &= x \\
\frac{dR_{\lambda}}{dx} &= -\lambda I
\end{aligned}
$$

forward propagation에서는 입력을 그대로 통과시키므로 network 구조를 깨지 않는다. 그러나 backward propagation에서는 상위에서 내려온 gradient에 $-\lambda$를 곱해 하위로 보낸다. 이 때문에, domain classifier가 계산한 loss gradient가 feature extractor로 역전파될 때는 부호가 뒤집혀, feature extractor가 domain-invariant한 feature를 학습하게 된다. 구현 면에서 보면 GRL은 파라미터가 없는 매우 단순한 layer다.

이를 반영한 최적화용 pseudo-objective는 다음과 같이 쓸 수 있다.

$$
\begin{aligned}
\tilde{E}(\theta_f,\theta_y,\theta_d) = &\sum_{i:d_i=0} L_y(G_y(G_f(x_i;\theta_f);\theta_y), y_i) \\
&+ \sum_{i=1}^{N} L_d(G_d(R_{\lambda}(G_f(x_i;\theta_f));\theta_d), d_i)
\end{aligned}
$$

원문 추출 텍스트에는 일부 식에서 domain loss의 두 번째 인수가 $y_i$처럼 잘못 보이는 부분이 있지만, 문맥상 domain classifier는 **class label이 아니라 domain label $d_i$**를 사용해야 한다. 이 부분은 OCR/LaTeX 추출 과정의 오류로 보이며, 본문 설명 전체와 GRL의 목적에 비추어 그렇게 해석하는 것이 타당하다.

이 논문은 또 이 방법을 이론적으로 $\mathcal{H}\Delta\mathcal{H}$-distance와 연결한다. domain adaptation 이론에서 target error는 대체로 **source error + domain discrepancy + 상수항**으로 상계된다. 논문은 domain classifier가 사실상 source와 target feature 분포의 구분 가능성을 측정하는 역할을 하며, reversed gradient를 통해 feature representation을 바꾸는 것은 결국 이 discrepancy를 줄이는 효과를 낸다고 설명한다. 즉, domain classifier를 속이는 feature를 학습하면 target generalization bound 측면에서도 유리하다는 해석이다.

훈련 절차도 구체적으로 제시한다. 배치 크기는 128이며, 각 batch의 절반은 labeled source, 나머지 절반은 unlabeled target으로 구성된다. 입력 이미지는 mean subtraction으로 전처리한다. loss는 label 쪽에 logistic regression loss, domain 쪽에 binomial cross-entropy를 썼다고 서술한다. 또한 adaptation strength를 나타내는 $\lambda$를 고정하지 않고, 학습 초반에는 domain classifier가 지나치게 noisy한 신호를 주지 않도록 점진적으로 키운다.

$$
\lambda_p = \frac{2}{1+\exp(-\gamma p)} - 1
$$

여기서 $p$는 training progress를 0에서 1까지 정규화한 값이고, $\gamma=10$이다. 이 스케줄은 초반에는 거의 0에 가까운 $\lambda$를 사용해 label learning을 우선 안정화하고, 학습이 진행될수록 domain confusion을 강하게 걸도록 설계되었다. learning rate도 다음과 같이 annealing한다.

$$
\mu_p = \frac{\mu_0}{(1+\alpha p)^{\beta}}
$$

논문에서는 $\mu_0=0.01$, $\alpha=10$, $\beta=0.75$, momentum은 0.9를 사용했다. SVHN 실험에서는 dropout과 $\ell_2$-norm restriction도 사용했다고 밝힌다.

아키텍처는 데이터셋마다 다르다. MNIST 계열에서는 LeNet-5 스타일의 비교적 작은 CNN을 썼고, SVHN 계열에서는 더 큰 CNN을, GTSRB에서는 traffic sign recognition용 CNN을 사용했다. Office dataset에서는 ImageNet으로 pretrain된 AlexNet을 fine-tune하며, domain classifier를 fc7 bottleneck에 부착했다. domain classifier는 대체로 fully connected 구조 $x \rightarrow 1024 \rightarrow 1024 \rightarrow 2$를 사용했다. 이 branch 구조는 다소 임의적이며 더 튜닝하면 더 나을 수 있다고 저자들도 인정한다.

## 4. 실험 및 결과

실험은 주로 image classification 기반 domain adaptation 설정에서 수행되었다. 논문은 단순히 성능 수치만 제시하는 것이 아니라, source-only 하한선과 train-on-target 상한선을 함께 보여 주어 adaptation이 실제로 얼마나 gap을 메웠는지 해석할 수 있게 했다. 또한 비교 대상으로는 shallow DA 방법인 **Subspace Alignment, SA**와 여러 기존 Office benchmark 방법들을 사용했다.

첫 번째 실험은 **MNIST $\rightarrow$ MNIST-M**이다. MNIST-M은 원래의 MNIST digit을 BSDS500 자연 이미지 패치 위에 합성하여 만든 target domain이다. 사람 눈에는 여전히 숫자를 식별하기 쉽지만, 배경이 복잡해져 source-only CNN에는 상당한 domain shift가 생긴다. 결과를 보면 source-only 성능은 **0.5749**, SA는 **0.6078**, 제안 방법은 **0.8149**, train-on-target upper bound는 **0.9891**이다. 즉 제안 방법은 매우 큰 폭의 향상을 달성했고, 표에서 제시한 기준으로 gap의 **57.9%**를 메웠다. 이 실험은 논문의 핵심 주장을 가장 직관적으로 보여 준다. 즉, source에서 잘 작동하는 분류 feature를 유지하면서도 target의 배경 변화에 둔감한 표현이 학습되었다는 것이다.

두 번째는 **Synthetic Numbers $\rightarrow$ SVHN**이다. 이 설정은 synthetic-to-real adaptation의 전형적인 사례다. source는 저자들이 직접 생성한 50만 장의 합성 숫자 이미지이며, target은 실제 거리 장면 숫자인 SVHN이다. 배경 clutter와 자연 이미지 특성 때문에 두 도메인 차이가 크다. 여기서 source-only는 **0.8665**, SA는 **0.8672**, 제안 방법은 **0.9048**, train-on-target은 **0.9244**였다. SA는 거의 개선을 보이지 못한 반면, 제안 방법은 upper bound와의 차이 중 **66.1%**를 메웠다. 즉, 이 방법은 단순한 feature subspace 정렬보다 훨씬 강한 적응 효과를 보였다.

세 번째는 **SVHN $\rightarrow$ MNIST** 방향이 표에 포함되어 있다. 결과는 source-only **0.5919**, SA **0.6157**, 제안 방법 **0.7107**, train-on-target **0.9951**이다. 논문은 이 방향에서는 adaptation이 어느 정도 성공했다고 설명한다. 흥미로운 점은 반대 방향인 **MNIST $\rightarrow$ SVHN**은 실패 사례라고 명시한다는 점이다. 즉, 모든 domain shift를 해결하는 만능 방법은 아니며, 특히 MNIST에서 학습한 feature가 SVHN 같은 더 복잡한 자연 장면 숫자로 일반화되기 어렵다는 점을 인정한다. 이 솔직한 실패 보고는 논문의 신뢰성을 높인다.

네 번째는 **Synthetic Signs $\rightarrow$ GTSRB**이다. 클래스 수가 43개로 늘어나 feature 분포가 더 복잡한 상황이다. source-only는 **0.7400**, SA는 **0.7635**, 제안 방법은 **0.8866**, train-on-target은 **0.9987**이다. 역시 상당한 개선이 있었고, gap의 **56.7%**를 메웠다. synthetic-to-real traffic sign adaptation에서도 효과가 있음을 보여 준다.

추가로 논문은 **semi-supervised domain adaptation** 가능성도 간단히 실험했다. GTSRB에서 소량의 labeled target data를 train set으로 두고 validation error 변화를 그린 Figure 4를 제시한다. 이 실험은 정량 표보다는 추세를 보여 주는 성격이며, 제안 방식이 labeled target이 조금 있을 때도 target-only 학습보다 더 낮은 error를 얻을 수 있음을 시사한다. 다만 저자들은 이 부분에 대해 “thorough verification is left for future work”라고 하여, 본 논문의 핵심 기여는 어디까지나 unsupervised DA라고 선을 긋는다.

마지막으로 **Office dataset** 실험은 이 논문의 대표적인 benchmark 결과다. Office는 Amazon, DSLR, Webcam 세 domain으로 이루어진 소규모 but 표준적인 domain adaptation 벤치마크다. 이 데이터셋은 크기가 작기 때문에 논문은 ImageNet pretrained AlexNet을 fine-tuning하는 전략을 택했다. 비교 과제는 Amazon $\rightarrow$ Webcam, DSLR $\rightarrow$ Webcam, Webcam $\rightarrow$ DSLR 세 가지다.

결과를 보면, 제안 방법은 각각 **0.673 ± 0.017**, **0.940 ± 0.008**, **0.937 ± 0.010**을 기록했다. 이는 표에 포함된 GFK, SA, DA-NBNN, DLID, DaNN, DDC 등 기존 방법보다 모두 높은 수치이며, 특히 가장 어려운 Amazon $\rightarrow$ Webcam에서 향상 폭이 크다. 논문은 이를 통해 당시 Office benchmark에서 **state-of-the-art**를 갱신했다고 주장한다.

정성적 분석으로는 t-SNE 시각화를 사용한다. 적응 전 feature는 source와 target이 뚜렷이 분리되어 있지만, 적응 후에는 두 분포가 훨씬 가깝게 겹친다. 논문은 이런 분포 overlap 증가가 target 정확도 향상과 잘 대응한다고 해석한다. 물론 t-SNE는 정량 지표는 아니지만, domain confusion이 실제 representation 공간에서 일어났다는 시각적 증거로는 설득력이 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **아이디어의 단순성과 구현의 우아함**이다. adversarial한 목표를 갖는 domain adaptation은 원래 min-max optimization처럼 복잡하게 보일 수 있는데, 저자들은 이를 GRL이라는 거의 한 줄짜리 개념으로 backpropagation 안에 녹여 넣었다. forward에서는 identity, backward에서는 gradient sign flip이라는 단순한 설계 덕분에, 기존 deep learning framework에 쉽게 붙일 수 있다. 논문이 Caffe extension으로 구현 예제를 공개하겠다고 한 것도 이 실용성을 강조하기 위한 것이다.

두 번째 강점은 **representation learning과 domain adaptation을 공동 최적화**한다는 점이다. 기존의 많은 방법은 고정된 feature 위에서 도메인 정렬을 하거나, autoencoder로 representation을 먼저 학습하고 classifier를 나중에 학습하는 2-stage 구조였다. 이 논문은 feature extractor, label predictor, domain classifier를 한 시스템 안에 두고 end-to-end로 학습한다. 이 덕분에 feature가 task-discriminative하면서 동시에 domain-invariant하도록 직접 압력을 받을 수 있다.

세 번째 강점은 실험의 설득력이다. synthetic-to-real, digit adaptation, traffic sign adaptation, Office benchmark까지 다양한 설정에서 좋은 결과를 보였고, 단순 baseline뿐 아니라 당시 강한 경쟁 방법들과 비교해서도 우수했다. 특히 Office dataset에서 SOTA를 달성한 것은 방법의 범용성을 뒷받침한다.

이론적 연결도 장점이다. 논문은 $\mathcal{H}\Delta\mathcal{H}$-distance를 이용해 왜 domain classifier를 헷갈리게 만드는 것이 target generalization에 도움이 되는지 설명한다. 이 이론은 엄밀한 완전 증명이라기보다 방법의 직관을 보강하는 수준이지만, 경험적 성공과 잘 맞물린다.

반면 한계도 분명하다. 첫째, 이 방법은 **완전한 domain alignment를 보장하지 않는다.** domain classifier를 속이는 것이 곧 class-conditional alignment까지 보장하는 것은 아니다. 즉, 전체 분포는 비슷해졌더라도 클래스별로 잘못 섞일 위험이 있다. 논문 본문에는 class-conditional mismatch에 대한 명시적 처리가 없다. 후속 연구들이 conditional adversarial adaptation이나 class-aware alignment로 발전한 이유가 여기에 있다.

둘째, **어려운 adaptation 방향에서는 실패**한다. 논문은 MNIST $\rightarrow$ SVHN에서는 자신들의 방법도 실패했다고 명시한다. 이는 domain shift가 너무 크거나 source feature가 지나치게 단순한 경우, domain confusion만으로는 충분하지 않을 수 있음을 보여 준다. 다시 말해, “domain-invariant”를 강제하는 것만으로 항상 좋은 target classification boundary가 생기지는 않는다.

셋째, 하이퍼파라미터와 부착 위치 문제다. 논문은 $\lambda$ 스케줄, domain classifier 구조, 그리고 어느 layer에 adaptation branch를 붙일지 등이 중요하다고 설명한다. 이들은 target label 없이 정해야 하므로 실제로는 쉽지 않다. 저자들은 source test error가 낮고 domain classifier error가 높을수록 adaptation이 잘 되는 경향이 있다고 말하지만, 이것이 일반적인 model selection 기준으로 충분한지는 논문만으로는 확실치 않다.

넷째, Office 실험은 **transductive setting**에 가깝다. 즉, unlabeled target 전체를 학습 중 사용할 수 있다고 가정한다. 이는 unsupervised DA 문헌에서는 흔한 설정이지만, 실제 deployment에서는 target 분포가 고정된 전량 데이터로 주어지지 않을 수도 있다. 따라서 실용적 의미를 해석할 때는 이 가정을 염두에 둘 필요가 있다.

또 하나 짚을 점은, 논문이 제시하는 domain classifier 성능과 source error를 hyperparameter tuning의 힌트로 쓰는 방식은 직관적이지만, 이 값들이 실제 target accuracy와 항상 일치한다고 보장되지는 않는다. 논문은 “good correspondence”를 관찰했다고만 말할 뿐, 강한 보장을 주지는 않는다. 따라서 이 부분은 경험적 heuristic에 가깝다.

종합하면, 이 논문은 방법론적으로 매우 강력하고 역사적으로도 중요한 출발점이지만, **global domain confusion만으로는 충분하지 않은 경우가 있으며**, 이후 연구들이 이를 보완해 나갔다는 맥락 속에서 읽는 것이 적절하다.

## 6. 결론

이 논문은 deep neural network에서 unsupervised domain adaptation을 수행하기 위한 매우 영향력 있는 방법을 제안했다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, source label prediction과 domain confusion을 동시에 만족하는 feature learning 문제를 명확히 정식화했다. 둘째, 이를 구현하기 위한 단순하고 실용적인 장치인 **Gradient Reversal Layer**를 제안했다. 셋째, 다양한 vision benchmark에서 강한 성능을 보이며 당시 기준으로 매우 경쟁력 있는 결과를 제시했다.

이 연구의 의미는 단지 성능 향상에만 있지 않다. 이후 수많은 adversarial adaptation, domain confusion, invariant representation 연구들이 이 논문의 틀 위에서 발전했다. 특히 “domain classifier를 두고, feature extractor가 이를 속이게 한다”는 아이디어는 후속 transfer learning, fairness, invariant learning 연구 전반으로 확장되었다. 실제 적용 측면에서도 simulator-to-real, synthetic-to-real, cross-camera, cross-sensor adaptation 같은 문제에 매우 자연스럽게 연결된다.

동시에 이 논문은 domain adaptation이 단순히 feature를 맞추는 문제만은 아니라는 사실도 보여 준다. 일부 어려운 설정에서는 실패하며, class-conditional alignment나 더 정교한 adversarial objective가 필요할 수 있다. 그럼에도 불구하고, **표준 backpropagation 안에서 adversarial domain adaptation을 가능하게 만든 첫 번째 간결한 해법**이라는 점에서 이 논문은 여전히 매우 중요한 위치를 차지한다. 오늘날에도 DANN 계열 방법의 원형으로서 읽을 가치가 크다.
