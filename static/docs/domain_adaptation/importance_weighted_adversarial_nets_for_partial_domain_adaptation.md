# Importance Weighted Adversarial Nets for Partial Domain Adaptation

* **저자**: Jing Zhang, Zewei Ding, Wanqing Li, Philip Ogunbona
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1803.09210](https://arxiv.org/abs/1803.09210)

## 1. 논문 개요

이 논문은 **partial domain adaptation** 문제를 다룬다. 이는 source domain의 클래스 집합이 더 크고, target domain은 그 일부 클래스만 포함하는 상황을 뜻한다. 기존의 unsupervised domain adaptation 방법들은 대체로 source와 target의 label space가 동일하다고 가정한다. 이 가정 아래에서는 두 도메인의 feature distribution 차이를 줄이면 지식 전달이 잘 일어난다고 본다. 그러나 이 논문이 다루는 상황에서는 이 가정이 성립하지 않는다. target에는 source의 일부 클래스만 존재하므로, source 전체와 target 전체를 무작정 정렬시키면 오히려 target에 존재하지 않는 source의 outlier classes까지 맞추려 하게 되어 **negative transfer**가 발생할 수 있다.

논문의 목표는 이러한 문제를 해결하기 위해, source 샘플들 중에서 **target과 공유될 가능성이 높은 샘플에는 큰 가중치**, target에 없는 outlier class에서 왔을 가능성이 높은 샘플에는 **작은 가중치**를 부여하는 adversarial adaptation 방법을 제안하는 것이다. 핵심은 target이 unlabeled라는 점이다. 즉, 어떤 source 클래스가 target에 실제로 존재하는지 미리 알 수 없고, target 클래스의 이름이나 개수도 모른다. 따라서 논문은 target label 정보를 전혀 사용하지 않고도, adversarial domain classifier의 출력을 이용해 source 샘플의 중요도를 추정하는 방식을 설계한다.

이 문제가 중요한 이유는 실용적인 응용에서 더 흔하기 때문이다. 현실에서는 대규모 source 데이터셋을 가지고 더 작은 target 도메인에 적응시키는 일이 많다. 예를 들어, 다양한 카테고리를 포함한 대형 데이터셋에서 특정 환경이나 특정 장치에서 취득된 더 제한된 클래스 집합으로 적응하는 경우가 그렇다. 이때 기존 full domain adaptation 기법을 그대로 적용하면 target에 없는 클래스까지 강제로 맞추면서 성능이 떨어질 수 있다. 따라서 partial setting을 명시적으로 고려하는 방법은 실제 transfer learning 시스템에서 매우 의미가 크다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 단순하다. **“target과 잘 겹치는 source 샘플만 골라서 domain alignment를 하자”**는 것이다. 하지만 target은 unlabeled이므로, 어떤 source 샘플이 target과 공유 클래스에 속하는지 직접 알 수 없다. 논문은 이 문제를 해결하기 위해 **두 개의 domain classifier**를 사용한다.

첫 번째 domain classifier는 현재 feature space에서 source 샘플이 target과 얼마나 분리되는지를 측정하는 역할을 한다. 만약 어떤 source 샘플이 domain classifier에 의해 “거의 확실히 source”로 판별된다면, 그 샘플 주변에는 target 샘플이 거의 없다는 뜻이므로 target에 없는 outlier class일 가능성이 높다. 반대로 source인지 target인지 구분하기 어려운 source 샘플은 target과 공유되는 클래스에서 왔을 가능성이 더 높다. 따라서 첫 번째 classifier의 출력은 source 샘플의 **importance weight**를 만드는 신호로 사용된다.

그다음 두 번째 domain classifier는 이렇게 계산된 가중치를 반영한 source 샘플들과 target 샘플을 가지고 실제 adversarial alignment를 수행한다. 즉, 첫 번째 classifier는 “어떤 source 샘플을 얼마나 믿을지”를 추정하고, 두 번째 classifier는 “중요한 source 샘플과 target을 정렬”하는 역할을 맡는다. 이 분리가 중요하다. 만약 하나의 동일한 domain classifier에 weighting까지 동시에 걸어버리면 이론적으로 깔끔한 divergence minimization 해석이 깨질 수 있기 때문이다. 논문은 이 점을 분명히 인식하고, weighting을 산출하는 classifier와 alignment를 담당하는 classifier를 분리해 설계한다.

기존 접근과의 차별점도 명확하다. RevGrad나 ADDA류 방법은 label space가 동일하다고 보고 전체 source와 target을 정렬한다. SAN은 partial transfer를 직접 다루지만 클래스별 domain classifier를 여러 개 둔다. 반면 이 논문은 **클래스별 classifier를 두지 않고도**, 오직 **두 개의 domain classifier**만으로 partial adaptation을 구현한다. 이 때문에 source 클래스 수가 매우 많아질 때 SAN보다 계산량과 파라미터 측면에서 더 확장 가능하다는 점을 강조한다. 또한 class-level weight를 따로 요구하지 않기 때문에, target의 class imbalance에도 더 유리할 수 있다고 주장한다.

## 3. 상세 방법 설명

논문은 source data를 $X_s \in \mathbb{R}^{D \times n_s}$, target data를 $X_t \in \mathbb{R}^{D \times n_t}$로 두고, source label space와 target label space의 관계를 $\mathcal{Y}_t \subseteq \mathcal{Y}_s$로 둔다. 즉 target 클래스는 source 클래스의 부분집합이다. 학습 시에는 labeled source data와 unlabeled target data만 사용한다.

전체 구조는 크게 네 부분으로 이루어진다. source feature extractor $F_s$, target feature extractor $F_t$, source classifier $C$, 그리고 두 개의 domain classifier $D$와 $D_0$이다. 논문은 shared feature extractor 대신 **unshared feature extractors**를 사용한다. 즉 source와 target이 같은 feature extractor를 공유하지 않고, 각각 $F_s$와 $F_t$를 갖는다. 이는 각 도메인의 domain-specific feature를 더 잘 포착하기 위함이다. 다만 $F_t$는 초기화 시 $F_s$의 파라미터를 복사해 시작하여 극단적인 degenerate solution을 피한다.

먼저 source discriminative model은 일반적인 supervised learning으로 학습된다. source feature extractor $F_s$와 classifier $C$는 source 분류 손실을 최소화한다.

$$
\min_{F_s, C} \mathcal{L}_s = \mathbb{E}_{\mathbf{x}, y \sim p_s(\mathbf{x}, y)} L(C(F_s(\mathbf{x})), y)
$$

여기서 $L$은 cross-entropy loss이다. 이 단계가 끝나면 $F_s$와 $C$는 source 분류에 잘 맞는 상태가 되고, 이후에는 고정된다.

그다음 일반 adversarial domain adaptation의 기본 형태를 보면, domain classifier $D$는 source와 target을 구분하려 하고, target feature extractor $F_t$는 그 구분을 어렵게 만들어 두 분포를 feature space에서 가깝게 한다. 이 기본 objective는 다음과 같다.

$$
\min_{F_t} \max_D \mathcal{L}(D, F_s, F_t) = \mathbb{E}_{\mathbf{x} \sim p_s(\mathbf{x})} [\log D(F_s(\mathbf{x}))] + \mathbb{E}_{\mathbf{x} \sim p_t(\mathbf{x})} [\log (1 - D(F_t(\mathbf{x})))]
$$

이때 feature space의 표본을 $\mathbf{z} = F(\mathbf{x})$라고 두면, 고정된 feature extractor에 대해 최적 domain classifier는 다음과 같다.

$$
D^*(\mathbf{z}) = \frac{p_s(\mathbf{z})}{p_s(\mathbf{z}) + p_t(\mathbf{z})}
$$

이 식은 GAN에서의 최적 discriminator와 동일한 형태다. 즉 어떤 feature가 source density가 크고 target density가 작으면 $D^*(\mathbf{z})$가 1에 가까워진다. 논문은 바로 이 성질을 이용한다. 만약 어떤 source 샘플의 feature $\mathbf{z}$에서 $D^*(\mathbf{z}) \approx 1$이라면, 그 영역은 target과 거의 겹치지 않으므로 target에 없는 outlier class에서 왔을 가능성이 높다.

그래서 source 샘플의 raw importance를 다음과 같이 정의한다.

$$
\tilde{w}(\mathbf{z}) = 1 - D^*(\mathbf{z}) \frac{1}{\frac{p_s(\mathbf{z})}{p_t(\mathbf{z})} + 1}
$$

이 식의 의미는 직관적이다. $D^*(\mathbf{z})$가 크면, 즉 source 쪽에 치우친 영역이면 weight는 작아진다. 반대로 source와 target이 잘 겹치는 영역이면 $D^*(\mathbf{z})$가 상대적으로 작아져 weight가 커진다. 논문은 이 weight가 사실상 density ratio와 연결되어 있어 합리적이라고 설명한다.

이 raw weight는 평균이 1이 되도록 정규화된다.

$$
w(\mathbf{z}) = \frac{\tilde{w}(\mathbf{z})} {\mathbb{E}_{\mathbf{z} \sim p_s(\mathbf{z})} \tilde{w}(\mathbf{z})}
$$

이렇게 하면 $w(\mathbf{z}) p_s(\mathbf{z})$가 여전히 확률밀도처럼 다뤄질 수 있다. 그런데 여기서 중요한 문제가 있다. 만약 이 weight를 같은 domain classifier에 바로 적용하면, weight가 discriminator 출력의 함수이므로 이론적 최적해 해석이 복잡해진다. 논문은 이를 피하기 위해, weight를 계산하는 첫 번째 classifier $D$와 실제 weighted adversarial matching을 수행하는 두 번째 classifier $D_0$를 분리한다.

따라서 weighted adversarial loss는 다음과 같이 된다.

$$
\begin{aligned}
\min_{F_t} \max_{D_0} \mathcal{L}_w(D_0, F_s, F_t) = & \; \mathbb{E}_{\mathbf{x} \sim p_s(\mathbf{x})} [w(\mathbf{z}) \log D_0(F_s(\mathbf{x}))] \\
&+ \mathbb{E}_{\mathbf{x} \sim p_t(\mathbf{x})} [\log (1 - D_0(F_t(\mathbf{x})))]
\end{aligned}
$$

여기서 $w(\mathbf{z})$는 첫 번째 classifier $D$로부터 이미 계산된 값이므로, 두 번째 classifier $D_0$ 입장에서는 상수처럼 취급된다. 이때 최적 $D_0^*$는 다음과 같다.

$$
D_0^*(\mathbf{z}) = \frac{w(\mathbf{z}) p_s(\mathbf{z})} {w(\mathbf{z}) p_s(\mathbf{z}) + p_t(\mathbf{z})}
$$

이를 objective에 대입하면 weighted source density와 target density 사이의 Jensen-Shannon divergence를 최소화하는 형태가 된다.

$$
\mathcal{L}_w(F_t) = -\log(4) + 2 \cdot JS(w(\mathbf{z}) p_s(\mathbf{z}) | p_t(\mathbf{z}))
$$

즉 이 방법은 원래 source density 전체를 target에 맞추는 것이 아니라, **가중된 source density** $w(\mathbf{z}) p_s(\mathbf{z})$를 target density에 맞춘다. 이것이 partial domain adaptation에 정확히 필요한 동작이다. outlier class 영역은 작은 weight를 받아 거의 무시되고, shared class 영역만 주로 정렬된다.

논문은 여기에 target data structure preservation을 위해 entropy minimization도 추가한다. target은 unlabeled이므로 feature를 정렬하는 과정에서 target 샘플의 클래스 구조가 무너질 수 있다. 이를 완화하려고 source classifier $C$의 target 예측 분포 entropy를 줄이는 항을 추가한다.

$$
\min_{F_t}
\mathbb{E}_{\mathbf{x} \sim p_t(\mathbf{x})}
H(C(F_t(\mathbf{x})))
$$

이 항은 target 예측을 더 confident하게 만들어 class boundary가 low-density region을 통과하게 유도한다. 다만 논문은 이 entropy minimization을 **$F_t$에만 적용**하고 classifier $C$에는 적용하지 않는다. 그 이유는 early training stage에서 domain shift가 큰 상태에서 classifier까지 함께 끌어당기면, target 샘플이 잘못된 클래스에 강하게 고정될 위험이 있다고 보기 때문이다. 이 판단은 논문의 중요한 설계 포인트다.

최종 objective는 세 단계로 정리된다. 첫째, $F_s$와 $C$를 source supervised learning으로 학습한다. 둘째, 첫 번째 domain classifier $D$를 이용해 source sample importance weight를 계산한다. 셋째, 두 번째 domain classifier $D_0$와 target feature extractor $F_t$가 minimax game을 하면서 weighted alignment를 수행하고, 동시에 target entropy minimization도 적용한다. 논문은 이 minimax optimization을 GAN처럼 번갈아 최적화하는 대신 **GRL(Gradient Reversal Layer)**을 사용해 end-to-end로 푼다.

전체 objective는 다음과 같이 쓸 수 있다.

$$
\begin{aligned}
\min_{F_s, C} \mathcal{L}_s(F_s, C) = & - \mathbb{E}_{\mathbf{x}, y \sim p_s(\mathbf{x}, y)} \sum_{k=1}^{K} \mathbb{1}_{[k=y]} \log C(F_s(\mathbf{x})) \\
\min_D \mathcal{L}_D(D, F_s, F_t) = & - \big( \mathbb{E}_{\mathbf{x} \sim p_s(\mathbf{x})}[\log D(F_s(\mathbf{x}))] \\
&+ \mathbb{E}_{\mathbf{x} \sim p_t(\mathbf{x})}[\log(1 - D(F_t(\mathbf{x})))] \big) \\
\min_{F_t} \max_{D_0} \mathcal{L}_w(C, D_0, F_s, F_t) = & \; \gamma \mathbb{E}_{\mathbf{x} \sim p_t(\mathbf{x})} H(C(F_t(\mathbf{x}))) \\
&+ \lambda \big( \mathbb{E}_{\mathbf{x} \sim p_s(\mathbf{x})}[w(\mathbf{z}) \log D_0(F_s(\mathbf{x}))] \\
&+ \mathbb{E}_{\mathbf{x} \sim p_t(\mathbf{x})}[\log(1 - D_0(F_t(\mathbf{x})))] \big)
\end{aligned}
$$

여기서 $\lambda$는 adversarial alignment의 강도를 조절하는 계수이고, $\gamma$는 target entropy term의 가중치이다. 본문에는 $\gamma$의 구체적인 수치가 전부 명시되어 있지는 않지만, 실험 표에서는 $\gamma = 0$인 변형과 기본 proposed를 비교한다. 또한 $\lambda$는 훈련 초기에 noisy signal을 줄이기 위해 점진적으로 증가시키는 schedule을 사용한다.

$$
\lambda = \frac{2u}{1 + \exp(-\alpha p)} - u
$$

여기서 $p$는 0에서 1까지 증가하는 training progress, $\alpha = 1$, 상한 $u = 0.1$이다. 즉 학습 초반에는 domain alignment를 약하게 하고, 학습이 진행될수록 조금 더 강하게 건다.

정리하면, 이 논문의 파이프라인은 다음과 같다. source로 $F_s$와 $C$를 학습한다. 현재의 $F_t$와 고정된 $F_s$를 이용해 첫 번째 domain classifier $D$가 source 중요도 weight를 산출한다. 이 weight를 source 샘플에 곱한 뒤 두 번째 domain classifier $D_0$와 $F_t$가 adversarial alignment를 수행한다. 동시에 target entropy minimization으로 target 구조를 보존한다. 최종적으로는 source classifier $C$를 target feature $F_t(x)$ 위에 직접 적용해 target 분류를 수행한다.

## 4. 실험 및 결과

실험은 세 종류의 object recognition benchmark에서 수행된다. 첫째는 Office+Caltech-10으로 Amazon, Webcam, DSLR, Caltech 네 도메인 간 적응을 다룬다. source는 10개 클래스, target은 그중 앞의 5개 클래스만 사용하여 partial setting을 만든다. 그래서 표기상 source는 A10, W10, D10, C10이고, target은 A5, W5, D5, C5로 쓴다.

둘째는 Office-31이다. 이 경우 source는 31개 클래스, target은 Office31과 Caltech-256이 공유하는 10개 클래스만 포함한다. 따라서 A31, W31, D31에서 A10, W10, D10으로의 적응 문제가 된다. 이 설정은 partiality가 더 강하고, outlier source classes가 훨씬 많기 때문에 더 도전적이다.

셋째는 더 큰 규모의 Caltech-256 $\rightarrow$ Office10 설정이다. source는 256개 클래스, target은 Office-31과 공유되는 10개 클래스만 가진다. 이는 대규모 source에서 작은 target subset으로 옮기는 상황을 시험한다.

비교 대상은 source-only baseline인 AlexNet+bottleneck, adversarial adaptation인 RevGrad, moment matching 기반 RTN, unshared extractor adversarial adaptation인 ADDA-grl, 그리고 partial transfer 전용 방법 SAN이다. 여기서 ADDA-grl은 논문 저자들이 GRL을 사용해 공정 비교를 위해 구성한 변형이며, 사실상 제안 방법에서 weighting만 제거한 버전이라고 볼 수 있다. 따라서 proposed와 ADDA-grl의 비교는 weighting scheme 자체의 효과를 가장 직접적으로 보여준다.

네트워크는 ImageNet pretrained AlexNet을 기반으로 한다. feature extractor는 fc8을 제거한 AlexNet에 256차원 bottleneck을 추가한다. 두 domain classifier는 모두 1024 $\rightarrow$ 1024 $\rightarrow$ 1 구조의 3-layer fully connected network이다. source extractor $F_s$는 source 분류로 먼저 fine-tuning된다.

결과를 보면, partial adaptation에서 제안 방법은 대부분의 비교 대상보다 뚜렷하게 우수하다. Office+Caltech-10 결과를 보면 평균 정확도는 AlexNet+bottleneck이 87.14, RevGrad가 79.83, RTN이 78.44, ADDA-grl이 92.31, proposed($\gamma=0$)가 93.72, 최종 proposed가 93.85이다. 즉 weighting이 없는 ADDA-grl보다도 개선되고, RevGrad나 RTN처럼 partial setting을 고려하지 않은 방법들보다 큰 폭으로 낫다.

Office-31의 더 어려운 partial setting에서는 차이가 더 분명하다. 평균 정확도는 AlexNet+bottleneck 76.32, RevGrad 66.64, RTN 78.80, ADDA-grl 79.41, SAN-selective 83.73, SAN-entropy 85.64, SAN 87.27, proposed($\gamma=0$) 86.73, proposed 87.57이다. 최종 proposed는 SAN보다도 평균이 약간 높다. 특히 A31 $\rightarrow$ W10, D31 $\rightarrow$ A10 같은 어려운 전이에서 강한 결과를 보인다. 예를 들어 D31 $\rightarrow$ A10에서 proposed는 89.46으로 SAN의 80.58보다 높다. 반면 A31 $\rightarrow$ D10에서는 SAN 81.28보다 proposed 78.98이 낮다. 즉 모든 개별 전이에서 일관되게 최고는 아니지만, 평균적으로는 매우 경쟁력 있다.

Caltech256 $\rightarrow$ Office10에서도 average accuracy가 Alex 49.86, RevGrad 61.80, RTN 71.56, SAN 85.83, proposed 84.14로 보고된다. 여기서는 SAN이 약간 더 높고 proposed는 근소하게 낮다. 따라서 논문의 주장인 “이전 domain adaptation 방법들보다 크게 우수하며, state-of-the-art partial transfer method와 비교 가능하다”는 표현이 적절하다. 모든 실험에서 압도적으로 1등이라고 말하는 것은 과장이다.

논문은 정량 결과뿐 아니라 t-SNE 시각화도 제시한다. A31 $\rightarrow$ W10의 bottleneck activation을 보면, RevGrad는 target 샘플이 source의 31개 전체 클래스 쪽으로 퍼져 negative transfer가 심하게 나타난다. 이는 partial setting에서 단순 adversarial alignment가 얼마나 위험한지 보여준다. RTN은 entropy minimization 덕분에 target 샘플이 모든 클래스에 완전히 흩어지지는 않지만, 여전히 outlier class의 영향이 남는다. ADDA-grl과 proposed를 비교하면, proposed에서는 target 데이터가 선택된 shared source classes와 더 잘 정렬된다. 이 비교는 weighting scheme의 효과를 시각적으로 뒷받침한다.

또한 논문은 source 샘플 weight를 시각화하여, 높은 weight를 받은 샘플이 주로 shared classes 0~9에 속하고, outlier classes 10~30은 낮은 weight를 받는 경향을 보인다고 설명한다. 이는 첫 번째 domain classifier가 실제로 outlier source samples를 어느 정도 식별하고 있음을 보여주는 정성적 근거다.

추가 분석으로 target class 개수를 31, 25, 20, 15, 10, 5로 변화시킨 실험도 수행한다. source는 항상 31개 클래스이고 target 클래스 수가 줄어드는 설정이다. 결과적으로 target 클래스 수가 줄어들수록 proposed의 상대적 이득이 더 커진다. 이는 partiality가 강해질수록 weighting의 필요성이 커진다는 논문의 핵심 메시지와 잘 맞는다. 반면 ADDA-grl은 target 클래스 수가 적어질수록 정확도가 더 떨어진다. 즉 weighting이 없는 일반 adversarial alignment는 partial setting이 심해질수록 더 취약해진다.

한편 non-partial setting에서도 실험을 수행해 제안 방법이 full domain adaptation 상황에서 심각한 성능 저하를 일으키지 않는지 확인한다. Office31에서는 Alex 69.15, RevGrad 73.75, RTN 72.87, ADDAgrl 73.90, proposed 73.35이고, OfficeCal10에서는 Alex 86.10, RevGrad 90.90, RTN 93.40, ADDAgrl 92.21, proposed 91.71이다. 즉 partial 전용 설계가 full setting에서 완전히 무너지는 것은 아니지만, 최상 성능이라고 보기도 어렵다. 논문이 말하는 “noticeable degradation이 없다”는 정도로 해석하는 것이 적절하다.

마지막으로 unshared feature extractor의 효과도 비교한다. 가장 어려운 A31 $\rightarrow$ W10에서 shared extractor는 71.5%, unshared extractor는 76.3%를 보였다. 이는 source와 target이 서로 다른 domain-specific 특성을 가질 때, feature extractor를 분리하는 선택이 실제로 유리할 수 있음을 뒷받침한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 partial domain adaptation 문제를 매우 직접적이고 설득력 있게 모델링했다는 점이다. 기존 adversarial adaptation이 실패하는 이유를 “label space mismatch 때문에 source 전체와 target 전체를 정렬하면 안 된다”는 관점에서 분명히 설명하고, 이를 해결하기 위해 **source importance weighting**이라는 간단하면서도 효과적인 장치를 넣었다. 특히 importance weight를 domain classifier 출력으로부터 유도한 점은 직관과 이론이 잘 연결되어 있다. 단순히 heuristic하게 샘플을 제거하는 것이 아니라, discriminator가 source라고 확신하는 샘플은 target과 겹치지 않을 가능성이 높다는 점을 활용했다.

두 번째 강점은 이론적 해석이다. 두 번째 domain classifier를 따로 둠으로써, 최종적으로는 $JS(w(\mathbf{z})p_s(\mathbf{z}) | p_t(\mathbf{z}))$를 줄이는 구조로 설명된다. 즉 “가중된 source 분포와 target 분포를 맞춘다”는 개념이 수학적으로 깔끔하게 정리된다. partial setting에서 무엇을 정렬해야 하는지를 명시적으로 정의했다는 점에서 의미가 있다.

세 번째 강점은 효율성과 확장성이다. SAN은 클래스마다 domain classifier를 두는 구조라 source 클래스 수가 많아지면 부담이 커진다. 반면 이 논문은 클래스 수와 무관하게 두 개의 domain classifier만 사용한다. source가 대규모일수록 이 설계는 실용적인 장점이 있다.

네 번째 강점은 실험 설계의 설득력이다. Office+Caltech-10, Office-31, Caltech256 $\rightarrow$ Office10처럼 partiality 강도가 다른 여러 설정을 사용했고, target class 수 변화 실험과 t-SNE 분석, learned weight visualization까지 제시했다. 특히 ADDA-grl과의 비교는 “weighting 자체의 기여”를 분리해서 보여준다는 점에서 좋다.

하지만 한계도 분명하다. 가장 먼저, weight가 domain classifier의 예측에 크게 의존한다. 초기 단계에서 $F_t$가 아직 충분히 정렬되지 않았을 때 첫 번째 classifier의 출력이 noisy할 수 있고, 그 경우 weight 자체가 부정확할 수 있다. 논문은 이를 완화하려고 $\lambda$를 작은 값에서 시작하는 schedule을 쓰지만, weight estimation의 안정성에 대한 보다 깊은 분석은 충분하지 않다.

또한 이 방법은 “source는 target의 모든 클래스를 포함한다”는 가정을 둔다. 즉 $\mathcal{Y}_t \subseteq \mathcal{Y}_s$가 핵심 전제다. 이 가정이 깨지는 open-set 또는 universal domain adaptation 환경에서는 바로 적용되기 어렵다. 논문도 이 범위를 넘는 문제를 다루지는 않는다.

더불어 target entropy minimization을 $F_t$에만 적용하는 설계는 흥미롭고 합리적이지만, 왜 이것이 항상 더 낫다고 볼 수 있는지는 실험적으로 아주 광범위하게 검증되지는 않았다. 논문은 $\gamma=0$ 버전과 기본 proposed를 비교하지만, entropy term의 세부 영향이나 sensitivity 분석은 제한적이다. 예를 들어 $\gamma$ 값 변화에 따른 안정성이나 데이터셋별 민감도는 본문에서 충분히 제시되지 않는다.

또 다른 한계는 방법의 클래스 인식 능력이 어디까지나 **간접적**이라는 점이다. SAN처럼 class-level probability를 명시적으로 쓰지 않기 때문에, coarse한 domain overlap signal만으로 source importance를 추정한다. 이것이 단순하고 장점이 되기도 하지만, 클래스 구조가 복잡하게 얽힌 경우에는 더 정교한 조건부 정렬보다 정보가 부족할 수 있다. 실제로 Caltech256 $\rightarrow$ Office10에서는 SAN이 약간 더 높은 평균 정확도를 보인다.

마지막으로, 논문은 주로 AlexNet 기반의 비교적 작은 규모 실험에 집중한다. 당시 기준으로는 충분하지만, 더 강한 backbone이나 더 복잡한 visual domain shift 환경에서 동일한 장점이 유지되는지는 이 본문만으로는 판단할 수 없다. 이는 논문 이후 후속 연구에서 추가적으로 검증되어야 할 부분이다.

## 6. 결론

이 논문은 unsupervised domain adaptation을 partial domain adaptation으로 확장한 연구로, 핵심 기여는 **source 샘플의 importance weight를 adversarial discriminator의 출력으로 추정하고, 이를 사용해 weighted source distribution과 target distribution을 정렬하는 것**이다. 이를 위해 두 개의 domain classifier를 사용하는 구조를 설계했고, 이론적으로는 weighted source density와 target density 사이의 Jensen-Shannon divergence를 줄이는 방식으로 해석했다. 또한 target entropy minimization을 결합해 unlabeled target의 구조를 보존하려 했다.

실험적으로 제안 방법은 partial setting에서 기존 RevGrad, RTN, ADDA-grl보다 명확히 우수하고, SAN과도 대체로 비슷하거나 더 나은 성능을 보인다. 특히 partiality가 강해질수록 이점이 커진다는 결과는 방법의 설계 의도와 잘 맞는다. 동시에 source 클래스 수에 비례해 많은 domain classifier를 필요로 하지 않는다는 점에서, 실용적인 확장성도 가진다.

이 연구는 이후 partial, open-set, universal domain adaptation 계열 연구들로 이어지는 중요한 아이디어를 담고 있다. 실제 응용에서도 큰 source 데이터셋에서 더 제한된 target 환경으로 적응해야 하는 상황은 매우 흔하므로, “무엇을 맞출지 선택적으로 결정하는 domain adaptation”이라는 관점은 지금도 유효하다. 이 논문은 바로 그 방향을 adversarial learning 관점에서 비교적 단순하고 명료하게 제시한 작업으로 볼 수 있다.
