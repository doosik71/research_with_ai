# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

이 논문은 computer vision에서 Bayesian deep learning이 실제로 어떤 종류의 uncertainty를 모델링해야 하는지 정면으로 다룬다. 저자들은 uncertainty를 크게 **aleatoric uncertainty** 와 **epistemic uncertainty** 로 나누고, 이 둘이 서로 대체 가능한 개념이 아니라 전혀 다른 역할을 가진다고 주장한다. 더 나아가, 입력 의존적인 **heteroscedastic aleatoric uncertainty** 와 모델의 불확실성을 나타내는 epistemic uncertainty를 하나의 unified framework 안에서 결합하고, 이를 **semantic segmentation** 과 **depth regression** 에 적용한다. 핵심 결론은 분명하다. 대규모 vision 데이터 환경에서는 aleatoric uncertainty를 명시적으로 모델링하는 것이 성능 향상과 robustness에 특히 중요하며, epistemic uncertainty는 out-of-distribution 상황이나 안전-critical failure를 감지하는 데 여전히 필요하다. 또한 저자들은 uncertainty를 단순히 “추가 출력”으로 다루지 않고, **loss attenuation** 으로 연결해 noisy data의 영향을 완화하는 새로운 loss formulation을 제안한다.

## 1. Paper Overview

이 논문의 연구 문제는 “딥러닝 비전 모델이 무엇을 모르는지 어떻게 표현할 것인가”이다. 일반적인 딥러닝 모델은 softmax score나 regression output을 내지만, 그것이 곧 신뢰도나 uncertainty를 의미하지는 않는다. 저자들은 실제로 perception 시스템의 오류가 치명적일 수 있다는 점을 강조하며, uncertainty estimation이 단순 부가 기능이 아니라 의사결정 안정성의 핵심이라고 본다. 문제는 기존 비전 모델들이 uncertainty를 제대로 표현하지 못했고, 특히 epistemic uncertainty는 computer vision에서는 모델링이 어렵다고 여겨졌다는 것이다. 이 논문은 그 공백을 메우기 위해 Bayesian deep learning 도구를 사용해 두 종류의 uncertainty를 함께 분석하고, 언제 무엇이 중요한지를 실험적으로 밝힌다.

또한 저자들은 big data regime에서는 epistemic uncertainty가 데이터가 많아질수록 상당 부분 설명될 수 있는 반면, **aleatoric uncertainty는 데이터가 아무리 많아도 사라지지 않는 observation noise** 라고 본다. 따라서 대규모 vision task에서는 aleatoric uncertainty를 명시적으로 모델링하는 편이 더 직접적인 실익을 준다고 주장한다. 그러나 동시에 aleatoric uncertainty만으로는 training distribution 밖의 예시를 식별할 수 없으므로, epistemic uncertainty도 여전히 필요하다고 말한다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 uncertainty를 두 종류로 분리해 이해하고, 이를 하나의 Bayesian deep learning framework 안에서 통합하는 것이다.

* **Aleatoric uncertainty**: 관측 자체에 내재한 noise. 센서 노이즈, motion blur, 멀리 있는 물체의 depth ambiguity 같은 것에 대응한다.
* **Epistemic uncertainty**: 모델 파라미터와 모델 자체에 대한 불확실성. 데이터가 충분해지면 줄어들 수 있으며, 보지 못한 상황에서 커진다.

논문의 novelty는 두 가지다. 첫째, heteroscedastic aleatoric uncertainty를 입력 의존적으로 예측하게 해 vision task의 noisy region을 모델이 스스로 식별하도록 만든다. 둘째, 이 uncertainty를 단순 부가 정보가 아니라 **loss attenuation** 과 연결해, noisy sample의 손실 기여를 자동으로 줄이는 학습 원리로 사용한다. 저자들은 regression뿐 아니라 classification에도 대응하는 formulation을 제시하며, 특히 classification 쪽 uncertainty modeling은 이 논문의 중요한 새 기여라고 명시한다.  

결국 이 논문은 “uncertainty를 잘 추정하자”가 아니라, **어떤 uncertainty가 어떤 역할을 하며, 이를 모델 구조·손실·추론 비용까지 포함한 전체 시스템으로 어떻게 설계할 것인가**를 제시한 논문이다.

## 3. Detailed Method Explanation

### 3.1 Epistemic uncertainty: Bayesian neural network 관점

저자들은 epistemic uncertainty를 Bayesian neural network(BNN)으로 설명한다. 기본 아이디어는 deterministic weight 대신 weight distribution을 두는 것이다. 예를 들어 weight prior를 다음처럼 둔다.

$$
\mathbf{W}\sim \mathcal{N}(0, I)
$$

그 뒤 데이터 $\mathbf{X}, \mathbf{Y}$ 가 주어졌을 때 posterior $p(\mathbf{W}\mid \mathbf{X}, \mathbf{Y})$ 를 구해 prediction 시 weight에 대해 marginalization 해야 한다. 이 posterior가 바로 “가능한 모델들에 대한 분포”이고, 여기서 나온 분산이 epistemic uncertainty다. 즉, 입력이 training distribution과 멀거나 데이터가 부족한 영역일수록, 어떤 모델이 맞는지 자신이 없기 때문에 epistemic uncertainty가 커진다.

논문은 exact Bayesian inference가 어렵기 때문에 approximate inference가 필요하다고 설명하고, 실험에서는 **Monte Carlo dropout** 을 사용해 epistemic uncertainty를 근사한다. 실험 설정상 convolutional layer마다 dropout $p=0.2$ 를 두고, **50 Monte Carlo dropout samples** 를 사용한다.

### 3.2 Aleatoric uncertainty: data noise 관점

Aleatoric uncertainty는 모델이 아니라 데이터의 불확실성이다. 논문은 이것을 다시

* **homoscedastic**: 입력과 무관하게 일정한 noise
* **heteroscedastic**: 입력에 따라 달라지는 noise

로 나눈다. 비전에서는 heteroscedastic 쪽이 특히 중요하다고 설명한다. 예를 들어 texture가 풍부하고 perspective cue가 뚜렷한 장면은 depth를 자신 있게 예측할 수 있지만, featureless wall처럼 단서가 빈약한 장면은 uncertainty가 높아진다. 즉, 모델이 입력마다 noise scale을 다르게 예측해야 한다는 것이다.  

Regression에서는 likelihood를 다음처럼 Gaussian으로 둘 수 있다.

$$
p(\mathbf{y}\mid \mathbf{f}^{\mathbf{W}}(\mathbf{x}))
=====================================================

\mathcal{N}(\mathbf{f}^{\mathbf{W}}(\mathbf{x}), \sigma^2)
$$

여기서 $\sigma$ 또는 $\sigma(\mathbf{x})$ 는 observation noise를 나타낸다. 핵심은 네트워크가 mean prediction뿐 아니라 noise scale도 함께 출력하게 만드는 것이다.

### 3.3 Unified model: 두 uncertainty의 결합

저자들은 heteroscedastic NN을 Bayesian NN으로 바꾸어, **aleatoric + epistemic uncertainty** 를 동시에 가지는 vision model을 구성한다. 말하자면,

* weight distribution으로 epistemic uncertainty를 표현하고
* 출력 또는 logit noise parameter로 aleatoric uncertainty를 표현하는

이중 구조다. 논문은 이 unified model을 통해 “aleatoric만”, “epistemic만”, “둘 다”를 각각 비교할 수 있게 만든다. 이 비교가 논문의 실험과 분석 전반의 핵심이다.

### 3.4 Loss attenuation: uncertainty가 손실을 조절하게 만들기

이 논문에서 가장 중요한 방법론적 포인트는 **aleatoric uncertainty가 learned attenuation으로 해석될 수 있다**는 점이다. 직관은 다음과 같다.

* 어떤 샘플이 매우 noisy하다면, 그 샘플의 오차를 모델이 과도하게 맞추려고 할 필요가 없다.
* 모델이 그 샘플의 uncertainty를 높게 예측하면, 해당 샘플의 residual loss 기여가 자동으로 줄어든다.
* 결과적으로 학습은 noisy observation에 덜 민감해지고, robustness가 올라간다.  

논문은 regression에서 이 원리를 먼저 설명하고, classification에 대해서도 complementary approach를 제안한다고 말한다. 현재 확보된 본문 조각에서는 classification 수식 전체가 완전하게 드러나진 않지만, 저자 스스로 classification용 uncertainty formulation을 이 논문의 novel contribution으로 내세운다. 따라서 이 논문의 핵심은 단순히 “dropout uncertainty”가 아니라, **uncertainty-aware objective를 통해 학습 자체를 바꾸었다**는 데 있다.

### 3.5 시각적 해석

Figure 1의 설명은 두 uncertainty의 차이를 매우 잘 보여 준다.

* **Aleatoric uncertainty** 는 object boundary, 멀리 있는 물체, depth edge처럼 본질적으로 noisy한 영역에서 높다.
* **Epistemic uncertainty** 는 semantically or visually challenging pixel처럼 모델이 확신하지 못하는 영역에서 높다.
* segmentation failure case에서 footpath를 잘못 분할했을 때, aleatoric보다 epistemic uncertainty가 더 크게 반응한다.  

즉, aleatoric은 “이 입력은 원래 애매하다”를, epistemic은 “내 모델이 이 상황을 잘 모른다”를 말해 준다.

## 4. Experiments and Findings

논문은 두 비전 과제를 실험 대상으로 삼는다.

* **per-pixel semantic segmentation**
* **depth regression**

또한 qualitative 결과 설명에서 segmentation은 **CamVid**, depth regression은 **Make3D** 데이터셋을 사용했다고 명시되어 있다. Make3D 예시에서는 ground-truth depth가 70m 이상 제공되지 않기 때문에, 먼 거리 영역에서 epistemic uncertainty signal이 크게 나타난다고 설명한다. 반면 aleatoric uncertainty는 depth edge나 distant points 주변에서 두드러진다.  

실험의 핵심 정량 메시지는 다음과 같다.

첫째, **aleatoric, epistemic 모두 성능을 개선하고, 둘을 함께 쓰면 더 좋다**고 저자들은 보고한다. 논문 초반 contribution에서 non-Bayesian baseline 대비 **1–3% 성능 향상** 을 주장하며, 이는 explicit aleatoric uncertainty representation이 noisy data 영향을 줄이기 때문이라고 설명한다.

둘째, uncertainty metric 자체의 품질도 분석한다. precision-recall curve를 통해 uncertainty가 높은 픽셀을 제거할수록 precision이 올라가며, 이는 uncertainty estimate가 실제 error와 상관관계가 있음을 보여 준다고 해석한다. calibration plot에서도 aleatoric, epistemic, 둘의 결합 모두 calibration mean squared error 개선을 보였다고 한다.  

셋째, 두 uncertainty가 포착하는 현상이 다르다는 점을 별도로 검증한다. training dataset 크기를 늘리면 **epistemic uncertainty는 감소**하지만 **aleatoric uncertainty는 거의 일정**하게 유지된다. 그리고 training distribution과 먼 test set에서는 epistemic uncertainty가 크게 증가한다. 이는 epistemic uncertainty가 충분한 데이터로 설명될 수 있지만, out-of-distribution 상황에서는 여전히 필요하다는 논문의 핵심 주장과 정확히 맞아떨어진다.  

넷째, real-time application 관점에서 trade-off도 분석한다. aleatoric model은 compute overhead가 거의 없지만, epistemic model은 MC dropout sampling 때문에 비싸다. 논문은 **50 Monte Carlo samples** 시 일부 구조에서는 **약 50배 slowdown** 이 날 수 있다고 지적한다. 따라서 성능과 안전성에는 epistemic uncertainty가 유용하지만, 배포 비용 관점에서는 aleatoric uncertainty가 훨씬 실용적일 수 있다는 메시지를 준다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 uncertainty를 개념적으로 매우 명확하게 분리했다는 점이다. 많은 후속 연구가 이 논문을 기준점으로 삼는 이유도 여기에 있다. aleatoric과 epistemic을 단순 정의 수준이 아니라, **어떤 현상에서 증가하는지, 데이터 양과 OOD 상황에 따라 어떻게 달라지는지**까지 실험으로 보여 준다.  

둘째, uncertainty를 loss 설계와 연결했다는 점이 강하다. 많은 논문이 uncertainty를 calibration이나 confidence score 수준에서만 다루는 반면, 이 논문은 **uncertainty가 학습 objective를 바꾸도록 설계**해 실제 성능 향상까지 이끈다. 이 때문에 contribution이 이론 설명에 그치지 않고 실질적인 optimization benefit으로 이어진다.  

셋째, 실무적 trade-off를 솔직하게 다룬다. epistemic uncertainty는 유용하지만 비용이 비싸고, aleatoric uncertainty는 cheap하면서도 big data vision regime에서 특히 가치가 크다고 정리한다. 이는 이후 실시간 perception 시스템 설계에 꽤 중요한 시사점을 준다.

### 한계

한계도 분명하다. 첫째, epistemic uncertainty 추정이 MC dropout에 크게 의존한다. 이는 practical approximation이지만 엄밀한 full Bayesian inference는 아니고, 특히 많은 dropout sample이 필요해 계산 비용이 크다. 저자들 스스로도 실시간 측면의 slowdown 문제를 인정한다.

둘째, 논문은 주로 segmentation과 depth regression에 집중한다. vision 전반에 대한 general claim을 제시하지만, detection, tracking, generative modeling 같은 더 넓은 task에 대한 직접적 검증은 제한적이다.

셋째, classification용 heteroscedastic formulation은 중요 기여이지만, 현재 확보된 본문 조각에서는 세부 유도식 전체를 모두 복원하기 어렵다. 따라서 이 부분은 논문의 핵심 메시지는 명확하지만, 수식 수준의 완전한 재현에는 원문 전체의 더 세밀한 확인이 필요하다. 지금 보고서에서는 논문이 직접 강조한 개념과 실험 해석을 중심으로 설명했다.  

### 해석

비판적으로 보면, 이 논문의 진짜 공헌은 “Bayesian deep learning을 vision에 썼다”보다도, **uncertainty를 역할별로 나누어 어떤 문제에서 무엇이 필요한지 정리한 것**이다. 저자들의 결론을 한 줄로 요약하면 이렇다.

* noisy large-scale vision training에는 aleatoric uncertainty가 특히 중요하고
* unseen situation detection이나 safety-critical failure analysis에는 epistemic uncertainty가 반드시 필요하다.

즉, 둘 중 하나가 “더 좋은 uncertainty”가 아니라, **서로 다른 failure mode를 다루는 보완적 도구**라는 것이다. 이 관점은 자율주행, 의료영상, 로보틱스처럼 고위험 비전 시스템에서 특히 중요하다.  

## 6. Conclusion

이 논문은 Bayesian deep learning for computer vision에서 필요한 uncertainty가 무엇인지 체계적으로 정리하고, **aleatoric uncertainty와 epistemic uncertainty를 통합한 unified framework** 를 제안했다. 방법론적으로는 입력 의존적 heteroscedastic uncertainty를 예측하게 하고, 이를 regression과 classification에서 **loss attenuation** 으로 해석해 noisy data에 robust한 학습을 가능하게 했다. 실험적으로는 semantic segmentation과 depth regression에서 성능 향상, calibration 개선, uncertainty quality 향상, OOD 상황에서 epistemic uncertainty의 중요성을 보여 주었다. 특히 대규모 vision 문제에서는 aleatoric uncertainty가 실용성과 성능 면에서 큰 가치를 가지며, epistemic uncertainty는 unseen failure를 감지하는 데 중요하다는 것이 논문의 핵심 결론이다.

이 논문이 이후 연구에 미친 영향은 상당하다. uncertainty를 confidence score의 문제로 축소하지 않고, **data noise와 model ignorance를 분리한 프레임** 을 제시했기 때문이다. 그래서 이 논문은 단순한 성능 개선 논문이라기보다, modern uncertainty-aware vision 시스템의 사고방식을 정리한 foundational paper로 읽는 것이 더 적절하다.
