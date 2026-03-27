# DACS: Domain Adaptation via Cross-domain Mixed Sampling

* **저자**: Wilhelm Tranheden, Viktor Olsson, Juliano Pinto, Lennart Svensson
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2007.08702](https://arxiv.org/abs/2007.08702)

## 1. 논문 개요

이 논문은 semantic segmentation에서의 unsupervised domain adaptation(UDA) 문제를 다룬다. 구체적으로는 source domain에는 라벨이 있지만 target domain에는 라벨이 전혀 없는 상황에서, source에서 학습한 segmentation 모델이 target에서도 잘 동작하도록 만드는 것이 목표다. 논문이 특히 겨냥하는 상황은 synthetic-to-real adaptation이며, 예를 들어 GTA5나 SYNTHIA 같은 합성 데이터로 학습한 모델을 Cityscapes 같은 실제 도로 장면에 적용하는 경우다.

문제의 핵심은 domain shift이다. semantic segmentation 네트워크는 같은 분포의 데이터에서는 높은 성능을 보이지만, 훈련 데이터와 테스트 데이터의 시각적 특성이 달라지면 성능이 급격히 떨어진다. 실제 응용, 특히 자율주행에서는 라벨링 비용이 매우 높기 때문에 target domain의 실제 이미지에 모두 수작업 라벨을 붙이는 것은 비현실적이다. 그래서 라벨이 있는 source와 라벨이 없는 target을 함께 사용해 적응하는 UDA가 중요하다.

기존 UDA segmentation 방법들 중 많은 방식은 pseudo-labeling에 기반한다. 즉, target 이미지에 대해 현재 모델이 예측한 결과를 임시 정답처럼 사용해 다시 학습한다. 하지만 domain gap이 크면 pseudo-label 품질이 낮아지고, 특히 쉬운 클래스에 예측이 편향되면서 어려운 클래스가 점점 사라지는 문제가 생긴다. 논문은 이 문제를 class conflation이라고 부르며, 예를 들어 sidewalk가 road로, rider가 person으로, terrain이 vegetation으로 계속 흡수되는 현상을 구체적으로 지적한다.

이 논문의 제안은 DACS(Domain Adaptation via Cross-domain mixed Sampling)이다. 핵심은 source 이미지와 target 이미지를 서로 섞어 새로운 mixed sample을 만든 뒤, 여기에 source의 ground-truth label과 target의 pseudo-label을 함께 섞은 label map을 붙여 학습하는 것이다. 저자들은 이 단순한 전략이 pseudo-label의 취약점을 완화하고, 기존 UDA benchmark에서 강한 성능을 낸다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 신뢰할 수 있는 source ground-truth와 불완전한 target pseudo-label을 같은 이미지 안에서 섞으면, target pseudo-label이 가진 구조적 오류를 완화할 수 있다는 것이다. 저자들은 기존 pseudo-label 기반 UDA가 target 데이터만 따로 다루면서 쉽고 자주 나타나는 클래스에 학습이 쏠린다고 본다. 반면 DACS는 source와 target을 한 장의 mixed image 안에 함께 넣음으로써, 모델이 두 domain을 이미지 단위로 쉽게 분리해서 학습하지 못하게 만든다.

조금 더 구체적으로 보면, source 이미지에서 일부 semantic class 영역을 선택해 잘라낸 뒤 이를 target 이미지 위에 붙인다. 그리고 label도 동일하게 source의 ground-truth label과 target의 pseudo-label을 같은 binary mask로 섞어 만든다. 이렇게 하면 mixed image 안에는 확실한 정답 픽셀과 불완전한 pseudo-label 픽셀이 공존하게 된다. 저자들의 설명에 따르면, 이때 pseudo-labeled 영역이 ground-truth 영역과 이웃하게 되므로, 네트워크가 domain별로 완전히 다른 class 분포를 암묵적으로 학습하는 것이 어려워진다.

기존 접근과의 차별점은 크게 두 가지다. 첫째, 많은 기존 pseudo-label 방법들은 낮은 confidence의 픽셀을 버리거나, 특정 이미지를 과샘플링하거나, class-balanced sampling 또는 uncertainty estimation을 통해 pseudo-label을 정제한다. 반면 DACS는 pseudo-label을 복잡하게 교정하기보다, augmentation 방식 자체를 바꿔 문제를 해결한다. 둘째, SSL에서 쓰이던 mixing 기반 consistency regularization을 UDA에 그대로 적용하는 naive mixing은 잘 작동하지 않는데, 논문은 이 실패 원인을 class conflation으로 분석하고, 이를 cross-domain mixing으로 해결했다는 점을 강조한다.

또 하나 중요한 포인트는, 이 논문이 mixed image의 realism 자체는 필수 조건이 아니라고 본다는 점이다. source와 target을 섞은 결과물이 시각적으로 다소 부자연스러울 수 있지만, 학습 관점에서는 강한 perturbation으로 작동하며 오히려 의미 있는 regularization이 될 수 있다는 해석이다. 즉, 자연스러운 합성 이미지를 만드는 것이 목적이 아니라, source의 확실한 supervision을 target 학습 신호와 결합하는 것이 목적이다.

## 3. 상세 방법 설명

전체 파이프라인은 매우 단순한 self-training 계열 구조 위에 cross-domain mixing을 얹은 형태다. 학습 시마다 source dataset $\mathcal{D}_S$에서 이미지-라벨 쌍 $(X_S, Y_S)$를 뽑고, target dataset $\mathcal{D}_T$에서 이미지 $X_T$를 뽑는다. 그런 다음 현재 segmentation network $f_{\theta}$를 이용해 $X_T$의 예측 semantic map $\hat{Y}_T$를 만든다. 이것이 target pseudo-label 역할을 한다.

그다음 DACS의 핵심 단계가 수행된다. source 이미지와 target 이미지를 binary mask를 통해 섞어 mixed image $X_M$를 만든다. 동시에 label도 같은 방식으로 섞어 mixed label $Y_M$를 만든다. 이때 source 쪽은 실제 정답 $Y_S$를 사용하고, target 쪽은 pseudo-label $\hat{Y}_T$를 사용한다. 논문에서 기본 mixing 전략으로 사용하는 것은 ClassMix이다. ClassMix에서는 source 이미지에 포함된 클래스들 중 절반을 선택하고, 그 클래스에 해당하는 픽셀들을 mask로 삼아 target 이미지 위에 붙인다. 즉, 직사각형 패치를 붙이는 CutMix와 달리 semantic class 경계를 따라 영역을 옮긴다.

이를 식으로 쓰면, 엄밀한 mask 식은 본문에 직접 명시되어 있지 않지만 개념적으로는 다음과 같이 이해할 수 있다.

$$
X_M = M \odot X_S + (1-M) \odot X_T
$$

$$
Y_M = M \odot Y_S + (1-M) \odot \hat{Y}_T
$$

여기서 $M$은 binary mask이고, source에서 선택된 클래스에 해당하는 위치에서는 1, 나머지 위치에서는 0이다. 실제 논문은 위 식을 직접 전개하지는 않지만, pseudocode와 설명상 이러한 구조가 분명하다.

저자들이 먼저 분석하는 것은 naive mixing이다. 이는 SSL에서처럼 target 이미지끼리만 섞는 방식이다. 즉, $X_{T_1}$과 $X_{T_2}$를 섞어 mixed image와 pseudo-label을 만든 뒤, source supervised loss와 함께 학습한다. 겉보기에는 자연스러운 확장처럼 보이지만, 실험 결과 일부 클래스가 거의 예측되지 않게 되며 심각한 class conflation이 발생한다. 논문은 이 현상이 pseudo-labeling을 UDA에 순진하게 적용했을 때 생기는 easy-to-transfer class bias와 같은 뿌리를 가진다고 해석한다.

이에 비해 DACS에서는 source ground-truth가 mixed label의 일부를 차지한다. 저자들의 설명을 따르면, cross-domain mixing은 두 가지 이유로 class conflation을 완화한다. 첫째, mixed image 전체가 pseudo-label에만 의존하지 않기 때문에, 이미지 전체 수준에서 conflated label만 존재하는 상황이 줄어든다. 둘째, ground-truth source 영역과 pseudo-labeled target 영역이 한 이미지 안에서 이웃하게 되어, 모델이 domain을 거칠게 구분해 서로 다른 클래스 분포를 학습하는 것이 어렵다. 다시 말해, domain gap을 회피하는 방식 대신 domain이 섞인 상태에서 segmentation을 해야 하므로, 더 일반화된 결정 경계를 학습하게 된다는 주장이다.

학습 목표는 다음 손실 함수로 정의된다.

$$
\mathcal{L}(\theta)=\mathbb{E}\left[H\big(f_{\theta}(X_S),Y_S\big)+\lambda H\big(f_{\theta}(X_M),Y_M\big)\right]
$$

여기서 $H$는 예측 semantic map과 label 사이의 cross-entropy loss이다. 첫 번째 항은 source supervised loss이고, 두 번째 항은 mixed image에 대한 loss다. 중요한 점은 $Y_M$이 source ground-truth와 target pseudo-label의 혼합이라는 것이다. 따라서 두 번째 항은 완전한 supervised loss는 아니지만, 논문은 이를 consistency regularization 성격의 학습으로 본다.

$\lambda$는 mixed sample loss가 전체 학습에 얼마나 반영될지를 조절하는 hyper-parameter다. 저자들은 고정값 대신 adaptive schedule을 사용한다. 각 이미지에 대해 모델 예측 confidence가 일정 threshold를 넘는 픽셀의 비율을 계산하고, 이를 $\lambda$로 사용한다. 즉, pseudo-label이 상대적으로 신뢰할 만할수록 unsupervised 성분의 영향이 커진다. threshold의 정확한 값은 본문 발췌에 명시되어 있지 않으므로, 그 부분은 추측할 수 없다.

Algorithm 1의 절차를 자연어로 정리하면 다음과 같다. 먼저 source 배치와 target 배치를 뽑는다. target 배치에 대해 pseudo-label을 생성한다. source와 target을 섞어 mixed image와 mixed label을 만든다. 네트워크는 source 이미지와 mixed image 모두에 대해 예측을 수행한다. 그리고 source 정답, mixed label을 기준으로 loss를 계산해 역전파한다. 이때 mixed label은 상수처럼 취급되므로 pseudo-label 생성 경로로 gradient가 흐르지 않는다. 이 과정을 $N$회 반복한다.

## 4. 실험 및 결과

실험은 두 개의 대표적인 synthetic-to-real semantic segmentation UDA benchmark에서 수행된다. 첫째는 GTA5 $\rightarrow$ Cityscapes이고, 둘째는 SYNTHIA $\rightarrow$ Cityscapes이다. Cityscapes는 target dataset이며 2,975장의 training image와 19개 클래스를 가진다. GTA5는 24,966장의 synthetic training image를 가지며 Cityscapes와 같은 19개 클래스를 갖는다. SYNTHIA는 9,400장의 synthetic training image를 가지며 Cityscapes의 19개 클래스 중 16개만 라벨이 존재한다.

평가 지표는 semantic segmentation에서 표준적으로 쓰이는 IoU와 mean IoU(mIoU)이다. 실험 모델은 DeepLab-v2 with ResNet101 backbone이며, ImageNet과 MSCOCO로 pretrained된 backbone을 사용한다. optimizer는 Nesterov acceleration이 포함된 SGD이고, 초기 learning rate는 $2.5 \times 10^{-4}$, weight decay는 $5 \times 10^{-4}$, momentum은 0.9이다. learning rate decay는 polynomial decay with exponent 0.9를 사용한다. source 이미지는 $760 \times 1280$, target 이미지는 $512 \times 1024$로 리사이즈한 뒤, 모두 $512 \times 512$ random crop을 사용한다. 학습은 2개의 source image와 2개의 mixed image로 이루어진 배치를 사용해 250k iterations 수행한다. mixed image에는 ClassMix 외에도 color jittering과 Gaussian blur를 추가로 적용한다.

### GTA5 $\rightarrow$ Cityscapes

가장 중요한 결과는 GTA5 $\rightarrow$ Cityscapes에서 DACS가 이전 방법들을 넘어서는 성능을 보였다는 점이다. source only baseline의 mIoU는 32.85였고, DACS는 52.14를 기록했다. 이는 표에 포함된 기존 방법들 중 최고였던 IAST의 51.5, PIT의 50.6, R-MRNet의 50.3, FDA의 50.45보다 높다. 따라서 저자들은 이 benchmark에서 state-of-the-art를 달성했다고 주장한다.

클래스별 성능을 보면 DACS는 road 89.90, building 87.87, vegetation 87.98, sky 88.76, person 67.20, truck 45.73, bus 50.19 등을 기록했다. 특히 기존 source baseline 대비 거의 모든 주요 클래스에서 큰 폭의 향상을 보였다. 예를 들어 sidewalk는 15.65에서 39.66으로, wall은 8.56에서 30.71로, fence는 15.17에서 39.52로, traffic sign은 15.00에서 52.79로 상승했다. 이는 DACS가 단순히 쉬운 클래스만 강화한 것이 아니라, 원래 적응이 어려운 클래스들에서도 개선이 있음을 보여준다.

정성적 결과에서도 DACS는 naive mixing보다 안정적이다. 논문 설명에 따르면, naive mixing은 sidewalk를 거의 항상 road로 예측하는 등 class conflation이 지속적으로 발생한다. 반면 DACS는 이런 오류를 크게 줄인다. 저자들은 Figure 5를 통해 source-only, naive mixing, DACS를 비교했으며, DACS가 시각적으로도 더 올바른 segmentation을 생성한다고 설명한다. 다만 제공된 발췌 텍스트에는 실제 figure 이미지가 포함되어 있지 않으므로, 여기서는 논문 설명 수준까지만 말할 수 있다.

### SYNTHIA $\rightarrow$ Cityscapes

SYNTHIA는 16개 클래스만 주어지므로, 문헌에 따라 13-class mIoU와 16-class mIoU를 모두 보고하는 경우가 있다. 논문도 이 두 기준 모두를 보고한다. DACS는 16-class 기준 mIoU 48.34, 13-class 기준 mIoU 54.81을 기록했다.

표를 보면 13-class 기준으로는 IAST가 57.0으로 더 높고, DACS는 그보다 낮다. 반면 16-class 기준에서는 IAST가 49.8, DACS가 48.34이므로 역시 최고 성능은 아니다. 즉, DACS의 가장 강한 실험적 기여는 GTA5 $\rightarrow$ Cityscapes에서의 SOTA이며, SYNTHIA에서도 강한 성능을 내지만 절대 최고는 아니다. 이 점은 보고서에서 분명히 구분해야 한다.

클래스별 성능으로는 DACS가 road 80.56, building 81.90, pole 37.20, vegetation 83.69, sky 90.77, person 67.61, rider 38.33, car 82.92 등을 기록했다. source baseline과 비교하면 큰 개선이 있다. 예를 들어 road는 36.30에서 80.56으로, rider는 11.34에서 38.33으로, bike는 20.66에서 47.58로 올랐다. 따라서 domain adaptation 효과 자체는 매우 분명하다.

### 평가 조건에 대한 논의

논문은 평가 프로토콜에 대해서도 흥미로운 문제 제기를 한다. Cityscapes에는 공개 test set 대신 validation set이 자주 최종 평가에 사용되는데, 많은 기존 방법들이 이 validation set 성능을 기준으로 early stopping을 하거나 민감한 hyper-parameter tuning을 수행했다고 지적한다. 저자들은 이것이 공정하지 않을 수 있다고 본다. DACS 역시 validation set 기준 early stopping을 쓰면 GTA5에서 52.14가 아니라 53.84까지 올라가고, source baseline도 32.85에서 35.68까지 오른다고 보고한다. 즉, validation fluctuation이 커서 best checkpoint만 보고하면 결과가 부풀려질 수 있다는 것이다.

또한 논문은 DACS 결과를 3회 실행 평균으로 보고한다. top-performing single run은 GTA5에서 54.09였다고 하므로, 평균 성능을 보고했다는 점에서 실험 보고 방식이 비교적 보수적이다. 이는 논문이 실험의 공정성을 중시한다는 신호로 볼 수 있다.

### 추가 실험: class conflation 분석과 mixing 전략

Table 3은 이 논문의 핵심 가설을 이해하는 데 매우 중요하다. 우선 naive mixing의 mIoU는 35.08로, source baseline 32.85보다 약간 높지만 DACS 52.14와는 큰 차이가 난다. 더 중요한 것은 per-class IoU에서 sidewalk 0.00, rider 0.00, train 0.00, bike 0.00 등 다수 클래스가 거의 완전히 붕괴한다는 점이다. 이는 class conflation이 실제로 심각하다는 직접 증거다.

pseudo-labeling만 사용하고 mixing을 제거한 경우는 더 나쁘다. mIoU는 22.97이며, sidewalk 0.03, wall 0.35, fence 0.03, pole 0.23, rider 0.01, train 0.00, bike 0.02처럼 더 많은 클래스가 사라진다. 저자들은 이를 근거로 conflation의 주된 원인이 mixing이 아니라 pseudo-labeling 자체에 있다고 해석한다. 즉, naive mixing도 문제를 일부 보여주지만, mixing이 문제의 근원이 아니라 target pseudo-label에 의존하는 학습이 본질적 원인이라는 것이다.

또 다른 흥미로운 비교는 distribution alignment 실험이다. 이는 target의 실제 class distribution $p$를 안다고 가정하고, 현재 예측 분포 $q$를 running average $\tilde{p}$와 함께 보정하는 방식이다.

$$
\tilde{q}=\text{Normalize}(q \times p / \tilde{p})
$$

이 방법은 현실적인 UDA 설정에서는 target ground-truth distribution을 모른다는 점에서 정당한 방법이 아니라고 저자들은 인정한다. 그럼에도 이 실험은 인위적으로 entropy를 주입하면 class conflation이 완화될 수 있음을 보여주는 참고 실험이다. 실제로 distribution alignment의 mIoU는 48.04이며, zero에 가까운 클래스들이 사라진다. 저자들은 이를 통해 “entropy injection”이 conflation 완화에 중요하다는 자신의 해석을 강화한다.

마지막으로 mixing strategy의 일반성을 보기 위해 ClassMix 대신 CutMix와 CowMix를 사용한 결과도 제시한다. DACS + CutMix는 48.69, DACS + CowMix는 48.30으로 기본 DACS의 52.14보다 낮지만, naive mixing에서 보이던 class conflation은 해결된 상태다. 이는 cross-domain mixing 자체가 핵심이고, 구체적 mask 생성 전략은 성능 차이를 만들 수 있으나 conflation 해소의 본질은 아니라는 해석을 뒷받침한다.

### source-target similarity에 대한 해석

논문은 DACS가 GTA5 $\rightarrow$ Cityscapes에서는 큰 이득을 보지만, SYNTHIA $\rightarrow$ Cityscapes에서는 상대적으로 이득이 작다는 점도 논의한다. 저자들은 이를 source와 target의 spatial class distribution 유사성 차이로 설명한다. GTA5와 Cityscapes는 카메라 시점과 장면 구성이 비슷해서 road는 아래, sky는 위, car는 중앙 부근에 놓이는 경향이 유사하다. 반면 SYNTHIA는 ground-level뿐 아니라 aerial-like perspective도 포함되어 있어 spatial layout이 더 다양하다. 이런 경우 source object를 target 이미지에 붙였을 때 더 비상식적인 mixed image가 만들어질 가능성이 높고, 이것이 학습에 불리할 수 있다는 가설이다. 이 부분은 논문의 해석이며, 이를 검증하는 별도의 정량 실험은 제공된 텍스트에 명시되어 있지 않다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 해결책이 모두 명확하다는 점이다. 저자들은 단순히 성능을 올렸다고 주장하는 것이 아니라, naive pseudo-label 기반 UDA에서 왜 문제가 생기는지 class conflation이라는 현상으로 구체화하고, 이를 실험적으로 보여준다. 그리고 DACS라는 매우 간단한 augmentation 변경만으로 이 문제를 상당 부분 해결한다. 복잡한 uncertainty module이나 adversarial alignment 없이도 강한 성능을 달성했다는 점은 실용적 가치가 크다.

또한 방법이 구조적으로 단순해서 기존 segmentation framework에 쉽게 붙일 수 있다는 장점이 있다. 실제로 DeepLab-v2 기반 설정에서 loss는 cross-entropy 두 항의 합으로 매우 간단하고, 추가되는 핵심은 mixed sample 생성뿐이다. 이는 구현 난이도와 재현성 면에서 유리하다. 더불어 ablation이 비교적 설득력 있게 구성되어 있어서, naive mixing, pseudo-label only, distribution alignment, 다른 mixing 전략 등을 통해 제안 방법의 작동 이유를 논리적으로 뒷받침한다.

실험 보고 방식도 장점이다. 저자들은 early stopping과 hyper-parameter tuning이 validation set을 과도하게 활용할 수 있다는 점을 공개적으로 지적하고, 자사의 결과를 평균 3회 실행 기준으로 제시한다. 이는 UDA benchmark에서 종종 간과되는 평가 공정성 문제를 환기한다는 점에서 가치가 있다.

한계도 분명하다. 첫째, DACS는 target pseudo-label의 품질 자체를 근본적으로 향상시키는 방법은 아니다. 잘못된 pseudo-label을 filtering하거나 uncertainty-aware correction으로 고치는 대신, source ground-truth를 섞어 문제를 완화하는 접근이다. 따라서 domain gap이 매우 큰 상황에서 pseudo-label이 지나치게 불안정하면 성능 이득이 제한될 수 있다. 저자들도 SYNTHIA 결과와 source-target similarity 논의를 통해 이 가능성을 인정한다.

둘째, mixed image의 의미적 타당성은 어느 정도 source와 target의 spatial layout 유사성에 의존한다. 논문은 realism이 필수는 아니라고 주장하지만, 동시에 GTA5와 Cityscapes가 서로 비슷할수록 더 잘 작동한다고 해석한다. 즉, DACS의 효과는 모든 종류의 domain shift에 동일하게 일반화된다고 보기 어렵다.

셋째, 분류된 원인 분석에는 설득력이 있지만 완전한 인과 검증은 아니다. 논문은 class conflation이 entropy minimization 혹은 pseudo-label bias와 연결된다고 설명하고, distribution alignment 실험으로 이를 간접 뒷받침한다. 그러나 왜 특정 클래스가 특히 사라지는지, spatial context와 class frequency 중 어떤 요인이 더 결정적인지에 대한 세밀한 분해 분석은 제공되지 않는다.

넷째, 일부 결과 해석에서는 주의가 필요하다. DACS는 GTA5 benchmark에서는 SOTA를 달성했지만, SYNTHIA에서는 최고 성능이 아니다. 따라서 이 방법을 “보편적으로 최고”라고 해석하면 과장이다. 정확히는 “특정 대표 benchmark에서 매우 강력하며, 특히 GTA5 $\rightarrow$ Cityscapes에서 두드러진 성과를 낸 방법”이라고 보는 것이 적절하다.

## 6. 결론

이 논문은 semantic segmentation를 위한 UDA에서 pseudo-label 기반 학습이 겪는 class conflation 문제를 분석하고, 이를 해결하기 위한 단순하면서도 효과적인 방법으로 DACS를 제안했다. 핵심 아이디어는 source 이미지와 target 이미지를 cross-domain으로 섞고, source ground-truth와 target pseudo-label을 함께 섞은 label로 학습하는 것이다. 이를 통해 모델이 domain별로 다른 class 구조를 암묵적으로 고착화하는 것을 막고, target 쪽 pseudo-label 학습을 더 안정화한다.

실험적으로 DACS는 GTA5 $\rightarrow$ Cityscapes에서 52.14 mIoU를 달성해 당시 SOTA를 갱신했고, SYNTHIA $\rightarrow$ Cityscapes에서도 강한 성능을 보였다. 추가 실험은 naive mixing과 pseudo-label only가 심각한 class conflation을 일으킨다는 점, 그리고 cross-domain mixing이 이 문제를 본질적으로 완화한다는 점을 잘 보여준다.

실제 적용 관점에서 이 연구의 중요성은, 복잡한 모듈 추가 없이도 데이터 augmentation 설계만으로 UDA 성능을 크게 높일 수 있음을 보여준 데 있다. 특히 합성 데이터와 실제 데이터 사이의 적응이 중요한 자율주행, 로보틱스, 도시 장면 해석 같은 영역에서 실용성이 높다. 향후 연구로는 더 큰 domain gap 상황에서의 강건성 분석, mixing mask 전략의 고도화, pseudo-label 품질 추정과의 결합, 그리고 source-target spatial mismatch가 큰 경우의 보정 기법 등이 자연스러운 확장 방향으로 보인다.
