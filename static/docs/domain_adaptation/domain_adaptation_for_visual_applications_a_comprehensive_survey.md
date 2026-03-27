# Domain Adaptation for Visual Applications: A Comprehensive Survey

* **저자**: Gabriela Csurka
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1702.05374](https://arxiv.org/abs/1702.05374)

## 1. 논문 개요

이 논문은 visual domain adaptation(DA) 분야를 폭넓게 정리한 survey이다. 핵심 목표는 컴퓨터 비전 응용에서 왜 domain adaptation이 필요한지 설명하고, transfer learning 전체 맥락 속에서 DA를 위치시킨 뒤, 당시까지 제안된 대표적인 방법들을 체계적으로 분류해 소개하는 것이다. 특히 image classification 중심의 전통적 shallow 방법과 deep learning 기반 방법을 함께 다루고, 더 나아가 object detection, segmentation, video analysis, attribute learning 등으로 확장된 응용까지 연결해 보여준다.

연구 문제가 되는 배경은 명확하다. 실제 환경에서는 학습 데이터와 테스트 데이터가 같은 분포를 따른다는 표준 머신러닝 가정이 자주 깨진다. 논문은 이를 domain shift라고 부르며, 배경, 조명, 촬영 위치, 자세 변화처럼 비교적 약한 차이부터, photo와 sketch, painting, NIR 영상처럼 표현 자체가 크게 다른 경우까지 포함한다고 설명한다. 이런 분포 차이는 source domain에서 잘 학습된 모델이 target domain에서 성능이 크게 떨어지게 만든다.

이 문제가 중요한 이유는 실제 서비스 환경에서 데이터 분포가 고객, 지역, 센서, 운영 환경에 따라 지속적으로 바뀌기 때문이다. 논문은 서비스 재배포 상황을 예로 들며, 새로운 환경마다 대량의 label을 다시 수집하는 것은 비용이 너무 크기 때문에, source의 labeled data와 target의 unlabeled 또는 소량 labeled data를 이용해 모델을 옮기는 기술이 필수적이라고 본다. 즉, 이 논문은 단순히 한 방법을 제안하는 것이 아니라, “왜 adaptation이 필요한가”와 “어떤 종류의 adaptation이 존재하는가”를 시야 넓게 정리하는 데 목적이 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 domain adaptation을 하나의 단일 기법으로 보지 않고, transfer learning의 하위 문제로 정식화한 뒤, 문제 설정과 해결 전략에 따라 체계적으로 분류하는 데 있다. 저자는 먼저 transfer learning과 domain adaptation의 개념적 경계를 분명히 하고, 이어서 homogeneous DA와 heterogeneous DA, unsupervised DA와 semi-supervised DA, shallow 방법과 deep 방법, 그리고 image classification을 넘는 다양한 vision 응용을 구분한다.

이 survey의 중요한 기여는 “방법의 나열”이 아니라 “정리의 축”을 제공한다는 점이다. 예를 들어 homogeneous DA에서는 같은 feature space 안에서 분포 차이를 줄이는 방법들을 instance re-weighting, parameter adaptation, feature augmentation, feature alignment, feature transformation, metric learning, optimal transport 등으로 분류한다. heterogeneous DA에서는 source와 target의 표현 공간이 다를 때 intermediate auxiliary domain을 쓰는 방법과 latent common space를 학습하는 symmetric/asymmetric transformation 방식으로 나눈다. deep DA에서는 deep feature를 shallow DA에 연결하는 접근, fine-tuning 중심 접근, 그리고 adaptation 자체를 network 안에 내장한 deepDA architecture로 다시 분해한다.

기존 접근과의 차별점은, 특정 하위 문제만 다루는 survey가 아니라 시각 인식 전체에서 domain adaptation이 어떻게 전개되었는지 큰 지도를 제공한다는 점이다. 또한 저자는 단순 분류 정확도 개선만이 아니라, object detection, tracking, segmentation, synthetic-to-real adaptation 같은 실용적 문제들까지 같이 다루면서, DA가 점점 더 복잡한 structured vision task로 이동하고 있음을 강조한다.

## 3. 상세 방법 설명

이 논문은 새로운 알고리즘을 하나 제안하는 논문이 아니라 survey이므로, “하나의 파이프라인”을 제시하지 않는다. 대신 domain adaptation 문제를 정의하는 수학적 틀과, 그 위에서 발전한 여러 방법군의 설계 원리를 설명한다.

먼저 domain은 feature space와 marginal distribution의 조합으로 정의된다. 즉, domain $\mathcal{D}$는 feature space $\mathcal{X} \subset \mathbb{R}^d$와 marginal distribution $P(\mathbf{X})$로 구성된다. task $\mathcal{T}$는 label space $\mathcal{Y}$와 conditional distribution $P(\mathbf{Y} \mid \mathbf{X})$로 정의된다. source domain과 target domain이 완전히 같으면 일반적인 supervised learning 문제이지만, $\mathcal{D}^s \neq \mathcal{D}^t$ 또는 $\mathcal{T}^s \neq \mathcal{T}^t$이면 transfer learning이 필요해진다.

domain adaptation은 그중에서도 보통 task는 같고 domain만 다른 경우를 다룬다. 논문은 전통적으로 $\mathcal{Y}^s=\mathcal{Y}^t$와 $P(\mathbf{Y}\mid \mathbf{X}^s)=P(\mathbf{Y}\mid \mathbf{X}^t)$를 가정하지만, 후자의 가정은 현실에서 너무 강하므로 실제로는 label space 공유만을 핵심 전제로 보는 쪽으로 완화된다고 설명한다. 또한 source에만 label이 있는 경우를 unsupervised DA, target에 소량 label이 있는 경우를 semi-supervised DA로 구분한다.

### 3.1 Homogeneous domain adaptation

homogeneous DA는 source와 target의 feature representation이 같은 경우, 즉 $\mathcal{X}^s=\mathcal{X}^t$이지만 $P(\mathbf{X}^s)\neq P(\mathbf{X}^t)$인 상황이다. 이 경우 방법들은 크게 다음과 같이 묶인다.

#### Instance re-weighting

가장 초기의 접근은 source sample마다 가중치를 다르게 주는 것이다. 핵심 직관은 source 전체를 동일하게 학습에 쓰지 말고, target 분포와 더 비슷한 source sample에 더 큰 비중을 두자는 것이다. 이는 covariate shift 가정, 즉 $P(\mathbf{Y}\mid\mathbf{X}^s)=P(\mathbf{Y}\mid\mathbf{X}^t)$이지만 입력 분포만 다르다는 상황에서 특히 자연스럽다.

가중치는 source와 target density ratio를 추정하거나, domain classifier로 source/target likelihood ratio를 구하거나, MMD(Maximum Mean Discrepancy) 같은 분포 거리로 계산할 수 있다. TrAdaBoost는 이 계열의 대표 예로, AdaBoost를 transfer learning에 맞게 바꾸어 target에서 틀린 샘플은 더 강조하고, source에서 틀린 샘플은 점점 덜 중요하게 만든다. 직관적으로 “target에 해가 되는 source 샘플을 تدريously 배제”하는 방식이다.

#### Parameter adaptation

이 부류는 source에서 학습된 classifier의 파라미터를 target에 맞게 수정한다. 대표적으로 Adaptive SVM(A-SVM)은 source classifier $f^s$에 perturbation function $\Delta f$를 더해 target decision boundary를 점진적으로 조정한다. 아이디어는 source 모델을 버리는 대신, target에 맞는 보정량만 추가로 학습하는 것이다.

이 방법들은 보통 target의 소량 labeled data가 필요하므로 semi-supervised DA에 가깝다. Domain Transfer SVM은 SVM 학습과 동시에 source-target 분포 차이(MMD)를 줄이고, A-MKL은 이를 다중 kernel로 일반화한다. DASVM은 transductive SVM 계열을 DA에 연결해 target unlabeled data를 더 직접적으로 학습 과정에 끌어들인다.

#### Feature augmentation

Daumé의 “Frustratingly Easy Domain Adaptation”은 가장 단순하지만 영향력 있는 아이디어 중 하나다. source feature $\mathbf{x}^s$를 $[\mathbf{x}^s,\mathbf{x}^s,\mathbf{0}]$로, target feature $\mathbf{x}^t$를 $[\mathbf{x}^t,\mathbf{0},\mathbf{x}^t]$로 확장하여, shared part와 domain-specific part를 동시에 학습하게 한다. 이 방식은 구현이 매우 간단하면서도 multi-source 확장도 자연스럽다.

Geodesic Flow Sampling(GFS)과 Geodesic Flow Kernel(GFK)은 feature augmentation의 더 기하학적 버전이다. source와 target을 Grassmann manifold 위의 subspace로 보고, 그 둘 사이 geodesic path를 따라 intermediate domain representation을 만들거나 적분한다. 즉, source에서 target으로 한 번에 점프하지 말고, 그 사이의 부드러운 경로를 따라가며 representation을 이어 붙이자는 생각이다.

#### Feature space alignment

이 부류는 source feature를 target feature 쪽으로 정렬한다. 대표적인 SA(Subspace Alignment)는 source PCA subspace와 target PCA subspace를 구한 뒤, source를 target subspace 기준으로 재표현한다. 논문에 제시된 알고리즘은 매우 단순하다. source 데이터 $\mathbf{X}^s$와 target 데이터 $\mathbf{X}^t$에 대해 PCA를 수행해 $\mathbf{P}_s,\mathbf{P}_t$를 얻고, 정렬된 source를 다음처럼 만든다.

$$
\mathbf{X}^{s}_{a} = \mathbf{X}^{s}\mathbf{P}_{s}\mathbf{P}_{s}^{\top}\mathbf{P}_{t}, \quad
\mathbf{X}^{t}_{a} = \mathbf{X}^{t}\mathbf{P}_{t}
$$

즉, source를 source subspace 안에서 표현한 뒤 그것을 target subspace 방향으로 회전시키는 구조다.

CORAL은 더 단순하다. source covariance를 whitening하고, target covariance로 re-coloring하여 2차 통계량을 맞춘다. 핵심은 평균보다 covariance alignment가 domain shift 완화에 중요하다는 것이다. 논문에 따르면 CORAL은 몇 줄의 MATLAB 코드로도 구현될 정도로 간단하다.

#### Feature transformation

이 계열은 source와 target을 어떤 latent space로 투영하여 그 공간에서 두 분포가 가까워지도록 만든다. TCA(Transfer Component Analysis)는 공통 latent feature를 찾으면서 source와 target의 marginal distribution 차이를 줄이고, 동시에 데이터 manifold의 local geometry를 유지하려 한다.

DIP와 SIE는 단순 평균 차이보다 더 정교한 분포 비교를 시도한다. DIP는 RKHS에서 분포 차이를 비교하면서 orthogonal projection을 강제하고, SIE는 probability distribution이 Riemannian manifold 위에 있다는 관점을 활용해 Hellinger distance로 분포 차이를 줄인다. TSC와 TJM은 sparse coding이나 kernel embedding을 통해 더 견고한 공통 표현을 학습한다.

MDA(Marginalized Denoising Autoencoder)는 매우 흥미롭다. 입력 feature에 dropout 형태의 noise가 들어간 상황에서 원래 feature를 복원하도록 representation을 학습한다. 일반 denoising autoencoder와 달리 noise를 실제로 샘플링하지 않고 marginalize해서 closed-form 해를 얻는다. 논문이 제시한 핵심 식은 다음과 같다.

$$
\mathbf{W} = (\mathbf{Q} + \omega \mathbf{I}_{D})^{-1}\mathbf{P}
$$

여기서 $\mathbf{P}$와 $\mathbf{Q}$는 noise level $p$를 반영한 통계량이며, $\omega$는 regularizer다. 여러 층을 쌓을 때는

$$
\mathbf{X}_{(k)} = \tanh(\mathbf{X}_{(k-1)}\mathbf{W}^{(k)})
$$

처럼 비선형을 끼워 넣어 stacked MDA로 확장할 수 있다. 이 구조는 이후 deep DA와도 자연스럽게 연결된다.

#### Supervised feature transformation 및 metric learning

class label을 활용하면 더 discriminative한 adaptation이 가능하다. Semi-supervised TCA는 label dependency term을 추가해 projection이 class structure와도 맞도록 만든다. MDA 계열도 source classifier의 출력을 보존하는 regularization을 추가하는 방식으로 확장된다.

metric learning 기반 방법은 source와 target 사이 거리를 직접 재설계한다. Information-Theoretic Metric Learning, DSCM, NBNN-DA, SaML-DA 등이 이에 속한다. 예를 들어 DSCM은 각 sample이 domain-specific class mean에 얼마나 가까운지를 softmax 형태 거리로 최적화하여, 같은 클래스는 더 가깝고 다른 클래스는 더 멀어지도록 한다.

#### Local transformation, optimal transport, landmark selection

global transformation만으로는 부족할 수 있기 때문에, 일부 방법은 sample-level local transformation을 학습한다. ATTM은 target을 Gaussian Mixture Model로 표현하고 locally linear한 translation을 추정한다. Optimal Transport(OTDA)는 source sample을 target sample 쪽으로 “운반”하는 transportation plan을 찾는다. 이는 분포 정렬을 개별 sample 매칭 관점으로 본다는 점에서 직관적이다.

landmark selection은 adaptation 이전에, source 중에서 target과 더 잘 맞는 일부 샘플만 골라 쓰는 전처리 방식이다. landmark는 일종의 binary re-weighting과 비슷하지만, 논문은 이를 adaptation 자체보다는 preprocessing으로 본다.

### 3.2 Multi-source domain adaptation

source domain이 여러 개 있을 때는 단순 결합이 항상 좋은 선택이 아니다. 각 source 간에도 shift가 있을 수 있기 때문이다. 그래서 multi-source DA는 source별 특수성을 보존하면서 target에 유용한 정보만 조합하려 한다.

기존 FA나 A-SVM도 자연스럽게 multi-source 확장 가능하다. 더 나아가 Domain Adaptation Machine은 source별 classifier의 decision value가 target unlabeled sample에서 target classifier와 유사해지도록 regularization한다. CP-MDA는 conditional probability 차이에 따라 source별 가중치를 조절한다. RDALRR은 각 source를 intermediate representation으로 옮긴 뒤 target으로 잘 reconstruction되도록 low-rank 구조를 강제한다.

또한 여러 source 가운데 어떤 source가 target에 더 유익한지를 평가하는 source weighting 문제도 중요하게 다뤄진다. 논문은 conditional probability 차이, local graph similarity, latent space에서의 KL divergence, leave-one-out 기반 source relevance 평가 등을 소개한다. 이는 negative transfer를 줄이기 위한 핵심 장치다.

### 3.3 Heterogeneous domain adaptation

heterogeneous DA(HDA)는 source와 target의 feature space가 다른 경우를 다룬다. 예를 들어 source는 text, target은 image일 수 있고, source는 RGB face, target은 NIR face일 수 있다. 이런 경우는 단순한 분포 차이뿐 아니라 representation gap까지 동시에 해결해야 한다.

한 계열의 방법은 auxiliary multi-view domain을 사용한다. 예를 들어 text와 image를 연결하고 싶다면, 웹에서 text와 image가 함께 존재하는 페이지를 intermediate domain으로 모을 수 있다. 이 경우 co-occurrence 정보를 통해 source와 target 사이를 잇는 translator 또는 latent topic space를 만든다. Mixed-Transfer, Transitive Transfer Learning, semantic propagation 계열 방법이 이에 해당한다.

다른 계열은 직접 공통 latent space를 학습한다. symmetric transformation 방식은 source와 target 각각에서 projection을 학습해 공통 공간으로 보낸다. HFA(Heterogeneous Feature Augmentation)는 먼저 양쪽을 공통 latent space에 embed한 뒤 feature augmentation을 수행한다. SDDL은 공통 잠재공간에서 class-wise discriminative dictionary를 학습한다. DAMA는 각 domain을 manifold로 보고, 구조를 보존하는 mapping을 각각 학습한다.

반대로 asymmetric transformation은 source를 target space로 직접 옮기려 한다. Asymmetric Regularized Cross-domain Transformation은 비선형 kernel 공간에서 domain-invariant mapping을 학습하고, Multiple Outlook Mapping은 클래스별 분포 구조를 맞추는 선형 변환을 찾는다. 이런 방법은 일반적으로 target에 소량의 labeled data가 필요한 semi-supervised 설정에 적합하다.

## 4. 실험 및 결과

이 논문은 survey이기 때문에 하나의 데이터셋에서 통일된 실험을 수행하지 않는다. 따라서 일반적인 연구 논문처럼 특정 baseline과 수치 결과를 비교하는 “자체 실험 섹션”이 없다. 오히려 저자는 왜 공정한 정량 비교가 어려운지 자체적으로 설명한다.

첫째, 많은 논문이 Office31, Office+Caltech10 같은 공용 벤치마크를 사용하지만, 실험 protocol이 서로 다르다. 어떤 논문은 source 일부만 샘플링하고, 어떤 논문은 전체를 사용하며, unsupervised setting과 semi-supervised setting이 섞여 있다. 둘째, 같은 방법이라도 구현마다 결과가 크게 다를 수 있다. 셋째, 오래된 shallow 방법은 SURFBOV 같은 전통 feature로 실험되었고, 이후 deep feature 기반 방법과 직접 비교하기 어렵다.

그럼에도 이 survey가 전달하는 중요한 실험적 메시지는 몇 가지가 있다.

먼저, deep convolutional feature 자체가 매우 강력한 baseline이라는 점이다. 논문은 DeCAF 같은 deep activation feature를 source와 target에 추출한 뒤 adaptation 없이 사용해도, Office31이나 Office+Caltech10에서는 많은 shallow DA 방법과 비슷하거나 더 좋은 결과를 내는 경우가 있음을 강조한다. 이는 deep representation이 category-level abstraction을 학습하고 일부 domain bias를 자연스럽게 줄이기 때문이라고 해석한다.

하지만 이 성능 우위가 모든 상황에 일반화되는 것은 아니다. photo와 painting, sketch, NIR처럼 modality 차이가 크거나 시각적 스타일 차이가 큰 경우에는 단순 deep feature만으로는 충분하지 않다고 설명한다. 그래서 discrepancy-based deepDA, adversarial deepDA, reconstruction-based deepDA, heterogeneous deepDA 같은 방법들이 등장한다.

응용 측면에서는 object detection, video event recognition, action recognition, tracking, semantic segmentation, pose estimation 등에서 DA가 활용되는 사례를 폭넓게 소개한다. 특히 synthetic-to-real adaptation은 중요한 흐름으로 제시된다. SYNTHIA, Virtual KITTI, GTA-V 같은 synthetic dataset은 annotation을 거의 공짜로 제공하지만, real-world와 차이가 크기 때문에 domain adaptation이 필요하다. 이 분야에서 domain confusion loss, contrastive loss, multi-task learning, synthetic-real joint training이 유망한 방향으로 언급된다.

결국 이 논문의 “결과”는 특정 숫자보다도 다음의 해석에 가깝다. 작은 고전 벤치마크에서는 deep feature와 deepDA가 매우 강력하며, 앞으로는 더 크고 더 어려운 heterogeneous/synthetic/structured-output benchmark가 필요하다는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 정리의 폭과 구조화 수준이다. domain adaptation을 단순히 몇몇 대표 논문 나열로 끝내지 않고, transfer learning의 큰 틀 속에서 formal definition부터 homogeneous/heterogeneous, shallow/deep, classification beyond classification, synthetic data, object detection, online adaptation까지 하나의 지도처럼 정돈한다. 특히 visual application에 초점을 맞추면서도 text, speech, multilingual classification 등 비전 밖 사례를 잠깐씩 연결해 DA의 일반성을 잘 보여준다.

두 번째 강점은 방법을 “무엇을 줄이려 하는가”라는 관점에서 재구성한다는 점이다. 어떤 방법은 sample importance를 조절하고, 어떤 방법은 classifier parameter를 보정하며, 어떤 방법은 covariance나 MMD를 맞추고, 어떤 방법은 adversarial objective로 domain confusion을 유도한다. 이런 정리는 독자가 새로운 논문을 볼 때도 어느 계열에 속하는지 빠르게 이해하게 해 준다.

세 번째 강점은 deep learning 전환기의 중요한 통찰을 담고 있다는 점이다. 저자는 deep feature가 이미 강한 baseline이라는 사실과, 그 위에서 adaptation을 network 내부로 통합하는 deepDA가 왜 더 유망한지 분명히 설명한다. 또한 synthetic-to-real, detection, tracking 같은 응용 확장이 왜 중요한지 잘 짚는다.

하지만 한계도 분명하다. 무엇보다 이 논문은 survey이므로 개별 방법의 수학적 세부 유도나 실험적 비교가 깊게 들어가지는 않는다. 많은 방법을 폭넓게 언급하는 대신, 각 방법의 objective function 전체나 optimization의 세부 증명은 대부분 생략된다. 따라서 입문자에게는 큰 그림을 주지만, 특정 방법을 실제 구현하거나 엄밀히 비교하려는 연구자에게는 원 논문을 추가로 봐야 한다.

또한 저자 스스로 인정하듯, 공정한 실험 비교가 부재하다. 이는 단순한 약점이라기보다 분야 자체의 한계에 대한 솔직한 진단이지만, 독자 입장에서는 “어떤 방법이 실제로 더 좋은가”에 대한 직접적인 답을 얻기 어렵다. 특히 2016년 전후의 deepDA 초창기 결과들은 이후 훨씬 큰 benchmark와 강력한 backbone이 등장하면서 해석이 달라질 수 있다.

비판적으로 보면, 이 논문은 당시 시점에서 매우 충실한 survey이지만, foundation model이나 self-supervised pretraining, large-scale vision-language model이 adaptation 문제를 어떻게 바꾸는지는 다루지 못한다. 다만 이는 논문의 연도상 당연한 제한이며, 오히려 당시까지의 고전적 DA 문제 설정을 이해하는 데는 장점이 된다.

## 6. 결론

이 논문은 visual domain adaptation 분야를 체계적으로 정리한 포괄적 survey로서, domain shift가 현실의 컴퓨터 비전 시스템에서 왜 본질적인 문제인지, 그리고 이를 해결하기 위해 어떤 계열의 방법이 발전해 왔는지를 넓고 균형 있게 설명한다. 핵심 기여는 새로운 알고리즘 제안이 아니라, transfer learning 안에서 DA를 형식적으로 자리매김하고, homogeneous DA, multi-source DA, heterogeneous DA, deep DA, 그리고 image classification을 넘어서는 여러 응용을 일관된 관점으로 정리한 데 있다.

실무적 관점에서 이 논문은 매우 유용하다. 새로운 환경에 모델을 배포해야 할 때, 문제를 먼저 “feature space가 같은가 다른가”, “target label이 전혀 없는가 조금 있는가”, “분포 정렬이 필요한가 구조적 출력까지 적응해야 하는가”로 분해하게 해 주기 때문이다. 연구 관점에서는 deep representation의 강력함과 동시에, 더 도전적인 benchmark와 structured prediction, synthetic-to-real, heterogeneous data adaptation이 앞으로 중요해질 것이라는 방향성을 제시한다.

정리하면, 이 논문은 visual domain adaptation의 고전적 기반을 이해하기 위한 매우 좋은 안내서다. 개별 알고리즘의 세부 구현서라기보다는, 분야 전체의 문제 설정과 해법의 계보를 잡아 주는 지도에 가깝다. 특히 이제 막 domain adaptation 연구를 시작하는 사람이나, 컴퓨터 비전에서 transfer learning의 역사적 맥락을 이해하고 싶은 사람에게 가치가 크다.
