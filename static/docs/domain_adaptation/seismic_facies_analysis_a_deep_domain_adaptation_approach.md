# Seismic Facies Analysis: A Deep Domain Adaptation Approach

* **저자**: M Quamer Nasim, Tannistha Maiti, Ayush Srivastava, Tarry Singh, Jie Mei
* **발표연도**: 2022
* **arXiv**: [https://arxiv.org/abs/2011.10510](https://arxiv.org/abs/2011.10510)

## 1. 논문 개요

이 논문은 seismic facies analysis를 위해 deep learning 기반의 semantic segmentation과 unsupervised deep domain adaptation을 결합한 접근을 제안한다. 논문의 핵심 배경은 두 가지다. 첫째, deep neural network는 일반적으로 많은 양의 labeled data가 있을 때 높은 성능을 보이지만, 실제 지구물리 데이터 해석에서는 라벨이 풍부하지 않은 경우가 많다. 둘째, 한 지역이나 조사 구역에서 학습한 모델이 다른 지역의 seismic data에 그대로 일반화되지 않는 distribution shift 문제가 자주 발생한다. 논문은 이 두 문제를 동시에 겨냥한다.

구체적으로 저자들은 offshore Netherlands의 F3 block 3D dataset을 source domain으로, Canada의 Penobscot 3D survey data를 target domain으로 설정한다. 그리고 두 도메인에서 reflection pattern이 유사한 세 개의 geological class를 대상으로 실험한다. 이 설정은 실제 현장에서 자주 맞닥뜨리는 문제를 잘 반영한다. 즉, 어떤 지질 구조에 대해서는 source domain에는 라벨이 있지만 target domain에는 라벨이 없거나 매우 부족하며, 또 두 데이터의 분포가 다르기 때문에 단순한 supervised transfer가 잘 작동하지 않을 수 있다.

논문의 중요성은 분명하다. seismic facies classification 또는 segmentation은 해석 자동화, 지질 구조 파악, 탐사 효율 향상과 직결된다. 그런데 라벨링은 고비용이고 전문가 의존적이다. 따라서 데이터 부족과 도메인 차이를 동시에 다루는 방법은 응용 가치가 높다. 이 논문은 하나의 architecture 수준 제안인 EarthAdaptNet과, 이를 바탕으로 한 unsupervised deep domain adaptation 모델 EAN-DDA를 통해 이러한 실제적 제약을 해결하려고 시도한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 두 단계로 이해할 수 있다.

첫 번째는 seismic image semantic segmentation을 위한 전용 네트워크 EarthAdaptNet, 줄여서 EAN을 설계한 것이다. 저자들은 소수 클래스(minority classes)에서 데이터 부족이 있는 상황을 특히 염두에 두고, decoder block 안에서 traditional dilated convolution 대신 transposed residual unit을 사용한다. 이 설계는 abstract 수준 정보만 보면, 복원 과정에서 보다 효과적으로 feature를 재구성하고 세부 spatial structure를 유지하려는 의도로 읽힌다. 즉, 일반적인 segmentation architecture를 그대로 쓰기보다 seismic reflection pattern의 구조적 특성과 class imbalance 또는 scarcity 문제를 고려한 architecture 변형을 제안한 것이다.

두 번째는 EAN에 CORAL(Correlation Alignment)을 결합해 EAN-DDA라는 unsupervised deep domain adaptation 네트워크를 구성한 것이다. 여기서 핵심 직관은 source domain과 target domain 사이의 feature distribution 차이를 줄이면, target domain에 라벨이 없어도 source에서 학습한 표현이 target에서 더 잘 작동할 수 있다는 점이다. CORAL은 보통 source와 target의 feature covariance structure를 정렬하도록 만드는 방식으로 알려져 있으며, 이 논문은 이를 seismic reflection classification 문제에 도입한다. 따라서 이 논문의 핵심은 단순히 “좋은 segmentation 모델을 만들었다”에 그치지 않고, “도메인이 다른 seismic survey 사이에서도 활용 가능한 representation을 학습하자”는 데 있다.

기존 접근과의 차별점은 제공된 텍스트 기준으로 두 가지다. 하나는 seismic facies segmentation을 위해 EAN이라는 bespoke architecture를 제안했다는 점이고, 다른 하나는 seismic reflection classification에 대해 unsupervised deep domain adaptation을 실험적으로 보여주었다는 점이다. 다만 논문 원문 전체의 method section이 제공되지 않았기 때문에, 기존 어떤 architecture와 layer-by-layer로 어떻게 다른지까지는 여기서 단정적으로 설명할 수 없다.

## 3. 상세 방법 설명

제공된 텍스트만 기준으로 보면, 논문은 서로 연결된 두 개의 방법론을 제안한다. 하나는 supervised semantic segmentation 모델인 EarthAdaptNet(EAN)이고, 다른 하나는 unsupervised deep domain adaptation 모델인 EAN-DDA이다.

먼저 EAN은 seismic images를 입력으로 받아 pixel-wise semantic segmentation을 수행하는 네트워크이다. 즉, 각 픽셀이 어느 geological facies class에 속하는지를 예측하는 구조다. abstract에 따르면 EAN은 “few classes have data scarcity” 상황을 다루기 위해 설계되었다. 이는 일부 클래스는 샘플 수가 적어서 일반적인 supervised training에서 쉽게 성능이 떨어질 수 있음을 뜻한다. 저자들은 decoder block에서 traditional dilated convolution 대신 transposed residual unit을 사용했다고 설명한다. 여기서 transposed residual unit은 이름상으로는 transposed convolution 계열의 upsampling 또는 reconstruction 기능과 residual learning을 결합한 모듈로 이해할 수 있다. 다시 말해 encoder에서 압축된 feature를 decoder에서 복원할 때, 단순한 dilated convolution보다 더 직접적으로 해상도를 복구하면서 residual connection을 통해 학습을 안정화했을 가능성이 높다. 그러나 제공된 텍스트에는 이 unit의 정확한 내부 수식이나 layer 구성은 없다. 따라서 이 부분은 구조적 의도 수준까지만 설명할 수 있고, 세부 연산 정의는 논문 본문 없이는 확정할 수 없다.

EAN의 학습 목표는 문맥상 semantic segmentation loss를 최소화하는 것이다. 다만 제공된 텍스트에는 cross-entropy를 사용했는지, class weighting을 적용했는지, Dice 계열 보조 손실을 썼는지 등의 정보가 명시되어 있지 않다. 따라서 손실 함수를 구체적인 식으로 재현하는 것은 불가능하다. 마찬가지로 optimizer, learning rate schedule, batch size, data augmentation 절차도 주어진 텍스트 안에는 없다.

다음으로 EAN-DDA는 EAN에 CORAL을 도입한 unsupervised deep domain adaptation 네트워크이다. 이 모델의 목적은 source domain인 F3에서 학습한 feature가 target domain인 Penobscot에도 유효하도록 source-target feature distribution을 정렬하는 것이다. CORAL의 일반적인 목적은 source feature covariance와 target feature covariance 사이 차이를 줄이는 데 있다. 직관적으로 쓰면, source와 target의 feature representation이 1차 평균뿐 아니라 2차 통계 구조까지 유사해지도록 만들어 도메인 차이를 줄이는 것이다. 제공된 텍스트에는 실제 식이 없지만, CORAL의 핵심은 대체로 다음과 같은 형태의 정렬 목표로 이해할 수 있다.

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \mathcal{L}_{CORAL}
$$

여기서 $\mathcal{L}_{task}$는 source domain에서의 supervised classification 또는 segmentation objective이고, $\mathcal{L}_{CORAL}$은 source와 target feature의 correlation 또는 covariance 구조 차이를 줄이는 항이다. $\lambda$는 두 목적 사이 균형을 조절하는 계수다. 다만 위 식은 CORAL류 방법의 일반적인 형태를 설명하기 위한 것이며, 논문 본문에서 실제로 동일한 식을 사용했다고 단정할 수는 없다. 제공된 텍스트에는 “introduce the CORAL method to the EAN to create an unsupervised deep domain adaptation network”라고만 되어 있기 때문이다.

또 하나 주의할 점은, abstract에서 EAN은 semantic segmentation 모델로 소개되지만 EAN-DDA는 “classification of seismic reflections from F3 and Penobscot”에 사용되었다고 적혀 있다. 즉, adaptation 실험에서는 pixel-level segmentation이 아니라 reflection class classification 형태로 문제가 재구성되었을 가능성이 있다. 이것은 중요한 차이다. segmentation과 classification은 출력 구조, loss 설계, 평가 방식이 다르기 때문이다. 그러나 제공된 텍스트만으로는 EAN에서 EAN-DDA로 넘어가면서 output head가 어떻게 바뀌는지, feature extractor는 어떤 식으로 공유되는지, target에서 어떤 단위의 샘플을 쓰는지까지는 확인할 수 없다. 따라서 이 논문의 방법론을 정확히 말하면, seismic segmentation backbone 또는 feature extractor로 EAN을 설계하고, 그 표현에 CORAL 기반 domain alignment를 결합하여 target domain classification 가능성을 실험한 것으로 이해하는 것이 가장 안전하다.

## 4. 실험 및 결과

실험은 두 데이터셋을 중심으로 진행된다. source domain은 offshore Netherlands의 F3 block 3D dataset이고, target domain은 Canada의 Penobscot 3D survey data이다. 두 데이터셋은 지리적으로도 다르고 취득 환경과 subsurface condition도 다를 가능성이 높기 때문에, domain shift를 실험하기에 적절한 조합이다. 저자들은 두 도메인에서 reflection pattern이 유사한 세 개의 geological class를 선택해 비교했다고 밝힌다. 이 점은 실험 설정의 현실성을 높이지만, 동시에 전체 facies taxonomy가 아니라 일부 유사 클래스에 한정된 평가라는 점도 의미한다.

EAN의 경우 semantic segmentation 성능이 보고된다. abstract에 따르면 EAN은 pixel-level accuracy가 84% 이상이며, minority classes에 대해서도 약 70%의 accuracy를 달성했다. 이 결과는 단순한 전체 정확도만 높고 소수 클래스는 놓치는 모델이 아니라, 데이터가 부족한 클래스에도 일정 수준의 성능을 낸다는 점을 보여준다. 특히 seismic facies 문제에서는 dominant class에 비해 minority class가 훨씬 해석적으로 중요할 수 있으므로, minority class 성능 보고는 의미가 있다. 저자들은 또한 EAN이 existing architectures보다 개선된 성능을 보였다고 주장한다. 다만 제공된 텍스트에는 비교 대상 모델 이름, 각 baseline의 수치, 통계적 유의성 여부는 없다. 따라서 “얼마나 크게 개선되었는지”, “어떤 baseline 대비 우수한지”는 현재 자료만으로는 세부 분석이 불가능하다.

EAN-DDA의 경우는 unsupervised domain adaptation 기반 classification 실험이 보고된다. target domain 라벨이 없다는 설정에서 Penobscot의 특정 클래스에 대해 최대 class accuracy 약 99%를 달성했고, overall accuracy는 50% 이상이라고 한다. 이 결과는 다소 비대칭적으로 해석할 필요가 있다. 한 클래스에서 매우 높은 정확도가 나온 것은 특정 reflection pattern이 source-target 사이에서 잘 정렬되었음을 시사하지만, overall accuracy가 50%를 약간 넘는 수준이라는 점은 도메인 적응이 전체적으로는 아직 어려운 문제임을 보여준다. 다시 말해 이 논문은 “완전한 해결”보다는 “라벨 없는 target seismic facies classification도 가능성이 있다”는 실증을 제시한 것으로 보는 편이 타당하다.

정리하면, 실험 결과는 두 메시지를 전달한다. 첫째, EAN은 segmentation 문제에서 비교적 높은 pixel accuracy와 minority class 성능을 동시에 확보했다. 둘째, EAN-DDA는 target domain 라벨이 없는 상황에서도 일부 클래스에서 매우 강한 분류 성능을 낼 수 있었고, 전체적으로도 50% 이상의 정확도를 확보했다. 다만 제공된 텍스트에는 confusion matrix, per-class precision/recall, IoU, F1-score, ablation study, qualitative visualization의 상세 내용이 포함되어 있지 않다. 따라서 성능의 안정성, 오류 양상, module별 기여도까지 깊게 평가하려면 논문 본문 전체가 필요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 실제 현장의 제약을 정면으로 다뤘다는 점이다. 많은 연구가 충분한 라벨이 있다고 가정하거나 단일 데이터셋 안에서 성능만 비교하는 데 비해, 이 논문은 labeled data scarcity와 domain shift라는 두 가지 현실 문제를 동시에 설정한다. 특히 source domain으로 F3, target domain으로 Penobscot을 사용한 것은 데이터 분포 차이가 있는 실제 seismic survey 간 전이를 염두에 둔 점에서 응용성이 높다.

두 번째 강점은 방법론의 구성 자체가 실용적이라는 점이다. 완전히 새로운 이론적 adaptation framework를 제안하기보다, segmentation 성능을 높이는 network design과 CORAL 기반 alignment를 결합해 비교적 해석 가능한 구조를 만든다. 이 때문에 현업 연구자 입장에서는 “어떤 feature extractor를 쓰고, 어떤 adaptation loss를 결합할지”라는 식으로 재현하거나 확장하기 쉬울 가능성이 있다. 또한 minority classes 성능을 별도로 언급했다는 점은 class imbalance 문제를 단순히 숨기지 않았다는 점에서 긍정적이다.

세 번째 강점은 결과 해석의 방향성이다. EAN은 supervised segmentation에서 강한 baseline 역할을 하고, EAN-DDA는 unlabeled target 환경에서의 가능성을 보인다. 즉, 이 논문은 한 모델의 성능 자랑에 그치지 않고, supervised-to-unsupervised transfer라는 더 넓은 워크플로를 제시한다.

반면 한계도 분명하다. 우선 제공된 텍스트 기준으로는 실험 대상 클래스가 세 개이며, 그것도 reflection pattern이 유사한 클래스만 선택되었다. 따라서 보다 복잡하고 다양한 facies taxonomy로 일반화될 수 있는지는 아직 알 수 없다. 또한 EAN-DDA의 결과에서 특정 클래스는 99%에 가까운 정확도를 보였지만 overall accuracy는 50% 이상 수준이므로, 모든 클래스에 대해 균형 잡힌 적응이 이뤄졌다고 보기는 어렵다. 이는 domain adaptation이 일부 클래스에는 잘 작동하지만 전체 클래스 구조를 안정적으로 맞추는 데는 한계가 있음을 시사한다.

또 다른 한계는 논문 요약 정보만으로는 method의 핵심 세부가 충분히 드러나지 않는다는 점이다. 예를 들어 transposed residual unit의 정확한 구조, CORAL이 어느 feature level에서 적용되는지, segmentation과 classification 사이 전환이 어떻게 구현되는지, class imbalance를 직접 완화하는 별도 전략이 있는지 등은 확인되지 않는다. 이는 본 논문의 한계라기보다 현재 제공된 텍스트의 한계이지만, 보고서 독자 입장에서는 중요한 불확실성이다.

비판적으로 보면, 이 논문은 target label이 없는 환경에서 실용적 가능성을 보여준다는 점에서 의미가 있지만, overall accuracy가 아주 높다고 보기 어려우므로 실제 deployment 단계에서는 추가적인 semi-supervised labeling, self-training, pseudo-label refinement, 또는 더 강한 domain-invariant representation 학습이 필요할 가능성이 높다. 즉, 이 연구는 매우 유용한 출발점이지만, 곧바로 범용 seismic interpretation 시스템으로 연결되었다고 평가하기는 어렵다.

## 6. 결론

이 논문은 seismic facies analysis를 위해 두 가지 기여를 한다. 첫째, seismic image semantic segmentation을 위한 EarthAdaptNet(EAN)을 제안하고, decoder에서 transposed residual unit을 사용해 pixel-level accuracy 84% 이상과 minority class accuracy 약 70%를 달성했다. 둘째, EAN에 CORAL 기반 domain adaptation을 결합한 EAN-DDA를 통해, source domain과 target domain 사이 분포 차이가 존재하고 target label이 없는 환경에서도 seismic reflection classification이 가능함을 보였다. 특히 Penobscot의 특정 클래스에서는 약 99% 수준의 class accuracy를 얻었고, overall accuracy도 50% 이상을 기록했다.

이 연구의 실제적 의미는 매우 크다. 탐사 및 해석 현장에서는 새로운 survey마다 충분한 라벨을 다시 만드는 것이 어렵기 때문에, 기존 지역에서 학습한 모델을 다른 지역으로 옮겨 쓰는 문제가 핵심이다. 이 논문은 바로 그 문제에 대해 deep domain adaptation이라는 설득력 있는 해법을 제시한다. 아직 전체 정확도와 일반화 범위 측면에서 개선 여지는 남아 있지만, labeled seismic data scarcity와 cross-survey transfer 문제를 함께 다룬다는 점에서 향후 연구의 기반이 될 가능성이 높다. 특히 이후 연구에서는 EAN-DDA를 더 강한 adaptation objective, 더 다양한 seismic facies class, 그리고 더 정교한 uncertainty-aware interpretation framework와 결합하는 방향으로 발전시킬 수 있을 것이다.

마지막으로, 이 보고서는 사용자가 제공한 arXiv 페이지 추출 텍스트를 바탕으로 작성되었다. 따라서 논문 본문 전체에 포함될 수 있는 상세 방정식, 정확한 network diagram, 세부 hyperparameter, 모든 baseline 비교표, ablation 실험 등은 확인 가능한 범위 밖에 있으며, 위 내용에서는 그러한 부분을 추측하지 않았다.
