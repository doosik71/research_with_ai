# Domain Adaptation for Medical Image Analysis: A Survey

* **저자**: Hao Guan, Mingxia Liu
* **발표연도**: 2021
* **arXiv**: [https://arxiv.org/abs/2102.09508](https://arxiv.org/abs/2102.09508)

## 1. 논문 개요

이 논문은 의료영상 분석에서의 domain adaptation(DA) 연구를 체계적으로 정리한 survey이다. 논문의 출발점은 의료영상 기반 machine learning 모델이 실제 임상 환경에서 매우 자주 겪는 **domain shift** 문제다. 즉, 학습에 사용한 source/reference domain과 실제 적용 대상인 target domain의 데이터 분포가 다르면, 모델 성능이 크게 저하될 수 있다는 점을 핵심 문제로 본다.

의료영상에서는 이런 분포 차이가 특히 심각하다. 같은 MRI라도 병원마다 scanner vendor가 다를 수 있고, 촬영 프로토콜이나 해상도, intensity distribution, 환자 집단의 구성도 달라진다. CT, MRI, PET처럼 modality 자체가 다를 수도 있다. 논문은 이러한 이질성이 자연영상보다 의료영상에서 더 큰 문제라고 설명한다. 자연영상 분야는 ImageNet 같은 대규모 라벨 데이터가 있지만, 의료영상은 라벨링이 비싸고 시간이 많이 들며 전문가 참여가 필수이기 때문이다. 따라서 단순히 target domain에서 많은 라벨을 추가 확보해 재학습하는 방식은 현실적으로 어렵다.

이 논문의 목표는 크게 세 가지다. 첫째, 의료영상에서 domain adaptation이 왜 중요한지를 배경부터 설명한다. 둘째, 기존 방법들을 체계적으로 정리한다. 특히 방법을 **shallow model 기반 DA**와 **deep model 기반 DA**로 나누고, 각 범주 안에서 다시 **supervised**, **semi-supervised**, **unsupervised** 방식으로 세분화한다. 셋째, 실제 연구에 사용된 benchmark dataset과 향후 과제를 함께 정리하여, 이 분야에 입문하거나 연구 방향을 잡으려는 사람에게 전체 지형도를 제공한다.

중요한 점은 이 논문이 새로운 단일 알고리즘을 제안하는 연구 논문이 아니라는 것이다. 대신 의료영상 DA 연구를 분류하고, 어떤 문제 설정들이 존재하며, 각 설정에서 어떤 전략이 쓰였는지를 폭넓게 요약한다. 따라서 이 논문의 기여는 특정 성능 개선 수치보다도, **분야의 구조화와 문제 설정의 명료화**에 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 의료영상의 domain adaptation을 단순히 “source에서 target으로 옮긴다”는 수준이 아니라, **여러 축으로 구분되는 문제 공간**으로 체계화하는 데 있다. 논문은 DA를 먼저 transfer learning의 한 특수한 경우로 정리한다. transfer learning에서는 domain이나 task 둘 중 하나 혹은 둘 다 달라질 수 있지만, domain adaptation에서는 **task는 동일하고 분포만 다르다**는 점을 명확히 한다. 예를 들어 source와 target 모두 뇌병변 segmentation을 수행하지만, 병원이나 scanner가 달라 데이터 분포가 다른 상황이 이에 해당한다.

논문이 제시하는 가장 중요한 정리 방식은 다음과 같다. 첫 번째 축은 **model type**으로, shallow DA와 deep DA를 나눈다. shallow DA는 사람이 설계한 feature와 전통적인 machine learning 모델 위에서 분포 차이를 보정하는 방식이고, deep DA는 CNN 같은 심층 신경망이 feature learning과 task learning을 함께 수행하면서 domain gap까지 줄이는 방식이다.

두 번째 축은 **label availability**다. target domain에 일부 라벨이 있으면 supervised DA, 적은 라벨과 많은 unlabeled 데이터가 함께 있으면 semi-supervised DA, target 라벨이 전혀 없으면 unsupervised DA로 구분한다. 의료영상의 현실을 고려하면, 논문은 특히 unsupervised DA가 매우 중요하다고 본다.

세 번째 축은 **modality difference**다. source와 target이 같은 modality면 single-modality DA, MRI와 CT처럼 modality가 다르면 cross-modality DA다. 네 번째는 **number of sources**로, single-source와 multi-source DA를 구분한다. 다섯 번째는 **adaptation step**으로, source와 target이 가까우면 one-step DA, 차이가 매우 크면 intermediate domain을 끼우는 multi-step DA를 고려한다.

즉, 이 논문의 핵심은 “의료영상 DA 연구를 shallow vs deep라는 큰 틀로 묶되, 실제 문제는 label, modality, source 수, adaptation 경로 등에 따라 달라진다”는 점을 분명히 한 데 있다. 기존 survey들이 주로 자연영상 중심이거나, 의료영상 전체 transfer learning을 넓게 다뤘다면, 이 논문은 **의료영상에서의 domain adaptation만을 독립 주제로 묶어 구조화했다는 점**에서 차별성이 있다.

## 3. 상세 방법 설명

이 논문은 survey이므로 하나의 통합 모델 아키텍처를 제시하지 않는다. 대신 다양한 방법들을 묶는 공통 원리와 대표 전략들을 설명한다. 따라서 “상세 방법 설명”은 개별 기법의 세부 구현보다, 논문이 정리한 **방법론적 축과 대표적인 학습 전략**을 이해하는 것이 핵심이다.

### 3.1 Domain adaptation의 수학적 문제 설정

논문은 DA를 다음처럼 정의한다. feature space와 label space를 각각 $\mathcal{X}$와 $\mathcal{Y}$라고 할 때, source domain $S$와 target domain $T$는 같은 $\mathcal{X} \times \mathcal{Y}$ 위에 정의되지만, 데이터 분포는 서로 다르다. 즉 source와 target의 분포를 각각 $P_s$, $P_t$라고 두면 일반적으로 $P_s \neq P_t$이다.

source domain에는 라벨이 있는 샘플 집합 $\mathcal{D}_S={(\mathbf{x}_i^S, y_i^S)}_{i=1}^{n_s}$가 있고, target domain에는 $\mathcal{D}_T={(\mathbf{x}_j^T)}_{j=1}^{n_t}$ 형태의 샘플이 존재한다. target에는 라벨이 있을 수도 있고 없을 수도 있다. domain adaptation의 목적은 source에서 배운 지식을 target에 옮겨, **같은 task를 target domain에서 잘 수행하도록 하는 것**이다.

이 식이 중요하게 말해 주는 것은, DA의 본질이 task 변경이 아니라 **분포 불일치 보정**이라는 점이다. 따라서 많은 방법들은 $P_s$와 $P_t$의 차이를 줄이는 방향으로 설계된다.

### 3.2 Shallow DA의 두 가지 핵심 전략

논문은 shallow DA에서 자주 쓰이는 전략을 크게 두 가지로 정리한다.

첫 번째는 **instance weighting**이다. 이 전략에서는 source domain의 모든 샘플을 동일하게 쓰지 않고, target과 더 유사한 source 샘플에 더 큰 weight를 준다. 직관적으로는 “source 전체를 믿지 말고 target에 가까운 source만 더 신뢰하자”는 접근이다. 이후 re-weighted source data로 classifier나 regressor를 학습한다. 예를 들어 logistic regression이나 random forest 같은 전통적인 모델이 여기에 사용된다.

두 번째는 **feature transformation**이다. 여기서는 source와 target을 원래 feature space에서 직접 맞추는 대신, 둘 다 새로운 shared latent space로 사상한다. 이 공통 표현 공간에서는 domain gap이 더 작아지도록 설계한다. low-rank representation, PCA 기반 subspace alignment 같은 기법이 여기에 속한다. 이 전략의 목적은 “같은 질병 패턴은 domain이 달라도 비슷한 좌표계에서 표현되도록 만들자”는 것이다.

### 3.3 Supervised shallow DA

supervised shallow DA에서는 target에 소량의 라벨이 있다고 가정한다. 논문이 소개한 예들을 보면, 이런 라벨은 두 가지 방식으로 활용된다. 하나는 target distribution을 좀 더 정확히 추정해 source 샘플 weight를 조정하는 것이고, 다른 하나는 공통 feature space를 만든 뒤 target label로 classifier를 미세 조정하는 것이다.

예를 들어 Alzheimer’s disease 분류 연구에서는 labeled target samples를 이용해 target 분포를 추정하고, source 샘플이 target에서 나타날 확률에 비례하도록 re-weighting한 뒤 logistic regression classifier를 학습한다. 또 다른 연구에서는 source와 target을 shared latent space로 보낸 뒤 boosting classifier를 학습한다. 여기서 핵심은 적은 target label이라도 adaptation 과정의 기준점 역할을 한다는 점이다.

### 3.4 Semi-supervised shallow DA

semi-supervised shallow DA에서는 일부 labeled target과 다수의 unlabeled target을 함께 사용한다. 논문은 ultrasound image classification의 2-step framework를 예로 든다. 먼저 PCA로 source와 target을 common latent space로 옮겨 global domain gap을 줄이고, 이후 transformed source domain에서 학습한 random forest classifier를 소량의 labeled target data로 fine-tune한다. 이 접근은 shallow 모델이라도 “공통 공간 정렬 + 소량의 target supervision”의 조합이 효과적일 수 있음을 보여준다.

### 3.5 Unsupervised shallow DA

unsupervised shallow DA는 target label이 전혀 없는 경우다. 이 경우 instance weighting이나 subspace alignment가 더 직접적으로 쓰인다. 예를 들어 CT 기반 폐질환 분류에서는 Gaussian texture feature를 뽑은 뒤, target과 유사한 source 샘플에 높은 weight를 부여해 weighted logistic classifier를 학습한다. 또 다른 연구에서는 fMRI feature를 추출하고, source/target subspace를 정렬한 뒤 그 공유 공간에서 discriminant analysis classifier를 훈련한다.

이 범주의 한계는 사람이 설계한 feature 품질에 매우 의존한다는 점이다. 그러나 의료영상에서 데이터가 적고, 전통적인 feature engineering이 여전히 의미 있는 상황에서는 실용적인 접근이 될 수 있다.

### 3.6 Multi-source shallow DA

논문은 multi-source DA도 별도로 다룬다. 예를 들어 여러 imaging center의 rs-fMRI 데이터를 source domain들로 보고, 각 source를 target 쪽으로 변환한 뒤 분류기를 학습하는 방식이 소개된다. 여기서는 low-rank regularization, graph embedding, multi-view sparse representation 등이 사용된다. 핵심은 source가 여러 개면 단순히 데이터를 합치면 끝나는 것이 아니라, source들 사이에도 이질성이 있어 이를 함께 다뤄야 한다는 점이다.

### 3.7 Deep DA의 큰 흐름

deep DA에서는 CNN이 feature extractor와 task model 역할을 동시에 수행한다. 논문은 크게 몇 가지 흐름을 정리한다.

첫 번째는 **CNN feature + shallow DA**의 조합이다. 즉 CNN으로 feature를 먼저 추출하고, 그 위에 TCA, CORAL, BDA 같은 전통적인 adaptation 기법을 적용하는 방식이다.

두 번째는 **fine-tuning 기반 transfer**다. source 혹은 ImageNet에서 pre-trained된 CNN을 target data로 fine-tune한다. supervised DA에서 매우 많이 쓰인다. 특히 MRI처럼 3D 구조가 중요한 경우, 2D CNN 대신 3D CNN이나 3D U-Net을 pre-train 후 target에서 fine-tune하는 전략도 소개된다.

세 번째는 **adversarial feature alignment**다. 이것이 unsupervised deep DA의 대표적인 축이다. 기본 구조는 DANN과 유사하다. feature extractor 혹은 segmentation network가 task를 잘 수행하는 feature를 만들도록 학습되는 동시에, domain discriminator는 그 feature가 source인지 target인지 구분하려 한다. 반대로 feature extractor는 discriminator를 속이도록 학습된다. 결과적으로 source와 target을 잘 구분하지 못하는, 즉 **domain-invariant feature**가 형성된다.

이를 개념적으로 쓰면, 전체 목적은 task loss와 domain confusion loss 사이의 균형으로 이해할 수 있다. 예를 들면
$$
\mathcal{L} = \mathcal{L}_{task} - \lambda \mathcal{L}_{domain}
$$
처럼 볼 수 있다. 여기서 $\mathcal{L}_{task}$는 segmentation이나 classification의 성능을 높이는 손실이고, $\mathcal{L}_{domain}$은 discriminator가 domain을 구분하는 손실이다. feature extractor 입장에서는 domain 구분이 어려워지게 만들어야 하므로 adversarial한 관계가 생긴다. 논문 원문은 개별 모델들의 손실식을 모두 상세히 전개하지는 않지만, DANN류 방법의 핵심 원리는 이와 같다.

### 3.8 Feature alignment

feature alignment는 source와 target의 feature distribution을 직접 맞추는 방향이다. DANN 계열이 대표적이고, MMD, CORAL, CMD 같은 분포 정렬 기준도 사용된다. segmentation 문제에서는 U-Net 계열 segmentor와 domain discriminator를 함께 훈련하는 경우가 많다. 어떤 연구는 여러 feature layer에 discriminator를 붙여 low-level과 high-level 모두의 domain gap을 줄이려 했고, 어떤 연구는 cross-modality 문제에서 저수준 특성 차이가 더 크다고 보고 low-level layer만 adaptation하기도 했다.

논문이 강조하는 점은 의료영상에서 domain gap이 항상 같은 위치에서 생기지 않는다는 것이다. scanner 차이는 low-level intensity나 texture 차이에 클 수 있고, disease phenotype 차이는 더 high-level semantic feature에도 영향을 줄 수 있다. 따라서 어떤 layer를 맞출지 자체가 설계 선택이 된다.

### 3.9 Image alignment

이 범주는 GAN, 특히 CycleGAN을 사용해 **이미지 자체를 target-like 혹은 source-like하게 변환**하는 방법이다. 예를 들어 source MRI를 target CT 스타일처럼 보이게 바꾸고, 다시 원래 MRI로 복원하는 cycle consistency를 사용해 paired sample 없이도 image-to-image translation을 수행한다.

cycle consistency의 핵심은 한 domain에서 다른 domain으로 갔다가 다시 돌아왔을 때 원래 입력을 유지해야 한다는 것이다. 개념적으로는
$$
\mathcal{L}_{cycle} = | G_{T \rightarrow S}(G_{S \rightarrow T}(x_S)) - x_S |
$$
같은 형태로 이해할 수 있다. 즉 source image $x_S$를 target style로 보낸 후 다시 source style로 되돌렸을 때 차이를 최소화한다. 논문은 이 아이디어가 의료영상에서 cross-modality adaptation, denoising, synthetic-to-real augmentation 등에 널리 쓰인다고 정리한다.

### 3.10 Image + Feature alignment

이 논문이 소개하는 더 강력한 흐름 중 하나는 image alignment와 feature alignment를 결합하는 방식이다. 먼저 CycleGAN으로 labeled source를 target-like image로 변환하고, 그 뒤 이 합성된 target-like image와 실제 target image를 CNN에 넣어 feature-level adversarial alignment를 수행한다. 이 방식은 “겉모습”과 “내부 표현”의 간극을 동시에 줄이려는 전략이다. 특히 CT-MRI 같은 cross-modality segmentation에서 유용하다고 정리된다.

### 3.11 Disentangled representation

이 접근에서는 이미지를 **domain-invariant content space**와 **domain-specific style space**로 분리해서 표현한다. 예를 들어 liver segmentation에서는 CT와 MRI가 서로 외형적 스타일은 다르지만, 해부학적 구조 정보는 어느 정도 공유될 수 있다고 본다. 따라서 공통 content representation에서 segmentation을 수행하면 modality 차이의 영향을 줄일 수 있다는 논리다. 이 방법은 단순 정렬보다 표현을 더 해석적으로 나눈다는 점에서 의미가 있다.

### 3.12 Self-ensemble, soft labels, feature learning

논문은 추가적으로 self-ensemble 기반 DA도 소개한다. 여기서는 student network와 teacher network를 두고, teacher를 student weight의 exponential moving average로 갱신한다. source labeled data로 task loss를 학습하고, unlabeled target data에 대해서는 teacher와 student의 예측이 일관되도록 consistency loss를 둔다. 이 방식은 target label 없이도 안정적인 pseudo-supervision을 제공하려는 아이디어다.

soft labels 접근은 target domain에서 명시적 라벨이 없더라도, source와 target 간 구조적 유사성을 이용해 heatmap 형태의 soft supervision을 만들고 이를 segmentation 학습에 활용한다. 또 auto-encoder를 사용해 medical-image-specific feature를 추가로 학습하는 feature learning 기반 방법도 정리된다.

정리하면, 이 survey가 설명하는 deep DA의 핵심은 결국 세 가지 축으로 볼 수 있다. 첫째, **어디를 맞출 것인가**: image level인지, feature level인지, 둘 다인지. 둘째, **어떻게 맞출 것인가**: adversarial learning인지, discrepancy minimization인지, self-ensembling인지. 셋째, **무엇을 활용할 것인가**: target label이 있는지, 없는지, modality가 같은지, source가 여러 개인지다.

## 4. 실험 및 결과

이 논문은 survey이므로 하나의 통일된 실험 프로토콜을 제시하지 않는다. 대신 다양한 task와 dataset에서 어떤 DA 방식이 쓰였는지를 폭넓게 요약한다. 따라서 여기서의 핵심은 “어떤 의료영상 문제들이 DA의 대표 응용 분야인가”와 “전반적으로 어떤 패턴의 성과가 보고되었는가”를 이해하는 것이다.

논문이 정리한 benchmark dataset은 매우 다양하다. 뇌 영상 분야에서는 ADNI, AIBL, CADDementia, IXI, ABIDE, BraTS, ISBI2015, WMH Challenge, HCP 등이 소개된다. 폐 영역에서는 NIH ChestXray14, DLCST, COPDGene이 있고, 심장 영역에서는 MM-WHS, NIH PLCO, NIH Chest가 있다. 안과 분야는 DRIVE, STARE, SINA, 유방 분야는 CBIS-DDSM, InBreast, CAMELYON, MIAS, 피부 분야는 MoleMap, HAM10000, ISIC, 복부 영역은 PROMISE12, BWH, LiTS 등이 포함된다. 즉 DA는 특정 장기나 task에 국한되지 않고, classification, segmentation, detection, registration 등 매우 넓은 범위에 적용되고 있다.

### 4.1 Shallow DA 관련 결과 해석

shallow DA 사례들에서는 대체로 **instance weighting**이나 **feature transformation**을 통해 source-only 또는 target-only 학습보다 더 좋은 결과를 얻었다고 보고된다. 예를 들어 AD classification에서는 ADNI, AIBL, CADDementia 간 adaptation이 도움이 되었고, brain tumor segmentation, brain tissue segmentation, skull stripping, white matter lesion segmentation 등에서도 re-weighting 기반 접근의 효과가 제시되었다. 다만 survey 성격상 논문은 개별 연구의 수치를 일괄 비교하지 않고, 각 연구가 “도메인 차이를 고려하지 않은 방식보다 낫다”는 방향의 결론을 요약한다.

### 4.2 Supervised deep DA 관련 결과 해석

supervised deep DA에서는 fine-tuning의 효과가 매우 자주 언급된다. 특히 source 혹은 ImageNet pre-training 후 target 소량 라벨로 fine-tune하면, scratch training보다 유리한 경우가 많다. 뇌 병변 segmentation, breast cancer classification, Alzheimer’s disease classification, brain tumor classification 등에서 이런 패턴이 반복된다. 또 3D medical images에서는 2D CNN보다 3D CNN 또는 3D U-Net backbone을 사전학습 후 fine-tune하는 전략이 유효하다고 정리된다.

논문이 특히 강조하는 사례 중 하나는 multi-step adaptation이다. 피부질환 분류에서 ImageNet에서 바로 target으로 가지 않고, 먼저 상대적으로 큰 의료영상 데이터셋을 intermediate domain으로 사용한 뒤 최종 target에 적응시키는 방식이 direct transfer보다 더 좋았다고 소개한다. 이는 source와 target 간 거리가 큰 경우, 중간 domain이 bridge 역할을 할 수 있음을 보여준다.

### 4.3 Unsupervised deep DA의 실험 경향

unsupervised deep DA는 최근 가장 활발한 분야로 소개된다. target label 없이도 성능을 개선할 수 있다는 점에서 실용성이 크기 때문이다. brain lesion segmentation, retina vessel segmentation, lung texture classification, knee MRI segmentation, mammogram mass detection, OCT lesion detection, cardiac segmentation, liver segmentation 등에서 adversarial feature alignment나 image translation 기반 기법이 적용되었다.

이들 연구의 공통된 결과 메시지는 다음과 같다. 첫째, **source-only 모델은 domain shift에 매우 취약하다**. 둘째, adversarial learning, MMD/CORAL/CMD, CycleGAN, self-ensembling 같은 adaptation 기법을 넣으면 target 성능이 개선되는 경우가 많다. 셋째, 같은 unsupervised DA라도 task 특성에 따라 효과적인 alignment 수준이 다르다. 예를 들어 segmentation에서는 feature alignment와 shape/size prior, edge guidance, boundary weighting 같은 task-specific inductive bias를 결합할 때 더 좋은 결과가 보고된다.

### 4.4 Cross-modality adaptation의 의미

논문이 소개하는 실험 중 특히 중요해 보이는 축은 CT와 MRI 사이의 cross-modality adaptation이다. 이 경우 domain gap은 단순한 scanner 차이보다 훨씬 크다. Chen 등, Dou 등, Yang 등의 연구는 CycleGAN 기반 image translation과 adversarial feature alignment, 혹은 disentangled representation을 결합해 cardiac segmentation이나 liver segmentation 성능을 개선했다고 보고한다. 이는 의료영상 DA가 단순히 동일 modality 내 site adaptation을 넘어서, modality 간 구조적 차이도 다룰 수 있음을 보여준다.

### 4.5 데이터셋 표의 의미

논문의 Table I은 매우 중요하다. 이 표는 단순 데이터셋 목록이 아니라, 어떤 장기/질환/작업에 어떤 DA 기법이 적용되었는지를 한눈에 보여준다. 예를 들어 brain 분야는 MRI, fMRI 중심의 classification/segmentation 연구가 많고, heart 분야는 CT-MRI cross-modality segmentation이 중요한 축이며, breast와 histology는 scanner/site 차이 및 stain/style 차이가 domain shift의 핵심 원인임을 시사한다. 즉, Table I은 “어떤 task에 어떤 유형의 DA가 잘 맞는가”를 파악하는 지도로 볼 수 있다.

다만 이 survey는 메타분석 논문이 아니므로, 서로 다른 논문들의 수치를 동일 기준으로 정량 비교해 “무조건 어떤 방법이 최고다”라고 결론내리지는 않는다. 데이터셋, task, 라벨 설정, 평가 지표가 서로 다르기 때문이다. 이 점은 이 논문의 장점이자 한계이기도 하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 의료영상 domain adaptation 연구를 매우 체계적으로 정리했다는 점이다. 단순히 관련 논문을 나열하는 것이 아니라, shallow/deep, supervised/semi-supervised/unsupervised, single-modality/cross-modality, single-source/multi-source, one-step/multi-step 같은 분류 체계를 제시해 독자가 전체 분야를 구조적으로 이해할 수 있게 한다. 특히 자연영상 중심 survey와 달리 의료영상에서 왜 domain shift가 더 심각한지, 그리고 어떤 데이터셋과 응용 과제가 대표적인지를 구체적으로 연결해 설명한 점이 유용하다.

또 다른 강점은 단순한 알고리즘 리뷰에 그치지 않고, benchmark datasets를 장기 및 응용별로 정리했다는 점이다. 이는 실제 연구자가 새로운 DA 연구를 기획할 때 어떤 데이터셋과 task를 참고할 수 있는지 빠르게 파악하게 해 준다. 더 나아가 논문은 3D/4D volumetric representation, limited training data, inter-modality heterogeneity, multi-source/multi-target adaptation 같은 과제를 명시하여, 단순 현황 정리를 넘어 향후 연구 방향까지 제안한다.

하지만 한계도 분명하다. 첫째, 이 논문은 survey이므로 개별 방법의 세부 수식, 구현 디테일, 손실 함수 정의를 깊게 다루지 않는다. 따라서 특정 모델을 실제 재현하고자 하는 독자에게는 원 논문들을 추가로 읽어야 할 필요가 크다. 둘째, 다양한 논문을 폭넓게 다루는 대신, 서로 다른 방법들 간의 공정한 정량 비교나 standardized benchmark 비교는 제공하지 않는다. 이는 분야 특성상 쉽지 않은 일이지만, 실무자 입장에서는 “어떤 방법을 먼저 시도해야 하는가”에 대한 직접적 지침은 제한적이다.

셋째, survey 시점의 한계도 있다. 논문은 2020년 전후 연구를 중심으로 정리하고 있으며, self-supervised learning, test-time adaptation, foundation model, vision transformer 기반 DA, diffusion-based translation 같은 이후 흐름은 포함되지 않는다. 물론 이는 논문 당시 시점의 자연스러운 한계이며, 논문 자체가 이를 숨기지는 않는다.

비판적으로 보면, 이 논문은 DA를 광범위하게 포괄하면서도 각 범주의 경계가 실제로는 겹친다는 점을 잘 인정하고 있다. 그러나 그만큼 독자가 읽을 때 “분류는 명확하지만, 실제 연구는 혼합형”이라는 사실도 함께 기억해야 한다. 예를 들어 image alignment와 feature alignment를 결합한 방식은 단일 카테고리로 깔끔히 나누기 어렵다. 그럼에도 이 survey의 분류는 이해를 위한 틀로서는 충분히 강력하다.

## 6. 결론

이 논문은 의료영상 분석에서 domain adaptation이 왜 핵심 문제인지 설득력 있게 정리하고, 기존 연구를 큰 지도처럼 구조화한 survey이다. 핵심 기여는 다음과 같이 요약할 수 있다. 첫째, 의료영상 DA의 문제 배경과 transfer learning 대비 정의를 명확히 했다. 둘째, 기존 방법을 shallow DA와 deep DA로 나누고, 각 범주를 supervised, semi-supervised, unsupervised 관점에서 세분화했다. 셋째, image alignment, feature alignment, disentangled representation, self-ensemble, multi-source/multi-target adaptation 등 다양한 흐름을 의료영상 응용과 연결해 설명했다. 넷째, benchmark dataset과 향후 연구 과제를 함께 정리해 후속 연구의 출발점을 제공했다.

실제 적용 측면에서 이 논문이 중요한 이유는, 의료 AI가 병원 간 일반화 실패라는 현실적 문제를 극복하는 데 DA가 핵심 도구가 될 수 있음을 보여주기 때문이다. 같은 질환을 다루더라도 촬영 장비와 프로토콜이 바뀌면 성능이 급락하는 문제는 임상 배치에서 치명적이다. 따라서 DA는 단지 학술적 흥미가 아니라, 의료 AI의 **현장 배치 가능성**을 높이는 핵심 기술이다.

향후 연구 측면에서는 특히 3D/4D 의료영상에 특화된 DA, 라벨 없는 target 환경을 위한 unsupervised DA 및 domain generalization, cross-modality adaptation, multi-source/multi-target adaptation이 중요하다는 논문의 전망이 설득력 있다. 다시 말해, 이 survey는 “의료영상 DA가 어디까지 왔는가”를 정리할 뿐 아니라, “앞으로 어디를 더 파야 하는가”까지 제시하는 지침서 역할을 한다.
