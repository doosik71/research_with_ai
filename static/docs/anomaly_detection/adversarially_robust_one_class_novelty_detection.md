# Adversarially Robust One-class Novelty Detection

## 논문 메타데이터

- 제목: Adversarially Robust One-class Novelty Detection
- 저자: Shao-Yuan Lo, Poojan Oza, Vishal M. Patel
- 발표 연도: 2021
- arXiv ID: 2108.11168
- arXiv URL: https://arxiv.org/abs/2108.11168
- DOI: 확인 불가

## 연구 배경 및 문제 정의

원클래스 novelty detection은 정상(known) 클래스만으로 학습해, 새로운(unknown) 클래스를 탐지하는 문제다. 최근에는 오토인코더 기반 방법이 널리 사용되지만, 딥러닝 모델은 적대적 공격에 취약하다. 본 논문은 기존 원클래스 novelty detector의 공격 취약성을 체계적으로 보여주고, 분류 문제에서 쓰이던 방어 기법이 원클래스 환경에서는 효과가 제한적임을 지적한다. 이를 해결하기 위해 원클래스 특성에 맞춘 잠재 공간 조작 기반 방어 기법 PrincipaLS를 제안한다.

## 핵심 기여

- 원클래스 novelty detector에 대한 적대적 공격 취약성을 체계적으로 분석.
- 분류용 방어 기법들이 원클래스 novelty detection에서 충분히 효과적이지 않음을 실험적으로 제시.
- 잠재 공간을 PCA 기반으로 정제하는 PrincipaLS(Principal Latent Space) 방어 기법 제안.
- 8가지 공격, 5개 데이터셋, 7개 novelty detector에 대한 광범위 실험 벤치마크 제공.

## 방법론 요약

### 1. 공격 설정

- 오토인코더 기반 novelty detector의 재구성 오류를 조작하는 형태의 공격을 설계.
- 정상 데이터는 재구성 오류를 증가시키고, 이상 데이터는 오류를 감소시키는 방향으로 적대적 예제를 생성.
- PGD, FGSM, MI-FGSM, MultAdv, Adversarial Framing 등 다양한 공격 기법을 원클래스 환경에 맞춰 수정.

### 2. PrincipaLS 방어 기법

- 핵심 아이디어: 원클래스 novelty detection은 정상 클래스의 표현만 유지하면 되므로, 잠재 공간을 크게 조작해도 성능이 유지됨.
- 방법: 잠재 공간에 대해 2단계 PCA를 적용.
  - Vector-PCA: 채널 차원에서 주요 성분을 추출하여 잠재 벡터를 정제.
  - Spatial-PCA: 공간 차원에서 추가 정제를 수행.
- 두 단계 PCA를 순차적으로 수행한 뒤 역변환하여 principal latent space를 구성하고, 이를 디코더에 입력.

### 3. 학습 방식

- principal components는 EMA 기반으로 incremental하게 학습.
- adversarial training과 결합 가능하며, PrincipaLS는 가벼운 모듈로 다양한 AE 기반 novelty detector에 부착 가능.

## 실험 설정과 결과

- 데이터셋: MNIST, Fashion-MNIST, CIFAR-10, MVTec-AD, UCSD Ped2 등.
- 모델: AE, VAE, AAE, ALOCC, GPND, ARAE, OC-GAN 등 7종.
- 공격: FGSM, PGD, MI-FGSM, MultAdv, AF 등 8종.
- 평가 지표: AUROC, FPR@95%TPR 등.

주요 결과는 다음과 같다.

- 기존 모델은 적대적 공격 하에서 AUROC가 급격히 붕괴.
- PrincipaLS는 다양한 공격 및 데이터셋에서 기존 방어 기법 대비 높은 AUROC를 유지.
- adversarial training(PGD-AT), feature denoising(FD) 등과 비교해도 robust 성능이 일관되게 우수.
- clean data 성능 저하 없이 robustness 향상 가능함을 보고.

## 한계 및 향후 연구 가능성

- 원클래스 novelty detection에 특화된 설계이므로 다중 클래스 환경에서는 직접 적용이 제한될 수 있음.
- PCA 기반 정제는 잠재 공간 구조에 의존적이므로, 비정형 구조나 다른 표현 학습 기법과의 결합 필요.
- adversarial robustness와 해석 가능성의 결합, 혹은 다른 신뢰성 요소(공정성, 프라이버시)와의 통합 연구가 필요.

## 실무적 또는 연구적 인사이트

- 원클래스 탐지 문제는 잠재 공간 조작에 대한 여지가 커, 분류 문제보다 방어 설계가 유리한 측면이 있다.
- PrincipaLS는 모델 구조를 크게 바꾸지 않고도 robustness를 높일 수 있는 실용적인 모듈이다.
- 적대적 공격을 고려하지 않은 novelty detector는 실제 보안/안전 환경에서 취약할 수 있으므로, 평가 시 공격 시나리오를 반드시 포함해야 한다.
