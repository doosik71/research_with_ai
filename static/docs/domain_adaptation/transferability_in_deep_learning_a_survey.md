# Transferability in Deep Learning: A Survey

- **저자**: Junguang Jiang, Yang Shu, Jianmin Wang, Mingsheng Long (Tsinghua University)
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2201.05867

## 1. 논문 개요

딥러닝의 눈부신 성공은 대규모 레이블 데이터에 대한 높은 의존성이라는 근본적인 한계를 수반한다. 새로운 태스크나 도메인마다 수백만 건의 데이터를 수집하고 주석을 다는 것은 현실적으로 불가능에 가깝다. 반면, 인간은 이전 경험에서 얻은 지식을 새로운 문제에 유연하게 적용한다. 이 논문은 그러한 인간적 능력에 대응하는 딥러닝의 핵심 성질인 **전이 가능성(transferability)**을 체계적으로 다루는 서베이다.

저자들이 제기하는 연구 문제는 명확하다: 딥러닝 모델이 어떻게 지식을 획득하고, 새로운 태스크와 도메인에 그 지식을 효과적으로 재사용할 수 있는가? 이 질문은 도메인 적응, 메타러닝, 사전학습 등 기존에 분리되어 연구되어온 여러 분야를 하나의 통일된 관점으로 묶는다. 저자들은 딥러닝의 전체 생명주기(lifecycle)인 **사전학습(pre-training)**과 **적응(adaptation)** 단계를 모두 포괄하여 전이 가능성을 탐구하며, 공정한 비교를 위한 오픈소스 라이브러리 TLlib과 벤치마크를 함께 제공한다.

## 2. 핵심 아이디어

이 논문의 핵심 기여는 다양한 딥러닝 분야에 산재해 있는 연구들을 **전이 가능성**이라는 단일 개념으로 통합하는 것이다.

저자들은 전이 가능성을 다음과 같이 정의한다.

**정의 (Transferability):** 소스 도메인 $\mathcal{S}$의 태스크 $t_{\mathcal{S}}$에서 지식을 획득하고, 이를 재사용하여 분포 이동($\mathcal{S} \neq \mathcal{T}$) 또는 태스크 불일치($t_{\mathcal{S}} \neq t_{\mathcal{T}}$)가 존재하는 타깃 도메인 $\mathcal{T}$의 태스크 $t_{\mathcal{T}}$의 일반화 오류를 감소시키는 능력.

기존 접근들은 도메인 적응, 연속 학습, few-shot 학습 등 부분적인 관점에서 전이 가능성을 다루었다. 이 논문은 이를 하나의 통일된 틀 안에서 바라보며, 사전학습 단계의 **generic transferability**(광범위한 다운스트림 태스크에 범용적으로 적용 가능한 지식)와 적응 단계의 **specific transferability**(특정 태스크나 도메인에 최적화된 지식 활용)를 구분하는 시각을 제공한다.

## 3. 상세 방법 설명

### 3.1 사전학습 (Pre-Training)

사전학습은 대규모 업스트림 데이터로부터 전이 가능한 표현을 학습하는 단계다.

#### 3.1.1 모델 아키텍처와 전이 가능성

모델 아키텍처는 전이 가능성에 결정적인 영향을 미친다. ResNet의 잔차 연결은 수백~수천 층의 깊은 네트워크 학습을 가능하게 하여 모델 용량을 크게 확장했고, Batch Normalization은 학습 안정성을 높였다. 그러나 BatchNorm은 분포 의존적인 이동 평균 통계를 사용하므로 전이에 불리하다는 문제가 있으며, BiT(Big Transfer)는 이를 GroupNorm으로 대체하여 강한 전이 성능을 달성했다.

Transformer는 국소 연결 가정(local connectivity assumption)을 제거하고 self-attention을 통해 모든 위치 간 전역 의존성을 동적으로 계산한다. 이 최소한의 귀납적 편향(inductive bias)은 대규모 사전학습 데이터로부터 더 풍부하고 전이 가능한 지식을 흡수할 수 있게 한다. Vision Transformer(ViT)는 이를 이미지 도메인으로 확장했으며, Aghajanyan et al.(2021)은 사전학습이 목적 함수의 본질적 차원(intrinsic dimension)을 줄임으로써 파인튜닝 시 탐색해야 하는 가설 공간을 압축한다는 사실을 경험적으로 보였다.

#### 3.1.2 지도 사전학습 (Supervised Pre-Training)

표준 지도 사전학습은 ImageNet 분류와 같은 대규모 레이블 데이터 기반의 업스트림 태스크로 모델을 학습한 후, 피처 생성기(feature generator)만 타깃 태스크에 재사용한다(태스크 특화 헤드는 폐기). Kornblith et al.(2019)은 사전학습 태스크의 정확도와 다운스트림 성능이 높은 상관관계를 가짐을 보였다.

데이터의 양과 질이 전이 가능성에 가장 중요한 요소다. Weakly Supervised Pre-training(WSP)은 소셜 미디어 해시태그를 레이블로 사용한 수십억 이미지로 학습하고, Semi-Supervised Pre-training(SSP)은 소규모 레이블 데이터와 대규모 비레이블 데이터를 결합한다. 반면, 무조건 많은 데이터가 좋은 것은 아니며 타깃 태스크와 관련성이 높은 데이터를 선별하는 것(DAT)이 중요하다는 연구도 있다.

**메타러닝(Meta-Learning):** 표준 사전학습은 다운스트림 적응에 여전히 수백~수천 개의 레이블 샘플과 많은 gradient 업데이트를 필요로 한다. 메타러닝은 이를 극복하기 위해 메타 지식 $\phi$를 학습한다. 학습 목표는 다음의 이중 수준 최적화(bi-level optimization) 문제다:

$$\phi^{*}=\arg\max_{\phi}\sum_{i=1}^{n}\log P(\theta_{i}(\phi)|\mathcal{D}^{\text{ts}}_{i}), \quad \text{where}\ \theta_{i}(\phi)=\arg\max_{\theta}\log P(\theta|\mathcal{D}^{\text{tr}}_{i},\phi)$$

MAML(Finn et al., 2017)은 단 몇 번의 gradient 업데이트만으로 새로운 태스크에 빠르게 일반화할 수 있는 초기화 $\phi$를 명시적으로 탐색한다. 태스크 $i$의 훈련 데이터로 한 번 업데이트하면:

$$\theta_{i}=\phi-\alpha\nabla_{\phi}L(\phi,\mathcal{D}_{i}^{\text{tr}})$$

그리고 이 파인튜닝된 파라미터가 테스트 데이터에서 잘 동작하도록 $\phi$를 최적화한다:

$$\min_{\phi}\sum_{i=1}^{n}L(\phi-\alpha\nabla_{\phi}L(\phi,\mathcal{D}_{i}^{\text{tr}}),\mathcal{D}_{i}^{\text{ts}})$$

**인과 학습(Causal Learning):** 분포 외(OOD) 도메인에 대한 일반화를 위해, 분포가 바뀌어도 변하지 않는 인과 메커니즘을 학습한다. IRM(Invariant Risk Minimization)은 모든 학습 환경 $e \in \mathcal{E}^{\text{tr}}$에서 동시에 최적인 분류기 $h$가 존재하도록 표현 $\psi$를 학습하는 제약 최적화 문제를 푼다:

$$\min_{\psi, h}\sum_{e\in\mathcal{E}^{\text{tr}}}\epsilon^{e}(h\circ\psi), \quad \text{subject to } h\in\arg\min_{\bar{h}}\epsilon^{e}(\bar{h}\circ\psi),\ \forall e\in\mathcal{E}^{\text{tr}}$$

#### 3.1.3 비지도 사전학습 (Unsupervised Pre-Training)

비레이블 데이터를 활용하는 자기지도학습(self-supervised learning) 기반 사전학습으로, 레이블 비용 없이 대규모 데이터를 학습에 활용한다.

**생성 학습(Generative Learning):** 입력의 일부를 훼손하거나 마스킹한 후 원본을 복원하도록 학습한다. 인코더 $f_\theta$가 훼손된 입력 $\tilde{\mathbf{x}}$를 잠재 표현 $\mathbf{z}$로 변환하고 디코더 $g_\theta$가 원본 $\mathbf{x}$를 재구성한다.

- **자기회귀 모델(Autoregressive, GPT 계열):** Language Modeling(LM)으로, 이전 문맥 조건부로 각 토큰을 예측: $\max_\theta \sum_{t=1}^T \log P_\theta(x_t|x_{t-k},\cdots,x_{t-1})$. 단방향 컨텍스트만 활용한다는 한계가 있다.
- **자동인코딩 모델(Autoencoding, BERT 계열):** Masked Language Modeling(MLM)으로, 랜덤 마스킹 후 나머지 토큰으로 마스킹된 토큰을 예측: $\max_\theta \sum_{x\in m(\mathbf{x})}\log P_\theta(x|\mathbf{x}_{\setminus m(\mathbf{x})})$. MAE는 이미지에 동일 원리를 적용, 매우 높은 비율(~75%)의 패치를 마스킹하여 저수준 단서에 의존하지 않도록 강제한다.

**대조 학습(Contrastive Learning):** 같은 데이터에서 생성된 서로 다른 뷰(view) 간 유사도를 최대화하고, 다른 인스턴스 간 유사도를 최소화한다. Instance Discrimination(InstDisc, MoCo, SimCLR 등)의 기본 목표는:

$$\min_\psi -\log\frac{\exp(\mathbf{q}\cdot\mathbf{k}_{+}/\tau)}{\sum_{j=0}^{K}\exp(\mathbf{q}\cdot\mathbf{k}_{j}/\tau)}$$

여기서 $\tau$는 온도 하이퍼파라미터, 분자는 긍정 쌍, 분모는 1개의 긍정 샘플과 $K$개의 부정 샘플에 대한 합이다. MoCo는 모멘텀 업데이트 인코더와 키 큐를 도입해 부정 샘플의 수와 일관성을 개선했고, SimCLR은 강한 데이터 증강 조합이 핵심임을 강조했다. BYOL과 SimSiam은 부정 샘플 없이도 stop-gradient 등의 기법으로 붕괴를 방지하며 전이 가능한 표현을 학습할 수 있음을 보였다.

Zhao et al.(2021)의 분석에 따르면, 지도 사전학습은 클래스 레이블이 정의하는 의미론적 부분에 과적합될 위험이 있어 다양한 다운스트림 태스크로의 전이 시 불리할 수 있다. 반면 대조 사전학습은 객체 전체를 더 총체적으로 모델링하여 광범위한 태스크에 더 나은 전이 가능성을 보인다.

### 3.2 적응 (Adaptation)

#### 3.2.1 태스크 적응 (Task Adaptation)

사전학습된 모델 $h_{\theta^0}$을 레이블 데이터 $\widehat{\mathcal{T}}=\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^m$이 있는 타깃 태스크에 적응시키는 과정이다. Feature transfer(동결)보다 fine-tuning(전체 파라미터 업데이트)이 일반적으로 더 나은 성능을 보인다.

**Catastrophic Forgetting 방지:**
바닐라 파인튜닝은 업스트림에서 획득한 지식을 잃는 치명적 망각(catastrophic forgetting) 위험이 있다. 주요 방법들은 다음과 같다:

- **층별 차별화(Layer-wise adaptation):** Yosinski et al.(2014)이 발견했듯 초기 층은 일반적 특징, 후기 층은 태스크 특화 특징을 담으므로 층마다 다른 학습률을 적용한다(DAN: 태스크 헤드에 10배 높은 학습률).
- **도메인 적응적 튜닝(Domain Adaptive Tuning):** ULMFiT, DAPT 등은 사전학습 태스크로 타깃 도메인 데이터를 먼저 적응 학습한 후 파인튜닝하는 2단계 방식을 사용한다.
- **정규화 튜닝(Regularization Tuning):** 일반적인 정규화 최적화 목표는 $\min_\theta \sum_{i=1}^m L(h_\theta(\mathbf{x}_i),\mathbf{y}_i) + \lambda\cdot\Omega(\theta)$. EWC는 Fisher 정보 행렬 $F$를 이용해 사전학습 파라미터 $\theta^0$에서 크게 벗어나지 않도록 제약한다: $\Omega(\theta)=\sum_j \frac{1}{2}F_j\|\theta_j-\theta_j^0\|_2^2$. DELTA는 특징 맵 수준에서, LWF는 출력 예측 수준에서 정규화한다. SMART는 입력에 소 perturbation을 가했을 때 출력이 크게 변하지 않도록 smoothness를 직접 강제한다.

**Negative Transfer 방지:** 사전학습이 항상 유익한 것은 아니며, 태스크 불일치나 도메인 차이가 클 때 negative transfer가 발생한다. Negative Transfer Gap(NTG)을 $NTG = \epsilon_\mathcal{T}(h_\theta(\mathcal{U},\mathcal{T})) - \epsilon_\mathcal{T}(h_\theta(\emptyset,\mathcal{T}))$로 정의하고, 이를 줄이기 위해 BSS(작은 특이값 억제), Zoo-tuning(여러 사전학습 모델에서 선택적 전이) 등의 방법이 제안된다. 전이 가능성 예측 지표로는 LEEP, LogME 등이 있다.

**파라미터 효율성(Parameter Efficiency):**
각 다운스트림 태스크마다 전체 파라미터 복사본을 저장하면 비용이 크다.

- **Residual Tuning:** Side-Tuning은 고정된 사전학습 모델에 작은 사이드 네트워크를 추가하여 합산. Adapter Tuning은 각 동결 층 내부에 잔차 어댑터 모듈을 삽입하며, 전체 파라미터의 3.6%만으로 BERT 전체 파인튜닝과 유사한 GLUE 성능을 달성한다.
- **Parameter Difference Tuning:** 전체 파라미터를 $\theta_{\text{task}}=\theta_{\text{pretrained}}\oplus\delta_{\text{task}}$로 분해하여 태스크 특화 차이 벡터 $\delta_{\text{task}}$만 저장한다. Diff Pruning은 $L_0$ 정규화로 $\delta_{\text{task}}$를 희소하게 만들고, Piggyback은 이진 마스크를 활용한다.

**데이터 효율성(Data Efficiency):**
- **Metric Learning:** Matching Net, ProtoNet 등은 레이블 데이터가 극히 적은 few-shot 환경에서 코사인 거리 기반 분류기를 활용하여 파라미터 업데이트 없이 근접 이웃 방식으로 분류한다.
- **Prompt Learning:** 입력 $\mathbf{x}$를 프롬프트 템플릿으로 변환하여 다운스트림 태스크를 사전학습 태스크의 형식으로 변환한다. GPT-3의 in-context learning, PET-TC, Prefix-Tuning, Instruction Tuning(FLAN) 등이 이 패러다임에 속한다.

#### 3.2.2 도메인 적응 (Domain Adaptation)

레이블이 있는 소스 도메인 $\widehat{\mathcal{S}}$에서 레이블이 없는 타깃 도메인 $\widehat{\mathcal{T}}$으로 지식을 전이하는 비지도 도메인 적응(UDA) 문제다. 타깃 위험(target risk) $\epsilon_\mathcal{T}(h)$를 소스 위험과 분포 거리로 바운딩하는 이론이 핵심이다.

**이론적 토대:** Ben-David et al.(2010a)의 이진 분류 바운드:
$$\epsilon_\mathcal{T}(h)\leq\epsilon_\mathcal{S}(h)+d_{\mathcal{H}\Delta\mathcal{H}}(\widehat{\mathcal{S}},\widehat{\mathcal{T}})+\epsilon_{ideal}+4\sqrt{\frac{2d\log(2m)+\log(\frac{2}{\delta})}{m}}$$
이 바운드는 $\mathcal{H}\Delta\mathcal{H}$-발산(두 도메인 간 분포 차이)과 ideal joint error(두 도메인에서 동시에 좋은 가설의 존재 가능성)로 구성된다. Zhang et al.(2019c)의 Disparity Discrepancy(DD)와 Margin Disparity Discrepancy(MDD)는 더 타이트한 바운드를 제공한다.

**통계 매칭(Statistics Matching):**
Maximum Mean Discrepancy(MMD)는 RKHS에서 두 분포의 평균 임베딩 차이를 측정한다:
$$d_{\text{MMD}}^2(\mathcal{S},\mathcal{T})=\|\mathbb{E}_{\mathbf{x}\sim\mathcal{S}}[\phi(\mathbf{x})]-\mathbb{E}_{\mathbf{x}\sim\mathcal{T}}[\phi(\mathbf{x})]\|_{\mathcal{H}_k}^2$$
DAN은 다중 커널 MMD(MK-MMD)를 복수의 층에 적용하고, JAN은 피처와 레이블의 결합 분포를 정렬하는 JMMD를 제안한다. AdaBN은 타깃 도메인의 BatchNorm 통계로 대체하는 방식으로 암묵적 정렬을 수행한다.

**도메인 적대적 학습(Domain Adversarial Learning):**
DANN은 GAN에서 영감을 받아 도메인 판별기 $D$와 피처 생성기 $\psi$ 간의 minimax 게임을 도입한다:

$$L_{\text{DANN}}(\psi)=\max_D\mathbb{E}_{\mathbf{x}^s\sim\widehat{\mathcal{S}}}\log[D(\mathbf{z}^s)]+\mathbb{E}_{\mathbf{x}^t\sim\widehat{\mathcal{T}}}\log[1-D(\mathbf{z}^t)]$$
$$\min_{\psi,h}\mathbb{E}_{(\mathbf{x}^s,\mathbf{y}^s)\sim\widehat{\mathcal{S}}}L_{\text{CE}}(h(\mathbf{z}^s),\mathbf{y}^s)+\lambda L_{\text{DANN}}(\psi)$$

CDAN은 피처 $\mathbf{z}$와 분류기 예측 $\hat{\mathbf{y}}$의 다중선형 맵 $\mathbf{z}\otimes\hat{\mathbf{y}}$를 판별기 입력으로 사용하여 결합 분포의 조건부 정렬을 달성한다. 적대적 학습은 전이 가능성(transferability)을 높이지만 판별 가능성(discriminability)을 희생시킬 수 있으며, BSP와 DSN이 이 딜레마를 완화한다.

**가설 적대적 학습(Hypothesis Adversarial Learning):**
MCD는 두 분류기의 출력 불일치를 최대화하여 $\mathcal{H}\Delta\mathcal{H}$-발산을 추정하고, 피처 생성기는 이를 최소화한다. MDD는 단일 적대적 분류기 $h'$를 사용하여 더 안정적인 최적화를 달성하며, 멀티클래스 설정으로 확장된 이론적 바운드를 제공한다:

$$L_{\text{MDD}}(h,\psi)=\max_{h'}\gamma\mathbb{E}_{\mathbf{x}^s\sim\widehat{\mathcal{S}}}\log[\sigma_{h(\psi(\mathbf{x}^s))}(h'(\psi(\mathbf{x}^s)))]+\mathbb{E}_{\mathbf{x}^t\sim\widehat{\mathcal{T}}}\log[1-\sigma_{h(\psi(\mathbf{x}^t))}(h'(\psi(\mathbf{x}^t)))]$$

**도메인 번역(Domain Translation):**
GAN 기반 번역 모델로 소스 도메인 데이터를 타깃 스타일로 변환한다. CycleGAN은 순환 일관성(cycle consistency) $F(G(\mathbf{x}))\approx\mathbf{x}$를 도입하여 모드 붕괴를 방지하고 내용 보존을 강제한다. 의미 일관성(semantic consistency)을 위해 번역 전후 레이블이 보존되어야 하며($f(\mathbf{x})=f(G(\mathbf{x}))$), CyCADA 등이 이를 위한 proxy 함수를 활용한다.

**반지도 학습(Semi-Supervised Learning):**
일관성 정규화(Self-Ensemble, MMT), 엔트로피 최소화(MCC), 의사 레이블링(CBST, GCE 손실 함수 사용) 등의 SSL 기법을 UDA에 적용한다. GCE 손실은 레이블 노이즈에 강건하다: $L_{\text{GCE}}(\mathbf{x},\tilde{y})=\frac{1}{q}(1-h_{\tilde{y}}(\mathbf{x})^q)$, 여기서 $q\in(0,1]$.

## 4. 실험 및 결과

저자들은 TLlib 라이브러리를 구축하여 재현 가능하고 공정한 벤치마크를 제공한다.

**사전학습 벤치마크:** GLUE 태스크에서 BERT에 비해 T5와 ERNIE가 평균 약 9점 이상 우수한 성능을 보인다. 이미지 인식에서는 SimCLR/BYOL 등 비지도 사전학습이 지도 사전학습(ImageNet supervised)과 비슷하거나 일부 태스크에서 우월한 전이 성능을 보인다. 도메인 변화 벤치마크(ImageNet-Sketch, ImageNet-R)에서는 ViT-Large가 ResNet50보다 20점 이상 높은 성능을 보이며, WSP(약한 지도 사전학습)가 표준 사전학습 대비 큰 폭의 향상을 달성한다.

**태스크 적응 벤치마크:** Adapter Tuning(전체 파라미터의 2.1% 사용)이 GLUE에서 바닐라 파인튜닝과 유사한 성능을 보인다. 이미지 분류에서는 BSS, DELTA 등의 정규화 방법이 특정 태스크(Cars, Aircraft 등)에서 바닐라 파인튜닝을 개선하지만, 평균적인 이득은 제한적이어서 태스크 적응 알고리즘의 효과는 사전학습 태스크와 타깃 태스크의 관련성에 크게 의존함을 보여준다.

**도메인 적응 벤치마크:** DomainNet(대규모)에서 MDD가 평균 51.8%로 최고 성능을 달성하며, DANN은 47.2%에 그친다. 많은 소규모 데이터셋에서 우수한 방법들이 대규모 데이터셋에서는 성능이 크게 저하되는 현상이 관찰되어, 대규모 설정에서의 연구가 중요함을 강조한다.

## 5. 강점, 한계

**강점:** 이 논문의 가장 큰 강점은 분산되어 있던 연구들을 전이 가능성이라는 단일 개념으로 통합하는 포괄적인 관점을 제공한다는 점이다. 이론(도메인 적응 일반화 바운드 등)과 실제 알고리즘을 체계적으로 연결하며, 공개 라이브러리와 벤치마크를 통해 재현 가능성을 보장한다. 사전학습과 적응 단계를 모두 아우르는 전체 생명주기 관점은 기존 서베이에서 보기 어렵다.

**한계 및 미해결 질문들:** 저자들이 명시적으로 인정하는 한계들이 있다. 첫째, 메타러닝과 인과 학습 방법은 소규모 데이터셋에서만 검증되었으며 대규모 사전학습 데이터로 확장될 경우 성능이 개선되는지 불명확하다. 둘째, 비지도 사전학습 태스크 설계는 여전히 경험적(heuristic)이며, 어떤 요소가 전이 가능성을 가능하게 하는지에 대한 견고한 이론이 부재하다. 셋째, 대조 학습에서의 강한 데이터 증강 전략이 텍스트나 그래프 등 다른 모달리티로 설계하기 어렵다. 넷째, 실제 개방형 시나리오에서는 소스와 타깃의 카테고리가 완전히 일치하지 않을 수 있어 도메인 적응의 가정이 성립하지 않는 경우가 많다. 마지막으로, Abnar et al.(2022)이 지적하듯 사전학습 정확도 향상이 항상 다운스트림 성능 향상으로 이어지지는 않으며, 이 포화 현상의 원인이 아직 불명확하다.

## 6. 결론

이 논문은 딥러닝의 전이 가능성을 사전학습과 적응이라는 전체 생명주기에 걸쳐 통합적으로 탐구한 최초의 체계적 서베이 중 하나다. 모델 아키텍처의 역할, 지도/비지도 사전학습 방법의 특성과 상충 관계, 태스크 및 도메인 적응의 핵심 원리와 알고리즘, 그리고 엄밀한 이론적 토대를 하나의 통일된 언어로 기술한다.

실용적 관점에서 이 서베이는 LLM의 파인튜닝, 비전-언어 모델의 zero-shot 전이, PEFT(Parameter-Efficient Fine-Tuning) 등 현재 AI 분야에서 가장 활발한 연구 방향들의 근거를 제공한다. TLlib 라이브러리는 실무자들이 다양한 전이 학습 방법을 공정하게 비교하고 신속하게 적용할 수 있는 도구를 제공한다. 전이 가능성의 적절한 학습 목표가 무엇인지, 그리고 catastrophic forgetting과 negative transfer를 어떻게 원칙적으로 해결할 것인지는 향후 연구가 계속해서 집중해야 할 핵심 미해결 과제로 남아 있다.
