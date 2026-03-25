# Gradual Domain Adaptation without Indexed Intermediate Domains

* **저자**: Hong-You Chen, Wei-Lun Chao
* **발표연도**: 2022
* **arXiv**: <https://arxiv.org/abs/2207.04587>

## 1. 논문 개요

이 논문은 **Gradual Domain Adaptation (GDA)** 를 실제 환경에서 더 넓게 사용할 수 있도록 만들기 위한 연구이다. 기존의 **Unsupervised Domain Adaptation (UDA)** 는 source domain의 라벨된 데이터와 target domain의 비라벨 데이터만을 이용해 적응을 수행하지만, source와 target 사이의 분포 차이가 너무 크면 pseudo-label이 부정확해지고 self-training이 쉽게 무너진다. 이 문제를 줄이기 위해 최근에는 source와 target 사이를 서서히 연결하는 여러 개의 **intermediate domain** 을 활용하는 GDA가 제안되었고, 실제로 성능 향상도 컸다.

그러나 기존 GDA는 중요한 전제를 둔다. 추가 비라벨 데이터가 이미 여러 intermediate domain으로 **잘 나뉘어 있고**, 또 source에서 target으로 이어지는 **순서(index)** 가 주어져 있어야 한다는 점이다. 예를 들어 시간 정보(year, timestamp) 같은 side information으로 순서를 정의하는 식이다. 현실에서는 이런 정보가 없거나 부정확할 수 있고, 심지어 privacy 등의 이유로 접근이 안 될 수도 있다. 이 논문은 바로 이 지점을 겨냥한다. 즉, **추가 비라벨 데이터는 존재하지만 domain별로 묶여 있지도 않고, 순서도 주어지지 않은 상황** 에서 어떻게 intermediate domain sequence를 자동으로 발견할 것인가를 다룬다.

저자들은 이를 위해 **IDOL (Intermediate DOmain Labeler)** 이라는 coarse-to-fine 프레임워크를 제안한다. 먼저 각 비라벨 샘플이 source 쪽에 가까운지 target 쪽에 가까운지를 나타내는 **coarse domain score** 를 부여한다. 그다음 이 거친 순서를 그대로 쓰지 않고, 현재 domain의 분류 지식을 다음 domain이 얼마나 잘 보존하는지를 보려는 **cycle-consistency 기반 refinement** 를 수행한다. 이렇게 얻은 sequence를 GDA 알고리즘에 입력하면, 미리 정의된 순서를 쓰는 경우와 비슷하거나 더 좋은 성능도 가능하다는 것이 핵심 주장이다.

이 문제가 중요한 이유는 명확하다. GDA는 domain gap이 큰 상황에서 매우 강력하지만, 기존에는 “중간 domain의 경계와 순서가 이미 알려져 있다”는 다소 강한 가정 때문에 활용 범위가 좁았다. 이 논문은 그 가정을 약화시켜, **정렬되지 않은 대규모 비라벨 데이터만 있어도 GDA를 적용할 수 있는 길** 을 열었다는 데 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 간단히 말해 다음과 같다. **좋은 intermediate domain sequence를 직접 최적화하기는 너무 어렵기 때문에, 먼저 대략적인 위치를 추정하고(coarse), 그 뒤에 분류 지식 보존 관점에서 정교하게 다듬는다(fine).** 즉, 단순히 “source와 target 사이의 어디쯤에 있는 샘플인가”만 보는 것이 아니라, “이 샘플들을 다음 단계 domain으로 사용했을 때 현재 모델의 discriminative knowledge가 얼마나 자연스럽게 이어지는가”까지 보겠다는 것이다.

기존 접근과의 차별점은 크게 두 가지다.

첫째, 기존 GDA는 대부분 intermediate domain이 이미 정의되어 있고 index도 있다고 가정했다. 반면 이 논문은 **grouping도 indexing도 전혀 없는 추가 비라벨 데이터 집합** 만을 입력으로 받는다. 즉, 문제 정의 자체가 더 어렵고 현실적이다.

둘째, 저자들은 단순한 거리 기반 정렬로 끝내지 않는다. source-target 양끝을 구분하는 **domain discriminator** 로 각 샘플의 대략적 위치를 잡은 뒤, 그 순서를 **cycle-consistency loss** 로 refinement한다. 이 refinement는 “다음 intermediate domain으로 self-training한 뒤, 다시 현재 domain으로 되돌아왔을 때 원래 예측을 얼마나 잘 복원하는가”를 본다. 이는 intermediate domain이 단지 분포적으로 중간에 있는 것뿐 아니라, **classification knowledge를 끊기지 않게 전달하는 좋은 bridge인지** 를 평가하는 장치다.

이 논문에서 특히 인상적인 부분은, refinement criterion을 라벨 없이 정의했다는 점이다. target 라벨이 없기 때문에 최종 target error를 직접 최적화할 수 없는데, 이를 대신해 **forward adaptation + backward adaptation 이후의 prediction consistency** 를 surrogate objective로 사용한다. 이 설계는 논문의 가장 독창적인 부분이다.

## 3. 상세 방법 설명

### 3.1 문제 설정

논문은 source domain의 라벨된 데이터 $\mathcal{S}$, target domain의 비라벨 데이터 $\mathcal{T}$, 그리고 source와 target 사이 어딘가에서 왔다고 믿는 추가 비라벨 데이터 전체 집합 $\mathcal{U}$ 를 입력으로 둔다. 기존 GDA에서는 $\mathcal{U}*1, \dots, \mathcal{U}*{M-1}$ 로 intermediate domain이 이미 나뉘어 있다고 가정하지만, 이 논문에서는 오직 하나의 큰 비라벨 집합 $\mathcal{U}$ 만 주어진다.

GDA의 기본 self-training은 다음처럼 순차적으로 진행된다.

$$
\theta_{m+1} = \texttt{ST}(\theta_m, \mathcal{U}_{m+1})
$$

여기서 $\theta_m$ 은 현재 domain까지 적응된 모델이고, $\texttt{ST}$ 는 pseudo-label을 이용한 self-training이다. 최종 target 모델은 source 모델 $\theta_{\mathcal{S}}$ 에서 시작해 intermediate domains와 target을 따라 순차적으로 적응한 결과가 된다.

직접 UDA에서는 source 모델로 곧바로 target에 pseudo-label을 붙이므로 domain gap이 크면 오류가 커진다. 반대로 GDA에서는 작은 gap의 연속으로 적응하므로 각 단계의 pseudo-label 품질이 더 좋아질 수 있다.

### 3.2 self-training 배경

일반적인 UDA self-training은 다음과 같이 정의된다.

$$
\theta_{\mathcal{T}} = \texttt{ST}(\theta, \mathcal{T})
= \arg\min_{\theta' \in \Theta}
\frac{1}{|\mathcal{T}|}
\sum_{i=1}^{|\mathcal{T}|}
\ell \Big(
f(x_i;\theta'),
\texttt{sharpen}(f(x_i;\theta))
\Big)
$$

여기서 $f(x;\theta)$ 는 classifier의 logits 혹은 prediction이고, $\texttt{sharpen}$ 은 argmax 기반 pseudo-label 생성으로 이해하면 된다. 핵심은 현재 모델 $\theta$ 가 target 데이터에 대해 내놓는 예측을 pseudo-label로 삼아, 새로운 모델 $\theta'$ 를 supervised하게 학습시키는 것이다.

GDA에서는 이 self-training을 한 번에 target으로 하지 않고 intermediate domains를 따라 반복한다.

$$
\theta_{\mathcal{T}} = \theta_M
= \texttt{ST}\big(\theta_{\mathcal{S}}, (\mathcal{U}_1,\ldots,\mathcal{U}_M)\big)
$$

여기서 $\mathcal{U}_M = \mathcal{T}$ 로 보면 된다.

논문은 Kumar et al.의 이론을 인용하여, consecutive domains 사이의 거리 $\rho(P_m, P_{m+1})$ 가 충분히 작으면 target error가 제어될 수 있음을 설명한다. 정리의 핵심은 다음 bound이다.

$$
\mathcal{L}(\theta_M, P_M)
\le
\beta^{M+1}
\Big(
\mathcal{L}(\theta_0, P_0)
+
\frac{4BR+\sqrt{2\log(2M/\delta)}}{\sqrt{|\mathcal{U}|}}
\Big),
\quad
\beta = \frac{2}{1-\rho R}
$$

직관적으로는, **각 단계의 domain shift가 작을수록 전체 적응이 안정적** 이라는 뜻이다. 따라서 좋은 intermediate domain sequence란, 이 $\rho(P_m, P_{m+1})$ 를 단계별로 작게 만드는 sequence라고 볼 수 있다.

### 3.3 IDOL의 전체 구조

IDOL의 목표는 큰 비라벨 집합 $\mathcal{U}$ 를 source에서 target으로 이어지는 순서로 정렬하고, 이를 $M-1$ 개 intermediate domain으로 chunking하는 것이다. 논문은 이를 직접 풀기 어렵다고 말한다. 이유는 두 가지다. 하나는 labeled target이 없어 최종 target loss를 직접 측정할 수 없고, 다른 하나는 가능한 domain sequence 조합이 너무 많아 combinatorial optimization이 되기 때문이다.

그래서 저자들은 **coarse stage + fine stage** 로 나눈다.

* coarse stage에서는 각 샘플에 source 쪽인지 target 쪽인지 나타내는 점수 $q_i$ 를 부여한다.
* fine stage에서는 이 점수를 초기값으로 삼아, cycle-consistency를 통해 더 좋은 sequence로 refinement한다.

### 3.4 Coarse stage: domain score 부여

#### (1) Classifier confidence

가장 단순한 방식은 source model의 confidence를 쓰는 것이다.

$$
q_i = \max f(x_i; \theta_{\mathcal{S}})
$$

혹은 점진적으로 현재 모델의 confidence가 높은 샘플부터 뽑는 식이다. 하지만 이 방식은 target 정보를 거의 사용하지 않고, 단지 “쉽게 분류되는 샘플”을 먼저 고르는 경향이 있어서 source-to-target의 실제 shift를 잘 반영하지 못한다.

#### (2) Manifold distance

source, target, intermediate 데이터를 source model의 feature로 embedding한 후 UMAP 같은 manifold learning으로 저차원 공간에 놓고, source와 target에 대한 상대적 거리를 계산한다. 논문은 다음 score를 사용한다.

$$
q_i =
\frac{
\min_{x^{\mathcal{T}}\in\mathcal{T}}
|\gamma(x_i)-\gamma(x^{\mathcal{T}})|_2
}{
\min_{x^{\mathcal{S}}\in\mathcal{S}}
|\gamma(x_i)-\gamma(x^{\mathcal{S}})|_2
}
$$

여기서 $\gamma(x)$ 는 UMAP을 거친 feature이다. 이 값은 샘플이 source보다 target에 더 가까운지, 혹은 반대인지를 반영한다. 다만 이 방식은 feature manifold의 품질에 의존한다.

#### (3) Domain discriminator

binary classifier $g(\cdot;\phi)$ 를 학습하여 source는 1, target은 0으로 구분한다. 학습 loss는 binary cross-entropy이다.

$$
\mathcal{L}(\phi) = -\frac{1}{|\mathcal{S}|}\sum_{x^{\mathcal{S}}\in \mathcal{S}} \log(\sigma(g(x^{\mathcal{S}};\phi))) - \frac{1}{|\mathcal{T}|}\sum_{x^{\mathcal{T}}\in \mathcal{T}} \log(1-\sigma(g(x^{\mathcal{T}};\phi)))
$$

학습 후 intermediate 샘플에 대해

$$
q_i = g(x_i;\phi)
$$

를 점수로 사용한다. 이 방식은 source-target 양끝을 직접 대비시킨다는 장점이 있다.

#### (4) Progressive domain discriminator

저자들이 가장 좋은 coarse scoring으로 보고한 방식이다. 문제는 기본 domain discriminator가 source나 target 근처 샘플에는 강하지만, 둘 다에서 멀리 떨어진 intermediate region의 샘플에는 calibration이 나빠질 수 있다는 점이다. 그래서 **점진적으로 intermediate 샘플을 양끝 학습 집합에 흡수하면서 discriminator를 재학습** 한다.

절차는 대략 이렇다.

1. 현재 source 집합과 target 집합으로 discriminator를 학습한다.
2. 남은 intermediate 샘플들에 대해 score $\hat q_i$ 를 예측한다.
3. score가 가장 높은 일부는 source side에 가깝다고 보고 source 쪽에 추가한다.
4. score가 가장 낮은 일부는 target side에 가깝다고 보고 target 쪽에 추가한다.
5. 이를 $K$ 라운드 반복한다.

이 방식은 source와 target 사이의 데이터 manifold를 progressively 따라가며 discriminator를 보정하는 효과가 있다. 논문 실험에서도 이 방법이 가장 강력한 coarse ordering을 제공한다.

### 3.5 Fine stage: cycle-consistency refinement

coarse score만으로 얻은 순서는 distributional position만 반영할 뿐, 그 domain이 실제로 **분류 지식을 잘 전달하는 bridge인지** 는 보장하지 않는다. 저자들은 이 문제를 해결하기 위해 cycle-consistency를 도입한다.

핵심 직관은 다음과 같다. 현재 domain $\mathcal{U}_m$ 에서 다음 domain $\mathcal{U}_{m+1}$ 로 self-training해 모델 $\theta_{m+1}$ 을 만든 뒤, 다시 $\mathcal{U}_m$ 으로 역방향 self-training해서 $\theta'_m$ 을 만들었을 때, $\theta'_m$ 이 원래 $\theta_m$ 과 비슷한 예측을 해야 한다. 즉, 좋은 다음 domain이라면 **forward로 넘어갔다가 backward로 되돌아와도 현재 domain의 discriminative knowledge를 잃지 않아야 한다.**

전체 cycle objective는 개념적으로 다음과 같이 표현된다.

$$
\arg\min_{(\mathcal{U}_1,\ldots,\mathcal{U}_{M-1})}
\mathbb{E}_{x\sim P_0}
\left[
\ell\big(
f(x;\theta'_0),
\texttt{sharpen}(f(x;\theta_0))
\big)
\right]
$$

subject to

$$
\theta_M = \texttt{ST}(\theta_0, (\mathcal{U}_1,\ldots,\mathcal{U}_M))
$$

$$
\theta'_0 = \texttt{ST}(\theta_M, (\mathcal{U}_{M-1},\ldots,\mathcal{U}_0))
$$

하지만 이것도 여전히 어렵기 때문에, 논문은 greedy하게 **다음 intermediate domain 하나씩** 찾는다. 즉, 현재 모델 $\theta_m$ 과 현재 domain $\mathcal{U}*m$ 이 있을 때, 남은 샘플 집합 $\mathcal{U}*{\setminus m}$ 에서 다음 domain $\mathcal{U}_{m+1}$ 을 고른다.

이를 위한 sub-problem은 다음이다.

$$
\arg\min_{\mathcal{U}_{m+1}\subset \mathcal{U}_{\setminus m}}
\frac{1}{|\mathcal{U}_m|}
\sum_{x\in \mathcal{U}_m}
\ell\big(
f(x;\theta'_m),
\texttt{sharpen}(f(x;\theta_m))
\big)
$$

subject to

$$
\theta_{m+1} = \texttt{ST}(\theta_m,\mathcal{U}_{m+1})
$$

$$
\theta'_m = \texttt{ST}(\theta_{m+1},\mathcal{U}_m)
$$

직관적으로, 현재 domain에서 다음 domain으로 갔다가 다시 돌아왔을 때 prediction이 잘 보존되는 후보 집합을 고르는 것이다.

### 3.6 Meta-reweighting으로의 relaxation

$\mathcal{U}_{m+1}$ 를 직접 subset으로 고르는 것은 discrete combinatorial problem이라 어렵다. 그래서 저자들은 각 샘플에 대해 binary selection 대신 연속적인 가중치 $q_i$ 를 둔다. 즉, $q_i=1$ 이면 다음 domain에 포함, $q_i=0$ 이면 제외라는 이상적 설정을 완화하여, $q_i\in\mathbb{R}$ 인 learnable weight로 놓는다.

그러면 다음 domain으로 self-training하는 단계는

$$
\texttt{ST}(\theta_m,\mathbf{q}) = \arg\min_{\theta\in\Theta} \frac{1}{N} \sum_{i=1}^N q_i \cdot \ell\big( f(x_i;\theta), \texttt{sharpen}(f(x_i;\theta_m)) \big)
$$

처럼 쓸 수 있다. 즉, 각 intermediate 샘플이 self-training loss에 얼마나 기여할지를 가중치로 조절하는 것이다.

논문은 meta-reweighting 절차를 사용한다. 대략적인 흐름은 다음과 같다.

먼저 현재 모델 $\theta_m$ 에서 시작해, candidate samples에 대해 weight $\mathbf{q}$ 를 곱한 self-training step을 수행하여 임시 모델 $\theta(\mathbf{q})$ 를 만든다. 그다음 이 임시 모델을 다시 현재 domain $\mathcal{U}_m$ 에 적응시키고, 최종적으로 현재 domain에서의 prediction consistency loss를 줄이는 방향으로 $\mathbf{q}$ 를 업데이트한다. 즉, **현재 domain의 지식을 잘 보존하게 만드는 샘플일수록 높은 weight를 받는다.**

업데이트 후에는 음수 weight를 막기 위해

$$
q_i \leftarrow \max{0, q_i}
$$

를 적용한다. 최종적으로 높은 $q_i$ 를 가진 샘플 상위 일부를 다음 intermediate domain으로 선택한다.

### 3.7 왜 coarse initialization이 필요한가

fine stage는 현재 domain $\mathcal{U}_m$ 을 기준으로 다음 domain 하나만 greedy하게 찾기 때문에, 자칫하면 현재 지식 보존에는 도움이 되지만 전체 source-to-target 경로에서는 엉뚱한 샘플이 선택될 수도 있다. 그래서 coarse score가 중요한 역할을 한다. coarse stage가 **전체적인 방향성**, 즉 source에서 target으로 가는 큰 흐름을 제공하고, fine stage는 그 흐름 안에서 **knowledge-preserving refinement** 를 하는 구조다. 저자들은 이 점을 실험으로도 강조한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

주요 실험은 두 benchmark에서 수행된다.

첫째는 **Rotated MNIST** 이다. MNIST 숫자를 $[0,60]$ 도 범위에서 회전시켜 gradual shift를 만들고, source는 $[0,5)$ 도, target은 $[55,60]$ 도, intermediate는 그 사이 영역으로 구성한다. 각 domain은 2000장 이미지이며 source와 target에는 validation용 1000장이 별도로 있다.

둘째는 **Portraits** 이다. 미국 고등학교 졸업앨범 사진을 연도별로 정렬한 real-world gender classification 데이터셋이다. 시간이 지남에 따라 hairstyle, fashion, lip curvature 같은 시각적 특성이 바뀌므로 자연스러운 domain shift가 존재한다. 여기서 pre-defined index는 year이다.

모델은 두 데이터셋에서 모두 convolutional neural network를 사용하고, 각 domain마다 20 epochs씩 학습한다. IDOL의 주요 hyperparameter는 progressive discriminator의 라운드 수 $K=2M$, refinement에서 step당 30 epochs 등이다. Rotated MNIST는 $M=19$, Portraits는 $M=7$ 로 설정된다.

비교 대상은 다음과 같다.

* Source only
* UDA on target only
* UDA on target + all intermediate pooled together
* GDA with pre-defined indexes
* GDA with random sequences
* IDOL with various coarse scores
* IDOL with/without refinement

### 4.2 메인 결과

가장 중요한 결과는 Table 1이다.

#### Rotated MNIST

* Source only: $31.9 \pm 1.7$
* UDA on target only: $33.0 \pm 2.2$
* UDA on target + all intermediate: $38.0 \pm 1.6$
* Pre-defined GDA: $87.9 \pm 1.2$
* Pre-defined GDA + refinement: $93.3 \pm 2.3$
* Random GDA: $39.5 \pm 2.0$
* Random GDA + refinement: $57.5 \pm 2.7$
* Classifier confidence coarse: $45.5 \pm 3.5$
* Manifold distance coarse: $72.4 \pm 3.1$
* Domain discriminator coarse: $82.1 \pm 2.7$
* Progressive domain discriminator coarse: $85.7 \pm 2.7$
* Progressive domain discriminator + refinement: $87.5 \pm 2.0$

이 결과는 메시지가 아주 분명하다. intermediate 데이터를 그냥 한데 모아 UDA를 해도 거의 도움이 안 되며, random sequence 역시 성능이 낮다. 반면 **좋은 sequence만 있으면 GDA가 dramatic하게 좋아진다.** 그리고 IDOL은 pre-defined index가 없어도 progressive discriminator만으로 $85.7$ 까지 올라가며, refinement 후에는 $87.5$ 로 pre-defined GDA의 $87.9$ 와 거의 맞먹는다.

#### Portraits

* Source only: $75.3 \pm 1.6$
* UDA on target only: $76.9 \pm 2.1$
* UDA on target + all intermediate: $78.9 \pm 3.0$
* Pre-defined GDA: $83.8 \pm 0.8$
* Pre-defined GDA + refinement: $85.8 \pm 0.4$
* Random GDA: $81.1 \pm 1.8$
* Random GDA + refinement: $82.5 \pm 2.2$
* Classifier confidence coarse: $79.3 \pm 1.7$
* Manifold distance coarse: $81.9 \pm 0.8$
* Domain discriminator coarse: $82.3 \pm 0.9$
* Progressive domain discriminator coarse: $83.4 \pm 0.8$
* Progressive domain discriminator + refinement: $85.5 \pm 1.0$

Portraits에서도 비슷한 경향이 나온다. 특히 refinement가 매우 강력해서, progressive discriminator coarse의 $83.4$ 에서 refinement 후 $85.5$ 로 상승하며, pre-defined + refinement의 $85.8$ 과 거의 동일하다.

### 4.3 refinement의 효과

논문은 refinement가 단순한 부가 장치가 아니라 핵심이라고 본다. supplementary Table 3를 보면 모든 coarse score 방식에서 refinement가 일관되게 성능을 올린다.

예를 들어 Rotated MNIST에서:

* confidence: $45.5 \to 62.5$
* manifold distance: $72.4 \to 82.4$
* domain discriminator: $82.1 \to 86.2$
* progressive discriminator: $85.7 \to 87.5$

Portraits에서도:

* confidence: $79.3 \to 83.6$
* manifold distance: $81.9 \to 85.2$
* domain discriminator: $82.3 \to 85.1$
* progressive discriminator: $82.3$ 또는 본문 기준 $83.4 \to 85.5$

즉, coarse ordering의 품질이 좋을수록 refinement가 더 좋은 최종 sequence를 만든다. 논문이 coarse-to-fine 구조를 택한 이유가 실험적으로 정당화된다.

### 4.4 Portraits 분석: year가 최선의 index인가?

논문은 흥미롭게도 Portraits에서 pre-defined year index가 항상 최선은 아니라고 주장한다. Figure 4에 따르면, year 순서로 adaptation하면 target accuracy가 단계별로 출렁이는 반면, IDOL sequence를 쓰면 성능이 더 안정적으로 증가하고 최종 accuracy도 더 높아진다.

IDOL sequence와 actual year의 상관계수는 $0.727$ 이다. 즉, 어느 정도는 연도를 반영하지만 완전히 같지는 않다. 이는 실제 portrait 스타일 변화가 단순 연도 하나로 설명되지 않기 때문이라고 해석한다. hairstyle, eyeglasses, fashion, lip curvature 등 여러 요인이 섞여 있고, 같은 해 안에서도 개인차가 있다. 따라서 IDOL은 “연도”가 아니라 **분류 지식 전달에 더 적합한 시각적 변화 축** 을 찾아낸 것으로 볼 수 있다.

이 부분은 논문의 중요한 포인트다. 즉, IDOL은 단지 pre-defined index를 흉내 내는 것이 아니라, 경우에 따라 **더 나은 domain sequence** 를 찾을 수도 있다.

### 4.5 class balance 관련 관찰

저자들은 intermediate domains가 target class distribution과 맞지 않을 수 있다는 점도 언급한다. 흥미롭게도 fine-grained indexes는 coarse ones보다 더 class-balanced했다. 논문은 각 domain에서 클래스별 샘플 수 비율의 worst-case imbalance를 보고했고, refinement 후 더 1에 가까워졌다고 한다.

* Rotated MNIST: $1.33 \to 1.23$
* Portraits: $1.27 \to 1.25$

이는 refinement가 단지 domain ordering만 개선하는 것이 아니라, 결과적으로 class distribution 면에서도 더 안정적인 intermediate domains를 만들 가능성을 시사한다.

### 4.6 부분 정보, outlier, noisy index 상황

논문은 현실적인 변형 실험도 수행한다.

첫째, intermediate domains가 그룹은 되어 있지만 순서가 없는 경우다. 이 경우 각 그룹의 평균 domain score를 이용하면 pre-defined order를 완벽히 복원할 수 있었다고 한다.

둘째, pre-defined domains가 너무 거칠게만 나뉜 경우다. 예를 들어 domain 수가 줄어들어 각 domain이 너무 넓어지면 Rotated MNIST에서 accuracy가 11% 감소했다. 하지만 IDOL은 더 fine-grained sequence를 다시 구성해 성능을 회복했다.

셋째, outlier domains가 포함된 경우다. 예를 들어 Rotated MNIST에서 intermediate range를 $[-30, 90]$ 도로 확장하면 source-target 사이를 벗어난 샘플이 포함된다. 이때 GDA accuracy는 $77.0 \pm 2.0$ 으로 떨어지지만, refinement를 적용하면 $81.3 \pm 1.4$ 로 회복된다.

즉, IDOL은 완벽한 setting뿐 아니라 **부분적으로 잘못된 intermediate data / sequence** 에 대해서도 robustness를 보인다.

### 4.7 intermediate data가 적거나 noisy한 경우

실제에서는 중간 데이터가 적거나, 일부만 clean index가 있고 나머지는 noisy index를 가질 수 있다. 논문은 30%만 intermediate data를 남기는 실험을 했다. clean index만 있어도 성능은 크게 감소했다.

* MNIST는 약 14% 성능 하락
* Portraits는 약 5% 성능 하락

특히 MNIST에서는 noisy-indexed data를 더 넣으면 오히려 성능이 더 나빠질 수 있었다. 그러나 IDOL은 이런 unindexed/noisy data에도 index를 새로 부여할 수 있으므로, 데이터가 많아질수록 안정적으로 성능이 향상되었다. 이는 IDOL이 **additional unlabeled data를 “쓸 수 있는 intermediate data”로 변환하는 도구** 라는 관점을 잘 보여준다.

### 4.8 CIFAR10-STL case study

추가로 논문은 CIFAR10-to-STL UDA에 IDOL을 적용한다. 여기서는 STL의 extra unlabeled set을 intermediate data로 사용하며, ImageNet에서 sub-sample된 데이터라 unknown/outlier domain과 class가 섞여 있다. 매우 어려운 setting이다.

결과는 Table 2와 같다.

* Source only: $76.6 \pm 0.4$
* UDA on target (lr=$10^{-4}$): $69.4 \pm 0.4$
* UDA on target (lr=$10^{-5}$): $75.1 \pm 0.3$
* UDA on target + all intermediate: $61.1 \pm 0.8$
* GDA with confidence: $77.1 \pm 0.5$
* GDA with IDOL: $78.1 \pm 0.4$

즉, 단순 UDA는 source only보다도 못할 수 있고, intermediate를 그냥 섞으면 더 망가진다. 하지만 IDOL을 통해 sequence를 구성하면 가장 높은 성능을 얻는다. 저자들은 이 실험을 통해 IDOL이 benchmark GDA setting뿐 아니라 **open-domain unlabeled data가 섞인 실제적 상황** 에도 적용 가능함을 보이려 한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정 자체가 현실적이라는 점이다. 기존 GDA는 index가 주어진 intermediate domains를 전제로 하지만, 실제 환경에서는 데이터가 그렇게 깔끔하게 나뉘어 있지 않다. 이 논문은 **“추가 비라벨 데이터는 있지만 정렬은 안 되어 있다”** 는 훨씬 현실적인 조건을 명시적으로 다룬다. 단순히 기존 방법의 성능 개선이 아니라, GDA의 적용 가능 범위를 넓히는 기여다.

두 번째 강점은 방법론의 구조가 설득력 있다는 점이다. coarse stage에서는 distributional position을, fine stage에서는 discriminative knowledge preservation을 본다. 이 둘은 서로 보완적이다. 단지 source-target 사이 중간에 있다고 좋은 domain이 되는 것은 아니고, 또 지식 보존만 보면 전체 경로에서 벗어난 샘플이 끼어들 수 있는데, IDOL은 이 두 관점을 함께 사용한다.

세 번째 강점은 cycle-consistency를 intermediate domain discovery에 적용한 발상이 참신하다는 점이다. 보통 cycle-consistency는 image translation이나 style transfer 같은 문제에서 많이 쓰이는데, 여기서는 **domain sequence quality를 label-free하게 평가하는 surrogate** 로 재해석했다. 이는 논문에서 가장 창의적인 부분이다.

네 번째 강점은 실험이 비교적 충실하다는 점이다. 단순 benchmark 성능 비교뿐 아니라, random sequence, outlier domains, partial information, noisy indexes, open-domain unlabeled set 등 실제 사용 시 마주칠 수 있는 상황들을 폭넓게 다룬다. 따라서 “이 방법이 단지 benchmark에 맞춘 것인가?”라는 의문에 어느 정도 답한다.

반면 한계도 분명하다.

가장 큰 한계는 이 방법이 여전히 **추가 비라벨 데이터가 source와 target 사이를 어느 정도 부드럽게 연결한다** 는 GDA의 기본 가정 위에 서 있다는 점이다. 논문도 limitation에서 이를 인정한다. 만약 intermediate data가 source-target 사이를 잇지 않고 완전히 엉뚱한 영역에 많이 분포한다면, IDOL도 일부를 intermediate domain으로 잘못 포함할 수 있다.

두 번째 한계는 계산 비용이다. progressive discriminator는 여러 라운드의 재학습이 필요하고, refinement는 meta-reweighting을 사용하므로 한 단계당 standard training보다 훨씬 비싸다. supplementary에 따르면 Portraits에서 refinement로 intermediate domain 하나를 찾는 데 수백 초가 걸린다. 즉, 개념은 우아하지만 큰 데이터셋이나 복잡한 backbone에서는 비용이 커질 수 있다.

세 번째 한계는 최적화의 greedy nature다. 논문도 이를 명시적으로 인정한다. 전체 sequence를 jointly optimize하지 않고 한 domain씩 순서대로 고르기 때문에 global optimum을 보장하지 않는다. coarse initialization이 이 문제를 완화하지만, 완전히 해결하지는 못한다.

네 번째로, hyperparameter 의존성도 있다. intermediate domain 개수 $M-1$, progressive training round 수 $K$, refinement step 수, pseudo-label filtering 비율 등이 모두 결과에 영향을 줄 수 있다. 논문은 target validation set으로 tuning했다고 적고 있는데, 실제 완전한 unsupervised adaptation 환경에서는 이런 tuning이 현실적으로 까다로울 수 있다.

비판적으로 보면, 이 논문은 “좋은 sequence를 찾는다”는 목표를 cycle-consistency로 근사했는데, 이 surrogate가 항상 최종 target accuracy와 정확히 일치한다고 보장되지는 않는다. 다만 실험에서는 상관이 꽤 높게 나타난다. 따라서 이 부분은 엄밀한 이론적 정당화가 더 보강되면 좋을 여지가 있다.

## 6. 결론

이 논문은 Gradual Domain Adaptation의 가장 큰 실용적 제약 중 하나였던 **indexed intermediate domains의 필요성** 을 정면으로 다룬다. 추가 비라벨 데이터가 주어져도 그것이 domain별로 나뉘어 있지 않고 순서도 없는 상황에서, IDOL은 먼저 progressive domain discriminator로 coarse ordering을 만들고, 이어서 cycle-consistency 기반 refinement로 discriminative knowledge가 잘 이어지는 sequence를 찾는다.

실험 결과는 매우 설득력 있다. IDOL은 pre-defined index 없이도 기존 GDA와 비슷한 수준의 성능을 내며, 어떤 경우에는 pre-defined sequence보다 더 좋은 sequence를 찾기도 한다. 특히 refinement는 coarse sequence를 안정적으로 개선하며, outlier/noisy-index/low-quality intermediate data 같은 현실적인 조건에서도 유효하다.

이 연구의 실제적 의미는 크다. 현실의 데이터는 시간이나 장소에 따라 서서히 변하지만, 그 변화가 깔끔한 domain label로 제공되지는 않는다. IDOL은 이런 데이터를 자동으로 정렬해 GDA가 활용할 수 있게 만드는 도구다. 따라서 향후 자율주행, 센서 드리프트, 의료 영상, 장기간 수집되는 시각 데이터 등 **시간에 따라 점진적으로 변하는 실제 응용 분야** 에서 중요한 역할을 할 가능성이 있다. 동시에, 더 큰 규모의 데이터셋과 더 다양한 shift 유형에서 IDOL을 시험하고, sequence discovery의 이론을 더 정교하게 다듬는 것이 후속 연구 과제가 될 것이다.
