# Scatter Component Analysis: A Unified Framework for Domain Adaptation and Domain Generalization

* **저자**: Muhammad Ghifary, David Balduzzi, W. Bastiaan Kleijn, and Mengjie Zhang
* **발표연도**: 2015
* **arXiv**: [https://arxiv.org/abs/1510.04373](https://arxiv.org/abs/1510.04373)

## 1. 논문 개요

이 논문은 서로 다른 데이터 분포 때문에 발생하는 dataset bias 문제를 줄이기 위해, **domain adaptation**과 **domain generalization**을 하나의 관점에서 다룰 수 있는 representation learning 방법인 **Scatter Component Analysis (SCA)**를 제안한다. 문제 설정은 분명하다. 분류기를 학습할 때 레이블은 source domain에만 있고, 실제 성능을 내야 하는 target domain은 분포가 다르다. Domain adaptation에서는 unlabeled target data를 학습 중에 볼 수 있지만, domain generalization에서는 target data조차 볼 수 없다. 논문은 이 둘이 매우 비슷한 문제임에도 기존 방법들이 대체로 서로 호환되지 않고, 계산량도 크다는 점을 핵심적인 실무 문제로 본다.

연구 문제는 다음과 같이 정리할 수 있다. 첫째, 서로 다른 도메인 사이의 분포 차이를 줄이면서도 분류에 유리한 feature representation을 어떻게 배울 것인가. 둘째, 그 방법이 domain adaptation과 domain generalization 모두에 적용될 수 있는가. 셋째, 최적화가 빠르고 정확하게 풀리는 형태로 설계할 수 있는가. 논문은 이 세 가지를 동시에 만족시키는 알고리즘을 목표로 한다.

이 문제가 중요한 이유는 매우 실용적이다. 실제 컴퓨터 비전 환경에서는 학습 데이터와 테스트 데이터가 동일한 조건에서 수집되지 않는 경우가 많다. 카메라 시점, 배경, 조명, 해상도, 객체 변형, 데이터 수집 프로토콜 등이 바뀌면 분포가 달라지고, 일반적인 supervised learning 가정인 “train/test가 같은 분포에서 왔다”는 전제가 깨진다. 따라서 레이블이 부족하고 도메인 간 편향이 큰 환경에서도 잘 작동하는 방법은 object recognition 같은 실전 문제에서 가치가 크다.

또한 저자들은 이 논문의 기여를 세 갈래로 제시한다. 첫째, **scatter**라는 단순한 기하학적 양을 도입해 도메인 차이, 클래스 분리도, 전체 데이터 분산을 하나의 공통 언어로 표현한다. 둘째, 이 scatter를 바탕으로 generalized eigenvalue problem으로 환원되는 빠른 알고리즘 SCA를 설계한다. 셋째, domain adaptation의 경우 domain scatter가 일반화 성능과 연결된다는 이론적 근거를 제시한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 직관적이다. 좋은 representation은 단순히 “도메인 차이를 줄이는 것”만으로는 충분하지 않다. 분류 문제를 잘 풀려면 서로 다른 클래스는 멀어져야 하고, 같은 클래스는 가까워져야 하며, 데이터 전체의 정보량도 너무 줄어들면 안 된다. 저자들은 이 요구를 네 가지 조건으로 정리한다.

첫째, 서로 다른 label을 가진 점들은 잘 분리되어야 한다. 둘째, 데이터 전체는 충분한 분산을 가져야 한다. 셋째, 같은 label을 공유하는 점들은 서로 가깝게 모여야 한다. 넷째, 서로 다른 domain 사이의 mismatch는 작아져야 한다. SCA는 이 네 가지를 각각 **between-class scatter**, **total scatter**, **within-class scatter**, **domain scatter**로 정량화하고, “좋은 것은 키우고 나쁜 것은 줄이는” 비율 최적화 문제로 묶는다.

이 논문의 차별점은 두 가지 층위에서 드러난다. 하나는 **통합성**이다. 기존의 많은 방법은 domain adaptation 전용이거나 domain generalization 전용이었다. 반면 SCA는 입력으로 어떤 도메인 구성을 넣느냐에 따라 두 문제를 모두 같은 알고리즘 틀 안에서 다룬다. 다른 하나는 **계산 효율성**이다. 논문은 최적화가 generalized eigenproblem으로 귀결되므로 fast and exact solution을 얻을 수 있다고 주장한다. 즉, 복잡한 반복 최적화 대신 선형대수 문제로 환원하여 계산 부담을 줄인다.

또한 저자들은 scatter 개념이 기존의 잘 알려진 개념들과 연결된다고 강조한다. Total scatter는 PCA와 연결되고, domain scatter는 MMD 또는 distributional variance와 연결되며, class scatter는 Fisher’s linear discriminant와 연결된다. 따라서 SCA는 완전히 새로운 개념을 제시한다기보다, 기존의 여러 목적을 하나의 공통 기하학적 틀로 재정리하고 이를 통합한 방법이라고 보는 것이 정확하다.

## 3. 상세 방법 설명

### 3.1 기본 수학적 배경: RKHS와 mean map

논문은 입력 공간을 직접 다루지 않고, **RKHS (Reproducing Kernel Hilbert Space)** 위에서 논의를 전개한다. 이유는 비선형 feature mapping을 명시적으로 계산하지 않고도 kernel trick을 통해 고차원 공간에서 선형 분리를 시도할 수 있기 때문이다. 입력 $x$는 feature map $\phi(x)$를 통해 RKHS $\mathcal{H}$로 옮겨지고, kernel은 $\kappa(t,x)=\langle \phi(t), \phi(x)\rangle$로 정의된다.

그 다음 중요한 개념이 **mean map**이다. 어떤 분포 $\mathbb{P}$를 RKHS 상의 한 점으로 보내는 방법으로, 다음과 같이 정의된다.

$$
\mu_{\mathbb{P}}=\mathbb{E}_{x\sim \mathbb{P}}[\phi(x)]
$$

즉, 분포를 feature space에서의 centroid로 나타내는 것이다. 이 표현을 쓰면 “도메인”도 하나의 점처럼 다룰 수 있고, 클래스 조건부분포 역시 하나의 점처럼 다룰 수 있다. 이 점이 논문의 통일적 서술의 출발점이다.

### 3.2 Scatter의 정의와 의미

논문이 제안하는 **scatter**는 분포가 자신의 centroid 주변에서 얼마나 퍼져 있는지를 나타내는 양이다. 정의는 다음과 같다.

$$
\Psi_{\phi}(\mathbb{P})=\mathbb{E}_{x\sim \mathbb{P}}\left[|\mu_{\mathbb{P}}-\phi(x)|_{\mathcal{H}}^2\right]
$$

쉽게 말하면, 분포의 평균으로부터 샘플들이 얼마나 떨어져 있는가를 평균 제곱거리로 측정한 것이다. 이것은 분산의 일반화된 형태로 이해할 수 있다. 논문은 finite sample에 대해 empirical distribution으로 이를 근사할 수 있음을 말하고, sample size가 커질수록 추정 오차가 줄어드는 bound도 제시한다.

특히 $\phi$가 identity map이면 scatter는 일반적인 total variance가 된다.

$$
\Psi(\mathbf{X})=\operatorname{Tr}\operatorname{Cov}(\mathbf{X})
$$

이 결과는 매우 중요하다. 왜냐하면 이후의 모든 목적함수가 결국 “variance-like quantity”를 여러 방향으로 나눈 것임을 보여주기 때문이다.

### 3.3 SCA가 사용하는 네 가지 scatter

#### (1) Total scatter

전체 데이터의 정보량과 변동성을 유지하기 위한 항이다. 여러 도메인의 평균 분포를 $\bar{\mathbb{P}}_X$라고 두면 total scatter는

$$
\text{total scatter}=\Psi_{\phi}(\bar{\mathbb{P}}_X)
$$

로 정의된다. 경험적으로는 feature matrix를 변환한 뒤의 전체 분산 trace와 연결된다. 이 항을 크게 만든다는 것은 representation이 지나치게 collapse되지 않고 데이터를 넓게 펼치도록 유도한다는 뜻이다. 논문은 이 관점에서 Kernel PCA가 total scatter만 최대화하는 특별한 경우라고 설명한다.

#### (2) Domain scatter

도메인 간 분포 차이를 줄이기 위한 항이다. 각 도메인 $\mathbb{P}^1_X,\dots,\mathbb{P}^m_X$를 mean embedding $\mu_{\mathbb{P}^d_X}$로 나타낸 뒤, 이 점들의 scatter를 계산한다.

$$
\Psi\big({\mu_{\mathbb{P}^1_X},\dots,\mu_{\mathbb{P}^m_X}}\big) = \frac{1}{m}\sum_{i=1}^{m}|\bar{\mu}-\mu_{\mathbb{P}^i}|^2
$$

여기서 $\bar{\mu}$는 도메인 centroid이다. 이 값이 작을수록 각 도메인의 평균 embedding이 서로 가깝다는 뜻이다. 논문은 이 양이 **distributional variance**와 일치하며, 두 도메인의 경우에는 **MMD의 제곱의 $1/4$**와 같다고 보인다.

$$
\Psi({\mu_{\mathbb{P}},\mu_{\mathbb{Q}}}) = \frac{1}{4}\operatorname{MMD}_{\mathcal{F}}^2[\mathbb{P},\mathbb{Q}]
$$

즉, SCA의 domain scatter 최소화는 도메인 정렬(domain alignment)을 kernel mean discrepancy 관점에서 수행하는 것과 본질적으로 같다.

#### (3) Within-class scatter

같은 클래스 내부의 샘플들을 더 응집시키기 위한 항이다. 클래스 $k$에 속한 샘플들이 centroid $\mu_k$ 주변에 얼마나 흩어져 있는지를 측정한다. 값이 작을수록 같은 클래스 샘플들이 조밀하게 모이게 된다.

#### (4) Between-class scatter

서로 다른 클래스 centroid들을 멀어지게 만드는 항이다. 이는 클래스 간 분리도를 높이는 역할을 한다. 논문은 이 두 항이 고전적인 Fisher LDA의 within/between scatter와 정확히 연결된다고 설명한다. 따라서 SCA는 단순한 domain alignment 기법이 아니라, supervised discriminative structure까지 포함하는 방법이다.

### 3.4 커널 기반 표현과 행렬 형태

논문은 직접 $\phi(x)$를 계산하지 않고 kernel matrix $\mathbf{K}$를 사용한다. 전체 샘플을 모은 뒤,

$$
[\mathbf{K}]_{ij}=\kappa(\mathbf{x}_i,\mathbf{x}_j)
$$

를 구성하고, 변환 행렬 $\mathbf{W}$를 $\mathbf{W}=\mathbf{\Phi}^{\top}\mathbf{B}$ 꼴로 두어 최종 feature를

$$
\mathbf{Z}=\mathbf{\Phi}\mathbf{W}=\mathbf{K}^{\top}\mathbf{B}
$$

처럼 표현한다. 이렇게 하면 무한 차원 RKHS에서도 계산이 가능하다.

논문은 네 가지 scatter를 모두 $\mathbf{B}$에 대한 trace form으로 바꾼다. 핵심적으로,

* total scatter는 $\operatorname{Tr}\left(\frac{1}{n}\mathbf{B}^{\top}\mathbf{K}\mathbf{K}\mathbf{B}\right)$,
* domain scatter는 $\operatorname{Tr}(\mathbf{B}^{\top}\mathbf{K}\mathbf{L}\mathbf{K}\mathbf{B})$,
* between-class scatter는 $\operatorname{Tr}(\mathbf{B}^{\top}\mathbf{P}\mathbf{B})$,
* within-class scatter는 $\operatorname{Tr}(\mathbf{B}^{\top}\mathbf{Q}\mathbf{B})$

형태로 정리된다.

여기서 $\mathbf{L}$은 도메인 구조를 반영하는 계수 행렬이고, $\mathbf{P}$와 $\mathbf{Q}$는 각각 class centroid separation과 class compactness를 반영하는 행렬이다.

### 3.5 최종 목적함수

SCA의 핵심 목적은 좋은 scatter는 크게, 나쁜 scatter는 작게 만드는 것이다. 논문은 이를 개념적으로 다음과 같은 비율로 쓴다.

$$
\sup \frac{{\text{total scatter}}+{\text{between-class scatter}}} {{\text{domain scatter}}+{\text{within-class scatter}}}
$$

이 식은 매우 직관적이다. 분류에 유리한 방향인 전체 분산 유지와 클래스 간 분리 증가는 분자에 들어가고, 도메인 차이와 클래스 내부 퍼짐은 분모에 들어간다.

실제 구현을 위해 하이퍼파라미터 $\beta$와 $\delta$를 넣으면, 최적화 문제는 다음과 같이 정리된다.

$$
\operatorname*{argmax}_{\mathbf{B}\in\mathbb{R}^{n\times k}}
\frac{
\operatorname{Tr}\big(\mathbf{B}^{\top}(\frac{1-\beta}{n}\mathbf{K}\mathbf{K}+\beta\mathbf{P})\mathbf{B}\big)
}{
\operatorname{Tr}\big(\mathbf{B}^{\top}(\delta\mathbf{K}\mathbf{L}\mathbf{K}+\mathbf{Q}+\mathbf{K})\mathbf{B}\big)
}
$$

여기서 $\beta$는 total scatter와 between-class scatter 사이의 trade-off를 조절하고, $\delta$는 domain scatter의 가중치를 조절한다. 분모의 $\mathbf{K}$ 항은 정규화 및 해의 scale control 역할을 하는 것으로 볼 수 있다.

이 비율형 목적함수는 제약식 형태로 바꾼 뒤 Lagrangian을 세우면 generalized eigenvalue problem으로 귀결된다.

$$
\left(\frac{1-\beta}{n}\mathbf{K}\mathbf{K}+\beta\mathbf{P}\right)\mathbf{B}^{*} = \left(\delta\mathbf{K}\mathbf{L}\mathbf{K}+\mathbf{K}+\mathbf{Q}\right)\mathbf{B}^{*}\mathbf{\Lambda}
$$

즉, 최적의 projection 방향은 위 식의 leading generalized eigenvectors로 주어진다. 이 점이 논문의 계산 효율성 주장과 직접 연결된다.

### 3.6 학습 및 추론 절차

논문이 제시한 알고리즘 흐름은 다음과 같다.

먼저 모든 학습 샘플로 kernel matrix $\mathbf{K}$를 만들고, 도메인 정보와 레이블 정보를 이용해 $\mathbf{L}$, $\mathbf{P}$, $\mathbf{Q}$를 구성한다. 그다음 kernel centering을 수행하고, generalized eigendecomposition으로 최적의 $\mathbf{B}^{*}$와 고유값 $\mathbf{\Lambda}$를 구한다. 마지막으로 target sample에 대해 학습 샘플과의 kernel matrix $\mathbf{K}^t$를 계산한 뒤 feature를

$$
\mathbf{Z}^{t}=\mathbf{K}^{t\top}\mathbf{B}^{*}\mathbf{\Lambda}^{-\frac{1}{2}}
$$

로 추출한다.

이 특징은 이후 별도의 classifier에 입력될 수 있다. 논문의 제공 텍스트에서는 최종 classifier 종류가 이 부분에서 명시적으로 서술되지 않았다. 따라서 “SCA가 feature extractor 역할을 하고 그 위에 분류기를 얹는다” 정도까지만 확실히 말할 수 있다.

### 3.7 이론적 해석

논문은 domain adaptation의 경우 SCA에 대한 theoretical bound를 제시한다고 소개한다. 구체적으로 domain scatter가 discrepancy distance를 제어할 수 있으며, discrepancy distance는 domain adaptation generalization bound와 관련된다고 설명한다. 다만 제공된 추출 텍스트에는 bound의 정식 진술과 증명 전개가 포함되어 있지 않다. 따라서 이론 결과의 방향성은 알 수 있지만, 정확한 정리의 형태나 가정, 상수 항까지는 여기서 복원할 수 없다.

## 4. 실험 및 결과

제공된 추출 텍스트에 따르면 저자들은 benchmark cross-domain object recognition datasets에서 광범위한 실험을 수행했고, SCA가 여러 state-of-the-art 방법보다 **훨씬 빠르면서도**, domain adaptation과 domain generalization 모두에서 **state-of-the-art 혹은 경쟁력 있는 정확도**를 달성했다고 주장한다. 또한 abstract와 introduction에서 일관되게 “속도와 정확도 모두 우수하다”는 점을 강조한다.

비교 대상으로는 domain adaptation 측면에서 TCA, SSTCA, TJM, SA, GFK, CORAL, DIP, TSC 등 feature transformation 기반 방법들이 문맥상 중요한 기준선으로 언급되고, domain generalization 측면에서는 DICA, Undo-Bias, UML, LRE-SVM 등의 관련 방법들이 선행연구로 정리된다. 그러나 현재 제공된 텍스트는 실험 섹션 본문과 결과 표, 데이터셋 이름, 정확한 수치, 통계적 유의성 분석, ablation 설정을 포함하지 않는다. 따라서 어떤 데이터셋에서 어떤 baseline을 얼마나 이겼는지, 그리고 어느 설정에서 특히 강했는지까지는 정확히 기술할 수 없다.

그럼에도 불구하고 실험 설계의 의도는 분명하다. 논문은 두 문제군을 모두 다뤄야 한다고 선언했기 때문에, 실험은 적어도 다음을 검증하려는 구조라고 이해할 수 있다. 첫째, domain mismatch를 줄이는 데 실제로 효과가 있는가. 둘째, class discrimination을 유지하거나 향상시키는가. 셋째, 기존 방법보다 학습 단계가 빠른가. 넷째, adaptation뿐 아니라 generalization에서도 같은 알고리즘이 작동하는가. 이 네 가지는 방법론의 설계 목표와 정확히 대응한다.

실험 결과의 중요성은 특히 “정확도만 높은 느린 방법”이 아니라 “빠르고 정확한 방법”을 지향한다는 점에 있다. 논문은 여러 기존 방법들이 계산적으로 무겁고 실시간 또는 빠른 학습이 필요한 상황에 부적합할 수 있다고 비판했기 때문에, SCA의 실험적 성과는 단순한 성능 경쟁이 아니라 계산 효율성까지 포함한 실용성 주장으로 읽어야 한다.

다만 엄밀하게 말하면, 제공된 텍스트만으로는 다음 사항을 확인할 수 없다. 어떤 kernel이 실험에서 최종적으로 사용되었는지, 하이퍼파라미터 $\beta$, $\delta$, $\sigma$, 차원 $k$를 어떻게 선택했는지, classifier가 무엇인지, adaptation/generalization별 세부 프로토콜이 무엇인지, deep feature 위에서 실험했는지 handcrafted feature 위에서 실험했는지 등은 여기서 단정할 수 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 통합 방식이 매우 깔끔하다**는 점이다. Domain adaptation과 domain generalization을 별개의 기술로 보지 않고, “도메인 차이는 줄이고, 클래스 구조는 유지 또는 강화하는 representation을 배운다”는 하나의 원리로 묶는다. 이는 개념적으로도 아름답고, 실제 설계도 단순해진다.

둘째 강점은 **해석 가능성**이다. SCA는 deep neural network처럼 강력하지만 불투명한 방법이 아니라, 각 항이 무엇을 의미하는지 명확하다. Total scatter는 정보 보존, domain scatter는 distribution alignment, within-class scatter는 클래스 응집도, between-class scatter는 클래스 분리도를 나타낸다. 따라서 목적함수의 각 부분이 왜 필요한지 설명하기 쉽다.

셋째 강점은 **기존 이론 및 방법과의 연결성**이다. PCA, MMD, distributional variance, Fisher LDA와의 관계를 명시적으로 보여줌으로써, SCA가 기존 방법들의 공통 구조를 포괄하는 프레임워크라는 인상을 준다. 새로운 방법을 제시하되, 그것이 기존 지식과 어떻게 연결되는지 보여주는 점은 학술적으로 설득력이 크다.

넷째 강점은 **계산 효율성**이다. generalized eigenvalue problem으로 환원된다는 점은 구현과 해석 모두에서 장점이 있다. 논문도 이것이 Kernel PCA 수준의 시간 복잡도와 유사하다고 주장한다. 대규모 반복 최적화가 필요한 방법들보다 실용적일 가능성이 있다.

반면 한계도 분명하다. 첫째, 이 방법은 기본적으로 **kernel method 기반**이므로, 샘플 수가 매우 클 때는 kernel matrix $\mathbf{K}\in\mathbb{R}^{n\times n}$ 저장과 분해 자체가 부담이 될 수 있다. 논문이 기존 최적화보다 빠르다고 주장하더라도, kernel 기반 방법의 메모리/시간 복잡도 한계는 남아 있다.

둘째, 도메인 정렬이 주로 mean embedding 수준의 scatter로 표현되므로, **고차 모멘트나 더 복잡한 구조적 차이**를 얼마나 충분히 포착하는지는 제한적일 수 있다. 특히 class-conditional shift가 복잡하거나 multimodal alignment가 중요한 상황에서는 평균 기반 정렬만으로 충분하지 않을 수 있다. 물론 within/between class scatter가 일부 보완을 하지만, 논문 텍스트만 보면 정렬의 핵심은 mean-based discrepancy에 가깝다.

셋째, supervised class scatter 항은 source labels에 의존한다. 따라서 source label이 noisy하거나 클래스 불균형이 심한 경우, class separation을 강조하는 항이 오히려 representation을 왜곡할 가능성도 있다. 제공된 텍스트에는 이런 강건성 분석이 있는지 나타나 있지 않다.

넷째, domain generalization 설정에서는 target data를 전혀 보지 않기 때문에, 여러 source domain에서 공통 불변 구조를 잡아내는 것이 핵심인데, scatter 기반 정렬이 얼마나 강한 out-of-domain generalization을 보장하는지는 추후 검증이 필요하다. 논문은 이론 bound를 domain adaptation에 대해서는 언급하지만, domain generalization에 대한 동등한 이론적 보장은 현재 제공된 텍스트에서는 보이지 않는다.

비판적으로 보면, SCA의 강점은 “여러 목적을 하나의 선형대수 프레임워크로 통합한 것”에 있고, 약점은 “그 통합이 평균적 통계량과 kernel 선형변환에 머무는 것”에 있다. 즉, 구조는 매우 정교하지만 표현력 측면에서는 현대적인 깊은 비선형 적응 모델보다 제한적일 여지가 있다. 다만 이 평가는 논문이 쓰인 시기와 맥락을 함께 봐야 공정하다. 2015년 전후의 domain adaptation/generalization 문헌에서 이 정도의 통합성과 계산 효율성은 충분히 의미 있는 기여다.

## 6. 결론

이 논문은 domain adaptation과 domain generalization을 함께 다룰 수 있는 통합적인 feature learning 방법으로 **Scatter Component Analysis (SCA)**를 제안했다. 핵심은 scatter라는 단순하고 기하학적인 개념을 사용해, 전체 데이터 분산 유지, 클래스 간 분리, 클래스 내 응집, 도메인 간 정렬을 하나의 목적함수 안에 결합한 것이다. 이 목적함수는 generalized eigenvalue problem으로 풀리기 때문에 계산적으로 효율적이며, 커널 기반 방식 덕분에 비선형 표현 학습도 가능하다.

논문의 주요 기여는 세 가지로 요약할 수 있다. 첫째, scatter를 중심으로 여러 기존 개념을 통합하는 이론적 틀을 제시했다. 둘째, adaptation과 generalization 양쪽에 적용 가능한 실제 알고리즘을 설계했다. 셋째, domain scatter와 일반화 성능 사이의 이론적 연결 가능성을 제시했다.

실제 적용 측면에서 이 연구는 도메인 편향이 강한 시각 인식 문제에서 여전히 중요한 통찰을 제공한다. 특히 레이블 부족, 분포 이동, 빠른 학습이라는 조건이 동시에 중요한 환경에서는, 해석 가능하고 계산 효율적인 representation learning 방법으로서 가치가 있다. 향후 연구 관점에서는, SCA의 scatter 기반 목적을 더 강한 비선형 모델이나 deep architecture와 결합하거나, mean-level alignment를 넘어 더 풍부한 distribution structure를 반영하도록 확장하는 방향이 자연스럽다.

마지막으로 분명히 해둘 점은, 현재 분석은 사용자가 제공한 추출 텍스트에 근거한다는 것이다. 텍스트가 “Conversion to HTML had a Fatal error” 이후 중단되어 있어, 실험 섹션의 정량 결과표와 이론 bound의 상세 정리, 결론 원문은 확인되지 않는다. 따라서 위 보고서는 논문의 핵심 아이디어와 수식 구조, 알고리즘 설계를 충실히 재구성한 것이며, 제공되지 않은 수치나 세부 프로토콜은 추측하지 않았다.
