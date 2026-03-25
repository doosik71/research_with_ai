# A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects

* **저자**: Thomas Bendokat, Ralf Zimmermann, P.-A. Absil
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2011.13699](https://arxiv.org/abs/2011.13699)

## 1. 논문 개요

이 논문은 Grassmann manifold $\mathrm{Gr}(n,p)$, 즉 $\mathbb{R}^n$ 안의 $p$차원 선형 부분공간들의 집합에 대해, 실제 계산에 바로 사용할 수 있는 기하학 공식과 알고리즘을 체계적으로 정리한 “핸드북” 성격의 논문이다. 논문의 목적은 추상적인 미분기하학 설명을 넘어서, 행렬 기반 알고리즘 설계에 필요한 표현 방식, 접공간, metric, exponential map, logarithm map, parallel transport, curvature, cut locus, conjugate locus 등을 하나의 일관된 틀에서 제공하는 데 있다.

연구 문제는 크게 두 가지다. 첫째, Grassmann manifold를 다루는 기존 문헌이 basis/ONB(orthonormal basis) 관점과 projector 관점으로 나뉘어 있어, 서로 연결이 약하고 구현 관점에서도 공식이 흩어져 있다는 점이다. 둘째, 실제 응용에서 중요한 logarithm map, cut locus, parallel transport, exponential derivative 같은 구성요소가 일부 관점에서만 알려져 있거나 수치적으로 덜 안정적이라는 점이다.

이 문제가 중요한 이유는 Grassmann manifold가 단순한 순수수학 대상이 아니라, subspace estimation, subspace tracking, low-rank optimization, dynamic low-rank approximation, model reduction, computer vision, machine learning 등 매우 넓은 응용 분야의 공통 기반이기 때문이다. 특히 고차원 데이터에서 부분공간 자체를 변수로 두고 최적화하거나 보간하는 문제에서는, manifold 위의 기하학 연산을 정확하고 효율적으로 계산하는 것이 알고리즘 성능을 좌우한다.

이 논문은 설명 논문(expository work)이면서도, 단순 정리 수준을 넘어 몇 가지 새로운 기여를 포함한다. 대표적으로 Grassmannian의 Riemannian logarithm을 계산하는 수정 알고리즘, cut locus와 conjugate locus의 보다 명시적인 기술, projector 관점에서의 parallel transport 공식, exponential map의 도함수 공식, 그리고 한 점에서 사라지는 Jacobi field 공식 등이 제시된다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 Grassmann manifold를 하나의 고정된 표현으로만 보지 않고, 서로 연결되는 여러 표현을 동시에 다루는 것이다. 구체적으로는 orthogonal group $\mathrm{O}(n)$의 quotient space 관점, Stiefel manifold $\mathrm{St}(n,p)$의 orthonormal basis 관점, 그리고 orthogonal projector $P=UU^T$ 관점을 모두 함께 사용한다. 논문은 이 세 관점이 실제로 같은 기하학을 서로 다른 좌표계에서 기술한 것임을 보여주고, 각 관점의 장점을 결합한다.

이 설계가 중요한 이유는 계산량과 해석 편의가 관점마다 다르기 때문이다. projector 관점은 개념적으로 명확하고 대칭 구조, curvature, cut locus 같은 기하학적 성질을 다루기 좋다. 반면 ONB/Stiefel 관점은 $n \times p$ 크기의 행렬만 쓰므로, $n \gg p$인 실제 계산에서 훨씬 효율적이다. 논문은 이러한 차이를 “하나를 버리고 하나를 택하는 문제”가 아니라, quotient structure를 통해 서로 오갈 수 있는 동일한 구조로 본다.

기존 접근과의 차별점은, 단순히 공식을 나열하는 것이 아니라 서로 다른 관점의 식을 대응시키고, 실제 알고리즘 관점에서 어떤 형태가 더 유리한지까지 설명한다는 점이다. 예를 들어 geodesic, logarithm, parallel transport를 projector 관점과 ONB 관점 모두에서 제시하며, 계산 복잡도가 $\mathcal{O}(np^2)$로 유지되도록 식을 정리한다. 또한 cut locus 위의 점까지 tangent space로 보낼 수 있도록 logarithm 알고리즘을 확장했다는 점이 눈에 띈다.

또 하나의 핵심 아이디어는 principal angles를 중심 도구로 삼는 것이다. 두 부분공간 사이의 각도 정보가 geodesic distance, cut locus, shortest geodesic의 비유일성, conjugate locus의 구조를 모두 설명하는 핵심 변수로 사용된다. 이 덕분에 추상적 개념이 “부분공간 사이 각도”라는 직관적인 양으로 바뀌며, 수치 알고리즘과 기하학 해석이 연결된다.

## 3. 상세 방법 설명

### 3.1 Grassmann manifold의 표현과 quotient 구조

논문은 Grassmann manifold를 먼저 orthogonal projector들의 집합으로 정의한다.

$$
\mathrm{Gr}(n,p)={P\in \mathbb{R}^{n\times n}\mid P^T=P,; P^2=P,; \operatorname{rank}(P)=p}.
$$

여기서 $P$는 어떤 $p$차원 부분공간으로의 orthogonal projection이다. 만약 $U\in\mathrm{St}(n,p)$가 그 부분공간의 orthonormal basis라면 $P=UU^T$이다. 따라서 하나의 subspace는 여러 개의 $U$로 표현되지만, projector는 유일하다.

Stiefel manifold는 다음과 같이 정의된다.

$$
\mathrm{St}(n,p)={U\in\mathbb{R}^{n\times p}\mid U^TU=I_p}.
$$

Grassmann manifold는 이를 다시 $\mathrm{O}(p)$로 quotient한 공간으로 볼 수 있다. 즉, $U$와 $UR$은 같은 부분공간을 나타내므로 같은 Grassmann point이다.

$$
[U]={UR\mid R\in \mathrm{O}(p)}.
$$

더 위에서는 orthogonal group $\mathrm{O}(n)$로부터도 quotient를 만들 수 있다. 이로부터

$$
\mathrm{Gr}(n,p)\cong \mathrm{O}(n)/(\mathrm{O}(p)\times \mathrm{O}(n-p))
$$

라는 구조가 얻어진다. 이 quotient 구조가 중요한 이유는, tangent space를 vertical/horizontal decomposition으로 나누고, Grassmann의 tangent vector를 더 큰 공간의 horizontal lift로 계산할 수 있게 해주기 때문이다.

### 3.2 접공간과 horizontal lift

Projector 관점에서 $P\in\mathrm{Gr}(n,p)$에서의 tangent vector $\Delta$는 대칭행렬이면서

$$
\Delta P + P\Delta = \Delta
$$

를 만족하는 행렬로 특징지어진다. 논문은 이 조건이 tangent space의 핵심 성질이라고 정리한다. 직관적으로는 $\Delta$가 부분공간 내부를 움직이는 것이 아니라, 부분공간과 그 직교여공간 사이를 섞는 방향이라는 뜻이다.

또한 $\Delta$는 commutator 형태로도 쓸 수 있다.

$$
\Delta = [\Omega, P]
$$

여기서 $\Omega$는 특정한 skew-symmetric matrix이다. 이 표현은 projector 관점에서 geodesic, exponential, parallel transport를 유도할 때 매우 중요하다.

Stiefel 관점에서는 tangent space가

$$
T_U\mathrm{St}(n,p)={D\in\mathbb{R}^{n\times p}\mid U^TD=-D^TU}
$$

로 주어지고, 여기서 다시 vertical part와 horizontal part로 나뉜다. Grassmann의 tangent vector에 해당하는 것은 horizontal part이며,

$$
\operatorname{\mathsf{Hor}}_U\mathrm{St}(n,p)={D\in\mathbb{R}^{n\times p}\mid U^TD=0}
$$

이다. projector tangent vector $\Delta$의 $U$로의 horizontal lift는 매우 간단하게

$$
\Delta_U^{\mathsf{hor}}=\Delta U
$$

로 계산된다. 이 식은 논문 전체에서 매우 자주 사용되는 연결고리다.

### 3.3 Riemannian metric

Grassmann manifold의 canonical metric은 quotient 구조에서 유도되며, projector 관점에서는 Euclidean trace metric의 절반과 일치한다.

$$
g_P^{\mathrm{Gr}}(\Delta_1,\Delta_2)=\frac{1}{2}\operatorname{tr}(\Delta_1\Delta_2).
$$

이 식은 매우 중요하다. 왜냐하면 추상적인 Riemannian metric이 실제로는 단순한 trace 연산으로 계산되기 때문이다. 논문은 이것이 ONB lift 관점에서는

$$
g_P^{\mathrm{Gr}}(\Delta_1,\Delta_2)=\operatorname{tr}((\Delta_{1,U}^{\mathsf{hor}})^T\Delta_{2,U}^{\mathsf{hor}})
$$

와 같음을 보인다. 따라서 Stiefel 쪽에서는 작은 크기의 $n\times p$ 행렬 연산만으로 metric을 계산할 수 있다.

### 3.4 Riemannian connection과 gradient

Grassmannian을 symmetric matrix 공간의 embedded submanifold로 보면, ambient space에서 미분한 뒤 tangent space로 사영해서 Levi-Civita connection을 얻을 수 있다. projector 관점에서 tangent projection은

$$
\Pi_{T_P\mathrm{Gr}}(S)=(I_n-P)SP+PS(I_n-P)
$$

로 주어진다. 그래서 vector field $X$의 방향미분을 $Y$ 방향에서 계산한 뒤 위 projection을 취하면 Riemannian connection이 된다.

Stiefel 관점에서는 horizontal space projection

$$
\Pi_{\operatorname{\mathsf{Hor}}_U\mathrm{St}}(Z)=(I_n-UU^T)Z
$$

를 사용한다. 이 방식은 gradient 계산에서도 쓰인다. 즉, Euclidean gradient를 구한 뒤 tangent/horizontal projection을 적용하면 Grassmann gradient를 얻는다. 논문은 이 내용을 기존 문헌과 연결해 정리하지만, 여기서는 새 이론이라기보다 계산 체계를 통합하는 역할이 크다.

### 3.5 Exponential map과 geodesic

Grassmannian의 geodesic은 quotient 위의 geodesic을 내려보내어 얻는다. projector 관점의 핵심 식은 다음이다.

$$
\operatorname{Exp}_P^{\mathrm{Gr}}(\Delta)=\exp_m([\Delta,P]),P,\exp_m(-[\Delta,P]).
$$

이 식은 개념적으로 깔끔하지만 $n\times n$ matrix exponential이 필요해서 큰 문제에서는 비효율적이다. 그래서 논문은 ONB/Stiefel 관점의 계산식도 함께 제시한다.

$\Delta_U^{\mathsf{hor}}$의 thin SVD를

$$
\Delta_U^{\mathsf{hor}}=\hat{Q}\Sigma V^T
$$

라고 하면 geodesic은

$$
\operatorname{Exp}_P^{\mathrm{Gr}}(t\Delta) = [UV\cos(t\Sigma)V^T+\hat{Q}\sin(t\Sigma)V^T+UV_\perp V_\perp^T]
$$

로 표현된다. 사실상 tangent direction의 singular values가 geodesic의 principal angle 속도를 결정하는 구조다. 이 식의 장점은 $n\gg p$일 때 훨씬 저렴하다는 것이다.

### 3.6 Exponential map의 도함수와 Jacobi field

논문은 geodesic 자체뿐 아니라 exponential map의 미분도 계산한다. 이는 Hermite interpolation이나 Jacobi field 계산에 필요하다. 방법은 $\Delta+t\tilde{\Delta}$의 lift에 대한 SVD가 어떻게 변하는지를 미분하는 것이다. 다만 singular value가 서로 달라야 한다는 가정이 필요한 경우가 있고, 이를 완화하기 위해 QR decomposition 기반의 대안적 계산법도 제안한다.

핵심 메시지는, exponential derivative를 explicit하게 쓸 수 있어 geodesic variation과 Jacobi field를 행렬 수준에서 계산할 수 있다는 점이다. 이는 일반적인 Grassmann optimization 논문에서는 자주 생략되는 부분인데, 이 논문은 이를 꽤 자세히 다룬다.

### 3.7 Parallel transport

Parallel transport는 한 점의 tangent vector를 geodesic을 따라 다른 점의 tangent space로 옮기는 연산이다. 논문은 projector 관점에서 다음의 간단한 공식을 제시한다.

$$
\mathbb{P}_{\Delta}(\operatorname{Exp}_P^{\mathrm{Gr}}(t\Gamma)) = \exp_m(t[\Gamma,P]), \Delta, \exp_m(-t[\Gamma,P]).
$$

이 식은 projector 관점에서의 명시적 parallel transport 공식이라는 점에서 논문의 원저 기여 중 하나다. ONB 관점에서는 SVD 기반의 더 계산 효율적인 식도 유도되며, 전체 연산량을 $\mathcal{O}(np^2)$ 정도로 유지한다.

### 3.8 Curvature

논문은 Grassmann manifold가 symmetric space임을 elementary한 방식으로 보인 뒤, sectional curvature를 explicit한 trace 식으로 제시한다. 대표적인 식은

$$
K_P(\Delta_1,\Delta_2) = 4 \, \frac{\operatorname{tr}(\Delta_1^2\Delta_2^2)-\operatorname{tr}((\Delta_1\Delta_2)^2)} {\operatorname{tr}(\Delta_1^2)\operatorname{tr}(\Delta_2^2)-(\operatorname{tr}(\Delta_1\Delta_2))^2}.
$$

또는 commutator norm으로도 쓸 수 있다.

$$
K_P(\Delta_1,\Delta_2) = 2 \, \frac{|[\Delta_1,\Delta_2]|_F^2} {|\Delta_1|_F^2|\Delta_2|_F^2-\langle \Delta_1,\Delta_2\rangle_0^2}.
$$

이 식은 curvature가 언제 0이 되는지, 언제 큰지 직관을 준다. 두 tangent direction이 commute하면 curvature가 0이 될 수 있고, 특정 경우에는 최대값 2까지 도달한다. 또한 $\mathrm{Gr}(n,1)$이나 $\mathrm{Gr}(n,n-1)$의 경우 curvature가 상수 1이라는 점도 다시 확인된다.

### 3.9 Cut locus와 logarithm map

논문의 가장 실질적인 신규 기여는 logarithm map 계산이다. Grassmannian에서 $P$와 $F$ 사이의 principal angles 중 하나라도 $\pi/2$이면 $F$는 $P$의 cut locus에 속한다. 이는 $U^TY$의 rank가 $p$보다 작아지는 경우와 같다.

$$
\operatorname{Cut}_P={F=YY^T\in\mathrm{Gr}(n,p)\mid \operatorname{rank}(U^TY)<p}.
$$

또한 geodesic distance는 principal angles $\theta_i$의 $\ell_2$ norm이다.

$$
\operatorname{dist}(\mathcal{U},\widetilde{\mathcal{U}}) = \left(\sum_{i=1}^p \theta_i^2\right)^{1/2}.
$$

논문은 기존 logarithm 알고리즘이 cut locus 근처에서 수치적으로 불안정하고, $(U^TY)^{-1}$ 같은 연산을 필요로 한다는 문제를 지적한다. 이를 개선한 것이 Algorithm 3이다.

Algorithm 3의 흐름은 다음과 같다. 먼저 $Y^TU$에 SVD를 적용해 Procrustes 정렬을 수행한다. 이를 통해 $Y$의 대표자를 $U$에 가장 잘 맞도록 회전시킨 $Y_*$를 만든다. 그다음 $(I-UU^T)Y_*$의 compact SVD를 구하고, 그 singular values에 element-wise $\arcsin$을 적용해서 tangent singular values를 복원한다. 최종적으로

$$
\Delta_U^{\mathsf{hor}}=\hat{Q}\Sigma R^T
$$

를 얻는다. 논문은 이 방법이 cut locus 밖에서는 유일한 Riemannian logarithm을 주고, cut locus 위에서는 shortest geodesic에 대응하는 가능한 tangent vector들 중 하나를 반환함을 보인다.

여기서 중요한 점은, SVD의 비유일성이 cut locus에서의 geodesic 비유일성과 정확히 대응한다는 것이다. 즉, 알고리즘의 모호성은 단순한 수치 오류가 아니라 manifold 기하학의 실제 구조를 반영한다.

### 3.10 Conjugate locus

논문은 conjugate locus도 principal angles로 설명한다. $p\le n/2$일 때, 두 부분공간 사이에 같은 principal angle이 두 개 이상 있거나, $p<n/2$에서 0 principal angle이 하나라도 있으면 conjugate locus에 속한다는 정리를 제시한다. 이 부분은 기존 문헌의 불완전한 설명을 보완하는 기여다.

핵심적으로, cut locus와 conjugate locus는 동일하지 않다. principal angle 하나만 $\pi/2$인 점은 cut locus에는 있지만 conjugate locus에는 없을 수 있다. 반대로 repeated principal angles는 shortest geodesic의 비유일성이 없어도 conjugate point를 만들 수 있다. 이 구분을 명확히 한 점이 이 논문의 이론적 가치다.

## 4. 실험 및 결과

이 논문은 일반적인 machine learning 논문처럼 대규모 benchmark 실험을 수행하는 형태는 아니다. 오히려 수학적 정리와 알고리즘 공식화가 중심이며, 실험은 주로 제안한 logarithm 알고리즘의 수치 안정성을 확인하는 역할을 한다. 따라서 “실험 및 결과”를 읽을 때도 분류 정확도나 SOTA 비교가 아니라, 알고리즘이 cut locus 근처에서 얼마나 잘 동작하는지를 보는 것이 핵심이다.

대표적인 수치 실험은 Section 5.3에서 제시된다. 저자들은 $U\in\mathrm{St}(1000,200)$인 무작위 subspace representative와, 최대 singular value가 $\pi/2$가 되도록 설정한 random horizontal tangent vector를 사용해, cut locus에 점점 가까워지는 subspace $U_1(\tau)$를 생성한다. 여기서 $\tau\to 0$이면 정확히 cut locus에 접근한다.

그 후 두 알고리즘을 비교한다. 하나는 논문이 제안한 new log algorithm이고, 다른 하나는 기존 standard log algorithm이다. 계산된 logarithm을 다시 exponential map에 넣어 복원한 subspace와 원래 subspace 사이의 distance error를 측정한다. 이 오차는 principal angles 기반의 subspace distance로 계산된다.

핵심 결과는 Figure 5.2의 정성적 결론이다. 제안한 new log algorithm은 cut locus에 매우 가까운 상황에서도 안정적으로 작은 오차를 유지한다. 반면 standard log algorithm은 cut locus 근처로 갈수록 오차가 빠르게 증가하며, 심지어 horizontal projection을 추가해도 성능 열화가 나타난다. 논문은 이 결과를 통해 제안 알고리즘이 이론적으로만 확장된 것이 아니라 실제 수치 계산에서도 더 안정적임을 보여준다.

다만 이 실험은 “정확히 어떤 평균 오차가 얼마였다”는 식의 풍부한 표를 제공하지는 않는다. 실험의 목적이 대규모 empirical validation이 아니라, cut locus 근처에서의 극단 상황(extreme-case behavior)을 검증하는 것이기 때문이다. 따라서 결과 해석도 “응용 task 성능 개선”보다는 “기하학 연산의 신뢰도 향상”에 초점을 맞춰야 한다.

또한 논문에는 FLOP count 표도 포함되어 있다. 예를 들어 Stiefel representative 기반의 Riemannian exponential, parallel transport, logarithm이 대체로 $\mathcal{O}(np^2)$ 복잡도 안에서 계산되도록 정리되어 있다. 이는 이론만이 아니라 실제 큰 $n$, 작은 $p$ 환경을 염두에 둔 설계임을 보여준다. 특히 logarithm의 경우 대략 $\sim 8np^2 + 2np + p^3 + p^2 + 2p$ 수준의 연산량으로 정리되어 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 Grassmann manifold의 기하학을 단순한 정의 모음이 아니라, 실제 계산 가능한 matrix formula의 체계로 정리했다는 점이다. 많은 논문이 Grassmannian을 optimization 배경 정도로만 간단히 소개하는 반면, 이 논문은 tangent space, metric, connection, exponential, logarithm, curvature, parallel transport, cut locus, conjugate locus, local coordinates를 한 문맥에서 다룬다. 따라서 연구자 입장에서는 참고서 역할을 하기에 매우 유용하다.

두 번째 강점은 표현 간의 다리 역할이다. projector 관점은 기하학 해석에 강하고, Stiefel/ONB 관점은 계산 효율이 좋다. 논문은 이 둘을 quotient structure와 horizontal lift로 연결해, 어느 식이 어느 상황에서 더 좋은지 분명하게 보여준다. 이는 관련 문헌이 분절되어 있다는 저자들의 문제의식을 잘 해결한다.

세 번째 강점은 신규 기여의 실질성이다. 특히 modified Grassmann logarithm 알고리즘은 단순히 수식을 다시 쓴 것이 아니라, cut locus까지 포함하는 확장성과 수치적 안정성이라는 장점을 가진다. parallel transport의 projector 공식과 conjugate locus의 더 완전한 설명도 이론적으로 의미가 크다.

네 번째 강점은 설명의 “algorithmic fitness”다. 논문은 추상적인 기하학 정리를 가능한 한 선형대수와 행렬 해석 수준에서 풀어낸다. 그래서 미분기하학 전문 배경이 없는 독자도, 어느 정도 선형대수 지식이 있다면 따라갈 수 있게 구성되어 있다. 특히 principal angles, SVD, QR decomposition을 중심 도구로 삼아 실제 구현과 이론을 자연스럽게 연결한다.

하지만 한계도 있다. 가장 먼저, 이 논문은 새로운 학습 모델이나 응용 시스템을 제안하는 논문이 아니므로, 실용적 가치가 간접적이다. 즉, 논문 자체가 어떤 computer vision task의 성능을 올려준다는 식의 직접적 메시지는 없다. 대신 다양한 알고리즘의 수학적 기반을 제공하는 역할에 가깝다.

둘째, 내용이 매우 방대하고 수식이 많아, 입문자가 처음부터 끝까지 따라가기는 쉽지 않다. 저자들이 “elementary” derivation을 지향했다고는 하지만, 실제로는 Lie group, quotient manifold, curvature tensor, Jacobi field 등 고급 개념이 계속 등장한다. 따라서 선형대수 친화적으로 설명된다는 장점과 별개로, 독해 난도는 높은 편이다.

셋째, numerical experiment는 logarithm 근처의 제한된 사례만 검증한다. 제안한 기하학 도구들이 실제 optimization, interpolation, model reduction, computer vision pipeline에서 얼마나 큰 차이를 만드는지는 이 논문에서 직접 보여주지 않는다. 물론 이는 논문의 목표가 handbook이라는 점을 고려하면 자연스러운 한계이지만, 응용 연구자 입장에서는 후속 검증이 필요하다.

넷째, 일부 결과는 특정 가정에 기대고 있다. 예를 들어 exponential derivative의 한 공식은 singular values가 서로 다르고 0이 아니어야 한다는 가정 아래 제시된다. QR 기반 대안도 완전한 만능은 아니고, rank-deficient에 가까운 경우 불안정성이 남을 수 있다고 논문 스스로 밝힌다. 즉, 모든 수치 난점을 완전히 제거한 것은 아니다.

비판적으로 보면, 이 논문은 “핸드북”으로서 매우 성공적이지만, 처음 접하는 독자가 전체 구조를 빠르게 파악하기엔 다소 장황하다. 또 어떤 공식이 실제 구현에서 가장 추천되는지에 대한 우선순위는 독자가 스스로 판단해야 하는 부분이 있다. 그럼에도 불구하고, Grassmann geometry를 연구하거나 구현하는 사람에게는 장기적으로 큰 가치를 주는 논문이다.

## 6. 결론

이 논문은 Grassmann manifold의 기본 기하학과 계산 공식을 projector 관점, Stiefel/ONB 관점, orthogonal group quotient 관점을 연결해 체계적으로 정리한 종합 참고서다. 핵심 기여는 단순한 정리보다 더 넓다. Riemannian exponential, logarithm, connection, gradient, parallel transport, sectional curvature, local coordinates, cut locus, conjugate locus를 모두 행렬 기반으로 서술했고, 특히 modified logarithm algorithm, projector 관점의 parallel transport 공식, conjugate locus의 더 완전한 기술 등은 새로운 연구 기여로 제시된다.

실제 적용 측면에서 이 연구는 manifold optimization, subspace interpolation, model reduction, low-rank methods, computer vision, statistical learning 등에서 공통 기반 도구로 쓰일 가능성이 높다. 특히 $n\gg p$ 환경에서 $\mathcal{O}(np^2)$ 수준의 계산을 염두에 두고 공식을 정리했다는 점은 대규모 응용에 중요하다.

향후 연구 관점에서는 두 방향이 자연스럽다. 하나는 이 논문에서 정리한 기하학 도구들을 실제 응용 알고리즘에 끼워 넣어 성능과 안정성 개선을 실증하는 것이다. 다른 하나는 complex Grassmannian, generalized manifold, 또는 보다 어려운 singular setting으로 이 결과를 확장하는 것이다. 요약하면, 이 논문은 Grassmann manifold 자체를 “이론적 대상”에서 “실제로 계산 가능한 작업 공간”으로 바꾸는 데 큰 역할을 하는 기반 논문이라고 볼 수 있다.
