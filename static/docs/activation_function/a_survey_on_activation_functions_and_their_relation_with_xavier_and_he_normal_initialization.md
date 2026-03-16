# A Survey on Activation Functions and their relation with Xavier and He Normal Initialization

## 논문 메타데이터

- 제목: A Survey on Activation Functions and their relation with Xavier and He Normal
  Initialization
- 저자: Leonid Datta
- 발표 연도: 2020
- arXiv ID: 2004.06632v1
- arXiv URL: <https://arxiv.org/abs/2004.06632v1>
- PDF: <https://arxiv.org/pdf/2004.06632v1.pdf>
- 카테고리: cs.NE
- DOI: 확인 불가

## 연구 배경 및 문제 정의

활성화 함수(activation function)는 신경망의 비선형성을 제공해 표현력을
확보하고, 역전파 시 gradient의 크기/흐름에 직접적인 영향을 준다.
가중치 초기화(weight initialization)는 학습 초기에 각 층의 분산(variance)이
지나치게 커지거나 작아지는 현상을 완화하여 학습 안정성과 수렴 속도에 큰
영향을 준다.

이 논문은 다음 질문에 초점을 둔다.

- “좋은” 활성화 함수가 갖춰야 할 성질은 무엇인가?
- 널리 쓰이는 초기화 방법인 Xavier 초기화와 He 초기화가 활성화 함수와 어떤 근본적 관계를 갖는가?

## 핵심 기여

- 활성화 함수가 갖추길 기대하는 핵심 성질(비선형성, 미분 가능성,
  연속성, boundedness, zero-centered, 낮은 계산 비용)을 정리하고 필요성을
  설명한다.
- 대표 활성화 함수 5종(sigmoid, tanh, ReLU, Leaky ReLU, PReLU)을
  정의·비교하고, 대표적 실패 모드(vanishing gradient, dead neuron)를
  연결해 설명한다.
- Xavier 초기화와 He 초기화의 아이디어(층 간 분산 보존)를 소개하고,
  활성화 함수 유형(특히 sigmoidal vs rectifier)에 따라 초기화 선택이 달라지는
  이유를 정성적으로 설명한다.

## 방법론 요약

본 논문은 “서베이(survey)”로서 새로운 알고리즘을 제안하거나 자체 대규모
실험을 수행하기보다는, 기존 문헌에서 널리 쓰이는 활성화/초기화 조합을
개념적으로 정리한다. 구성은 다음과 같다.

- (섹션 2) 활성화 함수의 기대 성질 → 대표적 문제(vanishing gradient,
  dead neuron) → 대표 활성화 함수별 성질/문제 비교
- (섹션 3) 초기화: 대칭성(symmetry) 문제와 랜덤 초기화의 필요성 → Xavier, He 초기화 소개
- (섹션 4–5) 활성화–초기화 관계에 대한 요약적 논의 및 결론

## 활성화 함수: 성질과 실패 모드

### 1) 활성화 함수가 갖추길 기대하는 성질

논문이 정리한 “중요/필요” 성질은 아래와 같다.

- **Nonlinear(비선형)**: 비선형 경계를 근사하고, 다층 네트워크가 단층
  선형 모델로 “붕괴(compress)”되는 것을 막는다.
- **Differentiable(미분 가능)** 및 **Continuous(연속)**: 역전파에서
  $\partial f/\partial z$가 필요하므로, 미분 가능성이 중요하며(일반적으로)
  연속성이 이를 뒷받침한다.
- **Bounded(유계)**: 여러 층을 통과하면서 값이 폭주(explode)하는 것을
  억제하는 데 도움이 될 수 있으나, 논문은 “중요하지만 필수는 아님”으로
  다룬다.
- **Zero-centered(0-중심)**: 출력이 양/음 모두를 가질 때, 다음 층 가중치
  업데이트가 한쪽 방향으로 편향되는 것을 완화해 학습을 돕는다는 관점이다.
- **낮은 계산 비용**: 함수와 도함수 계산이 비싸면 학습 시간/에너지 비용이 증가한다.

### 2) Vanishing gradient problem

입력 범위를 좁은 출력 범위로 “압축(compress)”하는 활성화 함수는 기울기가
작은 구간이 넓어질 수 있다. 역전파는 체인 룰로 여러 층의 미분값을 곱하므로,
작은 값이 반복 곱해지며 초기 층으로 갈수록 gradient가 0에 수렴하고 가중치
업데이트가 멈추는 현상이 발생한다(포화(saturation)).

### 3) Dead neuron problem (Dying ReLU)

특정 활성화 함수가 입력의 큰 부분을 0(또는 거의 0)으로 만들어 해당 뉴런이
출력에 기여하지 못하는 상태가 지속되는 문제다. ReLU 계열에서 특히 빈번하게
논의된다.

## 대표 활성화 함수 5종 분석 (정의·성질·장단점)

### 1) Sigmoid (logistic sigmoid)

- 정의: $f(x)=\\frac{1}{1+e^{-x}}$
- 장점: 연속/미분 가능/유계(0,1). “sigmoidal” 함수로서 단일 은닉층 + 유한
  뉴런 조건에서 보편 근사(universal approximation) 정리(Cybenko theorem)를
  인용한다.
- 단점(논문 관점):
  - 출력이 (0,1)로 **zero-centered가 아니다**.
  - 넓은 입력이 좁은 출력으로 압축되어 **vanishing gradient**가 발생하기 쉽다.
  - 지수(exp) 연산으로 계산 비용이 높다(도함수 계산은 비교적 단순).

### 2) Tanh

- 정의(논문 표기): $f(x)=\\frac{1-e^{-x}}{1+e^{-x}}$ (표준적으로는
  $\\tanh(x)$와 동치인 형태로 이해된다.)
- 장점: 출력 범위가 (-1,1)로 **zero-centered**이며, sigmoid의
  zero-centered 문제를 완화한다. sigmoidal 계열로서 보편 근사 맥락을 유지한다.
- 단점: 여전히 입력을 좁은 범위로 압축하므로 **vanishing gradient**에서
  자유롭지 않다. 계산 비용도 sigmoid와 유사하게 높다.

### 3) ReLU

- 정의: $f(x)=\\max(0,x)$
- 장점:
  - 양수 구간에서 도함수가 1로 유지되어(논문 서술) **vanishing gradient가 없다**는 주장(참고 문헌 인용)을 소개한다.
  - 계산 비용이 매우 낮고, 음수 값을 0으로 만들어 sparsity를 유도한다.
- 단점:
  - $x=0$에서 좌/우 미분이 달라 **엄밀히는 미분 불가능**(실무에서는 subgradient로 처리).
  - 음수 입력이 모두 0이 되므로 뉴런이 “꺼진 채로” 고착될 수 있어
    **dead neuron (dying ReLU)** 문제가 발생할 수 있다.

### 4) Leaky ReLU (LReLU)

- 정의(논문 예): $f(x)=\\begin{cases}0.01x & x\\le 0\\\\ x & \\text{otherwise}\\end{cases}$
- 장점: 음수 구간에 작은 기울기를 허용하여 ReLU의 dead neuron 문제를
  완화하려는 목적이다. zero-centered로 분류한다(음수 출력이 가능).
- 단점:
  - $x=0$에서 미분 불가능(좌/우 미분 불일치).
  - 음수 구간의 기울기(0.01)가 작아, 논문은 **vanishing gradient 위험이 “부분적으로” 존재**한다고 정리한다.

### 5) Parametric ReLU (PReLU)

- 정의: $f(x)=\\begin{cases}ax & x\\le 0\\\\ x & \\text{otherwise}\\end{cases}$,
  여기서 $a$는 학습되는 파라미터
- 해석: $a=0$이면 ReLU, $a=0.01$이면 LReLU로 환원되므로 “rectifier
  nonlinearity의 일반형”으로 설명한다.
- 장점: LReLU와 유사하되 음수 기울기를 학습으로 적응시킬 수 있다.
- 단점: 음수 구간의 $a$가 작으면 LReLU와 같은 이유로 **vanishing gradient 위험이 부분적으로 존재**할 수 있다고 정리한다.

### (요약) 논문이 제공하는 표 기반 비교

논문은 다음 관찰을 표로 정리한다.

- Sigmoid/tanh는 bounded·sigmoidal·연속·미분 가능하지만, sigmoid는
  zero-centered가 아니고 둘 다 vanishing gradient를 겪기 쉽다.
- ReLU/LReLU/PReLU는 계산 비용이 낮고(특히 ReLU 계열) vanishing
  gradient에 상대적으로 유리하지만, ReLU는 dead neuron 문제가 있고,
  LReLU/PReLU는 음수 구간 기울기 설정에 따라 “부분적” vanishing gradient
  위험이 있다.

## 가중치 초기화: Xavier vs He (공식과 직관)

### 1) 초기화가 필요한 이유(대칭성)

같은 값으로 모든 가중치를 초기화하면, 역전파에서 동일한 gradient를 받아
가중치들이 “그룹처럼” 움직이며 대칭성이 깨지지 않아 학습이 실패할 수 있다.
따라서 랜덤 초기화가 필요하며, 단순 랜덤 초기화는 층이 깊어질수록 분산
붕괴/폭주를 유발할 수 있어 분산을 통제하는 초기화가 등장한다.

### 2) Xavier initialization (Glorot/Bengio 계열로 소개)

논문은 Xavier 초기화를 다음 형태로 제시한다(표기는 층의 fan-in/fan-out을 사용).

$$
w \\sim U\\left[
-\\sqrt{\\frac{6}{n_{in}+n_{out}}},
\\; \\sqrt{\\frac{6}{n_{in}+n_{out}}}
\\right]
$$

핵심 아이디어는 각 층의 입력과 출력의 분산을 맞춰(분산 손실이 크지 않도록)
깊은 네트워크에서 학습이 초기부터 포화되는 문제를 완화하는 것이다. 논문은
Xavier의 이론적 전개가 “선형 구간(linear regime)” 가정과 연결된다는 논의를
함께 인용한다.

### 3) He normal initialization (rectifier nonlinearity를 위한 초기화로 소개)

논문은 He 초기화를 rectifier(ReLU/PReLU) 계열을 위한 초기화로 소개하며,
PReLU의 음수 기울기 $a$를 포함한 일반형을 다음과 같이 제시한다.

$$
w \\sim N\\left(0,\\; \\frac{2}{n_{in}(1+a^2)}\\right)
$$

직관은 ReLU 계열에서 음수 입력이 0으로 잘리며(또는 작은 기울기만 남으며)
분산이 감소하는 경향이 있으므로, 이를 보정하기 위해 Xavier보다 더 큰 분산
스케일이 필요하다는 것이다. 논문은 He 초기화가 forward와 backward 양쪽을
고려해(논문 서술) Xavier보다 더 빠르게 오차를 줄인다는 주장을 소개한다.

## 실험 설정과 결과 (논문 내에서의 “증거” 형태)

이 논문 자체는 서베이이므로 통일된 벤치마크 실험을 수행하지 않는다. 대신,
다음과 같은 문헌 기반 관찰을 근거로 든다.

- (He et al. 인용) ReLU 활성화 + Xavier 초기화로 22층 네트워크는 수렴하지만
  30층 네트워크는 수렴하지 못했다는 사례를 언급한다. 이를 “깊어질수록 입력
  분산이 지수적으로 작아질 수 있다”는 설명과 연결해 Xavier가 rectifier-깊은
  네트워크에서 실패할 수 있음을 논의한다.
- (응용 사례 인용) ReLU + He 초기화 조합이 U-Net 등 일부 SOTA 사례에서 좋은
  성능을 보였다는 관찰을 예시로 든다(논문 내 표 4는 EM segmentation challenge
  결과를 인용해 제시).

## 한계 및 향후 연구 가능성

이 서베이는 개념 정리의 성격이 강해 아래 한계가 있다.

- 다루는 활성화 함수가 5종으로 제한적이며, 2020년 전후 널리 쓰이기 시작한
  다른 함수(GELU, Swish 등)나 정규화/스킵 연결과의 상호작용은 분석 범위
  밖이다.
- 초기화-활성화의 관계를 엄밀한 수학적 유도(예: 평균/분산 전달, Jacobian
  스펙트럼)로 전개하기보다는 정성적 설명과 문헌 사례에 기대는 부분이 크다.
- LReLU/PReLU의 “부분적 vanishing gradient 위험”은 $a$의 크기와
  네트워크/데이터 분포에 따라 크게 달라질 수 있으나, 이를 정량적으로 비교하는
  실험은 제공하지 않는다.

## 실무적 또는 연구적 인사이트

- **깊은 네트워크에서 rectifier 계열(ReLU 등)을 쓰는 경우**: He 초기화(또는
  그 계열)가 분산 보존 관점에서 더 자연스럽고, Xavier를 그대로 적용하면
  깊어질수록 분산이 줄어 학습이 실패할 수 있다(논문이 인용한 22층 vs 30층
  사례).
- **sigmoidal 계열(sigmoid/tanh)을 쓰는 경우**: 포화(saturation)와 vanishing
  gradient를 특히 주의해야 하며, Xavier 초기화는 이러한 초기 포화 문제를
  완화하려는 목적과 연결된다. 다만 깊이가 커지면 sigmoid/tanh 자체의 구조적
  한계(포화)가 여전히 남는다.
- **ReLU의 dead neuron을 우려하는 경우**: LReLU/PReLU처럼 음수 구간에
  기울기를 남기는 변형은 “완전한 0 고착”을 완화할 수 있다. 단, 음수 구간
  기울기가 너무 작으면 학습 신호가 약해질 수 있어(논문 관점의 “부분적
  vanishing”) 초기화/학습률/정규화와 함께 조정해야 한다.
