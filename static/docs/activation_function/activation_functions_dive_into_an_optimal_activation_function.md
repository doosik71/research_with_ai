# Activation Functions: Dive into an optimal activation function

## 1. Paper Overview

이 논문은 **신경망에서 최적의 activation function을 찾는 문제**를 다룹니다. Activation function은 딥러닝 모델에서 비선형성을 추가하여 복잡한 패턴을 학습하도록 하는 핵심 요소입니다. 적절한 activation function의 선택은 모델의 학습 안정성, 수렴 속도, 최종 정확도에 큰 영향을 미칩니다.

저자는 하나의 고정된 activation function을 사용하는 대신 **여러 activation function의 가중합(weighted sum)을 학습 과정에서 최적화하여 사용**하는 접근을 제안합니다. 즉, ReLU, tanh, sin 등의 함수들을 조합하고, 학습 중에 각 함수의 기여도를 조정하여 **데이터와 네트워크 구조에 맞는 activation function을 자동으로 형성**하도록 합니다.

연구는 MNIST, Fashion-MNIST, KMNIST 데이터셋을 사용하여 실험을 수행하고, 네트워크의 각 층에서 어떤 형태의 activation function이 선호되는지 분석합니다.

---

## 2. Core Idea

### 핵심 아이디어

논문의 핵심 아이디어는 다음과 같습니다.

> **Activation function을 고정하지 않고 여러 activation function의 조합을 학습으로 결정한다.**

즉, 새로운 activation function을 직접 설계하는 대신 다음과 같은 형태를 사용합니다.

$$
A(x) = w_1 f_1(x) + w_2 f_2(x) + w_3 f_3(x)
$$

여기서

* $f_1(x)$ : ReLU
* $f_2(x)$ : tanh
* $f_3(x)$ : sin

그리고

* $w_1, w_2, w_3$ 는 학습 가능한 weight입니다.

이 방식의 장점은 다음과 같습니다.

* 특정 activation function에 의존하지 않음
* 데이터 및 레이어 특성에 맞는 activation 형태를 자동 학습
* activation function 자체도 모델 파라미터로 최적화

---

## 3. Detailed Method Explanation

### 3.1 Neural Network 기본 구조

논문은 먼저 일반적인 신경망 구조를 설명합니다.

입력 $x_i$ 와 출력 $y$ 관계는 다음과 같이 표현됩니다.

$$
y = f(x_i) + \epsilon
$$

여기서

* $f(x_i)$ : 모델이 학습해야 하는 함수
* $\epsilon$ : 오차

신경망의 기본 선형 모델은 다음과 같습니다.

$$
\hat{y} = w_i x_i + b
$$

벡터 형태로 확장하면

$$
\hat{y} = W^T X + B
$$

이며, 다층 신경망에서는

$$
\hat{y} = W_2^T(W_1^T X + B_1) + B_2
$$

와 같이 표현됩니다.

하지만 이 구조는 **선형 모델**이기 때문에 복잡한 함수 표현이 어렵습니다. 따라서 **activation function이 필요합니다.**

---

### 3.2 Activation Function 추가

신경망에 비선형성을 추가하면 다음과 같은 형태가 됩니다.

$$
\hat{y} = A_2(W_2^T A_1(W_1^T X + B_1) + B_2)
$$

여기서

* $A_1$
* $A_2$

는 activation function입니다.

---

### 3.3 Proposed Activation Function

논문에서 제안한 activation function은 다음과 같습니다.

$$
A(x) = \alpha_1 ReLU(x) + \alpha_2 \tanh(x) + \alpha_3 \sin(x)
$$

특징:

1. **Linear combination**
2. Weight는 학습으로 업데이트
3. 초기 weight에 따라 activation shape 변화

즉, 학습 과정에서 다음이 자동으로 결정됩니다.

* 어떤 activation이 중요한지
* 어떤 레이어에서 어떤 activation이 적합한지

---

### 3.4 Layer-wise Behavior

논문에서 관찰한 중요한 현상:

* **초기 layer**

  * ReLU 계열 선호
  * feature extraction에 유리

* **깊은 layer**

  * tanh / sinusoidal 계열 증가
  * convergence 안정성 증가

즉,

> 레이어 깊이에 따라 최적 activation 특성이 달라진다.

---

## 4. Experiments and Findings

### 4.1 Dataset

논문은 다음 3개 데이터셋에서 실험했습니다.

1. **MNIST**
2. **Fashion-MNIST**
3. **KMNIST**

모두 **이미지 분류 문제**입니다.

---

### 4.2 Baselines

비교 대상 activation function:

* ReLU
* tanh
* sin
* proposed weighted activation

---

### 4.3 Results

실험에서 관찰된 주요 결과:

1️⃣ **ReLU dominance**

초기 학습에서 ReLU가 다른 activation을 쉽게 압도하는 경향이 있음

2️⃣ **Layer dependency**

activation 선호도가 layer에 따라 달라짐

| Layer        | Preferred Activation |
| ------------ | -------------------- |
| Early layers | ReLU / LeakyReLU     |
| Deep layers  | tanh / sin           |

3️⃣ **Adaptive activation 효과**

제안된 방식은 다음 장점을 보임

* 더 안정적인 학습
* 다양한 activation 특성 활용

---

## 5. Strengths, Limitations, and Interpretation

### Strengths

1️⃣ **Adaptive activation learning**

activation function을 학습으로 결정

2️⃣ **Simple implementation**

기존 activation 함수 조합만 사용

3️⃣ **Layer-wise behavior insight**

레이어별 activation 선호 분석 제공

---

### Limitations

1️⃣ **limited function set**

사용한 activation function:

* ReLU
* tanh
* sin

더 많은 activation (Swish, GELU 등)을 고려하지 않음

2️⃣ **simple datasets**

실험 데이터셋이 비교적 단순

* MNIST 계열

3️⃣ **computational overhead**

activation function weight 학습 필요

---

### Interpretation

이 논문은 새로운 activation을 제안했다기보다는

> **Activation function을 학습 가능한 구조로 만드는 방향**

을 제시한 연구입니다.

최근 연구 흐름과도 연결됩니다.

예:

* ACON
* Meta activation
* Neural architecture search for activation

---

## 6. Conclusion

이 논문은 activation function을 고정된 함수로 사용하는 대신 **여러 activation function의 가중합 형태로 구성하고 그 가중치를 학습으로 최적화하는 방법**을 제안합니다.

주요 결론은 다음과 같습니다.

* activation function은 레이어별로 최적 형태가 다르다
* ReLU는 초기 layer에서 유리
* deeper layer에서는 smoother activation이 선호됨
* activation 자체도 학습 파라미터로 취급할 수 있다

이 연구는 **adaptive activation function 설계 방향**을 제시한다는 점에서 의미가 있습니다.

---

## Source

[https://arxiv.org/abs/2202.12065](https://arxiv.org/abs/2202.12065)
