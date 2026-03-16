# Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark

## 논문 메타데이터

- 제목: Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark
- 저자: Shiv Ram Dubey, Satish Kumar Singh, Bidyut Baran Chaudhuri
- 발표 연도: 2021
- arXiv ID: 2109.14545v3
- arXiv URL: <https://arxiv.org/abs/2109.14545v3>
- PDF: <https://arxiv.org/pdf/2109.14545v3.pdf>
- 카테고리: cs.LG
- 출판: Neurocomputing (논문 본문에 accept 표기)
- 코드: <https://github.com/shivram1987/ActivationFunctions>
- DOI: 본문 기준 확인 불가

## 연구 배경 및 문제 정의

딥러닝 모델은 층(layer)을 거치며 입력 데이터를 점진적으로 더 “선형적으로
분리 가능한” 추상 특징으로 변환하는 것을 목표로 한다. 이때 각 층은 선형 변환
(affine/conv 등)과 비선형 변환(activation function, AF)의 조합으로 구성된다.

AF는 다음과 같은 관점에서 학습을 좌우한다.

- 최적화 관점: 비선형성을 부여해 표현력을 확보하는 동시에, gradient 흐름을
  방해하지 않아야 한다.
- 계산 관점: 과도한 연산 비용을 추가하지 않으면서도 성능을 개선해야 한다.
- 데이터/표현 관점: 데이터 분포를 지나치게 왜곡하지 않고 학습을 안정화해야
  한다(예: zero-mean에 가까운 표현).

AF 관련 문헌은 방대하지만, “어떤 유형의 AF가 어떤 데이터/네트워크에서 유리한가”
를 **폭넓게 정리**하고, 서로 다른 네트워크와 데이터 모달리티에서 **실험적으로
비교**한 자료는 제한적이다. 본 논문은 (1) AF의 포괄적 분류/정리와 (2) 18개
활성화 함수를 여러 네트워크·데이터에서 비교하는 벤치마크를 함께 제공한다.

## 핵심 기여

논문이 명시한 기여를 요약하면 다음과 같다.

- Logistic Sigmoid/Tanh 기반, ReLU 기반, ELU 기반, 학습 기반(적응형) 등
  폭넓은 AF를 포괄하는 분류 체계 및 문헌 정리.
- AF의 특성(출력 범위, 단조성, smoothness 등)을 함께 제시해 선택 기준을 제공.
- 18개 AF를 대상으로 이미지(CIFAR10/100), 텍스트(독일어→영어 번역),
  음성(ASR)에서 여러 네트워크로 성능 비교를 수행.
- 결론/권고 사항을 “데이터 유형(모달리티) + 네트워크 유형”과 연결해 정리.

## 활성화 함수 분류 체계 (논문 구조 기반)

논문은 AF 발전을 개괄한 뒤, 대표 AF들을 다음과 같은 큰 범주로 나누어 정리한다.

### 1) Logistic Sigmoid / Tanh 기반

전통적 sigmoidal 계열과 그 변형들은 주로 다음 한계를 완화하는 방향으로 연구된다.

- **non zero-mean**(출력이 양수 위주로 치우침) 문제
- **zero-gradient**(포화로 인한 gradient 소실) 문제

다만 논문은 이러한 개선이 종종 함수 복잡도(연산량) 증가를 수반한다고 지적한다.

### 2) ReLU 기반 (Rectified Unit 계열)

ReLU와 그 변형은 딥러닝 표준으로 자리잡았지만, 논문은 ReLU의 대표 이슈를
다음 3가지로 정리한다.

- 음수 영역의 활용 부족(negative values under-utilization)
- 제한된 비선형성(limited nonlinearity)
- 출력이 unbounded

따라서 LReLU, PReLU, ABReLU 등은 음수 영역 처리, 형태 제어 등을 통해 위 이슈를
개선하려고 시도한다. 하지만 벤치마크에서는 “항상 ReLU를 확실히 이기는” 결과가
일관되게 나오지 않는다는 점도 함께 강조한다.

### 3) ELU 기반 (Exponential Unit 계열)

ELU/CELU/SELU 등 지수 기반(unit) 계열은 음수 영역을 더 부드럽게 활용해 포화 및
학습 안정성을 개선하려는 흐름으로 소개된다. 반면 논문은 일부 지수 계열이
**비매끄러운(non-smooth)** 형태로 인해 불리해질 수 있다는 점도 언급한다.

### 4) 학습 기반(적응형) AF (Learning based / Adaptive)

데이터/문제에 필요한 비선형 형태를 파라미터 학습으로 찾는 계열이다.
논문은 최근 이 범주가 인기를 얻고 있다고 정리하면서도,

- “좋은 base function / 파라미터 수” 설계가 어렵고
- 초기화가 좋지 않으면 학습이 발산(diverge)할 수 있음

을 주된 실무 리스크로 든다.

## AF 선택 관점: 특성(Characteristic) 체크리스트

논문은 AF를 비교할 때 다음과 같은 성질을 함께 보는 것이 유용하다고 정리한다.

- **출력 범위(output range)**: bounded vs unbounded, zero-mean 근접 여부
- **단조성(monotonicity)**: 단조 증가/비단조(예: 일부 modern AF는 비단조적 형태)
- **smoothness(매끄러움)**: 미분 가능성/연속성/기울기 안정성
- **계산 비용**: 추가 연산(지수/특수함수/분기 등)에 따른 학습 시간 증가 여부

이 관점은 “단일 점수로 AF를 고르는 것”이 아니라,
데이터·네트워크·하드웨어 제약에 따라 트레이드오프를 명시적으로 보는 틀에 가깝다.

## 실험 설정과 결과 (벤치마크 요약)

### 1) 평가 대상 AF (총 18개)

논문은 아래 18개 활성화 함수에 대해 실험 비교를 수행한다.

- Sigmoid, Tanh, Elliott, ReLU, LReLU, PReLU, ELU, SELU, GELU, CELU,
  Softplus, Swish, ABReLU, LiSHT, SRS, Mish, PAU, PDELU

### 2) 이미지 분류: CIFAR10 / CIFAR100

다양한 CNN 아키텍처(경량/중량 및 residual/skip 계열 포함)에서 정확도를 비교한다.

#### CIFAR10 (Table 8)에서의 관찰(발췌)

- MobileNet에서 ReLU(90.10±0.22) 대비 Softplus(91.05±0.22),
  CELU(91.04±0.17) 등이 더 높게 보고된다.
- ResNet50에서 ReLU(93.74±0.34) 대비 CELU(94.09±0.17)가 더 높게 보고된다.
- VGG16/GoogLeNet에서도 ReLU가 강력한 기준선으로 남지만,
  Mish/PDELU/Softplus 등 일부 AF가 동급 또는 근소 우위 성능을 보인다.

#### CIFAR100 (Table 9)에서의 관찰(발췌)

- MobileNet에서 Softplus(62.59±0.21)가 가장 높게 보고된다(표 내 기준).
- VGG16에서 Mish(68.13±0.40), PDELU(67.92±0.32), Softplus(67.70±0.19)
  등이 높은 편이며, ReLU(67.47±0.44)와 근접한 성능을 보인다.
- GoogLeNet에서 PDELU(74.48±1.23)가 가장 높게 보고되며,
  ReLU(74.05±1.69)도 경쟁력 있는 성능을 보인다.

### 3) 학습 시간(Training time) 비교 (Table 10의 메시지)

논문은 일부 AF가 정확도는 높지만 학습 시간 증가가 크다고 보고한다.
특히 PDELU와 SRS는 학습 시간이 크게 늘 수 있는 예로 언급되며,
ReLU/SELU/GELU/Softplus 등은 “정확도-시간”의 균형이 상대적으로 좋다는
결론을 제시한다.

### 4) 텍스트(번역) + 음성(ASR) 비교 (Table 11)

#### (a) German → English 번역(BLEU)

Seq2Seq(LSTM 기반 autoencoder) 모델에서,
AF를 dropout 이전 임베딩(feature embedding)에 적용한다.

- Epoch: 50, LR: 0.001, Batch size: 256
- Embedding size: 300, Dropout: 0.5
- Optimizer: Adam, Loss: cross-entropy
- Metric: BLEU(4-gram), 5회 평균±표준편차 보고

Table 11 기준 BLEU는 Tanh(20.93±0.91), SELU(20.85±0.64),
SRS(20.66±0.78), LiSHT(20.39±0.93) 등이 높게 보고된다.

#### (b) 음성 인식(ASR: LibriSpeech, DeepSpeech2 계열)

논문은 end-to-end ASR(Deep Speech 2 계열) 프레임워크를 사용하고,
CER/WER 평균을 5회 반복으로 보고한다.

Table 11 및 본문 요약에 따르면, PReLU/GELU/Swish/Mish/PAU 등이
ASR에서 “가장 적합”한 최근 AF로 관찰되었다고 정리한다
(여러 AF가 CER≈0.24, WER≈0.65 수준으로 보고됨).

## 한계 및 향후 연구 가능성

- AF의 성능은 최적화(학습률/스케줄), 정규화(BN 등), 초기화, 아키텍처와 강하게
  상호작용한다. 본 논문은 폭넓은 비교를 제공하지만, 모든 조합을 포괄할 수는 없다.
- 실험은 18개 AF로 “대표성 있는” 범위를 목표로 하지만, 이후 등장한 변형이나
  특정 도메인 특화 AF까지 포함하진 않는다.
- 번역/ASR 실험은 단일 구현(모델/프레임워크/데이터 설정)에 의존하므로,
  다른 구현이나 대규모 설정에서 동일 결론이 유지되는지는 추가 검증이 필요하다.

## 실무적 또는 연구적 인사이트

논문의 결론/권고를 실무 관점으로 재정리하면 다음과 같다.

- CNN에서 sigmoid/tanh는 수렴이 나빠질 수 있어 피하는 것이 권장되지만,
  RNN에서는 gate로 자주 사용된다.
- ReLU는 여전히 기본 선택으로 강력하나, Swish/Mish/PAU 같은 최근 AF도
  문제에 따라 시도할 가치가 있다.
- 이미지 분류에서는 residual 연결이 있는 네트워크에서
  ReLU/LReLU/ELU/GELU/CELU/PDELU 등이 유리하다고 정리한다.
- 파라미터 학습형(예: PReLU, PAU, PDELU)은 데이터에 적응하며 수렴이 좋을 수
  있지만, 초기화/파라미터 설계가 중요하고 학습 시간 증가 가능성이 있다.
- 모달리티별로 “좋은” AF가 달라질 수 있다.
  - 번역: Tanh, SELU, (그리고 PReLU/LiSHT/SRS/PAU)
  - ASR: PReLU, GELU, Swish, Mish, PAU
